import argparse
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import Optional

from nanovllm import LLM, SamplingParams


@dataclass
class RequestSpec:
    req_id: int
    prompt_tokens: list[int]
    output_len: int
    tier: str


@dataclass
class RequestStat:
    req_id: int
    tier: str
    target_output_tokens: int
    arrival_ts: float
    first_token_ts: Optional[float] = None
    finish_ts: Optional[float] = None
    last_token_ts: Optional[float] = None
    generated_tokens: int = 0
    inter_token_sum_s: float = 0.0
    inter_token_count: int = 0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * q / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    weight = rank - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def build_arrivals(num_requests: int, mean_interval_ms: int, pattern: str, rng: random.Random) -> list[float]:
    mean_s = max(mean_interval_ms, 0) / 1000.0
    arrivals = []
    if pattern == "fixed":
        for i in range(num_requests):
            arrivals.append(i * mean_s)
        return arrivals

    t = 0.0
    for i in range(num_requests):
        if i == 0 or mean_s == 0:
            arrivals.append(0.0)
            continue
        t += rng.expovariate(1.0 / mean_s)
        arrivals.append(t)
    return arrivals


def build_workload(args) -> tuple[list[RequestSpec], list[float]]:
    rng = random.Random(args.seed)
    arrivals = build_arrivals(args.num_requests, args.arrival_interval_ms, args.arrival_pattern, rng)
    requests: list[RequestSpec] = []

    def sample_prompt_tokens(base_len: int, jitter: int) -> list[int]:
        delta = rng.randint(-jitter, jitter) if jitter > 0 else 0
        cur_len = max(4, base_len + delta)
        return [rng.randint(0, 10000) for _ in range(cur_len)]

    if args.workload_profile == "uniform":
        for i in range(args.num_requests):
            requests.append(
                RequestSpec(
                    req_id=i,
                    prompt_tokens=sample_prompt_tokens(args.prompt_len, args.prompt_jitter),
                    output_len=args.output_len,
                    tier="uniform",
                )
            )
        return requests, arrivals

    short_ratio = min(max(args.short_ratio, 0.0), 1.0)
    for i in range(args.num_requests):
        is_short = rng.random() < short_ratio
        if is_short:
            req = RequestSpec(
                req_id=i,
                prompt_tokens=sample_prompt_tokens(args.short_prompt_len, args.short_prompt_jitter),
                output_len=args.short_output_len,
                tier="short",
            )
        else:
            req = RequestSpec(
                req_id=i,
                prompt_tokens=sample_prompt_tokens(args.long_prompt_len, args.long_prompt_jitter),
                output_len=args.long_output_len,
                tier="long",
            )
        requests.append(req)
    return requests, arrivals


def warmup(llm: LLM):
    llm.generate(
        [[1, 2, 3, 4, 5, 6, 7, 8]],
        SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, ignore_eos=True),
        use_tqdm=False,
    )


def summarize_request_stats(request_stats: list[RequestStat]) -> dict:
    latencies_ms = []
    ttft_ms = []
    itl_ms = []
    for st in request_stats:
        if st.finish_ts is None:
            continue
        latencies_ms.append((st.finish_ts - st.arrival_ts) * 1000.0)
        if st.first_token_ts is not None:
            ttft_ms.append((st.first_token_ts - st.arrival_ts) * 1000.0)
        if st.inter_token_count > 0:
            itl_ms.append(st.inter_token_sum_s / st.inter_token_count * 1000.0)

    return {
        "count": len(request_stats),
        "avg_ttft_ms": statistics.mean(ttft_ms) if ttft_ms else float("nan"),
        "avg_itl_ms_per_token": statistics.mean(itl_ms) if itl_ms else float("nan"),
        "avg_latency_ms": statistics.mean(latencies_ms) if latencies_ms else float("nan"),
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "p99_latency_ms": percentile(latencies_ms, 99),
    }


def fmt(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and (v != v):
        return "N/A"
    return f"{v:.{digits}f}"


def safe_delta(base: Optional[float], cur: Optional[float], lower_better: bool) -> str:
    if base is None or cur is None:
        return "N/A"
    if isinstance(base, float) and (base != base):
        return "N/A"
    if isinstance(cur, float) and (cur != cur):
        return "N/A"
    if abs(base) < 1e-9:
        return "N/A"
    pct = (base - cur) / base * 100.0 if lower_better else (cur - base) / base * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def cleanup_between_runs():
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def aggregate_trial_results(mode_name: str, runs: list[dict]) -> dict:
    if len(runs) == 1:
        out = dict(runs[0])
        out["mode"] = mode_name
        out["num_trials"] = 1
        return out

    metric_keys = [
        "throughput_rps",
        "avg_ttft_ms",
        "avg_itl_ms_per_token",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "makespan_s",
        "recomputed_prefill_tokens",
        "prefix_cache_hit_tokens",
        "avg_recomputed_prefill_tokens_per_request",
        "avg_prefix_cache_hit_tokens_per_request",
    ]
    out = dict(runs[0])
    out["mode"] = mode_name
    out["num_trials"] = len(runs)
    for key in metric_keys:
        vals = [r[key] for r in runs]
        out[key] = statistics.median(vals)
    return out


def run_once(
    *,
    mode_name: str,
    model: str,
    requests: list[RequestSpec],
    arrivals: list[float],
    temperature: float,
    top_p: float,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    chunked_prefill_size: int,
    enable_cb_prefill_liveness: bool,
    cb_prefill_reserve_ratio: float,
    cb_prefill_min_tokens: int,
    cb_prefill_min_seqs: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
    enable_resumable_priority: bool,
    resumable_priority_cached_tokens_weight: float,
    resumable_priority_remaining_prefill_tokens_weight: float,
    resumable_priority_waiting_time_weight: float,
    resumable_priority_preempt_count_weight: float,
) -> dict:
    llm = LLM(
        model,
        enable_continuous_batching=True,
        enable_chunked_prefill=True,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        chunked_prefill_size=chunked_prefill_size,
        enable_cb_prefill_liveness=enable_cb_prefill_liveness,
        cb_prefill_reserve_ratio=cb_prefill_reserve_ratio,
        cb_prefill_min_tokens=cb_prefill_min_tokens,
        cb_prefill_min_seqs=cb_prefill_min_seqs,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        enable_resumable_priority=enable_resumable_priority,
        resumable_priority_cached_tokens_weight=resumable_priority_cached_tokens_weight,
        resumable_priority_remaining_prefill_tokens_weight=resumable_priority_remaining_prefill_tokens_weight,
        resumable_priority_waiting_time_weight=resumable_priority_waiting_time_weight,
        resumable_priority_preempt_count_weight=resumable_priority_preempt_count_weight,
    )
    stats: dict[int, RequestStat] = {}
    sp_cache: dict[int, SamplingParams] = {}
    total_reqs = len(requests)
    total_recomputed_prefill_tokens = 0
    total_prefix_cache_hit_tokens = 0

    def get_sp(out_len: int) -> SamplingParams:
        if out_len not in sp_cache:
            sp_cache[out_len] = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=out_len,
                ignore_eos=True,
            )
        return sp_cache[out_len]

    t0 = 0.0
    t_end = 0.0
    first_arrival: Optional[float] = None
    try:
        warmup(llm)
        t0 = time.perf_counter()
        t_end = t0
        next_idx = 0
        done = 0

        while done < total_reqs:
            now = time.perf_counter()
            while next_idx < total_reqs and now - t0 >= arrivals[next_idx]:
                req = requests[next_idx]
                seq = llm.add_request(req.prompt_tokens, get_sp(req.output_len))
                stats[seq.seq_id] = RequestStat(
                    req_id=req.req_id,
                    tier=req.tier,
                    target_output_tokens=req.output_len,
                    arrival_ts=now,
                )
                if first_arrival is None:
                    first_arrival = now
                next_idx += 1

            if llm.is_finished():
                if next_idx >= total_reqs:
                    break
                target = t0 + arrivals[next_idx]
                sleep_s = max(0.0, min(0.001, target - time.perf_counter()))
                if sleep_s > 0:
                    time.sleep(sleep_s)
                continue

            _, _, metadata = llm.step(return_metadata=True)
            step_ts = time.perf_counter()
            total_recomputed_prefill_tokens += metadata["recomputed_prefill_tokens"]
            total_prefix_cache_hit_tokens += metadata["prefix_cache_hit_tokens"]

            for seq_id, should_sample in zip(metadata["seq_ids"], metadata["should_sample"]):
                if not should_sample:
                    continue
                st = stats[seq_id]
                st.generated_tokens += 1
                if st.first_token_ts is None:
                    st.first_token_ts = step_ts
                if st.last_token_ts is not None:
                    st.inter_token_sum_s += step_ts - st.last_token_ts
                    st.inter_token_count += 1
                st.last_token_ts = step_ts

            for st in stats.values():
                if st.finish_ts is None and st.generated_tokens >= st.target_output_tokens:
                    st.finish_ts = step_ts
                    done += 1

        t_end = time.perf_counter()
    finally:
        llm.exit()

    if first_arrival is None:
        first_arrival = t0
    makespan_s = max(t_end - first_arrival, 1e-9)
    summary = summarize_request_stats(list(stats.values()))
    return {
        "mode": mode_name,
        "num_requests": total_reqs,
        "arrival_interval_ms": ((arrivals[-1] - arrivals[0]) * 1000.0 / (total_reqs - 1)) if total_reqs > 1 else 0.0,
        "throughput_rps": total_reqs / makespan_s,
        "avg_ttft_ms": summary["avg_ttft_ms"],
        "avg_itl_ms_per_token": summary["avg_itl_ms_per_token"],
        "avg_latency_ms": summary["avg_latency_ms"],
        "p50_latency_ms": summary["p50_latency_ms"],
        "p95_latency_ms": summary["p95_latency_ms"],
        "p99_latency_ms": summary["p99_latency_ms"],
        "makespan_s": makespan_s,
        "recomputed_prefill_tokens": total_recomputed_prefill_tokens,
        "prefix_cache_hit_tokens": total_prefix_cache_hit_tokens,
        "avg_recomputed_prefill_tokens_per_request": total_recomputed_prefill_tokens / total_reqs,
        "avg_prefix_cache_hit_tokens_per_request": total_prefix_cache_hit_tokens / total_reqs,
        "enable_resumable_priority": enable_resumable_priority,
    }


def run_trials(mode_name: str, num_trials: int, run_kwargs: dict) -> dict:
    runs = []
    for i in range(num_trials):
        kwargs = dict(run_kwargs)
        kwargs["mode_name"] = f"{mode_name}_t{i+1}"
        runs.append(run_once(**kwargs))
        if i != num_trials - 1:
            cleanup_between_runs()
    return aggregate_trial_results(mode_name, runs)


def print_workload_summary(requests: list[RequestSpec], arrivals: list[float], profile: str, pattern: str):
    prompt_lens = [len(r.prompt_tokens) for r in requests]
    output_lens = [r.output_len for r in requests]
    tiers = {}
    for r in requests:
        tiers[r.tier] = tiers.get(r.tier, 0) + 1
    mean_interval = 0.0
    if len(arrivals) > 1:
        mean_interval = (arrivals[-1] - arrivals[0]) * 1000.0 / (len(arrivals) - 1)
    print("\n=== Workload Summary ===")
    print(f"Profile: {profile}, Arrival pattern: {pattern}")
    print(
        f"Requests: {len(requests)}, "
        f"Avg prompt len: {statistics.mean(prompt_lens):.1f}, "
        f"Avg output len: {statistics.mean(output_lens):.1f}, "
        f"Mean arrival interval: {mean_interval:.1f} ms"
    )
    print(f"Tier counts: {tiers}")


def print_report(priority_off: dict, priority_on: dict):
    print("\n=== Resumable Priority Comparison (CB=ON, CP=ON) ===")
    print(f"Requests: {priority_off['num_requests']}, Mean arrival interval: {priority_off['arrival_interval_ms']:.1f} ms")
    print("")
    print(f"{'Metric':<36} {'Priority OFF':>14} {'Priority ON':>14} {'Change':>14}")
    print("-" * 82)
    rows = [
        ("Throughput (req/s)", priority_off["throughput_rps"], priority_on["throughput_rps"], False),
        ("Avg TTFT (ms)", priority_off["avg_ttft_ms"], priority_on["avg_ttft_ms"], True),
        ("Avg ITL (ms/token)", priority_off["avg_itl_ms_per_token"], priority_on["avg_itl_ms_per_token"], True),
        ("Avg latency (ms)", priority_off["avg_latency_ms"], priority_on["avg_latency_ms"], True),
        ("P95 latency (ms)", priority_off["p95_latency_ms"], priority_on["p95_latency_ms"], True),
        ("Makespan (s)", priority_off["makespan_s"], priority_on["makespan_s"], True),
        ("Recomputed prefill tokens", priority_off["recomputed_prefill_tokens"], priority_on["recomputed_prefill_tokens"], True),
        ("Prefix cache hit tokens", priority_off["prefix_cache_hit_tokens"], priority_on["prefix_cache_hit_tokens"], False),
        (
            "Avg recomputed prefill / req",
            priority_off["avg_recomputed_prefill_tokens_per_request"],
            priority_on["avg_recomputed_prefill_tokens_per_request"],
            True,
        ),
        (
            "Avg prefix cache hit / req",
            priority_off["avg_prefix_cache_hit_tokens_per_request"],
            priority_on["avg_prefix_cache_hit_tokens_per_request"],
            False,
        ),
    ]
    for name, v_off, v_on, lower_better in rows:
        print(f"{name:<36} {fmt(v_off):>14} {fmt(v_on):>14} {safe_delta(v_off, v_on, lower_better):>14}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark resumable-priority scheduling under CB+CP")
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--num-requests", type=int, default=500)
    parser.add_argument("--arrival-interval-ms", type=int, default=50)
    parser.add_argument("--arrival-pattern", choices=["fixed", "poisson"], default="poisson")
    parser.add_argument("--workload-profile", choices=["uniform", "hetero"], default="hetero")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--prompt-jitter", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--short-ratio", type=float, default=0.7)
    parser.add_argument("--short-prompt-len", type=int, default=64)
    parser.add_argument("--short-prompt-jitter", type=int, default=16)
    parser.add_argument("--short-output-len", type=int, default=64)
    parser.add_argument("--long-prompt-len", type=int, default=1024)
    parser.add_argument("--long-prompt-jitter", type=int, default=128)
    parser.add_argument("--long-output-len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument("--disable-cb-prefill-liveness", action="store_true")
    parser.add_argument("--cb-prefill-reserve-ratio", type=float, default=0.2)
    parser.add_argument("--cb-prefill-min-tokens", type=int, default=512)
    parser.add_argument("--cb-prefill-min-seqs", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--resumable-priority-cached-tokens-weight", type=float, default=1.0)
    parser.add_argument("--resumable-priority-remaining-prefill-tokens-weight", type=float, default=1.0)
    parser.add_argument("--resumable-priority-waiting-time-weight", type=float, default=1.0)
    parser.add_argument("--resumable-priority-preempt-count-weight", type=float, default=1.0)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    args.enable_cb_prefill_liveness = not args.disable_cb_prefill_liveness
    assert args.num_requests > 0, "num-requests must be > 0"
    assert args.arrival_interval_ms >= 0, "arrival-interval-ms must be >= 0"
    assert args.trials > 0, "trials must be > 0"
    assert args.max_num_seqs > 0, "max-num-seqs must be > 0"
    assert args.max_num_batched_tokens > 0, "max-num-batched-tokens must be > 0"
    assert args.chunked_prefill_size > 0, "chunked-prefill-size must be > 0"
    assert 0.0 <= args.cb_prefill_reserve_ratio <= 1.0, "cb-prefill-reserve-ratio must be in [0, 1]"
    assert args.cb_prefill_min_tokens >= 0, "cb-prefill-min-tokens must be >= 0"
    assert args.cb_prefill_min_seqs >= 0, "cb-prefill-min-seqs must be >= 0"
    assert args.resumable_priority_cached_tokens_weight >= 0.0
    assert args.resumable_priority_remaining_prefill_tokens_weight >= 0.0
    assert args.resumable_priority_waiting_time_weight >= 0.0
    assert args.resumable_priority_preempt_count_weight >= 0.0
    if args.workload_profile == "uniform":
        assert args.prompt_len > 0 and args.output_len > 0
    else:
        assert args.short_prompt_len > 0 and args.long_prompt_len > 0
        assert args.short_output_len > 0 and args.long_output_len > 0

    requests, arrivals = build_workload(args)
    print_workload_summary(requests, arrivals, args.workload_profile, args.arrival_pattern)

    common_kwargs = dict(
        model=args.model,
        requests=requests,
        arrivals=arrivals,
        temperature=args.temperature,
        top_p=args.top_p,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        chunked_prefill_size=args.chunked_prefill_size,
        enable_cb_prefill_liveness=args.enable_cb_prefill_liveness,
        cb_prefill_reserve_ratio=args.cb_prefill_reserve_ratio,
        cb_prefill_min_tokens=args.cb_prefill_min_tokens,
        cb_prefill_min_seqs=args.cb_prefill_min_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        resumable_priority_cached_tokens_weight=args.resumable_priority_cached_tokens_weight,
        resumable_priority_remaining_prefill_tokens_weight=args.resumable_priority_remaining_prefill_tokens_weight,
        resumable_priority_waiting_time_weight=args.resumable_priority_waiting_time_weight,
        resumable_priority_preempt_count_weight=args.resumable_priority_preempt_count_weight,
    )

    print("Running resume_priority_off (CB=ON, CP=ON, resumable-priority=OFF) ...")
    priority_off = run_trials(
        mode_name="resume_priority_off",
        num_trials=args.trials,
        run_kwargs=dict(common_kwargs, enable_resumable_priority=False),
    )
    cleanup_between_runs()

    print("Running resume_priority_on (CB=ON, CP=ON, resumable-priority=ON) ...")
    priority_on = run_trials(
        mode_name="resume_priority_on",
        num_trials=args.trials,
        run_kwargs=dict(common_kwargs, enable_resumable_priority=True),
    )

    print_report(priority_off, priority_on)

    if args.output_json:
        payload = {
            "config": vars(args),
            "results": {
                "resume_priority_off": priority_off,
                "resume_priority_on": priority_on,
            },
            "timestamp": int(time.time()),
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report to {args.output_json}")


if __name__ == "__main__":
    main()
