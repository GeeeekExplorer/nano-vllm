import argparse
import json
import os
import random
import statistics
import subprocess
import threading
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


class GpuUtilSampler:

    def __init__(self, interval_ms: int = 200, gpu_id: int = 0):
        self.interval_s = max(interval_ms, 10) / 1000.0
        self.gpu_id = gpu_id
        self.samples: list[float] = []
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _query_once(self) -> float:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_id}",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip() or "nvidia-smi failed")
        line = res.stdout.strip().splitlines()[0].strip().replace("%", "")
        return float(line)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self._query_once())
            except Exception as e:
                self.error = str(e)
                break
            self._stop.wait(self.interval_s)

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def avg_util(self) -> Optional[float]:
        if not self.samples:
            return None
        return statistics.mean(self.samples)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * q / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    w = rank - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def build_arrivals(num_requests: int, mean_interval_ms: int, pattern: str, rng: random.Random) -> list[float]:
    mean_s = max(mean_interval_ms, 0) / 1000.0
    arrivals = []
    if pattern == "fixed":
        for i in range(num_requests):
            arrivals.append(i * mean_s)
        return arrivals

    # Poisson arrivals: exponentially distributed inter-arrival gaps.
    t = 0.0
    for i in range(num_requests):
        if i == 0:
            arrivals.append(0.0)
            continue
        if mean_s == 0:
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

    # hetero: short + long mixed, easier to reveal CB/CP benefits under load.
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


def run_once(
    *,
    mode_name: str,
    model: str,
    enable_cb: bool,
    enable_chunked_prefill: bool,
    requests: list[RequestSpec],
    arrivals: list[float],
    temperature: float,
    top_p: float,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    chunked_prefill_size: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
    gpu_sample_interval_ms: int,
    gpu_id: int,
) -> dict:
    llm = LLM(
        model,
        enable_continuous_batching=enable_cb,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        chunked_prefill_size=chunked_prefill_size,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
    )
    sampler = GpuUtilSampler(interval_ms=gpu_sample_interval_ms, gpu_id=gpu_id)
    stats: dict[int, RequestStat] = {}
    total_reqs = len(requests)
    sp_cache: dict[int, SamplingParams] = {}

    def get_sp(out_len: int) -> SamplingParams:
        if out_len not in sp_cache:
            sp_cache[out_len] = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=out_len,
                ignore_eos=True,
            )
        return sp_cache[out_len]

    t0 = time.perf_counter()
    t_end = t0
    first_arrival: Optional[float] = None
    try:
        warmup(llm)
        sampler.start()
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

            for seq_id, should_sample in zip(metadata["seq_ids"], metadata["should_sample"]):
                if not should_sample:
                    continue
                st = stats[seq_id]
                st.generated_tokens += 1
                if st.first_token_ts is None:
                    st.first_token_ts = step_ts
                if st.last_token_ts is not None:
                    st.inter_token_sum_s += (step_ts - st.last_token_ts)
                    st.inter_token_count += 1
                st.last_token_ts = step_ts

            for st in stats.values():
                if st.finish_ts is None and st.generated_tokens >= st.target_output_tokens:
                    st.finish_ts = step_ts
                    done += 1

        t_end = time.perf_counter()
    finally:
        sampler.stop()
        llm.exit()

    if first_arrival is None:
        first_arrival = t0
    makespan_s = max(t_end - first_arrival, 1e-9)

    all_stats = list(stats.values())
    summary = summarize_request_stats(all_stats)
    tiers = {}
    tier_names = sorted(set(st.tier for st in all_stats))
    for tier in tier_names:
        tier_stats = [st for st in all_stats if st.tier == tier]
        tiers[tier] = summarize_request_stats(tier_stats)

    return {
        "mode": mode_name,
        "num_requests": total_reqs,
        "arrival_interval_ms": arrivals[1] * 1000.0 - arrivals[0] * 1000.0 if total_reqs > 1 else 0.0,
        "throughput_rps": total_reqs / makespan_s,
        "avg_ttft_ms": summary["avg_ttft_ms"],
        "avg_itl_ms_per_token": summary["avg_itl_ms_per_token"],
        "avg_latency_ms": summary["avg_latency_ms"],
        "p50_latency_ms": summary["p50_latency_ms"],
        "p95_latency_ms": summary["p95_latency_ms"],
        "p99_latency_ms": summary["p99_latency_ms"],
        "gpu_util_avg_percent": sampler.avg_util,
        "gpu_util_samples": len(sampler.samples),
        "gpu_util_error": sampler.error,
        "makespan_s": makespan_s,
        "tiers": tiers,
        "enable_continuous_batching": enable_cb,
        "enable_chunked_prefill": enable_chunked_prefill,
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
    if lower_better:
        pct = (base - cur) / base * 100.0
    else:
        pct = (cur - base) / base * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


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


def print_pair_report(off: dict, on: dict):
    print("\n=== Pair Comparison: OFF/OFF vs ON/ON ===")
    print(f"Requests: {off['num_requests']}, Mean arrival interval: {off['arrival_interval_ms']:.1f} ms")
    print("")
    print(f"{'Metric':<28} {'OFF/OFF':>14} {'ON/ON':>14} {'Improvement':>14}")
    print("-" * 74)
    rows = [
        ("Throughput (req/s)", off["throughput_rps"], on["throughput_rps"], False),
        ("Avg TTFT (ms)", off["avg_ttft_ms"], on["avg_ttft_ms"], True),
        ("Avg ITL (ms/token)", off["avg_itl_ms_per_token"], on["avg_itl_ms_per_token"], True),
        ("Avg latency (ms)", off["avg_latency_ms"], on["avg_latency_ms"], True),
        ("P50 latency (ms)", off["p50_latency_ms"], on["p50_latency_ms"], True),
        ("P95 latency (ms)", off["p95_latency_ms"], on["p95_latency_ms"], True),
        ("P99 latency (ms)", off["p99_latency_ms"], on["p99_latency_ms"], True),
        ("GPU util avg (%)", off["gpu_util_avg_percent"], on["gpu_util_avg_percent"], False),
        ("Makespan (s)", off["makespan_s"], on["makespan_s"], True),
    ]
    for name, v_off, v_on, lower_better in rows:
        print(f"{name:<28} {fmt(v_off):>14} {fmt(v_on):>14} {safe_delta(v_off, v_on, lower_better):>14}")


def print_matrix_report(results: dict[str, dict]):
    baseline = results["off_off"]
    print("\n=== Matrix Comparison (CP/CB Value) ===")
    print(f"Requests: {baseline['num_requests']}, Mean arrival interval: {baseline['arrival_interval_ms']:.1f} ms")
    print("")
    print(f"{'Mode':<12} {'Throughput':>12} {'Avg TTFT':>12} {'Avg Lat':>12} {'P95 Lat':>12} {'Makespan':>12}")
    print("-" * 74)
    order = ["off_off", "off_on", "on_off", "on_on"]
    for mode in order:
        r = results[mode]
        print(
            f"{mode:<12} "
            f"{fmt(r['throughput_rps']):>12} "
            f"{fmt(r['avg_ttft_ms']):>12} "
            f"{fmt(r['avg_latency_ms']):>12} "
            f"{fmt(r['p95_latency_ms']):>12} "
            f"{fmt(r['makespan_s']):>12}"
        )

    print("\n=== Gain vs OFF/OFF Baseline ===")
    print(f"{'Variant':<16} {'Throughput':>14} {'Avg TTFT':>14} {'Avg Lat':>14} {'P95 Lat':>14} {'Makespan':>14}")
    print("-" * 90)
    variants = [
        ("CP only", "off_on"),
        ("CB only", "on_off"),
        ("CP + CB", "on_on"),
    ]
    for label, mode in variants:
        r = results[mode]
        print(
            f"{label:<16} "
            f"{safe_delta(baseline['throughput_rps'], r['throughput_rps'], False):>14} "
            f"{safe_delta(baseline['avg_ttft_ms'], r['avg_ttft_ms'], True):>14} "
            f"{safe_delta(baseline['avg_latency_ms'], r['avg_latency_ms'], True):>14} "
            f"{safe_delta(baseline['p95_latency_ms'], r['p95_latency_ms'], True):>14} "
            f"{safe_delta(baseline['makespan_s'], r['makespan_s'], True):>14}"
        )

    if baseline["tiers"]:
        print("\n=== Tier Avg TTFT Gain vs OFF/OFF ===")
        print(f"{'Tier':<10} {'CP only':>12} {'CB only':>12} {'CP+CB':>12}")
        print("-" * 52)
        for tier in sorted(baseline["tiers"].keys()):
            base_t = baseline["tiers"][tier]["avg_ttft_ms"]
            cp_t = results["off_on"]["tiers"].get(tier, {}).get("avg_ttft_ms", float("nan"))
            cb_t = results["on_off"]["tiers"].get(tier, {}).get("avg_ttft_ms", float("nan"))
            both_t = results["on_on"]["tiers"].get(tier, {}).get("avg_ttft_ms", float("nan"))
            print(
                f"{tier:<10} "
                f"{safe_delta(base_t, cp_t, True):>12} "
                f"{safe_delta(base_t, cb_t, True):>12} "
                f"{safe_delta(base_t, both_t, True):>12}"
            )


def cleanup_between_runs():
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark continuous batching + chunked prefill on online workload")
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--num-requests", type=int, default=500)
    parser.add_argument("--arrival-interval-ms", type=int, default=50)
    parser.add_argument("--arrival-pattern", choices=["fixed", "poisson"], default="poisson")
    parser.add_argument("--workload-profile", choices=["uniform", "hetero"], default="hetero")

    # Uniform profile args
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--prompt-jitter", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=64)

    # Heterogeneous profile args
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
    parser.add_argument("--compare-mode", choices=["pair", "matrix"], default="matrix")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=200)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.num_requests > 0, "num-requests must be > 0"
    assert args.arrival_interval_ms >= 0, "arrival-interval-ms must be >= 0"
    assert args.max_num_seqs > 0, "max-num-seqs must be > 0"
    assert args.max_num_batched_tokens > 0, "max-num-batched-tokens must be > 0"
    if args.workload_profile == "uniform":
        assert args.prompt_len > 0, "prompt-len must be > 0"
        assert args.output_len > 0, "output-len must be > 0"
    else:
        assert args.short_prompt_len > 0 and args.long_prompt_len > 0, "prompt lens must be > 0"
        assert args.short_output_len > 0 and args.long_output_len > 0, "output lens must be > 0"

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
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        gpu_sample_interval_ms=args.gpu_sample_interval_ms,
        gpu_id=args.gpu_id,
    )

    if args.compare_mode == "pair":
        mode_plan = [
            ("off_off", False, False),
            ("on_on", True, True),
        ]
    else:
        mode_plan = [
            ("off_off", False, False),
            ("off_on", False, True),
            ("on_off", True, False),
            ("on_on", True, True),
        ]

    results = {}
    for i, (name, cb, cp) in enumerate(mode_plan):
        print(f"Running {name} (CB={'ON' if cb else 'OFF'}, CP={'ON' if cp else 'OFF'}) ...")
        results[name] = run_once(
            mode_name=name,
            enable_cb=cb,
            enable_chunked_prefill=cp,
            **common_kwargs,
        )
        if i != len(mode_plan) - 1:
            cleanup_between_runs()

    if args.compare_mode == "pair":
        print_pair_report(results["off_off"], results["on_on"])
    else:
        print_matrix_report(results)

    if args.output_json:
        payload = {
            "config": vars(args),
            "results": results,
            "timestamp": int(time.time()),
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report to {args.output_json}")


if __name__ == "__main__":
    main()
