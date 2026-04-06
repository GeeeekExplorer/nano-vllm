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
    enable_cb_prefill_liveness: bool,
    cb_prefill_reserve_ratio: float,
    cb_prefill_min_tokens: int,
    cb_prefill_min_seqs: int,
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
        enable_cb_prefill_liveness=enable_cb_prefill_liveness,
        cb_prefill_reserve_ratio=cb_prefill_reserve_ratio,
        cb_prefill_min_tokens=cb_prefill_min_tokens,
        cb_prefill_min_seqs=cb_prefill_min_seqs,
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

    t0 = 0.0
    t_end = 0.0
    first_arrival: Optional[float] = None
    try:
        warmup(llm)
        t0 = time.perf_counter()
        t_end = t0
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
        "arrival_interval_ms": ((arrivals[-1] - arrivals[0]) * 1000.0 / (total_reqs - 1)) if total_reqs > 1 else 0.0,
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


def run_once_guarded(**kwargs) -> dict:
    mode_name = kwargs.get("mode_name", "unknown")
    try:
        return run_once(**kwargs)
    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if "out of memory" in lower or "cuda oom" in lower:
            print(f"[WARN] {mode_name} skipped due to OOM: {msg}")
            return {"mode": mode_name, "error": msg}
        if "illegal memory access" in lower:
            print(f"[WARN] {mode_name} skipped due to CUDA illegal memory access: {msg}")
            return {"mode": mode_name, "error": msg}
        raise


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


def parse_int_csv(raw: str) -> list[int]:
    xs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        xs.append(int(part))
    return xs


def parse_float_csv(raw: str) -> list[float]:
    xs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        xs.append(float(part))
    return xs


def is_nan_or_none(v: Optional[float]) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and (v != v):
        return True
    return False


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
        "gpu_util_avg_percent",
        "makespan_s",
    ]

    out = dict(runs[0])
    out["mode"] = mode_name
    out["num_trials"] = len(runs)

    for key in metric_keys:
        vals = [r[key] for r in runs if key in r and not is_nan_or_none(r[key])]
        out[key] = statistics.median(vals) if vals else float("nan")

    tiers = {}
    tier_names = set()
    for r in runs:
        tier_names.update(r.get("tiers", {}).keys())
    tier_metric_keys = ["avg_ttft_ms", "avg_itl_ms_per_token", "avg_latency_ms", "p95_latency_ms", "p99_latency_ms"]
    for tier in sorted(tier_names):
        tiers[tier] = {}
        for mkey in tier_metric_keys:
            vals = []
            for r in runs:
                v = r.get("tiers", {}).get(tier, {}).get(mkey, float("nan"))
                if not is_nan_or_none(v):
                    vals.append(v)
            tiers[tier][mkey] = statistics.median(vals) if vals else float("nan")
    out["tiers"] = tiers
    return out


def run_trials(mode_name: str, num_trials: int, run_kwargs: dict) -> dict:
    trial_runs = []
    for i in range(num_trials):
        trial_mode = f"{mode_name}_t{i+1}"
        kwargs = dict(run_kwargs)
        kwargs["mode_name"] = trial_mode
        r = run_once_guarded(**kwargs)
        if "error" in r:
            return r
        trial_runs.append(r)
        if i != num_trials - 1:
            cleanup_between_runs()
    return aggregate_trial_results(mode_name, trial_runs)


def dominates(base: dict, cand: dict) -> bool:
    # Comprehensive dominance criteria for "CP+CB全面优化".
    rules = [
        ("throughput_rps", False),
        ("avg_ttft_ms", True),
        ("avg_itl_ms_per_token", True),
        ("avg_latency_ms", True),
        ("p95_latency_ms", True),
        ("makespan_s", True),
    ]
    for key, lower_better in rules:
        b = base[key]
        c = cand[key]
        if is_nan_or_none(b) or is_nan_or_none(c):
            return False
        if lower_better:
            if c > b:
                return False
        else:
            if c < b:
                return False
    return True


def dominance_score(base: dict, cand: dict) -> float:
    # Aggregate score: positive means "better overall".
    rules = [
        ("throughput_rps", False),
        ("avg_ttft_ms", True),
        ("avg_itl_ms_per_token", True),
        ("avg_latency_ms", True),
        ("p95_latency_ms", True),
        ("makespan_s", True),
    ]
    score = 0.0
    for key, lower_better in rules:
        b = base[key]
        c = cand[key]
        if is_nan_or_none(b) or is_nan_or_none(c):
            continue
        if abs(b) < 1e-9:
            continue
        if lower_better:
            score += (b - c) / b
        else:
            score += (c - b) / b
    return score


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
    print(f"{'Mode':<12} {'Throughput':>12} {'Avg TTFT':>12} {'Avg ITL':>12} {'Avg Lat':>12} {'P95 Lat':>12} {'Makespan':>12}")
    print("-" * 74)
    order = ["off_off", "off_on", "on_off", "on_on"]
    for mode in order:
        r = results[mode]
        print(
            f"{mode:<12} "
            f"{fmt(r['throughput_rps']):>12} "
            f"{fmt(r['avg_ttft_ms']):>12} "
            f"{fmt(r['avg_itl_ms_per_token']):>12} "
            f"{fmt(r['avg_latency_ms']):>12} "
            f"{fmt(r['p95_latency_ms']):>12} "
            f"{fmt(r['makespan_s']):>12}"
        )

    print("\n=== Gain vs OFF/OFF Baseline ===")
    print(f"{'Variant':<16} {'Throughput':>14} {'Avg TTFT':>14} {'Avg ITL':>14} {'Avg Lat':>14} {'P95 Lat':>14} {'Makespan':>14}")
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
            f"{safe_delta(baseline['avg_itl_ms_per_token'], r['avg_itl_ms_per_token'], True):>14} "
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


def print_opt_report(base: dict, candidates: list[dict], best: dict, found_dominating: bool):
    print("\n=== Optimize ON/ON vs OFF/OFF Baseline ===")
    print("Goal: throughput up, and TTFT/ITL/latency/P95/makespan all down")
    print("")
    print(
        f"{'Candidate':<20} {'Chunk':>8} {'BatchedTok':>10} {'RsvR':>6} {'RsvTok':>8} {'RsvSeq':>8} "
        f"{'Thrpt':>10} {'TTFT':>10} {'ITL':>10} {'P95':>10} {'Mkspan':>10} {'Dominates':>10}"
    )
    print("-" * 144)
    for r in candidates:
        print(
            f"{r['mode']:<20} {r['chunked_prefill_size']:>8} {r['max_num_batched_tokens']:>10} "
            f"{r.get('cb_prefill_reserve_ratio', float('nan')):>6.2f} "
            f"{r.get('cb_prefill_min_tokens', -1):>8} "
            f"{r.get('cb_prefill_min_seqs', -1):>8} "
            f"{fmt(r['throughput_rps']):>10} {fmt(r['avg_ttft_ms']):>10} {fmt(r['avg_itl_ms_per_token']):>10} "
            f"{fmt(r['p95_latency_ms']):>10} {fmt(r['makespan_s']):>10} "
            f"{'YES' if r['dominates'] else 'NO':>10}"
        )

    print("\n=== Best Candidate ===")
    print(
        f"{best['mode']}, chunked_prefill_size={best['chunked_prefill_size']}, "
        f"max_num_batched_tokens={best['max_num_batched_tokens']}, "
        f"reserve_ratio={best.get('cb_prefill_reserve_ratio')}, "
        f"reserve_min_tokens={best.get('cb_prefill_min_tokens')}, "
        f"reserve_min_seqs={best.get('cb_prefill_min_seqs')}, "
        f"dominates={best['dominates']}"
    )
    print(
        "Gain vs OFF/OFF: "
        f"Thrpt {safe_delta(base['throughput_rps'], best['throughput_rps'], False)}, "
        f"TTFT {safe_delta(base['avg_ttft_ms'], best['avg_ttft_ms'], True)}, "
        f"ITL {safe_delta(base['avg_itl_ms_per_token'], best['avg_itl_ms_per_token'], True)}, "
        f"P95 {safe_delta(base['p95_latency_ms'], best['p95_latency_ms'], True)}, "
        f"Mkspan {safe_delta(base['makespan_s'], best['makespan_s'], True)}"
    )
    if found_dominating:
        print("Result: Found at least one ON/ON candidate that comprehensively dominates OFF/OFF.")
    else:
        print("Result: No strict comprehensive dominance found in this search space.")


def print_opt_search_report(scenarios: list[dict], global_best: dict, found_dominating: bool):
    print("\n=== Multi-Interval Search Summary ===")
    print(f"{'Interval(ms)':<14} {'BestCandidate':<20} {'Dominates':>10} {'Thrpt':>10} {'TTFT':>10} {'ITL':>10} {'P95':>10} {'Mkspan':>10}")
    print("-" * 106)
    for sc in scenarios:
        base = sc["baseline"]
        best = sc["best"]
        print(
            f"{sc['arrival_interval_ms']:<14} "
            f"{best['mode']:<20} "
            f"{'YES' if best['dominates'] else 'NO':>10} "
            f"{safe_delta(base['throughput_rps'], best['throughput_rps'], False):>10} "
            f"{safe_delta(base['avg_ttft_ms'], best['avg_ttft_ms'], True):>10} "
            f"{safe_delta(base['avg_itl_ms_per_token'], best['avg_itl_ms_per_token'], True):>10} "
            f"{safe_delta(base['p95_latency_ms'], best['p95_latency_ms'], True):>10} "
            f"{safe_delta(base['makespan_s'], best['makespan_s'], True):>10}"
        )

    print("\n=== Global Best ===")
    print(
        f"interval={global_best['arrival_interval_ms']}ms, mode={global_best['mode']}, "
        f"chunked_prefill_size={global_best['chunked_prefill_size']}, "
        f"max_num_batched_tokens={global_best['max_num_batched_tokens']}, "
        f"reserve_ratio={global_best.get('cb_prefill_reserve_ratio')}, "
        f"reserve_min_tokens={global_best.get('cb_prefill_min_tokens')}, "
        f"reserve_min_seqs={global_best.get('cb_prefill_min_seqs')}, "
        f"dominates={global_best['dominates']}"
    )
    if found_dominating:
        print("Result: Found strict comprehensive dominance (throughput up, all key latencies down).")
    else:
        print("Result: No strict comprehensive dominance found; printed the best tradeoff candidate.")


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
    parser.add_argument("--optimize-on-on", action="store_true")
    parser.add_argument("--opt-arrival-interval-ms", type=str, default="")
    parser.add_argument("--opt-chunked-prefill-sizes", type=str, default="256,512,768,1024,1536,2048")
    parser.add_argument("--opt-max-num-batched-tokens", type=str, default="")
    parser.add_argument("--opt-cb-prefill-reserve-ratios", type=str, default="")
    parser.add_argument("--opt-cb-prefill-min-tokens", type=str, default="")
    parser.add_argument("--opt-cb-prefill-min-seqs", type=str, default="")
    parser.add_argument("--opt-stop-on-first-dominating", action="store_true")
    parser.add_argument("--trials-per-config", type=int, default=1)
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
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=200)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    args.enable_cb_prefill_liveness = True
    if args.disable_cb_prefill_liveness:
        args.enable_cb_prefill_liveness = False
    assert args.num_requests > 0, "num-requests must be > 0"
    assert args.arrival_interval_ms >= 0, "arrival-interval-ms must be >= 0"
    assert args.max_num_seqs > 0, "max-num-seqs must be > 0"
    assert args.max_num_batched_tokens > 0, "max-num-batched-tokens must be > 0"
    assert args.trials_per_config > 0, "trials-per-config must be > 0"
    assert 0.0 <= args.cb_prefill_reserve_ratio <= 1.0, "cb-prefill-reserve-ratio must be in [0, 1]"
    assert args.cb_prefill_min_tokens >= 0, "cb-prefill-min-tokens must be >= 0"
    assert args.cb_prefill_min_seqs >= 0, "cb-prefill-min-seqs must be >= 0"
    if args.workload_profile == "uniform":
        assert args.prompt_len > 0, "prompt-len must be > 0"
        assert args.output_len > 0, "output-len must be > 0"
    else:
        assert args.short_prompt_len > 0 and args.long_prompt_len > 0, "prompt lens must be > 0"
        assert args.short_output_len > 0 and args.long_output_len > 0, "output lens must be > 0"

    if args.optimize_on_on:
        interval_list = parse_int_csv(args.opt_arrival_interval_ms)
        if not interval_list:
            interval_list = [args.arrival_interval_ms]
        chunk_sizes = parse_int_csv(args.opt_chunked_prefill_sizes)
        if not chunk_sizes:
            chunk_sizes = [args.chunked_prefill_size]
        max_batched_tokens_list = parse_int_csv(args.opt_max_num_batched_tokens)
        if not max_batched_tokens_list:
            max_batched_tokens_list = [args.max_num_batched_tokens]
        reserve_ratios = parse_float_csv(args.opt_cb_prefill_reserve_ratios)
        if not reserve_ratios:
            reserve_ratios = [args.cb_prefill_reserve_ratio]
        reserve_min_tokens_list = parse_int_csv(args.opt_cb_prefill_min_tokens)
        if not reserve_min_tokens_list:
            reserve_min_tokens_list = [args.cb_prefill_min_tokens]
        reserve_min_seqs_list = parse_int_csv(args.opt_cb_prefill_min_seqs)
        if not reserve_min_seqs_list:
            reserve_min_seqs_list = [args.cb_prefill_min_seqs]

        assert all(x >= 0 for x in interval_list), "opt-arrival-interval-ms values must be >= 0"
        assert all(x > 0 for x in chunk_sizes), "opt-chunked-prefill-sizes values must be > 0"
        assert all(x > 0 for x in max_batched_tokens_list), "opt-max-num-batched-tokens values must be > 0"
        assert all(0.0 <= x <= 1.0 for x in reserve_ratios), "opt-cb-prefill-reserve-ratios must be in [0, 1]"
        assert all(x >= 0 for x in reserve_min_tokens_list), "opt-cb-prefill-min-tokens values must be >= 0"
        assert all(x >= 0 for x in reserve_min_seqs_list), "opt-cb-prefill-min-seqs values must be >= 0"

        scenarios = []
        all_candidates = []
        found_dominating_any = False

        for interval_ms in interval_list:
            local_args = argparse.Namespace(**vars(args))
            local_args.arrival_interval_ms = interval_ms
            requests, arrivals = build_workload(local_args)
            print_workload_summary(requests, arrivals, local_args.workload_profile, local_args.arrival_pattern)

            common_kwargs = dict(
                model=local_args.model,
                requests=requests,
                arrivals=arrivals,
                temperature=local_args.temperature,
                top_p=local_args.top_p,
                max_model_len=local_args.max_model_len,
                max_num_seqs=local_args.max_num_seqs,
                max_num_batched_tokens=local_args.max_num_batched_tokens,
                chunked_prefill_size=local_args.chunked_prefill_size,
                enable_cb_prefill_liveness=local_args.enable_cb_prefill_liveness,
                cb_prefill_reserve_ratio=local_args.cb_prefill_reserve_ratio,
                cb_prefill_min_tokens=local_args.cb_prefill_min_tokens,
                cb_prefill_min_seqs=local_args.cb_prefill_min_seqs,
                tensor_parallel_size=local_args.tensor_parallel_size,
                enforce_eager=local_args.enforce_eager,
                gpu_sample_interval_ms=local_args.gpu_sample_interval_ms,
                gpu_id=local_args.gpu_id,
            )

            print(f"Running baseline off_off (CB=OFF, CP=OFF), arrival={interval_ms}ms ...")
            baseline = run_trials(
                mode_name=f"off_off_i{interval_ms}",
                num_trials=args.trials_per_config,
                run_kwargs=dict(
                    common_kwargs,
                    enable_cb=False,
                    enable_chunked_prefill=False,
                ),
            )
            if "error" in baseline:
                raise RuntimeError(
                    f"Baseline OOM at arrival_interval_ms={interval_ms}. "
                    "Please reduce load or lower max-num-batched-tokens/max-num-seqs."
                )
            cleanup_between_runs()

            candidates = []
            for mbt in max_batched_tokens_list:
                for csz in chunk_sizes:
                    for rsv_ratio in reserve_ratios:
                        for rsv_min_tok in reserve_min_tokens_list:
                            for rsv_min_seq in reserve_min_seqs_list:
                                name = (
                                    f"on_on_i{interval_ms}_c{csz}_b{mbt}_"
                                    f"r{rsv_ratio:.2f}_t{rsv_min_tok}_s{rsv_min_seq}"
                                )
                                print(f"Running candidate {name} (CB=ON, CP=ON) ...")
                                cand_kwargs = dict(common_kwargs)
                                cand_kwargs["max_num_batched_tokens"] = mbt
                                cand_kwargs["chunked_prefill_size"] = csz
                                cand_kwargs["cb_prefill_reserve_ratio"] = rsv_ratio
                                cand_kwargs["cb_prefill_min_tokens"] = rsv_min_tok
                                cand_kwargs["cb_prefill_min_seqs"] = rsv_min_seq
                                r = run_trials(
                                    mode_name=name,
                                    num_trials=args.trials_per_config,
                                    run_kwargs=dict(
                                        cand_kwargs,
                                        enable_cb=True,
                                        enable_chunked_prefill=True,
                                    ),
                                )
                                if "error" in r:
                                    cleanup_between_runs()
                                    continue
                                r["chunked_prefill_size"] = csz
                                r["max_num_batched_tokens"] = mbt
                                r["cb_prefill_reserve_ratio"] = rsv_ratio
                                r["cb_prefill_min_tokens"] = rsv_min_tok
                                r["cb_prefill_min_seqs"] = rsv_min_seq
                                r["arrival_interval_ms"] = interval_ms
                                r["dominates"] = dominates(baseline, r)
                                r["score"] = dominance_score(baseline, r)
                                candidates.append(r)
                                all_candidates.append(r)
                                cleanup_between_runs()

            if not candidates:
                print(f"[WARN] No valid ON/ON candidate at arrival={interval_ms}ms (all failed/OOM).")
                continue

            dominating = [x for x in candidates if x["dominates"]]
            if dominating:
                best = max(dominating, key=lambda x: (x["throughput_rps"], x["score"]))
                found_dominating = True
            else:
                best = max(candidates, key=lambda x: x["score"])
                found_dominating = False
            found_dominating_any = found_dominating_any or found_dominating

            scenarios.append(
                {
                    "arrival_interval_ms": interval_ms,
                    "baseline": baseline,
                    "candidates": candidates,
                    "best": best,
                    "found_dominating": found_dominating,
                }
            )
            print_opt_report(baseline, candidates, best, found_dominating)

            if args.opt_stop_on_first_dominating and found_dominating:
                break

        if not scenarios:
            raise RuntimeError("No optimization scenario completed successfully.")

        if found_dominating_any:
            dominating_all = [x for x in all_candidates if x["dominates"]]
            global_best = max(dominating_all, key=lambda x: (x["throughput_rps"], x["score"]))
        else:
            global_best = max(all_candidates, key=lambda x: x["score"])

        print_opt_search_report(scenarios, global_best, found_dominating_any)
        print(
            "\nReproduce best pair run with:\n"
            "python bench_cb_online.py "
            f"--model {args.model} "
            f"--compare-mode pair "
            f"--arrival-interval-ms {global_best['arrival_interval_ms']} "
            f"--chunked-prefill-size {global_best['chunked_prefill_size']} "
            f"--max-num-batched-tokens {global_best['max_num_batched_tokens']} "
            f"--cb-prefill-reserve-ratio {global_best.get('cb_prefill_reserve_ratio', args.cb_prefill_reserve_ratio)} "
            f"--cb-prefill-min-tokens {global_best.get('cb_prefill_min_tokens', args.cb_prefill_min_tokens)} "
            f"--cb-prefill-min-seqs {global_best.get('cb_prefill_min_seqs', args.cb_prefill_min_seqs)} "
            f"--trials-per-config {args.trials_per_config} "
            f"--num-requests {args.num_requests} "
            f"--workload-profile {args.workload_profile} "
            f"--arrival-pattern {args.arrival_pattern}"
        )

        if args.output_json:
            payload = {
                "config": vars(args),
                "scenarios": scenarios,
                "global_best": global_best,
                "found_dominating_any": found_dominating_any,
                "timestamp": int(time.time()),
            }
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\nSaved report to {args.output_json}")
        return

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
        results[name] = run_trials(
            mode_name=name,
            num_trials=args.trials_per_config,
            run_kwargs=dict(
                common_kwargs,
                enable_cb=cb,
                enable_chunked_prefill=cp,
            ),
        )
        if "error" in results[name]:
            raise RuntimeError(f"{name} failed: {results[name]['error']}")
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
