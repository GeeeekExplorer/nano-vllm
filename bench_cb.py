"""
Continuous Batching isolation benchmark.

Demonstrates CB value with uniform-length online arrivals (CP OFF).
Legacy mode alternates prefill/decode batches; CB mode mixes them per step,
filling decode gaps with new prefill immediately.

Usage:
    python bench_cb.py
    python bench_cb.py --num-requests 200 --arrival-interval-ms 30
"""
import os
import random
import time
from statistics import mean

import torch

from nanovllm import LLM, SamplingParams


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    w = rank - low
    return ordered[low] * (1.0 - w) + ordered[high] * w


def build_workload(num_requests: int, prompt_len: int, output_len: int,
                   arrival_interval_ms: int, seed: int = 0):
    rng = random.Random(seed)
    prompts = [[rng.randint(0, 10000) for _ in range(prompt_len)]
               for _ in range(num_requests)]
    # Poisson inter-arrival
    arrivals = [0.0]
    mean_s = arrival_interval_ms / 1000.0
    t = 0.0
    for _ in range(num_requests - 1):
        t += rng.expovariate(1.0 / mean_s) if mean_s > 0 else 0.0
        arrivals.append(t)
    return prompts, arrivals, output_len


def run_case(
    model_path: str,
    prompts: list[list[int]],
    arrivals: list[float],
    output_len: int,
    enable_cb: bool,
    max_num_batched_tokens: int,
    max_num_seqs: int,
):
    llm = LLM(
        model_path,
        enforce_eager=False,
        max_model_len=4096,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        enable_continuous_batching=enable_cb,
        enable_chunked_prefill=False,      # CP OFF to isolate CB
    )
    sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=output_len)

    # Warmup
    llm.generate([[1, 2, 3, 4]], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1), use_tqdm=False)
    torch.cuda.synchronize()

    # Per-request tracking
    seq_id_map: dict[int, int] = {}        # seq_id -> request index
    arrival_ts: dict[int, float] = {}      # req_idx -> wall time
    first_token_ts: dict[int, float] = {}  # req_idx -> wall time
    finish_ts: dict[int, float] = {}       # req_idx -> wall time
    token_counts: dict[int, int] = {}      # req_idx -> generated tokens
    last_token_ts: dict[int, float] = {}   # req_idx -> wall time
    itl_sum: dict[int, float] = {}         # req_idx -> sum of inter-token latency

    total = len(prompts)
    next_idx = 0
    done = 0

    t0 = time.perf_counter()
    while done < total:
        now = time.perf_counter()
        # Inject requests that have arrived
        while next_idx < total and (now - t0) >= arrivals[next_idx]:
            seq = llm.add_request(prompts[next_idx], sp)
            seq_id_map[seq.seq_id] = next_idx
            arrival_ts[next_idx] = now
            token_counts[next_idx] = 0
            itl_sum[next_idx] = 0.0
            next_idx += 1

        # If engine idle and more requests pending, sleep briefly
        if llm.is_finished():
            if next_idx >= total:
                break
            sleep_s = max(0.0, t0 + arrivals[next_idx] - time.perf_counter())
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.001))
            continue

        _, _, metadata = llm.step(return_metadata=True)
        step_ts = time.perf_counter()

        for seq_id, should_sample in zip(metadata["seq_ids"], metadata["should_sample"]):
            if not should_sample:
                continue
            req_idx = seq_id_map.get(seq_id)
            if req_idx is None:
                continue
            token_counts[req_idx] = token_counts.get(req_idx, 0) + 1
            if req_idx not in first_token_ts:
                first_token_ts[req_idx] = step_ts
            if req_idx in last_token_ts:
                itl_sum[req_idx] += step_ts - last_token_ts[req_idx]
            last_token_ts[req_idx] = step_ts
            if token_counts[req_idx] >= output_len:
                finish_ts[req_idx] = step_ts
                done += 1

    t_end = time.perf_counter()
    llm.exit()

    # Compute metrics
    ttft_ms = []
    latency_ms = []
    itl_ms = []
    for i in range(total):
        if i in first_token_ts:
            ttft_ms.append((first_token_ts[i] - arrival_ts[i]) * 1000.0)
        if i in finish_ts:
            latency_ms.append((finish_ts[i] - arrival_ts[i]) * 1000.0)
        cnt = token_counts.get(i, 0)
        if cnt > 1 and itl_sum.get(i, 0) > 0:
            itl_ms.append(itl_sum[i] / (cnt - 1) * 1000.0)

    makespan_s = t_end - t0
    return {
        "avg_ttft_ms": mean(ttft_ms) if ttft_ms else float("nan"),
        "p95_ttft_ms": percentile(ttft_ms, 0.95),
        "avg_itl_ms": mean(itl_ms) if itl_ms else float("nan"),
        "avg_latency_ms": mean(latency_ms) if latency_ms else float("nan"),
        "p95_latency_ms": percentile(latency_ms, 0.95),
        "throughput_rps": total / makespan_s,
        "makespan_s": makespan_s,
    }


def fmt(v, d=2):
    if v != v:
        return "N/A"
    return f"{v:.{d}f}"


def delta(base, cur, lower_better):
    if base != base or cur != cur or abs(base) < 1e-9:
        return "N/A"
    pct = ((base - cur) / base * 100.0) if lower_better else ((cur - base) / base * 100.0)
    return f"{'+' if pct >= 0 else ''}{pct:.1f}%"


def print_report(off: dict, on: dict, num_requests: int, interval_ms: int,
                 prompt_len: int, output_len: int):
    print(f"\n=== Continuous Batching Isolation Benchmark ===")
    print(f"Requests: {num_requests}, Prompt len: {prompt_len}, "
          f"Output len: {output_len}, Arrival interval: {interval_ms}ms (Poisson)")
    print(f"CP=OFF for both runs\n")
    rows = [
        ("Avg TTFT (ms)",      off["avg_ttft_ms"],      on["avg_ttft_ms"],      True),
        ("P95 TTFT (ms)",      off["p95_ttft_ms"],      on["p95_ttft_ms"],      True),
        ("Avg ITL (ms/tok)",   off["avg_itl_ms"],       on["avg_itl_ms"],       True),
        ("Avg Latency (ms)",   off["avg_latency_ms"],   on["avg_latency_ms"],   True),
        ("P95 Latency (ms)",   off["p95_latency_ms"],   on["p95_latency_ms"],   True),
        ("Throughput (req/s)", off["throughput_rps"],    on["throughput_rps"],    False),
        ("Makespan (s)",       off["makespan_s"],        on["makespan_s"],        True),
    ]
    print(f"{'Metric':<22} {'CB=OFF':>14} {'CB=ON':>14} {'Improvement':>14}")
    print("-" * 68)
    for name, v_off, v_on, lb in rows:
        print(f"{name:<22} {fmt(v_off):>14} {fmt(v_on):>14} {delta(v_off, v_on, lb):>14}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CB isolation benchmark (CP OFF)")
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--num-requests", type=int, default=300)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--arrival-interval-ms", type=int, default=40)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prompts, arrivals, output_len = build_workload(
        args.num_requests, args.prompt_len, args.output_len,
        args.arrival_interval_ms, args.seed,
    )

    common = dict(
        model_path=args.model,
        prompts=prompts,
        arrivals=arrivals,
        output_len=output_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )

    print("Running CB=OFF (legacy) ...")
    off = run_case(**common, enable_cb=False)
    torch.cuda.empty_cache()

    print("Running CB=ON ...")
    on = run_case(**common, enable_cb=True)

    print_report(off, on, args.num_requests, args.arrival_interval_ms,
                 args.prompt_len, args.output_len)


if __name__ == "__main__":
    main()


