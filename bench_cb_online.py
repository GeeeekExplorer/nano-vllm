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
class RequestStat:
    req_id: int
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


def build_workload(
    num_requests: int,
    prompt_len: int,
    prompt_jitter: int,
    arrival_interval_ms: int,
    seed: int,
) -> tuple[list[list[int]], list[float]]:
    rng = random.Random(seed)
    prompts = []
    arrivals = []
    for i in range(num_requests):
        delta = rng.randint(-prompt_jitter, prompt_jitter) if prompt_jitter > 0 else 0
        cur_len = max(4, prompt_len + delta)
        prompt = [rng.randint(0, 10000) for _ in range(cur_len)]
        prompts.append(prompt)
        arrivals.append(i * arrival_interval_ms / 1000.0)
    return prompts, arrivals


def warmup(llm: LLM):
    llm.generate(
        [[1, 2, 3, 4, 5, 6, 7, 8]],
        SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, ignore_eos=True),
        use_tqdm=False,
    )


def run_once(
    *,
    model: str,
    enable_cb: bool,
    prompts: list[list[int]],
    arrivals: list[float],
    temperature: float,
    top_p: float,
    output_len: int,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    enable_chunked_prefill: bool,
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
    total_reqs = len(prompts)
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=output_len,
        ignore_eos=True,
    )

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
                seq = llm.add_request(prompts[next_idx], sp)
                arrival_ts = now
                stats[seq.seq_id] = RequestStat(req_id=next_idx, arrival_ts=arrival_ts)
                if first_arrival is None:
                    first_arrival = arrival_ts
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

            # 生成长度固定（ignore_eos=True + fixed max_tokens），用 token 数判断完成更稳健
            for st in stats.values():
                if st.finish_ts is None and st.generated_tokens >= output_len:
                    st.finish_ts = step_ts
                    done += 1

        t_end = time.perf_counter()
    finally:
        sampler.stop()
        llm.exit()

    if first_arrival is None:
        first_arrival = t0
    makespan_s = max(t_end - first_arrival, 1e-9)

    latencies_ms = []
    ttft_ms = []
    itl_ms = []
    for st in stats.values():
        if st.finish_ts is None:
            continue
        latencies_ms.append((st.finish_ts - st.arrival_ts) * 1000.0)
        if st.first_token_ts is not None:
            ttft_ms.append((st.first_token_ts - st.arrival_ts) * 1000.0)
        if st.inter_token_count > 0:
            itl_ms.append(st.inter_token_sum_s / st.inter_token_count * 1000.0)

    return {
        "mode": "CB_ON" if enable_cb else "CB_OFF",
        "num_requests": total_reqs,
        "arrival_interval_ms": arrivals[1] * 1000.0 - arrivals[0] * 1000.0 if total_reqs > 1 else 0.0,
        "throughput_rps": total_reqs / makespan_s,
        "avg_ttft_ms": statistics.mean(ttft_ms) if ttft_ms else float("nan"),
        "avg_itl_ms_per_token": statistics.mean(itl_ms) if itl_ms else float("nan"),
        "avg_latency_ms": statistics.mean(latencies_ms) if latencies_ms else float("nan"),
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "p99_latency_ms": percentile(latencies_ms, 99),
        "gpu_util_avg_percent": sampler.avg_util,
        "gpu_util_samples": len(sampler.samples),
        "gpu_util_error": sampler.error,
        "makespan_s": makespan_s,
    }


def fmt(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and (v != v):
        return "N/A"
    return f"{v:.{digits}f}"


def safe_delta(off: Optional[float], on: Optional[float], lower_better: bool) -> str:
    if off is None or on is None:
        return "N/A"
    if isinstance(off, float) and (off != off):
        return "N/A"
    if isinstance(on, float) and (on != on):
        return "N/A"
    if abs(off) < 1e-9:
        return "N/A"
    if lower_better:
        pct = (off - on) / off * 100.0
    else:
        pct = (on - off) / off * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def print_report(off: dict, on: dict):
    print("\n=== Online Streaming Workload: CB OFF vs CB ON ===")
    print(f"Requests: {off['num_requests']}, Arrival Interval: {off['arrival_interval_ms']:.1f} ms")
    print("")
    print(f"{'Metric':<28} {'CB_OFF':>14} {'CB_ON':>14} {'Improvement':>14}")
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

    if off["gpu_util_avg_percent"] is None or on["gpu_util_avg_percent"] is None:
        print("\n[GPU util] N/A")
        if off.get("gpu_util_error"):
            print(f"  CB_OFF sampler error: {off['gpu_util_error']}")
        if on.get("gpu_util_error"):
            print(f"  CB_ON  sampler error: {on['gpu_util_error']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark continuous batching on online streaming workload")
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--arrival-interval-ms", type=int, default=50)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--prompt-jitter", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--chunked-prefill-size", type=int, default=2048)
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
    assert args.prompt_len > 0, "prompt-len must be > 0"
    assert args.output_len > 0, "output-len must be > 0"
    prompts, arrivals = build_workload(
        num_requests=args.num_requests,
        prompt_len=args.prompt_len,
        prompt_jitter=args.prompt_jitter,
        arrival_interval_ms=args.arrival_interval_ms,
        seed=args.seed,
    )

    common_kwargs = dict(
        model=args.model,
        prompts=prompts,
        arrivals=arrivals,
        temperature=args.temperature,
        top_p=args.top_p,
        output_len=args.output_len,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=args.enable_chunked_prefill,
        chunked_prefill_size=args.chunked_prefill_size,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        gpu_sample_interval_ms=args.gpu_sample_interval_ms,
        gpu_id=args.gpu_id,
    )

    print("Running CB_OFF ...")
    off = run_once(enable_cb=False, **common_kwargs)
    print("Running CB_ON ...")
    on = run_once(enable_cb=True, **common_kwargs)

    print_report(off, on)

    if args.output_json:
        payload = {
            "config": vars(args),
            "cb_off": off,
            "cb_on": on,
            "timestamp": int(time.time()),
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report to {args.output_json}")


if __name__ == "__main__":
    main()
