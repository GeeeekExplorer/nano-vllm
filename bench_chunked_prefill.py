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
    rank = (len(values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    ordered = sorted(values)
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def make_workload(
    seed: int,
    long_prompt_len: int,
    num_short: int,
    short_prompt_min_len: int,
    short_prompt_max_len: int,
    vocab_size: int = 10000,
):
    random.seed(seed)
    long_prompt = [random.randint(0, vocab_size) for _ in range(long_prompt_len)]
    short_prompts = [
        [random.randint(0, vocab_size) for _ in range(random.randint(short_prompt_min_len, short_prompt_max_len))]
        for _ in range(num_short)
    ]
    return long_prompt, short_prompts


def run_case(
    model_path: str,
    long_prompt: list[int],
    short_prompts: list[list[int]],
    enable_chunked_prefill: bool,
    chunked_prefill_size: int,
):
    long_max_tokens = 128
    short_max_tokens = 16
    max_model_len = len(long_prompt) + long_max_tokens if enable_chunked_prefill else len(long_prompt)
    llm = LLM(
        model_path,
        tensor_parallel_size=1,
        enforce_eager=False,
        max_model_len=max_model_len,
        max_num_batched_tokens=len(long_prompt),
        enable_chunked_prefill=enable_chunked_prefill,
        chunked_prefill_size=chunked_prefill_size,
    )

    # Warmup to reduce one-time compile / graph capture noise.
    llm.generate([[1, 2, 3, 4]], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1), use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    long_seq = llm.add_request(long_prompt, SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=long_max_tokens))
    short_seqs = [
        llm.add_request(prompt, SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=short_max_tokens))
        for prompt in short_prompts
    ]
    seqs = [long_seq] + short_seqs

    short_ids = {seq.seq_id for seq in short_seqs}
    first_scheduled_step: dict[int, int] = {}
    first_token_ms: dict[int, float] = {}
    finish_ms: dict[int, float] = {}

    step_idx = 0
    t0 = time.perf_counter()
    while not llm.is_finished():
        _, _, metadata = llm.step(return_metadata=True)
        step_idx += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        for seq_id in metadata["seq_ids"]:
            if seq_id in short_ids and seq_id not in first_scheduled_step:
                first_scheduled_step[seq_id] = step_idx

        for seq in seqs:
            if seq.seq_id not in first_token_ms and seq.num_completion_tokens > 0:
                first_token_ms[seq.seq_id] = elapsed_ms
            if seq.seq_id not in finish_ms and seq.is_finished:
                finish_ms[seq.seq_id] = elapsed_ms

    torch.cuda.synchronize()
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    long_ttft_ms = first_token_ms[long_seq.seq_id]
    long_done_ms = finish_ms[long_seq.seq_id]
    short_ttft_ms = [first_token_ms[seq.seq_id] for seq in short_seqs]
    blocked_short = sum(1 for seq in short_seqs if first_scheduled_step.get(seq.seq_id, 10**9) > 1)
    hol_blocking_rate = blocked_short / len(short_seqs)

    llm.exit()
    return {
        "long_ttft_ms": long_ttft_ms,
        "short_ttft_mean_ms": mean(short_ttft_ms),
        "short_ttft_p95_ms": percentile(short_ttft_ms, 0.95),
        "peak_memory_gb": peak_memory_gb,
        "hol_blocking_rate": hol_blocking_rate,
        "long_done_ms": long_done_ms,
    }


def print_result_table(full_prefill: dict, chunked_prefill: dict):
    print()
    print("=== Chunked Prefill 对比（1 x 32K Long + 32 x Short 并发）===")
    print("指标: 长请求 TTFT(ms), 短请求 TTFT 平均(ms), 峰值显存(GB), 队头阻塞率, 长请求完成时间(ms)")
    print()
    header = (
        f"{'方案':<20}"
        f"{'Long TTFT(ms)':>16}"
        f"{'Short TTFT Mean(ms)':>22}"
        f"{'Short TTFT P95(ms)':>20}"
        f"{'Peak Mem(GB)':>14}"
        f"{'HOL Rate':>12}"
        f"{'Long Done(ms)':>16}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'Full Prefill':<20}"
        f"{full_prefill['long_ttft_ms']:>16.2f}"
        f"{full_prefill['short_ttft_mean_ms']:>22.2f}"
        f"{full_prefill['short_ttft_p95_ms']:>20.2f}"
        f"{full_prefill['peak_memory_gb']:>14.2f}"
        f"{full_prefill['hol_blocking_rate']:>12.2%}"
        f"{full_prefill['long_done_ms']:>16.2f}"
    )
    print(
        f"{'Chunked Prefill(2048)':<20}"
        f"{chunked_prefill['long_ttft_ms']:>16.2f}"
        f"{chunked_prefill['short_ttft_mean_ms']:>22.2f}"
        f"{chunked_prefill['short_ttft_p95_ms']:>20.2f}"
        f"{chunked_prefill['peak_memory_gb']:>14.2f}"
        f"{chunked_prefill['hol_blocking_rate']:>12.2%}"
        f"{chunked_prefill['long_done_ms']:>16.2f}"
    )


def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    long_prompt, short_prompts = make_workload(
        seed=7,
        long_prompt_len=32768,
        num_short=32,
        short_prompt_min_len=32,
        short_prompt_max_len=256,
    )

    full_prefill = run_case(
        model_path=model_path,
        long_prompt=long_prompt,
        short_prompts=short_prompts,
        enable_chunked_prefill=False,
        chunked_prefill_size=2048,
    )
    chunked_prefill = run_case(
        model_path=model_path,
        long_prompt=long_prompt,
        short_prompts=short_prompts,
        enable_chunked_prefill=True,
        chunked_prefill_size=2048,
    )
    print_result_table(full_prefill, chunked_prefill)


if __name__ == "__main__":
    main()
