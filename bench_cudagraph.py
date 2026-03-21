"""
对比 CUDA Graphs（enforce_eager=False）和 Eager 模式（enforce_eager=True）的 decode 吞吐量。

测试重点在 decode 阶段，因为 CUDA Graphs 只在 decode 时启用（batch size <= 512 时）。
使用较多短 prompt + 较长 output，以放大 decode 阶段的耗时占比。
"""
import os
import time
from random import randint, seed
import torch.multiprocessing as mp

from nanovllm import LLM, SamplingParams


def _bench_worker(path: str, enforce_eager: bool, num_seqs: int,
                  max_input_len: int, max_output_len: int, result_queue):
    llm = LLM(path, enforce_eager=enforce_eager, max_model_len=4096)

    seed(42)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(32, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(64, max_output_len))
        for _ in range(num_seqs)
    ]

    # 预热，排除首次 CUDA 初始化的影响
    llm.generate(["warmup"], SamplingParams(temperature=0.6, max_tokens=4), use_tqdm=False)

    t = time.perf_counter()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - t

    total_output = sum(sp.max_tokens for sp in sampling_params)
    total_input = sum(len(p) for p in prompt_token_ids)
    throughput = total_output / elapsed

    result_queue.put({
        "total_input": total_input,
        "total_output": total_output,
        "elapsed": elapsed,
        "throughput": throughput,
    })


def run_bench(path: str, enforce_eager: bool, num_seqs: int,
              max_input_len: int, max_output_len: int) -> float:
    mode = "Eager (no CUDA Graphs)" if enforce_eager else "CUDA Graphs"
    print(f"\n{'='*60}")
    print(f"Mode: {mode}")
    print(f"{'='*60}")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_bench_worker,
        args=(path, enforce_eager, num_seqs, max_input_len, max_output_len, q),
    )
    p.start()
    p.join()

    result = q.get()
    print(f"  Seqs:              {num_seqs}")
    print(f"  Total input tok:   {result['total_input']}")
    print(f"  Total output tok:  {result['total_output']}")
    print(f"  Time:              {result['elapsed']:.2f}s")
    print(f"  Decode throughput: {result['throughput']:.1f} tok/s")
    return result["throughput"]


def main():
    path = os.path.expanduser("~/huggingface/Llama-3.2-1B-Instruct/")

    # decode 占比越高越能体现 CUDA Graphs 的优势：短 prompt + 长 output
    num_seqs = 128
    max_input_len = 128
    max_output_len = 512

    tp_eager = run_bench(path, enforce_eager=True,  num_seqs=num_seqs,
                         max_input_len=max_input_len, max_output_len=max_output_len)
    tp_graph = run_bench(path, enforce_eager=False, num_seqs=num_seqs,
                         max_input_len=max_input_len, max_output_len=max_output_len)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Eager:       {tp_eager:.1f} tok/s")
    print(f"  CUDA Graphs: {tp_graph:.1f} tok/s")
    speedup = tp_graph / tp_eager if tp_eager > 0 else 0
    print(f"  Speedup:     {speedup:.2f}x")


if __name__ == "__main__":
    main()
