import argparse
import time
from pathlib import Path
from random import randint, seed

import torch

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(Path(__file__).resolve().parent / "Qwen3-0.6B"))
    parser.add_argument("--device", choices=("auto", "cuda", "mps"), default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    memory_utilization = args.gpu_memory_utilization
    if memory_utilization is None:
        memory_utilization = 0.64 if device == "mps" else 0.9
    if not 0 < memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be between 0 and 1")
    if min(args.max_num_seqs, args.max_num_batched_tokens) < 1:
        raise ValueError("scheduler limits must be positive")
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    llm = LLM(args.model, device=device, enforce_eager=False, max_model_len=4096,
              gpu_memory_utilization=memory_utilization,
              max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_num_batched_tokens)

    try:
        prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

        llm.generate(["Benchmark: "], SamplingParams())
        start = time.time()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        elapsed = time.time() - start
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        print(f"Total: {total_tokens}tok, Time: {elapsed:.2f}s, Throughput: {total_tokens / elapsed:.2f}tok/s")
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
