import os
import sys
import time
import argparse
import subprocess
import json
import tempfile
from random import randint, seed
import torch
from nanovllm import LLM, SamplingParams
import nanovllm.layers.linear as _linear_module


def run_benchmark(model_path: str, quantization: str | None, num_seqs: int,
                  enforce_eager: bool, result_file: str | None = None):
    seed(0)
    _linear_module._INT8_QUANTIZED = False

    quant_str = quantization.upper() if quantization else "FP16"
    print(f"\n{'=' * 60}")
    print(f"Running Benchmark: {quant_str} (enforce_eager={enforce_eager})")
    print(f"{'=' * 60}")

    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        max_model_len=4096,
        quantization=quantization,
    )

    num_kv_blocks = llm.model_runner.config.num_kvcache_blocks
    print(f"KV Cache Blocks Allocated: {num_kv_blocks}")

    max_input_len = 1024
    max_output_len = 1024
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len))
        for _ in range(num_seqs)
    ]

    # Warmup
    print("Warming up engine...")
    llm.generate(["Benchmark: "], SamplingParams())

    print(f"Benchmarking {num_seqs} sequences...")
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    # Live tqdm progress bar with Prefill and Decode tok/s
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    elapsed = time.time() - t_start

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / elapsed
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    result = {
        "mode": quant_str,
        "tokens": total_tokens,
        "time": elapsed,
        "throughput": throughput,
        "peak_vram_gb": peak_vram_gb,
        "kv_blocks": num_kv_blocks,
    }

    print(f"\nResult [{quant_str}]:")
    print(f"  - Total Generated Tokens: {total_tokens:,}")
    print(f"  - Total Time:            {elapsed:.2f} s")
    print(f"  - Throughput:            {throughput:.2f} tok/s")
    print(f"  - Peak Memory:           {peak_vram_gb:.2f} GB")
    print(f"  - KV Cache Blocks:       {num_kv_blocks}")

    if result_file:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f)

    llm.exit()
    del llm
    torch.cuda.empty_cache()

    return result


def run_isolated(model_path: str, quantization: str | None, num_seqs: int, enforce_eager: bool):
    """Run benchmark in an isolated subprocess with live terminal progress bar."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        result_file = tmp.name

    try:
        cmd = [
            sys.executable,
            __file__,
            "--model", model_path,
            "--num-seqs", str(num_seqs),
            "--result-file", result_file,
        ]
        if quantization:
            cmd += ["--quantization", quantization]
        if enforce_eager:
            cmd += ["--enforce-eager"]

        # stdout and stderr stream directly to terminal so tqdm animates live
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Subprocess failed for quantization={quantization}")

        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        if os.path.exists(result_file):
            try:
                os.remove(result_file)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Benchmark Nano-vLLM (FP16 vs INT8)")
    parser.add_argument("--model", type=str, default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "int8"],
                        help="Run only one mode: None (FP16) or 'int8'")
    parser.add_argument("--compare", action="store_true", default=True,
                        help="Run both FP16 and INT8 in isolated subprocesses and compare (default: True)")
    parser.add_argument("--num-seqs", type=int, default=64,
                        help="Number of sequences to benchmark (default: 64)")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Enforce eager execution (disable CUDA graphs)")
    parser.add_argument("--result-file", type=str, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()
    model_path = os.path.expanduser(args.model)

    # Subprocess execution worker
    if args.result_file:
        run_benchmark(
            model_path,
            quantization=args.quantization,
            num_seqs=args.num_seqs,
            enforce_eager=args.enforce_eager,
            result_file=args.result_file,
        )
        return

    # Single-mode run
    if args.quantization is not None:
        run_benchmark(
            model_path,
            quantization=args.quantization,
            num_seqs=args.num_seqs,
            enforce_eager=args.enforce_eager,
        )
        return

    # Comparison run in isolated subprocesses (clean VRAM for both)
    print("\n[1/2] Running FP16 benchmark in isolated process...")
    res_fp16 = run_isolated(model_path, None, args.num_seqs, args.enforce_eager)

    print("\n[2/2] Running INT8 benchmark in isolated process...")
    res_int8 = run_isolated(model_path, "int8", args.num_seqs, args.enforce_eager)

    # Comparison summary table
    print(f"\n{'=' * 65}")
    print(f"{'COMPARISON SUMMARY':^65}")
    print(f"{'=' * 65}")
    print(f"{'Metric':<25} | {'FP16':<15} | {'INT8':<15} | {'Delta':<10}")
    print(f"{'-' * 65}")
    print(f"{'Throughput (tok/s)':<25} | {res_fp16['throughput']:<15.2f} | {res_int8['throughput']:<15.2f} | {res_int8['throughput']/res_fp16['throughput']:.2f}x")
    print(f"{'Elapsed Time (s)':<25} | {res_fp16['time']:<15.2f} | {res_int8['time']:<15.2f} | {res_int8['time']/res_fp16['time']:.2f}x")
    print(f"{'KV Cache Blocks':<25} | {res_fp16['kv_blocks']:<15} | {res_int8['kv_blocks']:<15} | {res_int8['kv_blocks']/res_fp16['kv_blocks']:.2f}x")
    print(f"{'Peak VRAM (GB)':<25} | {res_fp16['peak_vram_gb']:<15.2f} | {res_int8['peak_vram_gb']:<15.2f} | -")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
