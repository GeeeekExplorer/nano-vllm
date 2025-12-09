#!/usr/bin/env python3
"""
Case study helper for M2 decode pipeline.

Runs a "small decode" and a "large decode" workload with multiple SM splits
so we can reason about Attention/FFN resource allocation impact.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import torch

from nanovllm import LLM, SamplingParams


SM_SPLIT_HELP = (
    "Comma separated list like '16:16,20:12'. Each entry is parsed as "
    "ATTN_SM:FFN_SM and the script will run the workloads for every split."
)


def parse_args():
    parser = argparse.ArgumentParser(description="M2 decode pipeline case study")
    parser.add_argument(
        "--model",
        default=os.path.expanduser("./Qwen3-0.6B/"),
        help="Path to Qwen3 checkpoint (default: ./Qwen3-0.6B/)",
    )
    parser.add_argument("--prefill-gpu", type=int, default=0, help="Prefill GPU id")
    parser.add_argument("--decode-gpu", type=int, default=1, help="Decode GPU id")
    parser.add_argument(
        "--sm-splits",
        default="16:16,20:12",
        help=SM_SPLIT_HELP,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for both cases",
    )
    return parser.parse_args()


def parse_sm_splits(raw: str) -> List[Tuple[int, int]]:
    splits = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            attn_str, ffn_str = item.split(":")
            splits.append((int(attn_str), int(ffn_str)))
        except ValueError as exc:
            raise ValueError(f"Invalid --sm-splits entry: {item}") from exc
    if not splits:
        raise ValueError("No SM splits provided")
    return splits


def build_small_case(temperature: float) -> tuple[list[str], SamplingParams]:
    prompts = [
        "Summarize the solar system.",
        "List two use-cases for reinforcement learning.",
    ]
    return prompts, SamplingParams(max_tokens=24, temperature=temperature)


def build_large_case(temperature: float) -> tuple[list[str], SamplingParams]:
    base_prompt = (
        "Explain how multi-stage GPU pipelines overlap attention and FFN work "
        "when decoding large batches of tokens."
    )
    prompts = [base_prompt] * 6
    return prompts, SamplingParams(max_tokens=192, temperature=temperature)


def run_case(
    case_name: str,
    prompts: list[str],
    sampling_params: SamplingParams,
    sm_splits: List[Tuple[int, int]],
    model_path: str,
    prefill_gpu: int,
    decode_gpu: int,
) -> list[dict]:
    print("\n" + "=" * 80)
    print(f"{case_name} case: {len(prompts)} prompts, max_tokens={sampling_params.max_tokens}")
    print("=" * 80)
    results = []
    for attn_sm, ffn_sm in sm_splits:
        print(f"\n→ Split Attention:FFN = {attn_sm}:{ffn_sm}")
        llm = None
        try:
            llm = LLM(
                model_path,
                enforce_eager=True,
                tensor_parallel_size=1,
                enable_two_gpu_pd=True,
                prefill_device_id=prefill_gpu,
                decode_device_id=decode_gpu,
                enable_decode_pipeline=True,
                decode_attention_sm=attn_sm,
                decode_ffn_sm=ffn_sm,
                decode_pipeline_profiling=True,
            )
            _ = llm.generate(prompts, sampling_params, use_tqdm=False)
            stats = llm.decode_runner.get_pipeline_statistics()
            llm.decode_runner.print_pipeline_statistics()
            results.append(
                {
                    "split": f"{attn_sm}:{ffn_sm}",
                    "avg_total_time_ms": stats.get("avg_total_time", 0.0) * 1000
                    if stats.get("total_tokens", 0) > 0
                    else None,
                    "avg_comm_mib": stats.get("avg_comm_bytes_per_token", 0.0) / (1024**2),
                    "comm_tokens": stats.get("comm_tokens", 0),
                    "error": None,
                }
            )
        except Exception as exc:
            print(f"✗ Split {attn_sm}:{ffn_sm} failed: {exc}")
            results.append(
                {
                    "split": f"{attn_sm}:{ffn_sm}",
                    "avg_total_time_ms": None,
                    "avg_comm_mib": 0.0,
                    "comm_tokens": 0,
                    "error": str(exc),
                }
            )
        finally:
            if llm is not None:
                llm.exit()
    return results


def main():
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("Need at least 2 CUDA GPUs for this script")

    sm_splits = parse_sm_splits(args.sm_splits)
    small_prompts, small_sampling = build_small_case(args.temperature)
    large_prompts, large_sampling = build_large_case(args.temperature)

    small_results = run_case(
        "Small decode",
        small_prompts,
        small_sampling,
        sm_splits,
        args.model,
        args.prefill_gpu,
        args.decode_gpu,
    )
    large_results = run_case(
        "Large decode",
        large_prompts,
        large_sampling,
        sm_splits,
        args.model,
        args.prefill_gpu,
        args.decode_gpu,
    )

    print("\n" + "=" * 80)
    print("Summary (per token averages)")
    print("=" * 80)
    print("Case           | Split  | Avg total time (ms) | Avg A→F payload (MiB) | Tokens")
    print("-" * 80)
    for label, rows in (
        ("Small decode", small_results),
        ("Large decode", large_results),
    ):
        for row in rows:
            if row.get("error"):
                print(f"{label:13} | {row['split']:>6} | ERROR: {row['error']}")
                continue
            if row["avg_total_time_ms"] is None:
                avg_time_str = "n/a".rjust(19)
            else:
                avg_time_str = f"{row['avg_total_time_ms']:19.3f}"
            comm_str = f"{row['avg_comm_mib']:20.4f}"
            print(
                f"{label:13} | {row['split']:>6} | {avg_time_str} | "
                f"{comm_str} | {row['comm_tokens']:>6}"
            )


if __name__ == "__main__":
    main()
