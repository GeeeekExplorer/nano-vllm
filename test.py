#!/usr/bin/env python3
"""
Quick sanity test for Green Context splitting.

Usage:
  python test.py --device 0 --sm-splits 16 16
"""
from __future__ import annotations

import argparse
import sys
from typing import List

import torch

from green_ctx import split_device_green_ctx_by_sm_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Green Context splitting")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to split (default: 0)",
    )
    parser.add_argument(
        "--sm-splits",
        type=int,
        nargs="+",
        default=[16, 16],
        help="Requested SM counts for each partition",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=2048,
        help="Square matrix size for the sanity matmul on each stream",
    )
    return parser.parse_args()


def run_stream_matmul(stream: torch.cuda.Stream, dev: torch.device, size: int, tag: str) -> None:
    """Launch a simple matmul to verify the stream works."""
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(dev) >= (8, 0) else torch.float16
    with torch.cuda.stream(stream):
        a = torch.randn(size, size, device=dev, dtype=dtype)
        b = torch.randn(size, size, device=dev, dtype=dtype)
        c = a @ b
        c.sum().item()  # Force materialization
    print(f"[{tag}] matmul({size}x{size}) completed on stream {stream.cuda_stream}")


def main() -> int:

    
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available on this system.")
        return 1
    if args.device >= torch.cuda.device_count():
        print(f"Requested device {args.device}, but only {torch.cuda.device_count()} GPUs detected.")
        return 1

    dev = torch.device(f"cuda:{args.device}")
    print("=" * 80)
    print(f"Testing Green Context on GPU{args.device}: {torch.cuda.get_device_name(dev)}")
    print(f"Requested SM splits: {args.sm_splits}")

    try:
        streams, resources = split_device_green_ctx_by_sm_count(dev, args.sm_splits)
    except Exception as exc:  # noqa: BLE001
        print("Failed to create Green Context partitions:")
        print(f"  {exc}")
        print("Make sure CUDA 12.4+ and cuda-python are installed, and the GPU supports Green Context.")
        return 1

    actual_counts = [res.sm.smCount for res in resources]
    print(f"Actual SM allocation (last entry = remaining): {actual_counts}")

    for idx, stream in enumerate(streams[:-1]):  # skip remaining partition stream
        run_stream_matmul(stream, dev, args.matrix_size, f"Partition {idx} ({actual_counts[idx]} SMs)")

    torch.cuda.synchronize(dev)
    print("All stream tests finished successfully.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
