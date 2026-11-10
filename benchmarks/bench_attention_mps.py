import argparse
from statistics import median
from time import perf_counter

import torch

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import reset_context, set_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if min(args.batch_size, args.context_len, args.repeat) < 1:
        raise ValueError("batch size, context length, and repeat count must be positive")

    block_size = 256
    kv_heads, q_heads, head_dim = 8, 16, 128
    blocks_per_request = (args.context_len + block_size - 1) // block_size
    lengths = [max(1, args.context_len - (i % 4) * min(64, args.context_len // 4))
               for i in range(args.batch_size)]
    tables = torch.arange(args.batch_size * blocks_per_request, device="mps", dtype=torch.int32)
    tables = tables.reshape(args.batch_size, blocks_per_request)
    slots = [i * blocks_per_request * block_size + length - 1 for i, length in enumerate(lengths)]
    attention = Attention(q_heads, head_dim, head_dim ** -0.5, kv_heads)
    cache_shape = (args.batch_size * blocks_per_request, block_size, kv_heads, head_dim)
    attention.k_cache = torch.randn(cache_shape, device="mps", dtype=torch.bfloat16)
    attention.v_cache = torch.randn(cache_shape, device="mps", dtype=torch.bfloat16)
    query = torch.randn(args.batch_size, q_heads, head_dim, device="mps", dtype=torch.bfloat16)
    key = torch.randn(args.batch_size, kv_heads, head_dim, device="mps", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    set_context(False, slot_mapping=torch.tensor(slots, device="mps", dtype=torch.int32),
                block_tables=tables, context_lens=torch.tensor(lengths, device="mps", dtype=torch.int32),
                k_lens=lengths)
    try:
        for _ in range(5):
            attention(query, key, value)
        torch.mps.synchronize()
        timings = []
        for _ in range(args.repeat):
            start = perf_counter()
            attention(query, key, value)
            torch.mps.synchronize()
            timings.append(perf_counter() - start)
        print(f"batch={args.batch_size} context={args.context_len} "
              f"median={median(timings) * 1000:.3f} ms/layer")
    finally:
        reset_context()


if __name__ == "__main__":
    main()
