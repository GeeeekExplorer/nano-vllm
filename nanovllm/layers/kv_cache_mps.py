"""Write K/V rows into a paged Metal cache."""

from functools import lru_cache

import torch


_METAL_TYPES = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}

_SOURCE = """
kernel void store_kvcache(
    device const SCALAR_T* key,
    device const SCALAR_T* value,
    device SCALAR_T* k_cache,
    device SCALAR_T* v_cache,
    device const int* slot_mapping,
    constant uint& num_tokens,
    constant uint& width,
    constant uint& key_stride,
    constant uint& value_stride,
    constant uint& num_slots,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_tokens * width) return;

    uint token = tid / width;
    uint feature = tid % width;
    int slot = slot_mapping[token];
    if (slot < 0 || uint(slot) >= num_slots) return;

    uint cache_offset = uint(slot) * width + feature;
    k_cache[cache_offset] = key[token * key_stride + feature];
    v_cache[cache_offset] = value[token * value_stride + feature];
}
"""


@lru_cache(maxsize=len(_METAL_TYPES))
def _shader(dtype: torch.dtype):
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("this PyTorch build does not provide torch.mps.compile_shader")
    return torch.mps.compile_shader(_SOURCE.replace("SCALAR_T", _METAL_TYPES[dtype]))


def store_kvcache_mps(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("key and value must have shape [tokens, heads, head_dim]")
    num_tokens, num_heads, head_dim = key.shape
    width = num_heads * head_dim
    tensors = (key, value, k_cache, v_cache, slot_mapping)
    if any(t.device.type != "mps" for t in tensors):
        raise ValueError("all KV-cache tensors must be on MPS")
    if key.dtype not in _METAL_TYPES or any(t.dtype != key.dtype for t in (value, k_cache, v_cache)):
        raise ValueError("key, value, and caches must share a supported floating-point dtype")
    if slot_mapping.dtype != torch.int32 or slot_mapping.shape != (num_tokens,) or not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be contiguous int32 with one slot per token")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape or k_cache.shape[2:] != (num_heads, head_dim):
        raise ValueError("caches must have matching [blocks, block_size, heads, head_dim] shapes")
    if not k_cache.is_contiguous() or not v_cache.is_contiguous():
        raise ValueError("KV caches must be contiguous")
    if (key.stride(1), key.stride(2)) != (head_dim, 1):
        raise ValueError("each key token must have contiguous head data")
    if (value.stride(1), value.stride(2)) != (head_dim, 1):
        raise ValueError("each value token must have contiguous head data")
    if num_tokens == 0:
        return

    num_slots = k_cache.shape[0] * k_cache.shape[1]
    _shader(key.dtype).store_kvcache(
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_tokens,
        width,
        key.stride(0),
        value.stride(0),
        num_slots,
    )
