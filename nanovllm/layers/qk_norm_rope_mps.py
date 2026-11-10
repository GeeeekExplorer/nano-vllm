"""Fused Q/K normalization and rotary embedding for MPS."""

from functools import lru_cache

import torch


_METAL_TYPES = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}

_SOURCE = """
#include <metal_stdlib>
using namespace metal;

#define HEAD_DIM HEAD_DIM_VALUE
#define HALF_DIM (HEAD_DIM / 2)
#define PAIRS_PER_LANE (HEAD_DIM / 64)

// One SIMD group handles one token/head. Each lane owns rotary pairs across both halves.
// Keep PyTorch's dtype rounding between normalization and rotation.
kernel void qk_norm_rope(
    device SCALAR_T* q_out,
    device SCALAR_T* k_out,
    device const SCALAR_T* qkv,
    device const SCALAR_T* q_weight,
    device const SCALAR_T* k_weight,
    device const float* cos_sin,
    device const long* positions,
    constant uint& num_q_heads,
    constant uint& num_k_heads,
    constant uint& row_stride,
    constant float& eps,
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint heads = num_q_heads + num_k_heads;
    uint token = group / heads;
    uint head = group % heads;
    bool is_q = head < num_q_heads;
    uint in_base = token * row_stride + head * HEAD_DIM;
    uint rope_base = uint(positions[token]) * HEAD_DIM;

    float first[PAIRS_PER_LANE];
    float second[PAIRS_PER_LANE];
    float partial = 0.0f;
    for (uint i = 0; i < PAIRS_PER_LANE; ++i) {
        uint feature = lane + i * 32;
        first[i] = float(qkv[in_base + feature]);
        second[i] = float(qkv[in_base + feature + HALF_DIM]);
        partial += first[i] * first[i] + second[i] * second[i];
    }
    float inv_rms = rsqrt(simd_sum(partial) / float(HEAD_DIM) + eps);

    for (uint i = 0; i < PAIRS_PER_LANE; ++i) {
        uint feature = lane + i * 32;
        float w1 = is_q ? float(q_weight[feature]) : float(k_weight[feature]);
        float w2 = is_q ? float(q_weight[feature + HALF_DIM]) : float(k_weight[feature + HALF_DIM]);
        float x1 = float(SCALAR_T(float(SCALAR_T(first[i] * inv_rms)) * w1));
        float x2 = float(SCALAR_T(float(SCALAR_T(second[i] * inv_rms)) * w2));
        float c = cos_sin[rope_base + feature];
        float s = cos_sin[rope_base + feature + HALF_DIM];
        SCALAR_T y1 = SCALAR_T(x1 * c - x2 * s);
        SCALAR_T y2 = SCALAR_T(x2 * c + x1 * s);
        if (is_q) {
            uint out_base = (token * num_q_heads + head) * HEAD_DIM + feature;
            q_out[out_base] = y1;
            q_out[out_base + HALF_DIM] = y2;
        } else {
            uint out_base = (token * num_k_heads + head - num_q_heads) * HEAD_DIM + feature;
            k_out[out_base] = y1;
            k_out[out_base + HALF_DIM] = y2;
        }
    }
}
"""


@lru_cache(maxsize=12)
def _shader(dtype: torch.dtype, head_dim: int):
    source = _SOURCE.replace("HEAD_DIM_VALUE", str(head_dim)).replace("SCALAR_T", _METAL_TYPES[dtype])
    return torch.mps.compile_shader(source)


def supports_qk_norm_rope_mps(qkv: torch.Tensor, head_dim: int, cos_sin_cache: torch.Tensor) -> bool:
    return (qkv.device.type == "mps" and qkv.dtype in _METAL_TYPES and qkv.ndim == 2 and qkv.is_contiguous()
            and head_dim % 64 == 0 and cos_sin_cache.dtype == torch.float32
            and hasattr(torch.mps, "compile_shader"))


def qk_norm_rope_mps(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read Q and K from the start of each QKV row; return rotated Q and K."""
    if not supports_qk_norm_rope_mps(qkv, head_dim, cos_sin_cache):
        raise ValueError("unsupported inputs for the fused Q/K norm and rotary kernel")
    num_tokens, row_stride = qkv.shape
    heads = num_q_heads + num_k_heads
    if row_stride < heads * head_dim:
        raise ValueError("qkv rows are too short for the requested heads")
    if positions.dtype != torch.int64 or positions.shape != (num_tokens,) or not positions.is_contiguous():
        raise ValueError("positions must be contiguous int64 with one entry per token")
    if cos_sin_cache.shape[-1] != head_dim or cos_sin_cache.numel() % head_dim or not cos_sin_cache.is_contiguous():
        raise ValueError("cos_sin_cache must hold head_dim values per position")
    for weight in (q_weight, k_weight):
        if weight.shape != (head_dim,) or weight.dtype != qkv.dtype or not weight.is_contiguous():
            raise ValueError("norm weights must be contiguous [head_dim] tensors of the qkv dtype")
    q_out = torch.empty(num_tokens, num_q_heads, head_dim, dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty(num_tokens, num_k_heads, head_dim, dtype=qkv.dtype, device=qkv.device)
    if num_tokens == 0:
        return q_out, k_out
    _shader(qkv.dtype, head_dim).qk_norm_rope(
        q_out, k_out, qkv, q_weight, k_weight, cos_sin_cache, positions,
        num_q_heads, num_k_heads, row_stride, eps,
        threads=(num_tokens * heads * 32,), group_size=(32,),
    )
    return q_out, k_out
