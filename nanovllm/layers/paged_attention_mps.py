"""Paged decode attention for MPS."""

from functools import lru_cache

import torch


_METAL_TYPES = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}

_SHARED_SOURCE = """
#include <metal_stdlib>
using namespace metal;

#define HEAD_DIM HEAD_DIM_VALUE
#define Q_PER_KV Q_PER_KV_VALUE
#define FEATURES_PER_LANE ((HEAD_DIM + 31) / 32)
#define PARALLEL_GROUPS 4

// One threadgroup handles a request/KV head and its query heads.
// Four SIMD groups split positions; lanes split features.
kernel void paged_decode_attention_shared(
    device SCALAR_T* output,
    device const SCALAR_T* query,
    device const SCALAR_T* k_cache,
    device const SCALAR_T* v_cache,
    device const int* block_tables,
    device const int* context_lens,
    constant uint& num_heads,
    constant uint& num_kv_heads,
    constant uint& block_size,
    constant uint& table_stride,
    constant float& scale,
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint request = group / num_kv_heads;
    uint kv_head = group % num_kv_heads;
    uint q_base = (request * num_heads + kv_head * Q_PER_KV) * HEAD_DIM;
    uint context_len = uint(context_lens[request]);
    uint row_stride = num_kv_heads * HEAD_DIM;
    uint table_base = request * table_stride;
    uint subgroup = tid / 32;
    uint lane = tid % 32;

    float q[Q_PER_KV * FEATURES_PER_LANE];
    float accumulator[Q_PER_KV * FEATURES_PER_LANE];
    float max_score[Q_PER_KV];
    float denominator[Q_PER_KV];
    for (uint g = 0; g < Q_PER_KV; ++g) {
        max_score[g] = -INFINITY;
        denominator[g] = 0.0f;
        for (uint i = 0; i < FEATURES_PER_LANE; ++i) {
            uint feature = lane + i * 32;
            accumulator[g * FEATURES_PER_LANE + i] = 0.0f;
            q[g * FEATURES_PER_LANE + i] = feature < HEAD_DIM ? float(query[q_base + g * HEAD_DIM + feature]) * scale : 0.0f;
        }
    }

    for (uint block_start = 0, b = 0; block_start < context_len; block_start += block_size, ++b) {
        uint block_base = uint(block_tables[table_base + b]) * block_size * row_stride + kv_head * HEAD_DIM + lane;
        uint count = min(block_size, context_len - block_start);
        // Preserve the position modulo four across block boundaries.
        uint first = (subgroup + PARALLEL_GROUPS - block_start % PARALLEL_GROUPS) % PARALLEL_GROUPS;
        for (uint j = first; j < count; j += PARALLEL_GROUPS) {
            uint base = block_base + j * row_stride;
            float key[FEATURES_PER_LANE];
            float value[FEATURES_PER_LANE];
            for (uint i = 0; i < FEATURES_PER_LANE; ++i) {
                bool valid = lane + i * 32 < HEAD_DIM;
                key[i] = valid ? float(k_cache[base + i * 32]) : 0.0f;
                value[i] = valid ? float(v_cache[base + i * 32]) : 0.0f;
            }
            for (uint g = 0; g < Q_PER_KV; ++g) {
                float partial = 0.0f;
                for (uint i = 0; i < FEATURES_PER_LANE; ++i) partial += q[g * FEATURES_PER_LANE + i] * key[i];
                float score = simd_sum(partial);
                float weight;
                if (score > max_score[g]) {
                    // Rescale the online softmax sums when the maximum changes.
                    float rescale = exp(max_score[g] - score);
                    denominator[g] *= rescale;
                    for (uint i = 0; i < FEATURES_PER_LANE; ++i) accumulator[g * FEATURES_PER_LANE + i] *= rescale;
                    max_score[g] = score;
                    weight = 1.0f;
                } else {
                    weight = exp(score - max_score[g]);
                }
                denominator[g] += weight;
                for (uint i = 0; i < FEATURES_PER_LANE; ++i) accumulator[g * FEATURES_PER_LANE + i] += weight * value[i];
            }
        }
    }

    threadgroup float partial_output[PARALLEL_GROUPS * Q_PER_KV * HEAD_DIM];
    threadgroup float partial_max[PARALLEL_GROUPS * Q_PER_KV];
    threadgroup float partial_denominator[PARALLEL_GROUPS * Q_PER_KV];
    for (uint g = 0; g < Q_PER_KV; ++g) {
        for (uint i = 0; i < FEATURES_PER_LANE; ++i) {
            uint feature = lane + i * 32;
            if (feature < HEAD_DIM) {
                partial_output[(subgroup * Q_PER_KV + g) * HEAD_DIM + feature] = accumulator[g * FEATURES_PER_LANE + i];
            }
        }
        if (lane == 0) {
            partial_max[subgroup * Q_PER_KV + g] = max_score[g];
            partial_denominator[subgroup * Q_PER_KV + g] = denominator[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (subgroup == 0) {
        for (uint g = 0; g < Q_PER_KV; ++g) {
            float common_max = -INFINITY;
            for (uint s = 0; s < PARALLEL_GROUPS; ++s) common_max = max(common_max, partial_max[s * Q_PER_KV + g]);
            float merge[PARALLEL_GROUPS];
            float common_denominator = 0.0f;
            for (uint s = 0; s < PARALLEL_GROUPS; ++s) {
                float part = partial_denominator[s * Q_PER_KV + g];
                merge[s] = part > 0.0f ? exp(partial_max[s * Q_PER_KV + g] - common_max) : 0.0f;
                common_denominator += part * merge[s];
            }
            for (uint i = 0; i < FEATURES_PER_LANE; ++i) {
                uint feature = lane + i * 32;
                if (feature < HEAD_DIM) {
                    float numerator = 0.0f;
                    for (uint s = 0; s < PARALLEL_GROUPS; ++s) {
                        numerator += partial_output[(s * Q_PER_KV + g) * HEAD_DIM + feature] * merge[s];
                    }
                    output[q_base + g * HEAD_DIM + feature] = context_len
                        ? SCALAR_T(numerator / common_denominator) : SCALAR_T(0);
                }
            }
        }
    }
}
"""

_SHARED_THREADS = 128


@lru_cache(maxsize=12)
def _shared_shader(dtype: torch.dtype, head_dim: int, q_per_kv: int):
    source = (_SHARED_SOURCE.replace("HEAD_DIM_VALUE", str(head_dim))
              .replace("Q_PER_KV_VALUE", str(q_per_kv))
              .replace("SCALAR_T", _METAL_TYPES[dtype]))
    return torch.mps.compile_shader(source)


def _validate_attention_inputs(query, k_cache, v_cache, block_tables, context_lens):
    if query.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("expected query [batch, q_heads, head_dim] and matching paged caches")
    batch, num_heads, head_dim = query.shape
    num_kv_heads = k_cache.shape[2]
    if num_kv_heads < 1 or k_cache.shape[3] != head_dim or num_heads % num_kv_heads:
        raise ValueError("query and KV head dimensions do not match")
    if query.dtype not in _METAL_TYPES or any(t.dtype != query.dtype for t in (k_cache, v_cache)):
        raise ValueError("query and KV caches must share a supported dtype")
    if any(t.device.type != "mps" for t in (query, k_cache, v_cache, block_tables, context_lens)):
        raise ValueError("all attention inputs must be on MPS")
    if not query.is_contiguous() or not k_cache.is_contiguous() or not v_cache.is_contiguous():
        raise ValueError("query and KV caches must be contiguous")
    if block_tables.dtype != torch.int32 or block_tables.ndim != 2 or block_tables.shape[0] != batch or not block_tables.is_contiguous():
        raise ValueError("block_tables must be contiguous int32 [batch, blocks]")
    if context_lens.dtype != torch.int32 or context_lens.shape != (batch,) or not context_lens.is_contiguous():
        raise ValueError("context_lens must be contiguous int32 [batch]")
    return batch, num_heads, head_dim, num_kv_heads


def paged_decode_attention_shared_mps(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Read each cached K/V row once per KV head."""
    batch, num_heads, head_dim, num_kv_heads = _validate_attention_inputs(
        query, k_cache, v_cache, block_tables, context_lens,
    )
    output = torch.empty_like(query)
    _shared_shader(query.dtype, head_dim, num_heads // num_kv_heads).paged_decode_attention_shared(
        output, query, k_cache, v_cache, block_tables, context_lens,
        num_heads, num_kv_heads, k_cache.shape[1], block_tables.shape[1], scale,
        threads=(batch * num_kv_heads * _SHARED_THREADS,), group_size=(_SHARED_THREADS,),
    )
    return output
