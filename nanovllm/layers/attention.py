import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        layer_name: str = "",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.layer_name = layer_name
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # --- PD 分离 (layerwise): 在注意力计算前等待该层 KV 加载完成 ---
        self.maybe_wait_for_layer_load()

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)

        # --- PD 分离 (layerwise): 注意力计算后立即保存该层 KV ---
        self.maybe_save_kv_layer()

        return o

    def maybe_wait_for_layer_load(self):
        """
        在注意力计算前等待该层的 KV Cache 从远端加载完成（layerwise 模式）。

        对应 vllm-ascend PR #950 中 Attention 层在 forward 开始时调用
        connector.wait_for_layer_load(layer_name)，确保 Decode 节点在执行
        注意力计算前，该层的 KV Cache 已从 Prefill 节点传输完成。

        执行逻辑:
            # 从全局 context 获取 pd_connector（如果存在）:
            # connector = get_context().pd_connector
            # if connector is None:
            #     return
            # connector.wait_for_layer_load(self.layer_name)
            #
            # 注意:
            #   - 仅在 layerwise 模式下生效
            #   - 非 layerwise 模式下，所有层的 KV 在 start_load_kv 时一次性加载
            #   - 该方法会阻塞当前线程直到接收线程完成该层的 get 操作
        """
        pass

    def maybe_save_kv_layer(self):
        """
        在注意力计算后立即保存该层的 KV Cache 到远端（layerwise 模式）。

        对应 vllm-ascend PR #950 中 Attention 层在 forward 结束时调用
        connector.save_kv_layer(layer_name, kv_layer)，在 Prefill 节点上
        每计算完一层就立即开始传输该层的 KV Cache，实现计算与传输的流水线。

        执行逻辑:
            # connector = get_context().pd_connector
            # if connector is None:
            #     return
            # kv_layer = (self.k_cache, self.v_cache)  # 该层的 KV Cache 引用
            # connector.save_kv_layer(self.layer_name, kv_layer)
            #
            # 注意:
            #   - 仅在 layerwise 模式下生效
            #   - 仅 kv_producer（Prefill 节点）执行保存
            #   - 使用 CUDA/NPU Event 确保本层计算完成后再开始传输
            #   - 非 layerwise 模式下，所有层的 KV 在 wait_for_save 时批量保存
        """
        pass
