import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

# Triton JIT 编译的 kernel，用于将 key/value 写入 KV Cache
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
    idx = tl.program_id(0)  # 当前线程处理的样本索引
    slot = tl.load(slot_mapping_ptr + idx)  # 获取该样本对应的 cache slot
    if slot == -1: return  # -1 表示无效 slot，直接跳过
    key_offsets = idx * key_stride + tl.arange(0, D)  # 计算 key 的偏移
    value_offsets = idx * value_stride + tl.arange(0, D)  # 计算 value 的偏移
    key = tl.load(key_ptr + key_offsets)  # 读取 key
    value = tl.load(value_ptr + value_offsets)  # 读取 value
    cache_offsets = slot * D + tl.arange(0, D)  # 计算 cache 的偏移
    tl.store(k_cache_ptr + cache_offsets, key)  # 写入 key cache
    tl.store(v_cache_ptr + cache_offsets, value)  # 写入 value cache

# Python 封装，调用 Triton kernel 存储 KV Cache
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 检查张量布局和大小
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 启动 Triton kernel
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

# 注意力层实现，支持 KV Cache 和 FlashAttention
class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = head_dim    # 每个头的维度
        self.scale = scale          # softmax 缩放因子
        self.num_kv_heads = num_kv_heads  # KV 头数
        self.k_cache = self.v_cache = torch.tensor([])  # 初始化 KV Cache

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()  # 获取当前推理上下文
        k_cache, v_cache = self.k_cache, self.v_cache
        # 如果 cache 已经分配，写入当前步的 key/value
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        # prefill 阶段（批量填充），或 decode 阶段（单步生成）
        if context.is_prefill:
            if context.block_tables is not None:    # 如果有 prefix cache，直接用 cache
                k, v = k_cache, v_cache
            # 调用 FlashAttention 变长实现，支持高效批量推理
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode 阶段，单步生成，直接用 KV Cache
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o  # 返回注意力输出
