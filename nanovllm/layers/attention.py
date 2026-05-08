import torch
from torch import nn
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.tools.quant_attn_kvhead_based import decode_attn_quantkv_direct, \
    prefill_attn_quantkv_direct
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


@triton.jit
def _store_quantkv_kernel(
    key_ptr,    
    key_stride_b,
    key_stride_h,
    key_stride_d,
    value_ptr,  
    value_stride_b,
    value_stride_h,
    value_stride_d,
    k_cache_ptr,    
    v_cache_ptr,    
    k_scale_ptr,         
    v_scale_ptr,         
    slot_mapping_ptr,   
    stride_cache_h,      
    stride_cache_s,      
    stride_cache_d,      
    stride_scale_h,      
    stride_scale_s,      
    EPS: tl.constexpr,      
    BLOCK_H: tl.constexpr,  
    BLOCK_D: tl.constexpr,  
    TARGET_DTYPE: tl.constexpr,     
):
    token_idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot == -1:
        return
    offs_head = tl.arange(0, BLOCK_H)
    offs_dim = tl.arange(0, BLOCK_D)
    # k
    kv_offsets = token_idx * key_stride_b + offs_head[:, None] * key_stride_h + offs_dim[None, :] * key_stride_d
    kv = tl.load(key_ptr + kv_offsets)
    kv_abs_max = tl.max(tl.abs(kv), axis=1)
    kv_scale = tl.maximum(kv_abs_max / 127.0, EPS).to(TARGET_DTYPE)
    kv = tl.extra.cuda.libdevice.llrint(kv / kv_scale[:, None])
    kv = tl.maximum(tl.minimum(kv, 127), -127).to(tl.int8)
    cache_offsets = offs_head[:, None] * stride_cache_h + slot * stride_cache_s + offs_dim[None, :] * stride_cache_d
    tl.store(k_cache_ptr + cache_offsets, kv)
    scale_offsets = offs_head * stride_scale_h + slot * stride_scale_s
    tl.store(k_scale_ptr + scale_offsets, kv_scale)
    # v
    kv_offsets = token_idx * value_stride_b + offs_head[:, None] * value_stride_h + offs_dim[None, :] * value_stride_d
    kv = tl.load(value_ptr + kv_offsets)
    kv_abs_max = tl.max(tl.abs(kv), axis=1)
    kv_scale = tl.maximum(kv_abs_max/127.0, EPS).to(TARGET_DTYPE)
    kv = tl.extra.cuda.libdevice.llrint(kv / kv_scale[:, None])
    kv = tl.maximum(tl.minimum(kv, 127), -127).to(tl.int8)
    tl.store(v_cache_ptr + cache_offsets, kv)
    tl.store(v_scale_ptr + scale_offsets, kv_scale)

def store_quantkv(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    n, num_kv_heads, head_dim = key.shape
    if key.dtype == torch.float16:
        target_dtype = tl.float16
    elif key.dtype == torch.bfloat16:
        target_dtype = tl.bfloat16
    else:
        target_dtype = tl.float32
    _store_quantkv_kernel[(n,)](
        key,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_scale_cache.stride(0),
        k_scale_cache.stride(1),
        EPS=1e-8,
        BLOCK_H=num_kv_heads,
        BLOCK_D=head_dim,
        TARGET_DTYPE=target_dtype,
        num_warps = 1,
        num_stages = 1
    )

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale_cache = self.v_scale_cache = torch.tensor([])
        self.block_size = 256
        self.kv_quant = True
        self.GROUP_NUM = 2
        self.kv_store_stream = None
        self.kv_store_event = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if self.kv_quant:
            o = self._forward_quantized(q,k,v)
        else:
            o = self._forward_unquantized(q,k,v)
        return o

    def _forward_unquantized(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:  
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o

    def _forward_quantized(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        k_scale_cache, v_scale_cache = self.k_scale_cache, self.v_scale_cache
        if context.is_prefill:
            if context.block_tables is not None:
                with torch.cuda.nvtx.range("PreFlashQuantDir"):
                    o = prefill_attn_quantkv_direct(q, k, v, k_cache, v_cache, k_scale_cache, v_scale_cache,
                                                    max_seqlen_q=context.max_seqlen_q,
                                                    cu_seqlens_q=context.cu_seqlens_q,
                                                    context_lens=context.context_lens,
                                                    block_table=context.block_tables, block_size=self.block_size,
                                                    GROUP_NUM=self.GROUP_NUM,
                                                    softmax_scale=self.scale, causal=True)
            else:
                with torch.cuda.nvtx.range("PreFlashOri"):
                    o = flash_attn_varlen_func(q, k, v,
                                               max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                               max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                               softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:  # decode
            with torch.cuda.nvtx.range("DeFlashQuantDir"):
                o = decode_attn_quantkv_direct(q, k, v, k_cache, v_cache, k_scale_cache, v_scale_cache,
                                              context_lens=context.context_lens, block_table=context.block_tables,
                                              block_size=self.block_size,
                                              GROUP_NUM=self.GROUP_NUM,
                                              softmax_scale=self.scale, )
        if k_cache.numel() and v_cache.numel():
            if torch.cuda.is_current_stream_capturing():
                self.kv_store_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.kv_store_stream):
                with torch.cuda.nvtx.range("DeStore"):
                    store_quantkv(k, v, k_cache, v_cache, k_scale_cache, v_scale_cache, context.slot_mapping, )
            self.kv_store_event.record(self.kv_store_stream)
        return o


