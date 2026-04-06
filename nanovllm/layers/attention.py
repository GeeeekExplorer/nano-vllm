import torch
from torch import nn

from nanovllm.utils.context import get_context
from nanovllm.kvcache.base import BaseKVCache
from nanovllm.layers.flash_attn_backend import BaseFlashAttentionBackend, FlashAttentionRegistry


class Attention(nn.Module):
    """
    Attention module with pluggable KV cache and flash attention backends.
    
    This module supports:
    - Different KV cache implementations (FP16, INT8, INT4, etc.)
    - Different flash attention backends (standard, quantized-aware, custom)
    
    The separation allows you to:
    - Use INT8 cache with dequantization + standard flash-attn
    - OR use INT8 cache with custom INT8-aware flash-attn kernel
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        cache_backend: BaseKVCache | None = None,
        attn_backend: BaseFlashAttentionBackend | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.cache_backend = cache_backend
        
        # Flash attention backend - defaults to "default" implementation
        if attn_backend is None:
            # Import here to ensure registry is populated
            from nanovllm.layers.default_flash_attn import DefaultFlashAttention
            attn_backend = DefaultFlashAttention()
        self.attn_backend = attn_backend
        
        # Cache tensors (will be set by model runner)
        self.k_cache = self.v_cache = torch.tensor([])
        self.additional_cache_tensors = ()  # For quantized caches (scales, etc.)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # Store KV pairs in cache using the cache backend
        if k_cache.numel() and v_cache.numel():
            if self.cache_backend is not None:
                # Store supports additional tensors for quantized caches
                self.cache_backend.store(
                    k, v, k_cache, v_cache, context.slot_mapping,
                    *self.additional_cache_tensors
                )
            else:
                raise RuntimeError(
                    "No KV cache backend configured. This should not happen - "
                    "ensure DefaultKVCache is imported and registered."
                )
        
        # Perform attention using pluggable backend
        if context.is_prefill:
            # For prefill, use keys/values or cache depending on prefix caching
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
            
            o = self.attn_backend.prefill(
                q, k, v,
                scale=self.scale,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                block_table=context.block_tables,
            )
        else:  # decode
            # Pass all cache tensors to backend (quantized backends need scales, etc.)
            o = self.attn_backend.decode(
                q,
                k_cache,
                v_cache,
                scale=self.scale,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                *self.additional_cache_tensors,
            )
        
        return o
