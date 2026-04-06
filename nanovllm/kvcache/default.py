"""Default KV cache implementation for nano-vllm (no quantization)."""
import torch
import triton
import triton.language as tl
from typing import Tuple
from nanovllm.kvcache.base import BaseKVCache, KVCacheRegistry


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
    """
    Triton kernel for storing KV cache.
    This is the original nano-vllm implementation - no quantization.
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: 
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@KVCacheRegistry.register("default")
class DefaultKVCache(BaseKVCache):
    """
    Default KV cache - no quantization.
    
    This is the original nano-vllm implementation using Triton kernel
    for efficient KV cache storage. It stores keys and values in the
    same dtype as the model (typically fp16/bf16).
    """
    
    def allocate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Allocate cache tensors using model's dtype.
        
        Returns:
            Tuple of (k_cache, v_cache) tensors with shape:
            [num_blocks, block_size, num_heads, head_dim]
        """
        cache_shape = (
            self.num_blocks,
            self.block_size,
            self.num_heads,
            self.head_dim,
        )
        k_cache = torch.empty(cache_shape, dtype=self.dtype, device=self.device)
        v_cache = torch.empty(cache_shape, dtype=self.dtype, device=self.device)
        return k_cache, v_cache
    
    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """
        Store key-value pairs into cache using Triton kernel.
        
        This is the original nano-vllm Triton implementation - fast and efficient.
        
        Args:
            key: [num_tokens, num_heads, head_dim]
            value: [num_tokens, num_heads, head_dim]
            k_cache: [num_blocks, block_size, num_heads, head_dim]
            v_cache: [num_blocks, block_size, num_heads, head_dim]
            slot_mapping: [num_tokens] - indices indicating where to store each token
        """
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        
        # Stride assertions from original implementation
        assert key.stride(-1) == 1 and value.stride(-1) == 1, \
            "Key and value must be contiguous in last dimension"
        assert key.stride(1) == head_dim and value.stride(1) == head_dim, \
            "Key and value must have correct head stride"
        assert k_cache.stride(1) == D and v_cache.stride(1) == D, \
            "Cache must have correct block stride"
        assert slot_mapping.numel() == N, \
            "Slot mapping must have same length as number of tokens"
        
        # Launch Triton kernel (original nano-vllm implementation)
        store_kvcache_kernel[(N,)](
            key, key.stride(0),
            value, value.stride(0),
            k_cache, v_cache,
            slot_mapping,
            D
        )
    
    def retrieve(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve keys/values from cache (no-op for unquantized cache).
        
        Since this cache is already in FP16/BF16, we just return the
        cache tensors directly. Flash-attention can use them as-is.
        
        Args:
            k_cache: [num_blocks, block_size, num_heads, head_dim]
            v_cache: [num_blocks, block_size, num_heads, head_dim]
            
        Returns:
            (k_cache, v_cache) unchanged
        """
        return k_cache, v_cache
    
    def needs_dequantization(self) -> bool:
        """No dequantization needed for FP16/BF16 cache."""
        return False
