"""Default flash attention backend using flash-attn library."""
import torch
from typing import Optional
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.flash_attn_backend import BaseFlashAttentionBackend, FlashAttentionRegistry


@FlashAttentionRegistry.register("default")
class DefaultFlashAttention(BaseFlashAttentionBackend):
    """
    Default flash attention implementation using the flash-attn library.
    
    This is the original nano-vllm implementation:
    - Supports unquantized FP16/BF16 KV caches
    - Uses flash_attn_varlen_func for prefill
    - Uses flash_attn_with_kvcache for decode
    
    For quantized caches (INT8/INT4), you must implement a custom backend
    that either:
    1. Dequantizes cache before passing to flash-attn (simpler)
    2. Implements custom attention kernels that read quantized cache directly (faster)
    """
    
    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        max_seqlen_q: Optional[int],
        cu_seqlens_q: Optional[torch.Tensor],
        max_seqlen_k: Optional[int],
        cu_seqlens_k: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prefill attention using flash_attn_varlen_func.
        
        Handles variable-length sequences efficiently using cumulative
        sequence lengths, avoiding padding overhead.
        
        Args:
            q: [total_tokens, num_heads, head_dim]
            k: [total_tokens, num_kv_heads, head_dim]
            v: [total_tokens, num_kv_heads, head_dim]
            scale: Softmax scale
            max_seqlen_q: Maximum query sequence length
            cu_seqlens_q: Cumulative query sequence lengths
            max_seqlen_k: Maximum key sequence length
            cu_seqlens_k: Cumulative key sequence lengths
            block_table: Block table for prefix caching
            
        Returns:
            Output [total_tokens, num_heads, head_dim]
        """
        return flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=scale,
            causal=True,
            block_table=block_table,
        )
    
    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        scale: float,
        cache_seqlens: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        *additional_cache_tensors,
    ) -> torch.Tensor:
        """
        Decode attention using flash_attn_with_kvcache.
        
        Optimized for single-token generation, reading from paged KV cache.
        
        Args:
            q: [batch_size, num_heads, head_dim]
            k_cache: [num_blocks, block_size, num_kv_heads, head_dim] in FP16/BF16
            v_cache: [num_blocks, block_size, num_kv_heads, head_dim] in FP16/BF16
            scale: Softmax scale
            cache_seqlens: Length of each sequence in batch
            block_table: Block table for paged attention
            *additional_cache_tensors: Not used (for quantized backends)
            
        Returns:
            Output [batch_size, 1, num_heads, head_dim]
        """
        # Ensure q has batch dimension
        if q.ndim == 2:
            q = q.unsqueeze(1)  # [batch_size, 1, num_heads, head_dim]
        elif q.ndim == 3 and q.shape[1] != 1:
            # If q is [batch_size, num_heads, head_dim], add seq_len dim
            q = q.unsqueeze(1)
        
        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            softmax_scale=scale,
            causal=True,
        )
