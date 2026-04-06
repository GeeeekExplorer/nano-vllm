"""Base flash attention backend interface for nano-vllm."""
from abc import ABC, abstractmethod
import torch
from typing import Optional, Tuple


class BaseFlashAttentionBackend(ABC):
    """
    Base class for flash attention implementations.
    
    This abstraction allows different flash attention backends, making it easy to:
    - Support quantized KV caches (INT8/INT4) with custom kernels
    - Experiment with alternative attention implementations
    - Optimize for specific hardware or precision requirements
    """
    
    @abstractmethod
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
        Perform prefill attention (processing prompt tokens).
        
        Args:
            q: Query tensor [total_tokens, num_heads, head_dim]
            k: Key tensor [total_tokens, num_kv_heads, head_dim]
            v: Value tensor [total_tokens, num_kv_heads, head_dim]
            scale: Softmax scale (typically 1/sqrt(head_dim))
            max_seqlen_q: Maximum sequence length in the batch for queries
            cu_seqlens_q: Cumulative sequence lengths for queries
            max_seqlen_k: Maximum sequence length in the batch for keys
            cu_seqlens_k: Cumulative sequence lengths for keys
            block_table: Optional block table for prefix caching
            
        Returns:
            Output tensor [total_tokens, num_heads, head_dim]
        """
        pass
    
    @abstractmethod
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
        Perform decode attention (generating one token at a time).
        
        For unquantized caches:
            decode(q, k_cache, v_cache, scale, cache_seqlens, block_table)
        
        For quantized caches (INT8/INT4):
            decode(q, k_cache, v_cache, scale, cache_seqlens, block_table, 
                   k_scales, v_scales, ...)
            NOTE: Implementers must handle quantized cache reading/dequantization
        
        Args:
            q: Query tensor [batch_size, num_heads, head_dim]
            k_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim]
                     (may be quantized for INT8/INT4 backends)
            v_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim]
                     (may be quantized for INT8/INT4 backends)
            scale: Softmax scale
            cache_seqlens: Sequence length of each sequence in batch
            block_table: Block table mapping logical to physical blocks
            *additional_cache_tensors: For quantized caches (scales, zero-points, etc.)
            
        Returns:
            Output tensor [batch_size, 1, num_heads, head_dim]
        """
        pass
    
    @property
    def name(self) -> str:
        """Name of the backend implementation."""
        return self.__class__.__name__


class FlashAttentionRegistry:
    """
    Registry for flash attention backend implementations.
    
    This allows easy registration and retrieval of different attention backends,
    making it simple to plug in custom implementations for quantized caches.
    """
    
    _registry: dict[str, type[BaseFlashAttentionBackend]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Register a flash attention backend.
        
        Args:
            name: Name to register the backend under
            
        Returns:
            Decorator function
            
        Example:
            @FlashAttentionRegistry.register("default")
            class DefaultFlashAttention(BaseFlashAttentionBackend):
                ...
        """
        def decorator(backend_cls: type[BaseFlashAttentionBackend]):
            if name in cls._registry:
                raise ValueError(
                    f"Flash attention backend '{name}' is already registered. "
                    f"Existing: {cls._registry[name].__name__}, "
                    f"New: {backend_cls.__name__}"
                )
            cls._registry[name] = backend_cls
            return backend_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> type[BaseFlashAttentionBackend]:
        """
        Get a registered flash attention backend by name.
        
        Args:
            name: Name of the backend to retrieve
            
        Returns:
            The backend class
            
        Raises:
            KeyError: If backend is not registered
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"Flash attention backend '{name}' not found. "
                f"Available backends: {available}"
            )
        return cls._registry[name]
    
    @classmethod
    def list_backends(cls) -> list[str]:
        """List all registered backend names."""
        return list(cls._registry.keys())
