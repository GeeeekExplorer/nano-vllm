"""Base KV cache interface for nano-vllm."""
from abc import ABC, abstractmethod
import torch
from typing import Tuple


class BaseKVCache(ABC):
    """
    Base class for KV cache implementations.
    
    This abstraction allows different quantization strategies for KV cache,
    making it easy to experiment with various compression techniques like
    INT8, INT4, FP8, or custom quantization schemes.
    """
    
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        """
        Initialize KV cache.
        
        Args:
            num_blocks: Number of blocks to allocate
            block_size: Number of tokens per block
            num_heads: Number of attention heads (after tensor parallel split)
            head_dim: Dimension of each attention head
            dtype: Data type for cache storage
            device: Device to allocate cache on
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
    
    @abstractmethod
    def allocate(self) -> Tuple[torch.Tensor, ...]:
        """
        Allocate cache tensors.
        
        For unquantized caches (FP16/BF16):
            Returns (k_cache, v_cache)
        
        For quantized caches (INT8/INT4/FP8):
            Returns (k_cache, v_cache, k_scales, v_scales, ...)
            Additional tensors for quantization parameters
        
        Returns:
            Tuple of cache tensors (at minimum k_cache and v_cache)
        """
        pass
    
    @abstractmethod
    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *additional_tensors,
    ):
        """
        Store key-value pairs into cache (may quantize).
        
        For unquantized caches:
            store(key, value, k_cache, v_cache, slot_mapping)
        
        For quantized caches:
            store(key, value, k_cache, v_cache, slot_mapping, k_scales, v_scales, ...)
        
        Args:
            key: Key tensor [num_tokens, num_heads, head_dim] in FP16/BF16
            value: Value tensor [num_tokens, num_heads, head_dim] in FP16/BF16
            k_cache: Key cache tensor (may be quantized)
            v_cache: Value cache tensor (may be quantized)
            slot_mapping: [num_tokens] indices indicating where to store each token
            *additional_tensors: Extra tensors for quantization (scales, zero-points, etc.)
        """
        pass
    
    @abstractmethod
    def retrieve(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *additional_tensors,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve and dequantize keys/values from cache for attention.
        
        For unquantized caches (FP16/BF16):
            Returns (k_cache, v_cache) directly (no-op)
        
        For quantized caches (INT8/INT4):
            Dequantizes cache to FP16/BF16 for flash-attention
            NOTE: Implementers must either:
            1. Return dequantized tensors and use standard flash-attention
            2. Implement custom attention kernel that reads quantized cache directly
        
        This method exists to support quantized KV caches. Implementers of
        INT8/INT4 caches should either:
        - Implement dequantization here and use with standard attention
        - Return quantized tensors and implement custom flash-attention variant
        
        Args:
            k_cache: Key cache tensor (may be quantized)
            v_cache: Value cache tensor (may be quantized)
            *additional_tensors: Quantization parameters (scales, zero-points, etc.)
        
        Returns:
            (keys, values) in FP16/BF16 format for attention computation
        """
        pass
    
    def needs_dequantization(self) -> bool:
        """
        Whether this cache requires explicit dequantization before attention.
        
        Returns:
            True if retrieve() performs actual dequantization work
            False if cache is already in attention-compatible format (FP16/BF16)
        """
        # Default: assume no quantization (override for INT8/INT4/etc.)
        return False
    
    def get_cache_block_size_bytes(self) -> int:
        """
        Get the size in bytes of a single cache block.
        
        Returns:
            Size in bytes for one block (k + v)
        """
        # Default implementation - can be overridden for quantized caches
        block_bytes = (
            2 *  # k and v
            self.block_size *
            self.num_heads *
            self.head_dim *
            self.dtype.itemsize
        )
        return block_bytes
    
    @property
    def name(self) -> str:
        """Name of the cache implementation."""
        return self.__class__.__name__


class KVCacheRegistry:
    """
    Registry for KV cache implementations.
    
    This allows easy registration and retrieval of different KV cache strategies,
    making it simple to experiment with various quantization techniques.
    """
    
    _registry: dict[str, type[BaseKVCache]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Register a KV cache implementation.
        
        Args:
            name: Name to register the cache under
            
        Returns:
            Decorator function
            
        Example:
            @KVCacheRegistry.register("fp16")
            class FP16KVCache(BaseKVCache):
                ...
        """
        def decorator(cache_cls: type[BaseKVCache]) -> type[BaseKVCache]:
            if name in cls._registry:
                raise ValueError(f"KV cache '{name}' is already registered")
            cls._registry[name] = cache_cls
            return cache_cls
        
        return decorator
    
    @classmethod
    def get_cache_class(cls, name: str) -> type[BaseKVCache]:
        """
        Get a KV cache class by name.
        
        Args:
            name: Name of the cache
            
        Returns:
            KV cache class
            
        Raises:
            ValueError: If cache is not registered
        """
        if name not in cls._registry:
            raise ValueError(
                f"KV cache '{name}' not found in registry. "
                f"Available caches: {list(cls._registry.keys())}"
            )
        return cls._registry[name]
    
    @classmethod
    def create_cache(
        cls,
        name: str,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        **kwargs
    ) -> BaseKVCache:
        """
        Create a KV cache instance.
        
        Args:
            name: Name of the cache implementation
            num_blocks: Number of blocks to allocate
            block_size: Number of tokens per block
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            **kwargs: Additional arguments for the cache
            
        Returns:
            KV cache instance
        """
        cache_cls = cls.get_cache_class(name)
        return cache_cls(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            **kwargs
        )
    
    @classmethod
    def list_caches(cls) -> list[str]:
        """List all registered cache implementations."""
        return list(cls._registry.keys())
