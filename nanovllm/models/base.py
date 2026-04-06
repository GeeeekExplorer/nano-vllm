"""Base model interface for nano-vllm."""
from abc import ABC, abstractmethod
import torch
from torch import nn
from transformers import PretrainedConfig


class BaseModel(nn.Module, ABC):
    """
    Base class for all models in nano-vllm.
    
    All models must implement this interface to work with the model runner.
    This abstraction allows easy swapping of different model architectures
    and enables support for various quantization techniques.
    """
    
    # Mapping for packed modules (used during weight loading)
    # Format: {"original_name": ("packed_name", index_or_key)}
    packed_modules_mapping: dict[str, tuple[str, str | int]] = {}
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size] or [total_tokens]
            positions: Position indices for each token
            
        Returns:
            Hidden states from the model
        """
        pass
    
    @abstractmethod
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logits from hidden states.
        
        Args:
            hidden_states: Hidden states from the model
            
        Returns:
            Logits for vocabulary
        """
        pass
    
    def get_kvcache_modules(self) -> list[nn.Module]:
        """
        Get all modules that contain KV cache.
        
        Returns:
            List of modules with k_cache and v_cache attributes
        """
        kvcache_modules = []
        for module in self.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                kvcache_modules.append(module)
        return kvcache_modules
