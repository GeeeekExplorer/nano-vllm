"""
Model Registry for nano-vllm
This module provides a centralized way to register and load different model architectures.
"""

from typing import Dict, Type, Any
from transformers import PretrainedConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.cpm4 import Cpm4ForCausalLM


class ModelRegistry:
    """Registry for managing different model architectures."""
    
    def __init__(self):
        self._model_map: Dict[str, Type] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default supported models."""
        # Register by architecture name (from config.json)
        self.register("MiniCPMForCausalLM", Cpm4ForCausalLM)
        self.register("Qwen3ForCausalLM", Qwen3ForCausalLM)
        
    
    def register(self, model_name: str, model_class: Type):
        """
        Register a model class with a given name.
        
        Args:
            model_name: Name of the model (architecture or type)
            model_class: Model class to register
        """
        self._model_map[model_name] = model_class
    
    def get_model_class(self, model_identifier: str) -> Type:
        """
        Get model class by identifier.
        
        Args:
            model_identifier: Model architecture name or type
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model is not supported
        """
        if model_identifier in self._model_map:
            return self._model_map[model_identifier]
        
        raise ValueError(
            f"Unsupported model: {model_identifier}. "
            f"Supported models: {list(self._model_map.keys())}"
        )
    
    def create_model(self, hf_config: PretrainedConfig):
        """
        Create model instance based on HuggingFace config.
        
        Args:
            hf_config: HuggingFace configuration object
            
        Returns:
            Model instance
        """
        # Try to get model from architecture first
        if hasattr(hf_config, 'architectures') and hf_config.architectures:
            architecture = hf_config.architectures[0]
            try:
                model_class = self.get_model_class(architecture)
                return model_class(hf_config)
            except ValueError:
                pass
        
        # Try to get model from model_type
        if hasattr(hf_config, 'model_type') and hf_config.model_type:
            try:
                model_class = self.get_model_class(hf_config.model_type)
                return model_class(hf_config)
            except ValueError:
                pass
        
        # Fallback: try to infer from config class name
        config_class_name = hf_config.__class__.__name__
        if "MiniCPM" in config_class_name or "Cpm" in config_class_name:
            return Cpm4ForCausalLM(hf_config)
        elif "Qwen" in config_class_name:
            return Qwen3ForCausalLM(hf_config)
        
        # If all fails, raise error with helpful message
        available_models = list(self._model_map.keys())
        raise ValueError(
            f"Cannot determine model type from config. "
            f"Config architecture: {getattr(hf_config, 'architectures', None)}, "
            f"model_type: {getattr(hf_config, 'model_type', None)}, "
            f"config_class: {config_class_name}. "
            f"Available models: {available_models}"
        )
    
    def list_supported_models(self) -> Dict[str, Type]:
        """Return a copy of all registered models."""
        return self._model_map.copy()


# Global registry instance
model_registry = ModelRegistry()

# Convenience functions
def register_model(model_name: str, model_class: Type):
    """Register a new model globally."""
    model_registry.register(model_name, model_class)

def get_model_class(model_identifier: str) -> Type:
    """Get model class by identifier."""
    return model_registry.get_model_class(model_identifier)

def create_model(hf_config: PretrainedConfig):
    """Create model instance based on HuggingFace config."""
    return model_registry.create_model(hf_config)

def list_supported_models() -> Dict[str, Type]:
    """List all supported models."""
    return model_registry.list_supported_models()
