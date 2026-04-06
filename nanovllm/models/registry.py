"""Model registry for managing different model architectures."""
from typing import Type, Callable
from transformers import PretrainedConfig
from nanovllm.models.base import BaseModel


class ModelRegistry:
    """
    Registry for model architectures.
    
    This allows easy registration and retrieval of different model implementations,
    making it simple to add support for new architectures.
    """
    
    _registry: dict[str, Type[BaseModel]] = {}
    _arch_to_model: dict[str, str] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        architectures: list[str] | None = None
    ) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        """
        Register a model class.
        
        Args:
            name: Name to register the model under
            architectures: List of HuggingFace architecture names that map to this model
            
        Returns:
            Decorator function
            
        Example:
            @ModelRegistry.register("qwen3", architectures=["Qwen3ForCausalLM"])
            class Qwen3ForCausalLM(BaseModel):
                ...
        """
        def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            if name in cls._registry:
                raise ValueError(f"Model '{name}' is already registered")
            cls._registry[name] = model_cls
            
            # Map architecture names to model name
            if architectures:
                for arch in architectures:
                    cls._arch_to_model[arch] = name
            
            return model_cls
        
        return decorator
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseModel]:
        """
        Get a model class by name.
        
        Args:
            name: Name of the model
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model is not registered
        """
        if name not in cls._registry:
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {list(cls._registry.keys())}"
            )
        return cls._registry[name]
    
    @classmethod
    def create_model(cls, config: PretrainedConfig, model_name: str | None = None) -> BaseModel:
        """
        Create a model instance from config.
        
        Args:
            config: HuggingFace model config
            model_name: Optional model name. If not provided, will be inferred from config.
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model cannot be determined or is not registered
        """
        if model_name is None:
            # Try to infer from config architectures
            if not hasattr(config, "architectures") or not config.architectures:
                raise ValueError(
                    "Cannot infer model type from config. "
                    "Please specify model_name explicitly."
                )
            
            arch = config.architectures[0]
            if arch not in cls._arch_to_model:
                raise ValueError(
                    f"Architecture '{arch}' is not registered. "
                    f"Available architectures: {list(cls._arch_to_model.keys())}"
                )
            
            model_name = cls._arch_to_model[arch]
        
        model_cls = cls.get_model_class(model_name)
        return model_cls(config)
    
    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._registry.keys())
    
    @classmethod
    def list_architectures(cls) -> list[str]:
        """List all registered architecture names."""
        return list(cls._arch_to_model.keys())
