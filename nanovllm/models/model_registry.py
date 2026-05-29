import importlib

# model_type(来自 HF config) -> (模块路径, 类名)
# 用懒加载：只在真正需要某个模型时才 import 对应模块，避免无关依赖被牵连。
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "qwen3": ("nanovllm.models.qwen3", "Qwen3ForCausalLM"),
    "qwen3_moe": ("nanovllm.models.qwen3moe", "Qwen3MoeForCausalLM"),
}


def get_model_class(model_type: str):
    """根据 config.model_type 返回对应的 *ForCausalLM 类。"""
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_type: {model_type!r}. "
            f"Supported: {sorted(_MODEL_REGISTRY)}"
        )
    module_name, class_name = _MODEL_REGISTRY[model_type]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
