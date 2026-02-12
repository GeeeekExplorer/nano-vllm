from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.config import Config
from dataclasses import fields


class LLM(LLMEngine):
    def __init__(self, model: str, **kwargs) -> None:
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        super().__init__(config)
