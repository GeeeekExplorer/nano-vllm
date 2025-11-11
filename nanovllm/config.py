import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    is_multimodal: bool = False  # Enable multimodal support
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)

        # Multimodal models (e.g. Qwen3-VL) store the text settings in
        # hf_config.text_config.
        text_config = getattr(self.hf_config, "text_config", self.hf_config)

        max_position_embeddings = getattr(
            text_config,
            "max_position_embeddings",
            None,
        )
        if max_position_embeddings is not None:
            self.max_model_len = min(
                self.max_model_len,
                max_position_embeddings,
            )

        # eos may be defined within the text config
        eos_token_id = getattr(text_config, "eos_token_id", None)
        if eos_token_id is not None:
            self.eos = eos_token_id

        assert self.max_num_batched_tokens >= self.max_model_len
