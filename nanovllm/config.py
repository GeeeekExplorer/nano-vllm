import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    enable_continuous_batching: bool = False
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    enable_chunked_prefill: bool = False
    chunked_prefill_size: int = 2048
    enable_cb_prefill_liveness: bool = True
    cb_prefill_reserve_ratio: float = 0.2
    cb_prefill_min_tokens: int = 512
    cb_prefill_min_seqs: int = 1
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert self.chunked_prefill_size > 0
        assert 0.0 <= self.cb_prefill_reserve_ratio <= 1.0
        assert self.cb_prefill_min_tokens >= 0
        assert self.cb_prefill_min_seqs >= 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if not self.enable_chunked_prefill:
            assert self.max_num_batched_tokens >= self.max_model_len
