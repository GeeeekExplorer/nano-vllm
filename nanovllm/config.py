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
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Two-GPU PD separation (M1)
    enable_two_gpu_pd: bool = False
    prefill_device_id: int = 0
    decode_device_id: int = 1
    # GPU1 Attention/FFN pipeline (M2)
    enable_decode_pipeline: bool = False
    decode_attention_sm: int = 16
    decode_ffn_sm: int = 16
    decode_pipeline_profiling: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        # Two-GPU mode validation
        if self.enable_two_gpu_pd:
            assert self.tensor_parallel_size == 1, "Two-GPU PD separation requires tensor_parallel_size=1"
            assert self.prefill_device_id != self.decode_device_id, "Prefill and decode must use different GPUs"
        # M2 pipeline can only be enabled with two-GPU mode
        if self.enable_decode_pipeline:
            assert self.enable_two_gpu_pd, "Decode pipeline (M2) requires two-GPU PD separation (M1) to be enabled"
