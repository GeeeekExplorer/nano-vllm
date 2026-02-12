import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    # Path to the model checkpoint. Should be a local directory containing the model weights and config files.
    model: str
    # Maximum number of tokens to generate for each sequence. Will be used in the sampling process.
    max_num_batched_tokens: int = 16384
    # Maximum number of sequences to batch together in a single prefill or decode inference step.
    max_num_seqs: int = 512

    max_model_len: int = 4096
    # The target GPU memory utilization for the model runner processes.
    # Will be used to determine how many blocks to allocate for each sequence in the block manager.
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False

    hf_config: AutoConfig | None = None

    # End of sequence token id. Will be set after loading the tokenizer in the LLMEngine.
    eos: int = -1

    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len
