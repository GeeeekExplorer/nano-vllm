import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    # A hard cap on the total number of tokens processed in one "batch step"
    # across all active sequences.
    max_num_batched_tokens: int = 16384
    # The maximum number of concurrent sequences (requests) the engine will keep
    # "active" in scheduler at the same time.
    # FYI, a sequence is one independent token stream with its own KV cache.
    max_num_seqs: int = 512
    # prompt tokens + generated tokens <= max_model_len.
    # Applies per sequence, not per batch.
    max_model_len: int = 4096
    # Fraction of GPU memory allowed to use. The remaining 90% is reserved and
    # split between model weights (fixed) and KV cache (dynamic, grows as generation goes on).
    gpu_memory_utilization: float = 0.9
    # Number of GPUs to shard the model across.
    tensor_parallel_size: int = 1
    # if True, skip CUDA graph capture and always run eagerly. If False,
    # pre-capture CUDA graphs for decode.
    enforce_eager: bool = False
    # The rest of the system needs architecture details such as
    # num_hidden_layers, hidden_size, num_attention_heads, num_key_value_heads,
    # vocab_size, torch_dtype, etc. Rather than passing dozens of individual
    # fields, the whole hugging face config object is stored here.
    hf_config: AutoConfig | None = None
    # The end of sequence token id, init default to -1, which will set to the
    # actual value after tokenizer loads.
    eos: int = -1
    # Number of tokens in each KV cache block.
    kvcache_block_size: int = 256
    # Total number of KV cache blocks to allocate.
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # KV cache block size must be multiple of 256, align to Flash Attention.
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
