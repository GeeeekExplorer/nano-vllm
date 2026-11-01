import os
from dataclasses import dataclass
from transformers import AutoConfig

MAX_SPECULATIVE_BATCH_SIZE = 512   # the largest decode/verification CUDA-graph batch size
BLOCK_DRAFTS = ("dflash", "dspark")   # DeepSpec's block drafts: DFlash, and DSpark with its Markov head


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    enable_prefix_cache: bool = True
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    kv_cache_memory_bytes: int | None = None
    draft_model: str | None = None            # enables speculative decoding
    draft_type: str = "model"                 # "model": a draft LM with its own KV and decode steps; or one of BLOCK_DRAFTS
    num_speculative_tokens: int = 0           # k: drafts proposed per round; >= 1 iff draft_model is set; DFlash: its block size
    draft_kv_cache_memory_bytes: int | None = None    # the model draft's KV budget; DFlash's KV mirrors the target's blocks
    seed: int | None = None                   # seed of the engine's sampling generator
    record_step_timings: bool = False         # record a pair of CUDA events per engine step

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.kv_cache_memory_bytes is not None:
            if type(self.kv_cache_memory_bytes) is not int or self.kv_cache_memory_bytes <= 0:
                raise ValueError("kv_cache_memory_bytes must be a positive integer")
        if self.draft_model is not None:
            self._validate_speculative()
        elif self.num_speculative_tokens != 0:
            raise ValueError("num_speculative_tokens requires draft_model")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

    def _validate_speculative(self) -> None:
        if not os.path.isdir(self.draft_model):
            raise ValueError(f"draft_model is not a directory: {self.draft_model}")
        if self.draft_type not in ("model", *BLOCK_DRAFTS):
            raise ValueError(f"draft_type must be 'model', 'dflash' or 'dspark', not {self.draft_type!r}")
        if self.draft_type in BLOCK_DRAFTS:
            draft_config = AutoConfig.from_pretrained(self.draft_model)
            markov_rank = getattr(draft_config, "markov_rank", 0)
            if (markov_rank > 0) != (self.draft_type == "dspark"):
                raise ValueError(f"draft_type {self.draft_type!r} does not match a checkpoint with markov_rank {markov_rank}: "
                                 "DFlash has no Markov head, DSpark has one")
            if markov_rank > 0 and draft_config.markov_head_type != "vanilla":
                raise ValueError(f"only DSpark's vanilla Markov head is implemented, not {draft_config.markov_head_type!r}")
            block_size = draft_config.block_size
            if self.num_speculative_tokens not in (0, block_size):
                raise ValueError(f"a block draft proposes its block size ({block_size}) tokens per round")
            self.num_speculative_tokens = block_size
        if self.num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1 with a draft model")
        if self.tensor_parallel_size != 1:
            raise ValueError("speculative decoding supports tensor_parallel_size=1 only")
        if self.max_num_seqs > MAX_SPECULATIVE_BATCH_SIZE:
            raise ValueError(f"speculative decoding captures verification graphs for at most {MAX_SPECULATIVE_BATCH_SIZE} sequences")
        if self.enable_prefix_cache:
            raise ValueError("speculative decoding requires enable_prefix_cache=False")
        if self.kv_cache_memory_bytes is None:
            raise ValueError("speculative decoding requires an explicit kv_cache_memory_bytes")
        if self.draft_type in BLOCK_DRAFTS:
            return
        if self.draft_kv_cache_memory_bytes is None:
            raise ValueError("a model draft requires an explicit draft_kv_cache_memory_bytes")
        if type(self.draft_kv_cache_memory_bytes) is not int or self.draft_kv_cache_memory_bytes <= 0:
            raise ValueError("draft_kv_cache_memory_bytes must be a positive integer")
