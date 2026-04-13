import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
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

    # --- PD 分离式推理配置 ---
    # 对应 vllm-ascend PR #950 中的 KV transfer 配置
    pd_disagg_enabled: bool = False      # 是否启用 PD 分离
    pd_kv_role: str = "kv_producer"      # "kv_producer"(Prefill 节点) 或 "kv_consumer"(Decode 节点)
    pd_use_layerwise: bool = False       # 是否使用逐层传输模式
    pd_backend: str = "memcache"         # 存储后端: "memcache" 或 "yuanrong"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        if self.pd_disagg_enabled:
            assert self.pd_kv_role in ("kv_producer", "kv_consumer")
            assert self.pd_backend in ("memcache", "yuanrong")
