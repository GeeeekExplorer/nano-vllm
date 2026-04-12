import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class KVTransferConfig:
    """KV Cache 传输配置（CPU Offload 等场景）。

    Attributes:
        kv_connector:       连接器类名，如 "CPUOffloadConnector"
        kv_role:            连接器角色，"kv_both" 表示同时承担 send/recv
        swap_in_threshold:  swap in 触发阈值
        cpu_swap_space_gb:  CPU 端交换空间大小（GB）
    """
    kv_connector: str = ""
    kv_role: str = "kv_both"
    swap_in_threshold: int = 0
    cpu_swap_space_gb: float = 100.0


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
    kv_transfer_config: KVTransferConfig | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
