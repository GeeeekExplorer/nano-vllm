import os
from dataclasses import dataclass
from transformers import AutoConfig # HuggingFace 的配置加载类，用于读取模型目录下的 config.json

"""
存储整个推理引擎的配置参数
"""
@dataclass
class Config:
    model: str # 模型路径
    max_num_batched_tokens: int = 16384 # 防御性编程，用于约束scheduler的上限处理的token和seq数量，提前限制以适用于预分配内存和适应内部kernel等的高性能要求（避免少seq但很长 or 短seq但很多）
    max_num_seqs: int = 512
    max_model_len: int = 4096 # 单个seq允许的最大长度（prompt + output），默认 4096，如果模型本身支持更长，会被这个值截断；如果模型支持更短，会在 __post_init__ 里被修正
    gpu_memory_utilization: float = 0.9 # 最多可以用多少gpu显存，预先知道有多少显存可用，就能算出最多能存多少个 KV Block，从而推算出最大并发数（Batch Size）；在框架可用显存的 90% 额度内，扣除掉已用部分（模型参数等），剩下的全部分配给 KV Cache
    tensor_parallel_size: int = 1
    enforce_eager: bool = False # 是否强制使用 PyTorch Eager 模式（即不使用 CUDA Graph / torch.compile），默认尝试使用图优化加速；框架开发、调试、迭代可用强制eager
    hf_config: AutoConfig | None = None
    eos: int = -1 # 一个sequence生成结束的标志，初始为 -1，后续会在 LLMEngine 初始化时从 tokenizer 读取并覆盖
    kvcache_block_size: int = 256 # 越小内存碎片就越小，越大连续地址越多访存效率越高
    num_kvcache_blocks: int = -1 # 多少个KVcache block，会在 ModelRunner 初始化时根据显存大小动态计算

    def __post_init__(self): # @dataclass的优势
        assert os.path.isdir(self.model) # 检查路径是否存在
        assert self.kvcache_block_size % 256 == 0 # 直接原因：在 Sequence 类中直接硬编码了 block_size = 256；根本原因：FlashAttn 的经典配置
        assert 1 <= self.tensor_parallel_size <= 8 # 1-8卡TP
        self.hf_config = AutoConfig.from_pretrained(self.model) # 加载 HuggingFace 的 config.json，存入 self.hf_config
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings) # 在此作上述max_model_len修正
        assert self.max_num_batched_tokens >= self.max_model_len # batch 总 token 数必须大于等于单个序列的最大长度，否则连一个满长度的序列都跑不了
