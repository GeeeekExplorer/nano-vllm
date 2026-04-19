import os
from dataclasses import dataclass
from transformers import AutoConfig

# python
# @dataclass 是 Python 提供的“ 数据类 ”语法糖，适合“主要用来存配置/状态”的类。它会自动帮你生成常见样板代码！最核心是：
# - __init__ （按字段自动接收参数）
# - __repr__ （打印友好）
# - __eq__ （按字段比较是否相等）

# __post_init__ 是 dataclass 的“后置钩子”。
# 执行顺序是：
# 1. dataclass 自动生成的 __init__ 先把字段赋值完；
# 2. 然后自动调用你的 __post_init__ 。
# 它适合做三类事：
# - 参数校验（非法就 raise/assert）
# - 派生字段计算（根据输入算出新字段）
# - 配置归一化（把用户输入转成标准格式）

@dataclass
class Config:
    model: str # 模型权重目录路径（本项目假设是本地 huggingface safetensors 目录）。
    max_num_batched_tokens: int = 16384 # prefill 阶段 token 预算上限 ：scheduler 用它决定一次 prefill batch 最多塞多少 token
    max_num_seqs: int = 512 # 一个 step 最多并行多少条 seq（prefill 和 decode 都会受它影响）
    max_model_len: int = 4096 # 引擎允许的最大上下文长度上限（期望不超过模型本身的最大位置编码上限），主要用于 warmup / 估算 / 一些 buffer shape，没有用于调度时的长度检查
    gpu_memory_utilization: float = 0.9 # KV cache 预留策略：最多用总显存的 90% 来做推理 todo：为什么要这么做
    tensor_parallel_size: int = 1 # TP 卡数（= dist world_size）。>1 时会启动多进程、初始化 NCCL、启用 TP layer
    enforce_eager: bool = False # 强制 eager 执行开关：主要用于禁用 decode 的 cudagraph，todo：prefill不涉及吗
    hf_config: AutoConfig | None = None # HF 模型配置对象占位。实际会在 __post_init__ 里从 model 路径加载并写入
    eos: int = -1 # HF 模型配置对象占位。实际会在 __post_init__ 里从 model 路径加载并写入
    kvcache_block_size: int = 256 # 一个block的slot数
    num_kvcache_blocks: int = -1 # KV cache 物理 blocks 数量占位。会在 ModelRunner.allocate_kv_cache() 里根据显存算出来并写回 config

    def __post_init__(self):
        assert os.path.isdir(self.model) # 确保 model 路径存在且是目录
        assert self.kvcache_block_size % 256 == 0 # 要求 block_size 是 256 的倍数（和 flash-attn/paged KV 的对齐/性能假设有关）
        assert 1 <= self.tensor_parallel_size <= 8 # 限制 TP size 范围（作者做的简化约束
        self.hf_config = AutoConfig.from_pretrained(self.model) # 从本地目录读取 HF config（包含 dtype、层数、max_position_embeddings 等）
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings) # 把引擎配置的 max_model_len clamp 到模型本身允许的最大位置数上限，避免用户传一个比模型还大的值
        assert self.max_num_batched_tokens >= self.max_model_len # prefill token 预算至少要能容纳一条“最长上下文”的 seq，否则 warmup/推理会很尴尬
