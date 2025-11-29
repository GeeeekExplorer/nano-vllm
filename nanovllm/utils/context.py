from dataclasses import dataclass
import torch

"""
Context 代表模型在某一次前向传播（Forward Pass）时的全局上下文信息，让底层的模型算子（如 Attention 层）能够获取当前这一批次（Batch）数据的元数据
例如在推理时，一个 Batch 里的请求长度通常是不一样的（比如一个请求刚生成了 5 个词，另一个生成了 100 个词），
通常把这些请求的 Token 拼成一个长长的一维数组（Packed Tensor），那这些请求间的边界就需要被保存为元数据
"""
@dataclass
class Context:
    is_prefill: bool = False # 标记当前是否处于 Prefill 阶段，默认为Decode阶段
    cu_seqlens_q: torch.Tensor | None = None # Query 的累积序列长度 (Cumulative Sequence Lengths)，用于告诉算子，在拼起来的一维大数组里，第 i 个请求是从哪里开始，到哪里结束的
    cu_seqlens_k: torch.Tensor | None = None # Key 的累积序列长度
    max_seqlen_q: int = 0 # Batch 中最长的那个 Query 或 Key 的长度，用来分配资源
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None # 记录了当前 Batch 中正在处理的每一个 Token，它的 KV 数据应该写入到显存的哪个精确位置（Slot Index），即使可以用下面两个变量算出来，但是为了性能
    context_lens: torch.Tensor | None = None # 形状为 [batch_size] 的一维张量，记录了每个 Sequence 当前一共有多少个 Token，作用是控制读token时的范围和decode中新token在RoPE中的位置索引
    block_tables: torch.Tensor | None = None # 记录了第 i 个请求（seq）的第 j 个逻辑块，实际上存在显存的哪个物理块里

_CONTEXT = Context() # 初始化一个全局的 Context 实例，作为默认上下文

def get_context(): # 获取当前全局上下文
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None): # 设置全局上下文
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables) # 原来的 _CONTEXT 对象会被覆盖，随后通常会被 Python 的垃圾回收机制自动销毁

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
