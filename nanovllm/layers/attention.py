import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache # 分别用于变长输入的 FlashAttention 和带 KV cache 的Flashattention
from nanovllm.utils.context import get_context

"""
物理上有一个连续的全局 KV 存储空间；逻辑上通过 block id 和 block_size 把它划分为许多“slot”（每个 slot 对应一个 token 在全局 KV 索引上的起始位置，扁平化索引）
"""
@triton.jit # 表示下面定义的函数是一个将被 JIT 编译为 GPU 内核的 Triton kernel
def store_kvcache_kernel( # 用于把当前 step 的 k/v 写入全局 cache
    key_ptr, # 输入 key 的起始指针
    key_stride, # key 张量在第 0 维的步长（用于计算偏移）
    value_ptr,
    value_stride,
    k_cache_ptr, # 目标 key cache 的起始指针
    v_cache_ptr,
    slot_mapping_ptr, # slot 映射数组的指针，用来指明每个输入样本应写入缓存的槽位（或 -1 表示跳过），里面元素是token的位置
    D: tl.constexpr, # 常量表达式（tl.constexpr），表示每个token的隐藏层大小（通常为 num_heads * head_dim）；这很重要，Triton 需要知道块的大小，通常必须是 2 的幂次且是常量，以便优化
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D) # tl.arange(0, D): 生成一个向量 [0, 1, 2, ..., D-1]；结果 key_offsets 是一个长度为 D 的向量，包含了当前 Token 所有特征值在显存中的绝对地址偏移
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D) # slot 是行号，D 是行宽，加个向量是为了同时写
    tl.store(k_cache_ptr + cache_offsets, key) # 把之前读进来的 key 数据，写入到 k_cache_ptr 指向的显存位置，也是一样的指针+向量搭配读写
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor): # 作为 Triton 内核的包装器，接收 PyTorch 张量并调用 Triton kernel 写入缓存
    N, num_heads, head_dim = key.shape # N是全局token数
    D = num_heads * head_dim
    # 保证每个 slot 的 D 个元素在内存上是按预期连续/间隔正确的前提下才能安全用 Triton 的线性偏移 slot * D + arange(0, D) 读写
    assert key.stride(-1) == 1 and value.stride(-1) == 1 # 断言 key/value 最后一维的内存步长为 1（即最后一维是连续的），满足 Triton 批量 load/store 的要求
    assert key.stride(1) == head_dim and value.stride(1) == head_dim # 断言在第 1 维（head 维）上的步长等于 head_dim，用来保证按预期布局访问每个 head 的数据
    assert k_cache.stride(1) == D and v_cache.stride(1) == D # 确保每个slot在内存中占用连续的 D = num_kv_heads * head_dim 个元素
    assert slot_mapping.numel() == N # 断言每个输入样本都有一个 slot 映射项
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D) # [(N,)]代表一个线程块操作一个token，tensor传进kernel里面都变成首指针


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale, # softmax 的缩放因子 scale（为 1/sqrt(head_dim)）
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([]) # 初始化 k_cache 和 v_cache 为空的张量，实际运行中，真实的 cache 会在外面分配好后覆盖这个占位

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor): # 一个Token t 经过前面的层，变成了 q_t, k_t, v_t
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel(): # 如果 k_cache 与 v_cache 都不是空（即已分配并含元素）
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping) # 先存 k_t, v_t
        if context.is_prefill:
            if context.block_tables is not None: # 如果有prefix cache（比如 System Prompt），就直接拿来用；计算时模型不需要知道“哈希”或者“复用”的逻辑，它只看 block_tables 是否填好，填好了给flashattn用就行；我认为这里只是为了保证prefill中没有cache的情况，要不然直接用kcache
                k, v = k_cache, v_cache # 这是一个引用赋值，不会复制数据，意为把整个物理显存池传给 FlashAttention，让它配合 block_table 去查表
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables) # prefill调用的flashattn是输入q把所有token拼一起的，通过边界数组cu_seqlens_q区分；内核能够处理不等长序列而无需把它们都 pad 到最大长度（从而节省内存/计算）
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True) # decode调用的是每个seq只有一个token，通过 cache_seqlens (每个请求的长度) 区分；内部实现把新 k/v 插入 cache 操作并在同一 kernel 中完成 attention（读缓存 + 读新 k/v + softmax + 乘 v），从而最小化 kernel 启动与内存带宽开销
        return o

        """
        blocktable在flashattn中的底层查表实现见 https://github.com/Dao-AILab/flash-attention/blob/672381f72c927a4b4a92f30755dc5829c3d0eaa3/csrc/flash_attn/src/flash_fwd_kernel.h#L763
        """