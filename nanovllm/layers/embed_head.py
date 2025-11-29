import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context

"""
主要实现了词表并行（Vocab Parallelism）的 Embedding 层和 LM Head（将张量升维到词表大小的一个线性层，LM Head 输出 Logits 然后传给 Sampler）
简单说就是实现了模型和词表头和尾的转换
重点理解embedding时是用掩码作查表操作的分布式，而LM Head是用拼接来作矩阵乘法的分布式
"""
class VocabParallelEmbedding(nn.Module): # 将输入的 Token ID 转换为向量

    def __init__(
        self,
        num_embeddings: int, # 总词表大小
        embedding_dim: int, # 嵌入维度，即输出的hidden维度
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size() # 总进程数
        assert num_embeddings % self.tp_size == 0 # 总词表大小必须能被进程数整除，保证每个 GPU 分到的词表大小一致
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim)) # 只创建了 num_embeddings_per_partition 大小的权重，分布式分显存就在这
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size # 计算当前 GPU 应该从完整权重中截取哪一段，实际逻辑和init类似
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx) # 生成掩码 (mask)，判断输入的 Token ID 是否属于当前 GPU 负责的范围
            x = mask * (x - self.vocab_start_idx) # 将全局 ID 转换为局部 ID，然后方便查局部的向量表
        y = F.embedding(x, self.weight) # 所有 GPU 都会执行查表操作，但只有负责该 ID 的 GPU 查出的结果是有意义的，其他 GPU 查出的是垃圾数据
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y # 如果不属于当前 GPU，将结果置为全 0
            dist.all_reduce(y) # 对于任意一个位置，只有一个 GPU 的结果是非零的，allreduce后就得到了正确的 Embedding 向量
        return y


class ParallelLMHead(VocabParallelEmbedding): # 将隐藏层状态映射回词表大小的 Logits

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int, # 模型最后一层输出的隐藏状态的维度
        bias: bool = False, # 表明 ParallelLMHead 实现不支持偏置项，在并行层中很常见，为了简化计算和同步逻辑
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim) # 父类（即 VocabParallelEmbedding）会根据当前的并行环境，将 self.weight 初始化为完整词表的一个切片

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill: # 对prefill阶段的优化，因为只需要序列中最后一个 token的隐藏状态来预测下一个 token，如果不做处理，x 包含了整个序列的所有 token 的状态，用它去乘权重会做大量无用功
            last_indices = context.cu_seqlens_q[1:] - 1 # 例如一个 batch 中有两个序列，长度分别为 5 和 8，cu_seqlens_q 是 [0, 5, 13]，context.cu_seqlens_q[1:]会得到 [5, 13]，减 1 后得到 [4, 12]，正是每个序列最后一个 token 在展平的x张量中的索引
            x = x[last_indices].contiguous() # 在进行了索引操作之后，确保张量在内存中是连续存储的，避免后续操作因为内存不连续而报错或性能下降
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None # rank0 来负责存放从所有 GPU 收集来的 logits 切片，先开空间
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None # 用 torch.cat 在最后一个维度（词表维度）上将它们拼接起来
        return logits
