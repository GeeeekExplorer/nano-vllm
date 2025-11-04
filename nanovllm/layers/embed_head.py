import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist   # pytorch的分布式包

from nanovllm.utils.context import get_context

# 所谓的embedding，就是将一个seq中所有的token id转换成一个embedding_dim的向量
# token id list: [100] --> [100, vocab_size] 的one hot编码，乘以embedding矩阵[vocab, embedding_dim]
# 最终的结果为 [100, embedding_dim]


# 词表并行是对embedding矩阵按行切分，每个GPU负责一部分token的embedding，最后进行all_reduce合并结果
class VocabParallelEmbedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_rank = dist.get_rank()  # 该进程（GPU）在tensor parallel中的编号
        self.tp_size = dist.get_world_size()  # 总进程数（GPU数）
        assert num_embeddings % self.tp_size == 0   # num_embeddings 总词表大小
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size  # 当前GPU负责的词表大小
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank  # 当前GPU负责的词表开始索引
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition  # 结束索引
        # 当前GPU的权重，只存储自己负责的那一部分
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader  # 自定义加载方法，用于加载全局权重时只取shard
    
    # 权重加载函数  loaded weight：模型的全局权重  param:该GPU上的embedding权重矩阵[vocab_size/tp_size, embedding_dim]
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)  # 拿到该shard的词表数，也就说vocab_size/tp_size
        start_idx = self.tp_rank * shard_size  # 计算在全量矩阵中，本shard的起始索引
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size) # 从loaded weight中切分，形状为[vocab_size/tp_size, embedding_dim]
        param_data.copy_(loaded_weight) # 复制到param.data(最终拿到权重)

    def forward(self, x: torch.Tensor):
        # x [batch, seq_len]
        if self.tp_size > 1:  # 若GPU数>1
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)  # 定义mask，形状与X相同 布尔值
            x = mask * (x - self.vocab_start_idx) # 对所有token id做减法，乘以mask（不属于本shard的token id变成0，属于的不变）
        y = F.embedding(x, self.weight)  # 进行embedding    [batch, seq_len, embedding_dim]
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y  # [batch, 1, seq_len] * [batch, seq_len, embedding]
            # 广播 [batch, 1, seq_len]首先会变成 [batch, seq_len, seq_len] 再与y相乘 --> [batch, seq_len, embedding_dim]
            dist.all_reduce(y) # 对所有GPU得到的y进行逐元素加法，得到最终的embedding结果
        return y

# 从embedding-->token id
# LM Head:把模型的最后一层输出映射回词表的分类器
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        # prefill  x : [一个batch的序列token总长度, hidden_size]
        # decode   x : [batch, hidden_size]
        context = get_context()
        # 若是prefill阶段
        if context.is_prefill:
            # cu_seqlens_q [ batch+1] 
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous() # 拿到每个prompt最后一个token的hidden state, 将他们在内存中连续 [batch, hidden_size]
        # 若说decode阶段，不需要任何处理，直接计算logits
        logits = F.linear(x, self.weight) # 计算logits      linear层会自动对后一个矩阵做转置
        # [batch, embedding_dim] * [num_embeddings_per_partition, embedding_dim].T = [batch, num_embeddings_per_partition]
         
        if self.tp_size > 1:
            # 在主进程中创建buffer，用于接收所有GPU的logits
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            # all_logits[i] = [batch, vocab_size/tp_size] 
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None # 拼接所有logits
            # logits = [batch, vocab_size]
        return logits


# 大模型在进行推理的时候，往往要同时输入多条prompt，而每条prompt的长度又不同，如果将其组成一个矩阵的话避免不了需要padding
# 这样就大量浪费的显存（想象一下100个prompt，其中一个有100000个token,其余99个prompt每个仅有100左右个prompt）
# 但是为了组成矩阵就要创建一个[100, 1000000]的矩阵，太浪费了
# 真实的做法是将所有prompt拼成一个长条（行向量），通过index来区分不同prompt的token
# 假设seq A 5token, seq B  8token, seq C token, 首先将其拼成 [5+8+3=16, embedding_dim]
# 此时cu_seqlen_q = [0, 5, 5+8, 5+8+3] = [0, 5, 13, 16]
# 我们就可以清晰的知道A  [0,5], B [5,13]  C [13,16]
# cu_seqlen_q[1:]-1 = [4, 12, 15] 也就说每个prompt最后一个token的索引！

