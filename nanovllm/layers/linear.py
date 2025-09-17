import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

#实现并行线性层，在分布式环境中高效处理大规模模型的线性变换
#可以生成注意力机制中的QKV矩阵
#在MLP中进行前向传播
class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

#继承这个类，增加一步对x进行加密的操作
class QKVParallelLinear(ColumnParallelLinear):
    # QKV 并行线性层，用于分布式注意力机制中的 Query/Key/Value 权重切分和加载

    def __init__(
        self,
        hidden_size: int,              # 输入特征维度o
        head_size: int,                # 每个注意力头的维度
        total_num_heads: int,          # 总的注意力头数（Q头）
        total_num_kv_heads: int | None = None,  # 总的 KV 头数（可选，默认等于 Q头数）
        bias: bool = False,            # 是否使用偏置
    ):
        tp_size = dist.get_world_size()  # 并行卡数
        total_num_kv_heads = total_num_kv_heads or total_num_heads  # KV头数默认等于Q头数
        self.head_size = head_size      # 保存每头维度
        self.num_heads = divide(total_num_heads, tp_size)      # 当前卡负责的 Q头数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)  # 当前卡负责的 KV头数
        # 总输出维度 = Q头数 + 2*KV头数（Q,K,V拼接），每头 head_size
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)  # 调用父类构造函数

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        # 分布式加载权重分片
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]  # 分片类型只能是 q/k/v
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size  # Q分片大小
            shard_offset = 0                              # Q分片起始位置
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size  # K分片大小
            shard_offset = self.num_heads * self.head_size   # K分片起始位置
        else:
            shard_size = self.num_kv_heads * self.head_size  # V分片大小
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size  # V分片起始位置
        # 只加载本卡负责的分片
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 按 tp_size 分块，取本卡的分片
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)  # 拷贝权重到参数


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
