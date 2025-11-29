import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator): # 用于确保整除并返回整数商，用于将总维度按 tp_size 等整除为每个分片的大小
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module): # 所有后续线性变体的基类，负责基础参数初始化与并行元信息

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None, # 指明分片维度（用于 narrow/chunk）
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader # 给加载权重的逻辑提供 hook，用于按分片加载
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size)) # 若 bias 为 True，则创建 self.bias 参数，并同样绑定 weight_loader
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # forward给子类实现
        raise NotImplementedError


class ReplicatedLinear(LinearBase): # 权重在所有并行进程中完整复制（未分片），在未提供 tp_dim 时用（TP中不能并行的组件等）

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight) # 加载时直接把完整权重拷贝到参数，不做切分

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) # 标准线性变换（矩阵乘加）


class ColumnParallelLinear(LinearBase): # 列并行分片

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor): # 参数大小已经初始化了，就把加载的权重找到分片index加载进来
        param_data = param.data # 每次写 param.data 都会做一次属性访问，取出来放到局部变量可稍微提升速度
        shard_size = param_data.size(self.tp_dim) # 一个rank在TP的维度的分片大小
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) # 结果是维度大小分片但数值完整的，可直接参与后续操作也可cat


"""
在模型实现中，会把若干个线性投影（例如同时保存多条输出权重 W1, W2, W3）按输出维度在一起拼接成一个大矩阵保存/导出，以减少文件/IO 操作或符合上游 checkpoint 的布局，此类是为了适应此操作
但在并行（tensor-parallel）下，加载每个 Weight 时必须把该 Weight 按 rank 再切分并放到本 rank 的相应位置，output_sizes 就用于定位每个 Weight 在合并矩阵中的偏移与大小
"""
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int): # 把分片 id 暴露出去可以让 loader 按 weight 实际布局来调用，正常是读权重时把第 i 个被合并的权重张量传给 loader，就把 i 作为 loaded_shard_id
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size # sum(self.output_sizes[:loaded_shard_id]) 是此子output_size在合并矩阵中的全局起始行（所有 rank 的合并矩阵里），除以rank数量就是每个rank内部的需要填数据的起始行
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size # shard_size是rank本地需要填充的大小，和填充起始行相辅
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size) # 切片操作，在tpdim维度上作切片，便于后面把加载的权重数据切片copy过来
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank] # loadweight是单个子权重的全部，然后按rank数量分片
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear): # Q/K/V 特化的列并行，用于 transformer 中把 Q、K、V 三个投影（通常在权重里以 [Q|K|V] 拼接的形式存在，Q 有 total_num_heads 个头，K 和 V 各有 total_num_kv_heads 个头，两个头数量可能不一样，模型计算时对齐即可）在列并行场景下正确分配到不同进程/设备上

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size # 输出size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size # 大小
            shard_offset = 0 # 起点
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size # v排在k后面
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size) # 对本地预分配张量空间的分片和对加载来的权重数据的分片处理同上类
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase): # 行并行，即在输入维度上并行，各rank上算出来的是大小完整但数值不完整的结果，所以需要allreduce

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
        shard_size = param_data.size(self.tp_dim) # 已分片的张量
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
