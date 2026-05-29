import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from nanovllm.layers.utils import shard_slice, materialize_full


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


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
        loaded_weight = shard_slice(loaded_weight, self.tp_dim, start_idx, shard_size)
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
        loaded_weight = shard_slice(loaded_weight, self.tp_dim, self.tp_rank * shard_size, shard_size)
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

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
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = divide(total_num_heads, tp_size)
        # GQA + TP: 当 tp_size > total_num_kv_heads 时，每个 kv head 被复制到多个 rank
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        # 复制后等效的全局 kv head 数（每 rank num_kv_heads 个，共 tp_size 份）
        effective_total_kv_heads = self.num_kv_heads * tp_size
        output_size = (total_num_heads + 2 * effective_total_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        if loaded_shard_id == "q" or self.tp_size <= self.total_num_kv_heads:
            # 正常切分：每个 rank 取自己那一份（等价于 chunk(tp_size)[tp_rank]）
            chunk_idx = self.tp_rank
        else:
            # KV 复制：tp_size > total_num_kv_heads，多个 rank 共享同一个 kv head
            rep = self.tp_size // self.total_num_kv_heads
            chunk_idx = self.tp_rank // rep
        loaded_weight = shard_slice(loaded_weight, self.tp_dim, chunk_idx * shard_size, shard_size)
        param_data.copy_(loaded_weight)


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
        if param_data.ndim == 1:
            param_data.copy_(materialize_full(loaded_weight))
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = shard_slice(loaded_weight, self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y

class ExpertParallelLinear(RowParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_experts: int,
        bias: bool = False,
    ):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_experts % self.tp_size == 0
        self.num_local_experts = num_experts // self.tp_size
        self.experts_start_idx = self.tp_rank * self.num_local_experts
        self.experts_end_idx = self.experts_start_idx + self.num_local_experts
        self.weight = nn.Parameter(
            torch.empty(self.num_local_experts, output_size, input_size)
        )
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_local_experts, output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def _local_idx(self, expert_id: int) -> int:
        return expert_id - self.experts_start_idx

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_expert_id: int | None = None,
    ):
        if loaded_expert_id is None:
            param.data.copy_(loaded_weight)
            return
        if not (self.experts_start_idx <= loaded_expert_id < self.experts_end_idx):
            return
        local_idx = self._local_idx(loaded_expert_id)
        param.data[local_idx].copy_(loaded_weight)

    def forward(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        reduce_results: bool = True,
    ) -> torch.Tensor:
        x = x.view(-1, self.input_size)
        expert_ids = expert_ids.view(-1)
        y = x.new_zeros((x.size(0), self.output_size))
        for expert_id in range(self.experts_start_idx, self.experts_end_idx):
            token_ids = torch.where(expert_ids == expert_id)[0]
            if token_ids.numel() == 0:
                continue
            local_idx = self._local_idx(expert_id)
            bias = self.bias[local_idx] if self.bias is not None else None
            y[token_ids] = F.linear(x[token_ids], self.weight[local_idx], bias)
        if self.tp_size > 1 and reduce_results:
            dist.all_reduce(y)
        return y