import torch
import torch.nn as nn
from nanovllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.fused_moe.router.fused_topk_router import (FusedTopKRouter)
from nanovllm.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

class _ExpertMLP(nn.Module):
    # 复用 linear.py 中已有的并行线性层和 weight_loader 逻辑。
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )


class FusedMoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int | None = None,
        activation: str = "silu",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.global_num_experts = num_experts
        self.local_num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = tp_size
        self.activation = activation
        for expert_id in range(num_experts):
            self.add_module(str(expert_id), _ExpertMLP(hidden_size, intermediate_size))
        self.quant_method = UnquantizedFusedMoEMethod(
            num_experts=num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            num_local_experts=num_experts,
            activation=activation,
        )
        self.router = FusedTopKRouter(
            top_k=top_k,
            global_num_experts=self.global_num_experts,
            scoring_func="softmax",
        )

    def _get_expert(self, expert_id: int) -> _ExpertMLP:
        return getattr(self, str(expert_id))

    @property
    def w13_weight(self) -> torch.Tensor:
        """
            w13是所有专家门控FFN的gate, up权重矩阵的合并版本, shape(num_experts, hidden_size, 2 * intermediate_size)
        """
        experts = []
        for expert_id in range(self.num_experts):
            expert = self._get_expert(expert_id)
            experts.append(expert.gate_up_proj.weight.transpose(0, 1))
        return torch.stack(experts, dim=0).contiguous()

    @property
    def w2_weight(self) -> torch.Tensor:
        """
            w2是所有专家门控FFN的down权重矩阵的合并版本, shape(num_experts, intermediate_size, hidden_size)
        """
        experts = []
        for expert_id in range(self.num_experts):
            expert = self._get_expert(expert_id)
            experts.append(expert.down_proj.weight.transpose(0, 1))
        return torch.stack(experts, dim=0).contiguous()
        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor
    ) -> torch.Tensor:
        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        return final_hidden_states
