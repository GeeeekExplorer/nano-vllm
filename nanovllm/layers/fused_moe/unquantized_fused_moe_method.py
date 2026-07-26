import torch
import torch.nn as nn
import nanovllm.layers.fused_moe.modular_kernel as mk
from nanovllm.layers.fused_moe.fused_moe import TritonExperts
from nanovllm.layers.fused_moe.prepare_finalize import MoEPrepareAndFinalizeNoEP


class UnquantizedFusedMoEMethod(nn.Module):
    def __init__(
        self,
        num_experts: int,
        experts_per_token: int,
        hidden_dim: int,
        intermediate_size_per_partition: int,
        num_local_experts: int,
        activation: str = "silu",
    ):
       super().__init__()
       self.kernel = mk.FusedMoEModularKernel(
           prepare_finalize=MoEPrepareAndFinalizeNoEP(),
           fused_experts=TritonExperts(),
       )
       
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
    
    def forward(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        return self.kernel(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
        )
    
