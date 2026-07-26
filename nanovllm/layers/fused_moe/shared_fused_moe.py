from nanovllm.layers.fused_moe.layer import FusedMoE
import torch
class SharedFusedMoE(FusedMoE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor
    ) -> torch.Tensor:
        return super().forward(hidden_states, router_logits)
