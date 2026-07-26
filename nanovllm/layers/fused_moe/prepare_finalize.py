import nanovllm.layers.fused_moe.modular_kernel as mk
import torch

class MoEPrepareAndFinalizeNoEP(mk.FusedMoEPrepareAndFinalize):
    def __init__(self) -> None:
        super().__init__()

    def prepare(
        self,
        a1: torch.Tensor,          
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ):
        # vllm的实现中, prepare阶段会对a1进行量化, 以得到a1q和a1q_scale. 
        # 但是在这个实现中, 我们不进行量化, 所以这个函数什么都不做
        return

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        """权重聚合在算子中已经完成，这个函数只需要把结果从fused_expert_output复制到output即可"""
        if output is None:
            return fused_expert_output
        
        assert output.shape == fused_expert_output.shape, (
            f"output shape {output.shape} does not match fused_expert_output shape {fused_expert_output.shape}"
        )
        output.copy_(fused_expert_output, non_blocking=True)
        return output
