import torch
from abc import ABC, abstractmethod

"""
    moe模型为了进行高效的推理优化, 一般划分为：
    1. Prepare
    2. permute & unpermute
    3. Finalize

"""

class FusedMoEPrepareAndFinalize(ABC):
    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ):
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        raise NotImplementedError


class FusedMoEPermuteExpertsUnpermute(ABC):
    def __init__(
        self,
        max_num_tokens: int | None = None,
    ):
        self.max_num_tokens = max_num_tokens

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """
            E: 专家个数
            M: token数
            K: hidden size
            N: intermediate size, 这里就是w2.size(1)
            topk: 每个token路由的专家数
            使用这些符号是因为算子的Gemm中(M, K) @ (K, N) -> (M, N)
        """
        assert w1.dim() == 3 and w2.dim() == 3
        E, K, _ = w1.size()
        assert a1.dim() == 2
        assert topk_ids.dim() == 2
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        M = a1.size(0)
        topk = topk_ids.size(1)
        return E, M, w2.size(1), K, topk

    @abstractmethod
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
    ) -> None:
        raise NotImplementedError


class FusedMoEModularKernel(torch.nn.Module):
    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        fused_experts: FusedMoEPermuteExpertsUnpermute,
    ):
        super().__init__()
        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        global_num_experts: int = -1,
    ) -> torch.Tensor:
        if global_num_experts == -1:
            global_num_experts = w1.size(0)

        self.prepare_finalize.prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
        )

        fused_out = torch.empty_like(hidden_states)
        self.fused_experts.apply(
            output=fused_out,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
        )

        output = hidden_states if inplace else torch.empty_like(hidden_states)
        self.prepare_finalize.finalize(
            output,
            fused_out,
            topk_weights,
            topk_ids,
        )
        return output
