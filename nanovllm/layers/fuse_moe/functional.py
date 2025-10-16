import torch

from .grouped_gemm import grouped_gemm_forward


def fused_moe_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    m_sizes: torch.Tensor
    ) -> torch.Tensor:
    return grouped_gemm_forward(input, weight, m_sizes)
