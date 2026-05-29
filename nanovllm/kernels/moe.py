import torch
from torch import nn


def build_topk_dispatch(
    selected_experts: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a grouped dispatch plan for MoE top-k routing.

    Returns:
        sorted_token_ids: flattened token indices sorted by expert id
        sorted_topk_ids:  flattened top-k slot indices aligned with sorted_token_ids
        expert_offsets:   prefix-sum offsets of size [num_experts + 1]
        active_experts:   expert ids that have at least one routed token
    """
    assert selected_experts.dim() == 2
    num_tokens, top_k = selected_experts.shape
    device = selected_experts.device

    token_ids = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(num_tokens, top_k)
        .reshape(-1)
    )
    topk_ids = (
        torch.arange(top_k, device=device)
        .unsqueeze(0)
        .expand(num_tokens, top_k)
        .reshape(-1)
    )
    expert_ids = selected_experts.reshape(-1)

    order = torch.argsort(expert_ids)
    sorted_expert_ids = expert_ids[order]
    sorted_token_ids = token_ids[order]
    sorted_topk_ids = topk_ids[order]

    counts = torch.bincount(sorted_expert_ids, minlength=num_experts)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=device)
    expert_offsets[1:] = torch.cumsum(counts, dim=0)
    active_experts = torch.nonzero(counts, as_tuple=False).flatten()
    return sorted_token_ids, sorted_topk_ids, expert_offsets, active_experts


def grouped_moe_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    experts: nn.ModuleList,
    num_experts: int,
) -> torch.Tensor:
    """
    Grouped MoE execution path used by Qwen3-MoE.

    This function is the kernel integration point for future fused grouped-GEMM:
    - today: grouped dispatch + per-expert execution in PyTorch
    - future: replace per-expert loop with Triton/CUDA grouped GEMM kernel
    """
    sorted_token_ids, sorted_topk_ids, expert_offsets, active_experts = build_topk_dispatch(
        selected_experts, num_experts
    )
    out = torch.zeros_like(hidden_states)
    for expert_id in active_experts.tolist():
        start = expert_offsets[expert_id].item()
        end = expert_offsets[expert_id + 1].item()
        token_ids = sorted_token_ids[start:end]
        topk_ids = sorted_topk_ids[start:end]
        expert_out = experts[expert_id](hidden_states[token_ids])
        weighted_out = expert_out * routing_weights[token_ids, topk_ids, None]
        out.index_add_(0, token_ids, weighted_out)
    return out
