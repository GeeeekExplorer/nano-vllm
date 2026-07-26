import torch
import triton
import triton.language as tl
from nanovllm.layers.fused_moe.router.base_router import BaseRouter

@triton.jit
def topk_softmax_kernel(
    topk_weights_ptr,
    topk_ids_ptr,
    token_expert_indices_ptr,
    gating_output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    row_offsets = pid * N + cols
    logits = tl.load(
        gating_output_ptr + row_offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    # 稳定softmax
    logits = logits - tl.max(logits, axis=0)
    probs = tl.exp(logits)
    probs = probs / tl.sum(probs, axis=0)

    output_base = pid * TOPK
    for k in tl.static_range(TOPK):
        top_idx = tl.argmax(probs, axis=0)
        top_val = tl.max(probs, axis=0)

        tl.store(topk_weights_ptr + output_base + k, top_val)
        tl.store(topk_ids_ptr + output_base + k, top_idx.to(tl.int32))
        tl.store(token_expert_indices_ptr + output_base + k, k)

        probs = tl.where(cols == top_idx, float("-inf"), probs)

def invoke_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gating_output.ndim != 2:
        raise ValueError("gating_output must have shape [num_tokens, num_experts].")
    if topk_weights.ndim != 2 or topk_indices.ndim != 2:
        raise ValueError("top-k outputs must be rank-2 tensors.")
    if token_expert_indices.shape != topk_indices.shape:
        raise ValueError("token_expert_indices must match topk_indices shape.")

    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.size(1)

    if topk > num_experts:
        raise ValueError(
            f"topk ({topk}) must be <= number of experts ({num_experts})."
        )

    if topk_weights.shape != (num_tokens, topk):
        raise ValueError("topk_weights has an incompatible shape.")
    if topk_indices.shape != (num_tokens, topk):
        raise ValueError("topk_indices has an incompatible shape.")

    # 使得gating_output在内存中是连续的，以便triton内核高效访问
    gating_output = gating_output.contiguous()

    if not gating_output.is_cuda:
        routing_weights = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
        topk_values, topk_ids = torch.topk(routing_weights, k=topk, dim=-1)
        topk_weights.copy_(topk_values)
        topk_indices.copy_(topk_ids.to(torch.int32))
        token_expert_indices.copy_(
            torch.arange(topk, device=gating_output.device, dtype=torch.int32)
            .unsqueeze(0)
            .expand_as(token_expert_indices)
        )
        return topk_weights, topk_indices
    block_size = triton.next_power_of_2(num_experts)
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    elif block_size <= 256:
        num_warps = 2

    topk_softmax_kernel[(num_tokens,)](
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        M=num_tokens,
        N=num_experts,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return topk_weights, topk_indices


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    scoring_func: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    if scoring_func == "softmax":
        topk_weights, topk_ids = invoke_topk_softmax(
            topk_weights, topk_ids, token_expert_indices, gating_output
        )
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids, token_expert_indices
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")


class FusedTopKRouter(BaseRouter):
    """Default router using standard fused top-k routing."""
    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        scoring_func: str = "softmax",
        renormalize: bool = True,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func


    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using standard fused top-k."""
        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            scoring_func=self.scoring_func,
        )

        return topk_weights, topk_ids
