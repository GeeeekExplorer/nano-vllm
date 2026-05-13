import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers import Qwen3NextConfig
from typing import Callable

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.loader import sharded_weight_loader

# Import optimized GatedDeltaNet kernels
print("[Qwen3Next] 正在导入优化的 GatedDeltaNet kernels...")
try:
    from nanovllm.layers.ops import fused_recurrent_gated_delta_rule, chunk_gated_delta_rule
    HAS_FUSED_RECURRENT = fused_recurrent_gated_delta_rule is not None
    HAS_CHUNK = chunk_gated_delta_rule is not None
    
    if fused_recurrent_gated_delta_rule is not None:
        print("[Qwen3Next] ✓ 成功导入 fused_recurrent_gated_delta_rule (Triton kernel)")
    else:
        print("[Qwen3Next] ✗ fused_recurrent_gated_delta_rule 导入失败，将使用 Python fallback")
    
    if chunk_gated_delta_rule is not None:
        print("[Qwen3Next] ✓ 成功导入 chunk_gated_delta_rule (Triton kernel)")
    else:
        print("[Qwen3Next] ✗ chunk_gated_delta_rule 导入失败，将使用 Python fallback")
except ImportError as e:
    HAS_FUSED_RECURRENT = False
    HAS_CHUNK = False
    fused_recurrent_gated_delta_rule = None
    chunk_gated_delta_rule = None
    print("[Qwen3Next] ✗ 导入 GatedDeltaNet kernels 失败: {}".format(e))
    print("[Qwen3Next] 将使用 Python fallback 实现")


def mamba_v2_sharded_weight_loader(
    shard_spec: list[tuple[int, int, bool]],
    tp_size: int,
    tp_rank: int,
) -> Callable:
    """
    Create a weight loader for mamba v2 / GatedDeltaNet conv1d weights.
    This ensures that the projections are correctly sharded so that they can be 
    split into query, key, value. It also ensures that all the groups corresponding 
    to a head shard is placed together with it.
    
    Args:
        shard_spec: List of (full_dim, extra, duplicate_groups) tuples
            - full_dim: The model dimension (before TP)
            - extra: Expected overall increase of dimensions due to replication
            - duplicate_groups: Whether groups are duplicated across TP ranks
        tp_size: Tensor parallel size
        tp_rank: Current tensor parallel rank
    
    Returns:
        A weight loader function that takes (param, loaded_weight) and copies data
    """
    def loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        # Track boundary of (sharded) param, and loaded_weight, respectively
        boundary, loaded_boundary = 0, 0

        # Iterate over the shard specs
        for full_dim, extra, duplicate_groups in shard_spec:
            # full_dim is the model dim (before TP).
            # extra > 0 means there is expected overall increase of dimensions.
            # This is so because of replication.
            # duplicate_groups is used to map the tp_rank to the actual shard rank.
            # This is useful when there is replication of groups to accompany head shards.

            # Size of the loaded shard
            shard_size = full_dim // tp_size

            # Compute the rank into the loaded shard.
            # If there is replication, different TP shards will take from the same rank.
            # NOTE: currently we only support duplication in the case where num_groups == 1
            rank = 0 if duplicate_groups else tp_rank

            # Leftmost boundary index into loaded weight.
            loaded_skip = rank * shard_size
            loaded_start_idx = loaded_boundary + loaded_skip

            # Take these many dims from the loaded weight.
            take = min(shard_size, full_dim - extra - loaded_skip)

            # Always shard on dim 0
            param.data[
                boundary : (boundary + take), ...  # type: ignore[misc]
            ] = loaded_weight[
                loaded_start_idx : (
                    loaded_start_idx + take
                )  # type: ignore[misc]
            ]  # type: ignore[misc]

            # Move indexing boundaries
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader


class Qwen3NextRMSNorm(nn.Module):
    """RMSNorm implementation for Qwen3Next (uses 1.0 + weight instead of weight)"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, residual: torch.Tensor | None = None):
        # Align with HF Qwen3_5: norm only normalizes x; residual is NOT added into norm input.
        # When residual is None: return (norm(x), x) so caller gets residual = pre-norm x for the add.
        # When residual is not None: return (norm(x), residual) so caller adds the same residual after attn/mlp.
        orig_dtype = x.dtype
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen3Next is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        output = output.type_as(x)

        if residual is not None:
            return output, residual
        return output, x

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen3NextRMSNormGated(nn.Module):
    """RMSNorm with gate mechanism for GatedDeltaNet"""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    """Update causal conv1d state and compute output"""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    if activation == "silu":
        out = F.silu(out[:, :, -seq_len:])
    else:
        out = out[:, :, -seq_len:]
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    """L2 normalization"""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False
):
    """Recurrent gated delta rule implementation"""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    
    # Reshape: (seq_len, num_heads, head_dim) -> (batch=1, num_heads, seq_len, head_dim)
    if query.dim() == 3:
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        g = g.unsqueeze(0) if g.dim() == 2 else g
        beta = beta.unsqueeze(0) if beta.dim() == 2 else beta
    
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    # HF-style: apply scale even with L2 norm (aligns with HF implementation)
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    g = g.clamp(min=-50.0)
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=-50.0)

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)

    # Important for CUDA graph replay:
    # We must update the *original* `initial_state` storage in-place (copy back),
    # because Python-side state cache updates do not run during replay.
    initial_state_orig = initial_state
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state_orig is None
        else initial_state_orig.to(value)  # working copy (typically float32)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    # Persist updated recurrent state back into the original tensor object.
    # This enables state to survive across CUDA graph replay steps.
    if output_final_state and initial_state_orig is not None:
        initial_state_orig.copy_(last_recurrent_state.to(initial_state_orig.dtype))
        last_recurrent_state = initial_state_orig
    elif not output_final_state:
        last_recurrent_state = None
    
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    # Remove batch dimension if it was added
    if core_attn_out.shape[0] == 1:
        core_attn_out = core_attn_out.squeeze(0)
    
    return core_attn_out, last_recurrent_state


def torch_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False
):
    """Chunk-based gated delta rule implementation"""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    
    # Reshape: (seq_len, num_heads, head_dim) -> (batch=1, num_heads, seq_len, head_dim)
    if query.dim() == 3:
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        g = g.unsqueeze(0) if g.dim() == 2 else g
        beta = beta.unsqueeze(0) if beta.dim() == 2 else beta
    
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    # HF-style: apply scale even with L2 norm (aligns with HF implementation)
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    # Numerical stability: avoid -inf in g (e.g. from uninitialized or extreme weights)
    # which would make cumsum -inf and (g[i]-g[j]).exp() -> NaN. Clamp g to finite range.
    g = g.clamp(min=-50.0)
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=-50.0)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    # Remove batch dimension if it was added
    if core_attn_out.shape[0] == 1:
        core_attn_out = core_attn_out.squeeze(0)
    
    return core_attn_out, last_recurrent_state


class Qwen3NextAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # q_proj outputs num_heads * head_dim * 2 (query + gate)
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim * 2,
            bias=qkv_bias,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=qkv_bias,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # Always use q_norm and k_norm for Qwen3Next
        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # vLLM-style: cast to weight dtype
        hidden_states = hidden_states.to(self.q_proj.weight.dtype)
        # q_proj outputs query + gate (concatenated)
        # Shape: (seq_len, num_heads * head_dim * 2)
        q_gate = self.q_proj(hidden_states)
        # Reshape and split into query and gate (reshape for stride safety)
        q_gate = q_gate.reshape(-1, self.num_heads, self.head_dim * 2)
        query_states, gate = q_gate.split([self.head_dim, self.head_dim], dim=-1)
        k = self.k_proj(hidden_states)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        
        # Apply normalization
        # Qwen3NextRMSNorm returns (output, x) when called with one arg
        query_states, _ = self.q_norm(query_states)
        k, _ = self.k_norm(k)
        
        # Apply rotary embedding
        query_states, k = self.rotary_emb(positions, query_states, k)
        
        # Attention
        o = self.attn(query_states, k, v)
        # o shape: (seq_len, num_heads, head_dim)
        
        # Reshape gate and apply sigmoid
        # Flatten gate to match attention output shape
        gate = gate.flatten(1, -1)  # (seq_len, num_heads * head_dim)
        o_flat = o.flatten(1, -1)  # (seq_len, num_heads * head_dim)
        attn_output = o_flat * torch.sigmoid(gate)
        
        output = self.o_proj(attn_output)
        return output


class Qwen3NextMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
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
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        if isinstance(gate_up, tuple):
            gate_up = gate_up[0]
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        if isinstance(x, tuple):
            x = x[0]
        return x


class Qwen3NextSparseMoeBlock(nn.Module):
    """
    Sparse MoE block: gate (router) + experts (ModuleList) + optional shared expert.
    Param paths: mlp.gate.weight, mlp.experts.0.*, ... to match checkpoint.
    """

    def __init__(self, config: Qwen3NextConfig) -> None:
        super().__init__()
        num_experts = getattr(config, "num_experts", 0)
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0 for SparseMoeBlock")
        self.num_experts = num_experts
        top_k = getattr(config, "num_experts_per_tok", 1)
        self.top_k = min(top_k, num_experts)
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        shared_size = getattr(config, "shared_expert_intermediate_size", 0)
        hidden_act = getattr(config, "hidden_act", "silu")
        moe_intermediate_size = getattr(
            config, "moe_intermediate_size", config.intermediate_size
        )

        # Gate linear directly under mlp so path is mlp.gate.weight (match checkpoint)
        self.gate = ReplicatedLinear(
            config.hidden_size, num_experts, bias=False
        )
        # Direct ModuleList so param path is mlp.experts.0, experts.1, ... (match checkpoint)
        self.experts = nn.ModuleList([
            Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=moe_intermediate_size,
                hidden_act=hidden_act,
            )
            for _ in range(num_experts)
        ])
        if shared_size > 0:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_size,
                hidden_act=hidden_act,
            )
            self.shared_expert_gate = ReplicatedLinear(config.hidden_size, 1, bias=False)
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        # vLLM-style: cast to weight dtype to avoid float vs bfloat16 matmul
        hidden_states = hidden_states.to(self.gate.weight.dtype)
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        # Router: gate -> softmax -> top-k (was Qwen3NextTopKRouter)
        router_logits = self.gate(hidden_states)
        if isinstance(router_logits, tuple):
            router_logits = router_logits[0]
        probs = F.softmax(router_logits.float(), dim=-1).to(hidden_states.dtype)
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_k_weights = top_k_weights / (
                top_k_weights.sum(dim=-1, keepdim=True) + 1e-10
            )
        # Expert dispatch
        T, H = hidden_states.shape
        top_k = top_k_indices.size(-1)
        final = torch.zeros_like(hidden_states)
        for k in range(top_k):
            expert_idx = top_k_indices[:, k]
            w = top_k_weights[:, k]
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if not mask.any():
                    continue
                tokens_e = mask.nonzero(as_tuple=True)[0]
                x_e = hidden_states[tokens_e]
                out_e = self.experts[e](x_e)
                final[tokens_e] = (
                    final[tokens_e]
                    + out_e * w[tokens_e].unsqueeze(-1).to(out_e.dtype)
                )
        expert_output = final
        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states)
            gate_out = self.shared_expert_gate(hidden_states)
            if isinstance(gate_out, tuple):
                gate_out = gate_out[0]
            shared_out = shared_out * torch.sigmoid(gate_out)
            expert_output = expert_output + shared_out
        return expert_output.reshape(orig_shape)


class Qwen3NextGatedDeltaNet(nn.Module):
    """Gated Delta Net for linear attention in Qwen3Next"""

    def __init__(
        self,
        config: Qwen3NextConfig,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.hidden_size = config.hidden_size
        self.num_v_heads = getattr(config, "linear_num_value_heads", 32)
        self.num_k_heads = getattr(config, "linear_num_key_heads", 16)
        self.head_k_dim = getattr(config, "linear_key_head_dim", 128)
        self.head_v_dim = getattr(config, "linear_value_head_dim", 128)
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        
        self.conv_kernel_size = getattr(config, "linear_conv_kernel_dim", 4)
        self.layer_idx = layer_idx
        self.activation = getattr(config, "hidden_act", "silu")
        self.layer_norm_epsilon = config.rms_norm_eps
        
        # QKV projection
        self.conv_dim = self.key_dim * 2 + self.value_dim
        
        # Conv1d layer (for causal convolution)
        # Use ColumnParallelLinear instead of nn.Conv1d for tensor parallelism
        # The weight shape will be (conv_dim // tp_size, conv_kernel_size)
        # After unsqueeze(1), it becomes (conv_dim // tp_size, 1, conv_kernel_size)
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
        )
        # Add dimension for conv1d: (conv_dim, kernel_size) -> (conv_dim, 1, kernel_size)
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        
        # Set up custom weight loader for conv1d
        # The conv1d weight needs special handling because it contains Q, K, V projections
        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)
        
        # Remove default weight_loader and set custom one
        if hasattr(self.conv1d.weight, "weight_loader"):
            delattr(self.conv1d.weight, "weight_loader")
        
        # Set custom weight loader
        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [
                query_key_settings,  # Q projection
                query_key_settings,   # K projection
                value_settings,       # V projection
            ],
            tp_size,
            dist.get_rank(),
        )
        
        # Projection layers - Qwen3Next uses combined qkvz projection
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        
        self.in_proj_qkvz = ColumnParallelLinear(
            self.hidden_size,
            projection_size_qkvz,
            bias=False,
        )
        self.in_proj_ba = ColumnParallelLinear(
            self.hidden_size,
            projection_size_ba,
            bias=False,
        )
        
        # Time step projection parameters (sharded by TP along dim 0, vLLM-style)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads // tp_size))
        A = torch.empty(self.num_v_heads // tp_size).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log.weight_loader = sharded_weight_loader(0)
        self.dt_bias.weight_loader = sharded_weight_loader(0)

        # Normalization with gate
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        
        # Output projection
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
        )
        
        # Cache states (for incremental decoding)
        # For single sequence: use instance variables
        # For batch inference: use state_cache dict with sequence indices
        self.conv_state = None
        self.recurrent_state = None
        self.state_cache = {}  # Dict mapping sequence_id -> (conv_state, recurrent_state)
    
    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key`, `value`, `z`, `b`, `a` tensors from `mixed_qkvz` and `mixed_ba`.
        This follows the vllm implementation structure.
        """
        tp_size = dist.get_world_size()
        seq_len = mixed_qkvz.size(0)
        
        # Reshape mixed_qkvz: (seq_len, ...) -> (seq_len, num_k_heads/tp, ...)
        # The structure is: [q_per_k_head, k_per_k_head, v_per_k_head, z_per_k_head] * (num_v_heads / num_k_heads)
        num_v_per_k = self.num_v_heads // self.num_k_heads
        new_tensor_shape_qkvz = (seq_len, self.num_k_heads // tp_size,
                                  self.head_k_dim + self.head_k_dim + 
                                  num_v_per_k * self.head_v_dim + num_v_per_k * self.head_v_dim)
        new_tensor_shape_ba = (seq_len, self.num_k_heads // tp_size,
                               2 * num_v_per_k)
        
        mixed_qkvz = mixed_qkvz.reshape(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.reshape(*new_tensor_shape_ba)
        
        # Split along the last dimension
        split_arg_list_qkvz = [
            self.head_k_dim,  # q
            self.head_k_dim,  # k
            num_v_per_k * self.head_v_dim,  # v (grouped by num_v_per_k)
            num_v_per_k * self.head_v_dim,  # z (grouped by num_v_per_k)
        ]
        split_arg_list_ba = [
            num_v_per_k,  # b
            num_v_per_k,  # a
        ]
        
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=-1)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=-1)
        
        # Reshape value and z: (seq_len, num_k_heads/tp, num_v_per_k * head_v_dim) 
        # -> (seq_len, num_v_heads/tp, head_v_dim)
        value = value.reshape(seq_len, -1, self.head_v_dim)
        z = z.reshape(seq_len, -1, self.head_v_dim)
        b = b.reshape(seq_len, -1)
        a = a.reshape(seq_len, -1)
        
        return query, key, value, z, b, a
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
        sequence_id: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for GatedDeltaNet
        
        Args:
            hidden_states: (seq_len, hidden_size)
            positions: (seq_len,) - not used in linear attention but kept for compatibility
            sequence_id: Optional sequence ID for batch inference state management
        
        Returns:
            output: (seq_len, hidden_size)
        """
        seq_len = hidden_states.size(0)
        # vLLM-style: cast to weight dtype for linear layers
        hidden_states = hidden_states.to(self.in_proj_qkvz.weight.dtype)

        # Get or initialize states for this sequence
        # CRITICAL: Each sequence must have independent states to prevent cross-contamination
        if sequence_id is not None:
            # Batch inference: use state_cache with per-sequence isolation
            if sequence_id not in self.state_cache:
                # Initialize fresh states for new sequence
                self.state_cache[sequence_id] = (None, None)
            conv_state, recurrent_state = self.state_cache[sequence_id]
            # Ensure we get references, not copies
            if conv_state is None:
                conv_state = None  # Will be initialized later
            if recurrent_state is None:
                recurrent_state = None  # Will be initialized later
        else:
            # Single sequence: use instance variables
            conv_state = self.conv_state
            recurrent_state = self.recurrent_state
        
        # Part 1: Input Projection
        # ColumnParallelLinear returns (output, _) tuple in some implementations
        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        if isinstance(projected_states_qkvz, tuple):
            projected_states_qkvz = projected_states_qkvz[0]
        
        projected_states_ba = self.in_proj_ba(hidden_states)
        if isinstance(projected_states_ba, tuple):
            projected_states_ba = projected_states_ba[0]
        
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        
        # Flatten for convolution: (seq_len, num_heads, head_dim) -> (seq_len, num_heads * head_dim)
        query_flat = query.reshape(seq_len, -1)
        key_flat = key.reshape(seq_len, -1)
        value_flat = value.reshape(seq_len, -1)
        
        mixed_qkv = torch.cat([query_flat, key_flat, value_flat], dim=-1)
        # mixed_qkv shape: (seq_len, conv_dim // tp_size)
        # Transpose for conv1d: (seq_len, conv_dim // tp_size) -> (conv_dim // tp_size, seq_len)
        mixed_qkv = mixed_qkv.transpose(0, 1).unsqueeze(0)  # (1, conv_dim // tp_size, seq_len)
        
        # Part 2: Causal Convolution
        if conv_state is None:
            # First pass: initialize conv_state for this specific sequence
            # CRITICAL: Create new tensor for each sequence to ensure isolation
            conv_state = torch.zeros(
                1, self.conv_dim // dist.get_world_size(), self.conv_kernel_size - 1,
                dtype=mixed_qkv.dtype, device=mixed_qkv.device
            )
            # Update state cache immediately to ensure isolation
            if sequence_id is not None:
                self.state_cache[sequence_id] = (conv_state, recurrent_state)
            else:
                self.conv_state = conv_state
        
        # Apply causal convolution
        # Extract conv weights: (conv_dim // tp_size, 1, kernel_size) -> (conv_dim // tp_size, kernel_size)
        conv_weights = self.conv1d.weight.reshape(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv = torch_causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,  # (conv_dim // tp_size, kernel_size)
            self.conv1d.bias,
            self.activation,
        )
        
        # Update conv_state (it's modified in-place by torch_causal_conv1d_update)
        # CRITICAL: Always update state cache to ensure state persistence
        if sequence_id is not None:
            # Update state cache with current states (conv_state is modified in-place)
            self.state_cache[sequence_id] = (conv_state, recurrent_state)
        else:
            self.conv_state = conv_state
        
        # Transpose back: (1, conv_dim, seq_len) -> (seq_len, conv_dim)
        mixed_qkv = mixed_qkv.squeeze(0).transpose(0, 1)
        
        # Split back into query, key, value
        query_conv, key_conv, value_conv = torch.split(
            mixed_qkv,
            [self.key_dim // dist.get_world_size(),
             self.key_dim // dist.get_world_size(),
             self.value_dim // dist.get_world_size()],
            dim=-1
        )
        
        # Reshape back to (seq_len, num_heads, head_dim)
        query_conv = query_conv.reshape(seq_len, self.num_k_heads // dist.get_world_size(), self.head_k_dim)
        key_conv = key_conv.reshape(seq_len, self.num_k_heads // dist.get_world_size(), self.head_k_dim)
        value_conv = value_conv.reshape(seq_len, self.num_v_heads // dist.get_world_size(), self.head_v_dim)
        
        # Repeat query/key if num_v_heads > num_k_heads
        if self.num_v_heads // dist.get_world_size() > self.num_k_heads // dist.get_world_size():
            repeat_factor = (self.num_v_heads // dist.get_world_size()) // (self.num_k_heads // dist.get_world_size())
            query_conv = query_conv.repeat_interleave(repeat_factor, dim=1)
            key_conv = key_conv.repeat_interleave(repeat_factor, dim=1)
        
        # Compute gating parameters
        beta = b.sigmoid()  # (seq_len, num_v_heads)
        # g = -exp(A_log) * softplus(a + dt_bias)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        g = g.to(hidden_states.dtype)
        # Numerical stability for both Triton and torch paths
        g = g.clamp(min=-50.0)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=-50.0)

        # Part 3: Gated Delta Rule (Recurrent Attention)
        # CRITICAL: Determine mode based on sequence length and state availability
        use_recurrent = (seq_len == 1 and recurrent_state is not None)
        
        # Ensure recurrent_state is properly initialized for chunk mode if needed
        if not use_recurrent and recurrent_state is None:
            # Initialize recurrent_state for chunk mode (prefill)
            # Shape: (num_v_heads, head_v_dim, head_k_dim) per sequence
            tp_size = dist.get_world_size()
            num_v_heads_shard = self.num_v_heads // tp_size
            recurrent_state = torch.zeros(
                num_v_heads_shard,
                self.head_v_dim,
                self.head_k_dim,
                dtype=value_conv.dtype,
                device=value_conv.device
            )
        
        # Convert from (seq_len, num_heads, head_dim) to (B=1, T=seq_len, H, K) format
        # Add batch dimension
        q_batch = query_conv.unsqueeze(0)  # (1, seq_len, num_k_heads, head_k_dim)
        k_batch = key_conv.unsqueeze(0)    # (1, seq_len, num_k_heads, head_k_dim)
        v_batch = value_conv.unsqueeze(0)  # (1, seq_len, num_v_heads, head_v_dim)
        g_batch = g.unsqueeze(0)           # (1, seq_len, num_v_heads)
        beta_batch = beta.unsqueeze(0)     # (1, seq_len, num_v_heads)
        
        # Prepare initial_state if needed: (num_v_heads, head_v_dim, head_k_dim) -> (1, num_v_heads, head_v_dim, head_k_dim)
        initial_state_batch = None
        if recurrent_state is not None:
            if recurrent_state.dim() == 3:
                initial_state_batch = recurrent_state.unsqueeze(0)  # Add batch dim
            else:
                initial_state_batch = recurrent_state
        
        if use_recurrent and HAS_FUSED_RECURRENT and fused_recurrent_gated_delta_rule is not None:
            # Use optimized Triton kernel for recurrent mode
            # Note: inplace_final_state=False to avoid requiring ssm_state_indices
            # When inplace_final_state=True, kernel expects ssm_state_indices which we don't provide
            core_attn_out_batch, final_state_batch = fused_recurrent_gated_delta_rule(
                q=q_batch,
                k=k_batch,
                v=v_batch,
                g=g_batch,
                beta=beta_batch,
                initial_state=initial_state_batch,
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            # Remove batch dimension: (1, seq_len, num_v_heads, head_v_dim) -> (seq_len, num_v_heads, head_v_dim)
            core_attn_out = core_attn_out_batch.squeeze(0)
            if final_state_batch is not None:
                recurrent_state = final_state_batch.squeeze(0) if final_state_batch.dim() == 4 else final_state_batch
        elif not use_recurrent and HAS_CHUNK and chunk_gated_delta_rule is not None:
            # Use optimized Triton kernel for chunk mode
            core_attn_out_batch, final_state_batch = chunk_gated_delta_rule(
                q=q_batch,
                k=k_batch,
                v=v_batch,
                g=g_batch,
                beta=beta_batch,
                initial_state=initial_state_batch,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            # Remove batch dimension
            core_attn_out = core_attn_out_batch.squeeze(0)
            if final_state_batch is not None:
                recurrent_state = final_state_batch.squeeze(0) if final_state_batch.dim() == 4 else final_state_batch
        else:
            # Fallback to Python implementation
            if use_recurrent:
                core_attn_out, recurrent_state = torch_recurrent_gated_delta_rule(
                    query_conv,
                    key_conv,
                    value_conv,
                    g,
                    beta,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_attn_out, recurrent_state = torch_chunk_gated_delta_rule(
                    query_conv,
                    key_conv,
                    value_conv,
                    g,
                    beta,
                    chunk_size=64,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
        
        # CRITICAL: Always update state cache to ensure state persistence per sequence
        if sequence_id is not None:
            # Store updated states in cache for this specific sequence
            self.state_cache[sequence_id] = (conv_state, recurrent_state)
        else:
            self.recurrent_state = recurrent_state
        
        # Part 4: Output Projection with Gate
        # CRITICAL: Gate application must match vLLM implementation
        # core_attn_out: (seq_len, num_v_heads, head_v_dim)
        # z: (seq_len, num_v_heads, head_v_dim)
        z_shape_og = z.shape
        core_attn_out_flat = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z_flat = z.reshape(-1, z.shape[-1])
        
        # Apply normalization with gate: norm(core_attn_out) * silu(gate)
        # The gate (z) is applied via SiLU activation in Qwen3NextRMSNormGated
        core_attn_out_flat = self.norm(core_attn_out_flat, z_flat)
        core_attn_out = core_attn_out_flat.reshape(z_shape_og)
        
        # Flatten for output projection
        core_attn_out = core_attn_out.reshape(seq_len, -1)
        
        # Output projection
        # RowParallelLinear may return tuple in some implementations
        output = self.out_proj(core_attn_out)
        if isinstance(output, tuple):
            output = output[0]
        
        return output


class Qwen3NextDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3NextConfig,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_scaling = None  # TODO: MRoPE support

        # layer_types: "full_attention" or "linear_attention" per layer (vllm/transformers)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            self.layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
        else:
            self.layer_type = "full_attention"

        if self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=rope_scaling,
        )
            self.linear_attn = None
        elif self.layer_type == "linear_attention":
            self.self_attn = None
            self.linear_attn = Qwen3NextGatedDeltaNet(
                config=config,
                layer_idx=layer_idx,
            )
        else:
            raise ValueError(f"Invalid layer_type: {self.layer_type}")

        # MLP: standard MLP or Sparse MoE (vllm/transformers logic)
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        num_experts = getattr(config, "num_experts", 0)
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        use_moe = (
            (layer_idx not in mlp_only_layers)
            and (num_experts > 0)
            and ((layer_idx + 1) % decoder_sparse_step == 0)
        )
        if use_moe:
            self.mlp = Qwen3NextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3NextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        
        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        if self.layer_type == "full_attention":
            hidden_states = self.self_attn(positions, hidden_states)
        elif self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, positions)
        else:
            raise ValueError(f"Invalid layer_type: {self.layer_type}")
        
        hidden_states = hidden_states + residual
        # HF: residual = hidden_states, then norm(hidden_states); add residual after mlp
        residual = hidden_states
        hidden_states, residual = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states, residual


class Qwen3NextModel(nn.Module):

    def __init__(
        self,
        config: Qwen3NextConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3NextDecoderLayer(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 支持直接传入 embeddings（用于多模态）
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        # vLLM-style: keep activation dtype consistent with weights
        hidden_states = hidden_states.to(self.embed_tokens.weight.dtype)
        
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3NextForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("q_proj", None),
        "k_proj": ("k_proj", None),
        "v_proj": ("v_proj", None),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3NextConfig
    ) -> None:
        super().__init__()
        self.model = Qwen3NextModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values=None,  # 多模态参数（向后兼容）
        image_grid_thw=None,  # 多模态参数（向后兼容）
        **kwargs  # 其他参数
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        return self.lm_head(hidden_states)
