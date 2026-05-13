"""Qwen3.5 multimodal model (vision + language).

This module implements the Qwen3.5 vision-language model used by
nano-vllm-multimodal-qwen3_5. It does not implement or depend on Qwen3-VL.
"""

from __future__ import annotations

import os

# For diagnose_full_attention.py: when NANOVLLM_DEBUG_FA=1, layer 3 full_attention saves intermediates here
FA_DEBUG_SAVE = {}
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.qwen3_next import (
    Qwen3NextRMSNorm,
    Qwen3NextRMSNormGated,
    torch_causal_conv1d_update,
    torch_recurrent_gated_delta_rule,
    torch_chunk_gated_delta_rule,
    mamba_v2_sharded_weight_loader,
)
from nanovllm.utils.loader import sharded_weight_loader

# Triton FLA ops (same as vLLM fla/ops): prefer for numerical stability
try:
    from nanovllm.layers.ops import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
    from nanovllm.layers.ops import fused_recurrent_gated_delta_rule as fla_fused_recurrent_gated_delta_rule
    _HAS_FLA_TRITON = fla_chunk_gated_delta_rule is not None and fla_fused_recurrent_gated_delta_rule is not None
except Exception:
    fla_chunk_gated_delta_rule = None
    fla_fused_recurrent_gated_delta_rule = None
    _HAS_FLA_TRITON = False


# ---------------------------------------------------------------------------
# Text backbone (supports DeepStack injection)
# ---------------------------------------------------------------------------


class Qwen3_5TextRotaryEmbedding(nn.Module):
    """Rotary Embedding for Qwen3_5 with MRoPE support"""
    
    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.config = config
        
        rope_parameters = getattr(config, "rope_parameters", {})
        if isinstance(rope_parameters, dict):
            self.rope_type = rope_parameters.get("rope_type", "default")
            rope_theta = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 1000000))
            self.mrope_section = rope_parameters.get("mrope_section", [11, 11, 10])
            partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        else:
            # RopeParameters object or similar (e.g. from HF config)
            self.rope_type = getattr(rope_parameters, "rope_type", "default")
            rope_theta = getattr(rope_parameters, "rope_theta", getattr(config, "rope_theta", 1000000))
            self.mrope_section = getattr(rope_parameters, "mrope_section", [11, 11, 10])
            partial_rotary_factor = getattr(rope_parameters, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0
    
    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        args:
            freqs: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            freqs_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t
    
    def forward(self, x, position_ids):
        # Qwen3_5 MRoPE: position_ids can be (3, seq_len) or (3, batch, seq_len) for 3D (T,H,W),
        # or (batch, seq_len) for 1D which we expand to (3, batch, seq_len).
        if position_ids.ndim == 2:
            if position_ids.shape[0] == 3:
                # (3, seq_len) -> (3, 1, seq_len)
                position_ids = position_ids.unsqueeze(1)
            else:
                # (batch, seq_len) -> (3, batch, seq_len)
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        # position_ids now (3, batch, seq_len)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        
        # Force float32 computation
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3_5TextAttentionMerged(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        rms_norm_eps: float,
        qkv_bias: bool,
        head_dim: int | None,
        rope_theta: float,
        rope_scaling: tuple | None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
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
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.qkv_proj.weight.dtype)
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3_5TextAttention(nn.Module):
    """Qwen3_5 Text Attention with gate mechanism (similar to Qwen3Next)"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        rms_norm_eps: float,
        qkv_bias: bool,
        head_dim: int | None,
        rope_theta: float,
        rope_scaling: tuple | None,
        config=None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
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
        
        # Use Qwen3_5TextRotaryEmbedding if config is provided, otherwise fallback to standard RoPE
        if config is not None:
            self.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device=None)
        else:
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
        # Always use q_norm and k_norm for Qwen3_5
        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.q_proj.weight.dtype)
        # q_proj outputs query + gate (concatenated)
        q_gate = self.q_proj(hidden_states)
        q_gate = q_gate.reshape(-1, self.num_heads, self.head_dim * 2)
        query_states, gate = q_gate.split([self.head_dim, self.head_dim], dim=-1)
        k = self.k_proj(hidden_states)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        # Apply normalization (Qwen3NextRMSNorm returns (output, x) when called with one arg)
        query_states, _ = self.q_norm(query_states)
        k, _ = self.k_norm(k)
        
        # Apply rotary embedding
        # For Qwen3_5TextRotaryEmbedding, we need to handle MRoPE differently
        # For now, fallback to standard RoPE if not using MRoPE
        if isinstance(self.rotary_emb, Qwen3_5TextRotaryEmbedding):
            # For MRoPE, positions should be 3D (temporal, height, width)
            # But in nano-vllm, positions is typically 1D, so we use standard RoPE for now
            # TODO: Implement proper MRoPE support with 3D position_ids
            # For now, convert positions to 2D format expected by Qwen3_5TextRotaryEmbedding
            if positions.ndim == 1:
                # Convert 1D positions to 2D (batch_size=1, seq_len)
                positions_2d = positions.unsqueeze(0)
            else:
                positions_2d = positions
            
            cos, sin = self.rotary_emb(hidden_states, positions_2d)
            # Apply rotary embedding manually (partial RoPE: cos/sin only cover rotary_dim dims)
            rotary_dim = cos.shape[-1]
            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)
            # cos/sin shape: (bs, seq_len, rotary_dim); query_states/k shape: (seq_len, num_heads, head_dim)
            if cos.ndim == 3:
                cos = cos.squeeze(0).unsqueeze(1)  # (seq_len, 1, rotary_dim)
                sin = sin.squeeze(0).unsqueeze(1)
            else:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            q_rot = (query_states[..., :rotary_dim] * cos) + (rotate_half(query_states[..., :rotary_dim]) * sin)
            k_rot = (k[..., :rotary_dim] * cos) + (rotate_half(k[..., :rotary_dim]) * sin)
            query_states = torch.cat([q_rot, query_states[..., rotary_dim:]], dim=-1)
            k = torch.cat([k_rot, k[..., rotary_dim:]], dim=-1)
        else:
            query_states, k = self.rotary_emb(positions, query_states, k)
        
        # Attention
        o = self.attn(query_states, k, v)
        
        # Reshape gate and apply sigmoid
        gate = gate.flatten(1, -1)  # (seq_len, num_heads * head_dim)
        o_flat = o.flatten(1, -1)  # (seq_len, num_heads * head_dim)
        attn_output = o_flat * torch.sigmoid(gate)
        
        output = self.o_proj(attn_output)
        return output


class Qwen3_5TextMLPMerged(nn.Module):

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
        x = x.to(self.gate_up_proj.weight.dtype)
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated Delta Net for linear attention in Qwen3_5
    
    Qwen3_5 uses separate projections for qkv, z, b, a instead of combined qkvz and ba.
    """
    
    def __init__(
        self,
        config,
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
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
        )
        # Add dimension for conv1d: (conv_dim, kernel_size) -> (conv_dim, 1, kernel_size)
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        
        # Set up custom weight loader for conv1d
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
        
        # Projection layers - Qwen3_5 uses separate projections
        self.in_proj_qkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.key_dim, self.key_dim, self.value_dim],
            bias=False,
        )
        self.in_proj_z = ColumnParallelLinear(
            self.hidden_size,
            self.value_dim,
            bias=False,
        )
        self.in_proj_b = ColumnParallelLinear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
        )
        self.in_proj_a = ColumnParallelLinear(
            self.hidden_size,
            self.num_v_heads,
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
        # HF-style: per-sequence isolation via state_cache when sequence_id is provided
        self.conv_state = None
        self.recurrent_state = None
        self.state_cache = {}  # sequence_id -> (conv_state, recurrent_state)
        self._decode_step_counter = {}  # sequence_id -> decode steps (for debug)

        # CUDA graph decode state buffers: persistent tensors so in-place updates
        # are captured by the graph and replayed correctly.  Size must be >=
        # max_num_seqs (we pick 512 as a safe upper bound for nano-vllm).
        max_graph_bs = 512
        conv_dim_tp = self.conv_dim // tp_size
        num_v_heads_tp = self.num_v_heads // tp_size
        self.register_buffer(
            "_graph_conv_state",
            torch.zeros(
                max_graph_bs,
                conv_dim_tp,
                self.conv_kernel_size - 1,
                dtype=torch.bfloat16,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_graph_recurrent_state",
            torch.zeros(
                max_graph_bs,
                num_v_heads_tp,
                self.head_v_dim,
                self.head_k_dim,
                dtype=torch.bfloat16,
            ),
            persistent=False,
        )

    def reset_state(self):
        """Reset all cached states. Called after warmup to avoid state pollution."""
        self.conv_state = None
        self.recurrent_state = None
        self.state_cache.clear()
        self._decode_step_counter.clear()
        self._graph_conv_state.zero_()
        self._graph_recurrent_state.zero_()

    def _get_states(self, sequence_id: int | None):
        """Get conv_state and recurrent_state for this sequence."""
        if sequence_id is not None:
            if sequence_id not in self.state_cache:
                self.state_cache[sequence_id] = (None, None)
            return self.state_cache[sequence_id]
        return self.conv_state, self.recurrent_state

    def _set_states(self, conv_state, recurrent_state, sequence_id: int | None):
        """Store conv_state and recurrent_state for this sequence."""
        if sequence_id is not None:
            self.state_cache[sequence_id] = (conv_state, recurrent_state)
        else:
            self.conv_state = conv_state
            self.recurrent_state = recurrent_state

    def _forward_graph(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Graph-mode forward for batch decode using persistent state buffers.

        Args:
            hidden_states: (bs, hidden_size) - batch of decode tokens

        Returns:
            output: (bs, hidden_size)
        """
        bs = hidden_states.size(0)
        hidden_states = hidden_states.to(self.in_proj_qkv.weight.dtype)
        tp_size = dist.get_world_size()
        conv_dim_tp = self.conv_dim // tp_size

        # Persistent graph state buffers (in-place updates are captured by CUDA graph)
        conv_state = self._graph_conv_state[:bs]
        recurrent_state = self._graph_recurrent_state[:bs]

        # Part 1: Input Projection
        mixed_qkv = self.in_proj_qkv(hidden_states)
        if isinstance(mixed_qkv, tuple):
            mixed_qkv = mixed_qkv[0]

        z = self.in_proj_z(hidden_states)
        if isinstance(z, tuple):
            z = z[0]
        z = z.reshape(bs, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        if isinstance(b, tuple):
            b = b[0]
        a = self.in_proj_a(hidden_states)
        if isinstance(a, tuple):
            a = a[0]

        b = b.contiguous()
        a = a.contiguous()

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim // tp_size, self.key_dim // tp_size, self.value_dim // tp_size],
            dim=-1
        )

        query = query.reshape(bs, self.num_k_heads // tp_size, self.head_k_dim)
        key = key.reshape(bs, self.num_k_heads // tp_size, self.head_k_dim)
        value = value.reshape(bs, self.num_v_heads // tp_size, self.head_v_dim)

        query_flat = query.reshape(bs, -1)
        key_flat = key.reshape(bs, -1)
        value_flat = value.reshape(bs, -1)

        mixed_qkv_conv = torch.cat([query_flat, key_flat, value_flat], dim=-1)
        # causal_conv1d_update expects (batch, dim, seq_len); decode seq_len=1
        mixed_qkv_conv = mixed_qkv_conv.unsqueeze(-1)

        conv_weights = self.conv1d.weight.reshape(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        mixed_qkv_conv = torch_causal_conv1d_update(
            mixed_qkv_conv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
        )
        # conv_state is updated in-place by causal_conv1d_update
        mixed_qkv_conv = mixed_qkv_conv.squeeze(-1)

        query_conv, key_conv, value_conv = torch.split(
            mixed_qkv_conv,
            [self.key_dim // tp_size, self.key_dim // tp_size, self.value_dim // tp_size],
            dim=-1
        )

        query_conv = query_conv.reshape(bs, self.num_k_heads // tp_size, self.head_k_dim)
        key_conv = key_conv.reshape(bs, self.num_k_heads // tp_size, self.head_k_dim)
        value_conv = value_conv.reshape(bs, self.num_v_heads // tp_size, self.head_v_dim)

        if self.num_v_heads // tp_size > self.num_k_heads // tp_size:
            repeat_factor = (self.num_v_heads // tp_size) // (self.num_k_heads // tp_size)
            query_conv = query_conv.repeat_interleave(repeat_factor, dim=1)
            key_conv = key_conv.repeat_interleave(repeat_factor, dim=1)

        # Gating
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        g = g.to(hidden_states.dtype)
        g = g.clamp(min=-50.0)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=-50.0)

        use_float32 = bool(os.environ.get("NANOVLLM_GDN_FLOAT32"))
        if use_float32:
            query_conv = query_conv.float()
            key_conv = key_conv.float()
            value_conv = value_conv.float()
            g = g.float()
            beta = beta.float()

        # Triton recurrent decode path (batch)
        q_batch = query_conv.unsqueeze(1).contiguous()
        k_batch = key_conv.unsqueeze(1).contiguous()
        v_batch = value_conv.unsqueeze(1).contiguous()
        g_batch = g.unsqueeze(1).contiguous()
        beta_batch = beta.unsqueeze(1).contiguous()

        use_triton = (
            not os.environ.get("NANOVLLM_FORCE_TORCH_GDN")
            and not use_float32
            and _HAS_FLA_TRITON
            and hidden_states.dtype != torch.float32
        )

        if use_triton:
            # ssm_state_indices avoids Triton None compilation issue during graph capture.
            # Each batch element i maps to state slot i in the graph buffer.
            ssm_state_indices = torch.arange(bs, dtype=torch.int64, device=hidden_states.device)
            core_attn_out_batch, _ = fla_fused_recurrent_gated_delta_rule(
                q=q_batch,
                k=k_batch,
                v=v_batch,
                g=g_batch,
                beta=beta_batch,
                initial_state=recurrent_state,
                inplace_final_state=True,
                ssm_state_indices=ssm_state_indices,
                use_qk_l2norm_in_kernel=True,
            )
            # recurrent_state is updated in-place
        else:
            # Torch fallback: initial_state expects (N, H, K, V)
            recurrent_state_torch = recurrent_state.permute(0, 1, 3, 2).contiguous()
            core_attn_out_batch, recurrent_state_torch = torch_recurrent_gated_delta_rule(
                query_conv,
                key_conv,
                value_conv,
                g,
                beta,
                initial_state=recurrent_state_torch,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            # Copy back to graph buffer in-place so CUDA graph captures it
            recurrent_state[:, :, :, :] = recurrent_state_torch.permute(0, 1, 3, 2).contiguous()

        core_attn_out = core_attn_out_batch.squeeze(1)

        if use_float32:
            core_attn_out = core_attn_out.to(hidden_states.dtype)

        # Output projection with gate
        core_attn_out_flat = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z_flat = z.reshape(-1, z.shape[-1])
        core_attn_out_flat = self.norm(core_attn_out_flat, z_flat)
        core_attn_out = core_attn_out_flat.reshape(bs, -1, self.head_v_dim)
        core_attn_out = core_attn_out.reshape(bs, -1)

        output = self.out_proj(core_attn_out)
        if isinstance(output, tuple):
            output = output[0]

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
        sequence_id: int | None = None,
        use_graph: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for GatedDeltaNet (HF-style)

        Args:
            hidden_states: (seq_len, hidden_size) - ONE sequence only (no mixed batch)
            positions: (3, seq_len) for MRoPE
            sequence_id: for batch inference, isolates state per sequence
            use_graph: if True, use graph-mode batch path (no sequence_id dict)

        Returns:
            output: (seq_len, hidden_size)
        """
        if use_graph:
            return self._forward_graph(hidden_states)

        seq_len = hidden_states.size(0)
        hidden_states = hidden_states.to(self.in_proj_qkv.weight.dtype)

        conv_state, recurrent_state = self._get_states(sequence_id)

        def _gdn_debug_this_layer():
            if not os.environ.get("NANOVLLM_DEBUG_GDN"):
                return False
            try:
                return self.layer_idx == int(os.environ.get("NANOVLLM_DEBUG_GDN_LAYER", "0"))
            except ValueError:
                return self.layer_idx == 0

        # Part 1: Input Projection (separate projections for Qwen3_5)
        mixed_qkv = self.in_proj_qkv(hidden_states)
        # mixed_qkv is a tuple/list from MergedColumnParallelLinear
        if isinstance(mixed_qkv, tuple):
            mixed_qkv = mixed_qkv[0]
        if _gdn_debug_this_layer():
            print(f"[NV GDN L{self.layer_idx}] mixed_qkv: shape={list(mixed_qkv.shape)}, mean={mixed_qkv.mean():.6f}, std={mixed_qkv.std():.6f}, min={mixed_qkv.min():.6f}, max={mixed_qkv.max():.6f}")

        z = self.in_proj_z(hidden_states)
        if isinstance(z, tuple):
            z = z[0]
        z = z.reshape(seq_len, -1, self.head_v_dim)
        if _gdn_debug_this_layer():
            print(f"[NV GDN L{self.layer_idx}] z after proj: shape={list(z.shape)}, mean={z.mean():.6f}, std={z.std():.6f}")

        b = self.in_proj_b(hidden_states)
        if isinstance(b, tuple):
            b = b[0]
        a = self.in_proj_a(hidden_states)
        if isinstance(a, tuple):
            a = a[0]
        if _gdn_debug_this_layer():
            print(f"[NV GDN L{self.layer_idx}] b after proj: shape={list(b.shape)}, mean={b.mean():.6f}, std={b.std():.6f}")
            print(f"[NV GDN L{self.layer_idx}] a after proj: shape={list(a.shape)}, mean={a.mean():.6f}, std={a.std():.6f}")

        b = b.contiguous()
        a = a.contiguous()
        
        # Split mixed_qkv into query, key, value
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim // dist.get_world_size(),
             self.key_dim // dist.get_world_size(),
             self.value_dim // dist.get_world_size()],
            dim=-1
        )
        
        # Reshape to (seq_len, num_heads, head_dim)
        query = query.reshape(seq_len, self.num_k_heads // dist.get_world_size(), self.head_k_dim)
        key = key.reshape(seq_len, self.num_k_heads // dist.get_world_size(), self.head_k_dim)
        value = value.reshape(seq_len, self.num_v_heads // dist.get_world_size(), self.head_v_dim)
        # Flatten for convolution
        query_flat = query.reshape(seq_len, -1)
        key_flat = key.reshape(seq_len, -1)
        value_flat = value.reshape(seq_len, -1)
        
        mixed_qkv_conv = torch.cat([query_flat, key_flat, value_flat], dim=-1)
        # Transpose for conv1d: (seq_len, conv_dim) -> (conv_dim, seq_len)
        mixed_qkv_conv = mixed_qkv_conv.transpose(0, 1).unsqueeze(0)  # (1, conv_dim, seq_len)
        
        # Part 2: Causal Convolution
        # HF-style: per-sequence conv_state (isolated by sequence_id)
        if conv_state is None:
            conv_state = F.pad(mixed_qkv_conv, (self.conv_kernel_size - 1, 0))[:, :, :self.conv_kernel_size - 1].contiguous()

        # Apply causal convolution
        conv_weights = self.conv1d.weight.reshape(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_conv = torch_causal_conv1d_update(
            mixed_qkv_conv,
            conv_state,
            conv_weights,  # (conv_dim // tp_size, kernel_size)
            self.conv1d.bias,
            self.activation,
        )
        
        # Transpose back: (1, conv_dim, seq_len) -> (seq_len, conv_dim)
        mixed_qkv_conv = mixed_qkv_conv.squeeze(0).transpose(0, 1)
        if _gdn_debug_this_layer():
            print(f"[NV GDN L{self.layer_idx}] after conv1d: shape={list(mixed_qkv_conv.shape)}, mean={mixed_qkv_conv.mean():.6f}, std={mixed_qkv_conv.std():.6f}, min={mixed_qkv_conv.min():.6f}, max={mixed_qkv_conv.max():.6f}")
        # Scheme C: numerical stability check after conv1d
        if os.environ.get("NANOVLLM_DEBUG_STABILITY"):
            m_max = mixed_qkv_conv.abs().max().item()
            has_nan = torch.isnan(mixed_qkv_conv).any().item()
            has_inf = torch.isinf(mixed_qkv_conv).any().item()
            if has_nan or has_inf or m_max > 1e4:
                print(f"[GDN layer{self.layer_idx}] post-conv1d: max_abs={m_max:.4f} has_nan={has_nan} has_inf={has_inf}")
        
        # Split back into query, key, value
        query_conv, key_conv, value_conv = torch.split(
            mixed_qkv_conv,
            [self.key_dim // dist.get_world_size(),
             self.key_dim // dist.get_world_size(),
             self.value_dim // dist.get_world_size()],
            dim=-1
        )
        
        # Reshape back to (seq_len, num_heads, head_dim)
        query_conv = query_conv.reshape(seq_len, self.num_k_heads // dist.get_world_size(), self.head_k_dim)
        key_conv = key_conv.reshape(seq_len, self.num_k_heads // dist.get_world_size(), self.head_k_dim)
        value_conv = value_conv.reshape(seq_len, self.num_v_heads // dist.get_world_size(), self.head_v_dim)
        if _gdn_debug_this_layer():
            print(f"[NV GDN L{self.layer_idx}] query: shape={list(query_conv.shape)}, mean={query_conv.mean():.6f}, std={query_conv.std():.6f}")
            print(f"[NV GDN L{self.layer_idx}] key: shape={list(key_conv.shape)}, mean={key_conv.mean():.6f}, std={key_conv.std():.6f}")
            print(f"[NV GDN L{self.layer_idx}] value: shape={list(value_conv.shape)}, mean={value_conv.mean():.6f}, std={value_conv.std():.6f}")
        # Repeat query/key if num_v_heads > num_k_heads
        if self.num_v_heads // dist.get_world_size() > self.num_k_heads // dist.get_world_size():
            repeat_factor = (self.num_v_heads // dist.get_world_size()) // (self.num_k_heads // dist.get_world_size())
            query_conv = query_conv.repeat_interleave(repeat_factor, dim=1)
            key_conv = key_conv.repeat_interleave(repeat_factor, dim=1)
        
        # Compute gating parameters
        beta = b.sigmoid()  # (seq_len, num_v_heads)
        if _gdn_debug_this_layer():
            print(f"[NV GDN L{self.layer_idx}] beta (after sigmoid): shape={list(beta.shape)}, mean={beta.mean():.6f}, std={beta.std():.6f}")
        # g = -exp(A_log) * softplus(a + dt_bias)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        g = g.to(hidden_states.dtype)
        # Numerical stability: avoid -inf/NaN
        g = g.clamp(min=-50.0)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=-50.0)
        if _gdn_debug_this_layer():
            print(f"[NV GDN L{self.layer_idx}] g: shape={list(g.shape)}, mean={g.mean():.6f}, std={g.std():.6f}, min={g.min():.6f}, max={g.max():.6f}")
        # Scheme C: stability check for g/beta
        if os.environ.get("NANOVLLM_DEBUG_STABILITY"):
            g_min, g_max = g.min().item(), g.max().item()
            if torch.isnan(g).any() or torch.isinf(g).any() or g_min < -100 or g_max > 100:
                print(f"[GDN layer{self.layer_idx}] gating: g_range=[{g_min:.4f},{g_max:.4f}]")

        # Diagnostic: check for NaN before chunk (all layers, when NANOVLLM_DEBUG_LOGITS)
        if os.environ.get("NANOVLLM_DEBUG_LOGITS"):
            qn, kn, vn = query_conv.norm().item(), key_conv.norm().item(), value_conv.norm().item()
            has_nan = torch.isnan(query_conv).any() or torch.isnan(key_conv).any() or torch.isnan(value_conv).any()
            if has_nan:
                print(f"[Qwen3_5 GatedDeltaNet] layer{self.layer_idx} pre-chunk: q_norm={qn:.4f} k_norm={kn:.4f} v_norm={vn:.4f} has_nan=True")

        # Part 3: Gated Delta Rule (Recurrent Attention)
        # Prefer Triton FLA ops when available; use NANOVLLM_FORCE_TORCH_GDN=1 to force torch (for debugging NaN).
        # NANOVLLM_GDN_FLOAT32=1: run in float32 for numerical stability (skips Triton).
        use_float32 = bool(os.environ.get("NANOVLLM_GDN_FLOAT32"))
        debug_state = bool(os.environ.get("NANOVLLM_DEBUG_GDN_STATE"))
        is_capturing = (
            torch.cuda.is_available()
            and hidden_states.is_cuda
            and torch.cuda.is_current_stream_capturing()
        )
        if use_float32:
            query_conv = query_conv.float()
            key_conv = key_conv.float()
            value_conv = value_conv.float()
            g = g.float()
            beta = beta.float()
        use_recurrent = (seq_len == 1 and recurrent_state is not None)
        # Force first decode-state initialization with canonical layout (N, H, V, K)
        # so triton recurrent path can do in-place updates on a stable storage.
        if seq_len == 1 and recurrent_state is None:
            recurrent_state = torch.zeros(
                1,
                self.num_v_heads // dist.get_world_size(),
                self.head_v_dim,
                self.head_k_dim,
                dtype=value_conv.dtype,
                device=value_conv.device,
            )
            use_recurrent = True

        decode_step = None
        if seq_len == 1:
            seq_key = sequence_id if sequence_id is not None else -1
            decode_step = self._decode_step_counter.get(seq_key, 0) + 1
            self._decode_step_counter[seq_key] = decode_step
        if debug_state and not is_capturing:
            print(
                "[GDN state] layer=%d seq_id=%s use_recurrent=%s recurrent_none=%s"
                % (self.layer_idx, str(sequence_id), str(use_recurrent), str(recurrent_state is None))
            )
            if decode_step is not None and decode_step <= 3 and recurrent_state is not None:
                before_decode_sum = float(recurrent_state.float().sum().item())
                print(
                    "[GDN state] decode step=%d layer=%d seq_id=%s before_sum=%.6f"
                    % (decode_step, self.layer_idx, str(sequence_id), before_decode_sum)
                )
        # Canonical cache layout is (N, H, V, K).
        # NOTE: for Qwen3.5-0.8B, head_k_dim == head_v_dim (both 128), so shape-based
        # auto-detection of (K,V) vs (V,K) is ambiguous and can corrupt state.
        # Keep layout fixed and only convert explicitly at torch fallback boundaries.
        if debug_state and not is_capturing and recurrent_state is not None:
            print(
                "[GDN state] cache layout (N,H,V,K) shape=%s stride=%s contig=%s"
                % (
                    str(tuple(recurrent_state.shape)),
                    str(tuple(recurrent_state.stride())),
                    str(recurrent_state.is_contiguous()),
                )
            )
        use_triton = (
            not os.environ.get("NANOVLLM_FORCE_TORCH_GDN")
            and not use_float32
            and _HAS_FLA_TRITON
            and hidden_states.dtype != torch.float32
        )
        core_attn_out = None
        triton_succeeded = False
        if use_triton:
            try:
                # FLA Triton expects (B, T, H, K) / (B, T, H, V) / (B, T, H); initial_state (N, H, V, K)
                q_batch = query_conv.unsqueeze(0).contiguous()
                k_batch = key_conv.unsqueeze(0).contiguous()
                v_batch = value_conv.unsqueeze(0).contiguous()
                g_batch = g.unsqueeze(0).contiguous()
                beta_batch = beta.unsqueeze(0).contiguous()
                initial_state_batch = None
                if recurrent_state is not None:
                    # Cache layout is canonicalized to (N, H, V, K).
                    initial_state_batch = recurrent_state
                    if debug_state and not is_capturing:
                        before_sum = float(initial_state_batch.float().sum().item())
                        print(
                            "[GDN state] triton init shape=%s stride=%s contig=%s sum=%.6f"
                            % (
                                str(tuple(initial_state_batch.shape)),
                                str(tuple(initial_state_batch.stride())),
                                str(initial_state_batch.is_contiguous()),
                                before_sum,
                            )
                        )

                # HF-style: one sequence per call, no cu_seqlens
                cu_seqlens = None

                if use_recurrent:
                    core_attn_out_batch, final_state_batch = fla_fused_recurrent_gated_delta_rule(
                        q=q_batch,
                        k=k_batch,
                        v=v_batch,
                        g=g_batch,
                        beta=beta_batch,
                        initial_state=initial_state_batch,
                        inplace_final_state=True,
                        cu_seqlens=cu_seqlens,
                        use_qk_l2norm_in_kernel=True,
                    )
                else:
                    core_attn_out_batch, final_state_batch = fla_chunk_gated_delta_rule(
                        q=q_batch,
                        k=k_batch,
                        v=v_batch,
                        g=g_batch,
                        beta=beta_batch,
                        initial_state=initial_state_batch,
                        output_final_state=True,
                        cu_seqlens=cu_seqlens,
                        use_qk_l2norm_in_kernel=True,
                    )
                core_attn_out = core_attn_out_batch.squeeze(0)
                # NaN fallback: if Triton produced NaN, retry with torch (do not use corrupted state)
                if torch.isnan(core_attn_out).any():
                    if not getattr(Qwen3_5GatedDeltaNet, "_nan_fallback_logged", False):
                        print("[Qwen3_5 GatedDeltaNet] Triton produced NaN, falling back to torch (set NANOVLLM_FORCE_TORCH_GDN=1 to skip Triton)")
                        Qwen3_5GatedDeltaNet._nan_fallback_logged = True
                    core_attn_out = None
                else:
                    triton_succeeded = True
                    # For recurrent mode with inplace_final_state=True, state is
                    # already written back into `initial_state_batch` storage.
                    if (not use_recurrent) and final_state_batch is not None:
                        # Chunk path returns explicit final state in (N,H,V,K).
                        recurrent_state = final_state_batch.contiguous()
                    if debug_state and not is_capturing and recurrent_state is not None:
                        after_sum = float(recurrent_state.float().sum().item())
                        print(
                            "[GDN state] triton updated shape=%s stride=%s contig=%s sum=%.6f"
                            % (
                                str(tuple(recurrent_state.shape)),
                                str(tuple(recurrent_state.stride())),
                                str(recurrent_state.is_contiguous()),
                                after_sum,
                            )
                        )
            except Exception:
                use_triton = False

        if core_attn_out is None:
            if use_recurrent:
                # Torch fallback expects initial_state layout (N, H, K, V).
                recurrent_state_torch = (
                    recurrent_state.permute(0, 1, 3, 2).contiguous()
                    if recurrent_state is not None
                    else None
                )
                core_attn_out, recurrent_state = torch_recurrent_gated_delta_rule(
                    query_conv,
                    key_conv,
                    value_conv,
                    g,
                    beta,
                    initial_state=recurrent_state_torch,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                # Convert back to canonical cache layout (N, H, V, K).
                if recurrent_state is not None:
                    recurrent_state = recurrent_state.permute(0, 1, 3, 2).contiguous()
            else:
                recurrent_state_torch = (
                    recurrent_state.permute(0, 1, 3, 2).contiguous()
                    if recurrent_state is not None
                    else None
                )
                core_attn_out, recurrent_state = torch_chunk_gated_delta_rule(
                    query_conv,
                    key_conv,
                    value_conv,
                    g,
                    beta,
                    chunk_size=64,
                    initial_state=recurrent_state_torch,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                if recurrent_state is not None:
                    recurrent_state = recurrent_state.permute(0, 1, 3, 2).contiguous()
        if use_float32 and core_attn_out is not None:
            core_attn_out = core_attn_out.to(hidden_states.dtype)
            if recurrent_state is not None:
                recurrent_state = recurrent_state.to(hidden_states.dtype)
        if (
            debug_state
            and not is_capturing
            and decode_step is not None
            and decode_step <= 3
            and recurrent_state is not None
        ):
            after_decode_sum = float(recurrent_state.float().sum().item())
            print(
                "[GDN state] decode step=%d layer=%d seq_id=%s after_sum=%.6f"
                % (decode_step, self.layer_idx, str(sequence_id), after_decode_sum)
            )
        if _gdn_debug_this_layer() and core_attn_out is not None:
            print(f"[NV GDN L{self.layer_idx}] core_attn_out: shape={list(core_attn_out.shape)}, mean={core_attn_out.mean():.6f}, std={core_attn_out.std():.6f}, min={core_attn_out.min():.6f}, max={core_attn_out.max():.6f}")
        if os.environ.get("NANOVLLM_DEBUG_LOGITS") and core_attn_out is not None and torch.isnan(core_attn_out).any():
            backend = "triton" if triton_succeeded else "torch"
            print(f"[Qwen3_5 GatedDeltaNet] layer{self.layer_idx} post-chunk ({backend}): has_nan=True")
        # Scheme C: stability check after chunk/recurrent
        if os.environ.get("NANOVLLM_DEBUG_STABILITY") and core_attn_out is not None:
            c_max = core_attn_out.abs().max().item()
            has_nan = torch.isnan(core_attn_out).any().item()
            if has_nan or c_max > 1e4:
                backend = "triton" if triton_succeeded else "torch"
                print(f"[GDN layer{self.layer_idx}] post-chunk ({backend}): max_abs={c_max:.4f} has_nan={has_nan}")
        # Log backend once (Triton vs torch) and dtype
        if not getattr(Qwen3_5GatedDeltaNet, "_gdn_backend_logged", False):
            backend = "triton" if triton_succeeded else "torch"
            # print(f"[Qwen3_5 GatedDeltaNet] backend={backend} dtype={hidden_states.dtype}")
            Qwen3_5GatedDeltaNet._gdn_backend_logged = True

        # Persist states for this sequence (HF-style per-seq isolation)
        self._set_states(conv_state, recurrent_state, sequence_id)

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
        if _gdn_debug_this_layer():
            print(f"[NV GDN L{self.layer_idx}] after norm: shape={list(core_attn_out.shape)}, mean={core_attn_out.mean():.6f}, std={core_attn_out.std():.6f}")
        if os.environ.get("NANOVLLM_DEBUG_LOGITS") and torch.isnan(core_attn_out).any():
            print(f"[Qwen3_5 GatedDeltaNet] layer{self.layer_idx} post-norm: has_nan=True")
        
        # Flatten for output projection
        core_attn_out = core_attn_out.reshape(seq_len, -1)
        
        # Output projection
        # RowParallelLinear may return tuple in some implementations
        output = self.out_proj(core_attn_out)
        if isinstance(output, tuple):
            output = output[0]
        if _gdn_debug_this_layer():
            print(f"[NV GDN L{self.layer_idx}] output: shape={list(output.shape)}, mean={output.mean():.6f}, std={output.std():.6f}, min={output.min():.6f}, max={output.max():.6f}")
        if os.environ.get("NANOVLLM_DEBUG_LOGITS") and torch.isnan(output).any():
            print(f"[Qwen3_5 GatedDeltaNet] layer{self.layer_idx} post-out_proj: has_nan=True")

        return output


class Qwen3_5TextDecoderLayerMerged(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_scaling = None

        self.self_attn = Qwen3_5TextAttentionMerged(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=rope_scaling,
        )
        self.mlp = Qwen3_5TextMLPMerged(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3_5TextDecoderLayer(nn.Module):
    """Qwen3_5 Text Decoder Layer with layer_types support"""

    def __init__(
        self,
        config,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_scaling = None

        self.layer_idx = layer_idx
        # Support layer_types (full_attention or linear_attention)
        # Must match transformers default when layer_types is None: (i+1) % 4 -> linear else full
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            self.layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
        else:
            interval = getattr(config, "full_attention_interval", 4)
            self.layer_type = (
                "linear_attention" if (layer_idx + 1) % interval else "full_attention"
            )

        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5TextAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_position=config.max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, "attention_bias", False),
                head_dim=getattr(config, "head_dim", None),
                rope_theta=getattr(config, "rope_theta", 1000000),
                rope_scaling=rope_scaling,
                config=config,
            )
            self.linear_attn = None
        elif self.layer_type == "linear_attention":
            self.self_attn = None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config=config,
                layer_idx=layer_idx,
            )
        else:
            raise NotImplementedError(f"Layer type {self.layer_type} not implemented")

        self.mlp = Qwen3_5TextMLPMerged(
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
        sequence_ids: list[int] | None = None,
        sequence_lengths: list[int] | None = None,
        use_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Optional debug: check residual propagation at layer 1.
        if os.environ.get("NANOVLLM_DEBUG_RESIDUAL") and self.layer_idx == 1 and residual is not None:
            print(
                f"[DEBUG Layer 1] hidden_states.norm()={hidden_states.norm().item():.4f} "
                f"residual.norm()={residual.norm().item():.4f}"
            )
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # Scheme A: layer-wise diagnostics (NANOVLLM_DEBUG_LAYERS=1 prints every layer)
        if os.environ.get("NANOVLLM_DEBUG_LAYERS"):
            norm_val = hidden_states.norm().item() if hidden_states.numel() > 0 else float("nan")
            has_nan = torch.isnan(hidden_states).any().item()
            print(f"[Layer {self.layer_idx}] pre-attn norm={norm_val:.4f} has_nan={has_nan} type={self.layer_type}")
        elif os.environ.get("NANOVLLM_DEBUG_LOGITS") and torch.isnan(hidden_states).any():
            print(f"[Qwen3_5 DecoderLayer] layer{self.layer_idx} pre-attn (after input_layernorm): has_nan=True")

        if self.layer_type == "full_attention":
            debug_fa = os.environ.get("NANOVLLM_DEBUG_FA")
            try:
                debug_fa_layer = int(os.environ.get("NANOVLLM_DEBUG_FA_LAYER", "3"))
            except ValueError:
                debug_fa_layer = 3
            if debug_fa and self.layer_idx == debug_fa_layer:
                FA_DEBUG_SAVE["pre_attn_norm"] = hidden_states.detach().cpu().float()
            attn_out = self.self_attn(positions, hidden_states)
            if debug_fa and self.layer_idx == debug_fa_layer:
                FA_DEBUG_SAVE["attn_output"] = attn_out.detach().cpu().float()
            hidden_states = attn_out
        elif self.layer_type == "linear_attention":
            if use_graph:
                # CUDA graph mode: process entire batch at once using persistent state buffers
                hidden_states = self.linear_attn(hidden_states, positions, use_graph=True)
            elif sequence_lengths is not None and len(sequence_lengths) > 1 and sequence_ids is not None:
                # HF-style: per-sequence isolation - process each sequence separately to avoid state cross-contamination
                outputs = []
                offset = 0
                pos_2d = positions.dim() == 2
                for seq_len, seq_id in zip(sequence_lengths, sequence_ids):
                    h = hidden_states[offset : offset + seq_len]
                    p = positions[:, offset : offset + seq_len] if pos_2d else positions[offset : offset + seq_len]
                    out = self.linear_attn(h, p, sequence_id=seq_id)
                    outputs.append(out)
                    offset += seq_len
                hidden_states = torch.cat(outputs, dim=0)
            else:
                seq_id = sequence_ids[0] if (sequence_ids is not None and len(sequence_ids) == 1) else None
                hidden_states = self.linear_attn(hidden_states, positions, sequence_id=seq_id)
        else:
            raise NotImplementedError(f"Layer type {self.layer_type} not implemented")
        
        hidden_states = hidden_states + residual
        if os.environ.get("NANOVLLM_DEBUG_LAYERS"):
            norm_val = hidden_states.norm().item() if hidden_states.numel() > 0 else float("nan")
            has_nan = torch.isnan(hidden_states).any().item()
            print(f"[Layer {self.layer_idx}] post-attn norm={norm_val:.4f} has_nan={has_nan}")
        elif os.environ.get("NANOVLLM_DEBUG_LOGITS") and torch.isnan(hidden_states).any():
            print(f"[Qwen3_5 DecoderLayer] layer{self.layer_idx} post-attn ({self.layer_type}): has_nan=True")
        
        # HF: residual = hidden_states, then norm(hidden_states); add residual after mlp
        residual = hidden_states
        hidden_states, _ = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        if os.environ.get("NANOVLLM_DEBUG_LAYERS"):
            norm_val = hidden_states.norm().item() if hidden_states.numel() > 0 else float("nan")
            has_nan = torch.isnan(hidden_states).any().item()
            print(f"[Layer {self.layer_idx}] post-mlp norm={norm_val:.4f} has_nan={has_nan}")
        elif os.environ.get("NANOVLLM_DEBUG_LOGITS") and torch.isnan(hidden_states).any():
            print(f"[Qwen3_5 DecoderLayer] layer{self.layer_idx} post-mlp: has_nan=True")
        # Return full layer output as residual for next layer (align with vLLM/HF semantics)
        return hidden_states, hidden_states


class Qwen3_5TextModelMerged(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5TextDecoderLayerMerged(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        vision_token_count: int | None = None,
        visual_pos_mask: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        # vLLM-style: keep activation dtype consistent with weights
        hidden_states = hidden_states.to(self.embed_tokens.weight.dtype)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # HF: final norm only normalizes hidden_states (no residual merge)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3_5TextForCausalLMMerged(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen3_5TextModelMerged(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        vision_token_count: int | None = None,
        visual_pos_mask: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **_: dict,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            vision_token_count=vision_token_count,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        return self.lm_head(hidden_states)


class Qwen3_5TextModel(nn.Module):
    """Qwen3_5 Text Model"""

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5TextDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        vision_token_count: int | None = None,
        visual_pos_mask: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        sequence_ids: list[int] | None = None,
        sequence_lengths: list[int] | None = None,
        use_graph: bool = False,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        # vLLM-style: keep activation dtype consistent with weights
        hidden_states = hidden_states.to(self.embed_tokens.weight.dtype)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual,
                sequence_ids=sequence_ids,
                sequence_lengths=sequence_lengths,
                use_graph=use_graph,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3_5TextForCausalLM(nn.Module):
    """Qwen3_5 Text For Causal LM"""
    
    packed_modules_mapping = {
        "q_proj": ("q_proj", None),
        "k_proj": ("k_proj", None),
        "v_proj": ("v_proj", None),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen3_5TextModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        vision_token_count: int | None = None,
        visual_pos_mask: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        sequence_ids: list[int] | None = None,
        sequence_lengths: list[int] | None = None,
        use_graph: bool = False,
        **_: dict,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            vision_token_count=vision_token_count,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
            sequence_ids=sequence_ids,
            sequence_lengths=sequence_lengths,
            use_graph=use_graph,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# Vision encoder (Qwen3.5: vision tower only; no DeepStack)
# ---------------------------------------------------------------------------


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    inner = math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1 + torch.tanh(inner))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).to(q.dtype)
    sin = sin.unsqueeze(0).to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = getattr(config, "temporal_patch_size", 1)
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ]
        stride = kernel_size
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(2)
        hidden_states = self.proj(inputs)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        if getattr(config, "hidden_act", "gelu") == "gelu_pytorch_tanh":
            self.act_fn = gelu_pytorch_tanh
        else:
            self.act_fn = lambda x: F.gelu(x, approximate="tanh")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_fc2(hidden_states)
        return hidden_states


class Qwen3_5VisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=True)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_lengths: Sequence[int],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        outputs = []
        offset = 0
        cos, sin = position_embeddings
        for length in seq_lengths:
            chunk = hidden_states[offset : offset + length]
            cos_chunk = cos[offset : offset + length]
            sin_chunk = sin[offset : offset + length]

            qkv = self.qkv(chunk)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.reshape(length, self.num_heads, self.head_dim).transpose(0, 1)
            k = k.reshape(length, self.num_heads, self.head_dim).transpose(0, 1)
            v = v.reshape(length, self.num_heads, self.head_dim).transpose(0, 1)

            q, k = apply_rotary_pos_emb_vision(q, k, cos_chunk, sin_chunk)
            if q.dtype != v.dtype:
                q = q.to(v.dtype)
                k = k.to(v.dtype)

            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(v.dtype)
            attn_output = torch.matmul(attn_weights, v)

            attn_output = attn_output.transpose(0, 1).reshape(length, self.hidden_size)
            attn_output = self.proj(attn_output)
            outputs.append(attn_output)
            offset += length

        return torch.cat(outputs, dim=0)


class Qwen3_5VisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size, bias=True)
        self.act_fn = nn.GELU()
        self.merge_size = config.spatial_merge_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.reshape(-1, self.hidden_size)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_fc2(hidden_states)
        return hidden_states


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = Qwen3_5VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_lengths: Sequence[int],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = residual + self.attn(hidden_states, seq_lengths, position_embeddings)
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Qwen3_5VisionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Qwen3_5VisionPatchEmbed(config)
        self.hidden_size = config.hidden_size
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3_5VisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3_5VisionPatchMerger(config=config, use_postshuffle_norm=False)
        # Qwen3.5 has no DeepStack

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h = height // merge_size
            merged_w = width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row.view(1, 1, -1, 1)
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col.view(1, 1, 1, -1)
            )

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset: offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = grid_thw.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=device)

            h_floor = h_idxs.long()
            w_floor = w_idxs.long()
            h_ceil = (h_floor + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds.sum(dim=0)

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(int(t.item()), 1)
            pos_embed = (
                pos_embed.view(int(t.item()), h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        pixel_values = pixel_values.to(self.pos_embed.weight.dtype)
        seq_tokens = self.patch_embed(pixel_values)
        hidden_states = seq_tokens.reshape(-1, self.hidden_size)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(hidden_states.size(0), -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        seq_lengths = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()

        for block in self.blocks:
            hidden_states = block(hidden_states, seq_lengths, position_embeddings)

        hidden_states = self.merger(hidden_states)
        return hidden_states


class Qwen3VisionEncoder(nn.Module):
    def __init__(self, vision_config) -> None:
        super().__init__()
        self.config = vision_config
        self.vision = Qwen3_5VisionModel(vision_config)

    def _linear_patch_embed(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        proj = self.vision.patch_embed.proj
        weight = proj.weight.view(proj.out_channels, -1)
        bias = proj.bias
        return torch.nn.functional.linear(patch_tokens, weight, bias)

    def _run_vision_from_tokens(
        self,
        token_list: list[torch.Tensor],
        grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        proj = self.vision.patch_embed.proj
        device = proj.weight.device
        dtype = proj.weight.dtype

        tokens = torch.cat([t.to(device=device, dtype=dtype) for t in token_list], dim=0)
        grids = grid_thw.to(device=device, dtype=torch.int32)

        hidden_states = tokens

        pos_embeds = self.vision.fast_pos_embed_interpolate(grids).to(hidden_states.dtype)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.vision.rot_pos_emb(grids).to(hidden_states.dtype)
        rotary_pos_emb = rotary_pos_emb.reshape(hidden_states.size(0), -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        seq_lengths = (grids[:, 0] * grids[:, 1] * grids[:, 2]).tolist()

        for block in self.vision.blocks:
            hidden_states = block(hidden_states, seq_lengths, position_embeddings)

        hidden_states = self.vision.merger(hidden_states)

        split_sizes = (
            grids.prod(-1) // (self.config.spatial_merge_size**2)
        ).tolist()

        image_chunks = list(torch.split(hidden_states, split_sizes))
        return image_chunks

    def _normalize_pixel_inputs(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        in_channels = getattr(self.config, "in_channels", 3)
        num_dims = pixel_values.dim()

        channel_axis = None
        for axis in range(1, num_dims - 2):
            if pixel_values.shape[axis] == in_channels:
                channel_axis = axis
                break
        if channel_axis is None:
            channel_axis = 1

        permute_order = [0, channel_axis]
        temporal_axes = [
            axis for axis in range(1, num_dims - 2) if axis != channel_axis
        ]
        permute_order.extend(temporal_axes)
        permute_order.extend([num_dims - 2, num_dims - 1])

        pixel_values = pixel_values.permute(*permute_order).contiguous()

        batch = pixel_values.shape[0]
        channels = pixel_values.shape[1]
        height = pixel_values.shape[-2]
        width = pixel_values.shape[-1]

        temporal = int(math.prod(pixel_values.shape[2:-2]))
        pixel_values = pixel_values.reshape(
            batch,
            channels,
            temporal,
            height,
            width,
        )

        return pixel_values, batch, temporal, height, width

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pixel_values.dim() <= 3:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required for flattened inputs")
            grids = image_grid_thw.to(pixel_values.device).to(torch.int64)
            tokens_per_image = grids.prod(-1).tolist()
            if pixel_values.dim() == 3:
                batch, tokens, feature = pixel_values.shape
                flat = pixel_values.reshape(batch * tokens, feature)
            else:
                flat = pixel_values

            splits = torch.split(flat, tokens_per_image, dim=0)
            token_list = [self._linear_patch_embed(chunk) for chunk in splits]

            image_chunks = self._run_vision_from_tokens(token_list, grids)
            return torch.stack(list(image_chunks), dim=0)

        pixel_values, batch, temporal, height, width = self._normalize_pixel_inputs(
            pixel_values
        )

        if image_grid_thw is None:
            grid = torch.tensor(
                [
                    [
                        temporal,
                        height // self.config.patch_size,
                        width // self.config.patch_size,
                    ]
                ]
                * batch,
                device=pixel_values.device,
                dtype=torch.int32,
            )
            image_grid_thw = grid
        else:
            if image_grid_thw.dim() == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
            image_grid_thw = image_grid_thw.to(device=pixel_values.device, dtype=torch.int32)

        image_embeds = self.vision(pixel_values, image_grid_thw)
        split_sizes = (
            image_grid_thw.prod(-1) // (self.config.spatial_merge_size**2)
        ).tolist()

        image_chunks = torch.split(image_embeds, split_sizes)
        image_tokens = torch.stack(list(image_chunks), dim=0)
        return image_tokens


def create_vision_model(config, **kwargs) -> nn.Module:
    del kwargs
    return Qwen3VisionEncoder(config)


def get_vision_model(config, **kwargs) -> nn.Module:
    return create_vision_model(config, **kwargs)


# ---------------------------------------------------------------------------
# Multimodal wrapper
# ---------------------------------------------------------------------------


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Qwen3.5 conditional generation model (language + vision)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_config = getattr(config, "text_config", config)
        self.vision_config = getattr(config, "vision_config", None)

        if self.vision_config is None:
            raise ValueError("vision_config is missing; cannot build a multimodal model")

        self.visual = create_vision_model(self.vision_config)
        self.language_model = Qwen3_5TextForCausalLM(self.text_config)

        print("[Qwen3_5ForConditionalGeneration] Initialization complete")
        print(f"  - Vision encoder: {type(self.visual).__name__}")
        print(f"  - Language model: {type(self.language_model).__name__}")

        self.packed_modules_mapping = {
            "mlp.gate_proj": ("mlp.gate_up_proj", 0),
            "mlp.up_proj": ("mlp.gate_up_proj", 1),
            "q_proj": ("q_proj", None),
            "k_proj": ("k_proj", None),
            "v_proj": ("v_proj", None),
        }

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.model.embed_tokens(input_ids)

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: List[int] | torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Compute 3D positional indices for vision tokens (PR 43972)."""
        if isinstance(grid_thw, torch.Tensor):
            g0, g1, g2 = grid_thw[0].item(), grid_thw[1].item(), grid_thw[2].item()
        else:
            g0, g1, g2 = grid_thw[0], grid_thw[1], grid_thw[2]
        llm_grid_t = g0 // temp_merge_size
        llm_grid_h = g1 // spatial_merge_size
        llm_grid_w = g2 // spatial_merge_size
        image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
        if device is None:
            device = next(self.parameters()).device
        position_width = torch.arange(
            start_position, start_position + llm_grid_w, device=device, dtype=torch.long
        ).repeat(llm_grid_h * llm_grid_t)
        position_height = torch.arange(
            start_position, start_position + llm_grid_h, device=device, dtype=torch.long
        ).repeat_interleave(llm_grid_w * llm_grid_t)
        position_temporal = torch.full(
            (image_seq_length,), start_position, device=device, dtype=torch.long
        )
        position_temporal = position_temporal * time_interval
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def _build_3d_position_ids_for_sequence(
        self,
        seq_len: int,
        sorted_slices: list[dict],
        image_grid_thw: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Build (3, seq_len) position ids for one sequence from text/vision segments."""
        spatial_merge_size = getattr(self.vision_config, "spatial_merge_size", 1)
        chunks: list[torch.Tensor] = []
        current_pos = 0
        for seg_start, seg_end, is_vision, placeholder_idx in self._segment_ranges(
            seq_len, sorted_slices
        ):
            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue
            if is_vision and image_grid_thw is not None and placeholder_idx >= 0 and placeholder_idx < image_grid_thw.size(0):
                grid = image_grid_thw[placeholder_idx]
                pos_3d = self.get_vision_position_ids(
                    current_pos, grid, temp_merge_size=1,
                    spatial_merge_size=spatial_merge_size, device=device,
                )
                if pos_3d.size(1) == seg_len:
                    chunks.append(pos_3d)
                else:
                    ar = torch.arange(seg_len, device=device, dtype=torch.long) + current_pos
                    chunks.append(ar.view(1, -1).expand(3, -1))
            else:
                ar = torch.arange(seg_len, device=device, dtype=torch.long) + current_pos
                chunks.append(ar.view(1, -1).expand(3, -1))
            current_pos += seg_len
        if not chunks:
            ar = torch.arange(seq_len, device=device, dtype=torch.long)
            return ar.view(1, -1).expand(3, -1)
        return torch.cat(chunks, dim=1)

    @staticmethod
    def _segment_ranges(
        seq_len: int, sorted_slices: list[dict]
    ) -> list[tuple[int, int, bool, int]]:
        """Yield (start, end, is_vision, placeholder_idx) in order."""
        out: list[tuple[int, int, bool, int]] = []
        current = 0
        for s in sorted_slices:
            off = s["target_offset"]
            length = s["length"]
            pid = s.get("placeholder_idx", 0)
            if current < off:
                out.append((current, off, False, -1))
            out.append((off, off + length, True, pid))
            current = off + length
        if current < seq_len:
            out.append((current, seq_len, False, -1))
        return out

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        sequence_lengths: list[int] | None = None,
        sequence_ids: list[int] | None = None,
        vision_slices_per_seq: list[list[dict]] | None = None,
        image_grid_thw_per_seq: list[torch.Tensor | None] | None = None,
        use_graph: bool = False,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids and inputs_embeds cannot be None simultaneously")
            inputs_embeds = self.get_input_embeddings(input_ids)

        total_tokens = inputs_embeds.size(0)
        inputs_embeds = inputs_embeds.clone()

        visual_pos_mask = torch.zeros(
            total_tokens, dtype=torch.bool, device=inputs_embeds.device
        )
        vision_token_count = 0

        if vision_slices_per_seq:
            if sequence_lengths is None:
                raise ValueError("sequence_lengths must be provided to align visual features")
            if len(sequence_lengths) != len(vision_slices_per_seq):
                raise ValueError("sequence_lengths and vision_slices_per_seq have different lengths")
            if sum(sequence_lengths) != total_tokens:
                raise ValueError("sum of sequence_lengths does not match total input tokens")

            offsets = [0]
            for length in sequence_lengths:
                offsets.append(offsets[-1] + length)

            for seq_idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
                seq_slices = vision_slices_per_seq[seq_idx]
                if not seq_slices:
                    continue

                for slice_info in seq_slices:
                    token_slice = slice_info["tokens"].to(
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )
                    length = slice_info["length"]
                    target_offset = slice_info["target_offset"]

                    if token_slice.size(0) != length:
                        raise ValueError("Visual token slice length does not match the declared length")

                    target_start = start + target_offset
                    target_end = target_start + length
                    if target_end > end:
                        raise ValueError("Visual token target range is out of bounds")

                    inputs_embeds[target_start:target_end] = token_slice
                    visual_pos_mask[target_start:target_end] = True
                    vision_token_count += length

        elif pixel_values is not None:
            # Fallback path: process raw images when slices are not provided (legacy compatibility)
            if input_ids is None:
                raise ValueError("input_ids are required to locate visual placeholders")
            if sequence_lengths is None:
                raise ValueError("sequence_lengths are required to align visual features")
            if sum(sequence_lengths) != total_tokens:
                raise ValueError("sum of sequence_lengths does not match total input tokens")

            image_tokens = self.visual(pixel_values, image_grid_thw)
            if image_tokens is None or image_tokens.numel() == 0:
                raise ValueError("The vision encoder did not return valid image features")

            offsets = [0]
            for length in sequence_lengths:
                offsets.append(offsets[-1] + length)

            total_replaced = 0
            image_chunks = [image_tokens[i] for i in range(image_tokens.size(0))]
            image_iter = iter(image_chunks)

            for start, end in zip(offsets[:-1], offsets[1:]):
                seq_length = end - start
                if seq_length <= 0:
                    continue

                try:
                    token_slice = next(image_iter)
                except StopIteration:
                    break

                token_slice = token_slice.to(inputs_embeds.device, inputs_embeds.dtype)
                slice_len = token_slice.size(0)
                if slice_len > seq_length:
                    raise ValueError("Visual tokens exceed the available sequence length")

                inputs_embeds[start : start + slice_len] = token_slice
                visual_pos_mask[start : start + slice_len] = True
                total_replaced += slice_len

            vision_token_count = total_replaced

        if vision_token_count == 0:
            visual_pos_mask = None

        # Build 3D position_ids (PR 43972) when we have vision segments and grid per sequence
        if (
            vision_slices_per_seq is not None
            and image_grid_thw_per_seq is not None
            and len(image_grid_thw_per_seq) == len(sequence_lengths or [])
            and sum(sequence_lengths or []) == total_tokens
        ):
            device = inputs_embeds.device
            pos_chunks = []
            for seq_idx, seq_len in enumerate(sequence_lengths or []):
                slices = vision_slices_per_seq[seq_idx]
                sorted_slices = sorted(slices, key=lambda s: s["target_offset"])
                grid = image_grid_thw_per_seq[seq_idx]
                pos_3d = self._build_3d_position_ids_for_sequence(
                    seq_len, sorted_slices, grid, device
                )
                pos_chunks.append(pos_3d)
            if pos_chunks:
                positions = torch.cat(pos_chunks, dim=1)

        # For Qwen3.5, even pure text input needs 3D position_ids (T, H, W) for MRoPE
        if positions is None:
            positions = torch.arange(total_tokens, device=inputs_embeds.device)
            positions = positions.view(1, 1, -1).expand(3, 1, -1).reshape(3, total_tokens)
        elif positions.ndim == 1:
            if total_tokens != positions.size(0):
                positions = torch.arange(total_tokens, device=inputs_embeds.device)
            positions = positions.view(1, -1).expand(3, -1)

        if visual_pos_mask is not None and vision_token_count:
            visual_pos_mask = visual_pos_mask.to(inputs_embeds.device)
        else:
            visual_pos_mask = None

        # HF-style: per-sequence GDN state isolation
        # Use actual sequence_ids from the scheduler (seq.seq_id) so that
        # GDN state_cache keys remain stable even when batch composition changes.
        seq_ids = sequence_ids if sequence_ids is not None else (list(range(len(sequence_lengths))) if sequence_lengths else None)
        seq_lens = sequence_lengths

        hidden_states = self.language_model(
            input_ids=None,
            positions=positions,
            inputs_embeds=inputs_embeds,
            vision_token_count=vision_token_count,
            visual_pos_mask=visual_pos_mask,
            sequence_ids=seq_ids,
            sequence_lengths=seq_lens,
            use_graph=use_graph,
        )

        return hidden_states

    def compute_logits(self, hidden_states):
        """Compute logits (delegate to language model)"""
        return self.language_model.compute_logits(hidden_states)


def _qwen_multimodal_name_mapping(weight_name: str) -> str | None:
    """Weight name mapping for Qwen3.5 multimodal checkpoints."""
    # lm_head.* (top-level, some checkpoints save as "lm_head" without "model." prefix)
    if weight_name.startswith("lm_head."):
        return "language_model." + weight_name
    # model.lm_head.* (top-level lm_head under Qwen3_5ForConditionalGeneration)
    if weight_name.startswith("model.lm_head."):
        return "language_model." + weight_name[len("model.") :]
    if weight_name.startswith("model.language_model."):
        sub_name = weight_name[len("model.language_model.") :]
        text_model_prefixes = (
            "model.",
            "embed_tokens.",
            "layers.",
            "norm.",
            "rotary_emb.",
        )
        if sub_name.startswith(text_model_prefixes):
            if sub_name.startswith("model."):
                sub_name = sub_name[len("model.") :]
            sub_name = "language_model.model." + sub_name
        elif sub_name.startswith("lm_head."):
            sub_name = "language_model.lm_head." + sub_name[len("lm_head.") :]
        else:
            sub_name = "language_model." + sub_name
        return sub_name
    if weight_name.startswith("model.visual."):
        sub_name = weight_name[len("model.visual.") :]
        return "visual.vision." + sub_name
    return None


def load_qwen3_5_model(model_path, config):
    """
    Load Qwen3.5 (or Qwen3.5-MoE) multimodal model.

    Args:
        model_path: Model path
        config: Configuration object

    Returns:
        model: Qwen3_5ForConditionalGeneration instance
    """
    hf_config = config.hf_config
    model = Qwen3_5ForConditionalGeneration(hf_config)
    from nanovllm.utils.loader import load_model
    print("[load_qwen3_5_model] Loading Qwen3.5 multimodal weights...")
    load_model(model, model_path, name_mapping=_qwen_multimodal_name_mapping)
    return model
