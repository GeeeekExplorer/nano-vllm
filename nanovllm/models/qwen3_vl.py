"""Simplified implementation of the Qwen3-VL multimodal model.

This module inlines both the text backbone and the vision encoder to minimise
changes to other components.
"""

from __future__ import annotations

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
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope


# ---------------------------------------------------------------------------
# Text backbone (supports DeepStack injection)
# ---------------------------------------------------------------------------


class Qwen3VLTextAttention(nn.Module):

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
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3VLTextMLP(nn.Module):

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
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3VLTextDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_scaling = None

        self.self_attn = Qwen3VLTextAttention(
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
        self.mlp = Qwen3VLTextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3VLTextModel(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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

        residual = None
        if visual_pos_mask is not None:
            visual_pos_mask = visual_pos_mask.to(hidden_states.device)
        mask_tensor = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if (
                deepstack_visual_embeds is not None
                and vision_token_count is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):
                ds = deepstack_visual_embeds[layer_idx].to(
                    hidden_states.device, hidden_states.dtype
                )
                hidden_states = hidden_states.clone()
                if visual_pos_mask is not None:
                    if mask_tensor is None:
                        mask_tensor = visual_pos_mask.bool()
                    if mask_tensor.sum().item() != ds.size(0):
                        raise ValueError("DeepStack features do not match the visual mask length")
                    hidden_states[mask_tensor] += ds
                else:
                    hidden_states[:vision_token_count] += ds
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3VLTextForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen3VLTextModel(config)
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
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# Vision encoder (vision tower + DeepStack features)
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


class Qwen3VLVisionPatchEmbed(nn.Module):
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


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLVisionMLP(nn.Module):
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


class Qwen3VLVisionAttention(nn.Module):
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

            q = q.view(length, self.num_heads, self.head_dim).transpose(0, 1)
            k = k.view(length, self.num_heads, self.head_dim).transpose(0, 1)
            v = v.view(length, self.num_heads, self.head_dim).transpose(0, 1)

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


class Qwen3VLVisionPatchMerger(nn.Module):
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
            hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_fc2(hidden_states)
        return hidden_states


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = Qwen3VLVisionMLP(config)

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


class Qwen3VLVisionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.hidden_size = config.hidden_size
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

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

        deepstack_feature_lists: List[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, seq_lengths, position_embeddings)
            if layer_idx in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_idx)]
                deepstack_feature = merger(hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists


class Qwen3VisionEncoder(nn.Module):
    def __init__(self, vision_config) -> None:
        super().__init__()
        self.config = vision_config
        self.vision = Qwen3VLVisionModel(vision_config)

    def _linear_patch_embed(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        proj = self.vision.patch_embed.proj
        weight = proj.weight.view(proj.out_channels, -1)
        bias = proj.bias
        return torch.nn.functional.linear(patch_tokens, weight, bias)

    def _run_vision_from_tokens(
        self,
        token_list: list[torch.Tensor],
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
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

        deepstack_feature_lists: List[torch.Tensor] = []
        for layer_idx, block in enumerate(self.vision.blocks):
            hidden_states = block(hidden_states, seq_lengths, position_embeddings)
            if layer_idx in self.vision.deepstack_visual_indexes:
                merger = self.vision.deepstack_merger_list[
                    self.vision.deepstack_visual_indexes.index(layer_idx)
                ]
                deepstack_feature = merger(hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.vision.merger(hidden_states)

        split_sizes = (
            grids.prod(-1) // (self.config.spatial_merge_size**2)
        ).tolist()

        image_chunks = list(torch.split(hidden_states, split_sizes))

        deepstack_layers: List[List[torch.Tensor]] = []
        for feature in deepstack_feature_lists:
            per_batch = torch.split(feature, split_sizes)
            deepstack_layers.append(list(per_batch))

        return image_chunks, deepstack_layers

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
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
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

            return self._run_vision_from_tokens(token_list, grids)

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

        image_embeds, deepstack = self.vision(pixel_values, image_grid_thw)
        split_sizes = (
            image_grid_thw.prod(-1) // (self.config.spatial_merge_size**2)
        ).tolist()

        image_chunks = torch.split(image_embeds, split_sizes)
        image_tokens = torch.stack(list(image_chunks), dim=0)

        deepstack_layers: List[torch.Tensor] = []
        for feature in deepstack:
            per_batch = torch.split(feature, split_sizes)
            stacked = torch.stack(list(per_batch), dim=0)
            deepstack_layers.append(stacked)

        return image_tokens, deepstack_layers


def create_vision_model(config, **kwargs) -> nn.Module:
    del kwargs
    return Qwen3VisionEncoder(config)


def get_vision_model(config, **kwargs) -> nn.Module:
    return create_vision_model(config, **kwargs)


# ---------------------------------------------------------------------------
# Multimodal wrapper
# ---------------------------------------------------------------------------


class Qwen3VLForConditionalGeneration(nn.Module):
    """Qwen3-VL conditional generation model (language + vision)."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_config = getattr(config, "text_config", config)
        self.vision_config = getattr(config, "vision_config", None)

        if self.vision_config is None:
            raise ValueError("vision_config is missing; cannot build a multimodal model")

        self.visual = create_vision_model(self.vision_config)
        self.language_model = Qwen3VLTextForCausalLM(self.text_config)

        print("[Qwen3VLForConditionalGeneration] Initialization complete")
        print(f"  - Vision encoder: {type(self.visual).__name__}")
        print(f"  - Language model: {type(self.language_model).__name__}")
    
        self.packed_modules_mapping = {
            "mlp.gate_proj": ("mlp.gate_up_proj", 0),
            "mlp.up_proj": ("mlp.gate_up_proj", 1),
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
        }
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.model.embed_tokens(input_ids)
    
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        sequence_lengths: list[int] | None = None,
        vision_slices_per_seq: list[list[dict]] | None = None,
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
        deepstack_layers: list[torch.Tensor] | None = None
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

            deepstack_collect: list[list[torch.Tensor]] | None = None

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

                    deepstack_slice = slice_info.get("deepstack")
                    if deepstack_slice:
                        if deepstack_collect is None:
                            deepstack_collect = [[] for _ in range(len(deepstack_slice))]
                        for layer_idx, layer_tokens in enumerate(deepstack_slice):
                            if layer_tokens is None:
                                continue
                            deepstack_collect[layer_idx].append(
                                layer_tokens.to(
                                    device=inputs_embeds.device,
                                    dtype=inputs_embeds.dtype,
                                )
                            )

            if deepstack_collect is not None:
                hidden_dim = inputs_embeds.size(-1)
                deepstack_layers = [
                    torch.cat(layer_slices, dim=0)
                    if layer_slices
                    else torch.empty(0, hidden_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    for layer_slices in deepstack_collect
                ]
        elif pixel_values is not None:
            # Fallback path: process raw images when slices are not provided (legacy compatibility)
            if input_ids is None:
                raise ValueError("input_ids are required to locate visual placeholders")
            if sequence_lengths is None:
                raise ValueError("sequence_lengths are required to align visual features")
            if sum(sequence_lengths) != total_tokens:
                raise ValueError("sum of sequence_lengths does not match total input tokens")

            image_chunks, deepstack_layers_raw = self.visual(pixel_values, image_grid_thw)
            if not image_chunks:
                raise ValueError("The vision encoder did not return valid image features")

            offsets = [0]
            for length in sequence_lengths:
                offsets.append(offsets[-1] + length)

            total_replaced = 0
            deepstack_collect = None

            image_iter = iter(image_chunks)
            deepstack_iter = [iter(layer) for layer in deepstack_layers_raw] if deepstack_layers_raw else None

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

                if deepstack_iter:
                    if deepstack_collect is None:
                        deepstack_collect = [[] for _ in deepstack_layers_raw]
                    for layer_idx, iterator in enumerate(deepstack_iter):
                        try:
                            layer_slice = next(iterator).to(inputs_embeds.device, inputs_embeds.dtype)
                        except StopIteration:
                            continue
                        deepstack_collect[layer_idx].append(layer_slice)

            if deepstack_collect is not None:
                hidden_dim = inputs_embeds.size(-1)
                deepstack_layers = [
                    torch.cat(layer_slices, dim=0)
                    if layer_slices
                    else torch.empty(0, hidden_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    for layer_slices in deepstack_collect
                ]
            vision_token_count = total_replaced

        if vision_token_count == 0:
            visual_pos_mask = None
            deepstack_layers = None

        if positions is None:
            positions = torch.arange(
                inputs_embeds.size(0), device=inputs_embeds.device
            )

        if visual_pos_mask is not None and vision_token_count:
            visual_pos_mask = visual_pos_mask.to(inputs_embeds.device)
        else:
            visual_pos_mask = None
            deepstack_layers = None

        hidden_states = self.language_model(
            input_ids=None,
            positions=positions,
            inputs_embeds=inputs_embeds,
            vision_token_count=vision_token_count,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_layers,
        )

        return hidden_states
    
    def compute_logits(self, hidden_states):
        """Compute logits (delegate to language model)"""
        return self.language_model.compute_logits(hidden_states)


def load_qwen3_vl_model(model_path, config):
    """
    Load Qwen3-VL model
    
    Args:
        model_path: Model path
        config: Configuration object
    
    Returns:
        model: Qwen3VLForConditionalGeneration instance
    """
    hf_config = config.hf_config
    
    # Create model
    model = Qwen3VLForConditionalGeneration(hf_config)
    
    from nanovllm.utils.loader import load_model

    print("[load_qwen3_vl_model] Loading Qwen3-VL weights...")

    def name_mapping(weight_name: str) -> str | None:
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

    load_model(model, model_path, name_mapping=name_mapping)
    return model
