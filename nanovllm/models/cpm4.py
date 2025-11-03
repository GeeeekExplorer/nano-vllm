import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
import math
from functools import lru_cache


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    
    This is equivalent to the MiniCPM modeling implementation.
    """
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)


class MiniCPMLongRoPE(nn.Module):
    """MiniCPM LongRoPE implementation equivalent to modeling_minicpm.py"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        short_factor=None,
        long_factor=None,
        original_max_position_embeddings=None,
    ) -> None:
        super().__init__()
        self.dim = head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.short_factor = short_factor or [1.0] * (head_size // 2)
        self.long_factor = long_factor or [1.0] * (head_size // 2)
        self.original_max_position_embeddings = original_max_position_embeddings or max_position_embeddings
        
        # Calculate scaling factor (kept for compatibility, but not used to scale cos/sin amplitude)
        scale = (max_position_embeddings / self.original_max_position_embeddings)
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        
        # Create base inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Pre-compute cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device, torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=device)

        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype)
        )
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        # Do NOT scale cos/sin amplitude; only frequency is scaled by ext_factors
        self.register_buffer('cos_cached', emb.cos().to(dtype)* self.scaling_factor, persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype)* self.scaling_factor, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)
        max_pos = positions.max().item()
        
        # Rebuild cache if needed
        if max_pos >= self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_pos + 1, device=query.device, dtype=query.dtype)
        
        # Get cos/sin for the positions
        cos = self.cos_cached[positions]  # [num_tokens, head_dim]
        sin = self.sin_cached[positions]  # [num_tokens, head_dim]
        
        # Apply rotary embedding using the original nano-vllm method but with corrected math
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.dim)
        query = self._apply_rotary_emb(query, cos, sin).view(query_shape)
        
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.dim)
        key = self._apply_rotary_emb(key, cos, sin).view(key_shape)
        
        return query, key
    
    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding with corrected math matching modeling_minicpm.py"""
        # x: [num_tokens, num_heads, head_dim]
        # cos/sin: [num_tokens, head_dim] from _set_cos_sin_cache (already repeated)
        
        cos = cos.unsqueeze(1)  # [num_tokens, 1, head_dim] to broadcast over heads
        sin = sin.unsqueeze(1)  # [num_tokens, 1, head_dim] to broadcast over heads
        
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
        
        # Apply standard RoPE: (x * cos) + (rotate_half(x) * sin)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        rotate_half_x = torch.cat((-x2, x1), dim=-1)
        
        result = x * cos + rotate_half_x * sin
        return result.to(orig_dtype)


def get_cpm4_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """Get CPM4 LongRoPE implementation"""
    rotary_emb = MiniCPMLongRoPE(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
        short_factor=rope_scaling.get('short_factor') if rope_scaling else None,
        long_factor=rope_scaling.get('long_factor') if rope_scaling else None,
        original_max_position_embeddings=rope_scaling.get('original_max_position_embeddings') if rope_scaling else None,
    )
    return rotary_emb


class Cpm4Attention(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        apply_qk_norm: bool = False,
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
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position = max_position
        self.apply_qk_norm = apply_qk_norm
        

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
            bias=qkv_bias,
        )
        self.rotary_emb = get_cpm4_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if self.apply_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Apply Q/K normalization only if enabled
        if self.q_norm is not None:
            q_by_head = q.view(-1, self.num_heads, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            
            k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
        
        # Apply rotary embedding using nano-vllm interface
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Cpm4MLP(nn.Module):

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


class Cpm4DecoderLayer(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.self_attn = Cpm4Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            apply_qk_norm=getattr(config, 'apply_qk_norm', False),
        )
        self.mlp = Cpm4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # depth scaling like MiniCPM
        self.scale_depth = getattr(config, 'scale_depth', 1.0)
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # PreNorm + Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(positions, hidden_states)
        scale = self.scale_depth / math.sqrt(self.num_hidden_layers)
        hidden_states = residual + attn_out * scale

        # PreNorm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out * scale
        return hidden_states, residual


class Cpm4Model(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Cpm4DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # embedding scale from config (MiniCPM uses scale_emb)
        self.embed_scale = getattr(config, 'scale_emb', 1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # final RMSNorm without residual add to match MiniCPM
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Cpm4ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Cpm4Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # width scaling before logits: hidden_size/dim_model_base
        scale_width = self.config.hidden_size / max(1, getattr(self.config, 'dim_model_base', self.config.hidden_size))
        logits = self.lm_head(hidden_states / scale_width)
        return logits
