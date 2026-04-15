import torch
import torch.nn as nn
from transformers import AutoConfig

from nanovllm.layers.activation import Silu
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import WordEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import RotaryEmbedding


class Qwen2_5vlMLP(nn.Module):

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, gather_output=False)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = Silu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2_5vlAttention(nn.Module):

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_proj = ColumnParallelLinear(self.hidden_size, self.num_heads * self.head_dim, bias=True, gather_output=False)
        self.k_proj = ColumnParallelLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True, gather_output=False)
        self.v_proj = ColumnParallelLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True, gather_output=False)
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, self.max_position_embeddings, self.rope_theta)
        self.attn = Attention(self.num_heads, self.head_dim, config.sliding_window, self.num_kv_heads)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, k = self.rotary_emb(q, k, positions)
        x = self.attn(q, k, v)
        x = self.o_proj(x)
        return x


class Qwen2_5vlDecoderLayer(nn.Module):

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen2_5vlAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen2_5vlMLP(config)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Qwen2_5vlModel(nn.Module):

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.embed_tokens = WordEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2_5vlDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x, positions)
        x = self.norm(x)
        return x


class Qwen2_5vlForCausalLM(nn.Module):

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.model = Qwen2_5vlModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.model(x, positions)
        return x

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(x)