import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

'''
Each transformer decoder layer has two sub-networks:

Input
  │
  ├── (1) Attention  ← this is where Q, K, V projections live
  │
  └── (2) MLP        ← this is where Gate and Up projections live
  │
Output
'''

class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int, # width of representation vector, e.g. 4096, each token is described by 4096 different "features."
        num_heads: int, # number of attention heads, e.g. 16, each head can learn to pay attention to different things (e.g., one head might track grammar, another might track meaning). This is called Multi-Head Attention (MHA).
        # number of kv heads, notice this is smaller than num_heads, which is
        # called Grouped-Query Attention (GQA). Instead of each query head
        # having its own K and V, multiple query heads share the same K and V.
        # Here, every 2 query heads share 1 KV head (16/8 = 2). This saves
        # memory (less KV cache) while barely hurting quality. Q heads still
        # learn different patterns because their queries differ.
        num_kv_heads: int,
        # Within one sequence, the total number of tokens (prompt + generated
        # tokens) cannot exceed 131,072 (4096 * 32). The model was trained with
        # positional encoding table was pre-computed up to 131K positions.
        max_position: int = 4096 * 32,
        head_dim: int | None = None, # dimension of each attention head, 1024 / 16 = 64.
        rms_norm_eps: float = 1e-06, # A tiny number added to avoid division by zero in normalization. Pure numerical stability.
        qkv_bias: bool = False, # Whether the QKV linear layer has a bias term.
        rope_theta: float = 10000, # A base frequency for positional encoding (RoPE). Higher values let the model handle longer sequences.
        rope_scaling: tuple | None = None, # Optional scaling for extending context length beyond training. Not used here.
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
        # 1/sqrt(head_dim) before applying softmax, to prevent large dot product
        # values which can lead to small gradients.
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
        # Positional Encoding (RoPE)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta, # 1000000 for Qwen here.
            rope_scaling=rope_scaling, # None for Qwen here.
        )
        # This Attention class wraps Flash Attention and KV cache. It does the
        # core softmax(Q @ K^T / sqrt(d)) @ V computation, but also handles
        # writing K, V to the paged KV cache and reading from it during decode.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # When there's no bias in QKV, Qwen3 normalizes Q and K before applying
        # RoPE. This stabilizes training by preventing Q and K vectors from
        # growing too large.
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor, # pos ids of each token, e.g. [0, 1, 2, 3] for a 4-token prompt.
        hidden_states: torch.Tensor, # the token representations, shape [total_tokens, 1024]
    ) -> torch.Tensor:
        # Step 1: Project to Q, K, V in one shot.
        qkv = self.qkv_proj(hidden_states)
        # hidden_states: [total_tokens, 1024] → qkv: [total_tokens, 2048]
        #                                                   (1024 + 512 + 512)

        # Step 2: Split into Q, K, V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # q: [total_tokens, 1024]
        # k: [total_tokens, 512]
        # v: [total_tokens, 512]

        # Step 3: Reshape into separate heads
        q = q.view(-1, self.num_heads, self.head_dim)    # [total_tokens, 16, 64]
        k = k.view(-1, self.num_kv_heads, self.head_dim) # [total_tokens, 8, 64]
        v = v.view(-1, self.num_kv_heads, self.head_dim) # [total_tokens, 8, 64]

        # Step 4: Normalize Q and K (if no bias)
        if not self.qkv_bias:
            q = self.q_norm(q) # normalize each 64-dim vector
            k = self.k_norm(k)
        
        # Step 5: Apply positional encoding — rotate Q and K based on position
        q, k = self.rotary_emb(positions, q, k)

        # Step 6: Attention — the core computation
        # Writes K,V to KV cache, then computes softmax(Q @ K^T / scale) @ V
        o = self.attn(q, k, v) # output: [total_tokens, 16, 64]

        # Step 7: Combine heads and project back
        output = self.o_proj(o.flatten(1, -1))
        # o.flatten(1, -1): [total_tokens, 16, 64] → [total_tokens, 1024]
        # o_proj:           [total_tokens, 1024]    → [total_tokens, 1024]
        return output


# Attention lets tokens look at each other (inter-token communication). The MLP
# processes each token independently. Think of attention as "gathering
# information from other tokens" and MLP as "thinking about what was gathered."
class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,       # 1024 - input/output width
        intermediate_size: int, # 2816 - the "expansion" width
        hidden_act: str,        # "silu" - which activation function, e.g. SiLU for Qwen.
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,             # 1024 input width
            [intermediate_size] * 2, # [2816, 2816] - we compute gate and up projections together for efficiency.
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, # 2816 input width
            hidden_size,       # 1024 output width
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x) # [tokens, 1024] → [tokens, 5632] (we compute gate and up proj together for efficiency: 2816 for gate, 2816 for up)
        x = self.act_fn(gate_up)       # split, SiLU(gate) * up → [tokens, 2816]
        x = self.down_proj(x)          # [tokens, 2816] → [tokens, 1024]
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
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


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# Causal means the model can only look left (backward) when predicting the next token
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        # Attention Layer: input → q_proj, k_proj, v_proj → attention mechanism → o_proj → output
        "q_proj": ("qkv_proj", "q"), # q_proj → load into qkv_proj, shard_id="q", same applies to below.
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        # MLP Layer: input → gate_proj, up_proj → SiLU gating → down_proj → output
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
