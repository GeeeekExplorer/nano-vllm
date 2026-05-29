import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3MoeConfig
import torch.nn.functional as F

from nanovllm.layers import (
    SiluAndMul,
    Attention,
    RMSNorm,
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    get_rope,
    VocabParallelEmbedding,
    ParallelLMHead,
    shard_slice,
)
from nanovllm.kernels import fused_experts

class Qwen3MoeAttention(nn.Module):

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
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        # GQA + TP: kv_heads 能被 tp 整除时正常切分；tp 大于 kv_heads 时复制 kv head
        assert (self.total_num_kv_heads % tp_size == 0) or (tp_size % self.total_num_kv_heads == 0)
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
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
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
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


# MOE MLP
class Qwen3MoeMLP(nn.Module):

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

class Qwen3MoeTopKRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
    ) -> None:
        super().__init__()
        self.top_k = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.hidden_dim = hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        routing_weights = routing_weights.to(hidden_states.dtype)
        return routing_weights, selected_experts


class Qwen3MoeExperts(nn.Module):
    """所有 expert 的权重 stack 成 [E, ...]，配合可捕获的 fused grouped-GEMM。

    TP 切分方式与逐 expert 的 MLP 一致：
      - gate/up: 在 intermediate 维做 column-parallel（每个 rank 持有 I_local = I//tp）
      - down:    在 intermediate 维做 row-parallel（输出需在外层 all_reduce）
    权重布局：
      - w13: [E, 2*I_local, H]  前 I_local 行是 gate，后 I_local 行是 up
      - w2:  [E, H, I_local]
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        tp = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        assert intermediate_size % tp == 0
        I_local = intermediate_size // tp
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = I_local
        self.tp_rank = rank
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * I_local, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, I_local))
        self.w13.weight_loader = self.weight_loader
        self.w2.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight, expert_id: int, shard_id: str):
        I = self.intermediate_size_per_partition
        r = self.tp_rank
        if shard_id == "gate":
            param.data[expert_id, 0:I, :].copy_(shard_slice(loaded_weight, 0, r * I, I))
        elif shard_id == "up":
            param.data[expert_id, I:2 * I, :].copy_(shard_slice(loaded_weight, 0, r * I, I))
        elif shard_id == "down":
            param.data[expert_id, :, :].copy_(shard_slice(loaded_weight, 1, r * I, I))
        else:
            raise ValueError(f"unknown expert shard_id: {shard_id}")

    def forward(self, hidden_states, routing_weights, selected_experts):
        return fused_experts(hidden_states, self.w13, self.w2, routing_weights, selected_experts)


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self, 
        config:Qwen3MoeConfig
    )-> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        self.gate = Qwen3MoeTopKRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            norm_topk_prob=self.norm_topk_prob,
        )
        self.experts = Qwen3MoeExperts(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_dim,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        routing_weights, selected_experts = self.gate(hidden_states)
        out = self.experts(hidden_states, routing_weights, selected_experts)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(out)
        return out.view(orig_shape)


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttention(
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
        if (not layer_idx in config.mlp_only_layers) and (
            (layer_idx + 1) % config.decoder_sparse_step) == 0:
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = hidden_states + residual
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states

class Qwen3MoeModel(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    # MoE 专家权重单独处理：ckpt 里的 mlp.experts.{e}.{proj}.weight
    # -> stacked 参数 mlp.experts.{stack_param}，并带上 expert_id 与 shard_id。
    expert_modules_mapping = {
        "gate_proj": ("w13", "gate"),
        "up_proj": ("w13", "up"),
        "down_proj": ("w2", "down"),
    }

    def __init__(
        self,
        config: Qwen3MoeConfig
    ) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.num_experts = config.num_experts

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
