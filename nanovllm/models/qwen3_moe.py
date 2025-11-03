import os
from glob import glob

import torch
from torch import nn
import torch.distributed as dist
from safetensors import safe_open
from transformers import Qwen3MoeConfig

from nanovllm.utils.loader import default_weight_loader
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.fuse_moe import MergedColumnParallelFusedMoeLinear, RowParallelFusedMoeLinear, get_expert_counts_and_idx
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


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
        rope_scaling: tuple | None = None,
        sliding_window: int | None = None,
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
        self.sliding_window = sliding_window

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
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


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


class Qwen3MoeExperts(nn.ModuleList):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        for _ in range(self.num_experts):
            self.append(Qwen3MoeMLP(hidden_size, intermediate_size, hidden_act))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states

class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate = RowParallelLinear(hidden_size, num_experts, bias=False)
        self.experts = Qwen3MoeExperts(num_experts, hidden_size, intermediate_size, hidden_act)
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

    def route_tokens_to_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routing_weights = nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        return selected_experts, routing_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        router_logits = self.gate(hidden_states)
        selected_experts, routing_weights = self.route_tokens_to_experts(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, selected_experts, routing_weights)
        return final_hidden_states


class Qwen3MoeFusedSparseMoeBlock(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_selected = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.hidden_size = hidden_size
        self.moe_intermediate_size = intermediate_size

        self.gate = RowParallelLinear(hidden_size, num_experts, bias=False)
        self.gate_up_proj = MergedColumnParallelFusedMoeLinear(hidden_size, [intermediate_size] * 2, num_experts)
        self.down_proj = RowParallelFusedMoeLinear(intermediate_size, hidden_size, num_experts)
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        M, hidden_dim = hidden_states.shape
        router_logits = self.gate(hidden_states)
        routing_weights = nn.functional.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_selected, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        expanded_hidden = hidden_states.repeat_interleave(self.num_selected, dim=0)
        selected_experts = selected_experts.reshape(-1)
        m_sizes, sort_idx, inv_sort_idx = get_expert_counts_and_idx(
            selected_experts, self.num_experts
        )

        expanded_hidden = expanded_hidden[sort_idx]
        gate_up = self.gate_up_proj(expanded_hidden, m_sizes)
        expert_output = self.down_proj(self.act_fn(gate_up), m_sizes)
        expert_output = expert_output[inv_sort_idx]
        expert_output = expert_output.view(M, self.num_selected, hidden_dim)
        output = (expert_output * routing_weights.unsqueeze(-1)).sum(dim=1)

        return output


# Qwen3MoeSparseMoeBlock
class Qwen3MoeBlock(Qwen3MoeFusedSparseMoeBlock):
    pass


class Qwen3MoeRMSNorm(RMSNorm):
    pass


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            sliding_window=config.sliding_window,
        )
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeBlock(
                num_experts=config.num_experts,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                num_experts_per_tok=config.num_experts_per_tok,
                norm_topk_prob=config.norm_topk_prob,
                hidden_act=config.hidden_act,
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3MoeModel(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_ids,
            )
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

    def __init__(
        self,
        config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.num_experts = config.num_experts
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, position_ids)
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_model(
        self,
        path: str,
    ) -> None:
        for file in glob(os.path.join(path, "*.safetensors")):
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    weight_tensor = f.get_tensor(weight_name)
                    is_expert = "mlp.experts" in weight_name
                    is_loaded = False

                    # Process experts params name
                    if is_expert:
                        mlp_module_name, expert_module_name = weight_name.split(".experts.")
                        expert_idx = int(expert_module_name.split(".")[0])
                        proj_name = expert_module_name.replace(f"{expert_idx}.", "")
                        weight_name = f"{mlp_module_name}.{proj_name}"

                    # Load packed modules
                    for k in self.packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = self.packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            param = self.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            if is_expert:
                                weight_loader(param, weight_tensor, expert_idx, shard_id)
                            else:
                                weight_loader(param, weight_tensor, shard_id)
                            is_loaded = True
                            break

                    # Load other modules
                    if not is_loaded:
                        param = self.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        if is_expert:
                            weight_loader(param, weight_tensor, expert_idx)
                        else:
                            weight_loader(param, weight_tensor)
                        is_loaded = True
                    
                    assert is_loaded, f"Weight {weight_name} not loaded"
