import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3MoeConfig
from torch.nn import functional as F
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear, ReplicatedLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.fused_moe.shared_fused_moe import SharedFusedMoE

class Qwen3MoeMLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,    # input_size
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


class Qwen3MoeSparseMoeBlockOld(nn.Module):
    def __init__(
            self,
            config:  Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.n_routed_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        # 最简化的方式(测试)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config.hidden_size, config.moe_intermediate_size, config.hidden_act) for _ in range(self.n_routed_experts)]
        )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
        )

    def forward(self, hidden_states) -> torch.Tensor:
        # 形状: [batch*seq, hidden_size]
        batch_seq, hidden = hidden_states.shape
        router_logits = self.gate(hidden_states)  # [batch_seq, num_experts]

        # Softmax + top-k 选择（核心路由）
        routing_weights = F.softmax(router_logits, dim=-1)  # [batch_seq, num_experts]
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k,
                                                       dim=-1)  # weights [batch_seq, top_k], experts [batch_seq, top_k]
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)  # 归一化

        # 初始化输出
        final_hidden_states = torch.zeros_like(hidden_states)

        # 逐 token 计算（简化循环；实际用 scatter/gather 优化）
        for expert_idx in range(self.n_routed_experts):
            mask = (selected_experts == expert_idx).any(dim=-1)  # [batch_seq]
            if mask.sum() == 0:
                continue  # 跳过未选专家
            local_hidden = hidden_states[mask]  # 选中的 token
            expert_out = self.experts[expert_idx](local_hidden)  # 计算
            # 加权回原位置（简化）
            per_token_expert_mask = (selected_experts[mask] == expert_idx)  # [num_masked, top_k] bool
            per_token_weights = (routing_weights[mask] * per_token_expert_mask.float()).sum(dim=-1)  # [num_masked]
            final_hidden_states[mask] += per_token_weights.unsqueeze(-1) * expert_out

        return final_hidden_states

class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
            self,
            config:  Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.n_routed_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
        )
        
        self.experts = SharedFusedMoE(
            num_experts=self.n_routed_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
        )
        

    def forward(self, hidden_states) -> torch.Tensor:
        assert hidden_states.dim() == 2, (
            "Qwen3MoeSparseMoeBlock expects input hidden_states to have shape [batch*seq, hidden_size]"
        )
        num_tokens, hidden_dim = hidden_states.shape
        router_logits = self.gate(hidden_states)    # [num_tokens, num_experts]
        fused_out = self.experts(hidden_states, router_logits)  # [num_tokens, hidden_size]
        return fused_out
    
class Qwen3MoeAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,               # embedding后的词维度
            num_heads: int,
            num_kv_heads: int,
            max_position: int = 4096 * 32,
            head_dim: int | None = None,    # 每个注意力头的维度(hidden_size / num_head)：通过线性变换降维，而非简单截取输入矩阵X
            rms_norm_eps: float = 1e-06,
            qkv_bias: bool = False,
            rope_theta: float = 10000,
            rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        
        # assert self.total_num_kv_heads % tp_size == 0
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.rope_theta = rope_theta
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
            bias = False
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
        """
        打印Attention的forward的维度变换————decoder阶段
        hidden_states： torch.Size([1, 1024])   # 当batch_size = 1且seq_len = 1时，维度会变为(b * s, hidden_size)，推理过程(decode)，seq_len为1，而在prefill时，seq_len不为1
        qkv： torch.Size([1, 4096])             # (batch_size, seq_len, q_size + k_size + v_size)
        q.shape： torch.Size([1, 16, 128])      # (batch_size * seq_len, num_heads, head_dim)
        k.shape torch.Size([1, 8, 128])         # (batch_size * seq_len, num_kv_heads, head_dim)
        v.shape： torch.Size([1, 8, 128])       # (batch_size * seq_len, num_kv_heads, head_dim)
        o.shape： torch.Size([1, 1, 16, 128])   # (batch_size, seq_len, num_heads, head_dim)
        output.shape： torch.Size([1, 1024])    # (batch_size, seq_len * num_heads * head_dim)
        """
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

class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
            self,
            config:  Qwen3MoeConfig,
            mlp_kind: int = 0
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
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

        if mlp_kind == 0:
            self.mlp = Qwen3MoeSparseMoeBlock(config=config)
        else:
            self.mlp = Qwen3MoeMLP(config.hidden_size, config.moe_intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None,      # 残差
    ):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

class Qwen3MoeModel(nn.Module):
    def __init__(
            self,
            config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # self.layers = nn.ModuleList([ Qwen3MoeDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layers = nn.ModuleList()
        for layer_id in range(config.num_hidden_layers):
            if config.mlp_only_layers and layer_id in config.mlp_only_layers:
                self.layers.append(Qwen3MoeDecoderLayer(config, 1))
            else:
                if (layer_id + 1) % config.decoder_sparse_step == 0:
                    self.layers.append(Qwen3MoeDecoderLayer(config, 0))
                else:
                    self.layers.append(Qwen3MoeDecoderLayer(config, 1))
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
            config:  Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)
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
