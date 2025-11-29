import torch
from torch import nn
import torch.distributed as dist # dist 的方法（如 get_rank()）要求在使用前初始化分布式进程组
from transformers import Qwen3Config

"""
通过算子封装成模块，通过模块封装成层，通过层封装成模型
"""
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module): # 封装模型的自注意力 + 投影、并行逻辑

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
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5 # attention 缩放系数，通常为 1/sqrt(head_dim)
        self.qkv_bias = qkv_bias # 保存 qkv 是否带偏置

        self.qkv_proj = QKVParallelLinear( # 把 hidden_size 投影为合并的 q、k、v
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear( # 把拼接回的多头输出投影回 hidden_size
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
        if not self.qkv_bias: # 大多模型（如 Llama）直接拿 QK 去算 RoPE 和 Attention，但 Qwen 为了训练稳定性，在 RoPE 之前对 Q 和 K 进行了 RMSNorm；区分 Qwen 和其他模型的重要特征
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states) # 对输入做 qkv 并行投影，返回一个把 q、k、v 合并在一起的张量
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim) # 从四维变三维
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias: # 如果 qkv 投影没有偏置，则应用QNorm、KNorm
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1)) # 先把多头输出扁平化为最后一维拼接（flatten 从维度 1 到最后），然后通过 o_proj 投影回 hidden_size；o.flatten(1, -1) 意味着把除第一维外的维度都连成一个向量
        return output


class Qwen3MLP(nn.Module): # 其MLP就是TP的做法，先按列并行作第一层linear，再激活，再按行并行作第二层linear，最后GPU结果累加即可

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear( # 使用 MergedColumnParallelLinear，因为 SwiGLU 需要 Wg⋅x 和 Wu⋅x，这两个矩阵被合并成一个大矩阵进行并行计算，减少了 CUDA kernel 的调用次数
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
        gate_up = self.gate_up_proj(x) # 列并行TP
        x = self.act_fn(gate_up) # 激活
        x = self.down_proj(x) # 行并行TP
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention( # 根据 config 中的字段创建 Qwen3Attention
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
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 定义层内的输入 RMSNorm（通常在 attention 之前）
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 定义 attention 后的 RMSNorm（用于 residual/后续 MLP）

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


class Qwen3Model(nn.Module): # 顶层模型模块，包含 embedding、若干层以及最终 layer norm

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size) # 词表并行
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
        hidden_states, _ = self.norm(hidden_states, residual) # 全部层执行完之后，将最终 hidden_states 与 residual 一并输入最终 RMSNorm，输出的residual也没用了
        return hidden_states


class Qwen3ForCausalLM(nn.Module): # 封装模型 + lm_head（模型输出的是隐藏层数据，可以封装成Qwen3ForAnything）
    packed_modules_mapping = { # 类级别字典 packed_modules_mapping：用于说明 加载器中加载的权重名（如 q_proj/k_proj/v_proj）如何映射到 该代码的模块名（如 qkv_proj）以及对应的子分片键（"q"/"k"/"v" 或索引 0/1），便于从权重文件中恢复合并的/打包的参数
        "q_proj": ("qkv_proj", "q"), # 对应loader中的v，shard_id
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
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
        if config.tie_word_embeddings: # 如果配置要求将词向量与输出权重共享（tie embeddings），就将 lm_head 的权重直接指向/复制为 embed_tokens 的权重数据，从而实现权重共享；简单说就是tokenid和向量的表就只存一张就行了，反过来查也是查
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits( # 分出来logits计算是为了避免在 forward 里强制计算所有 token 的 logits，从而允许由用户（或推理引擎）手动控制何时计算 logits，以实现极致的性能优化
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
