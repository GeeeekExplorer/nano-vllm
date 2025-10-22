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
from nanovllm.utils.secure import get_security_config, orthogonal_matrix
from nanovllm.utils.trace import should_trace, print_tensor, print_line


class Qwen3Attention(nn.Module):

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

        #新增参数
        enable_vector_mask: bool = True,    # 是否启用向量掩码
        mask_scale: float = 0.05, 
        layer_id: int = 0,
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
        self.layer_id = layer_id

        # 软最大（QK^T）加密：使用正交矩阵 R，满足 R^{-1} = R^T，使 Q->Q R^T, K->K R^T，不改变 QK^T
        sec = get_security_config()
        if sec.enable_softmax_encrypt:
            # 使用 float32 存储 R，保持正交性，避免 bf16 破坏 R R^T ≈ I
            R = orthogonal_matrix(self.head_dim, dtype=torch.float32, device=torch.device('cpu'))
        else:
            R = torch.eye(self.head_dim, dtype=torch.float32, device='cpu')
        # 单一 R 即可：Q' = Q R, K' = K R，保证 Q'K'^T = QK^T
        self.encrypt_R = nn.Parameter(R, requires_grad=False)

        # 从安全配置读取线性噪声开关与强度
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            enable_mask=get_security_config().enable_linear_noise,
            mask_scale=get_security_config().noise_scale,
            layer_id=layer_id,
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
        #形成QKV矩阵
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        # 应用正交加密（TEE on CPU -> 先回到 CPU 再加密并发送给 GPU）
        from nanovllm.utils.secure import get_security_config
        sec = get_security_config()
        if sec.enable_softmax_encrypt and sec.encrypt_on_cpu:
            R_cpu = self.encrypt_R.detach().to(device="cpu", dtype=torch.float32)
            q_cpu = q.detach().to(device="cpu", dtype=torch.float32)
            k_cpu = k.detach().to(device="cpu", dtype=torch.float32)
            q_enc_cpu = torch.matmul(q_cpu, R_cpu)
            k_enc_cpu = torch.matmul(k_cpu, R_cpu)
            q_encrypted = q_enc_cpu.to(device=q.device, dtype=q.dtype)
            k_encrypted = k_enc_cpu.to(device=k.device, dtype=k.dtype)
        else:
            R = self.encrypt_R.to(device=q.device, dtype=q.dtype)
            q_encrypted = torch.matmul(q, R)
            k_encrypted = torch.matmul(k, R)

        o = self.attn(q_encrypted, k_encrypted, v)

        # 可视化与分数不变性验证（仅打印一次，且只在 layer_filter 命中时打印）
        from nanovllm.utils.trace import layer_enabled, get_trace_config
        if layer_enabled(self.layer_id) and should_trace(f"Qwen3Attention:{id(self)}") and get_security_config().enable_softmax_encrypt:
            cfg = get_trace_config()
            print_line(f"[TRACE][QK][L{self.layer_id}] 正交加密与分数不变性")
            try:
                # 取一个样本做对比（在 CPU 上计算参考分数）
                q0 = q.detach().to(device="cpu", dtype=torch.float32)
                k0 = k.detach().to(device="cpu", dtype=torch.float32)
                R0 = self.encrypt_R.detach().to(device="cpu", dtype=torch.float32)
                if q0.dim() == 3 and k0.dim() == 3:
                    qh = q0[0]  # [num_heads, head_dim]
                    kh = k0[0]
                    s_ref = torch.matmul(qh, kh.transpose(-1, -2))
                    s_enc = torch.matmul(qh @ R0, (kh @ R0).transpose(-1, -2))
                    abs_err = (s_ref - s_enc).abs().max().item()
                    denom = s_ref.abs().max().item() + 1e-6
                    rel_err = abs_err / denom

                    # RMS 不变性（取一个向量）
                    x = q0[0, 0]
                    xr = (q0[0, 0] @ R0)
                    rms = torch.sqrt((x.pow(2)).mean()).item()
                    rms_r = torch.sqrt((xr.pow(2)).mean()).item()
                    rms_rel = abs(rms_r - rms) / (abs(rms) + 1e-6)

                    if cfg.summary_only:
                        pass_score = abs_err <= 1e-5 or rel_err <= 1e-6
                        pass_rms = rms_rel <= 1e-6
                        print_line(f"[QK][score] PASS={pass_score} abs={abs_err:.2e} rel={rel_err:.2e}")
                        print_line(f"[QK][rms]   PASS={pass_rms} rel={rms_rel:.2e}")
                    else:
                        print_line(f"R {tuple(R0.shape)} | q/k {tuple(qh.shape)} | o {tuple(o.shape)}")
                        print_line(f"score abs={abs_err:.2e} rel={rel_err:.2e} | rms_rel={rms_rel:.2e}")
            except Exception as e:
                print_line(f"分数不变性验证失败: {e}")
        output = self.o_proj(o.flatten(1, -1))
        return output

#mlp部分也需要进行线性噪声加密吗
class Qwen3MLP(nn.Module):

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


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            layer_id=layer_id,
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
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, layer_id=i) for i in range(config.num_hidden_layers)])
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


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
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
