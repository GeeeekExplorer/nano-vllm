"""Multi-head Latent Attention (MLA) 实现。

MLA 是 DeepSeek-V2 提出的注意力机制，核心思想：
  - 不存储完整的 K、V（每头 head_dim 维），而是压缩为低秩隐向量 kv_c（kv_lora_rank 维）
  - KV Cache 只存 (kv_c, k_pe)，相比标准 MHA 大幅节省显存
  - 计算 Attention 时再通过 kv_b_proj 将 kv_c 展开为完整的 K_nope, V

在昇腾 910C 上进一步优化：
  - 使用融合算子 npu_kv_rmsnorm_rope_cache 将 RMSNorm + RoPE + Cache 写入一步完成
  - CPU Offload 时对压缩后的 kv_c / k_pe 做 scatter+burst，数据量远小于完整 KV

对应 vllm-ascend PR #1659 中的 mla_v1.py。
"""

import torch
from torch import nn

from nanovllm.utils.context import get_context
from nanovllm.layers.attention_utils import (
    wait_for_kv_layer_from_connector,
    maybe_save_kv_layer_to_connector,
)


class MLAAttention(nn.Module):
    """Multi-head Latent Attention 模块。

    KV 压缩流程：
      hidden → kv_a_proj_with_mqa → [kv_c (kv_lora_rank), k_pe (qk_rope_head_dim)]
      kv_c → kv_a_layernorm → kv_b_proj → [k_nope (num_heads * qk_nope_head_dim), v (num_heads * v_head_dim)]

    Query 压缩流程：
      hidden → q_a_proj → q_c (q_lora_rank)
      q_c → q_a_layernorm → q_b_proj → [q_nope (num_heads * qk_nope_head_dim), q_pe (num_heads * qk_rope_head_dim)]

    KV Cache 只存 [kv_c_cache, k_pe_cache]，不存完整 K/V。

    Args:
        hidden_size:       模型隐藏层维度
        num_heads:         注意力头数
        qk_nope_head_dim:  每头 query/key 的非 RoPE 部分维度
        qk_rope_head_dim:  每头 query/key 的 RoPE 部分维度
        v_head_dim:        每头 value 的维度
        q_lora_rank:       query 低秩压缩的秩
        kv_lora_rank:      KV 低秩压缩的秩
        rope_theta:        RoPE 的 theta 基数
        layer_name:        层名称标识，用于 CPU Offload 按层管理
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000.0,
        layer_name: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.layer_name = layer_name

        self.scaling = self.qk_head_dim ** -0.5

        # --- 以下权重矩阵在实际实现中为 nn.Linear / 并行 Linear ---
        # q_a_proj:            hidden_size → q_lora_rank        （查询压缩）
        # q_a_layernorm:       RMSNorm(q_lora_rank)
        # q_b_proj:            q_lora_rank → num_heads * qk_head_dim （查询展开）
        # kv_a_proj_with_mqa:  hidden_size → kv_lora_rank + qk_rope_head_dim （KV压缩+位置）
        # kv_a_layernorm:      RMSNorm(kv_lora_rank)
        # kv_b_proj:           kv_lora_rank → num_heads * (qk_nope_head_dim + v_head_dim) （KV展开）
        # o_proj:              num_heads * v_head_dim → hidden_size （输出投影）

        # KV Cache：压缩表示，比标准 MHA 小得多
        # kv_c_cache: [num_blocks, block_size, kv_lora_rank]
        # k_pe_cache: [num_blocks, block_size, qk_rope_head_dim]
        self.kv_cache: list[torch.Tensor] = []  # [kv_c_cache, k_pe_cache]

    def _mla_preprocess(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        slot_mapping: torch.Tensor,
        is_prefill: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """MLA 预处理：压缩投影 + RMSNorm + RoPE + KV Cache 写入。

        对应 vllm-ascend 中 AscendMLAImpl._mla_preprocess()，
        PR #1659 将 layer_name 加入签名以支持按层 CPU Offload。

        Args:
            layer_name:    当前层名称，用于 CPU Offload 按层定位
            hidden_states: 输入隐藏状态，shape = [num_tokens, hidden_size]
            slot_mapping:  每个 token 在 KV Cache 中的 slot 位置
            is_prefill:    是否为 prefill 阶段

        Returns:
            q_nope:  查询非 RoPE 部分，shape = [num_tokens, num_heads, qk_nope_head_dim]
            q_pe:    查询 RoPE 部分，shape = [num_tokens, num_heads, qk_rope_head_dim]
            k_nope:  键非 RoPE 部分，shape = [num_tokens, num_heads, qk_nope_head_dim]（prefill）
                     或从 kv_c_cache 中重建（decode）
            k_pe:    键 RoPE 部分，shape = [num_tokens, 1, qk_rope_head_dim]
            v:       值，shape = [num_tokens, num_heads, v_head_dim]（仅 prefill 返回）
                     decode 时为 None，由 attention kernel 内部从 cache 重建
        """
        # --- CPU Offload: 计算前先尝试从 CPU 换入该层的压缩 KV Cache ---
        wait_for_kv_layer_from_connector(layer_name, *self.kv_cache)

        # --- 执行逻辑 ---
        # 1. Query 压缩路径：
        #    q_c = q_a_proj(hidden_states)              # [num_tokens, q_lora_rank]
        #    q_c = q_a_layernorm(q_c)                   # RMSNorm 归一化
        #    q = q_b_proj(q_c)                          # [num_tokens, num_heads * qk_head_dim]
        #    q = q.view(num_tokens, num_heads, qk_head_dim)
        #    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
        #    对 q_pe 施加 RoPE 旋转
        #
        # 2. KV 压缩路径：
        #    kv_no_split = kv_a_proj_with_mqa(hidden_states)  # [num_tokens, kv_lora_rank + qk_rope_head_dim]
        #
        # 3. 在昇腾 910C 上，使用融合算子一步完成 RMSNorm + RoPE + Cache 写入：
        #    kv_c_normed, k_pe_roped = npu_kv_rmsnorm_rope_cache(
        #        kv_no_split, cos, sin, self.kv_cache, slot_mapping,
        #        kv_lora_rank, qk_rope_head_dim)
        #    （该融合算子在 AI Core 上执行，避免中间张量的 HBM 读写）
        #
        # 4. Prefill 分支：
        #    k_nope, v = kv_b_proj(kv_c_normed) 展开为完整 K_nope 和 V
        #    返回 (q_nope, q_pe, k_nope, k_pe_roped, v)
        #
        # 5. Decode 分支：
        #    k_nope 和 v 不在此处展开——由 decode attention kernel
        #    在读取 kv_c_cache 时通过 absorbed kv_b_proj 权重直接计算
        #    返回 (q_nope, q_pe, None, k_pe_roped, None)
        pass

    def forward(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """MLA forward：预处理 → Attention 计算 → 输出投影。

        对应 vllm-ascend 中 AscendMLAImpl.forward()，
        PR #1659 增加 layer_name 参数以支持 CPU Offload。

        Args:
            layer_name:    当前层名称
            hidden_states: 输入隐藏状态，shape = [num_tokens, hidden_size]

        Returns:
            output: 注意力输出，shape = [num_tokens, hidden_size]
        """
        context = get_context()

        # --- 执行逻辑 ---
        # 1. 调用 _mla_preprocess 获取 q_nope, q_pe, k_nope, k_pe, v
        #    q_nope, q_pe, k_nope, k_pe, v = self._mla_preprocess(
        #        layer_name, hidden_states, context.slot_mapping, context.is_prefill)
        #
        # 2. Prefill Attention（使用 flash_attn_varlen_func 或等效昇腾算子）：
        #    a. 拼接 q = concat(q_nope, q_pe) → [num_tokens, num_heads, qk_head_dim]
        #    b. 拼接 k = concat(k_nope, k_pe.expand(num_heads)) → [num_tokens, num_heads, qk_head_dim]
        #    c. o = flash_attn_varlen(q, k, v, ..., causal=True)
        #
        # 3. Decode Attention（使用 flash_attn_with_kvcache 的 MLA 变体）：
        #    a. 拼接 q = concat(q_nope, q_pe)
        #    b. attention kernel 内部直接从 kv_c_cache 和 k_pe_cache 读取，
        #       通过 absorbed kv_b_proj 权重在 on-the-fly 展开 k_nope 和 v
        #       （避免在 HBM 中存储展开后的完整 K/V，节省带宽）
        #    c. o = mla_decode_attention(q, kv_c_cache, k_pe_cache, kv_b_proj_weight, ...)
        #
        # 4. CPU Offload: prefill 后将该层压缩 KV Cache 卸载到 CPU
        #    maybe_save_kv_layer_to_connector(layer_name, *self.kv_cache)
        #
        # 5. 输出投影：
        #    output = o_proj(o.flatten(-2, -1))   # [num_tokens, hidden_size]
        #
        # 6. 返回 output
        pass
