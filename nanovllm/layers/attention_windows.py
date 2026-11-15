import torch
from torch import nn
from torch.nn import functional as F

from nanovllm.utils.context import get_context


class Attention(nn.Module):
    """Single-GPU Windows fallback for packed prefill and paged KV-cache decode."""

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def _cache_tokens(self, cache, block_table, length):
        block_size = cache.size(1)
        positions = torch.arange(length, device=cache.device)
        slots = block_table[positions // block_size].long() * block_size + positions % block_size
        return cache.view(-1, self.num_kv_heads, self.head_dim)[slots]

    def _attend(self, q, k, v, offset):
        q_len, k_len = q.size(0), k.size(0)
        query_positions = torch.arange(offset, offset + q_len, device=q.device)
        key_positions = torch.arange(k_len, device=q.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=causal_mask,
            scale=self.scale,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return output.squeeze(0).transpose(0, 1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        if self.k_cache.numel() and self.v_cache.numel():
            valid = context.slot_mapping >= 0
            slots = context.slot_mapping[valid].long()
            self.k_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(0, slots, k[valid])
            self.v_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(0, slots, v[valid])

        outputs = []
        if context.is_prefill:
            q_offsets = context.cu_seqlens_q.tolist()
            k_offsets = context.cu_seqlens_k.tolist()
            for i, (q_start, q_end) in enumerate(zip(q_offsets, q_offsets[1:])):
                k_len = k_offsets[i + 1] - k_offsets[i]
                if context.block_tables is None:
                    key = k[q_start:q_end]
                    value = v[q_start:q_end]
                else:
                    key = self._cache_tokens(self.k_cache, context.block_tables[i], k_len)
                    value = self._cache_tokens(self.v_cache, context.block_tables[i], k_len)
                outputs.append(self._attend(q[q_start:q_end], key, value, k_len - (q_end - q_start)))
        else:
            for i, k_len in enumerate(context.context_lens.tolist()):
                key = self._cache_tokens(self.k_cache, context.block_tables[i], k_len)
                value = self._cache_tokens(self.v_cache, context.block_tables[i], k_len)
                outputs.append(self._attend(q[i:i + 1], key, value, k_len - 1))
        return torch.cat(outputs, dim=0)
