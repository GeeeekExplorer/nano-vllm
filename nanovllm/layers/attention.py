import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    if key.device.type == "mps":
        from nanovllm.layers.kv_cache_mps import store_kvcache_mps

        return store_kvcache_mps(key, value, k_cache, v_cache, slot_mapping)
    if key.device.type == "cuda":
        from nanovllm.layers.kv_cache_cuda import store_kvcache_cuda

        return store_kvcache_cuda(key, value, k_cache, v_cache, slot_mapping)
    raise NotImplementedError(f"KV-cache storage is unavailable on {key.device.type}")


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if q.device.type == "mps":
            return self.forward_mps(q, k, v)
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o

    def forward_mps(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        has_cache = k_cache.numel() and v_cache.numel()
        if has_cache:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        # The Metal path reads each KV row once per group of query heads.
        if (has_cache and not context.is_prefill
                and (q.shape[0] > 1 or max(context.k_lens) > 192)
                and context.context_lens is not None
                and self.head_dim <= 256):
            from nanovllm.layers.paged_attention_mps import paged_decode_attention_shared_mps

            return paged_decode_attention_shared_mps(
                q.contiguous(), k_cache, v_cache, context.block_tables, context.context_lens, self.scale,
            )
        # Equal context lengths share one batched SDPA call.
        if has_cache and not context.is_prefill and len(set(context.k_lens)) == 1:
            k_len = context.k_lens[0]
            num_blocks = (k_len + k_cache.shape[1] - 1) // k_cache.shape[1]
            block_ids = context.block_tables[:, :num_blocks].long()
            batch_size = q.shape[0]
            keys = k_cache[block_ids].reshape(batch_size, -1, self.num_kv_heads, self.head_dim)[:, :k_len]
            values = v_cache[block_ids].reshape(batch_size, -1, self.num_kv_heads, self.head_dim)[:, :k_len]
            keys = keys.permute(0, 2, 1, 3)
            values = values.permute(0, 2, 1, 3)
            output = self._sdpa_mps(q.unsqueeze(2), keys, values)
            return output.squeeze(2)

        q_lens = context.q_lens if context.is_prefill else [1] * q.shape[0]
        k_lens = context.k_lens
        outputs = []
        offset = 0
        for i, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
            query = q[offset:offset + q_len].transpose(0, 1).unsqueeze(0)
            prefix_len = k_len - q_len
            if prefix_len and not has_cache:
                raise RuntimeError("prefix attention requires an allocated KV cache")
            if has_cache and (not context.is_prefill or prefix_len):
                num_blocks = (k_len + k_cache.shape[1] - 1) // k_cache.shape[1]
                block_ids = context.block_tables[i, :num_blocks].long()
                keys = k_cache[block_ids].reshape(-1, self.num_kv_heads, self.head_dim)[:k_len]
                values = v_cache[block_ids].reshape(-1, self.num_kv_heads, self.head_dim)[:k_len]
            else:
                keys = k[offset:offset + q_len]
                values = v[offset:offset + q_len]
            keys = keys.transpose(0, 1).unsqueeze(0)
            values = values.transpose(0, 1).unsqueeze(0)
            mask = None
            if context.is_prefill and prefix_len:
                query_positions = torch.arange(prefix_len, k_len, device=q.device)
                key_positions = torch.arange(k_len, device=q.device)
                mask = key_positions[None, :] <= query_positions[:, None]
            output = self._sdpa_mps(query, keys, values, mask, context.is_prefill and not prefix_len)
            outputs.append(output.squeeze(0).transpose(0, 1))
            offset += q_len
        return torch.cat(outputs, dim=0)

    def _sdpa_mps(self, query, keys, values, attn_mask=None, is_causal=False):
        if self.num_heads == self.num_kv_heads:
            return F.scaled_dot_product_attention(
                query, keys, values, attn_mask=attn_mask, is_causal=is_causal, scale=self.scale,
            )
        batch_size, _, q_len, head_dim = query.shape
        groups = self.num_heads // self.num_kv_heads
        # Broadcast K/V across their query heads without expanding the cache.
        query = query.contiguous().reshape(batch_size, self.num_kv_heads, groups, q_len, head_dim)
        keys = keys.contiguous()
        values = values.contiguous()
        output = F.scaled_dot_product_attention(
            query, keys.unsqueeze(2), values.unsqueeze(2),
            attn_mask=attn_mask, is_causal=is_causal, scale=self.scale,
        )
        return output.reshape(batch_size, self.num_heads, q_len, head_dim)
