import os
import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # pyright: ignore[reportMissingImports]
    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    HAS_FLASH_ATTN = False
from nanovllm.utils.context import get_context


def get_attention_backend() -> str:
    backend = os.getenv("NANOVLLM_ATTENTION_BACKEND", "auto").lower()
    if backend not in {"auto", "flash", "sdpa"}:
        raise ValueError(f"Invalid NANOVLLM_ATTENTION_BACKEND={backend!r}, expected one of auto|flash|sdpa.")
    if backend == "flash":
        if not HAS_FLASH_ATTN:
            raise RuntimeError("NANOVLLM_ATTENTION_BACKEND=flash, but flash-attn is not installed.")
        return "flash"
    if backend == "sdpa":
        return "sdpa"
    return "flash" if HAS_FLASH_ATTN else "sdpa"


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


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
        self.backend = get_attention_backend()
        self.k_cache = self.v_cache = torch.tensor([])

    @staticmethod
    def _build_causal_mask(q_len: int, k_len: int, q_start: int, device: torch.device) -> torch.Tensor:
        q_pos = q_start + torch.arange(q_len, device=device).unsqueeze(1)
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)
        return k_pos <= q_pos

    @staticmethod
    def _gather_from_cache(cache: torch.Tensor, block_table: torch.Tensor, seqlen: int) -> torch.Tensor:
        block_ids = block_table[block_table >= 0].to(dtype=torch.long)
        x = cache.index_select(0, block_ids).flatten(0, 1)
        return x[:seqlen]

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_start: int,
    ) -> torch.Tensor:
        q_len, k_len = q.size(0), k.size(0)
        if q_len == 0:
            return q
        attn_mask = self._build_causal_mask(q_len, k_len, q_start, q.device)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        kwargs = dict(attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=self.scale)
        if self.num_heads != self.num_kv_heads:
            kwargs["enable_gqa"] = True
        try:
            o = F.scaled_dot_product_attention(q, k, v, **kwargs)
        except TypeError:
            # Older torch versions may not support enable_gqa.
            if kwargs.get("enable_gqa", False):
                g = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(g, dim=1)
                v = v.repeat_interleave(g, dim=1)
            kwargs.pop("enable_gqa", None)
            o = F.scaled_dot_product_attention(q, k, v, **kwargs)
        return o.squeeze(0).transpose(0, 1)

    def _forward_sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context) -> torch.Tensor:
        outputs = []
        if context.is_prefill:
            cu_q = context.cu_seqlens_q
            cu_k = context.cu_seqlens_k
            for i in range(cu_q.numel() - 1):
                q_start, q_end = cu_q[i].item(), cu_q[i + 1].item()
                q_i = q[q_start:q_end]
                k_len = cu_k[i + 1].item() - cu_k[i].item()
                if context.block_tables is None:
                    k_i = k[cu_k[i].item():cu_k[i + 1].item()]
                    v_i = v[cu_k[i].item():cu_k[i + 1].item()]
                else:
                    k_i = self._gather_from_cache(self.k_cache, context.block_tables[i], k_len)
                    v_i = self._gather_from_cache(self.v_cache, context.block_tables[i], k_len)
                q_pos_start = k_len - q_i.size(0)
                outputs.append(self._sdpa(q_i, k_i, v_i, q_pos_start))
        else:
            for i in range(q.size(0)):
                q_i = q[i:i + 1]
                k_len = context.context_lens[i].item()
                k_i = self._gather_from_cache(self.k_cache, context.block_tables[i], k_len)
                v_i = self._gather_from_cache(self.v_cache, context.block_tables[i], k_len)
                outputs.append(self._sdpa(q_i, k_i, v_i, k_len - 1))
        if not outputs:
            return q.new_empty((0, self.num_heads, self.head_dim))
        return torch.cat(outputs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if self.backend == "sdpa":
            return self._forward_sdpa(q, k, v, context)
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