import torch
from torch import nn

# Triton is optional. When it's not available, we fall back to a torch KV-cache writer.
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _HAS_TRITON = True
except ModuleNotFoundError:
    triton = None
    tl = None
    _HAS_TRITON = False

# flash-attn is optional. When it's unavailable, we fall back to a torch attention implementation.
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # type: ignore
    _HAS_FLASH_ATTN = True
except ModuleNotFoundError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    _HAS_FLASH_ATTN = False
from nanovllm.utils.context import get_context


if _HAS_TRITON:
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
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    Write (K, V) into KV cache at positions given by `slot_mapping`.

    key/value: (N, num_kv_heads, head_dim)
    slot_mapping: (N,) where each entry is an absolute slot index, or -1 for "ignore".
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert slot_mapping.numel() == N

    if _HAS_TRITON:
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        store_kvcache_kernel[(N,)](
            key,
            key.stride(0),
            value,
            value.stride(0),
            k_cache,
            v_cache,
            slot_mapping,
            D,
        )
        return

    # Torch fallback: flatten caches to (num_slots, D) and assign.
    key_flat = key.reshape(N, D)
    value_flat = value.reshape(N, D)
    k_cache_flat = k_cache.contiguous().view(-1, D)
    v_cache_flat = v_cache.contiguous().view(-1, D)

    mask = slot_mapping != -1
    if mask.any():
        slots = slot_mapping[mask].to(torch.long)
        k_cache_flat[slots] = key_flat[mask]
        v_cache_flat[slots] = value_flat[mask]


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
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if _HAS_FLASH_ATTN:
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            else:
                # Correctness-oriented torch fallback.
                # Note: this path currently doesn't support prefix-cache (block_tables != None).
                if context.block_tables is not None:
                    raise NotImplementedError(
                        "Torch flash_attn fallback does not support prefix-cache (block_tables != None). "
                        "Install flash-attn or disable prefix caching for this debug run."
                    )
                assert context.cu_seqlens_q is not None and context.cu_seqlens_k is not None
                cu_q = context.cu_seqlens_q.tolist()
                cu_k = context.cu_seqlens_k.tolist()
                assert len(cu_q) == len(cu_k), "cu_seqlens_q/cu_seqlens_k length mismatch"

                outs = []
                # q: (total_q, num_heads, head_dim)
                # k/v: (total_k, num_kv_heads, head_dim)
                for start_q, end_q, start_k, end_k in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:]):
                    q_i = q[start_q:end_q]  # (L, H, D)
                    k_i = k[start_k:end_k]  # (L, Hk, D)
                    v_i = v[start_k:end_k]  # (L, Hk, D)

                    if self.num_heads != self.num_kv_heads:
                        if self.num_heads % self.num_kv_heads != 0:
                            raise ValueError("num_heads must be a multiple of num_kv_heads for torch fallback")
                        group = self.num_heads // self.num_kv_heads
                        k_i = k_i.repeat_interleave(group, dim=1)
                        v_i = v_i.repeat_interleave(group, dim=1)

                    # scaled_dot_product_attention expects (B, H, L, D)
                    q_t = q_i.transpose(0, 1).unsqueeze(0)
                    k_t = k_i.transpose(0, 1).unsqueeze(0)
                    v_t = v_i.transpose(0, 1).unsqueeze(0)

                    q_dtype = q_t.dtype
                    if q_t.device.type != "cuda" and q_dtype in (torch.float16, torch.bfloat16):
                        q_t = q_t.float()
                        k_t = k_t.float()
                        v_t = v_t.float()

                    attn = torch.nn.functional.scaled_dot_product_attention(
                        q_t, k_t, v_t, attn_mask=None, dropout_p=0.0, is_causal=True
                    )
                    out_i = attn.squeeze(0).transpose(0, 1).to(q_dtype)
                    outs.append(out_i)
                o = torch.cat(outs, dim=0)
        else:
            # decode
            if _HAS_FLASH_ATTN:
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
            else:
                # Correctness-oriented torch fallback for decode.
                assert context.context_lens is not None and context.block_tables is not None
                bs = q.shape[0]
                block_size = k_cache.shape[1]
                k_cache_flat = k_cache.contiguous().view(-1, self.num_kv_heads, self.head_dim)
                v_cache_flat = v_cache.contiguous().view(-1, self.num_kv_heads, self.head_dim)

                outs = []
                for i in range(bs):
                    seqlen = int(context.context_lens[i].item())
                    block_ids = context.block_tables[i].tolist()
                    block_ids = [bid for bid in block_ids if bid != -1]
                    if not block_ids:
                        raise RuntimeError("decode torch fallback: empty block_ids")

                    # token positions [0..seqlen-1] map to slots: block_id * block_size + offset
                    pos = torch.arange(seqlen, device=q.device, dtype=torch.long)
                    block_idx = (pos // block_size).to(torch.long)
                    within = (pos % block_size).to(torch.long)

                    block_id_tensor = torch.tensor(block_ids, device=q.device, dtype=torch.long)
                    slot_indices = block_id_tensor[block_idx] * block_size + within

                    K = k_cache_flat[slot_indices]  # (seqlen, Hk, D)
                    V = v_cache_flat[slot_indices]  # (seqlen, Hk, D)
                    if self.num_heads != self.num_kv_heads:
                        if self.num_heads % self.num_kv_heads != 0:
                            raise ValueError("num_heads must be a multiple of num_kv_heads for torch fallback")
                        group = self.num_heads // self.num_kv_heads
                        K = K.repeat_interleave(group, dim=1)
                        V = V.repeat_interleave(group, dim=1)

                    q_i = q[i]  # (H, D)
                    q_dtype = q_i.dtype
                    if q_i.device.type != "cuda" and q_dtype in (torch.float16, torch.bfloat16):
                        q_i = q_i.float()
                        K = K.float()
                        V = V.float()

                    # scores: (H, seqlen)
                    scores = torch.einsum("hd,thd->ht", q_i, K) * self.scale
                    attn = torch.softmax(scores.float(), dim=-1).to(V.dtype)
                    out_i = torch.einsum("ht,thd->hd", attn, V).to(q_dtype)
                    outs.append(out_i)
                o = torch.stack(outs, dim=0)
        return o
