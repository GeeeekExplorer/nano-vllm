"""Drop-in replacement for the FlashAttention APIs used by nano-vllm,
implemented with NVIDIA cuTile.

This module exports two functions matching the signatures consumed by
``nanovllm.layers.attention.Attention``:

    flash_attn_varlen_func(q, k, v, *, max_seqlen_q, cu_seqlens_q,
                           max_seqlen_k, cu_seqlens_k, softmax_scale,
                           causal=True, block_table=None) -> Tensor

    flash_attn_with_kvcache(q, k_cache, v_cache, *, cache_seqlens,
                            block_table, softmax_scale, causal=True) -> Tensor

Two cuTile kernels back the implementation:

* ``_varlen_attn_kernel``  -- packed Q/K/V varlen prefill (no paged KV).
* ``_paged_attn_kernel``   -- paged-KV attention; reused for prefill with a
                              prefix cache and for single-token decode.
"""

from __future__ import annotations

import math
import functools

import numpy as np
import torch
import cuda.tile as ct


_INV_LOG2 = 1.0 / math.log(2)
_ConstInt = ct.Constant[int]
_ConstBool = ct.Constant[bool]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@ct.kernel(occupancy=2)
def _varlen_attn_kernel(
    Q, K, V, Out,
    Cu_q, Cu_k,
    qk_scale: float,
    H: _ConstInt,
    GROUP: _ConstInt,
    TILE_M: _ConstInt,
    TILE_N: _ConstInt,
    TILE_D: _ConstInt,
    CAUSAL: _ConstBool,
):
    """Variable-length packed attention (no paged KV).

    ``Q`` is ``(total_q, NQH, D)``; ``K``/``V`` are ``(total_k, NKH, D)``.
    Sequence boundaries come from the prefix-sum tensors ``Cu_q`` and ``Cu_k``.
    Each CTA computes one ``(TILE_M, D)`` output tile for one ``(seq, head)``
    pair; tiles past the sequence end are inert (no scatter writes).
    """
    bid_m = ct.bid(0)
    bid_bh = ct.bid(1)
    batch_idx = bid_bh // H
    head_idx = bid_bh % H
    kv_head_idx = head_idx // GROUP

    q_start = ct.load(Cu_q, index=(batch_idx,), shape=(1,)).reshape(())
    q_end = ct.load(Cu_q, index=(batch_idx + 1,), shape=(1,)).reshape(())
    k_start = ct.load(Cu_k, index=(batch_idx,), shape=(1,)).reshape(())
    k_end = ct.load(Cu_k, index=(batch_idx + 1,), shape=(1,)).reshape(())
    seqlen_q = q_end - q_start
    seqlen_k = k_end - k_start

    qk_scale_log2 = qk_scale * _INV_LOG2

    # Q row indices (-1 marks invalid → gather pads with 0, scatter is a no-op).
    m_local = bid_m * TILE_M + ct.arange(TILE_M, dtype=np.int32)
    m_valid = m_local < seqlen_q
    m_safe = ct.where(m_valid, q_start + m_local,
                      ct.full((TILE_M,), -1, dtype=np.int32))
    d_offs = ct.arange(TILE_D, dtype=np.int32)

    q_tile = ct.gather(
        Q, (m_safe[:, None], head_idx, d_offs[None, :]),
        padding_value=0.0,
    )

    m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((TILE_M, 1), 1.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

    # Absolute key position for each query row (handles offset between
    # ``seqlen_q`` and ``seqlen_k`` -- e.g. prefix-cache prefill).
    kv_offset = seqlen_k - seqlen_q
    abs_m = (kv_offset + m_local)[:, None]

    if CAUSAL:
        last_key = kv_offset + (bid_m + 1) * TILE_M
        if last_key > seqlen_k:
            last_key = seqlen_k
        Tc = ct.cdiv(last_key, TILE_N)
    else:
        Tc = ct.cdiv(seqlen_k, TILE_N)

    n_offs_tile = ct.arange(TILE_N, dtype=np.int32)[None, :]

    for j in range(0, Tc):
        n_local = j * TILE_N + n_offs_tile           # (1, TILE_N)
        n_in_range = n_local < seqlen_k              # (1, TILE_N)
        n_safe = ct.where(
            n_in_range, k_start + n_local,
            ct.full((1, TILE_N), -1, dtype=np.int32),
        ).reshape((TILE_N,))

        k_nd = ct.gather(
            K, (n_safe[:, None], kv_head_idx, d_offs[None, :]),
            padding_value=0.0,
        )                                            # (TILE_N, TILE_D)
        k_dn = ct.transpose(k_nd)                    # (TILE_D, TILE_N)

        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
        qk = ct.mma(q_tile, k_dn, qk)

        valid = n_in_range
        if CAUSAL:
            valid = valid & (abs_m >= n_local)
        qk = qk + ct.where(valid, 0.0, -np.inf)

        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
        qk = qk * qk_scale_log2 - m_ij
        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        v_nd = ct.gather(
            V, (n_safe[:, None], kv_head_idx, d_offs[None, :]),
            padding_value=0.0,
        )
        acc = ct.mma(p.astype(Q.dtype), v_nd, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i)
    ct.scatter(
        Out, (m_safe[:, None], head_idx, d_offs[None, :]),
        acc.astype(Out.dtype),
    )


@ct.kernel(occupancy=2)
def _paged_attn_kernel(
    Q, K_cache, V_cache, Out,
    Block_table,
    Cu_q, Cu_k,
    qk_scale: float,
    H: _ConstInt,
    GROUP: _ConstInt,
    BLOCK_SIZE: _ConstInt,
    TILE_M: _ConstInt,
    TILE_N: _ConstInt,
    TILE_D: _ConstInt,
    BLOCKS_PER_TILE: _ConstInt,   # = TILE_N // BLOCK_SIZE  (when TILE_N >= BS)
    TILES_PER_BLOCK: _ConstInt,   # = BLOCK_SIZE // TILE_N (when BS >= TILE_N)
    BS_GE_TILE: _ConstBool,       # True iff BLOCK_SIZE >= TILE_N
    CAUSAL: _ConstBool,
):
    """Paged-KV attention.

    ``Q`` is packed ``(total_q, NQH, D)``.
    ``K_cache``/``V_cache`` are ``(num_blocks, BLOCK_SIZE, NKH, D)``.
    ``Block_table[b, j]`` gives the physical block index for the j-th logical
    block of sequence ``b``.

    Used for both varlen prefix-cache prefill (``seqlen_q`` per seq from
    ``Cu_q``) and decode (``seqlen_q == 1``).
    """
    bid_m = ct.bid(0)
    bid_bh = ct.bid(1)
    batch_idx = bid_bh // H
    head_idx = bid_bh % H
    kv_head_idx = head_idx // GROUP

    q_start = ct.load(Cu_q, index=(batch_idx,), shape=(1,)).reshape(())
    q_end = ct.load(Cu_q, index=(batch_idx + 1,), shape=(1,)).reshape(())
    k_start = ct.load(Cu_k, index=(batch_idx,), shape=(1,)).reshape(())
    k_end = ct.load(Cu_k, index=(batch_idx + 1,), shape=(1,)).reshape(())
    seqlen_q = q_end - q_start
    seqlen_k = k_end - k_start

    qk_scale_log2 = qk_scale * _INV_LOG2

    m_local = bid_m * TILE_M + ct.arange(TILE_M, dtype=np.int32)
    m_valid = m_local < seqlen_q
    m_safe = ct.where(m_valid, q_start + m_local,
                      ct.full((TILE_M,), -1, dtype=np.int32))
    d_offs = ct.arange(TILE_D, dtype=np.int32)

    q_tile = ct.gather(
        Q, (m_safe[:, None], head_idx, d_offs[None, :]),
        padding_value=0.0,
    )

    m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((TILE_M, 1), 1.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

    kv_offset = seqlen_k - seqlen_q
    abs_m = (kv_offset + m_local)[:, None]

    if CAUSAL:
        last_key = kv_offset + (bid_m + 1) * TILE_M
        if last_key > seqlen_k:
            last_key = seqlen_k
        Tc = ct.cdiv(last_key, TILE_N)
    else:
        Tc = ct.cdiv(seqlen_k, TILE_N)

    n_offs_tile = ct.arange(TILE_N, dtype=np.int32)[None, :]

    for j in range(0, Tc):
        n_local = j * TILE_N + n_offs_tile
        n_in_range = n_local < seqlen_k

        # Look up the physical block(s) for this N-tile via the block table.
        if BS_GE_TILE:
            # Each block holds ``TILES_PER_BLOCK`` N-tiles.
            jb = j // TILES_PER_BLOCK
            sub = j % TILES_PER_BLOCK
            phys = ct.load(Block_table, index=(batch_idx, jb), shape=(1, 1)).reshape(())
            k_4d = ct.load(
                K_cache,
                index=(phys, sub, kv_head_idx, 0),
                shape=(1, TILE_N, 1, TILE_D),
            )
            v_4d = ct.load(
                V_cache,
                index=(phys, sub, kv_head_idx, 0),
                shape=(1, TILE_N, 1, TILE_D),
            )
            k_nd = k_4d.reshape((TILE_N, TILE_D))
            v_nd = v_4d.reshape((TILE_N, TILE_D))
        else:
            # ``TILE_N`` spans multiple blocks; gather elementwise.
            # logical N positions for this tile -> block index + intra-block offset.
            local_n = j * TILE_N + ct.arange(TILE_N, dtype=np.int32)        # (TILE_N,)
            jb = local_n // BLOCK_SIZE
            within = local_n % BLOCK_SIZE
            jb_safe = ct.where(local_n < seqlen_k, jb,
                               ct.full((TILE_N,), 0, dtype=np.int32))
            phys_tile = ct.gather(
                Block_table, (batch_idx, jb_safe),
                padding_value=0,
            )                                                                # (TILE_N,)
            # Linear index into K_cache flattened along (block, BS): phys*BS + within
            kv_row = phys_tile * BLOCK_SIZE + within                         # (TILE_N,)
            kv_row = ct.where(local_n < seqlen_k, kv_row,
                              ct.full((TILE_N,), -1, dtype=np.int32))
            # K_cache logically (num_blocks, BS, NKH, D) -- gather treats first
            # two dims as fused via the wrapper view.
            k_nd = ct.gather(
                K_cache, (kv_row[:, None], kv_head_idx, d_offs[None, :]),
                padding_value=0.0,
            )
            v_nd = ct.gather(
                V_cache, (kv_row[:, None], kv_head_idx, d_offs[None, :]),
                padding_value=0.0,
            )
        k_dn = ct.transpose(k_nd)

        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
        qk = ct.mma(q_tile, k_dn, qk)

        valid = n_in_range
        if CAUSAL:
            valid = valid & (abs_m >= n_local)
        qk = qk + ct.where(valid, 0.0, -np.inf)

        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
        qk = qk * qk_scale_log2 - m_ij
        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        acc = ct.mma(p.astype(Q.dtype), v_nd, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i)
    ct.scatter(
        Out, (m_safe[:, None], head_idx, d_offs[None, :]),
        acc.astype(Out.dtype),
    )


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@functools.lru_cache(maxsize=64)
def _pick_tiles(seqlen_q_hint: int, head_dim: int, block_size: int) -> tuple[int, int]:
    """Pick (TILE_M, TILE_N). Decode (seqlen_q == 1) gets a small TILE_M."""
    if seqlen_q_hint <= 1:
        tile_m = 16
    elif seqlen_q_hint <= 16:
        tile_m = 16
    elif seqlen_q_hint <= 64:
        tile_m = 64
    else:
        tile_m = 64
    # TILE_N: a divisor of block_size when block_size is large, else a power of 2.
    tile_n = 64 if block_size >= 64 else block_size
    return tile_m, tile_n


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
    block_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """cuTile drop-in replacement for ``flash_attn.flash_attn_varlen_func``.

    Two regimes:

    * ``block_table is None`` -- ``q``, ``k``, ``v`` are packed ``(total, H, D)``.
    * ``block_table is not None`` -- prefix-cache prefill: ``q`` is packed and
      ``k``/``v`` are the paged KV cache ``(num_blocks, BS, NKH, D)``.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[-2]
    assert num_q_heads % num_kv_heads == 0
    group = num_q_heads // num_kv_heads
    batch = cu_seqlens_q.numel() - 1
    out = torch.empty_like(q)

    tile_m, tile_n = _pick_tiles(max_seqlen_q, head_dim, k.shape[1] if block_table is not None else tile_n_default(head_dim))
    grid_m = (max_seqlen_q + tile_m - 1) // tile_m
    grid = (max(grid_m, 1), batch * num_q_heads, 1)

    if block_table is None:
        ct.launch(
            torch.cuda.current_stream(), grid, _varlen_attn_kernel,
            (
                q, k, v, out,
                cu_seqlens_q, cu_seqlens_k,
                float(softmax_scale),
                num_q_heads, group,
                tile_m, tile_n, head_dim,
                bool(causal),
            ),
        )
    else:
        block_size = k.shape[1]
        bs_ge_tile = block_size >= tile_n
        if bs_ge_tile:
            assert block_size % tile_n == 0, \
                f"BLOCK_SIZE ({block_size}) must be a multiple of TILE_N ({tile_n})"
            tiles_per_block = block_size // tile_n
            blocks_per_tile = 1
        else:
            assert tile_n % block_size == 0
            tiles_per_block = 1
            blocks_per_tile = tile_n // block_size
        ct.launch(
            torch.cuda.current_stream(), grid, _paged_attn_kernel,
            (
                q, k, v, out,
                block_table,
                cu_seqlens_q, cu_seqlens_k,
                float(softmax_scale),
                num_q_heads, group,
                block_size,
                tile_m, tile_n, head_dim,
                blocks_per_tile, tiles_per_block,
                bool(bs_ge_tile),
                bool(causal),
            ),
        )
    return out


def tile_n_default(head_dim: int) -> int:
    return 64


# Cached scratch for cu_seqlens during decode.
_DECODE_SCRATCH: dict[tuple[torch.device, int], dict[str, torch.Tensor]] = {}


def _get_decode_scratch(device: torch.device, batch: int) -> dict[str, torch.Tensor]:
    key = (device, batch)
    s = _DECODE_SCRATCH.get(key)
    if s is None:
        s = {
            "cu_q": torch.arange(batch + 1, dtype=torch.int32, device=device),
            "cu_k": torch.zeros(batch + 1, dtype=torch.int32, device=device),
        }
        _DECODE_SCRATCH[key] = s
    return s


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """cuTile drop-in replacement for ``flash_attn.flash_attn_with_kvcache``.

    ``q`` arrives as ``(batch, 1, num_q_heads, head_dim)`` (one decode token per
    sequence); the result has the same shape.
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.dim() == 4 and q.shape[1] == 1
    batch, _, num_q_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    assert num_q_heads % num_kv_heads == 0
    group = num_q_heads // num_kv_heads

    q_packed = q.squeeze(1).contiguous()                  # (batch, NQH, D)
    out_packed = torch.empty_like(q_packed)

    scratch = _get_decode_scratch(q.device, batch)
    cu_q = scratch["cu_q"]                                # [0, 1, 2, ..., batch]
    # cu_k = cumulative cache_seqlens with leading 0. Index 0 is left at the
    # zero it was initialized with so this stays CUDA-graph-capturable.
    cu_k = scratch["cu_k"]
    torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32, out=cu_k[1:])

    tile_m, tile_n = _pick_tiles(1, head_dim, block_size)
    bs_ge_tile = block_size >= tile_n
    if bs_ge_tile:
        assert block_size % tile_n == 0
        tiles_per_block = block_size // tile_n
        blocks_per_tile = 1
    else:
        assert tile_n % block_size == 0
        tiles_per_block = 1
        blocks_per_tile = tile_n // block_size

    grid = (1, batch * num_q_heads, 1)
    ct.launch(
        torch.cuda.current_stream(), grid, _paged_attn_kernel,
        (
            q_packed, k_cache, v_cache, out_packed,
            block_table,
            cu_q, cu_k,
            float(softmax_scale),
            num_q_heads, group,
            block_size,
            tile_m, tile_n, head_dim,
            blocks_per_tile, tiles_per_block,
            bool(bs_ge_tile),
            bool(causal),
        ),
    )
    return out_packed.unsqueeze(1)
