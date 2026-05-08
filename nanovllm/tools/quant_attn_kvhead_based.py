import torch
import triton
import triton.language as tl

decode_num_warps = 1
decode_num_stages = 1
prefill_num_warps = 4
prefill_num_stages = 1
prefill_num_tileQ = 32
TILE_KV_PREFILL = 32
TILE_KV_DECODE = 32
assert prefill_num_tileQ == TILE_KV_PREFILL
NUM_KV_TILES_PER_BLOCK_PREFILL = 256 // TILE_KV_PREFILL      # 256/32 = 8
NUM_KV_TILES_PER_BLOCK_DECODE = 256 // TILE_KV_DECODE        # 256/32 = 8

@triton.jit
def _decode_quant_direct_kernel(
    q_ptr,      # [seqs_num, q_head_num, head_dim]
    new_k_ptr,  # [seqs_num, kv_head_num, head_dim]
    new_v_ptr,  # [seqs_num, kv_head_num, head_dim]
    stride_new_b,
    stride_new_h,
    stride_new_d,
    k_cache_ptr,    # [kv_head_num, num_slots, head_dim]
    v_cache_ptr,    # [kv_head_num, num_slots, head_dim]
    k_scale_ptr,    # [kv_head_num, num_slots]
    v_scale_ptr,    # [kv_head_num, num_slots]
    context_lens_ptr,   # [seqs_num]
    block_table_ptr,
    out_ptr,         # [seqs_num, q_head_num, head_dim]
    stride_q_b,      # q_head_num * head_dim
    stride_q_h,      # head_dim
    stride_q_d,      # 1
    stride_cache_h,  # num_slots * head_dim
    stride_cache_s,  # head_dim
    stride_cache_d,  # 1
    stride_scale_h,  # num_slots
    stride_scale_s,  # 1
    stride_bt_b,    # max_block_table_len
    stride_bt_blk,  # 1
    stride_out_b,   # = stride_q_b
    stride_out_h,   # head_dim
    stride_out_d,   # 1
    softmax_scale,  # 1/sqrt(dim)
    block_size,
    TILE_KV: tl.constexpr,      # tile_K
    BLOCK_DIM_MODEL: tl.constexpr,      # head_dim
    TARGET_DTYPE: tl.constexpr,
    NUM_KV_TILES_PER_BLOCK: tl.constexpr,
    GROUP_NUM: tl.constexpr,    # GQA
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    q_head_start = kv_head_idx * GROUP_NUM
    q_head_offsets = q_head_start + tl.arange(0, GROUP_NUM)
    ctx_len = tl.load(context_lens_ptr + batch_idx)
    ctx_len -= 1
    num_total_tiles = tl.cdiv(ctx_len, TILE_KV)
    if num_total_tiles <= 0 or ctx_len <= 0:
        return
    offs_dim = tl.arange(0, BLOCK_DIM_MODEL)
    offs_kv = tl.arange(0, TILE_KV)
    q_ptrs = q_ptr + batch_idx * stride_q_b + q_head_offsets[:, None] * stride_q_h + offs_dim[None, :] * stride_q_d
    # [GROUP_NUM, head_dim]
    q = tl.load(q_ptrs)
    m_i = tl.full((GROUP_NUM,), float("-inf"), tl.float32)
    li = tl.full((GROUP_NUM,), 0.0, tl.float32)
    oi = tl.zeros((GROUP_NUM, BLOCK_DIM_MODEL), dtype=tl.float32)
    curr_kv_block_idx = 0
    physical_block_idx = tl.load(block_table_ptr + batch_idx * stride_bt_b + curr_kv_block_idx * stride_bt_blk)
    for tile_idx in tl.range(0, num_total_tiles - 1):
        kv_block_idx = tile_idx // NUM_KV_TILES_PER_BLOCK
        kv_tile_idx = tile_idx % NUM_KV_TILES_PER_BLOCK
        if kv_block_idx != curr_kv_block_idx:
            curr_kv_block_idx = kv_block_idx
            physical_block_idx = tl.load(block_table_ptr + batch_idx * stride_bt_b + curr_kv_block_idx * stride_bt_blk)
        logic_in_block_offs = kv_tile_idx * TILE_KV
        token_in_block = logic_in_block_offs + offs_kv
        slot = physical_block_idx * block_size + token_in_block
        scale_offs = kv_head_idx * stride_scale_h + slot * stride_scale_s
        cache_offs = kv_head_idx * stride_cache_h + slot[:, None] * stride_cache_s + offs_dim[None, :] * stride_cache_d
        kvj = tl.load(k_cache_ptr + cache_offs).to(TARGET_DTYPE)
        kv_scale = tl.load(k_scale_ptr + scale_offs).to(TARGET_DTYPE)
        kvj = kvj * kv_scale[:, None]
        scores = tl.sum(q[:, None, :] * kvj[None, :, :], axis=2) * softmax_scale
        m_ij = tl.max(scores, axis=1)  # [GROUP_NUM]
        mj_new = tl.maximum(m_i, m_ij)  # [GROUP_NUM]
        alpha = tl.exp(m_i - mj_new)  # [GROUP_NUM]
        scores = tl.exp(scores - mj_new[:, None])  # [GROUP_NUM, tileK]
        li = alpha * li + tl.sum(scores, axis=1)  # [GROUP_NUM]
        kvj = tl.load(v_cache_ptr + cache_offs).to(TARGET_DTYPE)
        kv_scale = tl.load(v_scale_ptr + scale_offs).to(TARGET_DTYPE)
        kvj = kvj * kv_scale[:, None]
        # [GROUP_NUM, head_dim]
        oij = tl.sum(scores[:, :, None] * kvj[None, :, :], axis=1)
        oi = oi * alpha[:, None] + oij
        m_i = mj_new
    last_tile_idx = num_total_tiles - 1
    last_kv_block_idx = last_tile_idx // NUM_KV_TILES_PER_BLOCK
    last_kv_tile_idx = last_tile_idx % NUM_KV_TILES_PER_BLOCK
    if last_kv_block_idx != curr_kv_block_idx:
        physical_block_idx = tl.load(block_table_ptr + batch_idx * stride_bt_b + last_kv_block_idx * stride_bt_blk)
    logic_block_offs = last_kv_block_idx * block_size
    logic_in_block_offs = last_kv_tile_idx * TILE_KV
    token_in_block = logic_in_block_offs + offs_kv
    logical_token_idx = logic_block_offs + token_in_block
    valid = logical_token_idx < ctx_len
    slot = physical_block_idx * block_size + token_in_block
    scale_offs = kv_head_idx * stride_scale_h + slot * stride_scale_s
    cache_offs = kv_head_idx * stride_cache_h + slot[:, None] * stride_cache_s + offs_dim[None, :] * stride_cache_d
    kvj = tl.load(k_cache_ptr + cache_offs, mask=valid[:, None], other=0).to(TARGET_DTYPE)
    kv_scale = tl.load(k_scale_ptr + scale_offs, mask=valid, other=0.0).to(TARGET_DTYPE)
    kvj = kvj * kv_scale[:, None]
    scores = tl.sum(q[:, None, :] * kvj[None, :, :], axis=2) * softmax_scale
    scores = tl.where(valid[None, :], scores, float("-inf"))
    m_ij = tl.max(scores, axis=1)
    mj_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - mj_new)
    p = tl.exp(scores - mj_new[:, None])
    li = alpha * li + tl.sum(p, axis=1)
    kvj = tl.load(v_cache_ptr + cache_offs, mask=valid[:, None], other=0).to(TARGET_DTYPE)
    kv_scale = tl.load(v_scale_ptr + scale_offs, mask=valid, other=0.0).to(TARGET_DTYPE)
    kvj = kvj * kv_scale[:, None]
    oij = tl.sum(p[:, :, None] * kvj[None, :, :], axis=1)
    oi = oi * alpha[:, None] + oij
    m_i = mj_new
    cache_offs = batch_idx * stride_new_b + kv_head_idx * stride_new_h + offs_dim * stride_new_d
    new_kv = tl.load(new_k_ptr + cache_offs)
    scores = tl.sum(q * new_kv[None, :], axis=1)
    scores = scores * softmax_scale
    mj_new = tl.maximum(m_i, scores)
    alpha = tl.exp(m_i - mj_new)
    p = tl.exp(scores - mj_new)
    li = alpha * li + p
    new_kv = tl.load(new_v_ptr + cache_offs)
    oi = oi * alpha[:, None] + p[:, None] * new_kv[None, :]
    oi = oi / li[:, None]
    # [GROUP_NUM, head_dim]
    out_ptrs = out_ptr + batch_idx * stride_out_b + q_head_offsets[:, None] * stride_out_h + offs_dim[None, :] * stride_out_d
    tl.store(out_ptrs, oi.to(TARGET_DTYPE))

def decode_attn_quantkv_direct(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    GROUP_NUM: int,
    softmax_scale: float | None = None,
):
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads, num_slots, _ = k_cache.shape
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    out = torch.empty_like(q)
    if q.dtype == torch.float16:
        target_dtype = tl.float16
    elif q.dtype == torch.bfloat16:
        target_dtype = tl.bfloat16
    else:
        target_dtype = tl.float32

    _decode_quant_direct_kernel[(batch, num_kv_heads, )](
        q,
        new_k,
        new_v,
        new_k.stride(0),
        new_k.stride(1),
        new_k.stride(2),
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        context_lens,
        block_table,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        softmax_scale,
        block_size,
        TILE_KV=TILE_KV_DECODE,
        BLOCK_DIM_MODEL=head_dim,
        TARGET_DTYPE=target_dtype,
        NUM_KV_TILES_PER_BLOCK=NUM_KV_TILES_PER_BLOCK_DECODE,
        GROUP_NUM=GROUP_NUM,
        num_warps=decode_num_warps,
        num_stages=decode_num_stages,
    )
    return out

@triton.jit
def _prefill_quant_direct_kernel(
    q_ptr,                   # [total_q, q_head_num, head_dim] without prefix part
    k_ptr,                   # [total_k, kv_head_num, head_dim] total_q == total_k == total_v
    v_ptr,                   # [total_v, kv_head_num, head_dim]
    stride_kv_b,             # kv_head_num * head_dim
    stride_kv_h,             # head_dim
    stride_kv_d,             # 1
    k_cache_ptr,             # [kv_head_num, num_slots, head_dim]
    v_cache_ptr,             # [kv_head_num, num_slots, head_dim]
    k_scale_ptr,             # [kv_head_num, num_slots]
    v_scale_ptr,             # [kv_head_num, num_slots]
    cu_seqlens_q_ptr,        # [batch + 1]
    context_lens_ptr,        # [batch]
    block_table_ptr,         # [batch, max_num_blocks_per_seq]
    out_ptr,                 # [total_q, q_head_num, head_dim]
    stride_q_b,              # q_head_num * head_dim
    stride_q_h,              # head_dim
    stride_q_d,              # 1
    stride_cache_h,          # num_slots * head_dim
    stride_cache_s,          # head_dim
    stride_cache_d,          # 1
    stride_scale_h,          # num_slots
    stride_scale_s,          # 1
    stride_bt_b,             # max_block_table_len
    stride_bt_blk,           # 1
    stride_out_b,            # = stride_q_b
    stride_out_h,            # head_dim
    stride_out_d,            # 1
    softmax_scale,           # = 1/sqrt(dim)
    block_size,
    TILE_Q: tl.constexpr,
    TILE_KV: tl.constexpr,
    TILE_DIM_MODEL: tl.constexpr,   # head_dim
    TARGET_DTYPE: tl.constexpr,     # bfloat16
    NUM_KV_TILES_PER_BLOCK: tl.constexpr,
    GROUP_NUM: tl.constexpr,        # GQA
):
    batch_idx = tl.program_id(2)
    kv_head_idx = tl.program_id(1)
    q_tile_idx = tl.program_id(0)
    q_start = tl.load(cu_seqlens_q_ptr + batch_idx)
    q_end = tl.load(cu_seqlens_q_ptr + batch_idx + 1)
    q_len = q_end - q_start
    k_len = tl.load(context_lens_ptr + batch_idx)
    prefix_len = k_len - q_len
    q_tile_start = q_tile_idx * TILE_Q
    if q_tile_start >= q_len or q_len <= 0:
        return
    offs_dim = tl.arange(0, TILE_DIM_MODEL)
    offs_kv = tl.arange(0, TILE_KV)
    offs_rows = tl.arange(0, GROUP_NUM * TILE_Q)
    offs_group = offs_rows // TILE_Q
    offs_q_in_tile = offs_rows % TILE_Q
    logic_offs_q = q_tile_start + offs_q_in_tile    # [GROUP_NUM * TILE_Q]
    q_valid = logic_offs_q < q_len  #[tileQ] valid_q_num
    logic_offs_q_in_total = q_start + logic_offs_q
    q_head_offsets = kv_head_idx * GROUP_NUM + offs_group
    q_ptrs = (q_ptr + logic_offs_q_in_total[:, None] * stride_q_b + q_head_offsets[:, None] * stride_q_h + offs_dim[None, :] * stride_q_d)
    q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)   # [GROUP_NUM * tileQ, head_dim]
    m_i = tl.where(q_valid, float("-inf"), 0.0).to(tl.float32)
    l_i = tl.zeros((GROUP_NUM * TILE_Q,), dtype=tl.float32)
    oi = tl.zeros((GROUP_NUM * TILE_Q, TILE_DIM_MODEL), dtype=tl.float32)
    # prefix KV, from head-major kv cache
    num_prefix_tiles = tl.cdiv(prefix_len, TILE_KV)
    for tile_idx in tl.range(0, num_prefix_tiles):
        kv_block_idx = tile_idx // NUM_KV_TILES_PER_BLOCK
        kv_tile_idx = tile_idx % NUM_KV_TILES_PER_BLOCK
        physical_block_idx = tl.load(block_table_ptr + batch_idx * stride_bt_b + kv_block_idx * stride_bt_blk)
        logic_block_offs = kv_block_idx * block_size
        logic_in_block_offs = kv_tile_idx * TILE_KV
        token_in_block = logic_in_block_offs + offs_kv
        logical_k = logic_block_offs + token_in_block
        kv_valid = logical_k < prefix_len
        slot = physical_block_idx * block_size + token_in_block
        attn_mask = (q_valid[:, None] & kv_valid[None, :])  # no need casual mask
        scale_offs = kv_head_idx * stride_scale_h + slot * stride_scale_s
        cache_offs = kv_head_idx * stride_cache_h + slot[:, None] * stride_cache_s + offs_dim[None, :] * stride_cache_d
        kj = tl.load(k_cache_ptr + cache_offs, mask=kv_valid[:, None], other=0).to(TARGET_DTYPE)
        k_scale = tl.load(k_scale_ptr + scale_offs, mask=kv_valid, other=0.0).to(TARGET_DTYPE)
        kj = kj * k_scale[None, :]
        # [GROUP_NUM * tileQ, tileK]
        scores = tl.dot(q, tl.trans(kj)) * softmax_scale
        scores = tl.where(attn_mask, scores, float("-inf"))
        # [GROUP_NUM * tileQ]
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.where(attn_mask, tl.exp(scores - m_new[:, None]), 0.0).to(TARGET_DTYPE)
        # [GROUP_NUM * tileQ]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        vj = tl.load(v_cache_ptr + cache_offs, mask=kv_valid[:, None], other=0).to(TARGET_DTYPE)
        v_scale = tl.load(v_scale_ptr + scale_offs, mask=kv_valid, other=0.0).to(TARGET_DTYPE)
        vj = vj * v_scale[:, None]
        # [GROUP_NUM * tileQ, head_dim]
        oi = oi * alpha[:, None] + tl.dot(p, vj)
        m_i = m_new
    # post KV: from new kv [total_k, kv_head_num, head_dim]
    num_post_tiles = tl.cdiv(q_len, TILE_KV)
    for tile_idx in tl.range(0, num_post_tiles):
        post_logic_k = tile_idx * TILE_KV + offs_kv
        post_valid = post_logic_k < q_len
        attn_mask = q_valid[:, None] & post_valid[None, :] & (post_logic_k[None, :] <= logic_offs_q[:, None])
        postfix_global_k = q_start + post_logic_k
        cache_offs = postfix_global_k[:, None] * stride_kv_b + kv_head_idx * stride_kv_h + offs_dim[None, :] * stride_kv_d
        kj = tl.load(k_ptr + cache_offs, mask=post_valid[:, None], other=0.0)
        # [GROUP_NUM * tileQ, tileK]
        scores = tl.dot(q, tl.trans(kj))
        scores = scores * softmax_scale
        scores = tl.where(attn_mask, scores, float("-inf"))
        # [GROUP_NUM * tileQ]
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.where(attn_mask, tl.exp(scores - m_new[:, None]), 0.0).to(TARGET_DTYPE)
        # [GROUP_NUM * tileQ]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        vj = tl.load(v_ptr + cache_offs, mask=post_valid[:, None], other=0.0)
        oi = oi * alpha[:, None] + tl.dot(p, vj)
        m_i = m_new
    oi = oi / l_i[:, None]
    # [GROUP_NUM * tileQ, head_dim]
    out_ptrs = out_ptr + logic_offs_q_in_total[:, None] * stride_out_b + q_head_offsets[:, None] * stride_out_h + offs_dim[None, :] * stride_out_d
    tl.store(out_ptrs, oi.to(TARGET_DTYPE), mask=q_valid[:, None])

def prefill_attn_quantkv_direct(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    GROUP_NUM: int,
    softmax_scale: float | None = None,
    causal: bool = True,
):
    assert causal == True
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads, num_slots, _ = k_cache.shape
    batch = cu_seqlens_q.numel() - 1
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    out = torch.empty_like(q)
    if q.dtype == torch.float16:
        target_dtype = tl.float16
    elif q.dtype == torch.bfloat16:
        target_dtype = tl.bfloat16
    else:
        target_dtype = tl.float32

    if k_cache.numel():
        stride_cache_h = k_cache.stride(0)
        stride_cache_s = k_cache.stride(1)
        stride_cache_d = k_cache.stride(2)
        stride_scale_h = k_scale.stride(0)
        stride_scale_s = k_scale.stride(1)
    else:
        # warmup
        stride_cache_h = 0
        stride_cache_s = 0
        stride_cache_d = 0
        stride_scale_h = 0
        stride_scale_s = 0
    grid = (triton.cdiv(max_seqlen_q, prefill_num_tileQ), num_kv_heads, batch )
    _prefill_quant_direct_kernel[grid](
        q,
        k,
        v,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        cu_seqlens_q,
        context_lens,
        block_table,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        stride_cache_h,
        stride_cache_s,
        stride_cache_d,
        stride_scale_h,
        stride_scale_s,
        block_table.stride(0),
        block_table.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        softmax_scale,
        block_size,
        TILE_Q=prefill_num_tileQ,
        TILE_KV=TILE_KV_PREFILL,
        TILE_DIM_MODEL=head_dim,
        TARGET_DTYPE=target_dtype,
        NUM_KV_TILES_PER_BLOCK=NUM_KV_TILES_PER_BLOCK_PREFILL,
        GROUP_NUM=GROUP_NUM,
        num_warps=prefill_num_warps,
        num_stages=prefill_num_stages,
    )
