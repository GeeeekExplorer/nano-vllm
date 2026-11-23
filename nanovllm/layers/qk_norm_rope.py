import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope(Q, K, QW, KW, Pos, Cache, OutQ, OutK,
                  QH: tl.constexpr, KH: tl.constexpr, Q_STRIDE: tl.constexpr,
                  K_STRIDE: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr,
                  HEADS: tl.constexpr):
    token = tl.program_id(0)
    group = tl.program_id(1)
    groups_q: tl.constexpr = triton.cdiv(QH, HEADS)
    if group < groups_q:
        src, weight, dst = Q + token * Q_STRIDE, QW, OutQ + token * QH * D
        nheads = QH
        first = group * HEADS
    else:
        src, weight, dst = K + token * K_STRIDE, KW, OutK + token * KH * D
        nheads = KH
        first = (group - groups_q) * HEADS
    head = first + tl.arange(0, HEADS)
    dim = tl.arange(0, D)
    offset = head[:, None] * D + dim[None, :]
    x = tl.load(src + offset, head[:, None] < nheads, 0).to(tl.float32)
    w = tl.load(weight + dim).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    # Inductor keeps the normalization and weight product in FP32 within
    # compiled rms_forward. Its output is rounded on entry to RotaryEmbedding.
    x = (x * inv[:, None] * w[None, :]).to(tl.bfloat16).to(tl.float32)
    other = tl.gather(x, tl.broadcast_to((dim ^ (D // 2))[None, :], (HEADS, D)), 1)
    row = Cache + tl.load(Pos + token) * D
    cos = tl.load(row + dim % (D // 2))
    sin = tl.load(row + D // 2 + dim % (D // 2))
    y = x * cos[None, :] + tl.where(dim[None, :] < D // 2, -other, other) * sin[None, :]
    tl.store(dst + offset, y, head[:, None] < nheads)


def fused_qk_norm_rope(q, k, qw, kw, positions, cache, eps):
    tokens, qheads, dim = q.shape
    kheads = k.shape[1]
    out_q = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    out_k = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    # Match the compiled reference's FP32 reduction order on the SM80 path.
    heads = 4
    grid = (tokens, triton.cdiv(qheads, heads) + triton.cdiv(kheads, heads))
    _qk_norm_rope[grid](q, k, qw, kw, positions, cache, out_q, out_k, qheads, kheads,
                        q.stride(0), k.stride(0), dim, eps, heads, num_warps=4, enable_fp_fusion=True)
    return out_q, out_k
