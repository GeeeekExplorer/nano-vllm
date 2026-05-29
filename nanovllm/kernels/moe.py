import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 可捕获(CUDA-graph capturable)的 MoE dispatch + Triton fused grouped-GEMM
#
# 设计要点（全部满足 CUDA Graph 捕获要求：无 host 同步、shape 固定、无数据相关控制流）：
#   - moe_align_block_size: 用纯 torch 把 [T, top_k] 的路由结果按 expert 分组，
#     并 padding 到 block_size 的整数倍，输出**定长** buffer。
#   - fused_moe_kernel: grid 固定 (按 padded 上界)，每个 block 在 GPU 端读取自己负责的
#     expert id，再做该 token-block 与对应 expert 权重的 GEMM。
#
# 所有 per-(token, expert) 的中间量都用「展平索引 f = t*top_k + k」寻址。
# ---------------------------------------------------------------------------


def moe_align_block_size(
    topk_ids: torch.Tensor,   # [T, top_k] int，每个 token 选中的 expert id
    block_size: int,
    num_experts: int,
):
    """把 token-expert 对按 expert 分组并对齐到 block_size。

    返回（均为定长，shape 只依赖 T/top_k/num_experts/block_size，与具体路由值无关）：
        sorted_token_ids: [EM] —— 每个槽位对应的展平索引 f=t*top_k+k；padding 槽位 = num_valid(哨兵)
        expert_ids:       [EM // block_size] —— 每个 block 所属 expert id
        num_valid:        标量 tensor = T*top_k（有效槽位数，kernel 用它做 mask）
    """
    T, top_k = topk_ids.shape
    device = topk_ids.device
    num_valid = T * top_k
    # padded 上界：每个 expert 最多额外 padding (block_size-1)
    em = num_valid + num_experts * block_size
    em = (em + block_size - 1) // block_size * block_size
    num_blocks = em // block_size

    flat_expert = topk_ids.reshape(-1).to(torch.int64)            # [num_valid]
    order = torch.argsort(flat_expert, stable=True)               # 按 expert 分组
    sorted_expert = flat_expert[order]                            # [num_valid]

    # 每个 expert 的 token 数（定长 [num_experts]）
    counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    counts.scatter_add_(0, sorted_expert, torch.ones_like(sorted_expert))
    pad_counts = (counts + block_size - 1) // block_size * block_size  # 对齐
    blocks_per_expert = pad_counts // block_size

    # 每个 expert 在 padded 布局中的起始位置
    pad_offsets = torch.zeros(num_experts, dtype=torch.int64, device=device)
    pad_offsets[1:] = torch.cumsum(pad_counts, dim=0)[:-1]
    # 每个 expert 在 sorted(未 padding) 中的起始位置
    expert_start = torch.zeros(num_experts, dtype=torch.int64, device=device)
    expert_start[1:] = torch.cumsum(counts, dim=0)[:-1]

    # 每个 sorted token 在其 expert 组内的序号 -> padded 目标位置
    within = torch.arange(num_valid, device=device) - expert_start[sorted_expert]
    dest = pad_offsets[sorted_expert] + within                    # [num_valid]

    sorted_token_ids = torch.full((em,), num_valid, dtype=torch.int32, device=device)
    sorted_token_ids[dest] = order.to(torch.int32)
    # 注意: num_valid 是静态值(T*top_k)，作为 python int 返回，便于 Triton 当标量传入

    # 每个 block 的 expert id：在 block_start 处打标记(e+1)，再 cummax 前向填充
    block_start = pad_offsets // block_size                       # [num_experts]
    marker = torch.zeros(num_blocks + 1, dtype=torch.int64, device=device)
    valid = blocks_per_expert > 0
    idx = torch.where(valid, block_start, num_blocks)             # 无效 expert 丢到末尾(被裁掉)
    marker.scatter_(0, idx, torch.where(valid, torch.arange(num_experts, device=device) + 1, torch.zeros(num_experts, dtype=torch.int64, device=device)))
    marker = marker[:num_blocks]
    expert_ids = (torch.cummax(marker, dim=0).values - 1).clamp_min(0).to(torch.int32)

    return sorted_token_ids, expert_ids, num_valid


@triton.jit
def fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_valid_tokens,
    N, K,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    top_k: tl.constexpr,
    A_DIV_TOPK: tl.constexpr,        # GEMM1: A=hidden, 行索引 = f//top_k；GEMM2: A=inter, 行索引 = f
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    sorted_tokens = tl.load(sorted_token_ids_ptr + offs_m)        # 展平索引 f
    token_mask = sorted_tokens < num_valid_tokens

    expert_id = tl.load(expert_ids_ptr + pid_m)

    if A_DIV_TOPK:
        a_row = sorted_tokens // top_k
    else:
        a_row = sorted_tokens
    a_row = tl.where(token_mask, a_row, 0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + expert_id * stride_be + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        w = tl.load(topk_weights_ptr + sorted_tokens, mask=token_mask, other=0.0)
        acc = acc * w[:, None]

    acc = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + sorted_tokens[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=token_mask[:, None] & (offs_n[None, :] < N))


def _invoke(a, b, c, topk_weights, sorted_token_ids, expert_ids, num_valid,
            top_k, a_div_topk, mul_routed_weight, block_m):
    N = b.size(1)
    K = b.size(2)
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (sorted_token_ids.numel() // block_m * triton.cdiv(N, BLOCK_N),)
    fused_moe_kernel[grid](
        a, b, c,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_valid,
        N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1),
        top_k=top_k,
        A_DIV_TOPK=a_div_topk,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


def fused_experts(
    hidden_states: torch.Tensor,   # [T, H]
    w13: torch.Tensor,             # [E, 2*I, H]  (gate_up, 已按 TP 切好)
    w2: torch.Tensor,              # [E, H, I]    (down,    已按 TP 切好)
    topk_weights: torch.Tensor,    # [T, top_k]
    topk_ids: torch.Tensor,        # [T, top_k]
    block_m: int = 32,
) -> torch.Tensor:
    """可捕获的 fused MoE：返回 [T, H]（TP 下为本 rank 的部分和，需在外层 all_reduce）。"""
    T, H = hidden_states.shape
    E, two_I, _ = w13.shape
    I = two_I // 2
    top_k = topk_ids.size(1)
    num_experts = E

    sorted_token_ids, expert_ids, num_valid = moe_align_block_size(topk_ids, block_m, num_experts)

    flat = T * top_k
    inter1 = torch.empty((flat, two_I), dtype=hidden_states.dtype, device=hidden_states.device)
    # 展平后的 routing weights，按 f=t*top_k+k 寻址
    topk_weights_flat = topk_weights.reshape(-1)

    # GEMM1: hidden[f//top_k] @ w13[e]^T  -> inter1[f]
    _invoke(hidden_states, w13, inter1, topk_weights_flat,
            sorted_token_ids, expert_ids, num_valid,
            top_k=top_k, a_div_topk=True, mul_routed_weight=False, block_m=block_m)

    # silu_and_mul: [flat, 2I] -> [flat, I]
    inter2 = torch.empty((flat, I), dtype=hidden_states.dtype, device=hidden_states.device)
    gate = inter1[:, :I]
    up = inter1[:, I:]
    inter2 = torch.nn.functional.silu(gate) * up

    # GEMM2: inter2[f] @ w2[e]^T -> out_per[f]，并乘以 routing weight
    out_per = torch.empty((flat, H), dtype=hidden_states.dtype, device=hidden_states.device)
    _invoke(inter2, w2, out_per, topk_weights_flat,
            sorted_token_ids, expert_ids, num_valid,
            top_k=top_k, a_div_topk=False, mul_routed_weight=True, block_m=block_m)

    # 按 token 归约 top_k 个贡献
    out = out_per.view(T, top_k, H).sum(dim=1)
    return out
