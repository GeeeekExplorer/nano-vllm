import nanovllm.layers.fused_moe.modular_kernel as mk
import torch
import torch.nn.functional as F
import triton.language as tl
import triton
from nanovllm.utils.math_utils import round_up
from nanovllm.layers.activation import SiluAndMul
@triton.jit
def fused_moe_triton_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    # 矩阵维度
    N,
    K,
    EM,
    num_valid_tokens,
    # stride
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    topk: tl.constexpr,
    compute_type: tl.constexpr,
):
    # 计算位置
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(GROUP_SIZE_M, num_pid_m - first_pid_m)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # 划分数据
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_pad_ptr)
    
    # 如果pid_m对应的token起始位置超过了num_tokens_post_padded, 说明这个block没有有效token, 直接返回
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    # 将token_id为pad_token_id的token mask掉, 这些token是没有对应专家的, 也就是无效token
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    a_ptrs = a_ptr + (
        offs_token[:, None] // topk * stride_am + offs_k[None, :] * stride_ak
    )
    # b.shape(E, K, N)
    b_ptrs = b_ptr + (
        off_experts * stride_be + 
        (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs, 
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token, 
            mask=token_mask,
            other=0,
        )
        accumulator = accumulator * moe_weight[:, None]
    
    accumulator = accumulator.to(compute_type)
    # C.shape(topk, M, N)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator, mask=token_mask[:, None] & (offs_cn[None, :] < N))


def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,                        # (M, K)
    B: torch.Tensor,                        # (E, K, N)
    C: torch.Tensor,                        # (topk * M, N)
    topk_weights: torch.Tensor | None,      # (M, topk)
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor | None,
    num_tokens_post_pad: torch.Tensor,
    mul_routed_weight: bool,
    topk: int,
    block_size: int,
):
    assert topk_weights is not None or not mul_routed_weight, "topk_weights must be provided if mul_routed_weight is True"
    if sorted_token_ids is None or expert_ids is None:
        raise ValueError("sorted_token_ids and expert_ids must be provided")

    M, K = A.size(0), A.size(1)
    num_slots = M * topk
    BLOCK_SIZE_M = block_size
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 16
    EM = sorted_token_ids.size(0)

    if A.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif A.dtype == torch.float16:
        compute_type = tl.float16
    elif A.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {A.dtype}")

    grid = lambda META: (
        triton.cdiv(EM, BLOCK_SIZE_M) * triton.cdiv(B.size(2), BLOCK_SIZE_N),
    )
    fused_moe_triton_kernel[grid](
        a_ptr=A,
        b_ptr=B,
        c_ptr=C,
        topk_weights_ptr=topk_weights,
        sorted_token_ids_ptr=sorted_token_ids,
        expert_ids_ptr=expert_ids,
        num_tokens_post_pad_ptr=num_tokens_post_pad,
        N=B.size(2),
        K=K,
        EM=EM,
        num_valid_tokens=num_slots,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_be=B.stride(0),
        stride_bk=B.stride(1),
        stride_bn=B.stride(2),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=4,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        topk=topk,
        compute_type=compute_type,
    )



def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        参数：
            topk_ids: (num_tokens, top_k)
            block_size: int, triton内核的block size, 决定每次用HBM -> SRAM取出的数据个数
            num_experts: int
        Example:
            topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]]
            block_size = 4 并且 num_experts = 4:
            1. 将topk_ids展平为[2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3], 一共12个数
            2. 按照专家编号排序得到[3, 6, 9, 0, 4, 10, 1, 7, 11, 2, 5, 8] -> [expert1, expert2, expert3, expert4]
            3. 对每个专家的token数量进行padding(12), 使得每个专家的token数量都是block_size的整数倍 -> [4, 4, 4, 4]
            4. 得到排序后的token_id和对应的专家id:
                [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12]
    """
    # 最坏情况下会有多少token-expert对(进行padding后)
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    # 这个变量用来记录padding后的token-expert数量
    num_tokens_post_pad = torch.empty(
        (1,), dtype=torch.int32, device=topk_ids.device
    )
    num_token_slots = topk_ids.numel()
    pad_token_id = num_token_slots
    flatten_topk_ids = topk_ids.flatten()
    sorted_token_list: list[int] = []
    block_expert_list: list[int] = []
    for expert_id in range(num_experts):
        slot_indices = (flatten_topk_ids == expert_id).nonzero(as_tuple=False).flatten()
        slot_list = slot_indices.tolist()
        real_count = len(slot_list)
        padded_count = round_up(real_count, block_size) if real_count > 0 else 0
        pad_count = padded_count - real_count
        if pad_count > 0:
            slot_list.extend([pad_token_id] * pad_count)
        sorted_token_list.extend(slot_list)
        num_blocks_for_this_expert = padded_count // block_size
        block_expert_list.extend([expert_id] * num_blocks_for_this_expert)
    
    # 得到sort_ids
    num_tokens_post_pad[0] = len(sorted_token_list)
    if num_tokens_post_pad[0] > 0:
        sorted_ids[:num_tokens_post_pad[0]] = torch.tensor(
            sorted_token_list, dtype=torch.int32, device=topk_ids.device
        )
    if num_tokens_post_pad[0] < max_num_tokens_padded:
        sorted_ids[num_tokens_post_pad[0]:] = pad_token_id
    
    # 得到expert_ids
    actual_num_blocks = len(block_expert_list)
    if actual_num_blocks > 0:
        expert_ids[:actual_num_blocks] = torch.tensor(
            block_expert_list, dtype=torch.int32, device=topk_ids.device
        )
    if actual_num_blocks < max_num_m_blocks:
        expert_ids[actual_num_blocks:] = -1
        
    return sorted_ids, expert_ids, num_tokens_post_pad




class TritonExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
    ):
        self.BLOCK_SIZE = 64
        self.act_fn = SiluAndMul()

    def apply(
        self,
        output: torch.Tensor,           
        hidden_states: torch.Tensor,    
        w1: torch.Tensor,               
        w2: torch.Tensor,               
        topk_weights: torch.Tensor,     
        topk_ids: torch.Tensor,         
        activation: str,
        global_num_experts: int,
    ):
        """
            output: 输出结果, shape(num_tokens, hidden_size)
            hidden_states: shape(num_tokens, hidden_size)
            w1: shape(num_experts, hidden_size, 2 * intermediate_size)
            w2: shape(num_experts, intermediate_size, hidden_size)
            topk_weights: shape(num_tokens, top_k)
            topk_ids: shape(num_tokens, top_k)
        """
        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert topk_weights.is_contiguous(), "topk_weights must be contiguous"
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"

        num_experts, num_tokens, intermediate_size, hidden_size, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        
        assert topk_weights.shape == topk_ids.shape
        assert num_experts == global_num_experts
        assert w1.size(2) == 2 * intermediate_size
        assert w2.size(2) == hidden_size
        if activation != "silu":
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 按照expert 重排 token
        sorted_token_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
            topk_ids=topk_ids,
            block_size=self.BLOCK_SIZE,  
            num_experts=global_num_experts,
        )
        
        # 第一层expert matual: xw1
        gate_up = torch.empty(
            (num_tokens * top_k_num, 2 * intermediate_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        
        # C = A @ B --> (M, K) @ (E, K, N) -> (M * topk, N)
        invoke_fused_moe_triton_kernel(
            A=hidden_states,
            B=w1,
            C=gate_up,
            topk_weights=None,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_pad=num_tokens_post_pad,
            mul_routed_weight=False,
            topk=top_k_num,
            block_size=self.BLOCK_SIZE,
        )
        
        # 激活函数
        activated = self.act_fn(gate_up)
        
        # 第二层expert matmul: xw2
        expert_out = torch.empty(
            (num_tokens * top_k_num, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    
        # C = A @ B --> (M * topk, N) @ (E, N, K) -> (M * topk, K)
        invoke_fused_moe_triton_kernel(
            A=activated,
            B=w2,
            C=expert_out,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_pad=num_tokens_post_pad,
            mul_routed_weight=True,
            topk=1,
            block_size=self.BLOCK_SIZE,
        )
        
        
        output.copy_(
            expert_out.view(num_tokens, top_k_num, hidden_size).sum(dim=1)
        )
        return output
