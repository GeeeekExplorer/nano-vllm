"""昇腾 HCCL (Huawei Collective Communication Library) 工具函数。

HCCL 是昇腾 NPU 上替代 NCCL 的集合通信库，
用于多卡/多机之间的张量通信（AllGather, ReduceScatter 等）。
在 MLA + DP 并行场景下，用于跨 EngineCore 实例同步 KV Cache 元数据。
"""

import torch


def hccl_init_comm_group(
    world_size: int,
    rank: int,
    group_name: str = "default",
) -> object:
    """初始化 HCCL 通信组。

    Args:
        world_size: 通信组中的总进程数
        rank:       当前进程在通信组中的序号
        group_name: 通信组名称，用于区分不同用途的通信组
                    如 "dp_group"（数据并行）、"kv_transfer_group"（KV 传输）

    Returns:
        comm_group: HCCL 通信组句柄，供后续通信操作使用
    """
    # --- 执行逻辑 ---
    # 1. 调用 torch_npu.distributed 接口初始化进程组：
    #    torch.distributed.init_process_group(
    #        backend="hccl", world_size=world_size, rank=rank)
    # 2. 创建命名子组：
    #    group = torch.distributed.new_group(ranks=list(range(world_size)), backend="hccl")
    # 3. 验证 HCCL 链路连通性（ping-pong 测试）
    # 4. 返回通信组句柄
    return None


def hccl_all_gather(
    input_tensor: torch.Tensor,
    comm_group: object,
    world_size: int,
) -> torch.Tensor:
    """通过 HCCL 执行 AllGather 操作。

    在 MLA 中，fused_qkv_a_proj 的输出 (q_c, kv_no_split)
    可能需要跨 TP rank 做 AllGather 以获取完整的隐向量。

    Args:
        input_tensor: 当前 rank 的局部张量，shape = [num_tokens, local_dim]
        comm_group:   HCCL 通信组句柄
        world_size:   通信组大小

    Returns:
        gathered: AllGather 后的完整张量，shape = [num_tokens, local_dim * world_size]
    """
    # --- 执行逻辑 ---
    # 1. 分配输出 buffer：shape = [num_tokens, local_dim * world_size]
    # 2. 调用 HCCL AllGather：
    #    torch.distributed.all_gather_into_tensor(
    #        output, input_tensor, group=comm_group)
    # 3. 返回 gathered 张量
    return input_tensor


def hccl_reduce_scatter(
    input_tensor: torch.Tensor,
    comm_group: object,
    world_size: int,
) -> torch.Tensor:
    """通过 HCCL 执行 ReduceScatter 操作。

    MLA 的 output projection 后需要跨 TP rank 做 ReduceScatter。

    Args:
        input_tensor: 当前 rank 的局部结果，shape = [num_tokens, total_dim]
        comm_group:   HCCL 通信组句柄
        world_size:   通信组大小

    Returns:
        reduced: ReduceScatter 后的张量，shape = [num_tokens, total_dim // world_size]
    """
    # --- 执行逻辑 ---
    # 1. 分配输出 buffer：shape = [num_tokens, total_dim // world_size]
    # 2. 调用 HCCL ReduceScatter（求和归约）：
    #    torch.distributed.reduce_scatter_tensor(
    #        output, input_tensor, op=ReduceOp.SUM, group=comm_group)
    # 3. 返回 reduced 张量
    return input_tensor
