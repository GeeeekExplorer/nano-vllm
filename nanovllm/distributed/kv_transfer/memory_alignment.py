"""
难点一: 昇腾硬件 4MB 内存对齐约束。

问题:
    昇腾 NPU 要求 KV Cache 必须满足 4MB 边界对齐。GPU 上 KV Cache 通常以单个
    连续 tensor 存储 (shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim])，
    但在昇腾上 K 和 V 必须拆开成独立的对齐 tensor（tuple 存储），且每块内存的
    起始地址和大小都必须是 4MB 的整数倍。

解决方案:
    1. align_to_4mb(): 将任意字节大小向上对齐到 4MB 边界
    2. allocate_aligned_kv_cache(): 替代原始的单 tensor 分配，按对齐要求分别分配 K/V
    3. extract_aligned_kv_pointers(): 从对齐后的 KV Cache 中提取指针和大小，用于后端注册
    4. build_kv_cache_dict_for_register(): 构建适配分布式传输的 KV Cache 字典
"""

import torch

# 4MB = 4 * 1024 * 1024
_4MB = 4 * 1024 * 1024


def align_to_4mb(size_in_bytes: int) -> int:
    """
    将字节大小向上对齐到 4MB 边界。

    输入:
        size_in_bytes: int - 原始字节大小

    输出:
        int - 对齐后的字节大小（>= 原始大小，且是 4MB 的整数倍）

    执行逻辑:
        # 使用整数向上取整公式: (n + alignment - 1) // alignment * alignment
        # 例如: 3MB → 4MB, 5MB → 8MB, 4MB → 4MB
    """
    return (size_in_bytes + _4MB - 1) // _4MB * _4MB


def check_alignment(tensor: torch.Tensor) -> bool:
    """
    检查一个 tensor 的设备内存地址是否满足 4MB 对齐。

    输入:
        tensor: torch.Tensor - 待检查的设备 tensor

    输出:
        bool - True 如果起始地址是 4MB 对齐的

    执行逻辑:
        # return tensor.data_ptr() % _4MB == 0
    """
    return tensor.data_ptr() % _4MB == 0


def allocate_aligned_kv_cache(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    按昇腾 4MB 对齐要求分别分配 K Cache 和 V Cache。

    与 GPU 上的单 tensor 分配（shape: [2, num_layers, ...]）不同，
    昇腾需要每一层的 K/V 独立对齐分配。

    输入:
        num_layers: int - 模型层数
        num_blocks: int - KV Cache block 总数
        block_size: int - 每个 block 包含的 token 数
        num_kv_heads: int - 每个 TP rank 的 KV head 数
        head_dim: int - 每个 head 的维度
        dtype: torch.dtype - 数据类型

    输出:
        tuple[list[torch.Tensor], list[torch.Tensor]]:
            - k_caches: 每层一个 K Cache tensor, shape [num_blocks, block_size, num_kv_heads, head_dim]
            - v_caches: 每层一个 V Cache tensor, 同上

    执行逻辑:
        # 原始方式 (GPU, 单 tensor):
        #   kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        #   K 和 V 的内存是连续的，无对齐保证
        #
        # 对齐方式 (昇腾 NPU, tuple 存储):
        #   1. 计算单层单块的原始字节大小:
        #      raw_block_bytes = block_size * num_kv_heads * head_dim * dtype.itemsize
        #   2. 对齐到 4MB:
        #      aligned_block_bytes = align_to_4mb(raw_block_bytes)
        #   3. 计算对齐后每层需要的元素数（用 padding 填充对齐间隙）:
        #      aligned_elements_per_block = aligned_block_bytes // dtype.itemsize
        #   4. 为每一层独立分配 K 和 V:
        #      for layer in range(num_layers):
        #          k_caches.append(torch.empty(num_blocks, aligned_elements_per_block, dtype=dtype, device="npu"))
        #          v_caches.append(torch.empty(num_blocks, aligned_elements_per_block, dtype=dtype, device="npu"))
        #   5. 验证对齐:
        #      for t in k_caches + v_caches:
        #          assert check_alignment(t), "KV Cache 未满足 4MB 对齐"
        #
        # 注意: 对齐后的 tensor 实际可用部分仍为 [num_blocks, block_size, num_kv_heads, head_dim]，
        #       多出的 padding 区域不会被使用，但确保了每个 block 起始地址满足硬件要求。
    """
    k_caches = []
    v_caches = []
    return k_caches, v_caches


def extract_aligned_kv_pointers(
    k_caches: list[torch.Tensor],
    v_caches: list[torch.Tensor],
    num_blocks: int,
    block_size: int,
) -> tuple[list[int], list[int]]:
    """
    从对齐后的 KV Cache 中提取设备内存指针和字节大小，用于后端 register_buffer。

    输入:
        k_caches: list[torch.Tensor] - 每层的 K Cache tensor 列表
        v_caches: list[torch.Tensor] - 每层的 V Cache tensor 列表
        num_blocks: int - block 总数
        block_size: int - 每 block 的 token 数

    输出:
        tuple[list[int], list[int]]:
            - ptrs: 所有 KV Cache tensor 的设备内存起始指针列表
            - lengths: 每个 tensor 的字节大小列表

    执行逻辑:
        # ptrs = []
        # lengths = []
        # for cache in k_caches + v_caches:
        #     ptrs.append(cache.data_ptr())
        #     lengths.append(cache.numel() * cache.element_size())
        # return ptrs, lengths
        #
        # 这些指针和大小将传递给 Backend.register_buffer()，后端通过 RDMA/DMA
        # 在这些已对齐的内存地址上进行零拷贝传输。
    """
    ptrs = []
    lengths = []
    return ptrs, lengths


def build_kv_cache_dict_for_register(
    k_caches: list[torch.Tensor],
    v_caches: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    构建供 KVPoolWorker.register_kv_caches() 使用的字典格式。

    输入:
        k_caches: list[torch.Tensor] - 每层的 K Cache
        v_caches: list[torch.Tensor] - 每层的 V Cache

    输出:
        dict[str, torch.Tensor] - key 为 "layer.{i}.k" 或 "layer.{i}.v"

    执行逻辑:
        # kv_dict = {}
        # for i, (k, v) in enumerate(zip(k_caches, v_caches)):
        #     kv_dict[f"layer.{i}.k"] = k
        #     kv_dict[f"layer.{i}.v"] = v
        # return kv_dict
    """
    kv_dict = {}
    for i, (k, v) in enumerate(zip(k_caches, v_caches)):
        kv_dict[f"layer.{i}.k"] = k
        kv_dict[f"layer.{i}.v"] = v
    return kv_dict


def compute_aligned_block_bytes(
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    itemsize: int,
    is_mla: bool = False,
    mla_latent_dim: int = 0,
) -> int:
    """
    计算对齐后单个 KV block 的字节大小。

    输入:
        block_size: int - 每 block 的 token 数
        num_kv_heads: int - KV head 数
        head_dim: int - head 维度
        itemsize: int - 每个元素的字节数
        is_mla: bool - 是否为 MLA（Multi-head Latent Attention）模型
        mla_latent_dim: int - MLA 的 latent 维度（仅 is_mla=True 时使用）

    输出:
        int - 对齐后的 block 字节大小

    执行逻辑:
        # 标准 MHA 模型:
        #   raw_bytes = block_size * num_kv_heads * head_dim * itemsize
        #
        # MLA 模型 (如 DeepSeek):
        #   MLA 的 KV Cache 格式不同于标准 MHA:
        #   - 标准 MHA: K=[block_size, num_kv_heads, head_dim], V 同
        #   - MLA: latent=[block_size, mla_latent_dim]（K 和 V 共享压缩表示）
        #   raw_bytes = block_size * mla_latent_dim * itemsize
        #
        # 最后对齐: return align_to_4mb(raw_bytes)
    """
    if is_mla and mla_latent_dim > 0:
        raw_bytes = block_size * mla_latent_dim * itemsize
    else:
        raw_bytes = block_size * num_kv_heads * head_dim * itemsize
    return align_to_4mb(raw_bytes)
