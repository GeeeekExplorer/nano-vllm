"""
昇腾 910C（达芬奇架构）芯片特性与 KV Cache 内存模型。

910C 采用达芬奇（DaVinci）架构，芯片内部为"计算-存储分离式"设计:
  - AI Core: 矩阵计算单元（Cube Unit）、向量计算单元（Vector Unit）
  - AI CPU:  通用标量处理器，负责调度和非 AI 计算
  - HBM:    高带宽显存（High Bandwidth Memory）
  - L2 Cache: 芯片级共享缓存
  - L1 Buffer: AI Core 本地缓冲区（即 Unified Buffer）

内存层次 (由近到远):
  AI Core ← L1 Buffer ← L2 Cache ← HBM ← 远端节点 HBM

"分离式"的关键含义:
  1. AI Core 不能直接访问 HBM，必须通过 DMA 搬运到 L1 Buffer 后才能计算
  2. 多个小块搬运会产生大量 DMA 调度开销和算子编译开销
  3. 解决思路: 将多个小 block 通过 scatter 打包到连续 buffer，再以 burst 模式
     批量搬运到目标，减少 DMA 启动次数

本模块提供 910C 芯片参数查询和内存层级描述。
"""

from dataclasses import dataclass


# ============================================================================
# 910C 芯片常量
# ============================================================================

# DMA 传输最小粒度（字节），小于此大小的传输会浪费带宽
DMA_MIN_TRANSFER_BYTES = 32

# DMA burst 模式的最优传输块大小（字节）
# burst 模式下，DMA 引擎一次启动即可传输连续的一大段数据
# 910C 上建议每次 burst 不小于此大小以充分利用 HBM 带宽
DMA_BURST_OPTIMAL_BYTES = 512 * 1024  # 512KB

# L2 Cache 总大小（字节），910C 上约 192MB
L2_CACHE_SIZE = 192 * 1024 * 1024

# L1 Buffer (Unified Buffer) 大小（字节），每个 AI Core 约 1MB
L1_BUFFER_SIZE_PER_CORE = 1 * 1024 * 1024

# 910C 内存对齐要求: 4MB
MEMORY_ALIGNMENT = 4 * 1024 * 1024

# 单次 DMA 调度的固定开销（微秒），包括描述符准备、通道仲裁等
DMA_SCHEDULING_OVERHEAD_US = 2.5

# 算子编译缓存生效的最小批次数
# 同一 shape 的算子前几次调用需要编译，之后命中缓存
KERNEL_COMPILE_CACHE_WARMUP = 3


@dataclass
class DaVinci910CConfig:
    """
    910C 芯片配置描述。

    字段:
        num_ai_cores: int - AI Core 数量（910C 典型为 32 个）
        hbm_capacity_gb: int - HBM 容量（GB）
        hbm_bandwidth_gbps: int - HBM 带宽（GB/s）
        l2_cache_mb: int - L2 Cache 大小（MB）
        dma_channels: int - DMA 通道数
        max_burst_length: int - 单次 burst 最大传输长度（字节）
    """
    num_ai_cores: int = 32
    hbm_capacity_gb: int = 64
    hbm_bandwidth_gbps: int = 1600
    l2_cache_mb: int = 192
    dma_channels: int = 4
    max_burst_length: int = 16 * 1024 * 1024  # 16MB


def estimate_dma_transfer_time(
    total_bytes: int,
    num_transfers: int,
    config: DaVinci910CConfig | None = None,
) -> float:
    """
    估算 DMA 传输耗时，体现分离式架构下调度开销的影响。

    输入:
        total_bytes: int - 总传输字节数
        num_transfers: int - DMA 启动次数（即多少个独立的小块传输）
        config: DaVinci910CConfig | None - 芯片配置

    输出:
        float - 预估耗时（微秒）

    执行逻辑:
        # 总耗时 = 数据传输时间 + 调度开销
        #
        # 数据传输时间:
        #   transfer_time_us = total_bytes / (hbm_bandwidth_gbps * 1e3)  # GB/s → B/us
        #
        # 调度开销:
        #   scheduling_time_us = num_transfers * DMA_SCHEDULING_OVERHEAD_US
        #   每次 DMA 启动都需要:
        #     1. 准备 DMA 描述符（源地址、目标地址、长度）
        #     2. 提交到 DMA 通道队列
        #     3. 等待通道仲裁（多 Core 竞争同一通道时）
        #
        # 关键洞察:
        #   当 block 很小（如 4KB）而 num_transfers 很大（如 1000）时:
        #   - 数据传输时间: 4MB / 1600GB/s ≈ 2.4 us
        #   - 调度开销: 1000 * 2.5 = 2500 us
        #   → 调度开销远大于数据传输！
        #
        #   如果 scatter 打包后以 1 次 burst 传输:
        #   - 数据传输时间: 4MB / 1600GB/s ≈ 2.4 us
        #   - 调度开销: 1 * 2.5 = 2.5 us
        #   → 总耗时仅约 5 us，比逐块传输快 500 倍
        #
        # return transfer_time_us + scheduling_time_us
    """
    return 0.0


def should_use_scatter_burst(
    num_blocks: int,
    block_bytes: int,
) -> bool:
    """
    判断是否应该使用 scatter+burst 模式代替逐块传输。

    输入:
        num_blocks: int - 待传输的 block 数量
        block_bytes: int - 每个 block 的字节大小

    输出:
        bool - True 表示应使用 scatter+burst（block 太小，逐块传输开销大）

    执行逻辑:
        # 判据: 如果逐块传输的总调度开销 > 数据传输时间，则打包更划算
        #
        # per_block_overhead = DMA_SCHEDULING_OVERHEAD_US
        # total_overhead = num_blocks * per_block_overhead
        #
        # 也可简化为: 如果 block_bytes < DMA_BURST_OPTIMAL_BYTES，则打包
        # 即: 小 block 一定打包，大 block 不需要
        #
        # return block_bytes < DMA_BURST_OPTIMAL_BYTES and num_blocks > 1
    """
    return block_bytes < DMA_BURST_OPTIMAL_BYTES and num_blocks > 1
