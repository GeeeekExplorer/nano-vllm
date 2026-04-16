"""
以 32 卡 910C + DeepSeek R1 为例，
计算并验证 scatter 打包 + burst 传输优化是否真的有收益。
"""

# ============================================================================
# 第一部分: DeepSeek R1 模型参数 (基于 DeepSeek-V3 架构)
# ============================================================================

# 模型结构
NUM_LAYERS          = 61          # transformer 层数
HIDDEN_SIZE         = 7168        # 隐藏层维度
NUM_ATTENTION_HEADS = 128         # Q 头数

# MLA 关键参数
KV_LORA_RANK        = 512         # KV 低秩压缩维度
QK_ROPE_HEAD_DIM    = 64          # RoPE 部分的 head 维度
QK_NOPE_HEAD_DIM    = 128         # 非 RoPE 部分的 head 维度
V_HEAD_DIM          = 128         # V 的 head 维度

# MLA latent 维度 (存入 KV Cache 的实际维度)
MLA_LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM   # 512 + 64 = 576

# 数据类型
DTYPE_BYTES = 2                   # BF16 = 2 bytes/element

# 对比: 如果 DeepSeek R1 用标准 MHA 而非 MLA
NUM_KV_HEADS_IF_MHA = 128         # GQA 等效 KV 头数
HEAD_DIM_IF_MHA     = 128         # head 维度


# ============================================================================
# 第二部分: 32 卡 910C 部署配置
# ============================================================================

TOTAL_CARDS         = 32          # 总卡数
NUM_NODES           = 4           # 节点数 (每节点 8 卡)
CARDS_PER_NODE      = 8

# PD 分离: 2 节点 Prefill + 2 节点 Decode
PREFILL_NODES       = 2           # 16 卡 Prefill
DECODE_NODES        = 2           # 16 卡 Decode
TP_SIZE             = 8           # 每节点内 Tensor Parallel

# 910C 芯片参数
HBM_BANDWIDTH_GBps  = 1600        # HBM 带宽 (GB/s), 910C 理论峰值
DMA_OVERHEAD_US     = 2.5         # 单次 DMA 调度固定开销 (微秒)
HCCL_LAUNCH_US      = 5.0         # 单次 HCCL send/recv 启动开销 (微秒, 含握手)
MEMORY_ALIGN        = 4 * 1024 * 1024  # 4MB 对齐


# ============================================================================
# 第三部分: 推导计算
# ============================================================================

def calc_per_token_kv_bytes_mla():
    """MLA 模式下单个 token 的 KV Cache 字节数 (单层, 单 TP rank)"""
    # MLA: latent 不按 TP 分片, 每个 rank 存完整 latent
    return MLA_LATENT_DIM * DTYPE_BYTES   # 576 * 2 = 1152 bytes

def calc_per_token_kv_bytes_mha():
    """假设标准 MHA 模式下单个 token 的 KV Cache 字节数 (单层, 单 TP rank)"""
    num_kv_heads_per_rank = NUM_KV_HEADS_IF_MHA // TP_SIZE  # 128/8 = 16
    return 2 * num_kv_heads_per_rank * HEAD_DIM_IF_MHA * DTYPE_BYTES  # 2*16*128*2 = 8192

def calc_block_bytes(block_size, per_token_bytes):
    """单个 block 单层的字节数"""
    return block_size * per_token_bytes

def calc_transfer_metrics(prompt_tokens, block_size, per_token_bytes, label):
    """
    计算一个请求的 KV Cache 传输指标。

    传输粒度: 每个 block 的每一层是一次独立的 DMA/HCCL 操作。
    (因为各层的 cache tensor 在 HBM 中不连续)
    """
    num_blocks = prompt_tokens // block_size
    block_bytes = calc_block_bytes(block_size, per_token_bytes)

    # 总传输量
    total_bytes = num_blocks * NUM_LAYERS * block_bytes
    total_mb = total_bytes / (1024 * 1024)

    # --- 逐块传输 (Naive) ---
    num_dma_ops = num_blocks * NUM_LAYERS  # 每 block 每层一次 DMA
    transfer_time_us = total_bytes / (HBM_BANDWIDTH_GBps * 1e3)  # GB/s → bytes/us
    scheduling_us = num_dma_ops * DMA_OVERHEAD_US
    hccl_launch_us = num_dma_ops * HCCL_LAUNCH_US
    naive_total_us = transfer_time_us + scheduling_us + hccl_launch_us

    # --- scatter+burst ---
    # scatter: HBM 内拷贝 (将离散 block 打包到连续 buffer)
    scatter_us = total_bytes / (HBM_BANDWIDTH_GBps * 1e3)
    # burst: 1 次大块传输
    burst_transfer_us = total_bytes / (HBM_BANDWIDTH_GBps * 1e3)
    burst_overhead_us = DMA_OVERHEAD_US + HCCL_LAUNCH_US  # 仅 1 次调度
    # unscatter: 接收端拆包
    unscatter_us = scatter_us  # 与 scatter 对称
    sb_total_us = scatter_us + burst_transfer_us + burst_overhead_us + unscatter_us

    speedup = naive_total_us / sb_total_us if sb_total_us > 0 else 0

    return {
        "label": label,
        "prompt_tokens": prompt_tokens,
        "block_size": block_size,
        "block_bytes": block_bytes,
        "num_blocks": num_blocks,
        "num_dma_ops": num_dma_ops,
        "total_mb": total_mb,
        "transfer_us": transfer_time_us,
        "scheduling_us": scheduling_us,
        "hccl_launch_us": hccl_launch_us,
        "naive_total_us": naive_total_us,
        "scatter_us": scatter_us,
        "burst_transfer_us": burst_transfer_us,
        "burst_overhead_us": burst_overhead_us,
        "unscatter_us": unscatter_us,
        "sb_total_us": sb_total_us,
        "speedup": speedup,
        "scheduling_pct": scheduling_us / naive_total_us * 100 if naive_total_us > 0 else 0,
    }


# ============================================================================
# 第四部分: 数值验证
# ============================================================================

if __name__ == "__main__":
    mla_per_token = calc_per_token_kv_bytes_mla()
    mha_per_token = calc_per_token_kv_bytes_mha()

    print("=" * 90)
    print("DeepSeek R1 (671B MoE) on 32x 910C — KV Cache 传输分析")
    print("=" * 90)
    print()
    print(f"模型: DeepSeek R1, {NUM_LAYERS} layers, MLA latent_dim={MLA_LATENT_DIM}")
    print(f"部署: {TOTAL_CARDS} 卡 910C, {PREFILL_NODES}节点 Prefill + {DECODE_NODES}节点 Decode, TP={TP_SIZE}")
    print(f"芯片: HBM BW={HBM_BANDWIDTH_GBps} GB/s, DMA开销={DMA_OVERHEAD_US}μs/次, HCCL开销={HCCL_LAUNCH_US}μs/次")
    print()

    # --- 基础数据 ---
    print("─" * 90)
    print("▎基础数据: 每 token 每层 KV Cache 大小 (单 TP rank)")
    print("─" * 90)
    print(f"  MLA (实际):  {mla_per_token:>8,} bytes  ({MLA_LATENT_DIM} dims × {DTYPE_BYTES}B)")
    print(f"  MHA (假设):  {mha_per_token:>8,} bytes  (2 × {NUM_KV_HEADS_IF_MHA//TP_SIZE} heads × {HEAD_DIM_IF_MHA} dim × {DTYPE_BYTES}B)")
    print(f"  MLA 压缩比:  {mha_per_token / mla_per_token:.1f}x")
    print()

    # --- 不同 block_size 和 prompt 长度的分析 ---
    configs = [
        # (prompt_tokens, block_size)
        (2048,  16),
        (2048,  128),
        (2048,  256),
        (8192,  16),
        (8192,  128),
        (8192,  256),
        (32768, 16),
        (32768, 128),
        (32768, 256),
    ]

    print("─" * 90)
    print("▎MLA 模式 (DeepSeek R1 实际场景): 单请求 KV Cache 传输对比")
    print("─" * 90)
    print()
    print(f"{'prompt':>7} {'blk_sz':>6} │ {'blk字节':>8} {'块数':>5} {'DMA次数':>7} │"
          f" {'总数据':>8} {'朴素耗时':>10} {'SB耗时':>10} {'加速比':>6} │ {'调度占比':>8}")
    print("─" * 7 + " " + "─" * 6 + "─┼─" + "─" * 8 + " " + "─" * 5 + " " + "─" * 7 + "─┼─"
          + "─" * 8 + " " + "─" * 10 + " " + "─" * 10 + " " + "─" * 6 + "─┼─" + "─" * 8)

    results_mla = []
    for prompt_tokens, block_size in configs:
        r = calc_transfer_metrics(prompt_tokens, block_size, mla_per_token, "MLA")
        results_mla.append(r)

        naive_str = f"{r['naive_total_us']:,.0f}μs" if r['naive_total_us'] < 1e6 else f"{r['naive_total_us']/1e3:,.1f}ms"
        sb_str = f"{r['sb_total_us']:,.0f}μs" if r['sb_total_us'] < 1e6 else f"{r['sb_total_us']/1e3:,.1f}ms"

        print(f"{r['prompt_tokens']:>7,} {r['block_size']:>6} │ "
              f"{r['block_bytes']:>7,}B {r['num_blocks']:>5,} {r['num_dma_ops']:>7,} │ "
              f"{r['total_mb']:>7.1f}MB {naive_str:>10} {sb_str:>10} {r['speedup']:>5.1f}x │ "
              f"{r['scheduling_pct']:>7.1f}%")

    print()
    print("─" * 90)
    print("▎对比: 如果 DeepSeek R1 不用 MLA 而用标准 MHA (假设场景)")
    print("─" * 90)
    print()
    print(f"{'prompt':>7} {'blk_sz':>6} │ {'blk字节':>8} {'块数':>5} {'DMA次数':>7} │"
          f" {'总数据':>8} {'朴素耗时':>10} {'SB耗时':>10} {'加速比':>6} │ {'调度占比':>8}")
    print("─" * 7 + " " + "─" * 6 + "─┼─" + "─" * 8 + " " + "─" * 5 + " " + "─" * 7 + "─┼─"
          + "─" * 8 + " " + "─" * 10 + " " + "─" * 10 + " " + "─" * 6 + "─┼─" + "─" * 8)

    for prompt_tokens, block_size in configs:
        r = calc_transfer_metrics(prompt_tokens, block_size, mha_per_token, "MHA")

        naive_str = f"{r['naive_total_us']:,.0f}μs" if r['naive_total_us'] < 1e6 else f"{r['naive_total_us']/1e3:,.1f}ms"
        sb_str = f"{r['sb_total_us']:,.0f}μs" if r['sb_total_us'] < 1e6 else f"{r['sb_total_us']/1e3:,.1f}ms"

        print(f"{r['prompt_tokens']:>7,} {r['block_size']:>6} │ "
              f"{r['block_bytes']:>7,}B {r['num_blocks']:>5,} {r['num_dma_ops']:>7,} │ "
              f"{r['total_mb']:>7.1f}MB {naive_str:>10} {sb_str:>10} {r['speedup']:>5.1f}x │ "
              f"{r['scheduling_pct']:>7.1f}%")

    # --- 关键结论 ---
    print()
    print("=" * 90)
    print("▎关键结论")
    print("=" * 90)
    print()

    # 找到 MLA block_size=16 prompt=8192 的结果作为典型场景
    typical = [r for r in results_mla if r["block_size"] == 16 and r["prompt_tokens"] == 8192][0]

    print(f"  典型场景: DeepSeek R1, prompt=8192, block_size=16, MLA")
    print(f"  ├─ 单 block 单层数据量: {typical['block_bytes']:,} bytes ({typical['block_bytes']/1024:.1f} KB)")
    print(f"  ├─ DMA 操作次数: {typical['num_dma_ops']:,} 次")
    print(f"  ├─ 总传输数据量: {typical['total_mb']:.1f} MB")
    print(f"  ├─ 调度+启动开销占朴素总耗时: {typical['scheduling_pct']:.1f}%")
    print(f"  ├─ 朴素传输耗时: {typical['naive_total_us']:,.0f} μs ({typical['naive_total_us']/1e3:.1f} ms)")
    print(f"  ├─ scatter+burst 耗时: {typical['sb_total_us']:,.0f} μs ({typical['sb_total_us']/1e3:.2f} ms)")
    print(f"  └─ 加速比: {typical['speedup']:.1f}x")
    print()

    # 找到哪种配置加速比最低
    min_r = min(results_mla, key=lambda r: r["speedup"])
    print(f"  最小加速场景: prompt={min_r['prompt_tokens']}, block_size={min_r['block_size']}")
    print(f"  ├─ block 数据量: {min_r['block_bytes']:,} bytes ({min_r['block_bytes']/1024:.1f} KB)")
    print(f"  ├─ 调度占比: {min_r['scheduling_pct']:.1f}%")
    print(f"  └─ 加速比: {min_r['speedup']:.1f}x")
    print()

    max_r = max(results_mla, key=lambda r: r["speedup"])
    print(f"  最大加速场景: prompt={max_r['prompt_tokens']}, block_size={max_r['block_size']}")
    print(f"  ├─ block 数据量: {max_r['block_bytes']:,} bytes ({max_r['block_bytes']/1024:.1f} KB)")
    print(f"  ├─ 调度占比: {max_r['scheduling_pct']:.1f}%")
    print(f"  └─ 加速比: {max_r['speedup']:.1f}x")
    print()

    # 判断: 什么时候不值得
    print("─" * 90)
    print("▎scatter+burst 是否值得？")
    print("─" * 90)
    no_benefit = [r for r in results_mla if r["speedup"] < 2.0]
    if no_benefit:
        print(f"  加速比 < 2x 的场景 ({len(no_benefit)} 个):")
        for r in no_benefit:
            print(f"    prompt={r['prompt_tokens']}, block_size={r['block_size']}: "
                  f"加速比={r['speedup']:.1f}x, block={r['block_bytes']/1024:.0f}KB, 调度占比={r['scheduling_pct']:.1f}%")
    else:
        print("  所有 MLA 场景加速比均 ≥ 2x，scatter+burst 始终有收益")
    print()

    big_benefit = [r for r in results_mla if r["speedup"] >= 10.0]
    if big_benefit:
        print(f"  加速比 ≥ 10x 的场景 ({len(big_benefit)} 个):")
        for r in big_benefit:
            print(f"    prompt={r['prompt_tokens']}, block_size={r['block_size']}: "
                  f"加速比={r['speedup']:.1f}x, block={r['block_bytes']/1024:.0f}KB, DMA次数={r['num_dma_ops']:,}")

    print()
    print("─" * 90)
    print("▎MLA 的反直觉效应: 压缩 KV 反而让 scatter+burst 更必要")
    print("─" * 90)
    # MLA 把每 token KV 从 8192B 压缩到 1152B，block 变小了
    # block 越小 → 单次 DMA 数据越少 → 调度开销占比越高 → scatter+burst 收益越大
    for bs in [16, 128, 256]:
        mla_block = calc_block_bytes(bs, mla_per_token)
        mha_block = calc_block_bytes(bs, mha_per_token)
        print(f"  block_size={bs:>3}: MLA block={mla_block/1024:>6.1f}KB, MHA block={mha_block/1024:>6.1f}KB, "
              f"MLA是MHA的 {mla_block/mha_block*100:.1f}%")
    print()
    print('  -> MLA 的 KV 压缩使得单 block 数据量缩小到 MHA 的 ~14%')
    print('  -> 单 block 越小, 910C 上每次 DMA 的 "启动开销/有效载荷" 比越大')
    print('  -> 因此 MLA 模型比 MHA 模型更需要 scatter+burst 优化')
    print('  -> 这是 MLA 压缩带来的 "副作用" -- 压缩了数据量, 却放大了调度瓶颈')
