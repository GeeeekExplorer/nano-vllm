# decode_pipeline_case_study.py 输出解读

运行命令：
```
python decode_pipeline_case_study.py --prefill-gpu 5 --decode-gpu 6 --sm-splits 18:14
```

## 总览
- **Small decode**：2 个提示，`max_tokens=24`
- **Large decode**：6 个提示，`max_tokens=192`
- Green Context（`cuGreenCtxStreamCreate`）在当前驱动不可用，因此自动回退到串行执行，但仍会收集所有 A→F 通信统计与计时。

## Small decode
- 批量 token：48 个。
- 平均 Attention 时间：34.90 ms / token。
- 平均 FFN 时间：6.50 ms / token。
- 总延迟：41.40 ms / token。
- 平均 A→F 传输：0.1094 MiB / token。
- 每层 payload：全部 28 层均 ~0.0039 MiB（Attention 输出与 Residual 各 ~0.0020 MiB）。
- 最近一次 batch（batch_size=2，dtype=bfloat16）需要 0.2188 MiB。

**结论**：小批次下流水线开销占主导，即便失去 Green Context 也能看到串行路径的瓶颈位置（Attention ~5x FFN）。

## Large decode
- 批量 token：1152 个。
- 平均 Attention 时间：4.66 ms / token。
- 平均 FFN 时间：1.52 ms / token。
- 总延迟：6.18 ms / token。
- 平均 A→F 传输同样为 0.1094 MiB / token（batch 更大但 per-token payload 不变）。
- 最近 batch（batch_size=6）总 payload 0.6562 MiB。

**结论**：大批次显著 amortize 了 Attention/FFN 的 kernel 启动开销；因为串行执行，延迟≈Attention+FFN，两阶段差距约 3x，后续可以据此调整 SM 配比或尝试并行化。

## Summary 表
- `Avg total time` 列是每 token 平均 decode 时间（ms），Small case 明显更高。
- `Avg A→F payload (MiB)` 对两个 case 都是 0.1094 MiB/token（由 hidden_size, dtype 决定，几乎与 batch 无关）。
- `Tokens` 显示统计样本数量：小 case 48，大 case 1152。

## 下一步
1. 如需真正的 Green Context pipeline，需要在支持 `cuGreenCtxStreamCreate` 的 CUDA 12.4+ 环境中运行；届时统计会显示真正的 SM 并行延迟。
2. 可以再加其它 SM 配比（如 `20:12`、`24:8`），比较 Attention/FFN 平均时间变化，指导动态 producer-consumer 设计。
