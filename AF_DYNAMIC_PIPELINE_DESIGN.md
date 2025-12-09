# Attention/FFN Dynamic Producer-Consumer Blueprint

## 背景
- M2 阶段已经把 Attention (A) 与 FFN (F) 拆分成两个 stage，并在 `PipelineScheduler` 中通过 Green Context 保证 SM 隔离。
- 新增的层级通信统计 (`PipelineScheduler.layer_comm`) 能精确告诉我们每一层 A→F 之间传递的数据量（Attention 输出 + Residual）。
- `decode_pipeline_case_study.py` 可以跑「小 decode」和「大 decode」两种负载，并对不同的 SM 配比输出性能/通信指标，方便观察资源分配的灵敏度。

这些信息为后续的动态调度（M3 目标）提供了可观测性：我们知道每个 layer 的 payload 大小、不同 workload 对 SM 分配的敏感度，以及当前固定流水线的瓶颈。

## 建议的动态生产者/消费者架构
目标：把 GPU1 内部的 Attention stage 做成生产者（Producer），FFN stage 做成消费者（Consumer），并允许根据实时负载调整 SM 分配。

1. **AttentionProducer**
   - 在各层完成 Attention 之后立即把 `TokenPayload` 写入无锁 ring buffer。
   - `TokenPayload` 结构：`{token_id, layer_idx, hidden_ptr, residual_ptr, num_bytes}`。
   - 通过当前的 `forward_attention_stage()` 返回的 metadata 直接填充 `num_bytes`，无需额外统计。

2. **PayloadQueue**
   - 每个 layer 建一个小队列（深度 2~4），用于缓冲 Attention/FFN 的交接。
   - 队列项仅存放指针 + metadata，真正的 tensor 仍由 decode runner 持有，可通过事件 (`torch.cuda.Event`) 控制生命周期。
   - 当所有 layer 的队列都满时对 Producer 施加背压，反之 Consumer 空闲则唤醒 Producer。

3. **FFNConsumer**
   - 从对应 layer 的队列中取出 `TokenPayload`，执行 `forward_ffn_stage()`。
   - 完成后在 payload 上打事件，通知 Producer 该层的缓存可以复用。
   - 根据 `num_bytes` 与当前 SM 利用率判断是否需要更激进的再平衡。

4. **GreenManager 扩展**
   - 暴露 `rebalance()`，按 `attention_latency / ffn_latency`、`queue_depth`、`avg_comm_bytes` 计算新的 SM 配比。
   - 重平衡策略示例：`target_ratio = sqrt(attn_latency / ffn_latency)`，并对子阶段设置上下限。

5. **调度与监控**
   - `PipelineScheduler` 持续读取 `layer_comm` 与新的 queue 深度统计，形成如下指标：
     - `avg_comm_bytes_per_token`
     - `attention_backlog` / `ffn_backlog`
     - `producer_idle_time` / `consumer_idle_time`
   - 将指标写入 `print_pipeline_statistics()`，配合 `decode_pipeline_case_study.py` 的两种 workload，就能直观看出配比是否合理。

## 流程概览
```
Token i ───► AttentionProducer ──┐  layer=0 queue ─► FFNConsumer ───► logits
                                 ├─ layer=1 queue ─► FFNConsumer
                                 └─ ...
```

每层 queue 的深度控制 producer/consumer 的相位差；`num_bytes` 用来评估通信成本，并决定是否需要拆分得更细（例如把某些层合并到 Attention 或 FFN 侧，从而降低数据转移量）。

## 后续落地步骤
1. **验证通信成本**：使用 `llm.decode_runner.print_pipeline_statistics()` 查看每层的 MB/token，识别负载最重的层。
2. **跑大小 workload**：执行 `python decode_pipeline_case_study.py --sm-splits 16:16,20:12`，比较不同 SM 配比下的吞吐与 A→F payload。
3. **实现 queue 原型**：在 `PipelineScheduler` 中引入 per-layer ring buffer，先以内存事件同步验证正确性。
4. **接入动态 rebalance**：当 `attention_backlog` 明显大于 `ffn_backlog` 时，调用 `GreenManager.rebalance()` 迁移少量 SM。
5. **Auto-tuning**：记录多次 workload 的统计（例如 large vs. small decode），形成经验公式用于冷启动。

通过以上步骤就可以把静态的 M2 pipeline 逐步演化成真正的动态生产者/消费者体系，为后续 M3 做好准备。
