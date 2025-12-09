# 双卡 PD 分离 + Attention/FFN 动态调度规划

面向 2×GPU（例如 A800×2）环境，将 Prefill（PD）与 Decode 分别驻留在不同卡上，同时在 Decode 卡上继续推进 Attention/FFN 的 SM 动态划分。本文档替换原单卡计划，强调「卡级别调度 → 卡内流水线」的分层设计。

## 硬件 / 部署假设
- GPU0：Prefill 卡。负责大 batch 编码、KV cache 初始化、Prompt 阶段的所有算子。
- GPU1：Decode 卡。负责增量 Decode、Attention/FFN 双阶段流水以及最终 logits。
- 两卡之间通过 NVLink/PCIe 共享 KV cache 元数据；`Sequence.block_table`、`chunked_cache` 等需要显式跨卡同步。
- 每张卡由独立的 `torch.cuda.device` / stream 池管理；Scheduler 负责将请求在 Prefill 卡跑完后迁移到 Decode 卡。

## 阅读策略
- 依靠 `green_ctx.py` 中的 Green Context 工具理解 SM 切分方式。
- 只阅读 `.gitignore` 允许的源码与脚本；重点是 `nanovllm/engine/`、`nanovllm/layers/`、`nanovllm/models/`、`bench.py`。

---

## 阶段一：卡级 PD（prefill/decode）分离
目标：让 Prefill 完全固定在 GPU0，Decode 完全固定在 GPU1，并保证 KV cache/调度状态的一致性。此阶段不引入 Green Context，仅完成双卡管线。

1. **调度建模**
   - `Scheduler.schedule()` 引入 `PrefillShard`/`DecodeShard` 两种任务描述，记录所属 GPU。
   - `Scheduler.ready_for_prefill()` 只将请求交给 GPU0；Prefill 完成后将句柄（KV 元信息、序列 id、token 布局）放入跨卡队列，再由 `ready_for_decode()` 交给 GPU1。
2. **执行流程**
   - `nanovllm/engine/model_runner.py`
     - 拆分 `run_prefill()` 与 `run_decode_core()`，并由新的 `ModelRunner(device_id=0/1)` 实例化两个 worker。
     - Prefill 结束后调用 `kv_cache.mirror_to(device=1)` 或显式的 NCCL 传输接口，只同步必要的先进缓存块。
   - `nanovllm/models/qwen3.py`
     - `forward_prefill()` 仍走全串行模型，但绑定 `device=GPU0`。
     - `forward_decode()` 仅依赖 GPU1 的块；需要新增 `ensure_kv_resident(sequence, device=1)`。
3. **验证**
   - `python example_with_pipeline.py --prefill-device 0 --decode-device 1`。
   - `python bench.py --two-gpus`，比较 Prefill/Decode 各自的 GPU Util。

交付：master 获得卡级 PD 分离能力，Decode 卡提供唯一入口 `run_decode_core(device=1)`。

---

## 阶段二：GPU1 上的 Attention/FFN 双阶段流水
目标：在 Decode 卡内部继续拆分 Attention 与 FFN，先以固定 SM 配比实现流水，再引入 Green Context。

1. **Green Context 基础**
   - 新建 `nanovllm/engine/green_manager.py`，封装 `split_device_green_ctx_by_sm_count(device=1, attention_sm, ffn_sm)`。
   - 接口保持：
     ```python
     class GreenManager:
         def allocate(self, attention_sm: int, ffn_sm: int) -> tuple[torch.Stream, torch.Stream]
         def rebalance(self, attention_sm: int, ffn_sm: int) -> None
         def stats(self) -> dict[str, float]
     ```
2. **流水实现**
   - `run_decode_core()` 拆分 StageA（KV lookup + Attention）与 StageF（FFN+logits），两阶段分别绑定 Green Context 分出的 stream。
   - 使用 `torch.cuda.Event` 管控依赖：StageF 等待 StageA 对应 token 的 event；Prefill 卡无需介入。
   - 初版 SM 配比固定 1:1；利用 `bench.py` 统计 StageA/StageF 平均时延。
3. **诊断工具**
   - 在 GPU1 添加 `pipeline_debug` 日志：记录 token id、StageA/StageF queue 深度、SM 配比、整体吞吐。

交付：Decode 卡具备静态 SM 切分的 Attention/FFN 流水，Prefill 卡逻辑保持不变。

---

## 阶段三：跨阶段动态调度与回落策略
目标：结合两卡信息动态分配资源，并在异常情况下降级。

1. **DecodePipelineScheduler**
   - 新增 `nanovllm/engine/pipeline_scheduler.py`：维护 `attention_queue`、`ffn_queue`，根据最近 N 个 token 的 `avg_latency_*` 触发 `GreenManager.rebalance()`。
   - 在卡级调度层面加入 back-pressure：当 GPU1 解码 backlog > 阈值时，GPU0 暂停新的 Prefill，避免 KV 同步过量堆积。
2. **回退策略**
   - Green Context 初始化失败或重平衡异常时，自动切换为「GPU1 单 stream 顺序 decode」，不中断服务。
   - 提供 `--decode-pipeline=auto/static/disabled` CLI，以方便基准测试。
3. **监控指标**
   - 导出 Prometheus 友好的计数：Prefill/Decode 时延、StageA/StageF SM 利用率、跨卡 memcpy 体积。

交付：两张卡协同工作，Decode 卡内具备动态 SM 调度，Prefill 卡可根据 Decode 压力自适应限速。

---

## 里程碑
1. **M1：卡级 PD 分离**
   - PR：Scheduler/ModelRunner/Qwen3 双实例化、KV 同步接口、基础回归数据。
2. **M2：Decode 静态流水**
   - PR：GreenManager、StageA/StageF 拆分、固定配比评测。
3. **M3：动态调度器**
   - PR：DecodePipelineScheduler、动态重平衡、Prefill back-pressure。
4. **M4：文档与运维**
   - 更新 README、部署指南；记录 CUDA 12.4+、NCCL 要求、降级开关。

---

## 风险与对策
- **跨卡 KV 一致性**：Prefill 卡写入的块必须在 decode 前传输完成；需要 checksum 或版本号确保不会读旧块。
- **Green Context 依赖**：驱动版本不满足时立即回退到单 stream；同时保留 `--disable-green` 开关。
- **内存压力**：两卡各自维护 stream/缓冲区，注意分配上限；可按 `max_batch_size` 切分 KV。
- **Tensor Parallel 约束**：阶段性仅支持 `tensor_parallel_size = 1`；多卡 TP 需要额外的 AllReduce，同步成本高，暂时禁止与本方案共存。
- **故障恢复**：任何卡出错都需要清理另一张卡的队列，防止 KV 泄漏；引入 `SequenceLease` 统一回收。
