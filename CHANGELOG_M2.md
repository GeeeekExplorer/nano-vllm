# Changelog: M2 - GPU1 Attention/FFN Pipeline

## 概述
在 M1（两卡 PD 分离）基础上实现了**阶段二（M2）**：GPU1 内部的 Attention/FFN 静态流水线。

## 新增功能

### ✨ GPU1 两阶段流水线
- **Attention Stage**: 在专用 SM 分区执行 KV lookup + Attention
- **FFN Stage**: 在另一个 SM 分区执行 Feed-Forward + Normalization
- 两阶段通过 `torch.cuda.Event` 同步，可以在不同 token 上并行执行

### 🔧 Green Context 集成
- **GreenManager**: 封装 SM 资源分配逻辑
- 使用 `green_ctx.py` 工具切分 GPU1 的 SM 资源
- 为每个阶段创建独立的 CUDA stream
- 支持统计信息查询（tokens, latency, SM allocation）

### 📊 性能 Profiling
- 可选的逐 token 计时
- 统计 Attention/FFN 平均延迟
- 通过 `print_pipeline_statistics()` 查看详细信息

### 🛡️ 自动降级
- Green Context 初始化失败时自动降级到顺序执行
- 保持 M1 功能完整（两卡 PD 分离）
- 不影响系统稳定性

## 核心组件

### 1. GreenManager (`nanovllm/engine/green_manager.py`)
```python
class GreenManager:
    def __init__(device, attention_sm, ffn_sm, enable_rebalance)
    def allocate() -> (attention_stream, ffn_stream)
    def stats() -> dict
    def synchronize()
```

**职责**：
- 调用 `split_device_green_ctx_by_sm_count()` 切分 SM
- 管理两个 Green Context streams
- 为 M3 预留 `rebalance()` 接口

### 2. PipelineScheduler (`nanovllm/engine/pipeline_scheduler.py`)
```python
class PipelineScheduler:
    def __init__(model, green_manager, enable_profiling)
    def decode_token(input_ids, positions) -> hidden_states
    def get_statistics() -> dict
```

**职责**：
- 协调 Attention/FFN 两阶段执行
- 使用 Event 管理依赖关系
- 收集性能统计数据
- 提供顺序执行回退

### 3. 模型拆分接口
在三个层级添加拆分方法：
- **Qwen3DecoderLayer**: `forward_attention_stage()`, `forward_ffn_stage()`
- **Qwen3Model**: 遍历所有 layers 执行对应 stage
- **Qwen3ForCausalLM**: 顶层接口

## 配置项

### 新增配置
```python
enable_decode_pipeline: bool = False       # 启用 M2 流水线
decode_attention_sm: int = 16              # Attention 阶段 SM 数
decode_ffn_sm: int = 16                    # FFN 阶段 SM 数
decode_pipeline_profiling: bool = False    # 启用性能统计
```

### 配置约束
- `enable_decode_pipeline=True` 必须配合 `enable_two_gpu_pd=True`
- M2 不能单独使用，必须建立在 M1 基础上

## 使用示例

### 最小配置
```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "./model",
    enable_two_gpu_pd=True,          # M1
    enable_decode_pipeline=True,     # M2
    decode_attention_sm=16,
    decode_ffn_sm=16,
)
```

### 查看统计
```python
outputs = llm.generate(prompts, sampling_params)
llm.decode_runner.print_pipeline_statistics()
```

### 运行示例
```bash
python example_m2_pipeline.py   # 演示示例
python test_m2_pipeline.py      # 测试套件
```

## 性能预期

### 理想情况
| 阶段 | 延迟 | 说明 |
|------|------|------|
| Attention Only | 1.2 ms | 顺序执行 |
| FFN Only | 1.0 ms | 顺序执行 |
| **Sequential** | **2.2 ms** | 总延迟 = 1.2 + 1.0 |
| **Pipeline (M2)** | **~1.3 ms** | 接近 max(1.2, 1.0) |
| **Speedup** | **~1.7x** | 在多 token 情况下 |

### 实际因素
- **Batch Size**: 越大收益越明显（>= 4 推荐）
- **Token Count**: 需要一定 queue 深度（>= 32 tokens）
- **SM 配比**: 16:16 是初始配置，不同模型可能需要调整
- **硬件**: NVLink 和高 SM 数量的 GPU 效果更好

## 兼容性

### ✅ 兼容
- M1 模式（两卡 PD 分离）保持完全兼容
- 单卡模式不受影响
- 所有 M1 配置和 API 保持不变

### ⚠️ 要求
- **必须启用 M1**：`enable_two_gpu_pd=True`
- **CUDA 版本**：12.4+ （Green Context 需要）
- **cuda-python**：需要安装 `pip install cuda-python`
- **GPU 型号**：推荐 Ampere 架构及以上（A100, H100 等）

### ❌ 限制
- **不支持 CUDA Graph**：M2 pipeline 需要 `enforce_eager=True`
- **不支持 Tensor Parallel**：与 M1 一致，`tensor_parallel_size` 必须为 1
- **SM 数量要求**：GPU 需要足够的 SM（建议 ≥ 80）

## 文件变更

```
新增:
  nanovllm/engine/green_manager.py        # Green Context 管理器
  nanovllm/engine/pipeline_scheduler.py   # 两阶段流水线调度器
  example_m2_pipeline.py                  # M2 示例脚本
  test_m2_pipeline.py                     # M2 测试套件
  M2_DECODE_PIPELINE.md                   # M2 完整文档
  CHANGELOG_M2.md                         # 本文档

修改:
  nanovllm/config.py                      # +4 lines (M2 配置)
  nanovllm/engine/model_runner.py         # +45 lines (pipeline 集成)
  nanovllm/engine/llm_engine.py           # +2 lines (is_decode_runner 标志)
  nanovllm/models/qwen3.py                # +60 lines (拆分接口)
```

## 测试覆盖

### 自动化测试 (`test_m2_pipeline.py`)
1. ✅ **基础功能测试**：Pipeline 能否正常工作
2. ✅ **正确性测试**：M1+M2 vs M1-only 输出一致性
3. ✅ **性能测试**：M2 相对 M1 的加速比测量

### 手动验证
```bash
# 1. 检查 GPU 利用率
nvidia-smi dmon -s u

# 2. 运行示例并观察统计
python example_m2_pipeline.py

# 3. 对比 M1-only
python example_two_gpu.py  # M1 only
python example_m2_pipeline.py  # M1+M2
```

## 已知问题 & 限制

### 1. SM 配比固定
**现状**：M2 使用固定的 16:16 配比
**影响**：不同模型/workload 的最优配比不同
**解决**：M3 将支持动态 SM 重平衡

### 2. 浅流水线
**现状**：只有 2 级流水线（Attention, FFN）
**影响**：单 token 时无法并行
**解决**：需要一定的 decode queue 深度（>= 4 tokens）

### 3. CUDA Graph 不支持
**现状**：Pipeline 使用动态 stream，无法捕获到 CUDA Graph
**影响**：无法使用 CUDA Graph 优化
**解决**：M2 需要 `enforce_eager=True`

### 4. 初始化开销
**现状**：Green Context 初始化需要几百毫秒
**影响**：冷启动时间增加
**解决**：可接受（只在启动时发生一次）

## 故障排查

### Green Context 初始化失败
```
[GreenManager] Failed to initialize: CUDA error code=914
[GreenManager] Falling back to sequential decode
```

**检查清单**：
1. CUDA 版本：`nvcc --version` (需要 12.4+)
2. cuda-python：`pip show cuda-python`
3. GPU 支持：运行 `nvidia-smi` 检查 GPU 型号

### Pipeline 无性能提升
**可能原因**：
- Batch size 太小（<4）
- Max tokens 太少（<32）
- SM 配比不适合当前模型

**调试步骤**：
1. 启用 profiling：`decode_pipeline_profiling=True`
2. 查看统计：`llm.decode_runner.print_pipeline_statistics()`
3. 调整配比：尝试 `decode_attention_sm=20, decode_ffn_sm=12`

## Breaking Changes

无。M2 完全向后兼容，所有现有功能保持不变。

## 依赖更新

新增依赖：
```
cuda-python>=12.4  # Green Context 支持
```

安装：
```bash
pip install cuda-python
```

## 后续计划

### M3 阶段
- [ ] 动态 SM 重平衡（根据实际延迟调整配比）
- [ ] GPU0 back-pressure（避免 GPU1 过载）
- [ ] 更深流水线（per-layer 流水）
- [ ] 自适应降级策略

### 性能优化
- [ ] CUDA Graph 兼容性研究
- [ ] 异步 KV cache 传输
- [ ] 多 token 批处理优化

## 贡献者

实现基于 `DECODE_PIPELINE_PLAN.md` 的 M2 阶段规范。
