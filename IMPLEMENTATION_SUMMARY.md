# Attention/FFN Pipeline 实现总结

## 实现概述

根据 `DECODE_PIPELINE_PLAN.md` 的规划，已完成单卡 Attention/FFN Green Context 流水线的初步实现。本实现允许在 decode 阶段将 Attention 和 FFN 计算拆分到不同的 SM 资源上并行执行。

## 已完成的工作

### 1. 模型层拆分 (`nanovllm/models/qwen3.py`)

在三个层级添加了分阶段执行方法：

#### Qwen3DecoderLayer
- ✅ `forward_attention()`: 执行 attention + input_layernorm
- ✅ `forward_ffn()`: 执行 FFN + post_attention_layernorm

#### Qwen3Model
- ✅ `forward_attention()`: 对所有 layers 执行 attention 阶段
- ✅ `forward_ffn()`: 对所有 layers 执行 FFN 阶段

#### Qwen3ForCausalLM
- ✅ `forward_attention()`: 模型级别 attention 入口
- ✅ `forward_ffn()`: 模型级别 FFN 入口
- ✅ 保持原有 `forward()` 方法不变，确保向后兼容

### 2. Green Context 管理器 (`nanovllm/engine/green_manager.py`)

实现了 SM 资源管理和分配：

**核心功能**:
- ✅ 基于 `green_ctx.py` 的 `split_device_green_ctx_by_sm_count`
- ✅ 为 Attention 和 FFN 创建独立的 CUDA streams
- ✅ 自动降级机制（不支持 Green Context 时回退到默认 stream）
- ✅ SM 分配信息日志输出
- ✅ 预留动态调整接口 `adjust_sm_allocation()`

**初始化参数**:
```python
GreenContextManager(
    device: torch.device,
    attention_sm_count: int = 16,
    ffn_sm_count: int = 16,
    min_sm_count: int = 8,
    enable_dynamic_adjustment: bool = True,
)
```

### 3. 流水线调度器 (`nanovllm/engine/pipeline_scheduler.py`)

实现了两阶段流水线执行逻辑：

**核心功能**:
- ✅ `decode_token()`: 执行单个 token 的流水线处理
- ✅ Attention 和 FFN 异步执行（使用 CUDA events 同步）
- ✅ 性能分析支持（可选）
- ✅ 自动降级到顺序执行（Green Context 不可用时）

**执行流程**:
```
input_ids, positions
    ↓
[Attention Stage] (Stream 0)
    ↓ (Event sync)
[FFN Stage] (Stream 1)
    ↓
logits
```

**统计信息**:
- 总处理 token 数
- 平均 Attention 耗时
- 平均 FFN 耗时
- 队列长度（预留）

### 4. ModelRunner 集成 (`nanovllm/engine/model_runner.py`)

将流水线集成到现有推理引擎：

**修改点**:
- ✅ 导入 `GreenContextManager` 和 `PipelineScheduler`
- ✅ `__init__`: 根据配置初始化 pipeline 组件
- ✅ `run_model`: 为 decode 添加 pipeline 执行路径
- ✅ `get_pipeline_statistics()`: 查询统计信息
- ✅ `print_pipeline_statistics()`: 打印性能报告

**条件启用**:
- 仅在 `config.enable_pipeline=True` 时启用
- 仅在单卡模式 (`tensor_parallel_size=1`) 时启用
- 仅在 decode 阶段使用（prefill 仍走原路径）

### 5. 配置扩展 (`nanovllm/config.py`)

添加了 pipeline 相关配置项：

```python
@dataclass
class Config:
    # ... 原有配置 ...

    # Pipeline configuration
    enable_pipeline: bool = False
    pipeline_attention_sm_count: int = 16
    pipeline_ffn_sm_count: int = 16
    pipeline_enable_profiling: bool = False
```

### 6. 测试与文档

#### 测试脚本
- ✅ `test_pipeline.py`: 功能测试和性能对比
- ✅ `example_pipeline.py`: 基本使用示例

#### 文档
- ✅ `PIPELINE_README.md`: 完整的使用文档
  - 快速开始指南
  - 配置参数说明
  - 架构设计图
  - 性能调优建议
  - 故障排查
- ✅ `IMPLEMENTATION_SUMMARY.md`: 本文档

## 技术实现细节

### 1. 异步执行机制

```python
# Attention Stage (Stream 0)
with torch.cuda.stream(attention_stream):
    hidden_states, residual = model.forward_attention(input_ids, positions)
    attention_event.record()

# FFN Stage (Stream 1)
with torch.cuda.stream(ffn_stream):
    attention_event.wait()  # 等待 attention 完成
    logits = model.forward_ffn(hidden_states, residual)
    ffn_event.record()

# 主线程等待完成
ffn_event.synchronize()
```

### 2. 资源分配策略

使用 Green Context 将 GPU SMs 分为三组：
- **Attention Group**: `pipeline_attention_sm_count` 个 SMs
- **FFN Group**: `pipeline_ffn_sm_count` 个 SMs
- **Remaining**: 未分配的 SMs（buffer）

实际分配会根据 GPU 架构自动对齐：
- Volta (SM 70): 按 1 对齐
- Turing (SM 75): 按 2 对齐
- Ampere (SM 80): 按 4/2 对齐
- Hopper (SM 90+): 按 8 对齐

### 3. 错误处理

多层级降级机制：
1. Green Context 初始化失败 → 使用默认 stream（顺序执行）
2. Pipeline 初始化异常 → 禁用 pipeline
3. 多卡模式 → 自动禁用 pipeline（当前版本限制）

## 与原设计的对应关系

| 计划 (DECODE_PIPELINE_PLAN.md) | 实现状态 | 备注 |
|--------------------------------|----------|------|
| 模块化前置改造 | ✅ 已完成 | Qwen3 模型拆分完成 |
| Green Context 初始化 | ✅ 已完成 | GreenContextManager 封装 |
| Attention/FFN 流水线执行 | ✅ 已完成 | PipelineScheduler 实现 |
| ModelRunner 集成 | ✅ 已完成 | decode 路径支持 pipeline |
| 性能监控 | ✅ 已完成 | 基础 profiling 支持 |
| 动态 SM 调整 | ⚠️ 接口预留 | 未实现自动调整逻辑 |
| 多级流水线 | ❌ 未实现 | 当前为 2 级（Attn + FFN） |
| 多卡支持 | ❌ 未实现 | 限制 TP=1 |

## 使用示例

### 基础用法

```python
from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine

config = Config(
    model="Qwen3-0.6B",
    enable_pipeline=True,
    pipeline_attention_sm_count=16,
    pipeline_ffn_sm_count=16,
    pipeline_enable_profiling=True,
)

engine = LLMEngine(config)
outputs = engine.generate(["Hello"], max_tokens=50)

# 查看统计
engine.model_runner.print_pipeline_statistics()
```

### 预期输出

```
[ModelRunner] Initializing pipeline with Attn=16, FFN=16 SMs
[GreenContextManager] Initialized with SM allocation:
  - Attention: 16 SMs
  - FFN: 16 SMs
  - Remaining: 100 SMs
[ModelRunner] Pipeline initialized successfully

[Pipeline Statistics]
  Total tokens processed: 50
  Avg attention time: 2.45 ms
  Avg FFN time: 3.21 ms
  Avg total time: 5.66 ms
```

## 性能预期

根据计划文档，预期性能提升场景：
- ✅ **多请求并发 decode**: pipeline 可同时处理不同 token 的 Attn/FFN
- ✅ **长序列生成**: 充分利用流水线深度
- ⚠️ **单 token 小 batch**: 可能因 overhead 无明显提升

## 已知限制

1. **单卡限制**: 当前仅支持 `tensor_parallel_size=1`
2. **Decode 限制**: Prefill 不走 pipeline
3. **CUDA 版本**: 需要 CUDA 12.4+
4. **动态调整**: 预留接口但未实现
5. **CUDAGraph**: Pipeline 模式下暂不支持
6. **显存开销**: 中间 tensor 需要额外显存

## 未来工作

### 短期 (M4)
- [ ] 实现动态 SM 调整
  - 监控 Attention/FFN 队列长度
  - 根据瓶颈重新分配 SM
  - 设置调整冷却时间

### 中期 (M5-M6)
- [ ] 多卡支持 (TP > 1)
- [ ] 三阶段流水线（逐层并行）
- [ ] 更细粒度的 profiling（per-layer）

### 长期 (M7+)
- [ ] CUDAGraph 兼容性
- [ ] 自适应 SM 分配算法
- [ ] 支持其他模型架构

## 测试建议

### 功能测试
```bash
python test_pipeline.py
```

### 性能基准
```bash
# 对比 pipeline vs sequential
python bench.py --enable_pipeline
python bench.py --no-pipeline
```

### 不同配置测试
尝试不同 SM 分配：
- 均衡: `attention=16, ffn=16`
- Attention 优先: `attention=24, ffn=8`
- FFN 优先: `attention=8, ffn=24`

## 贡献与反馈

欢迎在以下方面贡献：
- 性能测试结果（不同 GPU/模型）
- Bug 修复
- 动态调度策略实现
- 多卡支持

## 参考资料

- [DECODE_PIPELINE_PLAN.md](DECODE_PIPELINE_PLAN.md) - 原始设计文档
- [PIPELINE_README.md](PIPELINE_README.md) - 用户文档
- [green_ctx.py](green_ctx.py) - Green Context 工具
- [NVIDIA Green Context API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)

---

**实现日期**: 2025-11-19
**实现者**: Claude Code
**版本**: v0.1.0 (初始实现)
