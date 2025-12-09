# Attention/FFN Pipeline 并行执行

## 概述

本实现基于 NVIDIA Green Context 技术，实现了单卡 decode 阶段的 Attention 与 FFN 流水线并行执行。通过将 GPU 的 SM（Streaming Multiprocessor）资源划分给不同的计算阶段，允许 Attention 和 FFN 同时处理不同 token，从而提升吞吐量。

## 功能特性

- ✅ **两阶段流水线**: 将模型前向传播拆分为 Attention 和 FFN 两个独立阶段
- ✅ **SM 级别资源划分**: 使用 Green Context 技术动态分配 SM 资源
- ✅ **异步执行**: 两阶段可并发执行，利用 CUDA streams 实现异步计算
- ✅ **性能监控**: 内置 profiling 功能，可查看各阶段耗时统计
- ✅ **自动降级**: 不支持 Green Context 时自动回退到顺序执行
- ✅ **兼容现有 API**: 完全兼容原有 `LLM.generate` 接口

## 系统要求

- CUDA 12.4+ (Green Context 支持)
- 支持的 GPU: Compute Capability 6.0+
- 推荐: Ampere 架构及以上 (SM 80+)
- Python 3.10+
- cuda-python (匹配您的 CUDA 版本)

## 快速开始

### 1. 基本使用

```python
from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine

# 创建配置，启用 pipeline
config = Config(
    model="Qwen3-0.6B",
    enable_pipeline=True,                # 启用流水线
    pipeline_attention_sm_count=16,      # Attention 阶段 SM 数量
    pipeline_ffn_sm_count=16,            # FFN 阶段 SM 数量
    pipeline_enable_profiling=True,      # 启用性能分析
)

# 初始化引擎
engine = LLMEngine(config)

# 正常使用 generate
outputs = engine.generate(
    prompts=["Hello, world!"],
    max_tokens=50,
)

# 查看流水线统计信息
engine.model_runner.print_pipeline_statistics()
```

### 2. 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_pipeline` | bool | False | 是否启用流水线执行 |
| `pipeline_attention_sm_count` | int | 16 | 分配给 Attention 阶段的 SM 数量 |
| `pipeline_ffn_sm_count` | int | 16 | 分配给 FFN 阶段的 SM 数量 |
| `pipeline_enable_profiling` | bool | False | 是否启用性能分析 |

**注意**:
- SM 数量会根据 GPU 架构自动向上对齐（例如 Ampere 架构按 8 对齐）
- 总 SM 分配不能超过 GPU 总 SM 数
- 初版仅支持 `tensor_parallel_size=1`（单卡）

### 3. 运行示例

```bash
# 运行基本示例
python example_pipeline.py

# 运行性能测试
python test_pipeline.py
```

## 架构设计

### 模块结构

```
nanovllm/
├── engine/
│   ├── green_manager.py         # Green Context 资源管理
│   ├── pipeline_scheduler.py    # 流水线调度器
│   └── model_runner.py          # 集成流水线执行路径
├── models/
│   └── qwen3.py                 # 模型拆分（attention/ffn）
└── config.py                    # Pipeline 配置
```

### 执行流程

```
┌─────────────────────────────────────────────────────────┐
│  Input: input_ids, positions                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  PipelineScheduler     │
        └────────────┬───────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
         ▼                        ▼
┌─────────────────┐      ┌─────────────────┐
│ Attention Stage │      │   FFN Stage     │
│  (Stream 0)     │─────▶│   (Stream 1)    │
│  16 SMs         │ sync │   16 SMs        │
└─────────────────┘      └─────────────────┘
         │                        │
         │  hidden_states         │  logits
         │  + residual            │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Sampler (next token)  │
         └────────────────────────┘
```

### 核心组件

#### 1. GreenContextManager
管理 SM 资源分配和 CUDA streams：
```python
# 初始化
green_manager = GreenContextManager(
    device=torch.device("cuda:0"),
    attention_sm_count=16,
    ffn_sm_count=16,
)

# 获取各阶段的 stream
attn_stream = green_manager.get_attention_stream()
ffn_stream = green_manager.get_ffn_stream()
```

#### 2. PipelineScheduler
协调 Attention 和 FFN 两阶段执行：
```python
# 执行 decode token
logits = pipeline_scheduler.decode_token(
    input_ids=input_ids,
    positions=positions,
)

# 获取性能统计
stats = pipeline_scheduler.get_statistics()
```

#### 3. 模型拆分
在 `Qwen3ForCausalLM` 中添加分阶段方法：
```python
# Attention 阶段
hidden_states, residual = model.forward_attention(input_ids, positions)

# FFN 阶段
hidden_states = model.forward_ffn(hidden_states, residual)
```

## 性能调优

### 1. SM 分配策略

根据模型特性调整 SM 分配：
- **Attention 密集型**: 增加 `pipeline_attention_sm_count`
- **FFN 密集型**: 增加 `pipeline_ffn_sm_count`
- **平衡型**: 保持两者相等（推荐初始值）

### 2. 性能监控

启用 profiling 查看各阶段耗时：
```python
config.pipeline_enable_profiling = True

# 运行后查看统计
engine.model_runner.print_pipeline_statistics()
```

输出示例：
```
[Pipeline Statistics]
  Total tokens processed: 128
  Avg attention time: 2.34 ms
  Avg FFN time: 3.12 ms
  Avg total time: 5.46 ms
```

### 3. 适用场景

- ✅ **高吞吐 decode**: 多个并发请求
- ✅ **长序列生成**: max_tokens > 50
- ✅ **Batch decode**: batch_size > 1
- ⚠️ **不适合**: 单个短序列（overhead 较大）

## 限制与已知问题

1. **仅支持 decode 阶段**: Prefill 仍使用顺序执行
2. **单卡限制**: 初版仅支持 `tensor_parallel_size=1`
3. **CUDA 版本**: 需要 CUDA 12.4+ 和对应驱动
4. **CUDAGraph 兼容性**: Pipeline 模式下暂不支持 CUDAGraph
5. **显存开销**: 中间张量需要额外显存

## 故障排查

### Green Context 初始化失败
```
[GreenContextManager] Failed to initialize Green Context: CUDA error code=914
```
**解决方案**:
- 检查 CUDA 版本 >= 12.4
- 减少 SM 分配数量
- 更新 NVIDIA 驱动

### Pipeline 自动禁用
```
[ModelRunner] Failed to initialize pipeline: ...
[ModelRunner] Falling back to sequential execution
```
**原因**: 系统不支持 Green Context，自动降级到顺序执行（不影响功能）

### 性能未提升
如果 pipeline 没有带来性能提升：
1. 检查 workload 是否足够大（建议 batch_size >= 4, max_tokens >= 32）
2. 调整 SM 分配比例
3. 启用 profiling 查看瓶颈阶段

## 开发路线图

- [x] **M1**: 基础流水线实现（Attention/FFN 拆分）
- [x] **M2**: Green Context 资源管理
- [x] **M3**: 性能监控与统计
- [ ] **M4**: 动态 SM 调整（根据队列长度）
- [ ] **M5**: 多卡支持（TP > 1）
- [ ] **M6**: 三阶段流水线（逐层并行）
- [ ] **M7**: CUDAGraph 兼容性

## 参考文档

- [CUDA Green Contexts](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)
- [DECODE_PIPELINE_PLAN.md](DECODE_PIPELINE_PLAN.md) - 详细设计文档
- [green_ctx.py](green_ctx.py) - Green Context 工具函数

## 贡献

欢迎提交 Issue 和 PR！特别关注：
- 性能测试结果（不同 GPU/模型/配置）
- Bug 修复
- 动态调度策略改进

## 许可

遵循 nano-vllm 主仓库许可协议。
