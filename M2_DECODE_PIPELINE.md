# M2: GPU1 Attention/FFN Pipeline

## 概述

实现了 DECODE_PIPELINE_PLAN.md 的**阶段二（M2）**：在 GPU1（Decode 卡）内部实现 Attention/FFN 静态流水线。

**在 M1 基础上新增**：
- GPU1 的 SM 资源被切分为两个分区：Attention 区和 FFN 区
- Attention 和 FFN 阶段在不同的 SM 分区上并行执行
- 使用 Green Context 进行 SM 级别的资源隔离
- 固定 SM 配比（默认 16:16）

## 架构

```
┌──────────────────────────────────────────────────────────┐
│                        GPU 0 (Prefill)                    │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Prompt Encoding + KV Cache Generation             │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬───────────────────────┘
                                   │ KV Sync
                                   ↓
┌──────────────────────────────────────────────────────────┐
│                  GPU 1 (Decode + Pipeline)                │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Attention Stage (16 SMs)                           │ │
│  │  • Embedding                                        │ │
│  │  • KV Lookup                                        │ │
│  │  • Attention Computation                            │ │
│  │  • LayerNorm                                        │ │
│  └─────────────────────────────────────────────────────┘ │
│           │                                                │
│           │ Event Sync                                     │
│           ↓                                                │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  FFN Stage (16 SMs)                                 │ │
│  │  • Post-Attention LayerNorm                         │ │
│  │  • Feed-Forward Network                             │ │
│  │  • Final LayerNorm                                  │ │
│  │  • LM Head (Logits)                                 │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. GreenManager (`nanovllm/engine/green_manager.py`)
负责 GPU1 的 SM 资源管理：
- 使用 `green_ctx.py` 切分 SM 资源
- 为 Attention 和 FFN 创建独立的 CUDA streams
- 提供 `allocate()` 和 `stats()` 接口
- M3 阶段将支持 `rebalance()` 动态调整

```python
green_manager = GreenManager(
    device=torch.device("cuda:1"),
    attention_sm=16,
    ffn_sm=16,
    enable_rebalance=False,  # M3 feature
)
```

### 2. PipelineScheduler (`nanovllm/engine/pipeline_scheduler.py`)
管理两阶段流水线执行：
- `decode_token()`: 主入口，协调两阶段执行
- 使用 `torch.cuda.Event` 同步 Attention → FFN 依赖
- 可选的性能 profiling（记录每阶段耗时）
- 回退到顺序执行（当 Green Context 不可用时）

```python
pipeline_scheduler = PipelineScheduler(
    model=model,
    green_manager=green_manager,
    enable_profiling=True,
)
```

### 3. 模型拆分接口（`nanovllm/models/qwen3.py`）
为流水线添加拆分接口：
- `forward_attention_stage()`: 执行 Attention 阶段
- `forward_ffn_stage()`: 执行 FFN 阶段
- 在 `Qwen3DecoderLayer`, `Qwen3Model`, `Qwen3ForCausalLM` 三层都实现

### 4. ModelRunner 集成
- 新增 `is_decode_runner` 参数标识 Decode runner
- 在 Decode runner 初始化时自动创建 pipeline
- `run_decode_core()` 优先使用 pipeline
- 提供 `get_pipeline_statistics()` 和 `print_pipeline_statistics()`

## 配置项

```python
# M1 配置（必需）
enable_two_gpu_pd = True
prefill_device_id = 0
decode_device_id = 1

# M2 新增配置
enable_decode_pipeline = True      # 启用 GPU1 流水线
decode_attention_sm = 16           # Attention 阶段 SM 数量
decode_ffn_sm = 16                 # FFN 阶段 SM 数量
decode_pipeline_profiling = False  # 启用性能统计
```

## 使用示例

### 基础用法

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "./Qwen3-0.6B/",
    # M1: 两卡 PD 分离
    enable_two_gpu_pd=True,
    prefill_device_id=0,
    decode_device_id=1,
    # M2: GPU1 流水线
    enable_decode_pipeline=True,
    decode_attention_sm=16,
    decode_ffn_sm=16,
    decode_pipeline_profiling=True,
)

outputs = llm.generate(
    ["Hello!", "What is AI?"],
    SamplingParams(temperature=0.7, max_tokens=64)
)
```

### 查看性能统计

```python
# 在 generate() 后
llm.decode_runner.print_pipeline_statistics()
```

输出示例：
```
[M2 Pipeline Statistics - GPU1 Decode Card]
  Total tokens: 256
  Avg attention time: 1.23 ms
  Avg FFN time: 0.98 ms
  Avg total time: 2.21 ms
  Attention SMs: 16
  FFN SMs: 16
```

## 运行示例与测试

```bash
# 运行 M2 示例（需要 2 张 GPU）
python example_m2_pipeline.py

# 运行测试套件
python test_m2_pipeline.py
```

测试覆盖：
1. **基础功能**：Pipeline 能否正确执行
2. **正确性**：M1+M2 与 M1-only 输出一致性
3. **性能**：M2 相对 M1 的加速比

## 性能特点

### 优势
- **阶段并行**：Attention 和 FFN 可以在不同 token 上并行
- **资源隔离**：两阶段不会争抢同一组 SMs
- **延迟降低**：理想情况下，decode 延迟可接近 max(T_attn, T_ffn) 而非 T_attn + T_ffn

### 当前限制（M2 静态配比）
- SM 配比固定（16:16），不根据负载动态调整
- 单 token 场景收益有限（流水线需要多个 token 才能充分利用）
- 小 batch size 下开销可能抵消收益

## 依赖要求

### CUDA 要求
- CUDA 12.4 或更高版本
- 支持 Green Context 的 GPU（Compute Capability ≥ 8.0 推荐）
- cuda-python 包（需安装 `pip install cuda-python`）

### 硬件要求
- 至少 2 张 CUDA GPU
- 推荐 NVLink 连接（加速 KV cache 传输）
- 推荐 80+ SMs 的 GPU（如 A100, H100）

## 已知限制

### 1. Green Context 依赖
- 需要 CUDA 12.4+ 和 cuda-python
- 不支持所有 GPU 型号
- 初始化失败时自动降级到顺序执行

### 2. 静态 SM 配比
- M2 阶段固定 16:16 配比
- 不同模型和 batch size 的最优配比不同
- M3 阶段将支持动态调整

### 3. 流水线深度
- 当前实现为浅流水线（2 阶段）
- 单 token 情况下无法并行
- 需要一定的 decode queue 深度才能充分利用

### 4. CUDA Graph 兼容性
- M2 pipeline 目前不支持 CUDA Graph 捕获
- 需要 `enforce_eager=True`

## 故障处理

### Green Context 初始化失败
**症状**：看到 "Failed to initialize Green Context" 警告

**原因**：
- CUDA 版本 < 12.4
- cuda-python 未安装或版本不匹配
- GPU 不支持 Green Context

**解决**：
1. 检查 CUDA 版本：`nvcc --version`
2. 安装 cuda-python：`pip install cuda-python`
3. 如果仍失败，系统会自动降级到顺序执行（M1-only）

### Pipeline 性能没有提升
**可能原因**：
- Batch size 太小（<4）
- Token 生成太少（<32）
- SM 配比不合适

**建议**：
- 增加 batch size 或 max_tokens
- 尝试调整 `decode_attention_sm` 和 `decode_ffn_sm`
- 使用 profiling 模式分析瓶颈

## 与 M1 的关系

M2 **必须**在 M1 的基础上启用：
```python
# 正确：M1 + M2
enable_two_gpu_pd = True          # M1 必需
enable_decode_pipeline = True     # M2 可选

# 错误：只启用 M2
enable_two_gpu_pd = False
enable_decode_pipeline = True     # 会报错
```

## 下一步：M3

M3 阶段将实现：
1. **动态 SM 重平衡**：根据 Attention/FFN 实际耗时动态调整配比
2. **GPU0 back-pressure**：GPU1 负载过高时，GPU0 自动限速
3. **更深流水线**：支持多 token 并行（每层独立流水）
4. **自适应降级**：异常情况下自动回退到 M1 或单卡模式

## 文件清单

**新增文件**：
- `nanovllm/engine/green_manager.py` - SM 资源管理器
- `nanovllm/engine/pipeline_scheduler.py` - 两阶段流水线调度器
- `example_m2_pipeline.py` - M2 示例脚本
- `test_m2_pipeline.py` - M2 测试套件
- `M2_DECODE_PIPELINE.md` - 本文档

**修改文件**：
- `nanovllm/config.py` - 添加 M2 配置项
- `nanovllm/engine/model_runner.py` - 集成 pipeline 到 decode_runner
- `nanovllm/models/qwen3.py` - 添加 Attention/FFN 拆分接口
- `nanovllm/engine/llm_engine.py` - 传递 is_decode_runner 标志

**保留文件**（已在 M1 使用）：
- `green_ctx.py` - Green Context 工具函数

## 贡献者

实现基于 `DECODE_PIPELINE_PLAN.md` 的 M2 阶段规范。
