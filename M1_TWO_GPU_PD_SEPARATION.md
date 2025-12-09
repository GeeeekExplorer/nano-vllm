# M1: Two-GPU PD (Prefill/Decode) Separation

## 概述

实现了 DECODE_PIPELINE_PLAN.md 的**阶段一（M1）**：卡级 PD 分离。

- **GPU0 (Prefill Card)**: 专门负责 Prefill 阶段，处理 prompt 编码和 KV cache 初始化
- **GPU1 (Decode Card)**: 专门负责 Decode 阶段，接收 GPU0 的 KV cache 后进行增量生成
- **KV Cache 同步**: Prefill 完成后，自动将相关的 KV cache 块从 GPU0 拷贝到 GPU1

## 架构变更

### 1. Config (`nanovllm/config.py`)
新增配置项：
```python
enable_two_gpu_pd: bool = False       # 启用两卡 PD 分离模式
prefill_device_id: int = 0            # Prefill 卡 ID（默认 GPU0）
decode_device_id: int = 1             # Decode 卡 ID（默认 GPU1）
```

### 2. ModelRunner (`nanovllm/engine/model_runner.py`)
- 新增 `device_id` 参数，支持指定运行设备（不再强制使用 rank）
- 新增 `sync_kv_cache_to()` 方法，用于跨 GPU 同步 KV cache 块
- 兼容原有的 tensor parallel 模式

### 3. Scheduler (`nanovllm/engine/scheduler.py`)
- 保留原有的 `schedule_prefill()` 和 `schedule_decode()` 分离接口
- 新增 `get_prefilled_sequences()` 方法，用于追踪刚完成 prefill 的序列

### 4. LLMEngine (`nanovllm/engine/llm_engine.py`)
- **单卡模式**：保持原有逻辑不变
- **两卡模式**：
  - 创建两个独立的 `ModelRunner` 实例（prefill_runner, decode_runner）
  - `_step_two_gpu()` 方法协调两卡工作：
    1. GPU0 执行 prefill
    2. 同步 KV cache 块到 GPU1
    3. GPU1 执行 decode

## 使用方法

### 基础示例

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "./Qwen3-0.6B/",
    enable_two_gpu_pd=True,      # 启用两卡模式
    prefill_device_id=0,         # Prefill 在 GPU0
    decode_device_id=1,          # Decode 在 GPU1
)

prompts = ["Hello, world!", "What is AI?"]
sampling_params = SamplingParams(temperature=0.7, max_tokens=64)
outputs = llm.generate(prompts, sampling_params)
```

### 运行示例脚本

```bash
# 基础示例
python example_two_gpu.py

# 测试套件（需要至少 2 张 GPU）
python test_two_gpu.py
```

## 限制与约束

1. **必须使用 `tensor_parallel_size=1`**
   - 两卡 PD 分离目前不支持与 tensor parallel 共存
   - 原因：跨卡 AllReduce 同步成本高，会抵消 PD 分离的收益

2. **GPU 数量要求**
   - 至少需要 2 张 GPU
   - `prefill_device_id` 和 `decode_device_id` 必须不同

3. **KV Cache 一致性**
   - Prefill 卡写入的 KV 块必须在 decode 前传输完成
   - 当前使用 `copy_()` 同步传输，后续可优化为异步传输

## 性能特点

### 优势
- **Prefill/Decode 并行**：GPU0 可以处理新的 prefill 请求，同时 GPU1 处理 decode
- **专用资源**：两个阶段不再争抢同一张卡的计算资源
- **吞吐提升**：理论上可以接近 2x 吞吐（取决于 P/D 比例）

### 当前版本的权衡
- KV cache 同步有额外开销（使用 P2P 拷贝，依赖 NVLink/PCIe 带宽）
- 两张卡的 KV cache 都需要分配内存（总内存使用 ~2x）

## 后续阶段

- **M2 (阶段二)**：在 GPU1 上添加 Attention/FFN 静态流水线
  - 使用 `green_ctx.py` 切分 SM
  - Attention 和 FFN 在不同 SM 分区并行执行

- **M3 (阶段三)**：动态调度与回退策略
  - 根据 GPU1 负载动态调整 GPU0 的 prefill 速率
  - SM 动态重平衡
  - 异常情况下自动降级

## 验证方法

### 手动验证
```bash
# 检查两张 GPU 是否都在工作
nvidia-smi dmon -s u

# 运行生成任务，观察 GPU 利用率
python example_two_gpu.py
```

### 自动化测试
```bash
python test_two_gpu.py
```

测试覆盖：
1. 基础功能：两卡能否正常协同工作
2. 正确性：两卡模式与单卡模式输出是否一致

## 文件清单

**新增文件**：
- `example_two_gpu.py` - 两卡模式示例脚本
- `test_two_gpu.py` - 两卡模式测试套件
- `M1_TWO_GPU_PD_SEPARATION.md` - 本文档

**修改文件**：
- `nanovllm/config.py` - 添加两卡配置项
- `nanovllm/engine/llm_engine.py` - 支持两卡协同
- `nanovllm/engine/model_runner.py` - 支持指定设备 + KV 同步
- `nanovllm/engine/scheduler.py` - 追踪 prefilled 序列

**保留文件**（用于后续阶段）：
- `green_ctx.py` - Green Context 工具（M2 需要）

## 已知问题

1. **KV cache 同步性能**
   - 当前使用同步拷贝，可能成为瓶颈
   - 后续可改为异步传输 + event 同步

2. **内存使用**
   - 两张卡都分配了完整的 KV cache
   - 可以优化为按需分配（GPU1 只分配 decode 需要的块）

3. **调度策略**
   - 当前 GPU0 可能过快生产 prefill 任务，GPU1 来不及消费
   - M3 阶段会添加 back-pressure 机制

## 贡献者

实现基于 `DECODE_PIPELINE_PLAN.md` 的设计规范。
