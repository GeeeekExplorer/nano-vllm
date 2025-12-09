# Pipeline 功能快速入门

## 5 分钟上手指南

### 1. 启用 Pipeline（只需 3 行代码）

将现有代码：
```python
from nanovllm import LLM, SamplingParams

llm = LLM("./Qwen3-0.6B/")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=50))
```

改为：
```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "./Qwen3-0.6B/",
    enable_pipeline=True,  # 添加这一行启用 pipeline
)
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=50))
```

就这么简单！✨

### 2. 自定义 SM 分配（可选）

```python
llm = LLM(
    "./Qwen3-0.6B/",
    enable_pipeline=True,
    pipeline_attention_sm_count=20,  # 给 Attention 分配 20 个 SMs
    pipeline_ffn_sm_count=12,        # 给 FFN 分配 12 个 SMs
)
```

### 3. 查看性能统计（可选）

```python
llm = LLM(
    "./Qwen3-0.6B/",
    enable_pipeline=True,
    pipeline_enable_profiling=True,  # 启用性能分析
)

outputs = llm.generate(prompts, sampling_params)

# 打印统计信息
llm.model_runner.print_pipeline_statistics()
```

输出示例：
```
[Pipeline Statistics]
  Total tokens processed: 256
  Avg attention time: 2.34 ms
  Avg FFN time: 3.12 ms
  Avg total time: 5.46 ms
```

## 完整示例

```python
from nanovllm import LLM, SamplingParams

# 初始化（启用 pipeline）
llm = LLM(
    "./Qwen3-0.6B/",
    enable_pipeline=True,
    pipeline_attention_sm_count=16,
    pipeline_ffn_sm_count=16,
    pipeline_enable_profiling=True,
)

# 正常使用
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
prompts = ["Hello, how are you?", "What is AI?"]
outputs = llm.generate(prompts, sampling_params)

# 查看结果
for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output['text']}\n")

# 查看性能
llm.model_runner.print_pipeline_statistics()
```

## 何时使用 Pipeline？

### ✅ 推荐使用场景
- 多个请求并发生成 (batch_size >= 4)
- 长序列生成 (max_tokens >= 50)
- 高吞吐量需求

### ⚠️ 不推荐场景
- 单个短序列 (max_tokens < 20)
- 低延迟优先场景
- 显存紧张时

## 系统要求

- ✅ CUDA 12.4+
- ✅ 单卡模式 (`tensor_parallel_size=1`)
- ✅ GPU: Volta 及以上 (Compute Capability 7.0+)

不满足要求时会自动禁用 pipeline，不影响正常使用。

## 故障排查

### Pipeline 未启用？

检查日志输出：
```
[ModelRunner] Initializing pipeline with Attn=16, FFN=16 SMs
[GreenContextManager] Initialized with SM allocation:
  - Attention: 16 SMs
  - FFN: 16 SMs
  - Remaining: 100 SMs
[ModelRunner] Pipeline initialized successfully
```

如果看到：
```
[GreenContextManager] Failed to initialize Green Context: ...
[ModelRunner] Falling back to sequential execution
```

说明系统不支持 Green Context，但不影响功能。

### 性能没有提升？

1. 增加 workload 大小：
   ```python
   # 更多并发请求
   prompts = ["prompt"] * 8

   # 更长生成长度
   sampling_params = SamplingParams(max_tokens=100)
   ```

2. 调整 SM 分配：
   ```python
   # 如果 Attention 是瓶颈
   pipeline_attention_sm_count=24
   pipeline_ffn_sm_count=8
   ```

3. 启用 profiling 查看各阶段耗时

## 更多资源

- 📖 [完整文档](PIPELINE_README.md)
- 🔧 [实现细节](IMPLEMENTATION_SUMMARY.md)
- 💡 [设计规划](DECODE_PIPELINE_PLAN.md)
- 📝 [示例代码](example_with_pipeline.py)
- 🧪 [测试脚本](test_pipeline.py)

## 配置参数速查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_pipeline` | bool | False | 启用/禁用 pipeline |
| `pipeline_attention_sm_count` | int | 16 | Attention 阶段 SM 数 |
| `pipeline_ffn_sm_count` | int | 16 | FFN 阶段 SM 数 |
| `pipeline_enable_profiling` | bool | False | 启用性能分析 |

---

**提示**: 所有 pipeline 配置都是可选的，不设置时使用默认值。
