# Quick Start: Two-GPU PD Separation

## 什么是两卡 PD 分离？

将 Prefill（提示词编码）和 Decode（逐词生成）分别放在两张独立的 GPU 上运行，实现更高的吞吐量。

**架构**：
```
┌─────────────┐         KV Cache          ┌─────────────┐
│   GPU 0     │  ───────────────────────> │   GPU 1     │
│  (Prefill)  │      Synchronization      │  (Decode)   │
└─────────────┘                           └─────────────┘
     ↓                                            ↓
处理新的 prompt                            增量生成 tokens
```

## 最小示例

```python
from nanovllm import LLM, SamplingParams

# 启用两卡模式只需要三个参数
llm = LLM(
    "./Qwen3-0.6B/",
    enable_two_gpu_pd=True,    # 1. 启用两卡模式
    prefill_device_id=0,       # 2. Prefill 用 GPU0
    decode_device_id=1,        # 3. Decode 用 GPU1
)

# 使用方式完全相同
outputs = llm.generate(
    ["Hello!", "What is AI?"],
    SamplingParams(temperature=0.7, max_tokens=64)
)
```

## 运行示例

```bash
# 需要至少 2 张 GPU
python example_two_gpu.py
```

## 性能对比

| 模式 | Prefill GPU | Decode GPU | 预期提升 |
|------|-------------|------------|---------|
| 单卡 | GPU0 | GPU0 | 基线 (1x) |
| 两卡 PD | GPU0 | GPU1 | ~1.5-2x |

**提升幅度取决于**：
- Prefill/Decode 时间比例
- NVLink vs PCIe 带宽（KV cache 传输速度）
- Batch size 大小

## 限制

1. **必须有 2 张及以上 GPU**
2. **不能与 `tensor_parallel_size > 1` 共存**
3. **KV cache 传输有额外开销**（建议使用 NVLink）

## 下一步

查看完整文档：[M1_TWO_GPU_PD_SEPARATION.md](M1_TWO_GPU_PD_SEPARATION.md)
