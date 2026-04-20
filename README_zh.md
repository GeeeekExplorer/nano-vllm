<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
<a href="README.md">English</a> | 简体中文
</p>

# Nano-vLLM

一个从零实现的轻量级 vLLM。

## 主要特性

* 🚀 **高速离线推理** - 推理速度可与 vLLM 相媲美
* 📖 **代码易读** - 约 1,200 行 Python 的简洁实现
* ⚡ **优化特性齐全** - Prefix caching、Tensor Parallelism、Torch compilation、CUDA graph 等

## 安装

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## 模型下载

如需手动下载模型权重，可使用以下命令：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 快速开始

使用方式可参考 `example.py`。API 基本与 vLLM 保持一致，只有 `LLM.generate` 方法存在少量差异：

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## 基准测试

基准脚本见 `bench.py`。

**测试配置：**
- 硬件：RTX 4070 Laptop (8GB)
- 模型：Qwen3-0.6B
- 总请求数：256 个序列
- 输入长度：在 100 到 1024 个 token 之间随机采样
- 输出长度：在 100 到 1024 个 token 之间随机采样

**性能结果：**
| 推理引擎 | 输出 Token 数 | 耗时 (s) | 吞吐量 (tokens/s) |
|---------|--------------|----------|-------------------|
| vLLM    | 133,966      | 98.37    | 1361.84           |
| Nano-vLLM | 133,966    | 93.41    | 1434.13           |

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
