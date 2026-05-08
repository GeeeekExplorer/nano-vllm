<p align="center">
<img width="300" src="assets/logo.png">
</p>


# Nano-vLLM-kv-compression

An improved implementation based on Nano-vLLM featuring int8 KV cache compression, head-major memory layout for coalesced access, and asynchronous stream pipelining that hides KV store latency behind attention computation.
## Key Features

* ⚡ **Int8 KV Cache Compression** — 50% memory reduction via dynamic per-head quantization
* 🔄 **Coalesced Layout** — Head-major reordering for warp-level memory coalescing
* 🎯 **GQA-Optimized Flash Attention** — group Q-head CTA mapping eliminates redundant KV loads
* 🔗 **Async KV Store Pipeline** — Multi-stream architecture overlaps KV quantization and cache writeback with attention computation

## Installation

```bash
pip install git+https://github.com/naalo2/nano-vLLM-kv-compression.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 3090 (24GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine          | Output Tokens | Time (s) | Throughput (tokens/s) |
|---------------------------|---------------|----------|-----------------------|
| Nano-vLLM                 | 133,966       | 33.05    | 4052.56               |
| Nano-vLLM-kv-compression  | 133,966       | 27.00    | 4962.21               |


