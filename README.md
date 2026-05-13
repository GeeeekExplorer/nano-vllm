<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.
* 🖼️ **Multimodal support** - Qwen3.5 text + vision with CUDA graph decode

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

### Qwen3 (text-only)
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

### Qwen3.5 (multimodal)
```bash
huggingface-cli download --resume-download Qwen/Qwen3.5-0.8B \
  --local-dir ~/huggingface/Qwen3.5-0.8B/ \
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

## Qwen3.5 Quick Start

`example_qwen3_5.py` demonstrates both text-only and multimodal (image+text) inference. CUDA graph decode is enabled by default for maximum throughput.

### Text-only batch
```bash
python example_qwen3_5.py --mode text --model ~/huggingface/Qwen3.5-0.8B/
```

### Multimodal (image + text)
```bash
python example_qwen3_5.py --mode multimodal --model ~/huggingface/Qwen3.5-0.8B/
```

### Python API
```python
from nanovllm import LLM, SamplingParams
from transformers import AutoProcessor

llm = LLM(
    "~/huggingface/Qwen3.5-0.8B/",
    is_multimodal=True,
    enforce_eager=False,  # CUDA graph decode by default
)
processor = AutoProcessor.from_pretrained("~/huggingface/Qwen3.5-0.8B/")

# Text-only
outputs = llm.generate([prompt_text], SamplingParams(temperature=0.6, max_tokens=256))

# Multimodal
requests = [{"text": chat_prompt, "images": [image]}]
outputs = llm.generate_multimodal(
    requests,
    [SamplingParams(temperature=0.6, max_tokens=256)],
    processor,
)
```

## Benchmark

See `bench.py` for Qwen3 benchmark and `bench_qwen3_5.py` for Qwen3.5 benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Qwen3 Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |

**Qwen3.5 Benchmark Commands:**
```bash
# Text-only throughput (same setup as Qwen3 benchmark)
python bench_qwen3_5.py --mode text --num-seqs 256 --temperature 0.6 --ignore-eos --seed 0 --model ~/huggingface/Qwen3.5-0.8B/

# Multimodal throughput
python bench_qwen3_5.py --mode multimodal --num-images 10 --model ~/huggingface/Qwen3.5-0.8B/
```

**Qwen3.5 Performance Results (Nano-vLLM only):**

| Mode | Requests | Prompt Tokens | Generated Tokens | Time (s) | Throughput (tokens/s) |
|------|----------|---------------|------------------|----------|-----------------------|
| Text (CUDA graph) | 256 | 142,827 | 133,966 | 162.23 | 825.79 |
| Multimodal (CUDA graph) | 10 | 2,998 | 5,114 | 11.18 | 457.38 |

*Configuration: H20, Qwen3.5-0.8B, temperature=0.6, ignore_eos=True (text).*

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)