# nano-vLLM

Lightweight Qwen3 inference on CUDA and Apple Silicon.

CUDA uses FlashAttention, Triton KV writes, CUDA graphs, and tensor parallelism. The MPS path uses Metal kernels for KV writes, Q/K normalization with rotary embedding, and paged decode attention. MPS runs on one GPU; prefill and unsupported shapes use PyTorch attention.

## Setup

Use Python 3.10–3.12. On macOS, install a PyTorch build with `torch.mps.compile_shader`.

```bash
python -m pip install -e .
```

Put [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) in `Qwen3-0.6B/`, or pass another local checkpoint with `--model`.

## Run

```bash
python example.py
python bench.py --model Qwen3-0.6B --device mps --gpu-memory-utilization 0.64
```

The MPS memory setting above was measured on a 48 GB M4 Pro. Adjust it for your machine. `example.py` also accepts `--device cuda`.

The benchmark uses the upstream seeded workload: 256 requests, input and output lengths of 100–1024 tokens, and 133,966 generated tokens.

| System | Time | Output throughput |
| --- | ---: | ---: |
| M4 Pro, 48 GB, MPS | 127.77 s | 1,048.51 tokens/s |
| RTX 4070 Laptop, 8 GB, CUDA ([upstream result](https://github.com/GeeeekExplorer/nano-vllm#benchmark)) | 93.41 s | 1,434.13 tokens/s |

These figures compare the same workload on different hardware. On the M4 Pro, the optimized MPS path measured 1,007.54 tokens/s against 911.15 tokens/s for the earlier path with the same `0.67` memory setting.

For a shorter local run, use `python -m benchmarks.bench_mps Qwen3-0.6B`. For one attention layer, use `python -m benchmarks.bench_attention_mps`.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

MPS tests cover KV writes, attention, prefix reuse, and generation. CUDA regression tests require a CUDA machine. The random checkpoint generator is `python -m tools.make_tiny_qwen3 /tmp/nano-vllm-tiny-qwen3`.

MIT licensed. See [LICENSE](LICENSE).
