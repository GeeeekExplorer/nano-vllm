# Setup Script Documentation

## Overview

The `setup_colab.sh` script provides automated setup for nano-vllm in both Google Colab and local environments.

## What It Does

### 1. Environment Detection
- Detects if running on Google Colab or local machine
- Sets appropriate model directories:
  - Colab: `/content/models/`
  - Local: `~/huggingface/`

### 2. GPU Verification
- Checks CUDA availability
- Reports GPU device name and CUDA version
- Continues even if GPU is not available (for testing)

### 3. Dependency Installation
Installs all required packages from `pyproject.toml`:
- `torch >= 2.4.0`
- `transformers >= 4.51.0`
- `triton >= 3.0.0` (skipped on macOS)
- `flash-attn` (may fail on systems without CUDA - script continues anyway)
- `xxhash`

### 4. Package Installation
- Installs nano-vllm as an editable package: `pip install -e .`
- This allows you to:
  - Import nano-vllm from anywhere
  - Make changes to the code and see them immediately
  - Run examples without path issues

### 5. Model Download
- Downloads **Qwen2.5-0.5B-Instruct** from HuggingFace
- Saves as `Qwen3-0.6B` in the model directory
- Downloads:
  - Model config
  - Tokenizer
  - Model weights

### 6. Testing
Runs two test suites:

#### a) `test_abstractions.py`
Tests the core registry system:
- Model registry (Qwen3 registered)
- KV cache registry (default cache registered)
- Flash attention registry (default backend registered)
- Base interface definitions

#### b) `example.py`
Full end-to-end test:
- Loads the downloaded model
- Uses **default KV cache** (FP16/BF16, no quantization)
- Uses **default flash attention backend**
- Generates text for sample prompts

## Usage

### Google Colab
```bash
!git clone https://github.com/your-repo/nano-vllm.git
%cd nano-vllm
!bash setup_colab.sh
```

### Local Machine
```bash
git clone https://github.com/your-repo/nano-vllm.git
cd nano-vllm
bash setup_colab.sh
```

## What Gets Created

```
~/huggingface/Qwen3-0.6B/          # Downloaded model (local)
/content/models/Qwen3-0.6B/         # Downloaded model (Colab)

nano-vllm/                          # Repository
├── nanovllm/                       # Installed as package
├── example.py                      # Updated to auto-detect model
├── test_abstractions.py            # Updated with new tests
└── setup_colab.sh                  # This script ✓
```

## Features of Updated Files

### `example.py`
- **Auto-detects model path** from multiple locations:
  1. Environment variable `MODEL_DIR`
  2. `~/huggingface/Qwen3-0.6B/`
  3. `/content/models/Qwen3-0.6B/` (Colab)
  4. `./models/Qwen3-0.6B/` (local)

- **Explicitly uses defaults**:
  - KV cache: `"default"` (FP16/BF16, no quantization)
  - Attention: `DefaultFlashAttention` backend

- **Better output formatting**:
  - Shows model path
  - Numbered query/response pairs
  - Separated sections

### `test_abstractions.py`
New tests added:
- ✅ Flash attention backend registry test
- ✅ Default KV cache verification
- ✅ Default flash attention backend verification
- ✅ Simplified config test (doesn't require model files)

## Troubleshooting

### flash-attn Installation Fails
This is expected on systems without CUDA. The script continues and nano-vllm may still work for testing purposes.

### Model Download Fails
- Check internet connection
- Try downloading manually: `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct`
- Set `HF_TOKEN` environment variable if using gated models

### GPU Not Available
The script runs CPU tests only. For GPU testing:
1. Ensure CUDA is installed
2. Verify with: `python -c "import torch; print(torch.cuda.is_available())"`
3. Re-run the script

### Import Errors
If `import nanovllm` fails after setup:
- Verify installation: `pip show nano-vllm`
- Re-install: `pip install -e .`
- Check Python version: `python --version` (requires 3.10-3.12)

## After Setup

You can now:
1. **Run the example**: `python example.py`
2. **Run benchmarks**: `python bench.py`
3. **Import nano-vllm**:
   ```python
   from nanovllm import LLM, SamplingParams
   llm = LLM("~/huggingface/Qwen3-0.6B/")
   ```
4. **Extend the framework**: See `integration.md`

## Clean Up

To remove downloaded models:
```bash
# Local
rm -rf ~/huggingface/Qwen3-0.6B/

# Colab
!rm -rf /content/models/
```

To uninstall nano-vllm:
```bash
pip uninstall nano-vllm
```
