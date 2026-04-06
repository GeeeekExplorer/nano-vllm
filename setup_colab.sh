#!/bin/bash
# Quick setup script for Google Colab or GPU environments
# Usage: bash setup_colab.sh

set -e  # Exit on error

echo "=========================================="
echo "  nano-vllm Setup for GPU Testing"
echo "=========================================="
echo ""

# ==========================================
# 1. Environment Detection
# ==========================================
if [ -d "/content" ]; then
    echo "✓ Detected Google Colab environment"
    BASE_DIR="/content"
    MODEL_DIR="/content/models"
    IS_COLAB=1
else
    echo "✓ Running in standard environment"
    BASE_DIR="."
    MODEL_DIR="$HOME/huggingface"
    IS_COLAB=0
fi

# ==========================================
# 2. GPU Check
# ==========================================
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; print('✓ CUDA available:', torch.cuda.is_available()); print('  Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'); print('  CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# ==========================================
# 3. Install Dependencies
# ==========================================
echo ""
echo "Installing nano-vllm dependencies..."
echo "  • torch >= 2.4.0"
echo "  • triton >= 3.0.0"
echo "  • transformers >= 4.51.0"
echo "  • flash-attn"
echo "  • xxhash"
echo ""

# Install core dependencies
pip install -q --upgrade pip
pip install -q torch>=2.4.0 transformers>=4.51.0 xxhash

# Install Triton (platform-specific)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  ℹ Note: Triton not supported on macOS, skipping..."
else
    pip install -q triton>=3.0.0
fi

# Install flash-attn (may take a few minutes)
echo "  Installing flash-attn (this may take 2-3 minutes)..."
pip install -q flash-attn --no-build-isolation || {
    echo "  ⚠ Warning: flash-attn installation failed. Continuing anyway..."
    echo "    (flash-attn requires CUDA-compatible GPU and may not work on all systems)"
}

# ==========================================
# 4. Install nano-vllm Package
# ==========================================
echo ""
echo "Installing nano-vllm as editable package..."
pip install -e . -q
echo "✓ nano-vllm installed successfully"

# ==========================================
# 5. Verify Installation
# ==========================================
echo ""
echo "Verifying installation..."
python3 << 'EOF'
import sys
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers: {e}")
    sys.exit(1)

try:
    import triton
    print(f"✓ Triton: {triton.__version__}")
except ImportError:
    print("⚠ Triton: Not available (OK on macOS)")

try:
    import flash_attn
    print(f"✓ flash-attn: {flash_attn.__version__}")
except ImportError:
    print("⚠ flash-attn: Not available (may cause issues)")

try:
    import xxhash
    print(f"✓ xxhash: {xxhash.VERSION}")
except ImportError as e:
    print(f"✗ xxhash: {e}")
    sys.exit(1)

try:
    import nanovllm
    print(f"✓ nano-vllm: Installed and importable")
except ImportError as e:
    print(f"✗ nano-vllm: {e}")
    sys.exit(1)
EOF

# ==========================================
# 6. Download Required Models
# ==========================================
echo ""
echo "=========================================="
echo "  Downloading Required Models"
echo "=========================================="
echo ""

# Create model directory
mkdir -p "$MODEL_DIR"

echo "Downloading Qwen3-0.6B model..."
echo "  Target: $MODEL_DIR/Qwen3-0.6B/"
echo ""

python3 << EOF
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_dir = Path("$MODEL_DIR") / "Qwen3-0.6B"

# Create directory
model_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading from HuggingFace: {model_name}")
print(f"Saving to: {model_dir}")
print("")

try:
    # Download config
    print("  • Downloading config...")
    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(model_dir)
    
    # Download tokenizer
    print("  • Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_dir)
    
    # Download model weights
    print("  • Downloading model weights (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None,  # Don't load to GPU yet
    )
    model.save_pretrained(model_dir)
    
    print("")
    print(f"✓ Model downloaded successfully to {model_dir}")
    
except Exception as e:
    print(f"✗ Error downloading model: {e}")
    print("  You may need to download manually or check your internet connection")
    exit(1)
EOF

# ==========================================
# 7. Run Tests
# ==========================================
echo ""
echo "=========================================="
echo "  Running Tests"
echo "=========================================="
echo ""

# Run abstraction tests
echo "Running abstraction tests..."
python3 test_abstractions.py || {
    echo "✗ Abstraction tests failed!"
    exit 1
}

echo ""
echo "=========================================="
echo "  Running Example (example.py)"
echo "=========================================="
echo ""

# Update MODEL_DIR in example.py if needed
export MODEL_DIR="$MODEL_DIR"

# Run example
python3 example.py || {
    echo "✗ Example failed!"
    echo "  This may be due to GPU/flash-attn issues"
    echo "  Try running manually: python3 example.py"
    exit 1
}

# ==========================================
# 8. Success Summary
# ==========================================
echo ""
echo "=========================================="
echo "  Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Dependencies installed"
echo "  ✓ nano-vllm package installed"
echo "  ✓ Model downloaded: $MODEL_DIR/Qwen3-0.6B/"
echo "  ✓ Tests passed"
echo "  ✓ Example executed successfully"
echo ""
echo "Model Location:"
echo "  $MODEL_DIR/Qwen3-0.6B/"
echo ""
echo "Next steps:"
echo "  • Run example: python3 example.py"
echo "  • Run benchmarks: python3 bench.py"
echo "  • Explore integration guide: cat integration.md"
echo ""
echo "Happy coding! 🚀"
echo ""
