# Integration Guide for nano-vllm

This guide explains how to extend nano-vllm with new models, KV cache implementations, and attention mechanisms.

---

## Architecture Overview

nano-vllm uses a **registry-based architecture** with three pluggable layers:

```
Models (inference logic)
   ↓
Attention (computation)
   ↓  
KV Cache (storage)
```

**Key principle**: Everything is registered in a central registry and discovered by name.

---

## Adding New Models

### What You Need to Implement

Your model must inherit from `BaseModel` and implement:
- `forward(input_ids, positions)` → returns hidden states
- `compute_logits(hidden_states)` → returns vocabulary logits

### Where to Define

Create a new file: `nanovllm/models/your_model.py`

```python
from nanovllm.models.base import BaseModel
from nanovllm.models.registry import ModelRegistry

@ModelRegistry.register("llama", architectures=["LlamaForCausalLM"])
class LlamaForCausalLM(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Build your model architecture here
        # Use Attention() module from nanovllm.layers.attention
    
    def forward(self, input_ids, positions):
        # Forward pass logic
        return hidden_states
    
    def compute_logits(self, hidden_states):
        # Project to vocabulary
        return logits
```

### How to Register

Use the `@ModelRegistry.register()` decorator with:
- **name**: Internal identifier (e.g., `"llama"`)
- **architectures**: HuggingFace config names (e.g., `["LlamaForCausalLM"]`)

### Where to Export

Add to `nanovllm/models/__init__.py`:
```python
from nanovllm.models.your_model import YourModel
```

### How to Use

```python
from nanovllm import LLM

# Automatic detection from HuggingFace config
llm = LLM(model="meta-llama/Llama-3-8B")

# Or explicit model name
llm = LLM(model="meta-llama/Llama-3-8B", model_name="llama")
```

### Important Notes

- **KV Cache**: Use the `Attention()` module from `nanovllm.layers.attention` - it handles caching automatically
- **Weight Packing**: If your model packs weights (e.g., QKV in one tensor), define `packed_modules_mapping`:
  ```python
  packed_modules_mapping = {
      "q_proj": ("qkv_proj", "q"),
      "k_proj": ("qkv_proj", "k"),
      "v_proj": ("qkv_proj", "v"),
  }
  ```

---

## Adding New KV Cache Types

### What You Need to Implement

Your cache must inherit from `BaseKVCache` and implement:
- `allocate()` → returns cache tensors (k_cache, v_cache, optional scales/zeros)
- `store(key, value, k_cache, v_cache, slot_mapping, *additional_tensors)` → stores (and optionally quantizes) KV pairs
- `retrieve(k_cache, v_cache, *additional_tensors)` → retrieves (and optionally dequantizes) KV pairs

### Where to Define

Create a new file: `nanovllm/kvcache/your_cache.py`

```python
from nanovllm.kvcache.base import BaseKVCache, KVCacheRegistry

@KVCacheRegistry.register("int8")
class INT8KVCache(BaseKVCache):
    def allocate(self):
        # Allocate INT8 storage + FP16 scales
        k_cache = torch.empty(..., dtype=torch.int8)
        v_cache = torch.empty(..., dtype=torch.int8)
        k_scales = torch.empty(..., dtype=torch.float16)
        v_scales = torch.empty(..., dtype=torch.float16)
        return k_cache, v_cache, k_scales, v_scales
    
    def store(self, key, value, k_cache, v_cache, slot_mapping, k_scales, v_scales):
        # Quantize FP16 → INT8 and store
        # Use Triton kernel for efficiency
        pass
    
    def retrieve(self, k_cache, v_cache, k_scales, v_scales):
        # Dequantize INT8 → FP16
        k_fp16 = k_cache.to(self.dtype) * k_scales
        v_fp16 = v_cache.to(self.dtype) * v_scales
        return k_fp16, v_fp16
    
    def needs_dequantization(self):
        return True  # INT8 requires dequant
```

### How to Register

Use the `@KVCacheRegistry.register()` decorator with a name (e.g., `"int8"`).

### Where to Export

Add to `nanovllm/kvcache/__init__.py`:
```python
from nanovllm.kvcache.your_cache import YourCache
```

### How to Use

```python
from nanovllm import LLM

llm = LLM(
    model="Qwen/Qwen-7B",
    kv_cache_dtype="int8",  # Registry lookup by name
)
```

### Important Notes

- **Additional Tensors**: Use `*additional_tensors` for scales, zero-points, etc.
- **Memory**: Override `get_cache_block_size_bytes()` for accurate memory planning
- **Performance**: For best speed, pair with a custom attention backend (see next section)

---

## Adding New Attention Mechanisms

### What You Need to Implement

Your backend must inherit from `BaseFlashAttentionBackend` and implement:
- `prefill(q, k, v, ...)` → attention during prompt processing
- `decode(q, k_cache, v_cache, ...)` → attention during token generation

### Where to Define

Create a new file: `nanovllm/layers/your_attention.py`

```python
from nanovllm.layers.flash_attn_backend import BaseFlashAttentionBackend, FlashAttentionRegistry

@FlashAttentionRegistry.register("int8")
class INT8FlashAttention(BaseFlashAttentionBackend):
    def prefill(self, q, k, v, scale, max_seqlen_q, cu_seqlens_q, ...):
        # Prefill: use standard flash-attn (K/V not cached yet)
        from flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func(q, k, v, ...)
    
    def decode(self, q, k_cache, v_cache, scale, cache_seqlens, block_table,
               k_scales, v_scales):  # Extra tensors for quantized cache
        # Decode: custom kernel that reads INT8 cache directly
        return custom_int8_kernel(q, k_cache, v_cache, k_scales, v_scales, ...)
```

### How to Register

Use the `@FlashAttentionRegistry.register()` decorator with a name (e.g., `"int8"`).

### Where to Use

Set the backend when creating an `Attention` module:

```python
from nanovllm.layers.attention import Attention
from nanovllm.layers.your_attention import YourAttention

attn = Attention(
    num_heads=32,
    head_dim=128,
    scale=0.0883,
    num_kv_heads=32,
    attn_backend=YourAttention(),  # Custom backend
)
```

Or configure it in your model's attention layer:

```python
class MyModelAttention(nn.Module):
    def __init__(self, config):
        use_int8 = getattr(config, "kv_cache_dtype", "default") == "int8"
        
        attn_backend = INT8FlashAttention() if use_int8 else None
        
        self.attn = Attention(..., attn_backend=attn_backend)
```

### Important Notes

- **Prefill vs Decode**: Different implementations are often needed:
  - **Prefill**: K/V are not cached yet (use standard attention)
  - **Decode**: K/V are in cache (read from paged storage)
- **Additional Tensors**: Use `*additional_cache_tensors` parameter to receive scales, zero-points, etc. from quantized caches
- **Shape Handling**: Handle both 2D and 3D query tensors properly

---

## Combining Components

### Example: INT8 Cache + INT8 Attention

**Best practice**: Pair quantized caches with matching attention backends for optimal performance.

**Directory structure:**
```
nanovllm/
├── kvcache/
│   ├── int8.py               # INT8KVCache
│   └── __init__.py
├── layers/
│   ├── int8_flash_attn.py    # INT8FlashAttention
│   └── __init__.py
```

**Auto-pairing in model:**
```python
class MyModelAttention(nn.Module):
    def __init__(self, config):
        cache_type = getattr(config, "kv_cache_dtype", "default")
        
        # Auto-select matching backend
        if cache_type == "int8":
            attn_backend = INT8FlashAttention()
        else:
            attn_backend = None  # Use default
        
        self.attn = Attention(..., attn_backend=attn_backend)
```

**Usage:**
```python
llm = LLM(model="Qwen/Qwen-7B", kv_cache_dtype="int8")
```

This automatically uses:
- `INT8KVCache` for storage (via registry lookup)
- `INT8FlashAttention` for computation (via model config)

---

## Quick Reference

### Models
| Step | Action |
|------|--------|
| **Define** | `nanovllm/models/your_model.py` |
| **Inherit** | `BaseModel` |
| **Implement** | `forward()`, `compute_logits()` |
| **Register** | `@ModelRegistry.register("name", architectures=[...])` |
| **Export** | Add to `nanovllm/models/__init__.py` |
| **Use** | `LLM(model="...", model_name="name")` |

### KV Caches
| Step | Action |
|------|--------|
| **Define** | `nanovllm/kvcache/your_cache.py` |
| **Inherit** | `BaseKVCache` |
| **Implement** | `allocate()`, `store()`, `retrieve()` |
| **Register** | `@KVCacheRegistry.register("name")` |
| **Export** | Add to `nanovllm/kvcache/__init__.py` |
| **Use** | `LLM(model="...", kv_cache_dtype="name")` |

### Attention Backends
| Step | Action |
|------|--------|
| **Define** | `nanovllm/layers/your_attention.py` |
| **Inherit** | `BaseFlashAttentionBackend` |
| **Implement** | `prefill()`, `decode()` |
| **Register** | `@FlashAttentionRegistry.register("name")` |
| **Use** | `Attention(..., attn_backend=YourBackend())` |

---


## Summary

nano-vllm's extension pattern:

1. **Inherit** from base class (`BaseModel`, `BaseKVCache`, `BaseFlashAttentionBackend`)
2. **Implement** required methods
3. **Register** with decorator (`@Registry.register("name")`)
4. **Export** in package `__init__.py`
5. **Use** via registry name

Everything is discovered through registries - no need to modify core code.
