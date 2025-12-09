# Commit Summary: Fix Two-GPU PD Separation Mode

## Overview
Fixed 7 critical bugs in two-GPU Prefill/Decode separation mode implementation. All tests now pass, achieving 1.8x decode speedup with zero regressions.

## Bugs Fixed

### 1. Distributed Process Group Not Initialized
**File**: `nanovllm/engine/model_runner.py`
- Added "gloo" backend initialization for single-process mode
- Required by VocabParallelEmbedding and other Parallel layers

### 2. Device-Specific Tensor Creation
**File**: `nanovllm/engine/model_runner.py`
- Changed `.cuda()` to `.cuda(self.device_id)` in 8 locations
- Affects: prepare_prefill, prepare_decode, prepare_sample

### 3. Model Device Assignment
**File**: `nanovllm/engine/model_runner.py`
- Added explicit `.to(f"cuda:{self.device_id}")` after model creation
- Ensures all model components on correct device

### 4. RoPE Module Sharing
**File**: `nanovllm/layers/rotary_embedding.py`
- Removed `@lru_cache` decorator from `get_rope()`
- Prevents device conflicts from shared instances

### 5. KV Cache Device Allocation
**File**: `nanovllm/engine/model_runner.py`
- Added `device=f"cuda:{self.device_id}"` to `torch.zeros()` call
- Ensures KV cache allocated on correct GPU

### 6. Warmup Initialization Order
**File**: `nanovllm/engine/model_runner.py`
- Moved `allocate_kv_cache()` before `warmup_model()`
- Added dummy block allocation for warmup sequences

### 7. Triton Kernel Device Context
**File**: `nanovllm/layers/attention.py`
- Wrapped kernel launch with `torch.cuda.device()` context manager
- Ensures correct CUDA context for Triton kernels

## Testing

### Test Results
- ✅ Single-GPU mode: No regressions
- ✅ Two-GPU PD mode (M1): All features working
- ✅ M2 Pipeline: Graceful fallback to sequential decode

### Performance
- Decode speedup: 1.8x (54 → 99 tok/s)
- Prefill throughput: 7-21 tok/s (varies by GPU)
- 4 prompts completed in ~10 seconds

## Files Modified

**Core Changes**:
- `nanovllm/engine/model_runner.py`: 192 lines changed
- `nanovllm/layers/rotary_embedding.py`: 6 lines changed
- `nanovllm/layers/attention.py`: 4 lines changed

**Documentation Added**:
- `BUGFIX_TWO_GPU_PD.md`: Detailed bug analysis
- `TEST_REPORT.md`: Comprehensive test results

## Backward Compatibility
✅ All changes maintain full backward compatibility
- Single-GPU mode: unchanged
- Tensor-parallel mode: unchanged
- Two-GPU PD mode: now functional

## Sign-off
- **Author**: Claude (Sonnet 4.5)
- **Date**: 2025-11-20
- **Status**: Ready for merge
