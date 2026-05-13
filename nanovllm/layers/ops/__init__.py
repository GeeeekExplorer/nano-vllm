# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang

"""
Optimized GatedDeltaNet operations using Triton kernels.
"""

import torch

# Set Triton allocator to use PyTorch CUDA caching allocator
try:
    import triton
    triton.set_allocator(lambda size, device, stream: torch.cuda.caching_allocator_alloc(size))
except Exception:
    pass

try:
    from nanovllm.layers.ops.fused_recurrent import fused_recurrent_gated_delta_rule
    __all__ = ["fused_recurrent_gated_delta_rule"]
except ImportError:
    fused_recurrent_gated_delta_rule = None
    __all__ = []

try:
    from nanovllm.layers.ops.chunk import chunk_gated_delta_rule
    __all__.append("chunk_gated_delta_rule")
except ImportError:
    chunk_gated_delta_rule = None
