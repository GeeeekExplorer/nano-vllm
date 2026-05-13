# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import triton
import triton.language as tl

# Use fast ops if available
if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    try:
        import triton.language.extra.libdevice as tldevice
        exp = tldevice.fast_expf
        log = tldevice.fast_logf
        log2 = tldevice.fast_log2f
    except ImportError:
        exp = tl.exp
        log = tl.log
        log2 = tl.log2
else:
    exp = tl.exp
    log = tl.log
    log2 = tl.log2

# Check if gather is supported
try:
    gather = tl.gather
except AttributeError:
    @triton.jit
    def gather(src, index, axis, _builder=None):
        """Fallback gather when tl.gather is not supported."""
        return None

# Check if make_tensor_descriptor is available
if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        """Fallback when TMA is not supported."""
        return None
