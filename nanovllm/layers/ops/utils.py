# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import contextlib
import functools
import os
from collections.abc import Callable
from typing import Any

import torch

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

SUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))
FLA_GDN_FIX_BT = os.getenv("FLA_GDN_FIX_BT", "0") == "1"

# Check if CUDA graphs should be used (simplified check)
use_cuda_graph = os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"

# Check if gather is supported
try:
    import triton.language as tl
    is_gather_supported = hasattr(tl, "gather")
except ImportError:
    is_gather_supported = False

# Simplified platform checks
try:
    is_nvidia = torch.cuda.is_available()
    is_nvidia_hopper = is_nvidia and (
        "NVIDIA H" in torch.cuda.get_device_name(0)
        or torch.cuda.get_device_capability()[0] >= 9
    )
except Exception:
    is_nvidia = False
    is_nvidia_hopper = False

try:
    is_amd = hasattr(torch.version, "hip") and torch.version.hip is not None
except Exception:
    is_amd = False

try:
    is_intel = hasattr(torch, "xpu") and torch.xpu.is_available()
except Exception:
    is_intel = False

is_intel_alchemist = False
if is_intel:
    try:
        is_intel_alchemist = "Intel(R) Arc(TM) A" in torch.xpu.get_device_name(0)
    except Exception:
        pass

# Check TMA support
try:
    import triton.language as tl
    is_tma_supported = (
        is_nvidia
        and torch.cuda.get_device_capability(0)[0] >= 9
        and (
            hasattr(tl, "_experimental_make_tensor_descriptor")
            or hasattr(tl, "make_tensor_descriptor")
        )
    )
except Exception:
    is_tma_supported = False

# Check shared memory
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    """Simplified shared memory check."""
    # Default to True for most cases
    return True


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.
    """
    cache_entries: list[tuple[tuple | None, dict | None, Any]] = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, cache_size
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(a is b for a, b in zip(args, last_args))
                and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                )
            ):
                cache_entries = (
                    cache_entries[:i]
                    + cache_entries[i + 1 :]
                    + [(args, kwargs, last_result)]
                )
                return last_result

        result = fn(*args, **kwargs)

        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.cuda.device(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper
