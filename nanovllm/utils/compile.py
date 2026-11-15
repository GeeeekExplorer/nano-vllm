import sys
import torch


def compile_if_supported(function):
    # The Windows path uses PyTorch's CUDA kernels without Triton/Inductor.
    return function if sys.platform == "win32" else torch.compile(function)
