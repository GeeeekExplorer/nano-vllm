import torch
from torch import nn
import torch.nn.functional as F
from nanovllm.utils.compile import compile_if_supported


class SiluAndMul(nn.Module):

    @compile_if_supported
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
