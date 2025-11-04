import torch
from torch import nn
import torch.nn.functional as F

# fused kernel
class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)  # 将x在最后维度hidden 等额分成2份
        return F.silu(x) * y   # 对第一部分进行silu激活，并于第二部分逐元素相乘
