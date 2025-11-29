import torch
from torch import nn

"""
RMSnorm与Layernorm相比，省去了减均值平移的操作，更高效
"""
class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6, # eps 是稳定项，用于避免除零或数值不稳定
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size)) # 缩放因子初始化为 1，起到逐通道缩放的作用，允许网络学习每个特征的最佳尺度

    @torch.compile # 注意这里的compile对调试的时候会有影响，x.float()时会出bug，调试需要强制eager
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype # 变type的时候先存下原type，可以在函数内使用更稳定或更快的浮点精度（如 float32）计算，但在接口上保持原始 dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True) # 平方又累加容易溢出
        x.mul_(torch.rsqrt(var + self.eps)) # rsqrt为平方根再倒数，此步后x已被归一化
        x = x.to(orig_dtype).mul_(self.weight) # 前面换高精度是为了避免数据溢出风险，而x又已被归一化，数值分布稳定，故低精度足以表达，就先降精度保性能了
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: # 对 x 与 residual 相加后再做 RMS 归一化
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype) # 比上面多了这两行
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None: # 如果没有传入 residual，走简单的 RMS 路径，使 RMSNorm 类更通用
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
