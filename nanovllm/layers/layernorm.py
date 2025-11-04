import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [batch, seq_len, hidden_dim]
        orig_dtype = x.dtype  # 保存dtype
        x = x.float()         # 转到float32
        var = x.pow(2).mean(dim=-1, keepdim=True) # 计算均方差,在hidden维度上进行归一化 var [batch, seq_len, 1]
        x.mul_(torch.rsqrt(var + self.eps)) # x = x / sqrt(var + eps)
        x = x.to(orig_dtype).mul_(self.weight) # 转回初始dtype      x = x * weight，在原tensor上进行修改
        return x

    # 残差+norm
    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())  # x = x + residual
        residual = x.to(orig_dtype)           # 再做常规RMSNorm    
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    # 若传入residual则做融合add+norm, 若没有传入，只做norm
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
