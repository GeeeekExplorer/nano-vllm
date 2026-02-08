import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32, copy=True)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32.mul_(torch.rsqrt(var + self.eps))
        x = x_fp32.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32, copy=True)

        x_fp32 = x_fp32.add_(residual.float())
        residual = x_fp32.to(orig_dtype, copy=True)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32.mul_(torch.rsqrt(var + self.eps))
        x = x_fp32.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
