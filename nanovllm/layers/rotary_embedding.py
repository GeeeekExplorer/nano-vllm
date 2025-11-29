from functools import lru_cache # 导入LRU用于缓存函数的返回结果，避免重复计算
import torch
from torch import nn


def apply_rotary_emb( # 此函数执行实际的旋转操作，把一个点 (x, y) 旋转θ角度，新的坐标 (x', y') 就是套用这个公式
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1) # 三角函数乘法，在半精度（fp16/bf16）下容易溢出或精度不足，故转float32；RoPE 将向量最后一个维度（特征维）切分为x1x2来模拟复数旋转
    y1 = x1 * cos - x2 * sin # 二维平面向量的旋转公式
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype) # 拼回去


class RotaryEmbedding(nn.Module): # 实现 RoPE（rotary positional embeddings），给 attention 的 Q/K 注入位置编码（高效、可缓存）

    def __init__(
        self,
        head_size: int,
        rotary_dim: int, # 需要进行旋转操作的维度大小（通常等于 head_size）
        max_position_embeddings: int, # 预计算的最大序列长度
        base: float, # 计算频率的基底（通常为 10000.0，长文本模型可能会更大），目的是确保部分向量不会转重
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size # 强制全旋转
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)) # inv_freq 是一个长度为 dim/2 的向量，表示不同维度的旋转频率（一系列几何级数递减的频率）
        t = torch.arange(max_position_embeddings, dtype=torch.float) # 绝对位置索引向量 t
        freqs = torch.einsum("i,j -> ij", t, inv_freq) # freqs[m, i] 存储了第 m 个 Token 在第 i 组旋转子空间中应该旋转的角度
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1) # 中间插入的这个 1，是为了让位置编码自动广播到所有的 Attention Heads 上，即同一个位置的 Token，在不同的 Attention Head 中使用的是相同的位置旋转角度；简单说就是在 y1 = x1 * cos - x2 * sin 乘的时候广播
        self.register_buffer("cos_sin_cache", cache, persistent=False) # 算 cos 和 sin 费时间，既然每个位置转的角度固定，就提前算好存进一张表里，用时查表即可；persistent=False:表示保存模型权重时不存这个，减小权重显存占用

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor, # 当前输入的 Token 索引，为整型张量batchsize*seqlength，内容是token绝对位置
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions] # 从预计算的缓存中取出对应的 cos 和 sin 组合
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin) # 分别对 Query 和 Key 进行旋转
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope( # 如果传入相同的参数，会直接返回之前创建好的 RotaryEmbedding 对象，而不是重新创建
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None # 为了简化代码直接禁用了长文本外推
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
