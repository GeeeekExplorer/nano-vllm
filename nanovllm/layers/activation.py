import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module): # SiLU，流行MLP激活方式，比传统的 ReLU 激活函数具有更强的表达能力，模块本身无参数（只是函数式组合），可被其它有参数的层（线性层）包裹使用

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor: # 这个层通常接在一个输出维度翻倍的线性层后面
        x, y = x.chunk(2, -1) # 输入 x 的最后一维必须能被 2 整除（因为此函数把最后一维均分为两部分）
        return F.silu(x) * y # 对第一半 x 应用 SiLU（又名 swish）激活，然后与第二半 y 做逐元素相乘（gated multiplication），输出和 x/y 的形状一致（即输入最后维度减半）
