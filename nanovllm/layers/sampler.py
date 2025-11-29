import torch
from torch import nn

"""
使用了 Gumbel-Max 技巧来实现高效的随机采样
"""
class Sampler(nn.Module):

    def __init__(self):
        super().__init__() # 调用父类的构造函数

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor): # 一个 Batch 往往包含来自不同用户的多个请求，温度为Tensor，就可以让每个样本使用其专有的温度
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) # 将温度张量的形状从 (batch_size) 变为 (batch_size, 1)，以便进行广播（broadcasting）操作
        probs = torch.softmax(logits, dim=-1) # 将其在词表维度上转换为概率分布
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1) # 通过概率分布除以随机噪声来保证生成的token的随机性
        return sample_tokens
