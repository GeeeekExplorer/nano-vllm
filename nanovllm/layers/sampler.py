from re import T
import torch
from torch import nn


# temperature 是控制“随机性/多样性”的参数。它通过调整 softmax 的“尖锐程度”改变分布熵。
# 公式：
# - 原始分布： p_i = softmax(z_i)
# - 加温度： p_i(T) = softmax(z_i / T)
# 效果：
# - T < 1 ：分布更尖锐，更偏向最大 logit（更“确定”、更像 greedy）
# - T = 1 ：原始分布
# - T > 1 ：分布更平坦，更随机（更多样但更容易跑偏）
# 你可以把它想成“把 logits 的差距放大或缩小”：
# - 除以小 T（比如 0.5）→ logits 差距变大 → top token 概率更接近 1
# - 除以大 T（比如 1.5）→ logits 差距变小 → 次优 token 更容易被采到
# 为什么在工程里必须有 temperature：
# - 生成任务很多不是“唯一正确答案”，需要多样性（对话、创作、头脑风暴）。
# - 但也不能太随机，否则会胡说。temperature 提供了一个简单、通用、可控的旋钮。
# nano-vllm 的 sampler 只做 temperature sampling；生产级系统常加 top-k/top-p、repetition penalty、min/max length、stop tokens、logprobs 等。

# temperature的示例 e.g.
# - 假设某步 logits 为： [2.0, 1.0, 0.5, 0.0, -1.0]
# - T=0.7 （更“保守”）
# - 缩放后： [2.857, 1.429, 0.714, 0, -1.429]
# - softmax 约： [0.700, 0.168, 0.082, 0.040, 0.010]
#   - 概率 0.7 的意思是： 重复很多次独立采样时，平均约 70% 会选中它 ，不是“单次一定选中”。
#   - 单次采样里它仍有 30% 概率不被选中（因为其余 token 总概率是 0.3）。
#   如果想“几乎每次都选 0.7 那个”，要么：
#   - 降低 temperature（让分布更尖），或
#   - 直接用 greedy（argmax，不做随机采样）。
# - 头部 token 概率很高，输出更稳定。
# - T=1.3 （更“发散”）
# - 缩放后： [1.538, 0.769, 0.385, 0, -0.769]
# - softmax 约： [0.478, 0.221, 0.151, 0.103, 0.047]
# - 分布更平，次优 token 更容易被采到，多样性提升。
# - 直觉： T 越小，分布越尖锐； T 越大，分布越平坦。

# temperature sampling和beam search都属于“解码策略（decoding strategy）”。
# - Temperature sampling ：在概率分布上按随机方式抽样（ temperature 控制随机程度）。
# - Beam search ：不是随机抽样，而是保留多个高分候选路径做启发式搜索，偏“找高概率序列”。
# 你可以这样记层级：
# - 顶层 ：解码策略
# - 分支 A（采样类） ：temperature、top-k、top-p 等（有随机性）
# - 分支 B（搜索类） ：beam search（通常更确定、计算更重）

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile # 让这个 forward 被 torch.compile 编译优化，减少 Python 开销、让采样这段更快，todo：是不是会做一些对计算图的调优
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 把 logits 转成 float32 做 softmax/采样，数值更稳定（fp16/bf16 做 softmax 容易溢出/精度差
        # 把 [batch] 变成 [batch, 1] ，方便广播到 [batch, vocab]，用于和logits计算。
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) 
        probs = torch.softmax(logits, dim=-1) # 把 logits 转成概率分布：每行和为 1

        # 这一行实现的是一种高效采样： Gumbel-Max / 指数噪声采样等价形式 。（todo：什么玩意？？？）
        # 直觉理解：
        # - 你想从 categorical 分布 probs 里抽样。
        # - 传统做法是 torch.multinomial(probs, 1) ，但可能更慢或有额外开销。
        # - 这里用“对每个候选 token 加随机性，再取 argmax”实现等价采样：
        # - exponential_(1) 生成与 probs 同形状的指数分布噪声 E
        # - probs / E 再 argmax ，等价于从 probs 采样一个类别（属于常见的采样技巧变体）
        # - clamp_min_ 防止除零
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
