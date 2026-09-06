import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # The softmax normalizer is row-constant, so sample in logit space.
        noise = torch.empty_like(logits).exponential_(1).clamp_min_(1e-10)
        sample_tokens = logits.sub_(noise.log_()).argmax(dim=-1)
        return sample_tokens
