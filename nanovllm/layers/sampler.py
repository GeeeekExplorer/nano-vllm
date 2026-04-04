import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor):
        logits = logits.float()
        greedy_tokens = logits.argmax(dim=-1)
        temperatures = temperatures.unsqueeze(dim=1)
        greedy_mask = temperatures <= 1e-10
        safe_temperatures = temperatures.clamp_min(1e-5)
        probs = torch.softmax(logits / safe_temperatures, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_ps.unsqueeze(dim=1)
        sorted_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
        probs.zero_().scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
        probs.div_(probs.sum(dim=-1, keepdim=True))

        sample_tokens = probs.div(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(greedy_mask.squeeze(dim=1), greedy_tokens, sample_tokens)
