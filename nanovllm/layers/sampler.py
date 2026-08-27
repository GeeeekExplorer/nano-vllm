import torch
from torch import nn


def apply_top_k_top_p(
    logits: torch.Tensor,
    top_ks: torch.Tensor | None,
    top_ps: torch.Tensor | None,
    max_top_k: int,
) -> torch.Tensor:
    if top_ks is None and top_ps is None:
        return logits

    if top_ps is None:
        top_k_values = logits.topk(max_top_k, dim=-1).values
        top_k_indices = top_ks.clamp_max(max_top_k).sub(1).unsqueeze(1)
        thresholds = top_k_values.gather(1, top_k_indices)
        thresholds.masked_fill_(top_ks.unsqueeze(1) > max_top_k, -float("inf"))
        return logits.masked_fill_(logits < thresholds, -float("inf"))

    sorted_logits, sorted_indices = logits.sort(dim=-1)
    if top_ks is not None:
        thresholds = sorted_logits.gather(1, (sorted_logits.size(1) - top_ks).unsqueeze(1))
        sorted_logits.masked_fill_(sorted_logits < thresholds, -float("inf"))

    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    top_p_mask = cumulative_probs <= 1 - top_ps.unsqueeze(1)
    top_p_mask[:, -1] = False
    sorted_logits.masked_fill_(top_p_mask, -float("inf"))
    return logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)


class Sampler(nn.Module):

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor | None = None,
        top_ps: torch.Tensor | None = None,
        max_top_k: int = 0,
    ):
        # Keep filtering eager: Inductor can fail on large-vocabulary top-p scans.
        if top_ks is None and top_ps is None:
            return self.random_sample(logits, temperatures)
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        logits = apply_top_k_top_p(logits, top_ks, top_ps, max_top_k)
        return self.random_sample(logits, None)

    @torch.compile
    def random_sample(self, logits: torch.Tensor, temperatures: torch.Tensor | None):
        if temperatures is not None:
            logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
