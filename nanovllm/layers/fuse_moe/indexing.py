from functools import partial

import torch


@partial(torch.compile, fullgraph=True, mode="max-autotune-no-cudagraphs")
@torch.no_grad()
def get_expert_counts_and_idx(s: torch.Tensor, E: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    E_arange = torch.arange(E, device=s.device, dtype=s.dtype)
    compare = E_arange[:, None] == s[None, :]
    counts = compare.sum(dim=1, dtype=torch.int32)

    s_arange = torch.arange(s.numel(), device=s.device, dtype=s.dtype)
    ranks_in_bin = compare.cumsum(dim=1, dtype=torch.int32)
    ranks_in_bin = ranks_in_bin[s, s_arange]
    offsets = counts.cumsum(dim=0, dtype=torch.int32) - counts
    idx = ranks_in_bin + offsets[s] - 1

    inv_idx = torch.empty_like(idx)
    inv_idx[idx] = s_arange.to(inv_idx.dtype)
    return counts, inv_idx, idx
