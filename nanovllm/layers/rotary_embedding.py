import math
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


class Llama3RotaryEmbedding(RotaryEmbedding):
    """Llama3 RoPE with frequency smoothing between NTK and linear interpolation."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        original_max_position_embeddings: int,
    ) -> None:
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        inv_freq = self._apply_scaling(
            inv_freq, factor, low_freq_factor, high_freq_factor, original_max_position_embeddings
        )
        # bypass RotaryEmbedding.__init__, build cache directly
        nn.Module.__init__(self)
        self.head_size = head_size
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @staticmethod
    def _apply_scaling(
        inv_freq: torch.Tensor,
        factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        original_max_position_embeddings: int,
    ) -> torch.Tensor:
        low_freq_wavelen = original_max_position_embeddings / low_freq_factor
        high_freq_wavelen = original_max_position_embeddings / high_freq_factor
        new_inv_freq = []
        for freq in inv_freq.tolist():
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_inv_freq.append(freq)
            elif wavelen > low_freq_wavelen:
                new_inv_freq.append(freq / factor)
            else:
                smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_inv_freq.append((1 - smooth) * freq / factor + smooth * freq)
        return torch.tensor(new_inv_freq, dtype=inv_freq.dtype)


def _rope_scaling_to_key(rope_scaling: dict | None):
    if rope_scaling is None:
        return None
    return tuple(sorted(rope_scaling.items()))


@lru_cache(maxsize=8)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: tuple | None = None,
):
    if rope_scaling is None:
        return RotaryEmbedding(head_size, rotary_dim, max_position, base)
    scaling_dict = dict(rope_scaling)
    rope_type = scaling_dict.get("rope_type", scaling_dict.get("type"))
    if rope_type == "llama3":
        return Llama3RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
            base=base,
            factor=scaling_dict["factor"],
            low_freq_factor=scaling_dict["low_freq_factor"],
            high_freq_factor=scaling_dict["high_freq_factor"],
            original_max_position_embeddings=scaling_dict["original_max_position_embeddings"],
        )
    raise ValueError(f"Unsupported rope_type: {rope_type}")
