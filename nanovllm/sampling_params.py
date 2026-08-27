from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_p: float = 1.0
    top_k: int = -1

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
        assert isinstance(self.top_k, int) and self.top_k >= -1, "top_k must be -1, 0, or a positive integer"
