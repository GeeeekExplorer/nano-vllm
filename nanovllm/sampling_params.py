from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    stop_token_ids: list[int] | None = None

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        if self.stop_token_ids is not None:
            assert isinstance(self.stop_token_ids, list) and all(
                type(token_id) is int and token_id >= 0 for token_id in self.stop_token_ids
            ), "stop_token_ids must be a list of non-negative integers"
