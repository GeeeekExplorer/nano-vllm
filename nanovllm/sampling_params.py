from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False # 忽略结束符的话就会一直输出到maxtoken数，不忽略就遇见结束符就结束

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
