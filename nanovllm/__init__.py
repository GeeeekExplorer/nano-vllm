from nanovllm.sampling_params import SamplingParams
from nanovllm.avg import (
    AVGDirector,
    AVGGenerationParams,
    AVGGenerationPlan,
    AVGState,
    AstroInfluence,
    PlayerEvent,
    PlotPoint,
)

__all__ = [
    "LLM",
    "SamplingParams",
    "AVGDirector",
    "AVGGenerationParams",
    "AVGGenerationPlan",
    "AVGState",
    "AstroInfluence",
    "PlayerEvent",
    "PlotPoint",
]


def __getattr__(name):
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
