from nanovllm.sampling_params import SamplingParams
from nanovllm.avg import (
    AVGDirector,
    CORE_STATE_KEYS,
    AVGGenerationParams,
    AVGGenerationPlan,
    AVGGenerationResult,
    AVGState,
    AstroInfluence,
    PLOT_POINT_KINDS,
    PlayerEvent,
    PlotPoint,
)

__all__ = [
    "LLM",
    "SamplingParams",
    "AVGDirector",
    "CORE_STATE_KEYS",
    "AVGGenerationParams",
    "AVGGenerationPlan",
    "AVGGenerationResult",
    "AVGState",
    "AstroInfluence",
    "PLOT_POINT_KINDS",
    "PlayerEvent",
    "PlotPoint",
]


def __getattr__(name):
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
