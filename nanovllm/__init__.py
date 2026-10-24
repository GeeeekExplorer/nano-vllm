from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    # Import the engine lazily so that lightweight submodules (sampling
    # parameters, sequence, block manager, scheduler) can be imported without
    # pulling in torch, transformers and flash-attn.
    if name == "LLM":
        from nanovllm.llm import LLM

        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
