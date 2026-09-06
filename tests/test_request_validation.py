from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def load_request_types():
    root = Path(__file__).parents[1]
    package = types.ModuleType("nanovllm")
    package.__path__ = [str(root / "nanovllm")]
    engine_package = types.ModuleType("nanovllm.engine")
    engine_package.__path__ = [str(root / "nanovllm" / "engine")]

    sampling_spec = spec_from_file_location(
        "nanovllm.sampling_params", root / "nanovllm" / "sampling_params.py"
    )
    if sampling_spec is None or sampling_spec.loader is None:
        raise RuntimeError("could not load sampling parameters")
    sampling_module = module_from_spec(sampling_spec)
    sampling_spec.loader.exec_module(sampling_module)

    config = types.ModuleType("nanovllm.config")
    config.Config = object
    sequence = types.ModuleType("nanovllm.engine.sequence")
    sequence.Sequence = object
    scheduler = types.ModuleType("nanovllm.engine.scheduler")
    scheduler.Scheduler = object
    model_runner = types.ModuleType("nanovllm.engine.model_runner")
    model_runner.ModelRunner = object

    engine_spec = spec_from_file_location(
        "nanovllm.engine.llm_engine",
        root / "nanovllm" / "engine" / "llm_engine.py",
    )
    if engine_spec is None or engine_spec.loader is None:
        raise RuntimeError("could not load LLM engine")
    engine_module = module_from_spec(engine_spec)
    with patch.dict(
        sys.modules,
        {
            "nanovllm": package,
            "nanovllm.config": config,
            "nanovllm.sampling_params": sampling_module,
            "nanovllm.engine": engine_package,
            "nanovllm.engine.sequence": sequence,
            "nanovllm.engine.scheduler": scheduler,
            "nanovllm.engine.model_runner": model_runner,
        },
    ):
        engine_spec.loader.exec_module(engine_module)
    return sampling_module.SamplingParams, engine_module.LLMEngine


SamplingParams, LLMEngine = load_request_types()


@pytest.mark.parametrize("max_tokens", [0, -1, 1.5, True])
def test_sampling_params_reject_invalid_max_tokens(max_tokens):
    with pytest.raises(ValueError, match="positive integer"):
        SamplingParams(max_tokens=max_tokens)


def test_request_length_rejects_total_over_limit():
    with pytest.raises(ValueError, match="9 tokens.*max_model_len=8"):
        SamplingParams(max_tokens=4).validate_request_length(5, 8)


def test_request_length_rejects_empty_prompt():
    with pytest.raises(ValueError, match="at least one token"):
        SamplingParams(max_tokens=1).validate_request_length(0, 8)


def test_request_length_accepts_total_at_limit():
    SamplingParams(max_tokens=4).validate_request_length(4, 8)


def test_engine_validates_string_prompt_after_tokenization():
    engine = object.__new__(LLMEngine)
    engine.max_model_len = 3
    engine.tokenizer = SimpleNamespace(encode=lambda _: [1, 2, 3])
    engine.scheduler = SimpleNamespace(add=lambda _: pytest.fail("scheduled"))

    with pytest.raises(ValueError, match="4 tokens.*max_model_len=3"):
        engine.add_request("prompt", SamplingParams(max_tokens=1))


@pytest.mark.parametrize("prompt", [[], ""])
def test_engine_rejects_empty_tokenized_prompt(prompt):
    engine = object.__new__(LLMEngine)
    engine.max_model_len = 8
    engine.tokenizer = SimpleNamespace(encode=lambda _: [])
    engine.scheduler = SimpleNamespace(add=lambda _: pytest.fail("scheduled"))

    with pytest.raises(ValueError, match="at least one token"):
        engine.add_request(prompt, SamplingParams(max_tokens=1))
