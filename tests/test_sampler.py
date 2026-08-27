from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler, apply_top_k_top_p
from nanovllm.sampling_params import SamplingParams


def test_sampling_params_validation_and_sequence_propagation():
    positional_params = SamplingParams(0.7, 12, True)
    assert positional_params.max_tokens == 12
    assert positional_params.ignore_eos
    assert positional_params.top_p == 1.0
    assert positional_params.top_k == -1

    params = SamplingParams(temperature=0.7, top_p=0.9, top_k=20)
    seq = Sequence([1, 2, 3], params)
    assert seq.temperature == 0.7
    assert seq.top_p == 0.9
    assert seq.top_k == 20

    SamplingParams(top_k=0)
    SamplingParams(top_k=-1)
    with pytest.raises(AssertionError):
        SamplingParams(top_p=0.0)
    with pytest.raises(AssertionError):
        SamplingParams(top_p=1.1)
    with pytest.raises(AssertionError):
        SamplingParams(top_k=-2)
    with pytest.raises(AssertionError):
        SamplingParams(top_k=1.5)


def test_top_k_filter_supports_mixed_rows():
    logits = torch.tensor([
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0, 0.0],
    ])
    filtered = apply_top_k_top_p(logits, torch.tensor([2, 5]), None, max_top_k=2)
    assert torch.isneginf(filtered[0, :3]).all()
    assert torch.equal(filtered[0, 3:], torch.tensor([3.0, 4.0]))
    assert torch.equal(filtered[1], torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]))


def test_top_p_filter_supports_mixed_rows():
    probs = torch.tensor([
        [0.05, 0.10, 0.15, 0.30, 0.40],
        [0.05, 0.10, 0.15, 0.30, 0.40],
    ])
    filtered = apply_top_k_top_p(probs.log(), None, torch.tensor([0.6, 1.0]), max_top_k=0)
    assert torch.isneginf(filtered[0, :3]).all()
    assert torch.isfinite(filtered[0, 3:]).all()
    assert torch.isfinite(filtered[1]).all()


def test_joint_top_k_top_p_filtering():
    probs = torch.tensor([[0.05, 0.10, 0.15, 0.30, 0.40]])
    filtered = apply_top_k_top_p(
        probs.log(),
        torch.tensor([3]),
        torch.tensor([0.7]),
        max_top_k=3,
    )
    assert torch.isneginf(filtered[0, :3]).all()
    assert torch.isfinite(filtered[0, 3:]).all()


def test_sampler_never_selects_filtered_tokens():
    logits = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 4.0],
    ]).repeat(64, 1)
    temperatures = torch.ones(128)
    top_ks = torch.tensor([2, 4]).repeat(64)
    top_ps = torch.tensor([1.0, 0.5]).repeat(64)
    tokens = Sampler()(logits, temperatures, top_ks, top_ps, max_top_k=2)
    assert set(tokens[0::2].tolist()) <= {2, 3}
    assert set(tokens[1::2].tolist()) == {3}


def test_disabled_filters_preserve_logits():
    logits = torch.randn(3, 17)
    assert apply_top_k_top_p(logits, None, None, max_top_k=0) is logits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prepare_sample_normalizes_mixed_and_oversized_top_k():
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=5))
    seqs = [
        Sequence([1], SamplingParams(temperature=0.2, top_k=2)),
        Sequence([1], SamplingParams(temperature=0.4, top_k=100, top_p=0.7)),
        Sequence([1], SamplingParams(temperature=0.6)),
    ]
    temperatures, top_ks, top_ps, max_top_k = runner.prepare_sample(seqs)
    torch.testing.assert_close(temperatures, torch.tensor([0.2, 0.4, 0.6], device="cuda"))
    assert top_ks.tolist() == [2, 5, 5]
    torch.testing.assert_close(top_ps, torch.tensor([1.0, 0.7, 1.0], device="cuda"))
    assert max_top_k == 2

    _, top_ks, top_ps, max_top_k = runner.prepare_sample([
        Sequence([1], SamplingParams(top_k=5)),
        Sequence([1], SamplingParams(top_k=100)),
    ])
    assert top_ks is None
    assert top_ps is None
    assert max_top_k == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_top_p_sampler_handles_large_vocab_cuda():
    logits = torch.randn(16, 151936, device="cuda", dtype=torch.bfloat16)
    temperatures = torch.full((16,), 0.8, device="cuda")
    top_ps = torch.full((16,), 0.9, device="cuda")
    tokens = Sampler()(logits, temperatures, top_ps=top_ps)
    assert tokens.shape == (16,)
