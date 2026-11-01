"""DSpark's propose: the drafts are drawn left to right, each row's logits biased by the Markov head on the token
before it (the anchor for row 0), and the q handed to acceptance is each row's distribution after its bias. The
sampling graphs capture the same rows; their body is checked here run eagerly, the capture itself on the GPU."""
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.batch_metadata import draft_block_metadata
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative import DFlashDrafter, SpeculativeDecoder, sampling_graph_key
from nanovllm.layers.spec_sampler import probs_from_logits, sample, sampling_tensors
from nanovllm.sampling_params import SamplingParams

CPU = torch.device("cpu")
# The Markov bias by previous token, as in the patch note's example: token 0 adds 2 to token 1's logit.
BIAS = torch.tensor([[0.0, 2.0, 0.0],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


def make_drafter(k: int, bias: torch.Tensor = BIAS) -> DFlashDrafter:
    """A drafter whose Markov head looks the bias up by previous token and whose hidden states are its logits."""
    config = SimpleNamespace(num_speculative_tokens=k, kvcache_block_size=256, max_model_len=64,
                             hf_config=SimpleNamespace(vocab_size=bias.shape[1]))
    target = SimpleNamespace(device=CPU, generator=torch.Generator().manual_seed(0))
    decoder = SpeculativeDecoder(target, object(), config)    # builds a ModelDrafter, unused here
    drafter = object.__new__(DFlashDrafter)
    drafter.decoder = decoder
    drafter.model = SimpleNamespace(markov_head=lambda prev: bias[prev], compute_logits=lambda hidden: hidden)
    drafter.sampling_noise = {}
    return drafter


def test_greedy_drafts_follow_the_markov_chain():
    # Base logits [2, 1, 0] at every row pick token 0; after token 0 the bias makes token 1 win, after token 1 it
    # does not. Anchor 0 gives 1, 0, 1; anchor 1 gives 0, 1, 0.
    seqs = [Sequence([2, 0], SamplingParams(temperature=0)), Sequence([2, 1], SamplingParams(temperature=0))]
    logits = torch.tensor([2.0, 1.0, 0.0]).expand(2, 3, 3)
    drafts, probs = make_drafter(k=3)._sample_markov(seqs, logits, sampling_tensors(seqs, CPU))
    assert drafts.tolist() == [[1, 0, 1], [0, 1, 0]] and probs is None


def test_q_is_each_row_distribution_after_its_bias():
    seqs = [Sequence([2, 0], SamplingParams(temperature=1.0)), Sequence([0, 1], SamplingParams(temperature=0.7, top_k=2))]
    sampling = sampling_tensors(seqs, CPU)
    logits = torch.randn(2, 4, 3, generator=torch.Generator().manual_seed(1))
    drafts, probs = make_drafter(k=4)._sample_markov(seqs, logits, sampling)
    prev = torch.tensor([0, 1])                               # the anchors
    for row in range(4):
        assert torch.equal(probs[:, row], probs_from_logits(logits[:, row] + BIAS[prev], sampling))
        prev = drafts[:, row]


def test_without_a_bias_the_rows_draw_what_dflash_draws():
    # One row does not truncate, so the rows are drawn over the whole vocabulary, with the noise of all k rows drawn
    # at once as DFlash draws it: with a zero Markov bias the same generator state gives DFlash's q and drafts and
    # leaves the generator where DFlash leaves it.
    seqs = [Sequence([1, 2], SamplingParams(temperature=1.0)),
            Sequence([3, 0], SamplingParams(temperature=0.7, top_k=3, top_p=0.9))]
    sampling = sampling_tensors(seqs, CPU)
    logits = torch.randn(2, 4, 6, generator=torch.Generator().manual_seed(2))
    drafter = make_drafter(k=4, bias=torch.zeros(6, 6))
    drafts, probs = drafter._sample_markov(seqs, logits, sampling)
    generator = torch.Generator().manual_seed(0)
    dflash_probs = probs_from_logits(logits, sampling)
    assert torch.equal(probs, dflash_probs)
    assert torch.equal(drafts, sample(dflash_probs, generator))
    assert torch.equal(drafter.decoder.target.generator.get_state(), generator.get_state())


def test_candidate_rows_give_the_whole_vocabulary_distribution():
    # Every row truncates, so each row is drawn among its top max_top_k + TIE_MARGIN logits (20 + 64). In the top-3
    # row, thirty tokens tie with the cut value 3, out to rank 31: the truncation keeps them, and so do the candidates.
    vocab_size, k = 200, 4
    seqs = [Sequence([1, 2], SamplingParams(temperature=0.7, top_k=3)),
            Sequence([3, 0], SamplingParams(temperature=1.0, top_k=20, top_p=0.8))]
    sampling = sampling_tensors(seqs, CPU)
    logits = torch.randn(2, k, vocab_size, generator=torch.Generator().manual_seed(5))
    logits[0] = logits[0].clamp(max=2.5)
    logits[0, :, :32] = torch.tensor([5.0, 4.0] + [3.0] * 30)
    drafts, probs = make_drafter(k, bias=torch.zeros(vocab_size, vocab_size))._sample_markov(seqs, logits, sampling)
    whole = probs_from_logits(logits, sampling)                       # no bias: each row's rule over the vocabulary
    assert torch.equal(probs > 0, whole > 0)
    assert torch.allclose(probs, whole, rtol=0, atol=1e-6)
    assert bool((probs.gather(-1, drafts.unsqueeze(-1)) > 0).all())


@pytest.mark.parametrize("params", [
    [SamplingParams(temperature=0), SamplingParams(temperature=0)],
    [SamplingParams(temperature=1.0), SamplingParams(temperature=0.7, top_k=3, top_p=0.9)],
    [SamplingParams(temperature=1.0, top_k=4), SamplingParams(temperature=0.7, top_k=3, top_p=0.9)],
], ids=["greedy", "random", "candidates"])
def test_sampling_graph_body_draws_what_the_eager_rows_draw(params):
    # The body a sampling graph captures, run eagerly: two sequences padded to four, the anchors read from the block
    # inputs as draft_block_metadata lays them out, the logits from the block outputs.
    k, vocab_size, padded = 3, 6, 4
    seqs = [Sequence([1, 2], params[0]), Sequence([3, 4], params[1])]
    for seq in seqs:
        seq.target_kv.block_table = [0]
    sampling = sampling_tensors(seqs, CPU)
    drafter = make_drafter(k, bias=torch.randn(vocab_size, vocab_size, generator=torch.Generator().manual_seed(3)))
    logits = torch.randn(padded, k, vocab_size, generator=torch.Generator().manual_seed(4))
    eager_drafts, eager_probs = drafter._sample_markov(seqs, logits[:2], sampling)
    eager_state = drafter.decoder.target.generator.get_state()

    metadata = draft_block_metadata(seqs, 256, k, vocab_size - 1, padded, 99, 1)
    drafter.graph_vars = {"outputs": logits.view(padded * k, vocab_size), "input_ids": torch.tensor(metadata.input_ids),
                          **drafter._sampling_buffers(padded)}
    key = sampling_graph_key(sampling)
    if key is not None:
        drafter._allocate_sampling_noise(key, padded)
    drafter.decoder.target.generator.manual_seed(0)                   # where the eager rows started
    drafter._load_sampling_inputs(2, sampling, key)
    drafter._sampling_forward(padded, key)
    buffers = drafter.graph_vars
    assert torch.equal(buffers["drafts"][:2], eager_drafts)
    assert eager_probs is None or torch.equal(buffers["probs"][:2], eager_probs)
    assert torch.equal(drafter.decoder.target.generator.get_state(), eager_state)
