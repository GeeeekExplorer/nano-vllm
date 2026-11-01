from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from nanovllm.engine import speculative
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative import SpeculativeDecoder
from nanovllm.layers.spec_sampler import probs_from_logits
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context

VOCAB = 16
BLOCK_SIZE = 4
K = 3
CPU = torch.device("cpu")


def target_rule(token: int) -> int:
    return (token + 1) % VOCAB


def draft_rule(token: int) -> int:
    return 0 if token == 5 else (token + 1) % VOCAB


def peaked(tokens: torch.Tensor) -> torch.Tensor:
    """Logits that put almost all mass on the given tokens."""
    return torch.full((*tokens.shape, VOCAB), -10.0).scatter_(-1, tokens.unsqueeze(-1), 10.0)


class FakeRunner:
    """Stands in for the target ModelRunner on CPU: records every forward's context, predicts by a token rule."""

    def __init__(self, rule):
        self.rule = rule
        self.block_size = BLOCK_SIZE
        self.device = CPU
        self.enforce_eager = True
        self.generator = torch.Generator().manual_seed(0)
        self.pad_block_id = 99
        self.graph_bs = [1, 2, 4, 8]
        self.graph_pool = None
        self.config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=VOCAB, hidden_size=VOCAB, dtype=torch.float32))
        self.model = SimpleNamespace(compute_logits=lambda hidden, all_logits=False: hidden)
        self.calls = []

    def run_model(self, input_ids, positions, is_prefill, *, need_logits=True, aux_out=None):
        context = get_context()
        self.calls.append({
            "is_prefill": is_prefill,
            "input_ids": input_ids.tolist(),
            "positions": positions.tolist(),
            "slot_mapping": context.slot_mapping.tolist(),
            "cu_seqlens_q": None if context.cu_seqlens_q is None else context.cu_seqlens_q.tolist(),
            "cu_seqlens_k": None if context.cu_seqlens_k is None else context.cu_seqlens_k.tolist(),
            "block_tables": None if context.block_tables is None else context.block_tables.tolist(),
        })
        return peaked(torch.tensor([self.rule(token) for token in input_ids.tolist()]))


class FakeDrafter:
    """Stands in for DFlashDrafter: proposes k drafts by draft_rule from the last token, peaked like the target."""
    aux_width, aux_dtype = 2, torch.float32

    def __init__(self, config, decoder):
        self.decoder = decoder

    def propose(self, seqs, sampling):
        rows = []
        for seq in seqs:
            token, row = seq.last_token, []
            for _ in range(self.decoder.num_drafts):
                token = draft_rule(token)
                row.append(token)
            rows.append(row)
        drafts = torch.tensor(rows)
        if sampling.temperatures is None:
            return drafts, None
        return drafts, probs_from_logits(peaked(drafts), sampling)

    def after_verify(self, seqs, aux):
        pass


def make_decoder():
    target = FakeRunner(target_rule)
    config = SimpleNamespace(num_speculative_tokens=K, kvcache_block_size=BLOCK_SIZE, max_model_len=64,
                             hf_config=SimpleNamespace(vocab_size=VOCAB))
    with mock.patch.object(speculative, "DFlashDrafter", FakeDrafter):    # the real one loads a checkpoint
        decoder = SpeculativeDecoder(target, config)
    return decoder, target, decoder.drafter


def make_sequence(prompt, first_token, target_blocks, **params):
    """A sequence right after its prefill step: the KV covers the prompt, first token appended."""
    seq = Sequence(prompt, SamplingParams(**params))
    seq.append_token(first_token)
    seq.target_kv.block_table = list(target_blocks)
    seq.target_kv.num_cached_tokens = len(seq) - 1
    return seq


@pytest.fixture(autouse=True)
def small_blocks():
    previous = Sequence.block_size
    Sequence.block_size = BLOCK_SIZE
    yield
    Sequence.block_size = previous


def test_full_acceptance_greedy():
    decoder, target, drafter = make_decoder()
    seq = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], temperature=0)
    appended = decoder.run_round([seq])
    assert appended == [[12, 13, 14, 15]]
    assert decoder.stats["accepted_3"] == 1 and decoder.stats["rounds"] == 1


def test_verification_forward_layout():
    decoder, target, drafter = make_decoder()
    seq = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], temperature=0)
    decoder.run_round([seq])
    (verify,) = target.calls
    assert verify["is_prefill"] and verify["input_ids"] == [11, 12, 13, 14]
    assert verify["positions"] == [3, 4, 5, 6]
    assert verify["slot_mapping"] == [15, 16, 17, 18]
    assert verify["cu_seqlens_q"] == [0, 4] and verify["cu_seqlens_k"] == [0, 7]
    assert verify["block_tables"] == [[3, 4]]


def test_partial_acceptance_greedy():
    decoder, target, drafter = make_decoder()
    seq = make_sequence([1, 2, 3], 4, target_blocks=[3, 4], temperature=0)
    appended = decoder.run_round([seq])
    assert appended == [[5, 6]]                       # drafts [5, 0, 1]; target says 6 after 5
    assert decoder.stats["accepted_1"] == 1


def test_rows_are_independent_in_a_batch():
    decoder, target, drafter = make_decoder()
    full = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], temperature=0)
    partial = make_sequence([1, 2, 3], 4, target_blocks=[8, 9], temperature=0)
    appended = decoder.run_round([full, partial])
    assert appended == [[12, 13, 14, 15], [5, 6]]
    (verify,) = target.calls
    assert verify["cu_seqlens_q"] == [0, 4, 8] and verify["cu_seqlens_k"] == [0, 7, 14]


def test_random_mode_with_peaked_distributions_matches_greedy():
    decoder, target, drafter = make_decoder()
    seq = make_sequence([1, 2, 3], 4, target_blocks=[3, 4], temperature=1.0)
    appended = decoder.run_round([seq])
    assert appended == [[5, 6]]


def test_round_entry_invariants_are_asserted():
    decoder, target, drafter = make_decoder()
    seq = make_sequence([1, 2, 3], 4, target_blocks=[3, 4], temperature=0)
    seq.target_kv.num_cached_tokens = len(seq)
    with pytest.raises(AssertionError):
        decoder.run_round([seq])


def test_phase_timings_are_recorded_when_enabled():
    decoder, target, drafter = make_decoder()
    decoder.enable_phase_timings(event_factory=FakeEvent, synchronize=lambda: None)
    seq = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], temperature=0)
    decoder.run_round([seq])
    phases = decoder.phase_timings()
    assert [phase["phase"] for phase in phases] == ["propose", "verify", "ctx_store", "accept", "commit_copy"]
    assert all(phase["duration_ms"] >= 0 for phase in phases)


class FakeEvent:
    clock = 0.0

    def record(self):
        self.at = FakeEvent.clock
        FakeEvent.clock += 1.0

    def elapsed_time(self, other):
        return other.at - self.at
