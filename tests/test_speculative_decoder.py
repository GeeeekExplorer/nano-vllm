from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.batch_metadata import prefill_metadata
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative import SpeculativeDecoder
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, set_context

VOCAB = 16
BLOCK_SIZE = 4
K = 3
CPU = torch.device("cpu")


def target_rule(token: int) -> int:
    return (token + 1) % VOCAB


def draft_rule(token: int) -> int:
    return 0 if token == 5 else (token + 1) % VOCAB


class FakeRunner:
    """Stands in for ModelRunner on CPU: records every forward's context, predicts by a token rule."""

    def __init__(self, role: str, rule):
        self.kv_role = role
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

    def prepare_prefill(self, seqs):
        metadata = prefill_metadata(seqs, self.kv_role, self.block_size)
        block_tables = None
        if metadata.has_cached_prefix:
            tables = [seq.kv(self.kv_role).block_table for seq in seqs]
            width = max(len(table) for table in tables)
            block_tables = torch.tensor([table + [-1] * (width - len(table)) for table in tables], dtype=torch.int32)
        set_context(True, torch.tensor(metadata.cu_seqlens_q), torch.tensor(metadata.cu_seqlens_k),
                    metadata.max_seqlen_q, metadata.max_seqlen_k, torch.tensor(metadata.slot_mapping), None, block_tables)
        return torch.tensor(metadata.input_ids), torch.tensor(metadata.positions)

    def run_model(self, input_ids, positions, is_prefill, *, need_logits=True, aux_out=None):
        context = get_context()
        self.calls.append({
            "is_prefill": is_prefill,
            "input_ids": input_ids.tolist(),
            "positions": positions.tolist(),
            "slot_mapping": context.slot_mapping.tolist(),
            "context_lens": None if context.context_lens is None else context.context_lens.tolist(),
            "cu_seqlens_q": None if context.cu_seqlens_q is None else context.cu_seqlens_q.tolist(),
            "cu_seqlens_k": None if context.cu_seqlens_k is None else context.cu_seqlens_k.tolist(),
            "block_tables": None if context.block_tables is None else context.block_tables.tolist(),
        })
        logits = torch.full((input_ids.numel(), VOCAB), -10.0)
        for row, token in enumerate(input_ids.tolist()):
            logits[row, self.rule(token)] = 10.0
        return logits


def make_decoder():
    target = FakeRunner("target", target_rule)
    draft = FakeRunner("draft", draft_rule)
    config = SimpleNamespace(num_speculative_tokens=K, kvcache_block_size=BLOCK_SIZE, max_model_len=64,
                             hf_config=SimpleNamespace(vocab_size=VOCAB))
    return SpeculativeDecoder(target, draft, config), target, draft


def make_sequence(prompt, first_token, target_blocks, draft_blocks, **params):
    """A sequence right after its prefill step: both KV states cover the prompt, first token appended."""
    seq = Sequence(prompt, SamplingParams(**params))
    seq.append_token(first_token)
    seq.target_kv.block_table = list(target_blocks)
    seq.draft_kv.block_table = list(draft_blocks)
    seq.target_kv.num_cached_tokens = len(seq) - 1
    seq.draft_kv.num_cached_tokens = len(seq) - 1
    return seq


@pytest.fixture(autouse=True)
def small_blocks():
    previous = Sequence.block_size
    Sequence.block_size = BLOCK_SIZE
    yield
    Sequence.block_size = previous


def test_full_acceptance_greedy():
    decoder, target, draft = make_decoder()
    seq = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], draft_blocks=[6, 7], temperature=0)
    appended = decoder.run_round([seq])
    assert appended == [[12, 13, 14, 15]]
    assert decoder.stats["accepted_3"] == 1 and decoder.stats["rounds"] == 1


def test_draft_steps_use_consecutive_positions_and_slots():
    decoder, target, draft = make_decoder()
    seq = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], draft_blocks=[6, 7], temperature=0)
    decoder.run_round([seq])
    steps = [call for call in draft.calls if not call["is_prefill"]]
    assert [call["input_ids"] for call in steps] == [[11], [12], [13]]
    assert [call["positions"] for call in steps] == [[3], [4], [5]]
    assert [call["context_lens"] for call in steps] == [[4], [5], [6]]
    assert [call["slot_mapping"] for call in steps] == [[27], [28], [29]]      # block 6*4 + 3, 7*4 + 0, 7*4 + 1
    assert all(call["block_tables"] == [[6, 7]] for call in steps)


def test_verification_forward_layout():
    decoder, target, draft = make_decoder()
    seq = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], draft_blocks=[6, 7], temperature=0)
    decoder.run_round([seq])
    (verify,) = target.calls
    assert verify["is_prefill"] and verify["input_ids"] == [11, 12, 13, 14]
    assert verify["positions"] == [3, 4, 5, 6]
    assert verify["slot_mapping"] == [15, 16, 17, 18]
    assert verify["cu_seqlens_q"] == [0, 4] and verify["cu_seqlens_k"] == [0, 7]
    assert verify["block_tables"] == [[3, 4]]


def test_partial_acceptance_greedy():
    decoder, target, draft = make_decoder()
    seq = make_sequence([1, 2, 3], 4, target_blocks=[3, 4], draft_blocks=[6, 7], temperature=0)
    appended = decoder.run_round([seq])
    assert appended == [[5, 6]]                       # drafts [5, 0, 1]; target says 6 after 5
    assert decoder.stats["accepted_1"] == 1


def test_rows_are_independent_in_a_batch():
    decoder, target, draft = make_decoder()
    full = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], draft_blocks=[6, 7], temperature=0)
    partial = make_sequence([1, 2, 3], 4, target_blocks=[8, 9], draft_blocks=[10, 11], temperature=0)
    appended = decoder.run_round([full, partial])
    assert appended == [[12, 13, 14, 15], [5, 6]]
    (verify,) = target.calls
    assert verify["cu_seqlens_q"] == [0, 4, 8] and verify["cu_seqlens_k"] == [0, 7, 14]


def test_catch_up_recomputes_the_missing_draft_position():
    decoder, target, draft = make_decoder()
    seq = make_sequence([8, 9, 10], 11, target_blocks=[3, 4, 5], draft_blocks=[6, 7, 12], temperature=0)
    for token in [12, 13, 14, 15]:                    # a fully accepted round was committed by the scheduler
        seq.append_token(token)
    seq.target_kv.num_cached_tokens = len(seq) - 1    # 7
    seq.draft_kv.num_cached_tokens = len(seq) - 2     # 6: the draft never saw token 14 at position 6
    decoder.run_round([seq])
    catch_up = draft.calls[0]                         # one decode step at position 6 with token 14, no logits
    assert not catch_up["is_prefill"] and catch_up["input_ids"] == [14] and catch_up["positions"] == [6]
    assert catch_up["context_lens"] == [7] and catch_up["slot_mapping"] == [30]     # block 7*4 + 2
    assert catch_up["block_tables"] == [[6, 7, 12]]
    assert seq.draft_kv.num_cached_tokens == 7 and seq.draft_kv.num_scheduled_tokens == 0
    assert draft.calls[1]["input_ids"] == [15]        # the first draft step follows immediately


def test_random_mode_with_peaked_distributions_matches_greedy():
    decoder, target, draft = make_decoder()
    seq = make_sequence([1, 2, 3], 4, target_blocks=[3, 4], draft_blocks=[6, 7], temperature=1.0)
    appended = decoder.run_round([seq])
    assert appended == [[5, 6]]


def test_round_entry_invariants_are_asserted():
    decoder, target, draft = make_decoder()
    seq = make_sequence([1, 2, 3], 4, target_blocks=[3, 4], draft_blocks=[6, 7], temperature=0)
    seq.target_kv.num_cached_tokens = len(seq)
    with pytest.raises(AssertionError):
        decoder.run_round([seq])


def test_prefill_draft_forwards_the_scheduled_chunk_without_logits():
    decoder, target, draft = make_decoder()
    seq = Sequence([8, 9, 10])
    seq.draft_kv.block_table = [6]
    seq.draft_kv.num_scheduled_tokens = 3
    decoder.prefill_draft([seq])
    (call,) = draft.calls
    assert call["is_prefill"] and call["input_ids"] == [8, 9, 10] and call["slot_mapping"] == [24, 25, 26]


def test_phase_timings_are_recorded_when_enabled():
    decoder, target, draft = make_decoder()
    decoder.enable_phase_timings(event_factory=FakeEvent, synchronize=lambda: None)
    seq = make_sequence([8, 9, 10], 11, target_blocks=[3, 4], draft_blocks=[6, 7], temperature=0)
    decoder.run_round([seq])
    phases = decoder.phase_timings()
    assert [phase["phase"] for phase in phases[:5]] == ["catch_up", "propose", "verify", "accept", "commit_copy"]
    assert all(phase["duration_ms"] >= 0 for phase in phases)


class FakeEvent:
    clock = 0.0

    def record(self):
        self.at = FakeEvent.clock
        FakeEvent.clock += 1.0

    def elapsed_time(self, other):
        return other.at - self.at
