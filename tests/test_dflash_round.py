"""The DFlash round's KV state machine, with the real SpeculativeDecoder and DFlashDrafter around fake models.

Tokens follow a rule: the token at position p is right(p). The fake target predicts right(p + 1) at position p
and exports (position, input token) as its aux features. The fake DFlash model keeps, per slot, where its context
K/V came from; each block forward checks that every context position below the anchor holds the features of the
committed token there, never a block's K/V or a rejected draft's features.
"""
import random
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.batch_metadata import prefill_metadata, slot_of
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculative import DFlashDrafter, SpeculativeDecoder
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, reset_context, set_context

VOCAB, MASK, BLOCK_SIZE, K = 16, 15, 4, 7
CPU = torch.device("cpu")


def right(position: int) -> int:
    return (3 * position + 1) % (VOCAB - 1)    # never the mask token


def peaked(tokens: list[int]) -> torch.Tensor:
    logits = torch.full((len(tokens), VOCAB), -10.0)
    logits[torch.arange(len(tokens)), torch.tensor(tokens)] = 10.0
    return logits


class FakeTarget:
    """Stands in for the target ModelRunner: records its KV by slot and exports (position, token) as aux."""

    def __init__(self):
        self.block_size = BLOCK_SIZE
        self.kv_role = "target"
        self.device = CPU
        self.generator = torch.Generator().manual_seed(0)
        self.pad_block_id = 99
        self.config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=VOCAB, dtype=torch.float32))
        self.model = SimpleNamespace(compute_logits=lambda hidden, all_logits=False: hidden,
                                     model=SimpleNamespace(aux_layer_ids=()))
        self.kv = {}

    def run_model(self, input_ids, positions, is_prefill, *, need_logits=True, aux_out=None):
        rows = list(zip(positions.tolist(), input_ids.tolist()))
        for (position, token), slot in zip(rows, get_context().slot_mapping.tolist()):
            if slot >= 0:
                self.kv[slot] = (position, token)
        if aux_out is not None:
            aux_out.copy_(torch.tensor(rows, dtype=aux_out.dtype))
        return peaked([right(position + 1) for position, _ in rows])

    def prefill(self, seqs, aux):
        """ModelRunner.run's prefill for the scheduled chunk, without sampling."""
        metadata = prefill_metadata(seqs, "target", self.block_size)
        set_context(True, slot_mapping=torch.tensor(metadata.slot_mapping))
        self.run_model(torch.tensor(metadata.input_ids), torch.tensor(metadata.positions), True, aux_out=aux)
        reset_context()


class FakeDFlash:
    """Stands in for DFlashDraftModel. is_right(position) says whether the draft row at position guesses right."""

    def __init__(self, decoder, is_right):
        self.mask_token_id, self.block_size, self.aux_layer_ids = MASK, K, (0,)
        self.markov_head = None    # DFlash: the k rows are sampled at once
        self.decoder, self.is_right = decoder, is_right
        self.cache = {}           # slot -> ("context", position, token) or ("block", position)
        self.bad_reads = []
        self.context_reads = 0

    def context_features(self, aux):
        return aux

    def store_context_kv(self, features, positions, slot_mapping):
        for (position, token), expected_position, slot in zip(features.tolist(), positions.tolist(), slot_mapping.tolist()):
            assert position == expected_position
            if slot >= 0:
                self.cache[slot] = ("context", int(position), int(token))

    def __call__(self, input_ids, positions):
        context = get_context()
        assert context.query_len_per_seq == K and context.causal is False
        for row, seq in enumerate(self.decoder.current_seqs):
            anchor = positions[row * K].item()
            assert anchor == len(seq) - 1 and context.context_lens[row].item() == anchor + K
            assert input_ids[row * K:(row + 1) * K].tolist() == [seq.last_token] + [MASK] * (K - 1)
            table = context.block_tables[row].tolist()
            for position in range(anchor):
                found = self.cache.get(slot_of(table, BLOCK_SIZE, position))
                self.context_reads += 1
                if found != ("context", position, seq.token_ids[position]):
                    self.bad_reads.append((row, position, found))
        for slot, position in zip(context.slot_mapping.tolist(), positions.tolist()):
            self.cache[slot] = ("block", position)
        return positions.clone()

    def compute_logits(self, hidden):
        guesses = [right(position + 1) if self.is_right(position) else (right(position + 1) + 1) % (VOCAB - 1)
                   for position in hidden.tolist()]
        return peaked(guesses)


class Decoder(SpeculativeDecoder):
    def run_round(self, seqs):
        self.current_seqs = seqs
        return super().run_round(seqs)


def make_decoder(is_right):
    config = SimpleNamespace(num_speculative_tokens=K, kvcache_block_size=BLOCK_SIZE, max_model_len=512,
                             hf_config=SimpleNamespace(vocab_size=VOCAB))
    target = FakeTarget()
    decoder = Decoder(target, object(), config)    # builds a ModelDrafter, replaced below
    drafter = object.__new__(DFlashDrafter)
    drafter.decoder, drafter.model = decoder, FakeDFlash(decoder, is_right)
    drafter.aux_width, drafter.aux_dtype, drafter.graphs = 2, torch.float32, {}
    decoder.drafter = drafter
    return decoder, target, drafter.model


def new_sequence(prompt_len: int, first_block: int) -> Sequence:
    seq = Sequence([right(position) for position in range(prompt_len)], SamplingParams(temperature=0.0, max_tokens=512))
    seq.target_kv.block_table = list(range(first_block, first_block + 60))
    return seq


def prefill(decoder, target, seqs, chunk=None):
    """The engine's prefill steps, chunk by chunk, then the first token appended (by the rule, as greedy would)."""
    while any(seq.num_cached_tokens < len(seq) for seq in seqs):
        for seq in seqs:
            seq.num_scheduled_tokens = min(chunk or len(seq), len(seq) - seq.num_cached_tokens)
        aux = decoder.prefill_aux(seqs)
        target.prefill(seqs, aux)
        decoder.prefill_draft(seqs, aux)
        for seq in seqs:
            seq.num_cached_tokens += seq.num_scheduled_tokens
    for seq in seqs:
        seq.append_token(right(len(seq)))


def commit(seqs, appended):
    """Scheduler.postprocess_speculative for a DFlash draft."""
    for seq, tokens in zip(seqs, appended):
        for token in tokens:
            seq.append_token(token)
        seq.target_kv.num_cached_tokens = len(seq) - 1


@pytest.fixture(autouse=True)
def small_blocks():
    previous = Sequence.block_size
    Sequence.block_size = BLOCK_SIZE
    yield
    Sequence.block_size = previous


def test_one_round_reads_and_writes():
    decoder, target, draft = make_decoder(lambda position: position + 1 in (6, 7))    # accept exactly 2
    seq = new_sequence(5, first_block=0)
    prefill(decoder, target, [seq])                       # t0 at position 5, L = 6
    appended = decoder.run_round([seq])
    assert appended == [[right(6), right(7), right(8)]]
    slots = {position: slot_of(seq.target_kv.block_table, BLOCK_SIZE, position) for position in range(14)}
    assert all(target.kv[slots[position]][0] == position for position in range(13))           # verify wrote 5..12
    assert slots[13] not in target.kv
    assert all(draft.cache[slots[position]][0] == "context" for position in range(13))       # context 0..12
    assert draft.context_reads == 5 and not draft.bad_reads                                  # the block read 0..4
    commit([seq], appended)
    assert len(seq) == 9 and seq.target_kv.num_cached_tokens == 8


def test_context_reads_come_from_committed_tokens_over_random_rounds():
    rng = random.Random(17011)
    decoder, target, draft = make_decoder(lambda position: rng.random() < 0.6)
    seqs = [new_sequence(3 + index, first_block=100 * index) for index in range(10)]
    prefill(decoder, target, seqs)
    for _ in range(20):
        commit(seqs, decoder.run_round(seqs))
    assert draft.context_reads > 1000 and not draft.bad_reads
    assert all(seq.token_ids == [right(position) for position in range(len(seq))] for seq in seqs)


def test_chunked_prefill_writes_the_whole_prompt():
    decoder, target, draft = make_decoder(lambda position: True)
    seq = new_sequence(7, first_block=0)
    prefill(decoder, target, [seq], chunk=3)
    assert [draft.cache[slot_of(seq.target_kv.block_table, BLOCK_SIZE, p)] for p in range(7)] == \
        [("context", p, right(p)) for p in range(7)]
    decoder.run_round([seq])
    assert draft.context_reads == 7 and not draft.bad_reads


def test_recompute_after_preemption_rewrites_the_context():
    rng = random.Random(110017)
    decoder, target, draft = make_decoder(lambda position: rng.random() < 0.6)
    seq = new_sequence(6, first_block=0)
    prefill(decoder, target, [seq])
    for _ in range(3):
        commit([seq], decoder.run_round([seq]))
    # preempted: its blocks are freed, then the whole sequence is prefilled again into new blocks
    seq.target_kv.block_table = list(range(500, 560))
    seq.target_kv.num_cached_tokens = 0
    prefill(decoder, target, [seq], chunk=5)
    assert all(draft.cache[slot_of(seq.target_kv.block_table, BLOCK_SIZE, p)] == ("context", p, seq.token_ids[p])
               for p in range(len(seq) - 1))
    for _ in range(5):
        commit([seq], decoder.run_round([seq]))
    assert not draft.bad_reads
