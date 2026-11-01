from types import SimpleNamespace

import pytest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE = 4
EOS = 7


@pytest.fixture(autouse=True)
def small_blocks():
    previous = Sequence.block_size
    Sequence.block_size = BLOCK_SIZE
    yield
    Sequence.block_size = previous


def make_scheduler(num_speculative_tokens, num_blocks=16, max_num_seqs=4):
    config = SimpleNamespace(max_num_seqs=max_num_seqs, max_num_batched_tokens=64, eos=EOS,
                             kvcache_block_size=BLOCK_SIZE, num_kvcache_blocks=num_blocks,
                             enable_prefix_cache=False, num_speculative_tokens=num_speculative_tokens)
    return Scheduler(config)


def prefilled(scheduler, prompt, first_token, **params):
    """Add a sequence and run its prefill step so it sits at the round-entry invariants."""
    seq = Sequence(prompt, SamplingParams(**params))
    scheduler.add(seq)
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and seqs == [seq]
    scheduler.postprocess(seqs, [first_token], True)
    return seq


def test_prefill_allocates_and_advances_the_kv_state():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)
    assert seq.target_kv.block_table == [0, 1] and seq.target_kv.num_cached_tokens == 5
    assert len(seq) == 6 and seq.num_cached_tokens == len(seq) - 1


def test_decode_scheduling_reserves_len_plus_k():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)   # len 6 -> 9 tokens
    seqs, is_prefill = scheduler.schedule()
    assert not is_prefill and seqs == [seq]
    assert len(seq.target_kv.block_table) == 3


def test_k0_decode_reserves_exactly_like_upstream():
    scheduler = make_scheduler(num_speculative_tokens=0)
    seq = prefilled(scheduler, [1, 2, 3, 4], first_token=9, max_tokens=8)        # len 5 -> block for position 4
    scheduler.schedule()
    assert len(seq.block_table) == 2


def test_decode_preempts_the_last_running_sequence_when_the_pool_is_short():
    scheduler = make_scheduler(num_speculative_tokens=3, num_blocks=6)
    first = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)    # 2 blocks each
    second = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)
    seqs, is_prefill = scheduler.schedule()                  # len 6: a round needs 9 tokens = 3 blocks, all 6 used
    assert not is_prefill and seqs == [first, second]
    scheduler.postprocess_speculative(seqs, [[10, 11, 12, 13], [10, 11, 12, 13]])   # len 10: 13 tokens = 4 blocks
    seqs, is_prefill = scheduler.schedule()                  # no free block: the last running seq is preempted
    assert not is_prefill and seqs == [first]
    assert second.status == SequenceStatus.WAITING and second.target_kv.block_table == []
    assert len(first.target_kv.block_table) == 4


def test_postprocess_speculative_keeps_the_kv_invariant():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)
    scheduler.schedule()
    scheduler.postprocess_speculative([seq], [[10, 11, 12, 13]])          # all three accepted + bonus
    assert seq.completion_token_ids == [9, 10, 11, 12, 13]
    assert seq.target_kv.num_cached_tokens == len(seq) - 1
    assert seq in scheduler.running


def test_postprocess_speculative_after_partial_acceptance():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)
    scheduler.schedule()
    scheduler.postprocess_speculative([seq], [[10, 11]])                   # one accepted + correction
    assert seq.target_kv.num_cached_tokens == len(seq) - 1


def test_postprocess_speculative_stops_at_eos_and_frees_the_blocks():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8)
    scheduler.schedule()
    scheduler.postprocess_speculative([seq], [[10, EOS, 12, 13]])
    assert seq.completion_token_ids == [9, 10, EOS]
    assert seq.is_finished and seq not in scheduler.running
    assert len(scheduler.block_manager.free_block_ids) == 16


def test_postprocess_speculative_stops_at_max_tokens():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=3)
    scheduler.schedule()
    scheduler.postprocess_speculative([seq], [[10, 11, 12, 13]])
    assert seq.completion_token_ids == [9, 10, 11]
    assert seq.is_finished


def test_postprocess_speculative_ignore_eos_continues():
    scheduler = make_scheduler(num_speculative_tokens=3)
    seq = prefilled(scheduler, [1, 2, 3, 4, 5], first_token=9, max_tokens=8, ignore_eos=True)
    scheduler.schedule()
    scheduler.postprocess_speculative([seq], [[EOS, 11]])
    assert seq.completion_token_ids == [9, EOS, 11] and not seq.is_finished
