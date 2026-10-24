from types import SimpleNamespace

import pytest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE = 256


def make_config(**overrides):
    values = dict(
        max_num_seqs=4,
        max_num_batched_tokens=1024,
        eos=-1,
        kvcache_block_size=BLOCK_SIZE,
        num_kvcache_blocks=64,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_seq(num_tokens, start=0, max_tokens=8):
    token_ids = list(range(start, start + num_tokens))
    return Sequence(token_ids, SamplingParams(temperature=1.0, max_tokens=max_tokens))


def test_prefill_admits_at_most_max_num_seqs():
    scheduler = Scheduler(make_config(max_num_seqs=2))
    for i in range(3):
        scheduler.add(make_seq(10, start=i * 1000))

    seqs, is_prefill = scheduler.schedule()

    assert is_prefill
    assert len(seqs) == 2
    assert all(seq.status is SequenceStatus.RUNNING for seq in seqs)
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 2


def test_chunked_prefill_keeps_the_sequence_waiting():
    scheduler = Scheduler(make_config(max_num_batched_tokens=64))
    seq = make_seq(200)
    scheduler.add(seq)

    seqs, is_prefill = scheduler.schedule()

    assert is_prefill
    assert seqs == [seq]
    assert seq.num_scheduled_tokens == 64
    assert seq.status is SequenceStatus.WAITING
    assert seq.block_table  # blocks are reserved when the request is admitted
    assert list(scheduler.waiting) == [seq]

    scheduler.postprocess(seqs, [0], True)

    assert seq.num_cached_tokens == 64
    assert seq.num_scheduled_tokens == 0
    assert len(seq) == 200  # no token is sampled before the prefill finishes

    seqs, is_prefill = scheduler.schedule()
    assert is_prefill
    assert seq.num_scheduled_tokens == 64


def test_prefix_cache_reduces_scheduled_prefill_tokens():
    scheduler = Scheduler(make_config())
    first = make_seq(600)
    scheduler.add(first)
    seqs, _ = scheduler.schedule()
    scheduler.postprocess(seqs, [1], True)

    second = Sequence(list(range(600)) + list(range(1000, 1100)))
    scheduler.add(second)

    seqs, is_prefill = scheduler.schedule()

    assert is_prefill
    assert second in seqs
    assert second.num_cached_tokens == 2 * BLOCK_SIZE
    # The tail block of the first request (88 tokens) is never cached, so the
    # second request re-runs those tokens plus its own 100 new ones.
    assert second.num_scheduled_tokens == 188
    assert len(second.block_table) == 3


def test_decode_schedules_one_token_per_sequence():
    scheduler = Scheduler(make_config())
    seq = make_seq(10)
    scheduler.add(seq)

    seqs, _ = scheduler.schedule()
    scheduler.postprocess(seqs, [7], True)

    assert seq.status is SequenceStatus.RUNNING
    assert len(seq) == 11

    seqs, is_prefill = scheduler.schedule()

    assert not is_prefill
    assert seqs == [seq]
    assert seq.num_scheduled_tokens == 1
    assert not seq.is_prefill


def test_postprocess_finishes_at_max_tokens():
    scheduler = Scheduler(make_config())
    seq = make_seq(10, max_tokens=1)
    scheduler.add(seq)

    seqs, _ = scheduler.schedule()
    scheduler.postprocess(seqs, [7], True)

    assert seq.status is SequenceStatus.FINISHED
    assert seq not in scheduler.running
    assert seq.block_table == []


def test_decode_preempts_another_sequence_when_blocks_run_out():
    scheduler = Scheduler(make_config(num_kvcache_blocks=2))
    a = make_seq(BLOCK_SIZE - 1, start=0)
    b = make_seq(BLOCK_SIZE - 1, start=10_000)
    scheduler.add(a)
    scheduler.add(b)

    seqs, _ = scheduler.schedule()
    scheduler.postprocess(seqs, [1, 2], True)

    seqs, is_prefill = scheduler.schedule()
    assert not is_prefill
    assert len(seqs) == 2
    scheduler.postprocess(seqs, [1, 2], False)

    assert len(scheduler.block_manager.free_block_ids) == 0

    seqs, is_prefill = scheduler.schedule()

    assert not is_prefill
    assert seqs == [a]
    assert b.status is SequenceStatus.WAITING
    assert b.block_table == []
    # The block freed from `b` is immediately reused by `a`'s next token.
    assert len(a.block_table) == 2
    assert len(scheduler.block_manager.free_block_ids) == 0


@pytest.mark.xfail(
    strict=False,
    reason="known bug: a single decode sequence exhausting the KV cache trips "
    "`assert scheduled_seqs` instead of finishing or being rejected (#274, #279)",
)
def test_single_sequence_cannot_outgrow_the_cache():
    scheduler = Scheduler(make_config(num_kvcache_blocks=1, max_num_seqs=1))
    seq = make_seq(BLOCK_SIZE - 1)
    scheduler.add(seq)

    seqs, _ = scheduler.schedule()
    scheduler.postprocess(seqs, [1], True)

    seqs, _ = scheduler.schedule()
    scheduler.postprocess(seqs, [1], False)

    # The next decode step needs a second block, but the cache only has one.
    scheduler.schedule()
