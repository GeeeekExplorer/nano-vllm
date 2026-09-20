from collections import deque
from itertools import islice
import random
from types import SimpleNamespace

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def seq(value, blocks=2):
    return Sequence([value] * (blocks * 256))


def cache(manager, request):
    manager.allocate(request, manager.can_allocate(request))
    request.num_scheduled_tokens = len(request) - request.num_cached_tokens
    manager.hash_blocks(request)
    manager.deallocate(request)


def populated():
    manager = BlockManager(4, 256)
    for value in (11, 22):
        cache(manager, seq(value))
    return manager


def test_preserves_waiting_prefix_and_remaining_queue_order():
    original, candidate = populated(), populated()
    order = list(candidate.free_block_ids)
    original.allocate(seq(33), 0)
    current = seq(33)
    candidate.allocate(current, 0, waiting_seqs=[current, seq(11)])
    assert original.can_allocate(seq(11)) == 0
    assert candidate.can_allocate(seq(11)) == 1
    assert list(candidate.free_block_ids) == [b for b in order if b not in current.block_table]


def test_retention_is_lazy_and_resolved_once():
    manager = populated()
    visits = []

    def waiting():
        visits.append(1)
        yield seq(11)

    manager.allocate(seq(33), 0, waiting_seqs=waiting())
    assert visits == [1]
    empty = BlockManager(4, 256)
    empty.allocate(seq(33), 0, waiting_seqs=waiting())
    assert visits == [1]


def test_lazy_lookup_after_allocating_uncached_blocks():
    manager = BlockManager(5, 256)
    cache(manager, seq(11))
    visits = []

    def waiting():
        visits.append(len(manager.used_block_ids))
        yield seq(11)

    current = seq(33, 4)
    manager.allocate(current, 0, waiting_seqs=waiting())
    assert visits == [3]
    manager.deallocate(current)
    assert manager.can_allocate(seq(11)) == 1


def test_all_retained_blocks_fall_back_in_original_order():
    manager = populated()
    order = list(manager.free_block_ids)
    current = seq(33, 4)
    manager.allocate(current, 0, waiting_seqs=[seq(11, 3), seq(22, 3)])
    assert current.block_table == order
    manager.deallocate(current)
    assert len(manager.free_block_ids) == 4


def test_lookup_is_bounded_read_only_and_stops_at_first_miss():
    manager = populated()
    waiting = [seq(99), seq(11)]
    before = repr((vars(manager), [vars(b) for b in manager.blocks], [vars(s) for s in waiting]))
    assert manager._get_retained_blocks(seq(33), islice(waiting, 1)) == set()
    retained = manager._get_retained_blocks(seq(33), islice(waiting, 2))
    assert len(retained) == 1
    assert before == repr((vars(manager), [vars(b) for b in manager.blocks], [vars(s) for s in waiting]))
    # A later hash may still exist, but cannot be used through a prefix hole.
    request = seq(11, 3)
    h = manager.compute_hash(request.block(0))
    del manager.hash_to_block_id[h]
    assert manager._get_retained_blocks(seq(33), [request]) == set()


def test_skips_current_and_partially_prefilled_requests():
    manager = populated()
    current = seq(11)
    partial = seq(22)
    partial.block_table = [0]
    assert manager._get_retained_blocks(current, [current, partial]) == set()


def test_live_shared_prefix_is_never_evicted():
    manager = BlockManager(8, 256)
    first, second = seq(11), seq(11)
    manager.allocate(first, 0)
    first.num_scheduled_tokens = len(first)
    manager.hash_blocks(first)
    manager.allocate(second, manager.can_allocate(second))
    shared = first.block_table[0]
    manager.allocate(seq(33, 5), 0, waiting_seqs=[seq(11)])
    assert manager.blocks[shared].ref_count == 2
    assert manager.blocks[shared].token_ids == [11] * 256


def test_lookup_exception_does_not_change_queue_or_leak_retention():
    manager = populated()
    order = list(manager.free_block_ids)

    def broken():
        raise RuntimeError("lookup failed")
        yield

    with pytest.raises(RuntimeError, match="lookup failed"):
        manager.allocate(seq(33), 0, waiting_seqs=broken())
    assert list(manager.free_block_ids) == order
    current = seq(33)
    manager.allocate(current, 0)
    assert current.block_table == order[:2]


def test_decode_boundary_retains_waiting_prefix():
    manager = BlockManager(6, 256)
    for value in (11, 22, 33):
        cache(manager, seq(value))
    current = seq(33)
    manager.allocate(current, manager.can_allocate(current))
    protected = manager.free_block_ids[0]
    current.append_token(1000)
    manager.may_append(current, waiting_seqs=[seq(11)])
    assert len(current.block_table) == 3
    assert current.block_table[-1] != protected
    assert manager.can_allocate(seq(11)) == 1


@pytest.mark.parametrize("window,hits", [(0, 0), (1, 0), (2, 1)])
def test_scheduler_wires_bounded_lookahead(window, hits):
    config = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1024, eos=-1,
                             kvcache_block_size=256, num_kvcache_blocks=4,
                             prefix_cache_lookahead=window)
    scheduler = Scheduler(config)
    scheduler.block_manager = populated()
    scheduler.waiting = deque([seq(33), seq(11)])
    scheduler.schedule()
    assert scheduler.block_manager.can_allocate(seq(11)) == hits


def test_scheduler_decode_looks_ahead_when_waiting_request_cannot_fit():
    config = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1024, eos=-1,
                             kvcache_block_size=256, num_kvcache_blocks=6,
                             prefix_cache_lookahead=2)
    scheduler = Scheduler(config)
    manager = scheduler.block_manager
    for value in (11, 22, 33):
        cache(manager, seq(value))
    current = seq(33)
    manager.allocate(current, manager.can_allocate(current))
    current.append_token(1000)
    scheduler.running.append(current)
    scheduler.waiting.extend([seq(99, 6), seq(11)])
    protected = manager.free_block_ids[0]
    scheduled, prefill = scheduler.schedule()
    assert scheduled == [current] and not prefill
    assert current.block_table[-1] != protected
    assert manager.can_allocate(seq(11)) == 1


def test_randomized_allocator_accounting_and_cached_tokens():
    rng = random.Random(42)
    manager = BlockManager(24, 256)
    live = []
    for _ in range(1000):
        if live and rng.random() < .4:
            manager.deallocate(live.pop(rng.randrange(len(live))))
        else:
            current = seq(rng.randrange(5), rng.randrange(2, 6))
            cached = manager.can_allocate(current)
            if cached >= 0:
                waiting = [seq(rng.randrange(5), 4) for _ in range(4)]
                manager.allocate(current, cached, waiting)
                for i in range(cached):
                    assert manager.blocks[current.block_table[i]].token_ids == current.block(i)
                current.num_scheduled_tokens = len(current) - current.num_cached_tokens
                manager.hash_blocks(current)
                live.append(current)
        references = [0] * len(manager.blocks)
        for request in live:
            for block_id in request.block_table:
                references[block_id] += 1
        assert references == [b.ref_count for b in manager.blocks]
        assert manager.used_block_ids == {i for i, count in enumerate(references) if count}
        assert set(manager.free_block_ids) == {i for i, count in enumerate(references) if not count}
        assert len(manager.free_block_ids) == len(set(manager.free_block_ids))
        assert all(manager.blocks[i].hash == h for h, i in manager.hash_to_block_id.items())
