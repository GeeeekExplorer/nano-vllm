from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence

BLOCK_SIZE = 256


def make_seq(num_tokens, start=0):
    return Sequence(list(range(start, start + num_tokens)))


def test_allocate_new_sequence_uses_one_block_per_256_tokens():
    manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
    seq = make_seq(600)

    assert manager.can_allocate(seq) == 0

    manager.allocate(seq, 0)

    assert len(seq.block_table) == 3
    assert len(manager.free_block_ids) == 5
    assert len(manager.used_block_ids) == 3
    assert seq.num_cached_tokens == 0
    assert all(manager.blocks[block_id].ref_count == 1 for block_id in seq.block_table)


def test_can_allocate_reports_insufficient_capacity():
    manager = BlockManager(num_blocks=2, block_size=BLOCK_SIZE)
    seq = make_seq(3 * BLOCK_SIZE)

    assert manager.can_allocate(seq) == -1


def test_prefix_cache_indexes_only_complete_blocks():
    manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
    seq = make_seq(600)
    manager.allocate(seq, 0)
    seq.num_scheduled_tokens = 600

    manager.hash_blocks(seq)

    blocks = [manager.blocks[block_id] for block_id in seq.block_table]
    assert blocks[0].hash != -1
    assert blocks[1].hash != -1
    # The tail block holds only 88 tokens and must never be reused.
    assert blocks[2].hash == -1


def test_prefix_cache_reuses_complete_blocks():
    manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
    first = make_seq(600)
    manager.allocate(first, 0)
    first.num_scheduled_tokens = 600
    manager.hash_blocks(first)
    cached_block_ids = list(first.block_table[:2])
    manager.deallocate(first)

    reused = make_seq(700)
    num_cached_blocks = manager.can_allocate(reused)
    assert num_cached_blocks == 2

    manager.allocate(reused, num_cached_blocks)

    assert reused.block_table[:2] == cached_block_ids
    assert reused.num_cached_tokens == 2 * BLOCK_SIZE
    assert len(reused.block_table) == 3


def test_shared_prefix_blocks_use_reference_counting():
    manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
    a = make_seq(700)
    manager.allocate(a, 0)
    a.num_scheduled_tokens = 700
    manager.hash_blocks(a)
    shared_ids = list(a.block_table[:2])
    assert len(manager.free_block_ids) == 5

    b = make_seq(700)
    manager.allocate(b, manager.can_allocate(b))

    assert b.block_table[:2] == shared_ids
    assert all(manager.blocks[block_id].ref_count == 2 for block_id in shared_ids)
    assert len(manager.free_block_ids) == 4

    manager.deallocate(a)
    assert all(manager.blocks[block_id].ref_count == 1 for block_id in shared_ids)
    assert all(block_id in manager.used_block_ids for block_id in shared_ids)
    assert len(manager.free_block_ids) == 5

    manager.deallocate(b)
    assert b.block_table == []
    assert len(manager.free_block_ids) == 8
    assert manager.used_block_ids == set()


def test_reusing_a_block_removes_its_stale_hash_index():
    manager = BlockManager(num_blocks=2, block_size=BLOCK_SIZE)
    seq = make_seq(BLOCK_SIZE)
    manager.allocate(seq, 0)
    seq.num_scheduled_tokens = BLOCK_SIZE
    manager.hash_blocks(seq)
    hash_value = manager.blocks[seq.block_table[0]].hash
    assert manager.hash_to_block_id[hash_value] == seq.block_table[0]

    manager.deallocate(seq)
    # Free list order is [1, 0] here, so the second allocation reuses the
    # block that still carries the hash.
    manager.allocate(make_seq(BLOCK_SIZE, start=1000), 0)
    manager.allocate(make_seq(BLOCK_SIZE, start=2000), 0)

    assert hash_value not in manager.hash_to_block_id
