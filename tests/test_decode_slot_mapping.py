"""Regression tests for the decode slot mapping.

`BlockManager.may_append` decides whether a new physical block is needed for
the next token, and `ModelRunner.prepare_decode` turns the sequence state into
a flat slot index in the KV cache. The two must agree, otherwise a token is
written to a slot that belongs to a different token or to an unallocated block.

A sequence keeps the invariant `num_cached_tokens == len(seq) - 1` while it is
being decoded: `Scheduler.postprocess` appends the sampled token to the
sequence, but its KV entry is only written by the next forward pass.
"""

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence

BLOCK_SIZE = 256


def prefill(manager, seq):
    num_cached_blocks = manager.can_allocate(seq)
    assert num_cached_blocks == 0
    manager.allocate(seq, num_cached_blocks)
    seq.num_cached_tokens = seq.num_tokens
    # The engine samples one token at the end of the prefill and appends it.
    seq.append_token(999)


@pytest.mark.parametrize(
    "prompt_len",
    [
        BLOCK_SIZE - 1,
        BLOCK_SIZE,
        BLOCK_SIZE + 1,
        2 * BLOCK_SIZE - 1,
        2 * BLOCK_SIZE,
        2 * BLOCK_SIZE + 1,
    ],
)
@pytest.mark.parametrize("decode_steps", [1, 4, BLOCK_SIZE])
def test_decode_slot_stays_inside_the_allocated_block(prompt_len, decode_steps):
    manager = BlockManager(num_blocks=64, block_size=BLOCK_SIZE)
    seq = Sequence(list(range(prompt_len)))
    prefill(manager, seq)

    for _ in range(decode_steps):
        assert manager.can_append(seq)
        manager.may_append(seq)

        token_index = len(seq) - 1
        logical_block = token_index // BLOCK_SIZE
        assert logical_block < len(seq.block_table)

        # Mirrors the slot computation in ModelRunner.prepare_decode.
        slot = seq.block_table[-1] * BLOCK_SIZE + seq.last_block_num_tokens - 1
        expected = seq.block_table[logical_block] * BLOCK_SIZE + token_index % BLOCK_SIZE
        assert slot == expected

        seq.num_cached_tokens += 1
        seq.append_token(999)


@pytest.mark.parametrize("num_tokens", [BLOCK_SIZE - 1, BLOCK_SIZE, BLOCK_SIZE + 1, 2 * BLOCK_SIZE])
def test_can_append_and_may_append_agree(num_tokens):
    manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
    seq = Sequence(list(range(num_tokens)))
    manager.allocate(seq, 0)
    seq.num_cached_tokens = num_tokens
    seq.append_token(999)

    blocks_before = len(seq.block_table)
    needs_new_block = len(seq) % BLOCK_SIZE == 1
    assert manager.can_append(seq) == (len(manager.free_block_ids) >= needs_new_block)

    manager.may_append(seq)

    assert len(seq.block_table) == blocks_before + int(needs_new_block)
