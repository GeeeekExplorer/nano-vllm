import pickle

import pytest

from nanovllm.engine.sequence import Sequence, SequenceStatus

BLOCK_SIZE = 256


@pytest.mark.parametrize(
    "num_tokens, num_blocks, last_block_num_tokens",
    [
        (1, 1, 1),
        (BLOCK_SIZE - 1, 1, BLOCK_SIZE - 1),
        (BLOCK_SIZE, 1, BLOCK_SIZE),
        (BLOCK_SIZE + 1, 2, 1),
        (2 * BLOCK_SIZE, 2, BLOCK_SIZE),
        (2 * BLOCK_SIZE + 1, 3, 1),
    ],
)
def test_block_geometry(num_tokens, num_blocks, last_block_num_tokens):
    seq = Sequence(list(range(num_tokens)))

    assert seq.num_blocks == num_blocks
    assert seq.last_block_num_tokens == last_block_num_tokens

    for i in range(num_blocks):
        start = i * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, num_tokens)
        assert seq.block(i) == list(range(start, end))


def test_append_token_updates_length_and_last_token():
    seq = Sequence([1, 2, 3])
    seq.append_token(4)

    assert len(seq) == 4
    assert seq.num_tokens == 4
    assert seq.last_token == 4
    assert seq.num_completion_tokens == 1
    assert seq.completion_token_ids == [4]


def test_pickle_round_trip_during_prefill():
    seq = Sequence([1, 2, 3, 4])
    seq.block_table = [7, 8]
    seq.num_cached_tokens = 2
    seq.num_scheduled_tokens = 2

    restored = pickle.loads(pickle.dumps(seq))

    assert restored.num_tokens == 4
    assert restored.num_prompt_tokens == 4
    assert restored.num_cached_tokens == 2
    assert restored.num_scheduled_tokens == 2
    assert restored.block_table == [7, 8]
    assert restored.token_ids == [1, 2, 3, 4]
    assert restored.last_token == 4


def test_status_starts_waiting():
    seq = Sequence([1])

    assert seq.status is SequenceStatus.WAITING
    assert seq.is_prefill
    assert not seq.is_finished
    assert seq.num_cached_tokens == 0
    assert seq.num_scheduled_tokens == 0
    assert seq.block_table == []
