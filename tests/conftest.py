import pytest

from nanovllm.engine.sequence import Sequence

BLOCK_SIZE = 256


@pytest.fixture(autouse=True)
def reset_sequence_block_size():
    """`Sequence.block_size` is global state set by `LLMEngine` from the config.

    Tests build `Sequence` objects directly, so keep them independent of
    whatever an engine instance or a previous test left behind.
    """
    original = Sequence.block_size
    Sequence.block_size = BLOCK_SIZE
    yield
    Sequence.block_size = original
