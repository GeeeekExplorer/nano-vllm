import hashlib
import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace

if importlib.util.find_spec("xxhash") is None:
    xxhash_stub = types.ModuleType("xxhash")

    class _XXH64:

        def __init__(self):
            self._hash = hashlib.blake2b(digest_size=8)

        def update(self, data):
            self._hash.update(data)

        def intdigest(self):
            return int.from_bytes(self._hash.digest(), "little")

    xxhash_stub.xxh64 = _XXH64
    sys.modules["xxhash"] = xxhash_stub

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler(max_num_batched_tokens=8, max_num_seqs=4, max_num_prefill_tokens_per_seq=None):
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_prefill_tokens_per_seq=max_num_prefill_tokens_per_seq,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=32,
    )
    Sequence.block_size = config.kvcache_block_size
    return Scheduler(config)


class SchedulerChunkedPrefillTest(unittest.TestCase):

    def test_default_prefill_limit_uses_full_batch_token_budget(self):
        scheduler = make_scheduler(max_num_batched_tokens=8)
        long_seq = Sequence(list(range(20)))
        short_seq = Sequence([100, 101])

        scheduler.add(long_seq)
        scheduler.add(short_seq)

        seqs, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [long_seq])
        self.assertEqual(long_seq.num_scheduled_tokens, 8)
        self.assertEqual(list(scheduler.waiting), [short_seq, long_seq])

    def test_prefill_limit_must_be_positive(self):
        with self.assertRaises(AssertionError):
            make_scheduler(max_num_batched_tokens=8, max_num_prefill_tokens_per_seq=0)

    def test_long_chunked_prefill_leaves_room_for_short_prefills(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=8,
            max_num_prefill_tokens_per_seq=4,
        )
        long_seq = Sequence(list(range(20)))
        short_seq_a = Sequence([100, 101])
        short_seq_b = Sequence([200, 201])

        scheduler.add(long_seq)
        scheduler.add(short_seq_a)
        scheduler.add(short_seq_b)

        seqs, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [long_seq, short_seq_a, short_seq_b])
        self.assertEqual([seq.num_scheduled_tokens for seq in seqs], [4, 2, 2])
        self.assertEqual(long_seq.status, SequenceStatus.WAITING)
        self.assertEqual(short_seq_a.status, SequenceStatus.RUNNING)
        self.assertEqual(short_seq_b.status, SequenceStatus.RUNNING)


if __name__ == "__main__":
    unittest.main()
