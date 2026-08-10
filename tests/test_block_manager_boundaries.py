import unittest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


class BlockManagerBoundaryTest(unittest.TestCase):
    def test_decode_allocates_at_lengths_257_and_513(self):
        block_size = 256
        manager = BlockManager(num_blocks=8, block_size=block_size)
        seq = Sequence(list(range(block_size)))
        manager.allocate(seq, num_cached_blocks=0)
        self.assertEqual(len(seq.block_table), 1)

        seq.append_token(1001)
        self.assertEqual(len(seq), 257)
        self.assertEqual(seq.last_block_num_tokens, 1)
        self.assertTrue(manager.can_append(seq))
        manager.may_append(seq)
        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(
            seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1,
            seq.block_table[-1] * block_size,
        )

        for token_id in range(1002, 1257):
            seq.append_token(token_id)
            manager.may_append(seq)
        self.assertEqual(len(seq), 512)
        self.assertEqual(len(seq.block_table), 2)

        seq.append_token(1257)
        self.assertEqual(len(seq), 513)
        self.assertEqual(seq.last_block_num_tokens, 1)
        self.assertTrue(manager.can_append(seq))
        manager.may_append(seq)
        self.assertEqual(len(seq.block_table), 3)


if __name__ == "__main__":
    unittest.main()
