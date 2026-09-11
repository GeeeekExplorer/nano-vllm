"""纯 CPU 调度回归：验证 running decode 会与 waiting prefill 同批。"""

from types import SimpleNamespace
import unittest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


class MixedSchedulerTest(unittest.TestCase):
    """验证 decode-first 调度的批构成与关键状态不变量。"""

    def test_decode_and_prefill_share_a_batch(self):
        """已有 decode 必须先入批，剩余预算可接纳一个 waiting prefill。"""
        # Sequence 固定以 256 token 为一块；用一个无需扩块的 decode 构造 running 状态。
        config = SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=512,
            eos=-1,
            kvcache_block_size=256,
            num_kvcache_blocks=4,
            prefill_reserve_blocks=0,
        )
        scheduler = Scheduler(config)

        decode = Sequence(list(range(256)), SamplingParams(max_tokens=2))
        cached_blocks, _ = scheduler.block_manager.can_allocate(decode)
        scheduler.block_manager.allocate(decode, cached_blocks)
        decode.num_cached_tokens = len(decode) - 1
        decode.is_prefill = False
        decode.status = SequenceStatus.RUNNING
        scheduler.running.append(decode)

        prefill = Sequence(list(range(256, 512)), SamplingParams(max_tokens=2))
        scheduler.waiting.append(prefill)

        scheduled, is_prefill, has_prefill = scheduler.schedule()

        self.assertEqual(scheduled, [decode, prefill])
        self.assertFalse(is_prefill)
        self.assertTrue(has_prefill)
        self.assertEqual(decode.num_scheduled_tokens, 1)
        self.assertTrue(decode.should_sample)
        self.assertEqual(prefill.num_scheduled_tokens, 256)
        self.assertTrue(prefill.should_sample)


if __name__ == "__main__":
    unittest.main()
