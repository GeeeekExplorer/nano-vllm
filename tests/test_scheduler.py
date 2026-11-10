import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class SchedulerAdmissionTest(unittest.TestCase):
    def test_active_limit_admits_waiting_requests_after_completion(self):
        config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=32, eos=-1,
                                 kvcache_block_size=256, num_kvcache_blocks=8)
        scheduler = Scheduler(config)
        requests = [Sequence([i + 1] * 10, SamplingParams(max_tokens=2, ignore_eos=True))
                    for i in range(3)]
        for request in requests:
            scheduler.add(request)

        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, requests[:2])
        scheduler.postprocess(scheduled, [20, 21], is_prefill)
        self.assertEqual(len(scheduler.running), 2)
        self.assertEqual(len(scheduler.waiting), 1)

        scheduled, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, requests[:2])
        scheduler.postprocess(scheduled, [22, 23], is_prefill)

        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, requests[2:])
        scheduler.postprocess(scheduled, [24], is_prefill)
        scheduled, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, requests[2:])
        scheduler.postprocess(scheduled, [25], is_prefill)
        self.assertTrue(scheduler.is_finished())


if __name__ == "__main__":
    unittest.main()
