import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy
import xxhash


def load_scheduler():
    # Use the real scheduler, sequences and block manager without importing the GPU engine.
    root = Path(__file__).resolve().parents[1] / "nanovllm"
    package = ModuleType("nanovllm")
    package.__path__ = [str(root)]
    engine = ModuleType("nanovllm.engine")
    engine.__path__ = [str(root / "engine")]
    config = ModuleType("nanovllm.config")
    config.Config = object
    with patch.dict(sys.modules, {
        "nanovllm": package,
        "nanovllm.engine": engine,
        "nanovllm.config": config,
    }):
        scheduler = importlib.import_module("nanovllm.engine.scheduler")
        sequence = importlib.import_module("nanovllm.engine.sequence")
        params = importlib.import_module("nanovllm.sampling_params")
    return scheduler.Scheduler, sequence.Sequence, params.SamplingParams


Scheduler, Sequence, SamplingParams = load_scheduler()


def make_scheduler(token_budget, max_num_seqs=4, num_blocks=16):
    return Scheduler(SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=token_budget,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=num_blocks,
    ))


class SchedulerTokenBudgetTest(unittest.TestCase):
    def test_decode_budget_after_multiple_prefill_steps(self):
        scheduler = make_scheduler(token_budget=2)
        requests = [Sequence([i], SamplingParams(max_tokens=3)) for i in range(4)]
        for seq in requests:
            scheduler.add(seq)

        for _ in range(2):
            seqs, is_prefill = scheduler.schedule()
            self.assertTrue(is_prefill)
            self.assertEqual(len(seqs), 2)
            scheduler.postprocess(seqs, [42] * len(seqs), is_prefill)

        seqs, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(seqs, requests[:2])
        self.assertEqual(sum(seq.num_scheduled_tokens for seq in seqs), 2)
        self.assertEqual(list(scheduler.running), requests)
        self.assertTrue(all(seq.num_scheduled_tokens == 0 for seq in requests[2:]))
        scheduler.postprocess(seqs, [42] * len(seqs), is_prefill)

        for _ in range(10):
            if scheduler.is_finished():
                break
            seqs, is_prefill = scheduler.schedule()
            self.assertLessEqual(sum(seq.num_scheduled_tokens for seq in seqs), 2)
            scheduler.postprocess(seqs, [42] * len(seqs), is_prefill)
        self.assertTrue(scheduler.is_finished())
        self.assertTrue(all(seq.num_completion_tokens == 3 for seq in requests))
        self.assertTrue(all(seq.is_finished and not seq.block_table for seq in requests))
        self.assertFalse(scheduler.block_manager.used_block_ids)
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 16)

    def test_decode_respects_both_limits(self):
        for token_budget, sequence_limit in ((1, 4), (2, 4), (4, 4), (8, 4), (8, 2)):
            with self.subTest(token_budget=token_budget, sequence_limit=sequence_limit):
                scheduler = make_scheduler(token_budget=4)
                for i in range(4):
                    scheduler.add(Sequence([i], SamplingParams(max_tokens=3)))
                seqs, is_prefill = scheduler.schedule()
                scheduler.postprocess(seqs, [42] * len(seqs), is_prefill)
                scheduler.max_num_batched_tokens = token_budget
                scheduler.max_num_seqs = sequence_limit

                seqs, is_prefill = scheduler.schedule()
                self.assertFalse(is_prefill)
                self.assertEqual(len(seqs), min(token_budget, sequence_limit, 4))
                self.assertTrue(all(seq.num_scheduled_tokens == 1 for seq in seqs))
                self.assertEqual(len(scheduler.running), 4)

    def test_block_boundary_preemption_preserves_budget_and_progress(self):
        scheduler = make_scheduler(token_budget=2, num_blocks=5)
        requests = [Sequence([i] * 256, SamplingParams(max_tokens=3)) for i in range(4)]
        for seq in requests:
            scheduler.add(seq)

        with patch.object(scheduler, "preempt", wraps=scheduler.preempt) as preempt:
            for _ in range(800):
                if scheduler.is_finished():
                    break
                seqs, is_prefill = scheduler.schedule()
                self.assertLessEqual(sum(seq.num_scheduled_tokens for seq in seqs), 2)
                scheduler.postprocess(seqs, [42] * len(seqs), is_prefill)
            self.assertGreater(preempt.call_count, 0)

        self.assertTrue(scheduler.is_finished())
        self.assertTrue(all(seq.num_completion_tokens == 3 for seq in requests))
        self.assertFalse(scheduler.block_manager.used_block_ids)
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 5)


if __name__ == "__main__":
    unittest.main()
