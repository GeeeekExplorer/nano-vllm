from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


class StopTokenIDsTest(unittest.TestCase):
    def setUp(self):
        self.block_size = patch.object(Sequence, "block_size", 256)
        self.block_size.start()
        self.addCleanup(self.block_size.stop)

    def scheduler(self, token_budget=1024):
        return Scheduler(SimpleNamespace(
            max_num_seqs=4, max_num_batched_tokens=token_budget, eos=99,
            kvcache_block_size=256, num_kvcache_blocks=8,
        ))

    def step(self, scheduler, tokens):
        seqs, is_prefill = scheduler.schedule()
        self.assertEqual(len(seqs), len(tokens))
        scheduler.postprocess(seqs, tokens, is_prefill)

    def assert_released(self, scheduler):
        self.assertTrue(scheduler.is_finished())
        manager = scheduler.block_manager
        self.assertFalse(manager.used_block_ids)
        self.assertEqual(len(manager.free_block_ids), len(manager.blocks))
        self.assertTrue(all(block.ref_count == 0 for block in manager.blocks))

    def test_stop_on_first_token_or_decode(self):
        for generated in ([7], [5, 8], [0]):
            with self.subTest(generated=generated):
                scheduler = self.scheduler()
                seq = Sequence([1, 2], SamplingParams(stop_token_ids=[0, 7, 8], max_tokens=8))
                scheduler.add(seq)
                for token in generated:
                    self.step(scheduler, [token])
                self.assertEqual(seq.completion_token_ids, generated)
                self.assertFalse(seq.block_table)
                self.assert_released(scheduler)

    def test_requests_have_independent_stop_ids(self):
        scheduler = self.scheduler()
        first = Sequence([1], SamplingParams(stop_token_ids=[7]))
        second = Sequence([2], SamplingParams(stop_token_ids=[8]))
        scheduler.add(first)
        scheduler.add(second)
        self.step(scheduler, [7, 7])
        self.assertTrue(first.is_finished)
        self.assertFalse(second.is_finished)
        self.assertTrue(second.block_table)
        self.step(scheduler, [8])
        self.assertEqual(second.completion_token_ids, [7, 8])
        self.assert_released(scheduler)

    def test_ignore_eos_does_not_ignore_explicit_stops(self):
        for stops, generated in (([7], [99, 7]), ([99], [99])):
            with self.subTest(stops=stops):
                scheduler = self.scheduler()
                seq = Sequence([1], SamplingParams(ignore_eos=True, stop_token_ids=stops))
                scheduler.add(seq)
                for token in generated:
                    self.step(scheduler, [token])
                self.assertEqual(seq.completion_token_ids, generated)
                self.assert_released(scheduler)

    def test_eos_and_length_limits_still_apply(self):
        for stops in (None, [], [7]):
            for ignore_eos, generated in ((False, [99]), (True, [4, 5])):
                with self.subTest(stops=stops, ignore_eos=ignore_eos):
                    scheduler = self.scheduler()
                    seq = Sequence([1], SamplingParams(max_tokens=2, ignore_eos=ignore_eos, stop_token_ids=stops))
                    scheduler.add(seq)
                    for token in generated:
                        self.step(scheduler, [token])
                    self.assertEqual(seq.completion_token_ids, generated)
                    self.assert_released(scheduler)

    def test_prompt_and_intermediate_prefill_chunks_do_not_stop(self):
        scheduler = self.scheduler(token_budget=256)
        seq = Sequence([7] * 513, SamplingParams(stop_token_ids=[7]))
        scheduler.add(seq)
        for cached_tokens in (256, 512):
            self.step(scheduler, [7])
            self.assertEqual(seq.num_cached_tokens, cached_tokens)
            self.assertEqual(seq.completion_token_ids, [])
            self.assertFalse(seq.is_finished)
        self.step(scheduler, [5])
        self.assertFalse(seq.is_finished)
        self.step(scheduler, [7])
        self.assertEqual(seq.completion_token_ids, [5, 7])
        self.assert_released(scheduler)

    def test_stopping_one_request_preserves_shared_prefix(self):
        scheduler = self.scheduler(token_budget=257)
        prefix = [1] * 256
        first = Sequence(prefix + [2], SamplingParams(stop_token_ids=[7]))
        scheduler.add(first)
        self.step(scheduler, [5])
        prefix_id = first.block_table[0]
        second = Sequence(prefix + [3], SamplingParams(stop_token_ids=[7]))
        scheduler.add(second)
        seqs, is_prefill = scheduler.schedule()
        self.assertEqual(seqs, [second])
        self.assertEqual(second.block_table[0], prefix_id)
        self.assertEqual(scheduler.block_manager.blocks[prefix_id].ref_count, 2)
        scheduler.postprocess(seqs, [7], is_prefill)
        self.assertTrue(second.is_finished)
        self.assertFalse(first.is_finished)
        self.assertEqual(scheduler.block_manager.blocks[prefix_id].ref_count, 1)
        self.step(scheduler, [7])
        self.assert_released(scheduler)

    def test_stop_ids_are_snapshotted_for_each_request(self):
        stop_ids = [7]
        seq = Sequence([1], SamplingParams(stop_token_ids=stop_ids))
        stop_ids.append(8)
        scheduler = self.scheduler()
        scheduler.add(seq)
        self.step(scheduler, [8])
        self.assertFalse(seq.is_finished)
        self.step(scheduler, [7])
        self.assert_released(scheduler)

    def test_generate_preserves_order_and_stopping_token(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = self.scheduler()
        engine.tokenizer = SimpleNamespace(decode=lambda ids: " ".join(map(str, ids)))
        run = Mock(side_effect=[[7, 7], [8]])
        engine.model_runner = SimpleNamespace(call=run)
        outputs = engine.generate(
            [[1], [2]], [SamplingParams(stop_token_ids=[7]), SamplingParams(stop_token_ids=[8])],
            use_tqdm=False,
        )
        self.assertEqual(outputs, [
            {"text": "7", "token_ids": [7]},
            {"text": "7 8", "token_ids": [7, 8]},
        ])
        self.assertEqual(run.call_count, 2)
        self.assert_released(engine.scheduler)

    def test_invalid_stop_ids_are_rejected(self):
        for stop_ids in ([-1], [1.5], ["7"], [True], "7", 7):
            with self.subTest(stop_ids=stop_ids):
                with self.assertRaisesRegex(AssertionError, "stop_token_ids"):
                    SamplingParams(stop_token_ids=stop_ids)

    def test_existing_positional_arguments_are_unchanged(self):
        params = SamplingParams(0.6, 8, True)
        self.assertEqual((params.temperature, params.max_tokens, params.ignore_eos), (0.6, 8, True))
        self.assertIsNone(params.stop_token_ids)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_sampler_to_scheduler(self):
        scheduler = self.scheduler()
        for stop_id in (7, 8):
            scheduler.add(Sequence([1], SamplingParams(stop_token_ids=[stop_id])))
        sampler = Sampler()
        with torch.inference_mode():
            for targets in ([7, 7], [8]):
                seqs, is_prefill = scheduler.schedule()
                logits = torch.full((len(seqs), 128), -float("inf"), device="cuda")
                indices = torch.tensor(targets, device="cuda").unsqueeze(1)
                logits.scatter_(1, indices, 0)
                tokens = sampler(logits, torch.ones(len(seqs), device="cuda")).tolist()
                self.assertEqual(tokens, targets)
                scheduler.postprocess(seqs, tokens, is_prefill)
        self.assert_released(scheduler)


if __name__ == "__main__":
    unittest.main()
