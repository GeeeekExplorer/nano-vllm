from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.context import get_context, reset_context, set_context


class DecodeModel(nn.Module):
    def forward(self, input_ids, positions):
        context = get_context()
        return torch.stack((input_ids, positions, context.context_lens, context.slot_mapping), dim=-1).float()

    def compute_logits(self, hidden_states):
        return hidden_states


def make_runner(max_num_seqs):
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        max_num_seqs=max_num_seqs, max_model_len=256,
        hf_config=SimpleNamespace(hidden_size=4),
    )
    runner.block_size = 256
    runner.enforce_eager = False
    runner.model = DecodeModel()
    return runner


class NoopGraph:
    def pool(self):
        return None


class CUDAGraphBucketTest(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_capture_covers_every_batch_limit(self):
        # Execute capture's real allocations and model calls on CPU.
        with patch("torch.cuda.CUDAGraph", NoopGraph), \
             patch("torch.cuda.graph", side_effect=lambda *args: nullcontext()), \
             patch("torch.cuda.synchronize"):
            for limit in range(1, 513):
                with self.subTest(limit=limit):
                    runner = make_runner(limit)
                    runner.capture_cudagraph()
                    buckets = runner.graph_bs
                    self.assertEqual(buckets, sorted(set(buckets)))
                    self.assertGreaterEqual(min(buckets), 1)
                    self.assertEqual(max(buckets), limit)
                    self.assertEqual(set(runner.graphs), set(buckets))
                    self.assertEqual(runner.graph_vars["input_ids"].numel(), limit)

    def test_default_and_large_limits_keep_existing_buckets(self):
        with patch("torch.cuda.CUDAGraph", NoopGraph), \
             patch("torch.cuda.graph", side_effect=lambda *args: nullcontext()), \
             patch("torch.cuda.synchronize"):
            for limit in (512, 513, 1024):
                with self.subTest(limit=limit):
                    runner = make_runner(limit)
                    runner.capture_cudagraph()
                    self.assertEqual(runner.graph_bs, [1, 2, 4, 8] + list(range(16, 513, 16)))
                    self.assertEqual(runner.graph_vars["input_ids"].numel(), 512)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_replay_matches_eager_across_batch_changes(self):
        with torch.device("cuda"), torch.inference_mode():
            for limit in (1, 3, 7, 10, 17, 33, 512):
                with self.subTest(limit=limit):
                    runner = make_runner(limit)
                    runner.capture_cudagraph()
                    for batch_size in (limit, 1, max(1, limit - 1), limit):
                        for step in range(3):
                            input_ids = torch.arange(batch_size, dtype=torch.int64) + step
                            positions = torch.full_like(input_ids, step + 5)
                            set_context(
                                False,
                                slot_mapping=torch.arange(batch_size, dtype=torch.int32) + step * 7,
                                context_lens=torch.full((batch_size,), step + 6, dtype=torch.int32),
                                block_tables=torch.zeros(batch_size, 1, dtype=torch.int32),
                            )
                            expected = runner.model(input_ids, positions)
                            actual = runner.run_model(input_ids, positions, False)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
