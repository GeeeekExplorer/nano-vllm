import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from nanovllm.engine.model_runner import ModelRunner


class FakeGraph:

    def __init__(self):
        self.replayed = False

    def replay(self):
        self.replayed = True


class CudaGraphReplayTest(unittest.TestCase):

    def devices(self):
        return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    def make_runner(self, block_table_width=4, device="cpu"):
        runner = ModelRunner.__new__(ModelRunner)
        runner.enforce_eager = False
        runner.graph_bs = [1, 2, 4]
        runner.graph_vars = {
            "block_tables": torch.empty(4, block_table_width, dtype=torch.int32, device=device),
        }
        return runner

    def test_capture_reserves_the_first_decode_block(self):
        runner = self.make_runner()
        runner.config = SimpleNamespace(max_model_len=4096)
        runner.block_size = 256

        self.assertEqual(runner._cudagraph_block_table_width(), 17)

    def test_replay_requires_runtime_inputs_to_fit_captured_buffers(self):
        runner = self.make_runner()
        context = SimpleNamespace(block_tables=torch.empty(2, 4, dtype=torch.int32))

        cases = [
            (torch.empty(2), False, False, True),
            (torch.empty(2), True, False, False),
            (torch.empty(5), False, False, False),
            (torch.empty(2), False, True, False),
        ]
        with patch("nanovllm.engine.model_runner.get_context", return_value=context):
            for input_ids, is_prefill, enforce_eager, expected in cases:
                runner.enforce_eager = enforce_eager
                with self.subTest(
                    batch_size=input_ids.size(0),
                    is_prefill=is_prefill,
                    enforce_eager=enforce_eager,
                ):
                    self.assertEqual(runner._can_use_cudagraph(input_ids, is_prefill), expected)

        context.block_tables = torch.empty(2, 5, dtype=torch.int32)
        with patch("nanovllm.engine.model_runner.get_context", return_value=context):
            runner.enforce_eager = False
            self.assertFalse(runner._can_use_cudagraph(torch.empty(2), False))

    def test_oversized_block_table_falls_back_to_eager(self):
        for device in self.devices():
            with self.subTest(device=device):
                runner = self.make_runner(device=device)
                runner.model = Mock()
                runner.model.return_value = torch.zeros(2, 8, device=device)
                runner.model.compute_logits.side_effect = lambda outputs: outputs
                context = SimpleNamespace(
                    block_tables=torch.empty(2, 5, dtype=torch.int32, device=device),
                )
                input_ids = torch.tensor([11, 12], device=device)
                positions = torch.tensor([256, 128], device=device)

                with patch("nanovllm.engine.model_runner.get_context", return_value=context):
                    runner.run_model(input_ids, positions, False)

                runner.model.assert_called_once_with(input_ids, positions)

    def test_replay_clears_stale_block_table_entries(self):
        for device in self.devices():
            with self.subTest(device=device):
                runner = self.make_runner(device=device)
                graph = FakeGraph()
                runner.graphs = {2: graph}
                runner.graph_vars.update({
                    "input_ids": torch.zeros(4, dtype=torch.int64, device=device),
                    "positions": torch.zeros(4, dtype=torch.int64, device=device),
                    "slot_mapping": torch.zeros(4, dtype=torch.int32, device=device),
                    "context_lens": torch.zeros(4, dtype=torch.int32, device=device),
                    "block_tables": torch.full((4, 4), 99, dtype=torch.int32, device=device),
                    "outputs": torch.zeros(4, 8, device=device),
                })
                runner.model = SimpleNamespace(compute_logits=lambda outputs: outputs)
                context = SimpleNamespace(
                    slot_mapping=torch.tensor([7, 8], dtype=torch.int32, device=device),
                    context_lens=torch.tensor([257, 129], dtype=torch.int32, device=device),
                    block_tables=torch.tensor([[1, 2], [3, -1]], dtype=torch.int32, device=device),
                )

                with patch("nanovllm.engine.model_runner.get_context", return_value=context):
                    runner.run_model(
                        torch.tensor([11, 12], device=device),
                        torch.tensor([256, 128], device=device),
                        False,
                    )

                expected = torch.tensor([
                    [1, 2, -1, -1],
                    [3, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                ], dtype=torch.int32, device=device)
                self.assertTrue(graph.replayed)
                self.assertTrue(torch.equal(runner.graph_vars["block_tables"], expected))


if __name__ == "__main__":
    unittest.main()
