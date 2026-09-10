import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from nanovllm.engine.model_runner import _sync_kv_cache_capacity


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class SyncKVCacheCapacityTest(unittest.TestCase):
    def test_single_rank_uses_local_capacity(self):
        with patch("nanovllm.engine.model_runner.torch.tensor") as tensor:
            self.assertEqual(_sync_kv_cache_capacity(17, 1), 17)
            tensor.assert_not_called()

    def test_multi_rank_uses_global_minimum(self):
        scalar = FakeScalar(17)

        def all_reduce(value, op):
            self.assertIs(value, scalar)
            self.assertEqual(op, dist.ReduceOp.MIN)
            value.value = 11

        with (
            patch("nanovllm.engine.model_runner.torch.tensor", return_value=scalar) as tensor,
            patch("nanovllm.engine.model_runner.dist.all_reduce", side_effect=all_reduce),
        ):
            self.assertEqual(_sync_kv_cache_capacity(17, 8), 11)

        tensor.assert_called_once_with(17, dtype=torch.int64, device="cuda")


if __name__ == "__main__":
    unittest.main()
