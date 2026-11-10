import unittest

import torch

from nanovllm.layers.attention import store_kvcache


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
class StoreKVCacheMPSTest(unittest.TestCase):
    def test_contiguous_and_strided_inputs(self):
        slots = torch.tensor([5, -1, 2], dtype=torch.int32, device="mps")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                contiguous = torch.arange(24, dtype=torch.float32, device="mps").to(dtype).reshape(3, 2, 4)
                fused = torch.arange(72, dtype=torch.float32, device="mps").to(dtype).reshape(3, 3, 2, 4)
                cases = (
                    (contiguous, contiguous + 100),
                    (fused[:, 1], fused[:, 2]),
                    (contiguous, fused[:, 2]),
                )
                for key, value in cases:
                    with self.subTest(key_stride=key.stride(0), value_stride=value.stride(0)):
                        cache_shape = (2, 4, 2, 4)
                        actual_k = torch.zeros(cache_shape, dtype=dtype, device="mps")
                        actual_v = torch.zeros_like(actual_k)
                        expected_k = torch.zeros_like(actual_k)
                        expected_v = torch.zeros_like(actual_k)
                        expected_k[1, 1] = key[0]
                        expected_v[1, 1] = value[0]
                        expected_k[0, 2] = key[2]
                        expected_v[0, 2] = value[2]

                        store_kvcache(key, value, actual_k, actual_v, slots)
                        torch.mps.synchronize()
                        self.assertTrue(torch.equal(actual_k, expected_k))
                        self.assertTrue(torch.equal(actual_v, expected_v))

    def test_non_power_of_two_width(self):
        key = torch.arange(5 * 3 * 7, dtype=torch.float32, device="mps").reshape(5, 3, 7)
        value = key + 200
        slots = torch.tensor([0, 7, 3, -1, 4], dtype=torch.int32, device="mps")
        actual_k = torch.zeros((2, 4, 3, 7), dtype=torch.float32, device="mps")
        actual_v = torch.zeros_like(actual_k)
        expected_k = torch.zeros_like(actual_k)
        expected_v = torch.zeros_like(actual_k)
        for token, (block, position) in ((0, (0, 0)), (1, (1, 3)), (2, (0, 3)), (4, (1, 0))):
            expected_k[block, position] = key[token]
            expected_v[block, position] = value[token]

        store_kvcache(key, value, actual_k, actual_v, slots)
        torch.mps.synchronize()
        self.assertTrue(torch.equal(actual_k, expected_k))
        self.assertTrue(torch.equal(actual_v, expected_v))


if __name__ == "__main__":
    unittest.main()
