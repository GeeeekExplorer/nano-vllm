import unittest

import torch
import torch.nn.functional as F

from nanovllm.layers.paged_attention_mps import paged_decode_attention_shared_mps


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
class PagedAttentionMPSTest(unittest.TestCase):
    def test_shared_kv_head_kernel_matches_reference(self):
        torch.manual_seed(15)
        # Odd block sizes exercise SIMD position offsets across block boundaries.
        cases = (
            (4, 2, 4, (1, 7, 9)),
            (2, 2, 3, (2, 5, 11)),
            (8, 2, 5, (0, 4, 23)),
            (16, 8, 16, (257, 511, 2047)),
            (6, 2, 7, (3, 130, 700)),
        )
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for head_dim in (8, 70, 128, 256):
                for num_heads, num_kv_heads, block_size, lengths in cases:
                    with self.subTest(dtype=dtype, head_dim=head_dim, heads=(num_heads, num_kv_heads), block=block_size):
                        blocks_per_request = (max(lengths) + block_size - 1) // block_size
                        num_blocks = blocks_per_request * len(lengths)
                        query = torch.randn(len(lengths), num_heads, head_dim, device="mps").to(dtype)
                        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device="mps").to(dtype)
                        v_cache = torch.randn_like(k_cache)
                        tables = torch.randperm(num_blocks).to(torch.int32).reshape(len(lengths), -1).to("mps")
                        context_lens = torch.tensor(lengths, device="mps", dtype=torch.int32)
                        actual = paged_decode_attention_shared_mps(
                            query, k_cache, v_cache, tables, context_lens, head_dim ** -0.5,
                        ).cpu().float()
                        groups = num_heads // num_kv_heads
                        for i, length in enumerate(lengths):
                            if length == 0:
                                self.assertEqual(actual[i].abs().max().item(), 0.0)
                                continue
                            block_ids = tables[i, : (length + block_size - 1) // block_size].cpu().tolist()
                            key = k_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:length].cpu().float()
                            value = v_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:length].cpu().float()
                            key = key.repeat_interleave(groups, dim=1).transpose(0, 1).unsqueeze(0)
                            value = value.repeat_interleave(groups, dim=1).transpose(0, 1).unsqueeze(0)
                            q = query[i].cpu().float().unsqueeze(0).unsqueeze(2)
                            expected = F.scaled_dot_product_attention(q, key, value).squeeze(2).squeeze(0)
                            tolerance = 1e-4 if dtype == torch.float32 else 2e-2
                            self.assertTrue(torch.allclose(actual[i], expected, atol=tolerance, rtol=tolerance))



if __name__ == "__main__":
    unittest.main()
