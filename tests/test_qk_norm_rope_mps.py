import unittest

import torch

from nanovllm.layers.qk_norm_rope_mps import qk_norm_rope_mps, supports_qk_norm_rope_mps


def reference(x, weight, cos_sin, positions, eps):
    """The PyTorch path on CPU: RMSNorm in float32, rounded to dtype, then the rotary embedding."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x = (x.to(dtype) * weight).float()
    cos, sin = cos_sin[positions].unsqueeze(1).chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(dtype)


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
class QKNormRopeMPSTest(unittest.TestCase):
    def test_matches_separate_norm_and_rotary(self):
        torch.manual_seed(21)
        eps = 1e-6
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for head_dim in (64, 128, 256):
                for num_q_heads, num_k_heads, num_tokens in ((4, 2, 1), (2, 2, 7), (16, 8, 300)):
                    with self.subTest(dtype=dtype, head_dim=head_dim, heads=(num_q_heads, num_k_heads)):
                        width = (num_q_heads + 2 * num_k_heads) * head_dim
                        qkv = torch.randn(num_tokens, width).to(dtype)
                        q_weight = (1 + 0.2 * torch.randn(head_dim)).to(dtype)
                        k_weight = (1 + 0.2 * torch.randn(head_dim)).to(dtype)
                        angles = torch.rand(50, head_dim // 2) * 6.28
                        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
                        positions = torch.randint(0, 50, (num_tokens,))
                        cache = cos_sin.unsqueeze(1).to("mps")
                        self.assertTrue(supports_qk_norm_rope_mps(qkv.to("mps"), head_dim, cache))
                        q, k = qk_norm_rope_mps(
                            qkv.to("mps"), positions.to("mps"), q_weight.to("mps"), k_weight.to("mps"), cache,
                            num_q_heads, num_k_heads, head_dim, eps,
                        )
                        q_size = num_q_heads * head_dim
                        k_size = num_k_heads * head_dim
                        expected_q = reference(qkv[:, :q_size].view(num_tokens, num_q_heads, head_dim), q_weight, cos_sin, positions, eps)
                        expected_k = reference(qkv[:, q_size:q_size + k_size].view(num_tokens, num_k_heads, head_dim), k_weight, cos_sin, positions, eps)
                        tolerance = 1e-5 if dtype == torch.float32 else 1e-2
                        for actual, expected in ((q, expected_q), (k, expected_k)):
                            self.assertEqual(actual.shape, expected.shape)
                            self.assertTrue(torch.allclose(actual.cpu().float(), expected.float(), atol=tolerance, rtol=tolerance))

    def test_unsupported_head_dim_is_reported(self):
        qkv = torch.randn(2, 4 * 16, device="mps")
        cache = torch.zeros(8, 1, 16, device="mps")
        self.assertFalse(supports_qk_norm_rope_mps(qkv, 16, cache))


if __name__ == "__main__":
    unittest.main()
