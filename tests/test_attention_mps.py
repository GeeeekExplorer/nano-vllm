import unittest

import torch

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import reset_context, set_context


def reference_attention(q, k, v, query_start):
    # CPU reference with direct token indexing.
    q = q.float().transpose(0, 1)
    k = k.float().repeat_interleave(q.shape[0] // k.shape[1], dim=1).transpose(0, 1)
    v = v.float().repeat_interleave(q.shape[0] // v.shape[1], dim=1).transpose(0, 1)
    scores = q @ k.transpose(-1, -2) * (q.shape[-1] ** -0.5)
    query_positions = torch.arange(query_start, query_start + q.shape[1])
    key_positions = torch.arange(k.shape[1])
    scores.masked_fill_(key_positions[None, None, :] > query_positions[None, :, None], float("-inf"))
    return (scores.softmax(-1) @ v).transpose(0, 1)


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
class AttentionMPSTest(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_prefill_and_decode_with_two_sequences(self):
        torch.manual_seed(7)
        q = torch.randn(5, 4, 8, device="mps")
        k = torch.randn(5, 2, 8, device="mps")
        v = torch.randn(5, 2, 8, device="mps")
        attention = Attention(4, 8, 8 ** -0.5, 2)
        attention.k_cache = torch.zeros(2, 4, 2, 8, device="mps")
        attention.v_cache = torch.zeros_like(attention.k_cache)
        set_context(True, slot_mapping=torch.tensor([0, 1, 2, 4, 5], dtype=torch.int32, device="mps"),
                    q_lens=[3, 2], k_lens=[3, 2])
        output = attention(q, k, v)
        expected = torch.cat((reference_attention(q[:3].cpu(), k[:3].cpu(), v[:3].cpu(), 0),
                              reference_attention(q[3:].cpu(), k[3:].cpu(), v[3:].cpu(), 0)))
        self.assertTrue(torch.allclose(output.cpu(), expected, atol=1e-4, rtol=1e-4))

        q_next = torch.randn(2, 4, 8, device="mps")
        k_next = torch.randn(2, 2, 8, device="mps")
        v_next = torch.randn(2, 2, 8, device="mps")
        set_context(False, slot_mapping=torch.tensor([3, 6], dtype=torch.int32, device="mps"),
                    block_tables=torch.tensor([[0], [1]], dtype=torch.int32, device="mps"),
                    context_lens=torch.tensor([4, 3], dtype=torch.int32, device="mps"), k_lens=[4, 3])
        decoded = attention(q_next, k_next, v_next)
        expected_decoded = torch.cat((
            reference_attention(q_next[:1].cpu(), torch.cat((k[:3], k_next[:1])).cpu(),
                                torch.cat((v[:3], v_next[:1])).cpu(), 3),
            reference_attention(q_next[1:].cpu(), torch.cat((k[3:], k_next[1:])).cpu(),
                                torch.cat((v[3:], v_next[1:])).cpu(), 2),
        ))
        self.assertTrue(torch.allclose(decoded.cpu(), expected_decoded, atol=1e-4, rtol=1e-4))

    def test_prefill_with_cached_prefix(self):
        torch.manual_seed(8)
        q = torch.randn(2, 4, 8, device="mps")
        k = torch.randn(4, 2, 8, device="mps")
        v = torch.randn(4, 2, 8, device="mps")
        attention = Attention(4, 8, 8 ** -0.5, 2)
        attention.k_cache = torch.zeros(1, 4, 2, 8, device="mps")
        attention.v_cache = torch.zeros_like(attention.k_cache)
        attention.k_cache[0, :2] = k[:2]
        attention.v_cache[0, :2] = v[:2]
        set_context(True, slot_mapping=torch.tensor([2, 3], dtype=torch.int32, device="mps"),
                    block_tables=torch.tensor([[0]], dtype=torch.int32, device="mps"),
                    q_lens=[2], k_lens=[4])
        output = attention(q, k[2:], v[2:])
        expected = reference_attention(q.cpu(), k.cpu(), v.cpu(), 2)
        self.assertTrue(torch.allclose(output.cpu(), expected, atol=1e-4, rtol=1e-4))

    def test_batched_decode_with_equal_lengths(self):
        torch.manual_seed(9)
        attention = Attention(4, 8, 8 ** -0.5, 2)
        attention.k_cache = torch.zeros(2, 4, 2, 8, device="mps")
        attention.v_cache = torch.zeros_like(attention.k_cache)
        old_k = torch.randn(2, 2, 2, 8, device="mps")
        old_v = torch.randn(2, 2, 2, 8, device="mps")
        attention.k_cache[:, :2] = old_k
        attention.v_cache[:, :2] = old_v
        q = torch.randn(2, 4, 8, device="mps")
        k = torch.randn(2, 2, 8, device="mps")
        v = torch.randn(2, 2, 8, device="mps")
        set_context(False, slot_mapping=torch.tensor([2, 6], dtype=torch.int32, device="mps"),
                    block_tables=torch.tensor([[0], [1]], dtype=torch.int32, device="mps"),
                    context_lens=torch.tensor([3, 3], dtype=torch.int32, device="mps"), k_lens=[3, 3])
        output = attention(q, k, v)
        expected = torch.cat([
            reference_attention(q[i:i + 1].cpu(), torch.cat((old_k[i], k[i:i + 1])).cpu(),
                                torch.cat((old_v[i], v[i:i + 1])).cpu(), 2)
            for i in range(2)
        ])
        self.assertTrue(torch.allclose(output.cpu(), expected, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
