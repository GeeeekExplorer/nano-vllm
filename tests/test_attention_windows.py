import unittest

import torch

from nanovllm.layers.attention_windows import Attention
from nanovllm.utils.context import reset_context, set_context


def reference(q, k, v, offset):
    # Expand one KV head to both query heads, then use explicit causal scores.
    k = k.expand(-1, q.size(1), -1)
    v = v.expand(-1, q.size(1), -1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / q.size(-1) ** 0.5
    q_pos = torch.arange(offset, offset + q.size(0))
    k_pos = torch.arange(k.size(0))
    scores.masked_fill_(k_pos[None, None, :] > q_pos[None, :, None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


class WindowsAttentionTest(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_prefill_without_cache(self):
        torch.manual_seed(1)
        attention = Attention(2, 4, 0.5, 1)
        q = torch.randn(3, 2, 4)
        k = torch.randn(3, 1, 4)
        v = torch.randn(3, 1, 4)
        set_context(True, torch.tensor([0, 3]), torch.tensor([0, 3]))
        torch.testing.assert_close(attention(q, k, v), reference(q, k, v, 0))

    def test_prefill_multiple_sequences_without_cache(self):
        torch.manual_seed(4)
        attention = Attention(2, 4, 0.5, 1)
        q = torch.randn(5, 2, 4)
        k = torch.randn(5, 1, 4)
        v = torch.randn(5, 1, 4)
        offsets = torch.tensor([0, 2, 5])
        set_context(True, offsets, offsets)
        expected = torch.cat((reference(q[:2], k[:2], v[:2], 0),
                              reference(q[2:], k[2:], v[2:], 0)))
        torch.testing.assert_close(attention(q, k, v), expected)

    def test_prefill_with_prefix_and_paged_cache(self):
        torch.manual_seed(2)
        attention = Attention(2, 4, 0.5, 1)
        attention.k_cache = torch.zeros(4, 2, 1, 4)
        attention.v_cache = torch.zeros_like(attention.k_cache)
        old_k = torch.randn(2, 1, 4)
        old_v = torch.randn(2, 1, 4)
        attention.k_cache.view(-1, 1, 4)[2:4] = old_k
        attention.v_cache.view(-1, 1, 4)[2:4] = old_v
        q = torch.randn(2, 2, 4)
        k = torch.randn(2, 1, 4)
        v = torch.randn(2, 1, 4)
        set_context(True, torch.tensor([0, 2]), torch.tensor([0, 4]),
                    slot_mapping=torch.tensor([4, 5]), block_tables=torch.tensor([[1, 2]]))
        actual = attention(q, k, v)
        expected = reference(q, torch.cat((old_k, k)), torch.cat((old_v, v)), 2)
        torch.testing.assert_close(actual, expected)

    def test_decode_from_paged_cache(self):
        torch.manual_seed(3)
        attention = Attention(2, 4, 0.5, 1)
        attention.k_cache = torch.zeros(4, 2, 1, 4)
        attention.v_cache = torch.zeros_like(attention.k_cache)
        old_k = torch.randn(3, 1, 4)
        old_v = torch.randn(3, 1, 4)
        attention.k_cache.view(-1, 1, 4)[2:5] = old_k
        attention.v_cache.view(-1, 1, 4)[2:5] = old_v
        q = torch.randn(1, 2, 4)
        k = torch.randn(1, 1, 4)
        v = torch.randn(1, 1, 4)
        set_context(False, slot_mapping=torch.tensor([5]), context_lens=torch.tensor([4]),
                    block_tables=torch.tensor([[1, 2]]))
        actual = attention(q, k, v)
        expected = reference(q, torch.cat((old_k, k)), torch.cat((old_v, v)), 3)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
