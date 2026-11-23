import unittest
from unittest.mock import patch

import torch
from torch import nn

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.qk_norm_rope import fused_qk_norm_rope
from nanovllm.layers.rotary_embedding import RotaryEmbedding
from nanovllm.models.qwen3 import Qwen3Attention
from nanovllm.utils.context import reset_context, set_context


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "requires CUDA BF16")
class TestDecodeQKNormRoPE(unittest.TestCase):
    @torch.inference_mode()
    def test_packed_qkv_matches_compiled_reference(self):
        gen = torch.Generator(device="cuda").manual_seed(23)
        dim, eps = 128, 1e-6
        rope = RotaryEmbedding(dim, dim, 4096, 1_000_000).cuda()
        qnorm, knorm = [RMSNorm(dim, eps).cuda().bfloat16() for _ in range(2)]
        for norm in (qnorm, knorm):
            norm.weight.normal_(mean=1, std=0.25, generator=gen)
        # Include a partially filled head group, as can occur with tensor parallelism.
        for batch, qheads, kheads in ((1, 16, 8), (8, 4, 2), (256, 16, 8)):
            with self.subTest(batch=batch, qheads=qheads, kheads=kheads):
                packed = torch.randn((batch, qheads + 2 * kheads, dim), device="cuda",
                                     dtype=torch.bfloat16, generator=gen)
                q, k, _ = packed.split([qheads, kheads, kheads], dim=1)
                k[0, 0].zero_()
                before = packed.clone()
                pos = torch.randint(4096, (batch,), device="cuda", generator=gen)
                pos[0] = 4095
                if batch > 1:
                    pos[-1] = 0
                expected = rope(pos, qnorm(q), knorm(k))
                actual = fused_qk_norm_rope(q, k, qnorm.weight, knorm.weight, pos, rope.cos_sin_cache, eps)
                for output, ref in zip(actual, expected):
                    torch.testing.assert_close(output, ref, rtol=0.016, atol=0.016)
                self.assertTrue(torch.equal(packed, before))

    @torch.inference_mode()
    def test_attention_dispatch(self):
        self.addCleanup(reset_context)
        # Supply packed QKV directly; only the normalization/rotation dispatch is under test.
        for flag, prefill, dtype, dim, bias, should_fuse in (
            (True, False, torch.bfloat16, 128, False, True),
            (False, False, torch.bfloat16, 128, False, False),
            (True, True, torch.bfloat16, 128, False, False),
            (True, False, torch.float16, 128, False, False),
            (True, False, torch.bfloat16, 64, False, False),
            (True, False, torch.bfloat16, 128, True, False),
        ):
            with self.subTest(flag=flag, prefill=prefill, dtype=dtype, dim=dim, bias=bias):
                with patch("torch.distributed.get_world_size", return_value=1), \
                     patch("torch.distributed.get_rank", return_value=0):
                    attn = Qwen3Attention(128, 4, 2, head_dim=dim, qkv_bias=bias, max_position=256).cuda()
                attn.qkv_proj = attn.o_proj = nn.Identity()
                if not bias:
                    attn.q_norm.to(dtype)
                    attn.k_norm.to(dtype)
                pos = torch.tensor([0, 31], device="cuda")
                packed = torch.randn((2, 8 * dim), dtype=dtype, device="cuda")
                set_context(prefill)
                with patch.object(attn.attn, "forward", side_effect=lambda q, k, v: q):
                    expected = attn(pos, packed)
                    attn.fuse_decode_qk_rope = flag
                    with patch("nanovllm.models.qwen3.fused_qk_norm_rope", wraps=fused_qk_norm_rope) as kernel:
                        actual = attn(pos, packed)
                    self.assertEqual(kernel.call_count, int(should_fuse))
                    tolerance = 0.016 if should_fuse else 0
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
