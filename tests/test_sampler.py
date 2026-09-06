import sys
import types
import unittest

import torch


try:
    import flash_attn  # noqa: F401
except ImportError:
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.flash_attn_varlen_func = None
    flash_attn.flash_attn_with_kvcache = None
    sys.modules["flash_attn"] = flash_attn

from nanovllm.layers.sampler import Sampler


@torch.compile
def compiled_softmax_reference(logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    return probs.div_(noise).argmax(dim=-1)


class TestSampler(unittest.TestCase):

    def test_exponential_race_is_algebraically_equivalent(self):
        generator = torch.Generator().manual_seed(20260906)
        for batch_size, vocab_size in [(1, 2), (8, 97), (32, 1000)]:
            for dtype in [torch.float32, torch.bfloat16]:
                with self.subTest(batch_size=batch_size, vocab_size=vocab_size, dtype=dtype):
                    logits = torch.randn(batch_size, vocab_size, generator=generator).to(dtype).float()
                    temperatures = torch.rand(batch_size, generator=generator).add_(0.1)
                    logits.div_(temperatures.unsqueeze(1))
                    noise = torch.empty_like(logits).exponential_(1).clamp_min_(1e-10)

                    expected = torch.softmax(logits, dim=-1).div_(noise).argmax(dim=-1)
                    actual = logits.sub(noise.log()).argmax(dim=-1)

                    torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_compiled_cuda_matches_softmax_reference_with_the_same_seed(self):
        generator = torch.Generator(device="cuda").manual_seed(20260906)
        for batch_size in [1, 8, 32]:
            with self.subTest(batch_size=batch_size):
                logits = torch.randn(
                    batch_size,
                    4096,
                    generator=generator,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                temperatures = torch.rand(batch_size, generator=generator, device="cuda").add_(0.1)

                torch.manual_seed(1)
                compiled_softmax_reference(logits.clone(), temperatures)
                torch.manual_seed(1)
                Sampler()(logits.clone(), temperatures)

                torch.manual_seed(1234)
                expected = compiled_softmax_reference(logits.clone(), temperatures)
                torch.manual_seed(1234)
                actual = Sampler()(logits.clone(), temperatures)

                torch.testing.assert_close(actual, expected)

    def test_is_invariant_to_a_shared_logit_offset(self):
        logits = torch.randn(16, 257)
        temperatures = torch.linspace(0.2, 1.5, 16)
        sampler = Sampler()
        sampler(logits.clone(), temperatures)

        torch.manual_seed(4321)
        baseline = sampler(logits.clone(), temperatures)
        torch.manual_seed(4321)
        shifted = sampler(logits.add(1000), temperatures)

        torch.testing.assert_close(shifted, baseline)

    def test_matches_the_expected_distribution(self):
        probabilities = torch.tensor([0.1, 0.3, 0.6])
        logits = probabilities.log().repeat(50_000, 1)
        temperatures = torch.ones(logits.size(0))

        torch.manual_seed(9876)
        tokens = Sampler()(logits, temperatures)
        frequencies = tokens.bincount(minlength=3).float().div_(tokens.numel())

        torch.testing.assert_close(frequencies, probabilities, atol=0.01, rtol=0)


if __name__ == "__main__":
    unittest.main()
