import sys
import types
import unittest


# Importing nano-vllm loads the attention module, even though these CPU-only tests
# never execute attention. Provide a lightweight substitute when flash-attn is not
# installed in the test environment.
try:
    import flash_attn  # noqa: F401
except ImportError:
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.flash_attn_varlen_func = None
    flash_attn.flash_attn_with_kvcache = None
    sys.modules["flash_attn"] = flash_attn

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


class FakeTokenizer:

    def __init__(self):
        self.batch_calls = 0
        self.encode_calls = 0

    @staticmethod
    def _tokenize(prompt):
        return [ord(char) for char in prompt] or [0]

    def __call__(self, prompts, **kwargs):
        self.batch_calls += 1
        self.batch_kwargs = kwargs
        return {"input_ids": [self._tokenize(prompt) for prompt in prompts]}

    def encode(self, prompt):
        self.encode_calls += 1
        return self._tokenize(prompt)

    def decode(self, token_ids):
        return ""


class FakeScheduler:

    def __init__(self):
        self.seqs = []

    def add(self, seq):
        self.seqs.append(seq)

    def is_finished(self):
        return True


class TestLLMEngineGenerate(unittest.TestCase):

    def make_engine(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine.tokenizer = FakeTokenizer()
        engine.scheduler = FakeScheduler()
        return engine

    def test_string_prompts_are_tokenized_as_one_batch(self):
        engine = self.make_engine()
        prompts = [f"prompt {i}" for i in range(32)]

        outputs = engine.generate(prompts, SamplingParams(), use_tqdm=False)

        self.assertEqual(outputs, [])
        self.assertEqual(engine.tokenizer.batch_calls, 1)
        self.assertEqual(engine.tokenizer.encode_calls, 0)
        self.assertEqual(
            engine.tokenizer.batch_kwargs,
            {
                "add_special_tokens": True,
                "padding": False,
                "truncation": False,
                "return_attention_mask": False,
                "return_token_type_ids": False,
            },
        )
        self.assertEqual(
            [seq.prompt_token_ids for seq in engine.scheduler.seqs],
            [FakeTokenizer._tokenize(prompt) for prompt in prompts],
        )

    def test_small_string_batches_keep_per_prompt_encoding(self):
        engine = self.make_engine()
        prompts = [f"prompt {i}" for i in range(31)]

        engine.generate(prompts, SamplingParams(), use_tqdm=False)

        self.assertEqual(engine.tokenizer.batch_calls, 0)
        self.assertEqual(engine.tokenizer.encode_calls, 31)
        self.assertEqual(
            [seq.prompt_token_ids for seq in engine.scheduler.seqs],
            [FakeTokenizer._tokenize(prompt) for prompt in prompts],
        )

    def test_token_id_prompts_bypass_tokenizer(self):
        engine = self.make_engine()
        prompts = [[1, 2], [3, 4]]

        engine.generate(prompts, SamplingParams(), use_tqdm=False)

        self.assertEqual(engine.tokenizer.batch_calls, 0)
        self.assertEqual(engine.tokenizer.encode_calls, 0)
        self.assertEqual([seq.prompt_token_ids for seq in engine.scheduler.seqs], prompts)

    def test_empty_prompt_list_bypasses_tokenizer(self):
        engine = self.make_engine()

        outputs = engine.generate([], SamplingParams(), use_tqdm=False)

        self.assertEqual(outputs, [])
        self.assertEqual(engine.tokenizer.batch_calls, 0)


if __name__ == "__main__":
    unittest.main()
