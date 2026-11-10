import tempfile
import unittest
from pathlib import Path

import torch

from tools.make_tiny_qwen3 import create_model
from nanovllm import LLM, SamplingParams


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
class MPSInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.tempdir.name)
        create_model(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_chunked_prefill_and_prefix_reuse(self):
        llm = LLM(str(self.model_path), device="mps", max_model_len=320,
                  max_num_batched_tokens=128, max_num_seqs=1, gpu_memory_utilization=0.5)
        try:
            prompt = [4 + i % 124 for i in range(300)]
            params = SamplingParams(max_tokens=3, ignore_eos=True)
            first = llm.generate([prompt], params, use_tqdm=False)
            self.assertEqual(len(first[0]["token_ids"]), 3)
            self.assertGreater(len(llm.scheduler.block_manager.hash_to_block_id), 0)
            second = llm.generate([prompt], params, use_tqdm=False)
            self.assertEqual(len(second[0]["token_ids"]), 3)
        finally:
            llm.exit()


if __name__ == "__main__":
    unittest.main()
