from contextlib import ExitStack
import unittest
from unittest.mock import patch

from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine


class EngineOptionsTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Exercise constructor argument handling without weights or GPU workers.
        self.post_init = self.stack.enter_context(patch.object(Config, "__post_init__"))
        self.process_context = self.stack.enter_context(patch("nanovllm.engine.llm_engine.mp.get_context"))
        self.model_runner = self.stack.enter_context(patch("nanovllm.engine.llm_engine.ModelRunner"))
        self.stack.enter_context(patch("nanovllm.engine.llm_engine.AutoTokenizer.from_pretrained"))
        self.stack.enter_context(patch("nanovllm.engine.llm_engine.Scheduler"))
        self.stack.enter_context(patch("nanovllm.engine.llm_engine.atexit.register"))

    def test_unknown_options_fail_before_initialization(self):
        for key in ("max_num_seq", "tensor_parallel_szie", "dtype"):
            with self.subTest(option=key):
                with self.assertRaisesRegex(TypeError, key):
                    LLMEngine("unused-model", **{key: 8})
        self.post_init.assert_not_called()
        self.process_context.assert_not_called()
        self.model_runner.assert_not_called()

    def test_valid_options_reach_model_runner(self):
        LLMEngine("unused-model", max_num_seqs=10, enforce_eager=True)
        config, rank, events = self.model_runner.call_args.args
        self.assertEqual(config.model, "unused-model")
        self.assertEqual(config.max_num_seqs, 10)
        self.assertTrue(config.enforce_eager)
        self.assertEqual(config.tensor_parallel_size, 1)
        self.assertEqual(rank, 0)
        self.assertEqual(events, [])

    def test_defaults_still_initialize(self):
        LLMEngine("unused-model")
        config = self.model_runner.call_args.args[0]
        self.assertEqual(config.max_num_seqs, 512)
        self.assertFalse(config.enforce_eager)
        self.post_init.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
