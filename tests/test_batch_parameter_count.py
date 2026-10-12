"""Exercise batch admission without loading a GPU model."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


def load_engine():
    dependencies = {}
    for name in (
        "tqdm.auto",
        "transformers",
        "torch",
        "torch.multiprocessing",
        "nanovllm",
        "nanovllm.config",
        "nanovllm.sampling_params",
        "nanovllm.engine",
        "nanovllm.engine.sequence",
        "nanovllm.engine.scheduler",
        "nanovllm.engine.model_runner",
    ):
        dependencies[name] = ModuleType(name)
    dependencies["tqdm.auto"].tqdm = Mock()
    dependencies["transformers"].AutoTokenizer = Mock()
    for name, attribute in (
        ("nanovllm.config", "Config"),
        ("nanovllm.sampling_params", "SamplingParams"),
        ("nanovllm.engine.sequence", "Sequence"),
        ("nanovllm.engine.scheduler", "Scheduler"),
        ("nanovllm.engine.model_runner", "ModelRunner"),
    ):
        setattr(dependencies[name], attribute, type(attribute, (), {}))
    path = Path(__file__).parents[1] / "nanovllm/engine/llm_engine.py"
    spec = importlib.util.spec_from_file_location("batch_test_engine", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


class BatchParameterCountTest(unittest.TestCase):
    def setUp(self):
        self.module = load_engine()
        self.engine = self.module.LLMEngine.__new__(self.module.LLMEngine)
        self.engine.add_request = Mock()
        self.engine.is_finished = Mock(return_value=True)
        self.engine.tokenizer = Mock()

    def test_mismatched_counts_fail_before_admitting_any_requests(self):
        for prompts, params in (
            (["a", "b"], [object()]),
            (["a"], [object(), object()]),
            (["a"], []),
            ([], [object()]),
        ):
            with self.subTest(prompts=prompts, params=params):
                with self.assertRaisesRegex(ValueError, "one entry per prompt"):
                    self.engine.generate(prompts, params, use_tqdm=False)
                self.engine.add_request.assert_not_called()
                self.module.tqdm.assert_not_called()

    def test_matching_counts_keep_parameter_order(self):
        params = [object(), object()]
        self.engine.generate(["a", "b"], params, use_tqdm=False)
        self.assertEqual(
            self.engine.add_request.call_args_list,
            [unittest.mock.call("a", params[0]), unittest.mock.call("b", params[1])],
        )

    def test_single_sampling_params_still_broadcasts(self):
        params = self.module.SamplingParams()
        self.engine.generate([[1], [2]], params, use_tqdm=False)
        self.assertEqual(
            self.engine.add_request.call_args_list,
            [unittest.mock.call([1], params), unittest.mock.call([2], params)],
        )

    def test_empty_batch_with_empty_parameters_is_valid(self):
        self.assertEqual(self.engine.generate([], [], use_tqdm=False), [])
        self.engine.add_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
