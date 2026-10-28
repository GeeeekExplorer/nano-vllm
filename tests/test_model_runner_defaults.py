import importlib.util
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch


def load_model_runner():
    """Load the real runner without importing model or kernel dependencies."""
    modules = {}
    for name, attrs in {
        "nanovllm": {"__path__": []},
        "nanovllm.config": {"Config": object},
        "nanovllm.engine.sequence": {"Sequence": object},
        "nanovllm.models.qwen3": {"Qwen3ForCausalLM": Mock()},
        "nanovllm.layers.sampler": {"Sampler": Mock()},
        "nanovllm.utils.context": {
            "set_context": Mock(),
            "get_context": Mock(),
            "reset_context": Mock(),
        },
        "nanovllm.utils.loader": {"load_model": Mock()},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        modules[name] = module
    path = Path(__file__).resolve().parents[1] / "nanovllm/engine/model_runner.py"
    spec = importlib.util.spec_from_file_location("model_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class TestModelRunnerDefaults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_model_runner()

    def setUp(self):
        self.original_dtype = torch.get_default_dtype()
        self.original_device = torch.get_default_device()
        self.addCleanup(torch.set_default_dtype, self.original_dtype)
        self.addCleanup(torch.set_default_device, self.original_device)

    def check_initialization(self, device, failure=None, eager=False):
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float64)
        config = SimpleNamespace(
            hf_config=SimpleNamespace(dtype=torch.float16),
            kvcache_block_size=256,
            enforce_eager=eager,
            tensor_parallel_size=1,
            model="unused",
        )
        error = RuntimeError(f"injected {failure} failure")
        stages = []

        def stage(name):
            def run(*args, **kwargs):
                tensor = torch.empty(0)
                self.assertEqual(tensor.device.type, "meta")
                self.assertEqual(tensor.dtype, torch.float16)
                stages.append(name)
                if name == failure:
                    raise error

            return run

        def test_device(value):
            return "meta" if value == "cuda" else value

        torch_proxy = SimpleNamespace(
            cuda=SimpleNamespace(set_device=Mock()),
            get_default_dtype=torch.get_default_dtype,
            set_default_dtype=torch.set_default_dtype,
            set_default_device=lambda value: torch.set_default_device(test_device(value)),
            device=lambda value: torch.device(test_device(value)),
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.module, "torch", torch_proxy))
            stack.enter_context(patch.object(self.module.dist, "init_process_group"))
            for name in ("Qwen3ForCausalLM", "load_model", "Sampler"):
                stack.enter_context(patch.object(self.module, name, side_effect=stage(name)))
            for name in ("warmup_model", "allocate_kv_cache", "capture_cudagraph"):
                stack.enter_context(
                    patch.object(self.module.ModelRunner, name, side_effect=stage(name))
                )
            if failure:
                with self.assertRaises(RuntimeError) as caught:
                    self.module.ModelRunner(config, 0, [])
                self.assertIs(caught.exception, error)
            else:
                self.module.ModelRunner(config, 0, [])

        self.assertEqual(torch.get_default_dtype(), torch.float64)
        tensor = torch.empty(0)
        self.assertEqual(tensor.dtype, torch.float64)
        self.assertEqual(tensor.device.type, device)
        if failure:
            self.assertEqual(stages[-1], failure)
        else:
            self.assertEqual(
                stages,
                ["Qwen3ForCausalLM", "load_model", "Sampler", "warmup_model", "allocate_kv_cache"]
                + ([] if eager else ["capture_cudagraph"]),
            )

    def test_failures_restore_defaults(self):
        for device in ("cpu", "meta"):
            for failure in (
                "Qwen3ForCausalLM",
                "load_model",
                "Sampler",
                "warmup_model",
                "allocate_kv_cache",
                "capture_cudagraph",
            ):
                with self.subTest(device=device, failure=failure):
                    self.check_initialization(device, failure=failure)

    def test_success_restores_defaults(self):
        for device in ("cpu", "meta"):
            for eager in (True, False):
                with self.subTest(device=device, eager=eager):
                    self.check_initialization(device, eager=eager)


if __name__ == "__main__":
    unittest.main()
