"""CPU-only shutdown tests; model and GPU dependencies are replaced at import."""

import importlib.util
import subprocess
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


@dataclass
class Config:
    model: str
    tensor_parallel_size: int = 1
    kvcache_block_size: int = 256
    eos: int = -1


def load_engine():
    modules = {}
    for name, attrs in {
        "tqdm": {},
        "tqdm.auto": {"tqdm": Mock()},
        "transformers": {"AutoTokenizer": Mock()},
        "torch": {},
        "torch.multiprocessing": {"get_context": Mock()},
        "nanovllm": {},
        "nanovllm.engine": {},
        "nanovllm.config": {"Config": Config},
        "nanovllm.sampling_params": {"SamplingParams": type("SamplingParams", (), {})},
        "nanovllm.engine.sequence": {"Sequence": type("Sequence", (), {})},
        "nanovllm.engine.scheduler": {"Scheduler": Mock()},
        "nanovllm.engine.model_runner": {"ModelRunner": Mock()},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        modules[name] = module
    modules["torch"].multiprocessing = modules["torch.multiprocessing"]
    path = Path(__file__).resolve().parents[1] / "nanovllm/engine/llm_engine.py"
    spec = importlib.util.spec_from_file_location("engine_exit_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class EngineExitTests(unittest.TestCase):
    def test_repeated_exit_and_callback_clean_up_once(self):
        module = load_engine()
        workers = [Mock(), Mock()]
        module.mp.get_context.return_value.Process.side_effect = workers
        with patch.object(module.atexit, "register") as register, \
             patch.object(module.atexit, "unregister") as unregister:
            engine = module.LLMEngine("unused", tensor_parallel_size=3)
            runner = engine.model_runner
            register.assert_called_once_with(engine.exit)

            engine.exit()
            engine.exit()
            register.call_args.args[0]()  # A previously obtained callback is safe too.

            runner.call.assert_called_once_with("exit")
            for worker in workers:
                worker.join.assert_called_once_with()
            # exit() is also an atexit callback and must not mutate that registry.
            unregister.assert_not_called()
            self.assertFalse(hasattr(engine, "model_runner"))

    def test_runner_error_is_propagated_and_callback_is_preserved(self):
        module = load_engine()
        with patch.object(module.atexit, "register"), \
             patch.object(module.atexit, "unregister") as unregister:
            engine = module.LLMEngine("unused")
            runner = engine.model_runner
            runner.call.side_effect = RuntimeError("shutdown failed")

            with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
                engine.exit()

            self.assertIs(engine.model_runner, runner)
            unregister.assert_not_called()

    def test_interrupted_worker_join_can_be_retried_without_stopping_runner_twice(self):
        module = load_engine()
        workers = [Mock(), Mock()]
        module.mp.get_context.return_value.Process.side_effect = workers
        workers[1].join.side_effect = [KeyboardInterrupt, None]
        with patch.object(module.atexit, "register") as register:
            engine = module.LLMEngine("unused", tensor_parallel_size=3)
            runner = engine.model_runner

            with self.assertRaises(KeyboardInterrupt):
                engine.exit()
            self.assertFalse(hasattr(engine, "model_runner"))
            runner.call.assert_called_once_with("exit")

            # The interpreter callback must still finish outstanding joins.
            register.call_args.args[0]()
            self.assertEqual(workers[1].join.call_count, 2)
            runner.call.assert_called_once_with("exit")
            engine.exit()
            self.assertEqual(workers[1].join.call_count, 2)

    def test_interpreter_shutdown_after_automatic_or_manual_exit(self):
        for manual_exit in (False, True):
            with self.subTest(manual_exit=manual_exit):
                script = f"""
from test_engine_exit import load_engine
module = load_engine()
engine = module.LLMEngine("unused")
engine.model_runner.call.side_effect = lambda name: print("runner:" + name)
if {manual_exit!r}:
    engine.exit()
"""
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=Path(__file__).resolve().parent,
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "runner:exit\n")
                self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
