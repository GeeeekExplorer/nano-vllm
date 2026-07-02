import unittest
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch


def load_llm_engine():
    root = Path(__file__).resolve().parents[1]
    llm_engine_path = root / "nanovllm" / "engine" / "llm_engine.py"

    tqdm_module = types.ModuleType("tqdm")
    tqdm_auto_module = types.ModuleType("tqdm.auto")
    tqdm_auto_module.tqdm = object

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = object

    torch_module = types.ModuleType("torch")
    torch_module.__path__ = []
    torch_mp_module = types.ModuleType("torch.multiprocessing")

    nanovllm_module = types.ModuleType("nanovllm")
    nanovllm_module.__path__ = []
    engine_module = types.ModuleType("nanovllm.engine")
    engine_module.__path__ = []

    config_module = types.ModuleType("nanovllm.config")
    config_module.Config = type("Config", (), {})

    sampling_params_module = types.ModuleType("nanovllm.sampling_params")
    sampling_params_module.SamplingParams = type("SamplingParams", (), {})

    sequence_module = types.ModuleType("nanovllm.engine.sequence")
    sequence_module.Sequence = type("Sequence", (), {})

    scheduler_module = types.ModuleType("nanovllm.engine.scheduler")
    scheduler_module.Scheduler = type("Scheduler", (), {})

    model_runner_module = types.ModuleType("nanovllm.engine.model_runner")
    model_runner_module.ModelRunner = type("ModelRunner", (), {})

    modules = {
        "tqdm": tqdm_module,
        "tqdm.auto": tqdm_auto_module,
        "transformers": transformers_module,
        "torch": torch_module,
        "torch.multiprocessing": torch_mp_module,
        "nanovllm": nanovllm_module,
        "nanovllm.engine": engine_module,
        "nanovllm.config": config_module,
        "nanovllm.sampling_params": sampling_params_module,
        "nanovllm.engine.sequence": sequence_module,
        "nanovllm.engine.scheduler": scheduler_module,
        "nanovllm.engine.model_runner": model_runner_module,
    }

    spec = importlib.util.spec_from_file_location("llm_engine_under_test", llm_engine_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.LLMEngine


LLMEngine = load_llm_engine()


class DummyModelRunner:

    def __init__(self):
        self.calls = []

    def call(self, method_name, *args):
        self.calls.append((method_name, args))


class DummyScheduler:

    def __init__(self, finished=True):
        self.finished = finished
        self.reset_count = 0

    def is_finished(self):
        return self.finished

    def reset(self):
        self.reset_count += 1


class SleepWakeTest(unittest.TestCase):

    def make_engine(self, finished=True, sleeping=False):
        engine = object.__new__(LLMEngine)
        engine.model_runner = DummyModelRunner()
        engine.scheduler = DummyScheduler(finished)
        engine._is_sleeping = sleeping
        return engine

    def test_sleep_releases_kv_cache_when_idle(self):
        engine = self.make_engine()

        engine.sleep()

        self.assertTrue(engine._is_sleeping)
        self.assertEqual(engine.model_runner.calls, [("sleep", (1,))])
        self.assertEqual(engine.scheduler.reset_count, 1)

    def test_sleep_rejects_inflight_requests(self):
        engine = self.make_engine(finished=False)

        with self.assertRaisesRegex(RuntimeError, "in progress"):
            engine.sleep()

        self.assertFalse(engine._is_sleeping)
        self.assertEqual(engine.model_runner.calls, [])
        self.assertEqual(engine.scheduler.reset_count, 0)

    def test_wake_up_reallocates_kv_cache(self):
        engine = self.make_engine(sleeping=True)

        engine.wake_up()

        self.assertFalse(engine._is_sleeping)
        self.assertEqual(engine.model_runner.calls, [("wake_up", (["kv_cache"],))])
        self.assertEqual(engine.scheduler.reset_count, 1)

    def test_unsupported_sleep_level_is_rejected(self):
        engine = self.make_engine()

        with self.assertRaises(NotImplementedError):
            engine.sleep(level=2)

    def test_unsupported_wake_tags_are_rejected(self):
        engine = self.make_engine(sleeping=True)

        with self.assertRaises(NotImplementedError):
            engine.wake_up(tags=["weights"])

    def test_inference_is_rejected_while_sleeping(self):
        engine = self.make_engine(sleeping=True)

        with self.assertRaisesRegex(RuntimeError, "sleeping"):
            engine._assert_awake()


if __name__ == "__main__":
    unittest.main()
