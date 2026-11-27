import pickle
import unittest
from contextlib import suppress
from multiprocessing.shared_memory import SharedMemory
from tempfile import TemporaryDirectory
from unittest.mock import DEFAULT, patch

from transformers import Qwen3Config

from nanovllm.config import Config
from nanovllm.engine import model_runner
from nanovllm.engine.model_runner import ModelRunner


class TestEngineIPC(unittest.TestCase):
    def test_separate_engines(self):
        with TemporaryDirectory() as model:
            Qwen3Config().save_pretrained(model)
            a = Config(model, tensor_parallel_size=2, enforce_eager=True)
            b = Config(model, tensor_parallel_size=2, enforce_eager=True, master_port=2334)
        self.assertEqual(a.master_port, 2333)
        self.assertNotEqual(a.shm_name, b.shm_name)

        def unlink(shm):
            with suppress(FileNotFoundError):
                shm.unlink()

        def open_segment(name, **kwargs):
            self.assertIn(name, (a.shm_name, b.shm_name))
            shm = SharedMemory(name, **kwargs)
            self.addCleanup(shm.close)
            if kwargs.get("create"):
                self.addCleanup(unlink, shm)
            return shm

        with patch.multiple(model_runner, torch=DEFAULT, dist=DEFAULT, SharedMemory=open_segment,
                            Qwen3ForCausalLM=DEFAULT, load_model=DEFAULT, Sampler=DEFAULT) as mocks, \
             patch.multiple(ModelRunner, warmup_model=DEFAULT, allocate_kv_cache=DEFAULT, loop=DEFAULT):
            pairs = []
            for config in (a, b):
                owner = ModelRunner(config, 0, [])
                worker = ModelRunner(pickle.loads(pickle.dumps(config)), 1, None)
                mocks["dist"].init_process_group.assert_any_call(
                    "nccl", f"tcp://localhost:{config.master_port}", world_size=2, rank=0)
                mocks["dist"].init_process_group.assert_called_with(
                    "nccl", f"tcp://localhost:{config.master_port}", world_size=2, rank=1)
                self.assertEqual(owner.shm.name, config.shm_name)
                self.assertEqual(worker.shm.name, config.shm_name)
                pairs.append((owner, worker))

            for owner, worker in pairs:
                worker.exit()
                open_segment(owner.shm.name).close()  # still named until rank 0 exits
                owner.exit()
                with self.assertRaises(FileNotFoundError):
                    open_segment(owner.shm.name)
