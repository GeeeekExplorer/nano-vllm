import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


def load_config():
    # Avoid the GPU imports in nanovllm.__init__ and mock model configuration loading.
    transformers = ModuleType("transformers")
    transformers.AutoConfig = type("AutoConfig", (), {})
    path = Path(__file__).resolve().parents[1] / "nanovllm/config.py"
    spec = importlib.util.spec_from_file_location("_config_block_size_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"transformers": transformers, spec.name: module}):
        spec.loader.exec_module(module)
    return module


config_module = load_config()


class ConfigBlockSizeTest(unittest.TestCase):
    def setUp(self):
        self.model_path = str(Path(__file__).resolve().parents[1])
        patcher = patch.object(config_module, "AutoConfig")
        self.auto_config = patcher.start()
        self.addCleanup(patcher.stop)
        self.auto_config.from_pretrained.return_value = SimpleNamespace(
            max_position_embeddings=8192,
        )

    def test_rejects_non_positive_block_sizes_before_loading_model_config(self):
        for block_size in (0, -256, -512):
            with self.subTest(block_size=block_size):
                with self.assertRaisesRegex(ValueError, "positive multiple of 256"):
                    config_module.Config(self.model_path, kvcache_block_size=block_size)
        self.auto_config.from_pretrained.assert_not_called()

    def test_rejects_non_multiple_block_sizes(self):
        for block_size in (1, 128, 257):
            with self.subTest(block_size=block_size):
                with self.assertRaisesRegex(ValueError, "positive multiple of 256"):
                    config_module.Config(self.model_path, kvcache_block_size=block_size)

    def test_accepts_positive_multiples(self):
        for block_size in (256, 512, 1024):
            with self.subTest(block_size=block_size):
                config = config_module.Config(self.model_path, kvcache_block_size=block_size)
                self.assertEqual(config.kvcache_block_size, block_size)
                self.assertEqual(config.max_model_len, 4096)
                self.assertIs(config.hf_config, self.auto_config.from_pretrained.return_value)
        self.assertEqual(self.auto_config.from_pretrained.call_count, 3)

    def test_default_block_size_is_valid(self):
        self.assertEqual(config_module.Config(self.model_path).kvcache_block_size, 256)


if __name__ == "__main__":
    unittest.main()
