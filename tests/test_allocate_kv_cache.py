import importlib.util
import json
import sys
import types
from types import SimpleNamespace
import torch
import torch.nn as nn
from transformers import AutoConfig


def _import_model_runner():
    orig_compile = torch.compile
    if "triton" not in sys.modules and importlib.util.find_spec("triton") is None:
        triton_mod = types.ModuleType("triton")
        triton_mod.__path__ = []
        triton_mod.jit = lambda fn=None, **kw: fn if fn else (lambda f: f)

        tl = types.ModuleType("triton.language")
        tl.constexpr = int
        triton_mod.language = tl

        sys.modules["triton"] = triton_mod
        sys.modules["triton.language"] = tl
        torch.compile = lambda fn=None, **kw: fn if fn else (lambda f: f)

    if "flash_attn" not in sys.modules and importlib.util.find_spec("flash_attn") is None:
        fa = types.ModuleType("flash_attn")
        fa.flash_attn_varlen_func = lambda *a, **kw: None
        fa.flash_attn_with_kvcache = lambda *a, **kw: None
        sys.modules["flash_attn"] = fa

    try:
        from nanovllm.engine.model_runner import ModelRunner
    finally:
        torch.compile = orig_compile
    return ModelRunner


class _DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])


class _DummyModel(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_DummyAttention() for _ in range(num_layers)])


def _make_runner(hf_config, block_size: int = 256, world_size: int = 1):
    ModelRunner = _import_model_runner()
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = block_size
    runner.world_size = world_size
    runner.config = SimpleNamespace(
        hf_config=hf_config,
        gpu_memory_utilization=0.9,
        num_kvcache_blocks=-1,
    )
    runner.model = _DummyModel(hf_config.num_hidden_layers)
    return runner


def _patch_cuda_stats(monkeypatch):
    # 32 MiB simulated GPU memory keeps test tensor allocations small on CPU.
    mem = 32 * 1024**2
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (mem, mem))
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {"allocated_bytes.all.peak": 0, "allocated_bytes.all.current": 0},
    )


def test_allocate_kv_cache_handles_none_head_dim(monkeypatch):
    _patch_cuda_stats(monkeypatch)
    hf_config = SimpleNamespace(
        num_key_value_heads=2,
        num_attention_heads=4,
        hidden_size=256,
        num_hidden_layers=2,
        head_dim=None,
        dtype=torch.bfloat16,
    )
    runner = _make_runner(hf_config)
    runner.allocate_kv_cache()

    expected_head_dim = 256 // 4
    assert runner.config.num_kvcache_blocks > 0
    assert runner.kv_cache.shape == (
        2,
        hf_config.num_hidden_layers,
        runner.config.num_kvcache_blocks,
        256,
        hf_config.num_key_value_heads,
        expected_head_dim,
    )


def test_allocate_kv_cache_handles_missing_head_dim(monkeypatch):
    _patch_cuda_stats(monkeypatch)
    hf_config = SimpleNamespace(
        num_key_value_heads=2,
        num_attention_heads=4,
        hidden_size=256,
        num_hidden_layers=2,
        dtype=torch.bfloat16,
    )
    runner = _make_runner(hf_config)
    runner.allocate_kv_cache()

    assert runner.kv_cache.shape[-1] == 64


def test_allocate_kv_cache_uses_explicit_head_dim(monkeypatch):
    _patch_cuda_stats(monkeypatch)
    hf_config = SimpleNamespace(
        num_key_value_heads=8,
        num_attention_heads=16,
        hidden_size=1024,
        num_hidden_layers=2,
        head_dim=128,
        dtype=torch.bfloat16,
    )
    runner = _make_runner(hf_config)
    runner.allocate_kv_cache()

    assert runner.kv_cache.shape[-1] == 128


def test_allocate_kv_cache_with_hf_config_null_head_dim(monkeypatch, tmp_path):
    _patch_cuda_stats(monkeypatch)
    (tmp_path / "config.json").write_text(
        json.dumps({
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "head_dim": None,
            "torch_dtype": "bfloat16",
            "vocab_size": 128,
        }),
        encoding="utf-8",
    )
    hf_config = AutoConfig.from_pretrained(tmp_path)
    runner = _make_runner(hf_config)
    runner.allocate_kv_cache()

    assert runner.kv_cache.shape[-1] == 64
