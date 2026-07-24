"""Regression tests for weightless RMSNorm / FlashNorm support (GitHub #220).

FlashNorm folds RMSNorm scales into the next Linear so decoder-layer norms
have no learnable weight. nano-vLLM previously always allocated ones weights
and always multiplied, which:
  1. fails to load HF checkpoints that omit `*.input_layernorm.weight`, and
  2. wastes a multiply even when the folded weight is mathematically ones.

Run:
  python tests/test_flash_norm_rmsnorm.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_layernorm():
    """Load RMSNorm without importing the full nanovllm package."""
    path = ROOT / "nanovllm" / "layers" / "layernorm.py"
    spec = importlib.util.spec_from_file_location("nanovllm.layers.layernorm", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.RMSNorm


RMSNorm = _load_layernorm()


def _pure_rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference: normalize only, no scale multiply."""
    orig = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(orig)


def test_default_rmsnorm_still_has_learnable_weight():
    """Default path must stay compatible with ordinary Qwen3 checkpoints."""
    m = RMSNorm(8)
    assert m.weight is not None
    assert isinstance(m.weight, torch.nn.Parameter)
    assert list(m.parameters())  # non-empty
    assert torch.allclose(m.weight, torch.ones(8))


def test_weightless_rmsnorm_has_no_weight_parameter():
    """Reproduce #220 requirement: optional RMSNorm without a weight tensor."""
    m = RMSNorm(8, has_weight=False)
    assert m.weight is None
    assert list(m.named_parameters()) == []
    # state_dict must not claim a missing weight key
    assert "weight" not in m.state_dict()


def test_weightless_forward_matches_pure_rms():
    """has_weight=False must equal normalize-only (no ones multiply)."""
    torch.manual_seed(0)
    m = RMSNorm(16, has_weight=False)
    x = torch.randn(4, 16, dtype=torch.float32)
    y = m(x)
    assert torch.allclose(y, _pure_rms(x), atol=1e-5, rtol=1e-5)


def test_weightless_add_rms_forward():
    # Use fp16 so residual = x.to(orig_dtype) materializes a copy before
    # the in-place rsqrt (fp32 path aliases residual with x; pre-existing).
    torch.manual_seed(1)
    m = RMSNorm(16, has_weight=False)
    x = torch.randn(3, 16, dtype=torch.float16)
    residual = torch.randn(3, 16, dtype=torch.float16)
    expected_residual = (x.float() + residual.float()).to(torch.float16)
    expected_out = _pure_rms(expected_residual)
    out, new_residual = m(x.clone(), residual.clone())
    assert torch.allclose(new_residual.float(), expected_residual.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(out.float(), expected_out.float(), atol=2e-3, rtol=2e-3)


def test_weighted_ones_equals_weightless():
    """Identity check: ones scale == weightless path (FlashNorm math)."""
    torch.manual_seed(2)
    weighted = RMSNorm(12, has_weight=True)
    weightless = RMSNorm(12, has_weight=False)
    x = torch.randn(5, 12)
    assert torch.allclose(weighted(x.clone()), weightless(x.clone()), atol=1e-5, rtol=1e-5)


def test_weighted_nontrivial_scale_differs_from_weightless():
    """Sanity: a non-ones weight must change the output vs weightless."""
    torch.manual_seed(3)
    weighted = RMSNorm(12, has_weight=True)
    with torch.no_grad():
        weighted.weight.copy_(torch.linspace(0.5, 1.5, 12))
    weightless = RMSNorm(12, has_weight=False)
    x = torch.randn(5, 12)
    assert not torch.allclose(
        weighted(x.clone()), weightless(x.clone()), atol=1e-4
    )


def test_flash_norm_config_flag_wiring():
    """Decoder layers should map config.flash_norm -> has_weight=False.

    Mirrors Qwen3DecoderLayer construction without needing torch.distributed.
    """
    def layer_has_weight(config) -> bool:
        return not getattr(config, "flash_norm", False)

    normal = SimpleNamespace(flash_norm=False)
    flash = SimpleNamespace(flash_norm=True)
    missing = SimpleNamespace()  # ordinary HF configs omit the field

    assert layer_has_weight(normal) is True
    assert layer_has_weight(flash) is False
    assert layer_has_weight(missing) is True

    m_flash = RMSNorm(8, has_weight=layer_has_weight(flash))
    m_normal = RMSNorm(8, has_weight=layer_has_weight(missing))
    assert m_flash.weight is None
    assert m_normal.weight is not None


def test_repro_load_failure_without_weightless_support():
    """Document the checkpoint-load failure that #220 fixes.

    FlashNorm HF repos omit decoder-layer `*.layernorm.weight` tensors.
    A module that still registers those Parameters cannot load such a
    state_dict with strict=True (extra/missing keys).
    """
    # Simulated flashified layer: only the linear exists, no norm.weight
    class FlashLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = RMSNorm(4, has_weight=False)
            self.proj = torch.nn.Linear(4, 4, bias=False)

    class LegacyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = RMSNorm(4, has_weight=True)
            self.proj = torch.nn.Linear(4, 4, bias=False)

    ckpt = {
        "proj.weight": torch.eye(4),
        # intentionally no input_layernorm.weight
    }

    flash = FlashLayer()
    missing, unexpected = flash.load_state_dict(ckpt, strict=False)
    assert missing == []
    assert unexpected == []

    legacy = LegacyLayer()
    missing, unexpected = legacy.load_state_dict(ckpt, strict=False)
    # Legacy still expects the folded-away norm weight -> load incomplete
    assert any("input_layernorm.weight" in k for k in missing)


if __name__ == "__main__":
    tests = [
        test_default_rmsnorm_still_has_learnable_weight,
        test_weightless_rmsnorm_has_no_weight_parameter,
        test_weightless_forward_matches_pure_rms,
        test_weightless_add_rms_forward,
        test_weighted_ones_equals_weightless,
        test_weighted_nontrivial_scale_differs_from_weightless,
        test_flash_norm_config_flag_wiring,
        test_repro_load_failure_without_weightless_support,
    ]
    for fn in tests:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
