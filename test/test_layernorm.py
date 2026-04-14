import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location(
    "layernorm",
    Path(__file__).parents[1] / "nanovllm" / "layers" / "layernorm.py",
)
layernorm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layernorm)
RMSNorm = layernorm.RMSNorm


def test_rms_norm_does_not_modify_fp32_input():
    norm = RMSNorm(4)
    x = torch.randn(2, 4, dtype=torch.float32)
    expected = x.clone()

    norm(x)

    torch.testing.assert_close(x, expected)
