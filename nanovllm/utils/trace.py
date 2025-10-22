from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class TraceConfig:
    enabled: bool = False
    head_items: int = 4
    max_calls_per_key: int = 1
    # 仅在指定层打印（例如 0 表示只打印第 0 层），None 表示所有层
    layer_filter: int | None = 0
    # 只输出指标（PASS/FAIL + 误差），不打印张量形状与样例
    summary_only: bool = True


_TRACE_CONFIG = TraceConfig()
_TRACE_COUNTS: Dict[str, int] = {}


def get_trace_config() -> TraceConfig:
    return _TRACE_CONFIG


def set_trace_config(**kwargs):
    global _TRACE_CONFIG
    for k, v in kwargs.items():
        if hasattr(_TRACE_CONFIG, k):
            setattr(_TRACE_CONFIG, k, v)


def _flatten_head(t: torch.Tensor, n: int) -> list:
    try:
        flat = t.detach().flatten()
        take = min(n, flat.numel())
        return flat[:take].tolist()
    except Exception:
        return []


def should_trace(key: str) -> bool:
    if not _TRACE_CONFIG.enabled:
        return False
    c = _TRACE_COUNTS.get(key, 0)
    if c >= _TRACE_CONFIG.max_calls_per_key:
        return False
    _TRACE_COUNTS[key] = c + 1
    return True


def print_tensor(name: str, t: torch.Tensor):
    try:
        info = f"{name}: shape={tuple(t.shape)} device={t.device} dtype={t.dtype}"
        vals = _flatten_head(t, _TRACE_CONFIG.head_items)
        if vals:
            info += f" head={vals}"
        print(info)
    except Exception as e:
        print(f"{name}: <print failed: {e}>")


def print_line(msg: str):
    print(msg)


def layer_enabled(layer_id: int | None) -> bool:
    if not _TRACE_CONFIG.enabled:
        return False
    if layer_id is None:
        return True
    return _TRACE_CONFIG.layer_filter is None or _TRACE_CONFIG.layer_filter == layer_id
