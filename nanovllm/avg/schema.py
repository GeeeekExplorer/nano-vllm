from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


CORE_STATE_KEYS = (
    "affection",
    "trust",
    "stress",
    "reason",
    "explore",
    "danger",
)

PLOT_POINT_KINDS = ("scene", "route", "ending")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_state_values(values: Mapping[str, float] | None, default: float = 0.5) -> dict[str, float]:
    normalized = {key: clamp01(default) for key in CORE_STATE_KEYS}
    if values:
        for key, value in values.items():
            validate_state_key(key)
            normalized[key] = clamp01(value)
    return normalized


def validate_state_key(key: str) -> None:
    if key not in CORE_STATE_KEYS:
        allowed = ", ".join(CORE_STATE_KEYS)
        raise ValueError(f"unknown AVG state key {key!r}; expected one of: {allowed}")


@dataclass(slots=True)
class PlayerEvent:
    action: str
    delta: dict[str, float] = field(default_factory=dict)
    tag: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("PlayerEvent.action must not be empty")
        for key in self.delta:
            validate_state_key(key)


@dataclass(slots=True)
class AVGState:
    values: dict[str, float] = field(default_factory=normalize_state_values)

    def __post_init__(self) -> None:
        self.values = normalize_state_values(self.values)

    @classmethod
    def neutral(cls, value: float = 0.5) -> "AVGState":
        return cls({key: value for key in CORE_STATE_KEYS})

    def apply_delta(self, delta: Mapping[str, float]) -> "AVGState":
        values = dict(self.values)
        for key, value in delta.items():
            validate_state_key(key)
            values[key] = clamp01(values[key] + float(value))
        return AVGState(values)


@dataclass(slots=True)
class AstroInfluence:
    timestamp: datetime
    bias: dict[str, float]
    weights: dict[str, float] | None = None
    seed: int | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise TypeError("AstroInfluence.timestamp must be a datetime")
        for key in self.bias:
            validate_state_key(key)
        if self.weights is not None:
            for key, value in self.weights.items():
                validate_state_key(key)
                if float(value) < 0:
                    raise ValueError("AstroInfluence.weights values must be non-negative")


@dataclass(slots=True)
class PlotPoint:
    id: str
    kind: str
    target: dict[str, float]
    description: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("PlotPoint.id must not be empty")
        if self.kind not in PLOT_POINT_KINDS:
            allowed = ", ".join(PLOT_POINT_KINDS)
            raise ValueError(f"unknown PlotPoint.kind {self.kind!r}; expected one of: {allowed}")
        if not self.description:
            raise ValueError("PlotPoint.description must not be empty")
        self.target = normalize_state_values(self.target)


@dataclass(slots=True)
class AVGGenerationParams:
    max_tokens: int = 256
    temperature: float = 0.7
    style: str = "日式青春悬疑AVG"
    banned_words: list[str] = field(default_factory=list)
    required_words: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("AVGGenerationParams.max_tokens must be positive")
        if self.temperature <= 1e-10:
            raise ValueError("AVGGenerationParams.temperature must be greater than 0")


@dataclass(slots=True)
class AVGGenerationPlan:
    state: AVGState
    plot_point: PlotPoint
    prompt: str
    astro: AstroInfluence | None = None
