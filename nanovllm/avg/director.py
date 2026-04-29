from __future__ import annotations

from collections.abc import Sequence as SequenceABC

from nanovllm.avg.schema import (
    AVGGenerationParams,
    AVGGenerationPlan,
    AVGState,
    AstroInfluence,
    CORE_STATE_KEYS,
    PlayerEvent,
    PlotPoint,
    clamp01,
)
from nanovllm.sampling_params import SamplingParams


class AVGDirector:
    def __init__(
        self,
        llm,
        plot_points: SequenceABC[PlotPoint],
        base_state: AVGState | None = None,
        default_weights: dict[str, float] | None = None,
    ) -> None:
        if not plot_points:
            raise ValueError("AVGDirector requires at least one PlotPoint")
        self.llm = llm
        self.plot_points = list(plot_points)
        self.base_state = base_state or AVGState.neutral()
        self.default_weights = self._normalize_weights(default_weights)

    def build_state(
        self,
        events: SequenceABC[PlayerEvent],
        astro: AstroInfluence | None = None,
    ) -> AVGState:
        state = self.base_state
        for event in events:
            state = state.apply_delta(event.delta)
        if astro is not None:
            state = state.apply_delta(astro.bias)
        return state

    def select_plot_point(
        self,
        state: AVGState,
        astro: AstroInfluence | None = None,
    ) -> PlotPoint:
        weights = dict(self.default_weights)
        if astro and astro.weights:
            for key, value in astro.weights.items():
                weights[key] = float(value)
        return min(self.plot_points, key=lambda point: self._distance(state, point, weights))

    def plan(
        self,
        events: SequenceABC[PlayerEvent],
        astro: AstroInfluence | None = None,
        params: AVGGenerationParams | None = None,
    ) -> AVGGenerationPlan:
        params = params or AVGGenerationParams()
        state = self.build_state(events, astro)
        plot_point = self.select_plot_point(state, astro)
        prompt = self.build_prompt(state, plot_point, params, astro)
        return AVGGenerationPlan(state=state, plot_point=plot_point, prompt=prompt, astro=astro)

    def generate(
        self,
        events: SequenceABC[PlayerEvent],
        astro: AstroInfluence | None = None,
        params: AVGGenerationParams | None = None,
        use_tqdm: bool = False,
    ) -> str:
        params = params or AVGGenerationParams()
        plan = self.plan(events, astro, params)
        sampling_params = SamplingParams(
            temperature=params.temperature,
            max_tokens=params.max_tokens,
        )
        outputs = self.llm.generate([plan.prompt], sampling_params, use_tqdm=use_tqdm)
        first = outputs[0]
        return first["text"] if isinstance(first, dict) else str(first)

    def build_prompt(
        self,
        state: AVGState,
        plot_point: PlotPoint,
        params: AVGGenerationParams,
        astro: AstroInfluence | None = None,
    ) -> str:
        state_text = ", ".join(f"{key}={state.values[key]:.2f}" for key in CORE_STATE_KEYS)
        lines = [
            "你是一个本地 AVG 剧本生成器。",
            f"写作风格：{params.style}。",
            "输出目标：生成一小段适合 AVG 演出的中文文本，包含场景旁白和角色台词。",
            "输出要求：只输出正文，不解释规则，不展示状态参数，不提及星体算法或随机化机制。",
            f"剧情节点：{plot_point.id} ({plot_point.kind})。",
            f"节点描述：{plot_point.description}",
            f"隐性状态参考：{state_text}",
        ]
        if astro is not None:
            lines.append("隐性氛围：当前外部星体影响已折算进状态，请只体现为情绪、压力、距离感和剧情倾向。")
        if params.required_words:
            lines.append("必须自然包含这些元素：" + "、".join(params.required_words))
        if params.banned_words:
            lines.append("禁止出现这些词语：" + "、".join(params.banned_words))
        lines.append("正文：")
        return "\n".join(lines)

    def _distance(self, state: AVGState, plot_point: PlotPoint, weights: dict[str, float]) -> float:
        return sum(
            weights[key] * (state.values[key] - plot_point.target[key]) ** 2
            for key in CORE_STATE_KEYS
        )

    def _normalize_weights(self, weights: dict[str, float] | None) -> dict[str, float]:
        normalized = {key: 1.0 for key in CORE_STATE_KEYS}
        if weights:
            for key, value in weights.items():
                if key not in normalized:
                    allowed = ", ".join(CORE_STATE_KEYS)
                    raise ValueError(f"unknown AVG state key {key!r}; expected one of: {allowed}")
                value = float(value)
                if value < 0:
                    raise ValueError("weights values must be non-negative")
                normalized[key] = value
        return normalized
