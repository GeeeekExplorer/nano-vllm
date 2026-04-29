from __future__ import annotations

from collections.abc import Mapping, Sequence as SequenceABC

from nanovllm.avg.constants import CORE_STATE_KEYS
from nanovllm.avg.prompt import build_avg_prompt
from nanovllm.avg.schema import (
    AVGGenerationParams,
    AVGGenerationPlan,
    AVGGenerationResult,
    AVGState,
    AstroInfluence,
    PlayerEvent,
    PlotPoint,
)
from nanovllm.sampling_params import SamplingParams


class AVGDirector:
    def __init__(
        self,
        llm,
        plot_points: SequenceABC[PlotPoint | Mapping],
        base_state: AVGState | None = None,
        default_weights: dict[str, float] | None = None,
    ) -> None:
        if not plot_points:
            raise ValueError("AVGDirector requires at least one PlotPoint")
        self.llm = llm
        self.plot_points = [self._coerce_plot_point(point) for point in plot_points]
        self.base_state = base_state or AVGState.neutral()
        self.default_weights = self._normalize_weights(default_weights)

    def build_state(
        self,
        events: SequenceABC[PlayerEvent | Mapping],
        astro: AstroInfluence | Mapping | None = None,
    ) -> AVGState:
        astro = self._coerce_astro(astro)
        state = self.base_state
        for event in events:
            event = self._coerce_event(event)
            state = state.apply_delta(event.delta)
        if astro is not None:
            state = state.apply_delta(astro.bias)
        return state

    def select_plot_point(
        self,
        state: AVGState,
        astro: AstroInfluence | Mapping | None = None,
    ) -> PlotPoint:
        astro = self._coerce_astro(astro)
        weights = dict(self.default_weights)
        if astro and astro.weights:
            for key, value in astro.weights.items():
                weights[key] = float(value)
        return min(self.plot_points, key=lambda point: self._distance(state, point, weights))

    def plan(
        self,
        events: SequenceABC[PlayerEvent | Mapping],
        astro: AstroInfluence | Mapping | None = None,
        params: AVGGenerationParams | None = None,
    ) -> AVGGenerationPlan:
        params = params or AVGGenerationParams()
        astro = self._coerce_astro(astro)
        state = self.build_state(events, astro)
        plot_point = self.select_plot_point(state, astro)
        prompt = self.build_prompt(state, plot_point, params, astro)
        return AVGGenerationPlan(state=state, plot_point=plot_point, prompt=prompt, astro=astro)

    def generate(
        self,
        events: SequenceABC[PlayerEvent | Mapping],
        astro: AstroInfluence | Mapping | None = None,
        params: AVGGenerationParams | None = None,
        use_tqdm: bool = False,
    ) -> str:
        return self.generate_with_plan(events, astro, params, use_tqdm).text

    def generate_with_plan(
        self,
        events: SequenceABC[PlayerEvent | Mapping],
        astro: AstroInfluence | Mapping | None = None,
        params: AVGGenerationParams | None = None,
        use_tqdm: bool = False,
    ) -> AVGGenerationResult:
        params = params or AVGGenerationParams()
        astro = self._coerce_astro(astro)
        plan = self.plan(events, astro, params)
        self._apply_seed(astro)
        sampling_params = SamplingParams(
            temperature=params.temperature,
            max_tokens=params.max_tokens,
        )
        outputs = self.llm.generate([plan.prompt], sampling_params, use_tqdm=use_tqdm)
        first = outputs[0]
        text = first["text"] if isinstance(first, dict) else str(first)
        return AVGGenerationResult(
            text=text,
            state=plan.state,
            plot_point=plan.plot_point,
            prompt=plan.prompt,
            astro=plan.astro,
        )

    def build_prompt(
        self,
        state: AVGState,
        plot_point: PlotPoint,
        params: AVGGenerationParams,
        astro: AstroInfluence | None = None,
    ) -> str:
        return build_avg_prompt(state, plot_point, params, astro)

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

    def _apply_seed(self, astro: AstroInfluence | None) -> None:
        if astro is None or astro.seed is None:
            return
        import torch

        torch.manual_seed(astro.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(astro.seed)

    def _coerce_event(self, event: PlayerEvent | Mapping) -> PlayerEvent:
        if isinstance(event, PlayerEvent):
            return event
        if isinstance(event, Mapping):
            return PlayerEvent.from_mapping(event)
        raise TypeError("events must contain PlayerEvent objects or mapping payloads")

    def _coerce_astro(self, astro: AstroInfluence | Mapping | None) -> AstroInfluence | None:
        if astro is None or isinstance(astro, AstroInfluence):
            return astro
        if isinstance(astro, Mapping):
            return AstroInfluence.from_mapping(astro)
        raise TypeError("astro must be AstroInfluence, mapping payload, or None")

    def _coerce_plot_point(self, plot_point: PlotPoint | Mapping) -> PlotPoint:
        if isinstance(plot_point, PlotPoint):
            return plot_point
        if isinstance(plot_point, Mapping):
            return PlotPoint.from_mapping(plot_point)
        raise TypeError("plot_points must contain PlotPoint objects or mapping payloads")
