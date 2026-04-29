from datetime import datetime, timezone

from nanovllm.avg import (
    AVGDirector,
    AVGGenerationParams,
    AVGGenerationResult,
    AstroInfluence,
    PlayerEvent,
    PlotPoint,
)


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.calls.append((prompts, sampling_params, use_tqdm))
        return [{"text": "旁白：窗边的光暗了下来。\n她说：\"我一直在等你。\""}]


def make_director():
    return AVGDirector(
        llm=FakeLLM(),
        plot_points=[
            PlotPoint(
                id="calm_scene",
                kind="scene",
                target={
                    "affection": 0.5,
                    "trust": 0.5,
                    "stress": 0.2,
                    "reason": 0.5,
                    "explore": 0.5,
                    "danger": 0.1,
                },
                description="安全的放学路上，角色之间的距离慢慢拉近。",
            ),
            PlotPoint(
                id="danger_scene",
                kind="scene",
                target={
                    "affection": 0.5,
                    "trust": 0.5,
                    "stress": 0.8,
                    "reason": 0.5,
                    "explore": 0.5,
                    "danger": 0.9,
                },
                description="旧校舍的深处传来异响，危险正在靠近。",
            ),
        ],
    )


def test_player_events_update_state_and_clamp_values():
    director = make_director()
    state = director.build_state([
        PlayerEvent(action="helped", delta={"affection": 0.7, "trust": -0.8}),
    ])

    assert state.values["affection"] == 1.0
    assert state.values["trust"] == 0.0


def test_astro_bias_changes_final_state():
    director = make_director()
    astro = AstroInfluence(
        timestamp=datetime.now(timezone.utc),
        bias={"danger": 0.3, "stress": 0.2},
    )

    state = director.build_state([], astro=astro)

    assert state.values["danger"] == 0.8
    assert state.values["stress"] == 0.7


def test_astro_weights_affect_plot_point_selection():
    director = make_director()
    state = director.build_state([
        PlayerEvent(action="nervous", delta={"stress": 0.3, "danger": -0.2}),
    ])
    astro = AstroInfluence(
        timestamp=datetime.now(timezone.utc),
        bias={},
        weights={"danger": 10.0},
    )

    selected = director.select_plot_point(state, astro=astro)

    assert selected.id == "calm_scene"


def test_prompt_contains_scene_state_style_and_constraints():
    director = make_director()
    params = AVGGenerationParams(
        style="日式青春悬疑AVG",
        required_words=["窗边"],
        banned_words=["星座"],
    )
    plan = director.plan([], params=params)

    assert "剧情节点" in plan.prompt
    assert "隐性状态参考" in plan.prompt
    assert "日式青春悬疑AVG" in plan.prompt
    assert "窗边" in plan.prompt
    assert "星座" in plan.prompt


def test_generate_uses_existing_llm_generate_contract():
    director = make_director()
    text = director.generate([], params=AVGGenerationParams(max_tokens=32, temperature=0.6))

    assert "旁白" in text
    assert director.llm.calls[0][1].max_tokens == 32
    assert director.llm.calls[0][1].temperature == 0.6


def test_generate_with_plan_returns_debuggable_result():
    director = make_director()
    result = director.generate_with_plan([], params=AVGGenerationParams(max_tokens=32))

    assert isinstance(result, AVGGenerationResult)
    assert "旁白" in result.text
    assert result.plot_point.id == "calm_scene"
    assert "剧情节点" in result.prompt


def test_astro_seed_is_applied_before_generation():
    director = make_director()
    astro = AstroInfluence(
        timestamp=datetime.now(timezone.utc),
        bias={},
        seed=1234,
    )

    director.generate([], astro=astro)

    try:
        import torch
    except ImportError:
        return
    assert torch.initial_seed() == 1234


def test_mapping_payloads_keep_unknown_integration_fields():
    director = AVGDirector(
        llm=FakeLLM(),
        plot_points=[
            {
                "id": "mapping_scene",
                "kind": "scene",
                "target": {"affection": 0.7, "galaxy_mood": "blue"},
                "description": "用字典配置的剧情节点。",
            }
        ],
    )
    astro = {
        "timestamp": "2026-04-30T12:00:00+08:00",
        "bias": {"affection": 0.2, "orbital_phase": "rising"},
        "weights": {"affection": 2.0, "unknown_weight": 7},
        "extra_payload": {"raw": True},
    }
    result = director.generate_with_plan(
        events=[
            {
                "action": "raw_choice",
                "delta": {"trust": 0.1, "external_score": 99},
                "source": "game_runtime",
            }
        ],
        astro=astro,
    )

    assert result.state.values["affection"] == 0.7
    assert result.state.values["trust"] == 0.6
    assert result.astro.metadata["unknown_bias"] == {"orbital_phase": "rising"}
    assert result.astro.metadata["unknown_weights"] == {"unknown_weight": 7}
    assert result.astro.metadata["extra_fields"]["extra_payload"] == {"raw": True}
