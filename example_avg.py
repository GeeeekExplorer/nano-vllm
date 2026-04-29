import os
from datetime import datetime, timezone

from nanovllm import LLM
from nanovllm.avg import (
    AVGDirector,
    AVGGenerationParams,
    AstroInfluence,
    PlayerEvent,
    PlotPoint,
)


def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)

    director = AVGDirector(
        llm=llm,
        plot_points=[
            PlotPoint(
                id="library_confession",
                kind="scene",
                target={
                    "affection": 0.8,
                    "trust": 0.7,
                    "stress": 0.3,
                    "reason": 0.4,
                    "explore": 0.6,
                    "danger": 0.2,
                },
                description="黄昏的图书馆里，角色终于愿意说出隐藏的心事。",
            ),
            PlotPoint(
                id="forbidden_corridor",
                kind="scene",
                target={
                    "affection": 0.4,
                    "trust": 0.3,
                    "stress": 0.8,
                    "reason": 0.6,
                    "explore": 0.9,
                    "danger": 0.8,
                },
                description="深夜的旧校舍走廊尽头，某个被封存的真相开始松动。",
            ),
        ],
    )

    events = [
        PlayerEvent(action="helped_heroine", delta={"affection": 0.2, "trust": 0.1}),
        PlayerEvent(action="opened_forbidden_door", delta={"danger": 0.2, "explore": 0.2}),
    ]
    astro = AstroInfluence(
        timestamp=datetime.now(timezone.utc),
        bias={"stress": 0.08, "danger": 0.06},
        weights={"danger": 1.3},
        seed=42,
        metadata={"source": "external_astro_randomizer"},
    )
    params = AVGGenerationParams(
        style="日式青春悬疑AVG",
        max_tokens=220,
        temperature=0.7,
        required_words=["窗边", "沉默"],
        banned_words=["星座", "算法"],
    )

    print(director.generate(events=events, astro=astro, params=params))


if __name__ == "__main__":
    main()
