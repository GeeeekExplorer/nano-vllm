from __future__ import annotations

from nanovllm.avg.constants import CORE_STATE_KEYS
from nanovllm.avg.schema import AVGGenerationParams, AVGState, AstroInfluence, PlotPoint


def build_avg_prompt(
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
