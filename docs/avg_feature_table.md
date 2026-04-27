# AVG 星体影响接口与功能表 v1

## 版本目标

本版本把“星体运动随机化模块”和“本地模型 AVG 文本生成模块”解耦。星体模块负责根据现实星体运动拟合数据、游戏内星体运行和随机扰动产出影响向量；`nano-vllm` 侧只接收结果，用于剧情节点收敛和隐性文本约束。

## 功能表

| 模块 | 功能 | 输入 | 输出 | 负责人/边界 | v1 状态 |
|---|---|---|---|---|---|
| 玩家行为量化 | 将玩家历史操作转为 AVG 状态参数 | 玩家事件列表 | `affection/trust/stress/reason/explore/danger` 等核心状态 | AVG 模块 | 实现 |
| 星体随机化接口 | 接收外部星体模块计算结果 | `AstroInfluence` | 状态偏置、权重、seed、metadata | 只定义接口，不实现星体算法 | 实现 |
| 星体运动计算 | 拟合现实星体运动并映射到游戏内星体作用 | 现实时间戳、星体数据、随机化规则 | `AstroInfluence` | 外部星体模块 | 外部实现 |
| 剧情节点收敛 | 根据玩家状态和星体影响选择目标剧情节点 | AVG 状态、星体 bias/weights、剧情节点表 | `PlotPoint` | AVG 模块 | 实现 |
| Prompt 约束生成 | 将剧情节点、角色状态、隐性氛围约束组织成模型输入 | `PlotPoint`、状态、生成参数 | prompt 文本 | AVG 模块 | 实现 |
| 本地模型生成 | 调用 `nano-vllm` 生成 AVG 文本 | prompt、采样参数 | 旁白+台词 | 复用现有 `LLM.generate` | 实现 |
| Token 级硬约束 | 禁止词、必需词、格式控制等强约束 | 约束参数 | 受限 token 采样 | 预留接口，暂不改 sampler | 预留 |

## 逐功能上手方案

### 玩家行为量化

目标：把玩家之前的选择转成可计算的剧情状态。

上手步骤：

1. 将每个玩家操作记录为 `PlayerEvent`。
2. 为事件填写 `delta`，例如帮助角色提升 `affection` 和 `trust`。
3. 使用 `AVGDirector.build_state(events)` 得到最终状态。
4. 检查状态值是否保持在 `0.0~1.0`。
5. 将常见玩家行为整理成一张配置表，方便剧情策划维护。

验收标准：能把一组玩家历史操作稳定转换成 `affection/trust/stress/reason/explore/danger`。

### 星体随机化接口

目标：让外部星体模块可以安全接入 AVG 系统。

上手步骤：

1. 外部星体模块根据现实时间戳和星体算法计算结果。
2. 将结果封装成 `AstroInfluence`。
3. 至少提供 `timestamp` 和 `bias`。
4. 如果某些状态维度在当前天象下更重要，则提供 `weights`。
5. 如果需要复现同一次结果，则提供 `seed`。

验收标准：AVG 模块只接收 `AstroInfluence`，不依赖外部星体模块内部实现。

### 星体运动计算

目标：明确这一块由外部模块实现，AVG 侧只消费结果。

上手步骤：

1. 外部模块读取或拟合现实星体运动数据。
2. 将现实星体状态映射到游戏内星体运行状态。
3. 加入项目需要的随机扰动规则。
4. 将最终影响转换为 `bias` 和可选 `weights`。
5. 输出 `AstroInfluence` 交给 AVG 生成链路。

验收标准：外部模块可以独立测试，AVG 侧不需要知道任何天文计算细节。

### 剧情节点收敛

目标：根据玩家状态和星体影响选择最合适的剧情落点。

上手步骤：

1. 为每个剧情节点配置一个 `PlotPoint`。
2. 给每个节点填写目标状态 `target`。
3. 调用 `AVGDirector.select_plot_point(state, astro)`。
4. 对比不同 `bias` 和 `weights` 下的节点选择结果。
5. 调整剧情节点目标值，让选择结果符合策划预期。

验收标准：同一组玩家行为在不同星体影响下，可以合理收敛到不同剧情节点。

### Prompt 约束生成

目标：把计算结果转成模型能理解的文本约束。

上手步骤：

1. 调用 `AVGDirector.plan(events, astro, params)`。
2. 查看返回的 `prompt`。
3. 确认 prompt 中包含剧情节点、节点描述、隐性状态和写作风格。
4. 配置 `required_words` 和 `banned_words`，观察 prompt 约束变化。
5. 根据实际输出效果迭代 prompt 话术。

验收标准：prompt 不暴露内部参数给玩家，但能稳定引导模型生成目标风格文本。

### 本地模型生成

目标：完成从剧情约束到 AVG 文本的最后一步。

上手步骤：

1. 初始化本地 `LLM`。
2. 初始化 `AVGDirector` 并传入剧情节点。
3. 调用 `director.generate(events, astro, params)`。
4. 检查输出是否包含旁白和角色台词。
5. 根据生成质量调整 `temperature`、`max_tokens` 和剧情节点描述。

验收标准：可以生成一段可放入 AVG 演出流程的中文旁白+台词。

### Token 级硬约束

目标：明确当前可用能力和后续增强方式。

上手步骤：

1. 在 `AVGGenerationParams` 中填写 `banned_words` 和 `required_words`。
2. 通过 `plan()` 检查这些约束是否进入 prompt。
3. 用真实模型生成文本，观察约束遵守程度。
4. 如果必须强制禁止某些词，后续扩展 sampler 或 decode 后处理。
5. 如果必须稳定输出 JSON，再追加格式约束模块。

验收标准：v1 能表达约束需求，强制执行能力作为后续开发项单独评估。

## 接口边界

`AstroInfluence` 是星体模块和 AVG 模块之间的稳定数据契约。AVG 模块接收现实时间戳、状态偏置、权重、seed 和 metadata，但不计算星体位置，也不解释天文含义。

```python
AstroInfluence(
    timestamp=timestamp,
    bias={"danger": 0.12, "trust": -0.04},
    weights={"danger": 1.4},
    seed=42,
    metadata={"source": "external_astro_randomizer"},
)
```

## 交付说明

- v1 重点是稳定接口、剧情收敛和可展示的本地模型生成流程。
- 星体影响默认是隐性的，只改变剧情倾向、氛围和节点选择，不要求模型直接写出星体机制。
- `banned_words` 和 `required_words` 已进入生成参数，第一版通过 prompt 约束生效，后续可升级为 token 级硬约束。
