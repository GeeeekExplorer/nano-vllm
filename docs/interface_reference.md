# AVG 与 nano-vLLM 接口参考

## 接口分层

| 层级 | 接口 | 作用 |
|---|---|---|
| 模型层 | `LLM`、`SamplingParams` | 加载本地模型并生成文本 |
| AVG 层 | `AVGDirector`、`PlayerEvent`、`AVGState`、`PlotPoint` | 管理玩家状态、剧情节点和生成流程 |
| 星体接入层 | `AstroInfluence` | 接收外部星体随机化模块输出 |
| 调试层 | `AVGGenerationPlan`、`AVGGenerationResult` | 记录生成依据和最终结果 |

## 常量

| 常量 | 数据类型 | 说明 |
|---|---|---|
| `CORE_STATE_KEYS` | `tuple[str, ...]` | AVG 核心状态字段：`affection/trust/stress/reason/explore/danger` |
| `PLOT_POINT_KINDS` | `tuple[str, ...]` | 剧情节点类型：`scene/route/ending` |

## 模型层接口

### `LLM`

用途：本地模型推理入口。

```python
llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(prompts, sampling_params)
```

| 参数 | 数据类型 | 说明 |
|---|---|---|
| `model_path` | `str` | 本地模型目录 |
| `prompts` | `list[str] | list[list[int]]` | prompt 文本或 token id |
| `sampling_params` | `SamplingParams | list[SamplingParams]` | 采样参数 |

返回：`list[dict]`，每项包含 `text` 和 `token_ids`。

### `SamplingParams`

| 字段 | 数据类型 | 说明 |
|---|---|---|
| `temperature` | `float` | 随机性，必须大于 0 |
| `max_tokens` | `int` | 最大生成 token 数 |
| `ignore_eos` | `bool` | 是否忽略结束符 |

## AVG 数据接口

### `PlayerEvent`

| 字段 | 数据类型 | 说明 |
|---|---|---|
| `action` | `str` | 玩家行为名 |
| `delta` | `dict[str, float]` | 对核心状态的增量 |
| `tag` | `str | None` | 可选分类标签 |
| `metadata` | `dict[str, Any] | None` | 外部扩展信息 |

支持 `PlayerEvent.from_mapping(payload)`，未知字段会进入 `metadata`。

### `AVGState`

| 字段 | 数据类型 | 说明 |
|---|---|---|
| `values` | `dict[str, float]` | 六个核心状态值，范围为 `0.0~1.0` |

默认中性状态为 `0.5`。状态更新会自动 clamp 到 `0.0~1.0`。

### `AstroInfluence`

| 字段 | 数据类型 | 说明 |
|---|---|---|
| `timestamp` | `datetime` | 现实时间戳 |
| `bias` | `dict[str, float]` | 对 AVG 状态的偏置 |
| `weights` | `dict[str, float] | None` | 对剧情收敛权重的调整 |
| `seed` | `int | None` | 生成前设置 PyTorch 随机种子 |
| `metadata` | `dict[str, Any] | None` | 外部星体模块调试信息 |

支持 `AstroInfluence.from_mapping(payload)`。`timestamp` 可传 `datetime`、ISO 字符串或 Unix 时间戳。

### `PlotPoint`

| 字段 | 数据类型 | 说明 |
|---|---|---|
| `id` | `str` | 剧情节点 ID |
| `kind` | `str` | `scene`、`route` 或 `ending` |
| `target` | `dict[str, float]` | 节点目标状态 |
| `description` | `str` | 节点剧情描述 |
| `metadata` | `dict[str, Any] | None` | 策划或运行时扩展信息 |

支持 `PlotPoint.from_mapping(payload)`，未知 target 字段会进入 `metadata["unknown_target"]`。

### `AVGGenerationParams`

| 字段 | 数据类型 | 说明 |
|---|---|---|
| `max_tokens` | `int` | 最大生成 token 数 |
| `temperature` | `float` | 生成随机性 |
| `style` | `str` | 写作风格 |
| `banned_words` | `list[str]` | 禁止词，v1 通过 prompt 约束 |
| `required_words` | `list[str]` | 必须出现元素，v1 通过 prompt 约束 |

## 编排接口

### `AVGDirector`

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `build_state(events, astro)` | 玩家事件、星体影响 | `AVGState` | 计算最终状态 |
| `select_plot_point(state, astro)` | 状态、星体影响 | `PlotPoint` | 选择剧情节点 |
| `build_prompt(state, plot_point, params, astro)` | 状态、节点、生成参数、星体影响 | `str` | 构造 prompt |
| `plan(events, astro, params)` | 玩家事件、星体影响、生成参数 | `AVGGenerationPlan` | 只生成计划，不调用模型 |
| `generate(events, astro, params)` | 玩家事件、星体影响、生成参数 | `str` | 返回最终文本 |
| `generate_with_plan(events, astro, params)` | 玩家事件、星体影响、生成参数 | `AVGGenerationResult` | 返回文本和调试信息 |

### `AVGGenerationPlan`

| 字段 | 数据类型 | 说明 |
|---|---|---|
| `state` | `AVGState` | 最终状态 |
| `plot_point` | `PlotPoint` | 选中的剧情节点 |
| `prompt` | `str` | 即将送入模型的 prompt |
| `astro` | `AstroInfluence | None` | 本次使用的星体影响 |

### `AVGGenerationResult`

| 字段 | 数据类型 | 说明 |
|---|---|---|
| `text` | `str` | 最终生成文本 |
| `state` | `AVGState` | 最终状态 |
| `plot_point` | `PlotPoint` | 选中的剧情节点 |
| `prompt` | `str` | 实际使用的 prompt |
| `astro` | `AstroInfluence | None` | 本次使用的星体影响 |

## 宽松输入规则

对接初期允许外部系统传普通字典。规则如下：

| 情况 | 处理方式 |
|---|---|
| 字段属于 `CORE_STATE_KEYS` | 转成 `float` 并进入核心状态计算 |
| 字段不属于 `CORE_STATE_KEYS` | 保存在 `metadata` |
| 顶层未知字段 | 保存在 `metadata["extra_fields"]` |
| 未知 `delta` 字段 | 保存在 `metadata["unknown_delta"]` |
| 未知 `bias` 字段 | 保存在 `metadata["unknown_bias"]` |
| 未知 `weights` 字段 | 保存在 `metadata["unknown_weights"]` |
| 未知 `target` 字段 | 保存在 `metadata["unknown_target"]` |

这套规则支持先完成系统联调，再逐步沉淀正式字段清单。
