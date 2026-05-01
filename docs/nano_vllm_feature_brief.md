# nano-vLLM 本地模型推理功能说明

## 一句话说明

`nano-vLLM` 是一个轻量级本地大语言模型推理引擎，用来把已经下载到本地的模型运行起来，并为游戏系统提供高效、可控、可私有化部署的文本生成能力。

在本项目中，它承担“本地模型生成底座”的角色：上层 AVG 系统负责剧情逻辑、玩家状态和星体影响，`nano-vLLM` 负责把这些约束转化为实际可读的剧情文本。

## 面向上级的价值说明

- 本地运行：模型和数据都在本机或自有服务器内运行，不依赖外部在线 API，利于隐私、成本和稳定性控制。
- 快速生成：项目使用 KV Cache、批量调度、CUDA Graph 等推理优化手段，目标是提升多请求文本生成吞吐。
- 可控扩展：推理核心和游戏业务逻辑分层，上层可以接入玩家行为、剧情节点、星体影响等系统，而不需要反复改底层模型执行逻辑。
- 代码轻量：实现规模较小，适合内部理解、改造和二次开发。
- 适配 AVG：可以作为游戏内“旁白、台词、分支剧情文本”的生成引擎，配合外部规则系统实现约束生成。

## 功能表

| 模块 | 功能 | 输入 | 输出 | 项目价值 | 当前状态 |
|---|---|---|---|---|---|
| 本地模型加载 | 从本地模型目录读取模型配置、权重和分词器 | 模型路径 | 可执行的本地语言模型 | 支持离线部署，减少外部依赖 | 已实现 |
| 文本生成接口 | 提供类似 vLLM 的 `LLM.generate` 调用方式 | prompt、采样参数 | 生成文本和 token id | 上层系统可直接调用 | 已实现 |
| 采样参数控制 | 控制生成长度、温度和 EOS 行为 | `SamplingParams` | 不同随机性和长度的文本 | 支持剧情文本多样性 | 已实现 |
| 请求调度 | 管理等待中和运行中的生成请求 | 多条生成请求 | 批量执行计划 | 提升多请求处理效率 | 已实现 |
| Prefill 阶段 | 处理输入 prompt，建立上下文缓存 | 输入 token | 首次上下文状态 | 提升后续生成效率 | 已实现 |
| Decode 阶段 | 逐 token 生成后续文本 | 上下文缓存、上一个 token | 新 token | 支持连续文本生成 | 已实现 |
| KV Cache 管理 | 缓存历史上下文的 key/value 数据 | token block | 可复用上下文缓存 | 降低重复计算成本 | 已实现 |
| Prefix Cache | 对相同前缀复用缓存 | 相同或相似 prompt 前缀 | 复用后的缓存命中 | 适合多角色、多分支共用背景设定 | 已实现 |
| Tensor Parallel | 将模型计算拆分到多张 GPU | 多 GPU 配置 | 并行推理结果 | 为更大模型预留扩展能力 | 已实现 |
| CUDA Graph | 捕获固定形状推理图以减少运行开销 | decode batch | 更快的执行路径 | 提升稳定吞吐 | 已实现 |
| Qwen3 模型结构 | 内置 Qwen3 Causal LM 推理结构 | Qwen3 模型权重 | logits/token 生成能力 | 直接支持当前计划使用的小模型 | 已实现 |
| AVG 生成适配 | 接收剧情节点、状态和星体影响约束生成文本 | AVG prompt | 旁白+台词 | 支撑游戏文本生成落地 | 已新增 |
| AVG 调试结果 | 返回生成文本、剧情节点、最终状态和 prompt | 玩家事件、星体影响、生成参数 | `AVGGenerationResult` | 便于游戏侧接入、复盘和调参 | 已新增 |
| Token 级硬约束 | 禁止词、必需词、格式控制等强约束 | 约束参数 | 受限生成结果 | 后续提升文本安全性和格式稳定性 | 预留 |

## 逐功能上手方案

### 本地模型加载

目标：让团队理解模型不是在线调用，而是从本地目录加载配置、分词器和权重。

上手步骤：

1. 准备 Qwen3 模型目录，例如 `~/huggingface/Qwen3-0.6B/`。
2. 确认目录内包含 `config.json`、分词器文件和 `.safetensors` 权重。
3. 运行 `example.py`，观察模型是否能完成初始化。
4. 查看 `nanovllm/config.py`，理解模型路径、最大长度、GPU 显存利用率等配置项。
5. 查看 `nanovllm/utils/loader.py`，理解权重是如何从 safetensors 文件加载到模型结构中的。

验收标准：能说明“模型文件在本地，启动时加载到 GPU，后续生成不依赖外部 API”。

### 文本生成接口

目标：掌握上层系统如何调用本地模型生成文本。

上手步骤：

1. 阅读 `example.py` 中的 `LLM(...)` 和 `llm.generate(...)` 调用。
2. 用一条简单 prompt 调用生成，例如“请写一句 AVG 旁白”。
3. 查看返回值中的 `text` 和 `token_ids`。
4. 修改 prompt 内容，观察输出文本如何变化。
5. 对照 `nanovllm/engine/llm_engine.py`，理解 `generate` 会把请求加入调度器并循环执行。

验收标准：能写出一段最小调用代码，并解释 prompt 输入和 text 输出的关系。

### 采样参数控制

目标：理解生成长度和随机性如何影响剧情文本。

上手步骤：

1. 查看 `nanovllm/sampling_params.py`，确认当前支持 `temperature`、`max_tokens`、`ignore_eos`。
2. 分别用 `temperature=0.3` 和 `temperature=0.9` 生成同一 prompt，对比文本稳定性。
3. 分别设置 `max_tokens=32` 和 `max_tokens=256`，观察输出长度差异。
4. 在 AVG 场景中将短输出用于一句台词，长输出用于一段旁白+台词。

验收标准：能说明“温度控制随机性，最大 token 数控制输出长度”。

### 请求调度

目标：理解多个文本请求如何被合并处理，提高吞吐。

上手步骤：

1. 在 `example.py` 中传入多个 prompts。
2. 观察进度条中 Prefill 和 Decode 的 token/s 指标。
3. 阅读 `nanovllm/engine/scheduler.py`，理解 `waiting` 和 `running` 两个队列。
4. 重点看 `schedule()`，理解它如何优先处理 prefill，再处理 decode。
5. 调整 prompt 数量，观察批量生成对整体耗时的影响。

验收标准：能说明“调度器把多个请求组织成批次，减少单条请求逐个执行的浪费”。

### Prefill 阶段

目标：理解模型第一次读取 prompt 时发生了什么。

上手步骤：

1. 查看 `nanovllm/engine/model_runner.py` 的 `prepare_prefill()`。
2. 关注 `input_ids`、`positions`、`cu_seqlens_q`、`slot_mapping` 的构造。
3. 用较长 prompt 运行生成，观察 prefill 阶段吞吐。
4. 理解 prefill 的作用是把输入上下文写入 KV Cache，为后续 decode 做准备。

验收标准：能说明“prefill 是处理已有输入，不是逐字生成新文本”。

### Decode 阶段

目标：理解模型如何逐 token 生成后续文本。

上手步骤：

1. 查看 `nanovllm/engine/model_runner.py` 的 `prepare_decode()`。
2. 关注每条序列只取 `last_token` 作为本轮输入。
3. 查看 `nanovllm/engine/scheduler.py` 的 `postprocess()`，理解新 token 如何追加到序列。
4. 观察生成过程中的 Decode token/s 指标。

验收标准：能说明“decode 是在已有 KV Cache 基础上，一轮生成一个新 token”。

### KV Cache 管理

目标：理解为什么本地推理需要缓存历史上下文。

上手步骤：

1. 阅读 `nanovllm/engine/block_manager.py` 中的 `Block` 和 `BlockManager`。
2. 理解每个序列会被拆成多个 token block。
3. 查看 `allocate()` 和 `deallocate()`，理解缓存块如何分配和释放。
4. 查看 `nanovllm/layers/attention.py` 的 `store_kvcache()`，理解 key/value 如何写入缓存。

验收标准：能说明“KV Cache 让模型不必每次重新计算完整历史上下文”。

### Prefix Cache

目标：理解多个请求共享相同背景设定时如何复用缓存。

上手步骤：

1. 准备多条具有相同开头的 prompt，例如相同世界观设定加不同角色台词。
2. 阅读 `BlockManager.compute_hash()` 和 `allocate()`。
3. 观察 `seq.num_cached_tokens` 在缓存命中时如何增加。
4. 将该能力映射到 AVG：共用世界观、章节背景、角色档案时可减少重复计算。

验收标准：能说明“相同前缀可以复用已经算过的上下文，适合分支剧情批量生成”。

### Tensor Parallel

目标：理解多 GPU 推理扩展方式。

上手步骤：

1. 查看 `LLM(..., tensor_parallel_size=1)` 的参数。
2. 阅读 `nanovllm/layers/linear.py` 中的 `ColumnParallelLinear` 和 `RowParallelLinear`。
3. 理解部分权重按 GPU 拆分，结果再通过通信合并。
4. 在有多张 GPU 的环境中尝试设置 `tensor_parallel_size=2`。

验收标准：能说明“Tensor Parallel 是把模型计算拆到多张 GPU 上运行，用于更大模型或更高吞吐”。

### CUDA Graph

目标：理解为什么固定形状的 decode 可以被加速。

上手步骤：

1. 对比 `LLM(..., enforce_eager=True)` 和 `enforce_eager=False`。
2. 查看 `nanovllm/engine/model_runner.py` 的 `capture_cudagraph()`。
3. 理解 CUDA Graph 会提前捕获一批常见 batch size 的执行图。
4. 用 benchmark 观察开启 CUDA Graph 后 decode 阶段的性能变化。

验收标准：能说明“CUDA Graph 减少重复调度开销，适合稳定的推理循环”。

### Qwen3 模型结构

目标：理解当前本地模型支持的基础结构。

上手步骤：

1. 阅读 `nanovllm/models/qwen3.py`。
2. 按顺序理解 `Embedding -> DecoderLayer -> Attention/MLP -> Norm -> LM Head`。
3. 查看 `Qwen3ForCausalLM.compute_logits()`，理解 hidden states 如何转成词表概率。
4. 将模型结构和生成流程对应起来：prompt 进入模型，logits 进入 sampler，sampler 选出新 token。

验收标准：能用一句话解释“Qwen3 模型负责根据上下文预测下一个 token”。

### AVG 生成适配

目标：让上层剧情系统可以把约束转为本地模型输入。

上手步骤：

1. 查看 `example_avg.py`，理解 `PlayerEvent`、`AstroInfluence`、`PlotPoint` 的使用方式。
2. 阅读 `nanovllm/avg/director.py`，理解 `build_state -> select_plot_point -> build_prompt -> generate` 的流程。
3. 修改剧情节点描述，观察最终生成文本风格变化。
4. 修改 `AstroInfluence.bias` 和 `weights`，观察收敛到的剧情节点是否变化。

验收标准：能跑通“玩家操作+星体影响 -> 剧情节点 -> 旁白+台词”的完整链路。

### AVG 调试结果

目标：让游戏侧不仅拿到文本，也能拿到生成依据，方便排查和调参。

上手步骤：

1. 使用 `director.generate_with_plan(events, astro, params)`。
2. 查看返回对象的 `text`，这是可放入游戏的生成文本。
3. 查看 `plot_point`，确认本次收敛到了哪个剧情节点。
4. 查看 `state`，确认玩家行为和星体影响叠加后的最终状态。
5. 查看 `prompt`，用于调试模型为什么这样生成。

验收标准：能把一次生成的“输入依据、收敛结果、模型 prompt、最终文本”完整记录下来。

### Token 级硬约束

目标：明确当前预留能力和后续升级方向。

上手步骤：

1. 查看 `AVGGenerationParams` 中的 `banned_words` 和 `required_words`。
2. 在 `example_avg.py` 中配置禁止词和必需词。
3. 查看 prompt 中如何把这些约束交给模型。
4. 后续如果需要强制生效，再扩展 `nanovllm/layers/sampler.py` 或 decode 后处理逻辑。

验收标准：能说明“v1 已有约束入口，当前通过 prompt 生效；严格禁止词或格式控制属于后续增强”。

## 在 AVG 项目中的位置

```text
玩家行为
  -> AVG 状态量化
  -> 星体影响接口
  -> 剧情节点收敛
  -> Prompt 约束生成
  -> nano-vLLM 本地模型推理
  -> 旁白和台词文本
```

`nano-vLLM` 不负责设计剧情规则，也不负责计算星体运动。它负责把上层系统已经整理好的剧情约束、人物状态和写作要求，稳定地送入本地模型并产出文本。

## 交付边界

- 已具备本地模型推理、批量调度、KV Cache、Prefix Cache、Qwen3 推理和 AVG 适配接口。
- 星体运动拟合、游戏内星体运行和随机化算法由外部模块实现。
- 当前 token 级硬约束为接口预留，第一版通过 prompt 约束文本生成。
- 实际生成质量依赖所选模型、prompt 设计和剧情节点配置，需要结合游戏内容继续调优。

## 可对外汇报文本

本项目已具备一个轻量级本地大语言模型推理底座，可在本地环境中加载并运行开源语言模型，为 AVG 游戏提供剧情文本生成能力。系统采用分层设计：底层 `nano-vLLM` 负责模型加载、推理加速、请求调度和文本生成；上层 AVG 模块负责玩家状态、剧情节点和星体影响接口。

这样的设计可以让剧情逻辑、星体随机化逻辑和模型推理逻辑相互解耦。后续如果调整星体算法或剧情规则，不需要重写模型推理引擎；如果更换本地模型，也可以尽量保持上层游戏逻辑稳定。第一版已完成本地生成链路、AVG 状态约束接口和面向星体影响模块的数据接入边界，适合作为后续游戏文本生成系统的基础版本。
