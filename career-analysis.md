# LLM 细分方向分析与学习路线

## 个人背景

- 2 年云服务开发（Docker/K8s/API/分布式）
- 1.5 年自动驾驶感知（YOLOv5 训练、ONNX/OM 转换、车端软件）
- 无深度学习/神经网络理论基础
- 目标：转入 LLM/Agent 方向，追求更高薪资和技术热度

---

## 行业对比：自动驾驶感知 vs LLM/Agent

### 薪资水平

| 维度 | 自动驾驶感知 | LLM/Agent |
|------|------------|-----------|
| 应届/初级 | 25-40w | 30-50w |
| 3-5年 | 40-70w | 50-100w |
| 资深/专家 | 70-120w | 80-200w+ |
| 薪资天花板 | 较高但有限 | 极高（头部公司股票溢价大） |
| 涨薪速度 | 稳定 | 跳槽涨幅大，但波动也大 |

### 岗位数量

| 维度 | 自动驾驶感知 | LLM/Agent |
|------|------------|-----------|
| 当前岗位量 | 收缩中 | 爆发增长中 |
| 招聘方 | 车企+Tier1（华为、大疆、小鹏、蔚来等） | 几乎所有互联网/AI 公司 + 传统企业转型 |
| 地域分布 | 集中（上海、北京、深圳、合肥） | 广泛（几乎所有一线+新一线城市） |
| 岗位多样性 | 窄（感知/规控/仿真） | 广（推理部署、RAG、Agent、微调、应用层） |

LLM/Agent 的岗位数量大约是自动驾驶感知的 **5-10 倍**。

### 未来发展

**自动驾驶感知：**
- 技术路线趋于收敛（BEV+Transformer → 端到端），需要的人在变少
- 头部玩家已形成壁垒（华为、特斯拉、Waymo），中小公司生存空间小
- L2+/L3 量产落地会持续，但增长是线性的
- 端到端方案成熟后，感知岗位需求会进一步减少

**LLM/Agent：**
- 仍处于技术爆发早期，新范式不断出现（MCP、Agent 框架、多模态）
- 应用场景在快速扩展：编程助手、客服、搜索、办公、教育、医疗...
- 不仅 AI 公司需要，传统企业也在大规模引入
- Agent 是当前最热方向，2025-2027 预计持续高增长
- 风险：泡沫存在，部分应用层岗位可能被 AI 自身替代

### 结论

**更看好 LLM/Agent，明显领先。** 市场天花板更高，个人背景更匹配，转入门槛更低（3-6 个月 vs 1-2 年）。

---

## LLM 细分方向详解

### 方向全景

```
LLM 技术栈
├── 1. 模型训练（预训练/微调）
├── 2. 推理引擎与部署
├── 3. RAG 系统
├── 4. Agent 框架与应用
├── 5. 数据工程
└── 6. 评测与安全
```

### 1. 模型训练（预训练 / 微调 / RLHF）

| 项目 | 说明 |
|------|------|
| 做什么 | 从零训练大模型，或在已有模型上微调 |
| 技术栈 | PyTorch、DeepSpeed、Megatron、FSDP、数据清洗 |
| 门槛 | **极高** — 需要深度学习理论 + 大规模分布式训练经验 |
| 薪资 | 最高档，但岗位少 |
| **匹配度** | **★☆☆☆☆ 不推荐** — 没有 ML 基础，补课周期 1-2 年 |

### 2. 推理引擎与部署 — 最推荐

| 项目 | 说明 |
|------|------|
| 做什么 | 让大模型高效运行：推理加速、显存优化、多机部署、API 服务 |
| 技术栈 | vLLM/SGLang、CUDA/Triton、Docker/K8s、ONNX、TensorRT |
| 门槛 | **中等** — 系统工程能力 > ML 理论 |
| 薪资 | 50-120w，需求旺盛 |
| **匹配度** | **★★★★★** |

**为什么最适合：**
- 云服务开发 2 年 → 懂容器化、分布式、API 服务，这是推理部署的核心
- ONNX/OM 转换经验 → 直接对口模型格式转换和推理优化
- 刚读完 nano-vllm 源码 → 已理解 PagedAttention、KV Cache、CUDA Graph
- **不需要深度 ML 理论**，重点是系统工程

**推荐开源项目：**

| 项目 | Star | 学什么 | 优先级 |
|------|------|--------|--------|
| [vLLM](https://github.com/vllm-project/vllm) | 50k+ | 工业级推理引擎，已读完 nano 版 | **必学** |
| [SGLang](https://github.com/sgl-project/sglang) | 10k+ | RadixAttention、更激进的调度 | **必学** |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | 75k+ | CPU/边缘推理、量化 | 推荐 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | 10k+ | NVIDIA 官方 LLM 推理优化 | 推荐 |
| [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII) | 2k+ | 微软的推理部署方案 | 了解 |

### 3. RAG 系统 — 次推荐

| 项目 | 说明 |
|------|------|
| 做什么 | 检索增强生成：让 LLM 基于外部知识库回答问题 |
| 技术栈 | 向量数据库、Embedding、检索策略、Prompt 工程 |
| 门槛 | **低-中等** — 工程为主，ML 理论需求少 |
| 薪资 | 40-80w，岗位数量最多 |
| **匹配度** | **★★★★☆** |

**为什么适合：**
- 本质是后端工程（数据库 + API + 管道编排），云服务背景直接适用
- 入门快，1-2 个月能做出项目
- 但天花板相对低，容易同质化竞争

**推荐开源项目：**

| 项目 | 学什么 |
|------|--------|
| [LangChain](https://github.com/langchain-ai/langchain) | RAG 管道编排，生态最大 |
| [LlamaIndex](https://github.com/run-llama/llama_index) | 专注检索和索引策略 |
| [Milvus](https://github.com/milvus-io/milvus) | 向量数据库（Go 写的，了解原理即可） |
| [RAGFlow](https://github.com/infiniflow/ragflow) | 国产 RAG 引擎，文档解析做得好 |

### 4. Agent 框架与应用 — 值得投入

| 项目 | 说明 |
|------|------|
| 做什么 | 让 LLM 使用工具、多步推理、协作完成复杂任务 |
| 技术栈 | Agent 框架、Tool Use、MCP 协议、工作流编排 |
| 门槛 | **中等** — 系统设计 + Prompt 工程 + 工具链集成 |
| 薪资 | 50-100w，当前最热方向 |
| **匹配度** | **★★★★☆** |

**为什么适合：**
- Agent 本质是"让 LLM 调用 API 完成任务"，做过云服务 API 开发
- 2025-2026 最火方向，MCP 协议刚起步，先入者有红利
- 但目前框架变化极快，今天学的明天可能过时

**推荐开源项目：**

| 项目 | 学什么 |
|------|--------|
| [LangGraph](https://github.com/langchain-ai/langgraph) | 有状态 Agent 工作流 |
| [CrewAI](https://github.com/crewAIInc/crewAI) | 多 Agent 协作 |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | OpenAI 官方 Agent 框架 |
| [Dify](https://github.com/langgenius/dify) | 低代码 LLM 应用平台（国产） |
| [Claude Code](https://github.com/anthropics/claude-code) | Agent 实践范例 |

### 5. 数据工程 & 6. 评测与安全

| 匹配度 | **★★☆☆☆** — 能做但不是最优路径 |
|---------|------|
| 数据工程 | 训练数据清洗、标注管线，偏重 ETL，薪资中等 |
| 评测安全 | Red Teaming、对齐评估，岗位少但在增长 |

---

## 推荐学习路线（3 个月）

### 第 1 个月：推理部署（核心竞争力）

- 精读 vLLM 源码（已有 nano-vllm 基础，事半功倍）
- 跑通 SGLang，对比两者调度策略差异
- 学习量化基础：GPTQ、AWQ、llama.cpp 的 GGUF
- 实践：用 vLLM 部署一个 7B 模型，写 OpenAI 兼容 API

### 第 2 个月：RAG + Agent（拓宽应用面）

- LangChain/LlamaIndex 搭建 RAG 系统
- 向量数据库选型与使用（Milvus 或 Chroma）
- LangGraph 构建多步 Agent
- 实践：做一个能检索文档 + 调用工具的 Agent

### 第 3 个月：项目整合 + 面试准备

- 一个完整项目：推理部署 + RAG + Agent 串起来
- 学习 MCP 协议（Agent 通信标准）
- 准备技术博客或 GitHub 项目展示
- 刷面经：推理优化、KV Cache、调度策略

### 核心结论

> **推理部署是主赛道**（云服务 + ONNX 经验 + nano-vllm 源码阅读 = 天然匹配），RAG/Agent 是辅助技能树。三者组合就是市场上最抢手的 "LLM 工程师" 画像。
