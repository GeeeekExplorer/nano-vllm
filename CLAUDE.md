# nano-vllm 源码阅读上下文

## 阅读目标

帮助用户系统理解 nano-vllm 的完整推理流程，重点关注：
- 每个模块"做什么"以及"为什么这样设计"
- 各模块之间的调用关系和数据流向
- 关键优化（KV Cache、前缀缓存、CUDA Graph、张量并行）的实现方式

## 推荐阅读顺序

按以下顺序逐模块展开，每次专注一个模块：

0. **让项目跑起来**：安装依赖、下载模型、运行 `example.py`，确认端到端流程正常
1. **入口与配置**：`llm.py` → `config.py` → `sampling_params.py` → `example.py`
2. **引擎主循环**：`engine/llm_engine.py`
3. **请求生命周期**：`engine/sequence.py`
4. **调度逻辑**：`engine/scheduler.py`
5. **KV Cache 管理**：`engine/block_manager.py`
6. **GPU 执行**：`engine/model_runner.py`
7. **神经网络层**：`layers/attention.py` → `layers/linear.py` → 其余 layers
8. **模型架构 + 工具**：`models/qwen3.py` → `utils/loader.py` → `utils/context.py`

## 讲解方式约定

- 先读代码，再解释；不要凭印象描述细节
- 解释设计决策时，联系 LLM 推理的工程背景（为什么需要这个优化）
- 标注当前模块与其他模块的接口点（调用者 / 被调用者）
- 用中文解释，代码保持原样
- **用户的所有补充提问和回答，无需询问，自动追加到 NOTES.md 的「补充问答」章节，并更新进度**

## 关键概念速查

- **Prefill vs Decode**：prefill 一次性处理全部 prompt tokens；decode 每步生成一个新 token
- **KV Cache 块**：256 tokens/block，逻辑块映射到物理 GPU 内存
- **前缀缓存**：对 token 序列哈希，相同前缀复用已有 KV 块
- **张量并行**：多 GPU 按列/行切分权重矩阵，rank 0 负责调度通信
- **CUDA Graph**：提前录制固定形状的 decode 步骤，回放时跳过 Python overhead

## 当前进度

（Claude 在阅读过程中更新此处，记录已完成的模块）

- [x] 模块 0：让项目跑起来（安装 + 运行 example.py）
- [x] 模块 1：入口与配置
- [x] 模块 2：引擎主循环
- [x] 模块 3：请求生命周期
- [x] 模块 4：调度逻辑
- [x] 模块 5：KV Cache 管理
- [ ] 模块 6：GPU 执行  ← 当前
- [ ] 模块 5：KV Cache 管理
- [x] 模块 6：GPU 执行
- [x] 模块 7：神经网络层
- [x] 模块 8：模型架构 + 工具
