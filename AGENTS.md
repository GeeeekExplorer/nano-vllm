# Repository Guidelines

## Project Structure & Module Organization
`nanovllm/` 是核心包。`engine/` 负责调度、序列与 block 管理，`layers/` 放注意力、线性层、归一化等基础组件，`models/` 放模型实现，`utils/` 放加载与上下文工具。仓库根目录下的 `example.py` 用于最小可运行示例，`bench.py` 用于离线吞吐基准，`assets/` 存放文档资源。

## Build, Test, and Development Commands
推荐使用 Python 3.10 到 3.12。

```bash
pip install -e .
python example.py
python bench.py
python -m build
```

`pip install -e .` 以可编辑模式安装本项目依赖。`python example.py` 验证基础推理流程。`python bench.py` 运行本地性能测试。`python -m build` 生成源码包和 wheel，用于检查发布构建是否正常。

## Coding Style & Naming Conventions
遵循现有 Python 风格：4 空格缩进，模块与函数使用 `snake_case`，类名使用 `PascalCase`，常量使用全大写。新增代码应保持实现短小、可读，优先复用 `nanovllm/engine/` 与 `nanovllm/layers/` 现有抽象。公开接口应尽量与 vLLM 用法保持一致，例如 `LLM` 和 `SamplingParams`。

当前仓库未配置 `ruff`、`black` 或 `mypy`。提交前至少运行相关脚本，确保导入顺序、命名和注释风格与现有文件一致。

## Testing Guidelines
仓库目前没有独立 `tests/` 目录，验证主要依赖示例脚本和基准脚本。新增功能或修复缺陷时，建议补充 `tests/` 下的 `pytest` 用例，文件命名采用 `test_<feature>.py`。如果改动影响推理行为，请同时提供最小复现输入，必要时更新 `example.py` 或补充基准对比数据。

## Commit & Pull Request Guidelines
最近提交历史以简短祈使句为主，例如 `support qwen2`、`simplify`、`compile random sampling`。请沿用这种风格，单个提交聚焦一个变更点。提交 PR 时应说明动机、核心改动、验证方式和影响范围；如果修改生成结果、性能或模型兼容性，请附上命令、关键输出或对比数据。

## Security & Configuration Tips
不要把本地模型权重、缓存目录或私有 token 提交到仓库。示例默认使用 `~/huggingface/Qwen3-0.6B/`；如需调整路径，优先通过本地环境配置处理，不要硬编码机器专属目录。
