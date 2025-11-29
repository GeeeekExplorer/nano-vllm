from nanovllm.engine.llm_engine import LLMEngine

"""
在 `nano-vllm` 的实现中，`LLMEngine` 类已经承担了所有的工作，包括：
- 初始化模型和分词器
- 管理调度器（Scheduler）和模型执行者（ModelRunner）
- 直接实现了面向用户的 `generate` 方法（包含 `while` 循环和进度条），这在标准的 vLLM 架构中通常是 `LLM` 类的职责

这里的 LLM 类仅仅是一个别名，为了保持与 vLLM API 的一致性，让用户可以像使用 vLLM 一样导入和使用它
"""
class LLM(LLMEngine):
    pass
