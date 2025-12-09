# Pipeline 功能变更日志

## 版本 v0.1.0 - 初始实现 (2025-11-19)

### 新增功能 ✨

#### 1. Attention/FFN 两阶段流水线并行
- 实现基于 NVIDIA Green Context 的 SM 级别资源划分
- Decode 阶段支持 Attention 和 FFN 并行执行
- 自动降级机制（不支持 Green Context 时回退顺序执行）

#### 2. 新增模块

**核心引擎**:
- `nanovllm/engine/green_manager.py` - Green Context 资源管理器
  - SM 资源分配和切分
  - CUDA Stream 管理
  - 动态调整接口（预留）

- `nanovllm/engine/pipeline_scheduler.py` - 流水线调度器
  - 两阶段异步执行
  - CUDA Event 同步
  - 性能统计和监控

**配置扩展**:
- `nanovllm/config.py` - 新增 4 个 pipeline 配置参数
  - `enable_pipeline`: 启用/禁用开关
  - `pipeline_attention_sm_count`: Attention SM 数量
  - `pipeline_ffn_sm_count`: FFN SM 数量
  - `pipeline_enable_profiling`: 性能分析开关

#### 3. 模型改动

**`nanovllm/models/qwen3.py`**:
- 在三个层级添加分阶段执行方法：
  - `Qwen3DecoderLayer.forward_attention()`
  - `Qwen3DecoderLayer.forward_ffn()`
  - `Qwen3Model.forward_attention()`
  - `Qwen3Model.forward_ffn()`
  - `Qwen3ForCausalLM.forward_attention()`
  - `Qwen3ForCausalLM.forward_ffn()`
- 保持原有 `forward()` 方法完全兼容

#### 4. 引擎集成

**`nanovllm/engine/model_runner.py`**:
- Pipeline 组件初始化（仅单卡 + pipeline 启用时）
- `run_model()` 中添加 pipeline 执行路径
- 新增方法：
  - `get_pipeline_statistics()`: 获取统计信息
  - `print_pipeline_statistics()`: 打印性能报告

#### 5. 工具和示例

**示例脚本**:
- `example_pipeline.py` - 基础使用示例
- `example_with_pipeline.py` - 完整示例（包含性能监控）
- `test_pipeline.py` - 功能测试和性能对比

**文档**:
- `PIPELINE_README.md` - 完整用户文档
  - 快速开始
  - 架构设计
  - 性能调优
  - 故障排查
- `IMPLEMENTATION_SUMMARY.md` - 实现技术文档
- `QUICKSTART_PIPELINE.md` - 5 分钟快速入门
- `CHANGELOG_PIPELINE.md` - 本文档

**辅助工具**:
- `green_ctx.py` - Green Context 工具函数（来自 flashinfer）

### 修改的文件 📝

#### 核心代码修改
1. `nanovllm/config.py`
   - 新增 4 个 pipeline 配置字段

2. `nanovllm/models/qwen3.py`
   - `Qwen3DecoderLayer`: +2 方法
   - `Qwen3Model`: +2 方法
   - `Qwen3ForCausalLM`: +2 方法
   - 原有功能完全兼容

3. `nanovllm/engine/model_runner.py`
   - 导入 pipeline 模块
   - `__init__`: 初始化 pipeline 组件
   - `run_model`: 添加 pipeline 执行分支
   - 新增 2 个统计方法

#### 兼容性保证
- ✅ 不启用 pipeline 时，行为完全不变
- ✅ 现有 API (`LLM.generate`) 完全兼容
- ✅ 配置参数向后兼容（默认禁用）
- ✅ 不支持 Green Context 时自动降级

### 新增文件 📄

```
根目录:
  - PIPELINE_README.md           完整文档
  - IMPLEMENTATION_SUMMARY.md    实现总结
  - QUICKSTART_PIPELINE.md       快速入门
  - CHANGELOG_PIPELINE.md        变更日志
  - example_pipeline.py          基础示例
  - example_with_pipeline.py     完整示例
  - test_pipeline.py             测试脚本
  - green_ctx.py                 Green Context 工具

nanovllm/engine/:
  - green_manager.py             Green Context 管理器
  - pipeline_scheduler.py        流水线调度器
```

### 系统要求 🔧

**新增依赖**:
- CUDA 12.4+ (Green Context 支持)
- cuda-python (匹配 CUDA 版本)
- GPU: Compute Capability 6.0+ (推荐 8.0+)

**限制**:
- 仅支持单卡模式 (`tensor_parallel_size=1`)
- 仅在 decode 阶段启用（prefill 仍顺序执行）
- 不支持 CUDAGraph（pipeline 模式下）

### 性能特性 ⚡

**优化场景**:
- ✅ 多请求并发 decode (batch_size >= 4)
- ✅ 长序列生成 (max_tokens >= 50)
- ✅ 高吞吐场景

**监控支持**:
- Per-stage 耗时统计
- 总吞吐量计算
- 队列长度监控（预留）

### 使用方式 📖

**最简启用**:
```python
llm = LLM("./model", enable_pipeline=True)
```

**完整配置**:
```python
llm = LLM(
    "./model",
    enable_pipeline=True,
    pipeline_attention_sm_count=16,
    pipeline_ffn_sm_count=16,
    pipeline_enable_profiling=True,
)
```

**查看统计**:
```python
llm.model_runner.print_pipeline_statistics()
```

### 已知问题 ⚠️

1. **单卡限制**: 多卡场景 (TP > 1) 暂不支持
2. **显存开销**: 中间张量占用额外显存
3. **小 workload**: 单 token 短序列可能无性能提升
4. **动态调整**: SM 自动调整功能未实现（接口已预留）

### 下一步计划 🚀

**短期 (M4)**:
- [ ] 实现动态 SM 调整
- [ ] 更细粒度的性能分析

**中期 (M5-M6)**:
- [ ] 多卡支持 (TP > 1)
- [ ] 三阶段流水线（逐层并行）
- [ ] Per-layer profiling

**长期 (M7+)**:
- [ ] CUDAGraph 兼容性
- [ ] 自适应 SM 分配算法
- [ ] 支持其他模型架构

### 测试覆盖 ✅

- [x] 基础功能测试
- [x] Pipeline vs Sequential 性能对比
- [x] 不同 SM 分配配置测试
- [x] 降级机制验证
- [x] 语法正确性检查

### 迁移指南 🔄

从现有代码迁移到支持 pipeline：

**步骤 1**: 更新代码（可选）
```python
# 旧代码
llm = LLM("./model")

# 新代码（向后兼容）
llm = LLM("./model", enable_pipeline=True)
```

**步骤 2**: 无需修改其他代码
- `generate()` 调用方式不变
- 输出格式不变
- 现有功能完全兼容

**步骤 3**: 验证启用（可选）
```python
stats = llm.model_runner.get_pipeline_statistics()
if stats.get("enabled", False):
    print("Pipeline enabled!")
```

### 参考资料 📚

- [PIPELINE_README.md](PIPELINE_README.md) - 用户文档
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - 技术文档
- [DECODE_PIPELINE_PLAN.md](DECODE_PIPELINE_PLAN.md) - 设计规划
- [NVIDIA Green Context API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)

### 贡献者 👥

- 初始实现: Claude Code
- 设计参考: DECODE_PIPELINE_PLAN.md
- Green Context 工具: flashinfer 项目

---

**版本**: v0.1.0
**发布日期**: 2025-11-19
**状态**: 初始实现完成，可用于测试
