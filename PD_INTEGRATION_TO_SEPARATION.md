# NanoVLLM：从函数出发看 PD 一体到 PD 分离（低智博士友好版）

这份笔记按**函数调用顺序**讲解 Prefill/Decode（PD）如何从一张 GPU 串行执行，演进成 M1 的双卡分离。只要跟着函数看，就能理解背后的动机和收益。

---

## 1. 核心函数导读（看懂这些名字就成功一半）
| 模块 | 关键函数 / 字段 | 作用 |
| --- | --- | --- |
| `nanovllm/config.py` | `Config.enable_two_gpu_pd`, `prefill_device_id`, `decode_device_id` | 控制是否启用 PD 分离以及两张卡的编号；`__post_init__` 里校验「两卡不同 & tensor_parallel_size=1」。 |
| `nanovllm/engine/llm_engine.py` | `LLMEngine.__init__`, `_step_single_gpu()`, `_step_two_gpu()` | `__init__` 根据配置决定创建一个 `ModelRunner`（一体模式）或两个 `ModelRunner`（分离模式）。`_step_single_gpu` 与 `_step_two_gpu` 分别定义两种调度流程。 |
| `nanovllm/engine/scheduler.py` | `schedule_prefill()`, `schedule_decode()`, `get_prefilled_sequences()` | 显式拆分 Prefill/Decode 队列：前者只喂 GPU0，后者只喂 GPU1，并记录哪些序列刚 Prefill 完成。 |
| `nanovllm/engine/model_runner.py` | `ModelRunner.run_prefill()`, `run_decode_core()`, `sync_kv_cache_to()` | 真正执行模型推理；在 M1 中被实例化为 prefill runner & decode runner。`sync_kv_cache_to()` 提供 KV cache 跨卡复制能力。 |
| `example_two_gpu.py` / `test_two_gpu.py` | `LLM(..., enable_two_gpu_pd=True)` | 实际演示 / 验证分离后的执行流。 |

搞懂这些函数谁调谁、在哪个 GPU 上，就掌握了整个 PD 分离。

---

## 2. 单卡 PD 一体：`_step_single_gpu()` 的故事
### 函数链
1. `LLMEngine.add_request()`：把 prompt 封装成 `Sequence` 丢进 `Scheduler.prefill_queue`。
2. `LLMEngine._step_single_gpu()`：
   - 调 `Scheduler.schedule()`，内部先走 `schedule_prefill()`，不行再走 `schedule_decode()`。
   - 得到一个批次 `(seqs, is_prefill)` 后，调用 `self.model_runner.call("run", seqs, is_prefill)`。
3. `ModelRunner.run()`：
   - 根据 `is_prefill` 选择 `prepare_prefill()` 或 `prepare_decode()` 构造输入；
   - 分别走 `run_prefill()`（一次性全 prompt 前向）或 `run_decode_core()`（增量 decode），两者统统跑在同一个 `cuda:<rank>`；
   - `Sampler` 采样 token，返回给引擎。
4. `Scheduler.postprocess()`：把生成 token 追加进 `Sequence`，判断是否结束。

### 痛点
- Prefill 和 Decode 都跑在同一张 GPU，`LLMEngine._step_single_gpu()` 只能串行调 `run_prefill()`、再调 `run_decode_core()`。
- Prefill 时 decode 队列停摆，Decode 时无法新建 prompt，GPU 在两类算子之间频繁切换、cache 反复抖动。
- 结果就是吞吐被慢阶段限制（在 `GREEN_CONTEXT_PERFORMANCE_ANALYSIS.md` 里只看到 40~50 tok/s 的水平）。

---

## 3. 双卡 PD 分离（M1）：`_step_two_gpu()` 如何分工
### 函数级流程
1. **构造两个 `ModelRunner`**  
   `LLMEngine.__init__` 在 `enable_two_gpu_pd=True` 时会创建：
   ```python
   self.prefill_runner = ModelRunner(..., device_id=prefill_device_id, is_decode_runner=False)
   self.decode_runner  = ModelRunner(..., device_id=decode_device_id,  is_decode_runner=True)
   ```
   - 两个实例各自 `torch.cuda.set_device()` 到不同的 GPU。
   - Decode runner 可以进一步启用 `pipeline_scheduler`（M2），但对 M1 来说只要 `run_decode_core()` 单流执行即可。

2. **调度拆分**  
   `_step_two_gpu()` 先问 `Scheduler.ready_for_prefill()`，如果队列非空：
   - `schedule_prefill()` 返回一批序列，并把它们从 `prefill_queue` 移到 `decode_queue`。
   - 引擎调用 `prefill_runner.run(..., is_prefill=True)`，Prefill 只在 GPU0 进行。

3. **KV cache 跨卡同步**  
   Prefill 完成后 `_step_two_gpu()` 会收集 `seq.block_table`，然后：
   ```python
   self.decode_runner.kv_cache[:, :, block_idx].copy_(
       self.prefill_runner.kv_cache[:, :, block_idx]
   )
   ```
   这一步就是 `ModelRunner.sync_kv_cache_to()` 的手工版本：直接把 Prefill 的 KV cache 块复制到 GPU1（同 NVLink/PCIe）。

4. **Decode 独立执行**  
   再问 `Scheduler.ready_for_decode()`，若有序列，调用 `schedule_decode()` 把它们从 `decode_queue` 取出，然后交给
   ```python
   self.decode_runner.run(..., is_prefill=False)
   ```
   Decode 阶段只使用 GPU1 的 `run_decode_core()`，Prefill 卡可以继续服务下一批 prompt。

5. **结果回写**  
   `_step_two_gpu()` 在 GPU1 生成 token 后，再次调用 `Scheduler.postprocess()` 更新状态。Prefill 卡不需要等待，继续循环下一轮。

### 直观时间线
```
GPU0 (prefill_runner.run):
  [Prefill Req A]     [Prefill Req B]     [Prefill Req C] ...

GPU1 (decode_runner.run):
           [Decode Req Aaaaaaaaa] [Decode Req Bbbbbbbbb] ...
```
两张 GPU 只有在 Prefill 刚完成时通过 KV cache copy “握一次手”，其他时间完全并行。

---

## 4. 函数细节：KV cache 与上下文是怎么搬的？
1. `Sequence.block_table` 记录了该序列持有的 KV cache block 索引。Prefill 阶段 `ModelRunner.prepare_prefill()` 会把这些 block 映射到实际显存地址。
2. Prefill runner 在 `run_prefill()` 里写入 `self.kv_cache[:, :, block_idx]`。
3. `_step_two_gpu()` 遍历 block 索引，使用 `torch.cuda.device(prefill_gpu)` 上的 `copy_` 把数据搬到 `decode_runner.kv_cache`。
4. Decode runner 在 `prepare_decode()` 中用刚同步过来的 block 构造 `slot_mapping`，确保 `run_decode_core()` 能直接访问 Prefill 生成的上下文。
5. `Scheduler.get_prefilled_sequences()` 可用于更精细的队列统计（比如调试日志），因为它会记住上一轮 `schedule_prefill()` 返回了哪些序列、需要同步哪些 block。

总结：KV cache 像接力棒，通过 `seq.block_table → kv_cache.copy_()` 完成 GPU 之间的传递，Decode 全程不用再向 GPU0 读数据。

---

## 5. 为什么值得？（函数视角下的收益和代价）
| 类别 | 单卡 `_step_single_gpu` | 双卡 `_step_two_gpu` | 原因 & 代价 |
| --- | --- | --- | --- |
| GPU 利用率 | Prefill/Decode 抢同一张卡，被慢阶段拖住 | Prefill & Decode 各自有 runner，同步点极少 | 调度函数把两条队列分开执行 |
| 吞吐 | ~40–50 tok/s（参考 `GREEN_CONTEXT_PERFORMANCE_ANALYSIS.md` 单卡） | ~99 tok/s（两卡 PD 模式） | `_step_two_gpu` 把 Prefill 与 Decode 并行化 |
| Prompt 等待时间 | 新请求得等 decode 完 | `schedule_prefill` 只要有空就继续喂 GPU0 | Prefill runner 不再被 decode 阻塞 |
| 资源 | 只需 1 张 GPU，KV cache 单份 | 2 张 GPU，各自保留一份 KV cache | `ModelRunner.allocate_kv_cache()` 在两卡上都执行，显存 ≈ 2x |
| 约束 | 兼容 tensor parallel | `Config.__post_init__` 强制 `tensor_parallel_size == 1` | 跨卡 AllReduce 成本太大，暂不支持 |

---

## 6. 如何在代码里打开开关并验证
```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "./Qwen3-0.6B/",
    enable_two_gpu_pd=True,
    prefill_device_id=0,
    decode_device_id=1,
)
outputs = llm.generate(
    prompts=["Hello NanoVLLM!"],
    sampling_params=SamplingParams(max_tokens=64),
)
```
运行 `python example_two_gpu.py` 会在终端看到：
```
[LLMEngine] Initializing Two-GPU PD separation mode
  - Prefill GPU: 0
  - Decode GPU: 1
```
说明 `LLMEngine.__init__` 已走到双 runner 分支。想持续验证，直接执行 `python test_two_gpu.py`，脚本会调用 `_step_two_gpu()` 并检查 Prefill/Decode 结果一致性。

---

## 7. 低智博士记忆口诀
1. **先记函数链**：`LLMEngine._step_* → Scheduler.schedule_* → ModelRunner.run_*`。
2. **再想硬件图**：Prefill runner 固定 GPU0，Decode runner 固定 GPU1，靠 `kv_cache.copy_()` 接力。
3. **最后记收益**：`_step_two_gpu()` 让两个函数序列平行运行，吞吐翻倍，为后续 M2（decode 内部 `pipeline_scheduler`) 奠定基础。

理解了这些函数的职责，你就真正搞懂了 NanoVLLM 从 PD 一体到 PD 分离的原理与目的。接下来若继续阅读 `M2_DECODE_PIPELINE.md`，会发现 decode runner 还能在 GPU1 内部再拆函数、继续加速。

