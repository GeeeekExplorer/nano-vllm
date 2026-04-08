# nano-vllm 源码学习笔记

---

## 模块 0：让项目跑起来

### 环境要求

| 项目 | 要求 | 说明 |
|------|------|------|
| Python | >=3.10, <3.13 | |
| GPU | NVIDIA（CUDA） | flash-attn / triton 不支持 Apple Silicon |
| torch | >=2.4.0 | |
| flash-attn | 最新 | 需与 CUDA 版本匹配编译 |
| triton | >=3.0.0 | CUDA 专属 |
| xxhash | 任意 | 用于前缀缓存的 token 块哈希 |

### 运行步骤

```bash
pip install -e .
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
python example.py
```

### example.py 关键点

- 模型路径写死在 `~/huggingface/Qwen3-0.6B/`
- `enforce_eager=True`：禁用 CUDA Graph，便于调试
- `tensor_parallel_size=1`：单 GPU 模式
- `generate()` 接收已格式化的字符串（带 chat template），不是原始问题

### Chat Template 示例

`tokenizer.apply_chat_template(...)` 会把对话格式化为：

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
<|im_end|>
<|im_start|>user
introduce yourself<|im_end|>
<|im_start|>assistant

```

每个 `<|im_start|>role ... <|im_end|>` 是一轮对话，末尾的 `<|im_start|>assistant\n` 是触发模型开始回答的信号。

---

## 模块 1：入口与配置

### 文件列表

| 文件 | 作用 |
|------|------|
| `nanovllm/__init__.py` | 只暴露 `LLM` 和 `SamplingParams` 两个符号 |
| `nanovllm/llm.py` | `LLM` 是 `LLMEngine` 的空壳子类，预留扩展口 |
| `nanovllm/config.py` | 全局配置 dataclass，`__post_init__` 做校验和自动填充 |
| `nanovllm/sampling_params.py` | 采样参数，`temperature > 1e-10` 强制校验 |

### Config 核心字段

| 字段 | 默认值 | 含义 |
|------|--------|------|
| `max_num_batched_tokens` | 16384 | 一个批次所有请求的 token 总数上限 |
| `max_num_seqs` | 512 | 一个批次最多并发请求数 |
| `max_model_len` | 4096 | 单条请求最长 token 数（取配置与模型位置编码上限的最小值）|
| `gpu_memory_utilization` | 0.9 | 90% 显存用于 KV Cache，10% 留给其他 |
| `kvcache_block_size` | 256 | KV Cache 每块的 token 数，必须是 256 的倍数 |
| `num_kvcache_blocks` | -1 | -1 表示自动根据显存计算 |
| `enforce_eager` | False | True = 禁用 CUDA Graph，便于调试 |

### 设计要点

- **最小化公开接口**：`__init__.py` 只暴露两个符号，内部复杂度对用户不可见
- **命名隔离层**：`LLM(pass)` 把用户 API 和引擎实现分开，未来扩展不动核心
- **校验前置**：Config 在构造时拦截所有非法参数，引擎代码可信任配置合法
- **`ignore_eos=True`**：跑 benchmark 时用，强制生成到 max_tokens 不提前停止

---

## 补充问答

### 显存（VRAM）vs 内存（RAM）

| | 内存（RAM） | 显存（VRAM） |
|---|---|---|
| 位置 | 焊在主板上，CPU 旁边 | 焊在显卡上，GPU 旁边 |
| 谁用 | CPU | GPU（只能直接访问显存）|
| 容量 | 16～64 GB | 8～80 GB |
| 速度 | 快 | 极快（专为并行设计）|
| 互通 | 需通过 PCIe 总线搬运数据 | — |

- GPU 要处理内存里的数据，必须先通过 PCIe 总线复制到显存，有额外开销
- 显存满了直接 OOM 崩溃，没有"溢出到内存"的退路
- **Apple M1 Pro 例外**：CPU、GPU、Neural Engine 集成在同一块 SoC 上，共享统一内存（Unified Memory），无需复制；但软件层（CUDA / flash-attn / triton）不兼容 Apple Metal，所以 nano-vllm 仍跑不起来

### 生成前知道要生成多少 token 吗？

**不知道。** 模型每步只预测下一个 token，遇到 `<EOS>` 才停止，提前无法预知总长度。

- `max_tokens`：兜底上限，防止模型无限生成；模型提前输出 `<EOS>` 则立即停止
- 这是 BlockManager 必须**动态**分配 KV Cache 块的根本原因——无法提前预留

### rank 是什么？一般有多少个？

rank 是多 GPU 场景里每张卡的编号，从 0 开始。张量并行时把模型权重切成若干份，每张卡存一份、算一部分，rank 用来区分"你是第几张卡、负责哪一份"。nano-vllm 最多支持 8 个 rank（即 8 张卡）。

### 为什么 rank 0 在主进程，rank 1+ 在子进程？

主进程承担了 tokenize、调度决策、收集输出、进度条等 GPU 计算以外的工作，rank 0 放主进程逻辑最顺畅。若 rank 0 也放子进程，还需额外通信层传输数据，白白增加复杂度。子进程（rank 1+）只做一件事：等 rank 0 通知 → 执行 GPU 计算 → 返回结果 → 继续等。

### seqs 的具体结构（以两条请求为例）

`seqs` 是一个 `Sequence` 对象列表，每个对象携带一条请求的完整状态：

```
# 刚进入引擎，prefill 前：
seq_id=0: token_ids=[123,456,789,...], num_tokens=20, status=WAITING, block_table=[]
seq_id=1: token_ids=[321,654,987,...], num_tokens=18, status=WAITING, block_table=[]

# prefill 后，decode 第一步生成了 token 888：
seq_id=0: token_ids=[123,...,888], num_tokens=21, status=RUNNING, block_table=[Block(5)]
           prompt_token_ids = token_ids[:20]
           completion_token_ids = token_ids[20:] = [888]
```

### 为什么在循环中收集完成的请求，而不是最后统一收集？

已完成请求的 KV Cache 占用的显存需要立刻释放，让新请求填进来，保持 GPU 满载。如果等所有请求完成再统一释放，先完成的请求的显存会一直白白占着，导致后续请求无法进批次，吞吐量下降。

### 为什么结果不能直接用列表顺序，要用 seq_id 做 key？

不同请求完成时间不同，先完成的先写入结果。如果用列表追加，最终顺序是"完成顺序"而非"提交顺序"。用 `seq_id` 做字典 key，最后按 key 排序取出，无论谁先完成，都能保证返回顺序和输入顺序一致。

### seqs 的流转构成了推理全过程吗？

**是的，seqs 是整个推理流程的主线**。一条 Sequence 的一生：

```
用户提交 prompt
    → tokenize → 创建 Sequence（status=WAITING）
    → Scheduler 分配 KV Cache 块（status=RUNNING）
    → Prefill：prompt 所有 token 一次性过模型，KV Cache 写入显存
    → Decode 循环：每步追加一个新 token，直到 <EOS> 或达到 max_tokens
    → status=FINISHED，从调度队列移除，显存释放
    → completion_token_ids detokenize 成文本返回给用户
```

所有其他组件（Scheduler、BlockManager、ModelRunner）都是围绕 seqs 提供服务的：
- **Scheduler**：决定哪些 seqs 进批次、何时挂起
- **BlockManager**：为 seqs 分配和释放显存块
- **ModelRunner**：接收 seqs，在 GPU 上跑 forward，返回 next token

### Python 是如何一层一层操作 GPU 的？

```
Python 代码（torch API）
    ↓ pybind11
PyTorch C++ 核心（libtorch）—— 真正的 Tensor 对象、算子分发
    ↓
CUDA Runtime API（libcudart）—— cudaMalloc / cudaMemcpy / cudaLaunchKernel
    ↓
CUDA Kernel（.cu 编译成 GPU 机器码）—— 真正的并行计算
    ↓
GPU 硬件（数千个 CUDA core 同时执行）
```

- `flash-attn`：NVIDIA 工程师手写 CUDA C++ kernel，极致优化 attention 计算
- `triton`：Python-like 语法写 kernel → 编译成 PTX → GPU 机器码（`store_kvcache` 用此方式）
- CPU 只负责"发令"，所有数字运算在 GPU 上并行执行

### ModelRunner 关键设计

| 设计 | 原因 |
|------|------|
| `set_default_device("cuda")` 初始化期间开启 | 模型构建和权重加载直接在显存，不经过内存 |
| warmup → 量峰值显存 → 分配 KV Cache | 不运行一遍就不知道模型实际用多少显存 |
| `kv_cache` 一次性分配，各层共享切片 | 避免碎片化，显存连续，访问高效 |
| `pin_memory=True` + `non_blocking=True` | 锁页内存，DMA 直传，CPU/GPU 并行传输 |
| CUDA Graph（decode 专用）| prefill 形状不固定无法录图；decode 形状固定，回放跳过 Python overhead |
| 共享内存（SharedMemory）+ Event 信号 | rank 0 写数据，rank 1+ 等信号读数据，零拷贝多卡同步 |

### 前缀缓存：命中概率和缓存级别

前缀缓存是**显存级别**的缓存（不是磁盘/内存），对每个满 256 token 的块计算哈希，命中时直接复用显存里的 KV 数据，跳过整段 Attention 计算。

命中概率：
- 所有请求共享同一 system prompt（生产最常见）→ **极高**，第一条之后几乎全命中
- 多轮对话历史相同 → 高
- 完全随机 prompt → 几乎为 0

哈希表存在 CPU 内存，KV Cache 数据在显存。命中时只需把物理块编号填入 `block_table`，实际 KV 数据一字节不动。

### CPU/内存 vs GPU/显存 的职责划分

| 永远在内存（CPU侧） | 永远在显存（GPU侧） |
|-------------------|--------------------|
| Sequence 对象、调度队列 | 模型权重（固定不动）|
| block_table 映射（只存块编号）| KV Cache 物理块（实际数据）|
| 前缀缓存哈希表 | 当前批次的中间激活值 |
| token_ids 列表 | — |

每步跨侧传输的数据极少：
- 内存→显存：decode 时只传 `last_token`（1个int/条），prefill 时传全部 token_ids
- 显存→内存：`next_token_id`（1个int/条）

模型权重和 KV Cache 始终在显存，CPU 只知道"块编号"，不碰实际 KV 数据——这是整个设计高效的核心。

### token_ids 的 id 是什么意思？实际 token 存在哪里？

模型有一个词表（约 151,936 个词条），每个词条有固定编号。引擎内部全程只处理整数编号，字符串只存在两端：

```
用户输入 "北京"  →  tokenizer.encode()  →  [9707, Δ, ...]  →  模型计算
模型输出 [9707]  →  tokenizer.decode()  →  "北"  →  返回用户
```

实际词条字符串存储在 `tokenizer.json` / `vocab.json` 里。

### 序列化优化详解（step-by-step）

多进程传输 seq 时，prefill 和 decode 传输内容不同：

- **Prefill**（`num_completion_tokens == 0`）：传完整 `token_ids`，GPU 需要全部 prompt token
- **Decode**（`num_completion_tokens > 0`）：只传 `last_token`（1 个 int），KV Cache 里已有历史信息

差距随生成长度线性增大：生成第 200 步时，无优化传 250 个 int，有优化传 1 个 int，**约 250 倍差距**。单 GPU 时无影响，多 GPU 张量并行时效果显著。

### 多进程通信手段

nano-vllm 使用 PyTorch multiprocessing（底层：共享内存 + pickle）。常见 IPC 方式对比：

| 方式 | 速度 | 说明 |
|------|------|------|
| 共享内存 | 极快，近零拷贝 | nano-vllm 底层使用 |
| 管道/socket | 中等 | 小消息控制信号 |
| pickle over queue | 中等，有序列化开销 | Python 多进程默认 |

### 谁修改 seq 的哪些字段

| 阶段 | 操作者 | 修改的字段 |
|------|--------|------------|
| 诞生 | `LLMEngine.add_request()` | 初始化所有字段，`status=WAITING` |
| 进入批次 | `Scheduler.schedule()` | `status=RUNNING`，填 `block_table`，设 `num_cached_tokens` |
| 每步 decode | `Scheduler.postprocess()` | `append_token()`：`token_ids +1`，`last_token`，`num_tokens +1` |
| 生成结束 | `Scheduler.postprocess()` | `status=FINISHED`，BlockManager 释放显存块 |
| 结果收割 | `LLMEngine.generate()` | 读 `completion_token_ids`，seq 对象被 GC 回收 |

seq 本身就是调度的真相来源，Scheduler 通过直接读写 seq 字段完成所有状态管理，不另外维护副本。

### GPU / NPU / TPU 的区别

| | GPU | NPU | TPU |
|---|---|---|---|
| 全称 | 图形/通用并行处理器 | 神经网络处理器 | 张量处理单元 |
| 谁做 | NVIDIA / AMD / Apple | 各家（Apple、高通、华为）| Google |
| 擅长 | 训练 + 推理，灵活 | 低功耗端侧推理 | 超大模型训练 |
| 灵活性 | 高 | 低（只跑固定格式模型）| 中 |
| 在哪用 | 服务器、PC | 手机、笔记本 | Google 数据中心 |
| nano-vllm | ✅ NVIDIA GPU | ❌ | ❌ |

M1 Pro 内部三者都有：CPU（10核）负责逻辑、GPU（16核）负责并行计算、Neural Engine（16核，即 NPU）负责低功耗 AI 推理（Siri / Face ID）。

---

## 模块 2：引擎主循环（`engine/llm_engine.py`）

### 职责

LLMEngine 是协调者，不做计算，只驱动循环、传递数据、收集结果。

### 初始化关键点

| 操作 | 原因 |
|------|------|
| `mp.get_context("spawn")` | CUDA 不支持 fork，子进程必须用 spawn 重新启动 |
| rank 0 在主进程，rank 1+ 在子进程 | 主进程承担 tokenize / 调度 / 收集输出等非 GPU 工作 |
| `config.eos` 在 tokenizer 加载后才填入 | 模型路径确定了才知道 eos token id |
| `atexit.register(self.exit)` | 程序退出时自动清理子进程 |

### step() —— 系统心跳

```
scheduler.schedule()     → 决定这批处理哪些 seq，prefill 还是 decode
model_runner.call("run") → GPU forward，返回 next token ids
scheduler.postprocess()  → 追加 token，标记完成的 seq
```

### generate() 主循环

- 所有 prompt 先全部 `add_request()`，再驱动 `while` 循环
- 用 `seq_id` 做 dict key 收集结果，最后 `sorted` 保证输出顺序 = 输入顺序
- `num_tokens` 正数 = prefill 吞吐，负数 = decode 吞吐（用于进度条显示）

---

## 模块 3：请求生命周期（`engine/sequence.py`）

### 状态机

```
WAITING ──► RUNNING ──► FINISHED（单向，不可逆）
```

### 关键字段

| 字段 | 含义 | 是否变化 |
|------|------|----------|
| `seq_id` | 全局唯一编号（类级自增计数器）| 不变 |
| `token_ids` | prompt + 生成的所有 token | 每步 decode +1 |
| `num_prompt_tokens` | prompt 长度分界线 | 不变 |
| `num_tokens` | 当前总长度 | 每步 +1 |
| `num_cached_tokens` | 前缀缓存命中的 token 数 | prefill 时确定 |
| `block_table` | 逻辑块→物理显存块的映射 | 随生成动态扩展 |
| `last_token` | 最新生成的 token（序列化优化用）| 每步更新 |

### block 分块示意（block_size=256）

```
600 个 token：
  块 0: token[0:256]   → 256 个（满块）
  块 1: token[256:512] → 256 个（满块）
  块 2: token[512:600] → 88 个（不满，last_block_num_tokens=88）
  block_table = [PhysBlock#7, PhysBlock#3, PhysBlock#11]
```

### 序列化优化（`__getstate__` / `__setstate__`）

多进程传输时：
- **Prefill**：传完整 `token_ids`（GPU 需要全部 prompt token）
- **Decode**：只传 `last_token`（GPU 只需要上一步生成的 1 个 token）

避免每步 decode 都传几百个 token，进程间通信量恒定为 1 个 int/条请求。

### 生命周期总览

```
LLMEngine.add_request()  → 创建 Sequence
Scheduler                → 改 status，填 block_table
Scheduler.postprocess()  → append_token()，标记 FINISHED
generate()               → 收集 completion_token_ids，丢弃 Sequence
```

---

## 模块 4：调度逻辑（`engine/scheduler.py`）

### 两个队列

- `waiting`：deque，新请求入队，等待显存分配
- `running`：deque，已分配显存、正在推理的请求

### schedule() 决策树

```
waiting 非空 且 能装下新请求？
    是 → prefill 批次（一次尽量多装）→ 直接返回，不做 decode
    否 → decode 批次
            某条请求显存不够追加？
                是 → 踢掉 running 队尾（生成最少，重做代价最低）
                否 → 正常推进
```

**prefill 停止条件（满足任一即停）：**
1. `num_batched_tokens + len(seq) > max_num_batched_tokens`（token 总数超限）
2. `not block_manager.can_allocate(seq)`（显存不够分块）

### preempt()：抢占

被踢请求：`status → WAITING`，显存释放，插回 waiting **队头**（优先下次重新 prefill）。
代价：KV Cache 丢失，下次从头 prefill。

### postprocess() 结束条件

满足任一即标记 FINISHED，释放显存，移出 running：
- 生成了 EOS token（且 `ignore_eos=False`）
- `num_completion_tokens == max_tokens`

### 设计要点

| 决策 | 原因 |
|------|------|
| prefill 优先 | 新请求尽快开始，减少等待延迟 |
| 踢队尾 | 生成最少，重做代价最低 |
| deque 而非 list | 两端 O(1)，appendleft/pop 高效 |
| prefill 计 token 数，decode 计 seq 数 | prefill 受计算量瓶颈，decode 受并发数瓶颈 |

### prefill 受计算量限制，decode 受并发数限制

- **Prefill（计算密集）**：token 数翻倍，计算量翻 4 倍（平方关系）。CUDA core 满载，瓶颈是算力 → 用 `max_num_batched_tokens` 限制总 token 数
- **Decode（访存密集）**：每步每条请求只算 1 个 token，但要读取全部历史 KV Cache。并发请求越多，显存带宽压力越大，CUDA core 大量空转 → 用 `max_num_seqs` 限制并发数

### nano-vllm vs 真实 vLLM

nano-vllm 保留了 vLLM 三个核心创新（PagedAttention / 前缀缓存 / 连续批处理），砍掉了所有生产复杂度：

| 维度 | nano-vllm | 真实 vLLM |
|------|-----------|-----------|
| prefill 阻塞 decode | 完全阻塞 | Chunked Prefill：切块交替执行，decode 延迟平滑 |
| 抢占方式 | 仅 recompute（丢 KV Cache 重算）| recompute + swap（换出到 CPU 内存）|
| 模型支持 | 仅 Qwen3 | 200+ 种 |
| 量化 | 无 | AWQ/GPTQ/FP8/INT4 |
| 并行策略 | 仅张量并行 | TP + 流水线并行 + 专家并行 |
| 服务形式 | 离线批量 | 在线 HTTP 服务，流式输出，动态接入 |
| 代码规模 | ~1,200 行 | 100,000+ 行 |

---

## 模块 5：KV Cache 管理（`engine/block_manager.py`）

### Block 字段

| 字段 | 含义 |
|------|------|
| `block_id` | 物理块编号 |
| `ref_count` | 共享引用计数，降到 0 才真正释放 |
| `hash` | 该块 token 序列的链式哈希，-1 表示未登记（不满块或新块）|
| `token_ids` | 该块存的 token 列表，用于哈希碰撞校验 |

### BlockManager 核心数据结构

| 结构 | 用途 |
|------|------|
| `blocks[i]` | 全部物理块的状态 |
| `hash_to_block_id` | 前缀缓存目录：哈希 → 块编号 |
| `free_block_ids` | 空闲块队列（deque，FIFO）|
| `used_block_ids` | 当前被引用的块（set）|

### 链式哈希

```
块0 hash = H(tokens[0:256])
块1 hash = H(块0.hash + tokens[256:512])
块2 hash = H(块1.hash + tokens[512:768])
```

混入前块哈希，防止不同前缀的相同内容错误命中缓存。只对满块（256 token）计哈希，不满块内容还在增长，哈希不稳定。

**为什么不同前缀不能共享同一块 KV Cache？**

KV Cache 存的是"这个 token 在这个上下文中的表示"，不是 token 本身。RoPE 位置编码让 K、V 向量随位置变化——同样的 token 出现在位置 256 和位置 512，K、V 值完全不同。前缀不同 → 位置和上下文不同 → K、V 值不同 → 不能复用。

反例：序列 A 前缀是垃圾内容，序列 B 前缀是 system prompt，两者第 1 块 token 内容相同。若只对块本身哈希，B 会拿到 A 的 KV Cache，但 A 的 K、V 是在位置 256+ 的垃圾上下文里算的，B 拿来用会导致 attention 计算错误，生成结果乱掉。链式哈希保证只有"从头到这一块所有 token 完全相同"时哈希才相同，复用才安全。

### allocate() 前缀缓存逻辑

```
对 seq 的每一块：
  命中（hash 在目录 且 token 内容一致）→ ref_count+1，跳过 GPU 计算
  未命中 → 取空闲块，正常计算，顺手登记哈希
一旦某块未命中，后续所有块强制未命中（cache_miss 标志）
```

### may_append() 三种情况

| `len(seq) % block_size` | 含义 | 操作 |
|--------------------------|------|------|
| `== 1` | 新块第一个 token | 取空闲块，追加到 block_table |
| `== 0` | 刚填满一块 | 封存，计算并登记哈希（为后续 seq 提供缓存）|
| 其他 | token 落在块中间 | 什么都不做 |

### vLLM 的链式哈希对比

vLLM 同样是链式哈希（Merkle Tree），但更复杂：

| 维度 | nano-vllm | vLLM |
|------|-----------|------|
| 哈希算法 | xxhash64（固定）| SHA256（默认）/ xxhash128 / 可配置 |
| 额外输入 | 无 | extra_keys：LoRA ID、多模态哈希、cache salt |
| 序列化 | 直接拼二进制 | pickle / CBOR |
| 安全性 | 非加密 | SHA256 抵御碰撞和侧信道攻击 |

vLLM 用 SHA256 的原因：多租户场景需要 cache salt（随机盐）防止跨用户缓存探测攻击，xxhash 不够安全。extra_keys 让不同 LoRA 权重、不同模态输入的 KV Cache 自动隔离，nano-vllm 无此需求所以没有。

### KV Cache 管理图解

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BlockManager 全局数据结构                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  free_block_ids (deque)          used_block_ids (set)                        ║
║  ┌──┬──┬──┬──┬──┐               { 0, 1, 3 }                                 ║
║  │4 │5 │6 │7 │8 │                                                            ║
║  └──┴──┴──┴──┴──┘                                                            ║
║       空闲，可分配                    已占用                                   ║
║                                                                              ║
║  hash_to_block_id (dict)         blocks[i] (list)                           ║
║  ┌──────────────┬────────┐       ┌────┬──────────┬──────┬───────────────┐   ║
║  │ 0xABCD...    │   0    │       │ id │ ref_count│ hash │  token_ids    │   ║
║  │ 0x1234...    │   1    │       ├────┼──────────┼──────┼───────────────┤   ║
║  │ 0x5678...    │   3    │       │  0 │    2     │0xABCD│ [1,2,3...256] │←共享║
║  └──────────────┴────────┘       │  1 │    1     │0x1234│ [257...512]   │   ║
║    前缀缓存目录                    │  2 │    0     │  -1  │ []            │←空闲║
║                                  │  3 │    1     │0x5678│ [100...355]   │   ║
║                                  │  4 │    0     │  -1  │ []            │←空闲║
║                                  └────┴──────────┴──────┴───────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║              场景：两条请求共享 system prompt（前缀缓存命中）                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  seq_A: [system(256)] [问题A(80)]   seq_B: [system(256)] [问题B(100)]        ║
║          ↑↑↑↑↑↑↑↑↑↑                         ↑↑↑↑↑↑↑↑↑↑                    ║
║                        完全相同                                               ║
║                                                                              ║
║  seq_A prefill（先）：system 块 cache miss → 分配 Blk#0，GPU 算，登记哈希     ║
║                        问题A 块不满 → 分配 Blk#1，GPU 算                      ║
║  seq_A.block_table = [0, 1]，num_cached_tokens = 0                          ║
║                                                                              ║
║  seq_B prefill（后）：system 块 hash=0xABCD → HIT！直接复用 Blk#0            ║
║                        Blk#0.ref_count: 1→2                                  ║
║                        问题B 块不满 → 分配 Blk#3，GPU 只算 100 token          ║
║  seq_B.block_table = [0, 3]，num_cached_tokens = 256                        ║
║                                                                              ║
║  显存：                                                                       ║
║  Blk#0 ████████████████  seq_A + seq_B 共享，ref=2                          ║
║  Blk#1 ████████████      seq_A 独占，ref=1                                   ║
║  Blk#3 ████████          seq_B 独占，ref=1                                   ║
║  Blk#2 ░░░░░░░░░░░░░░░   空闲（hash 保留，可被后续命中）                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                    decode 阶段：may_append 的三个时机                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  num_tokens: ...254  255  256  257  258  ...  511  512  513  ...             ║
║                           ↑              ↑         ↑                        ║
║                      % 256==0       % 256==0   % 256==1                     ║
║                      封存登记哈希    封存登记哈希   开新块                     ║
║                                                                              ║
║  token 257（% 256 == 1）→ 开新块：                                           ║
║  block_table: [ Blk#0(封存✓) ]  →  [ Blk#0, Blk#X ]                        ║
║                                                                              ║
║  token 512（% 256 == 0）→ 封存：                                             ║
║  Blk#X 填满，计算链式哈希并登记，后续 seq 可命中此块                          ║
║                                                                              ║
║  其他时刻（落在块中间）→ 什么都不做                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                    deallocate：释放但不清空哈希                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  seq_A 完成，deallocate(seq_A)：                                             ║
║    Blk#0: ref 2→1（seq_B 还用，不释放）                                      ║
║    Blk#1: ref 1→0 → 放回 free，但 hash/token_ids 保留                        ║
║                                                                              ║
║  seq_C 来了，前缀和 seq_A 一样：                                              ║
║    块0：命中 Blk#0（ref 1→2）                                                ║
║    块1：hash 查到 block_id=1，Blk#1 在 free 里 → 重新激活，不用重算           ║
║                                                                              ║
║  若 Blk#1 被抢走（free 被新任务取走）→ reset()，hash 清空 → cache miss，正常分配║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### num_cached_tokens 的三处用途

| 位置 | 用法 | 作用 |
|------|------|------|
| `scheduler.py:35` | `len(seq) - seq.num_cached_tokens` | 统计本批次真实计算量，缓存命中的 token 不计入 |
| `model_runner.py:137-139` | `seq[num_cached_tokens:]` / `seqlen_q = seqlen - num_cached_tokens` | **GPU 直接跳过缓存 token**，只算新部分的 Q；K/V 仍包含全部（Flash Attention varlen 特性）|
| `sequence.py:55` | `num_cached_blocks` property | 派生缓存块数，预留扩展用 |

关键：没有这个字段，GPU 仍会收到全部 token，前缀缓存等于白做。它是调度层和 GPU 执行层之间的桥梁。

**用途二详解（Q/K/V 与 Attention）：**

- Q（Query）= 当前 token 在问"我需要什么信息"
- K（Key）= 每个历史 token 在说"我是什么类型的信息"
- V（Value）= 每个历史 token 的实际内容
- Attention = 用 Q 匹配所有 K 得到权重，对 V 加权求和

缓存命中时：已缓存 token 的 K/V 已在显存，只需计算新 token 的 Q/K/V。Flash Attention varlen 模式支持 `seqlen_q ≠ seqlen_k`：

```
有缓存（命中 256，剩 80 新 token）：
  seqlen_q = 80   ← Q 只算新 token
  seqlen_k = 336  ← K 包含全部（含缓存）
  节省：Q/K/V 计算量 -76%，Attention 点积 -52%
```

system prompt 越长、命中率越高，收益越大（命中 95% 时可节省 90% 计算量）。

### 释放策略

`deallocate()` 只做 `ref_count--`，降到 0 才放回空闲队列，**但 hash/token_ids 保留**。空闲块仍可被后续 seq 命中缓存，被复用时才在 `reset()` 里清空。

---

## 模块 8：模型架构 + 工具

### utils/context.py——全局黑板

```
ModelRunner → set_context(is_prefill, cu_seqlens, slot_mapping, ...)
每层 Attention/LMHead → get_context() 直接读取
forward 结束 → reset_context() 清空
```

不通过函数参数层层传递，保持模型代码干净。

### utils/loader.py——权重名称翻译

权重文件命名（HuggingFace）和模型结构命名（nano-vllm）不一致：

| 文件中 | 代码中 | 原因 |
|--------|--------|------|
| `q_proj` / `k_proj` / `v_proj` | `qkv_proj`（shard_id=q/k/v）| Q/K/V 合并为一次 matmul |
| `gate_proj` / `up_proj` | `gate_up_proj`（shard_id=0/1）| FFN 升维合并 |

`packed_modules_mapping` 定义翻译规则，`weight_loader` 负责写入合并矩阵的正确偏移位置。

### models/qwen3.py——组装结构

四层嵌套：`Qwen3ForCausalLM → Qwen3Model → Qwen3DecoderLayer × 28 → Qwen3Attention + Qwen3MLP`

- `forward` 和 `compute_logits` 分开：CUDA Graph 只录制 `forward`（形状可固定），LM Head 单独调用
- 残差在层间传递（`hidden_states, residual` 对），延迟到 `add_rms_forward` 里融合，减少显存读写
- Qwen3 特有：`q_norm` / `k_norm`（QKNorm），没有 bias 时对 Q/K 额外做 RMSNorm

---

## 关键概念

### Temperature（采样温度）

控制生成文本的随机程度，核心操作是 **logits ÷ temperature**（`layers/sampler.py:12`）：

```python
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
probs = torch.softmax(logits, dim=-1)
# Gumbel-max 采样（等价于 multinomial，更高效）
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

| 值 | 效果 |
|----|------|
| 接近 0 | 分布尖锐，输出接近确定性（greedy） |
| 1.0 | 使用模型原始概率，不做修改 |
| > 1.0 | 分布平坦，输出更随机、更有创意 |

nano-vllm 通过 `assert temperature > 1e-10` 强制禁止 greedy 采样（`sampling_params.py:11`），避免数值溢出。

### Prefill vs Decode

- **Prefill**：一次性处理全部 prompt tokens，生成初始 KV Cache
- **Decode**：每步只生成一个新 token，复用已有 KV Cache

### KV Cache 块

- 每块 256 tokens，逻辑块映射到物理 GPU 内存
- 前缀缓存：对 token 序列哈希，相同前缀复用已有 KV 块（`engine/block_manager.py`）

### 张量并行

- 多 GPU 按列/行切分权重矩阵
- rank 0 负责调度和通信，其余 rank 只执行计算

### Attention 类型与 Flash Attention 模式

**Attention 数学结构演化：**

| 类型 | Q头数 | K/V头数 | 特点 |
|------|-------|---------|------|
| MHA | H | H | 原始多头，表达力强，KV Cache 大 |
| MQA | H | 1 | K/V 共享，KV Cache 缩小 H 倍 |
| GQA | H | G（1<G<H）| 折中，主流（Qwen3/LLaMA3/Gemma）|
| Cross Attention | — | — | Q/K/V 来自不同序列，用于编解码器 |

**Flash Attention 三种调用模式：**

| 模式 | 函数 | 用途 |
|------|------|------|
| 标准 | `flash_attn_func` | 训练，序列等长 |
| 变长 | `flash_attn_varlen_func` | prefill，序列不等长，用 cu_seqlens 标界 |
| KV Cache | `flash_attn_with_kvcache` | decode，Q 只有 1 个新 token |

Flash Attention 不改变数学定义，只改变计算方式（tile 分块，SRAM 计算），显存从 O(n²) 降到 O(n)，速度快 2-4 倍。

### CUDA Graph

录制一次 decode forward pass 的所有 GPU kernel，之后每步直接回放，CPU 只发一条命令，跳过几百次 kernel launch 的调度开销。

- **约束**：tensor 形状必须固定 → nano-vllm 预录 [1,2,4,8,16,32,...] 多份图，运行时向上取整匹配
- **传数据**：回放前原地修改录制时的 tensor 值（插槽机制）
- **性能**：decode 阶段约提升 40%，batch size 越小收益越大
- **prefill 不用**：每次 token 总数不同，形状不固定，无法录图
- `enforce_eager=True` 禁用，走普通 PyTorch forward

### nano-vllm 移植到国产 GPU

需要改造四处（难度递增）：

| 依赖 | 改造方式 |
|------|----------|
| NCCL → 昇腾HCCL/摩尔线程MCCL | 几行，改 `init_process_group` 参数 |
| `torch.cuda.*` → `torch.npu.*` / `torch.musa.*` | 几十行，API 基本一一对应 |
| Flash Attention → 平台原生 Attention | 中等，需要写适配层 |
| Triton kernel → 平台原生算子 | 困难，可先用 PyTorch 原生替换（性能损失 20-40%）|
| CUDA Graph | 直接禁用（`enforce_eager=True`），国产 GPU 暂无等价功能 |

### vLLM 对其他硬件的支持

vLLM 有 Platform 插件架构，官方支持 NVIDIA/AMD/AWS Neuron/Google TPU/Intel/CPU，社区维护华为昇腾（vllm-ascend）和摩尔线程。新硬件只需实现 `Platform` 接口，上层代码不变——这是 nano-vllm（写死 CUDA）和 vLLM（硬件抽象层）的核心架构差异之一。

---

## 进度

- [x] 模块 0：让项目跑起来
- [x] 模块 1：入口与配置（`llm.py` / `config.py` / `sampling_params.py`）
- [x] 模块 2：引擎主循环（`engine/llm_engine.py`）
- [x] 模块 3：请求生命周期（`engine/sequence.py`）
- [x] 模块 4：调度逻辑（`engine/scheduler.py`）
- [x] 模块 5：KV Cache 管理（`engine/block_manager.py`）
- [x] 模块 6：GPU 执行（`engine/model_runner.py`）
- [x] 模块 7：神经网络层（`layers/`）

### 各层通俗说明

| 层 | 类比 | 实际作用 |
|---|---|---|
| `VocabParallelEmbedding` | 查档案：token id → 详细档案卡 | token 编号 → 向量（多卡按词表范围切分）|
| `RMSNorm` | 调音量：防止数字爆炸 | 把向量缩放到合理范围，`add_rms_forward` 融合加残差操作 |
| `ColumnParallelLinear` | 重新整理档案（按输出切分）| 矩阵乘，各卡算部分输出，无需通信 |
| `RowParallelLinear` | 合并档案（按输入切分）| 矩阵乘，AllReduce 汇总，一次通信 |
| `QKVParallelLinear` | 从档案提取 Q/K/V 三份摘要 | 一次 matmul 同时产出 Q/K/V，权重加载时按 q/k/v 分片写入 |
| `RotaryEmbedding` | 给档案打序号 | 按位置旋转 Q/K 向量，让 Attention 感知相对位置 |
| `Attention` | 多轮互相参考 | 存 K/V 入缓存，prefill 用 varlen，decode 用 with_kvcache |
| `SiluAndMul` | 带开关的深加工 | FFN 激活：切两半，一半做门控，一半做内容，相乘 |
| `ParallelLMHead` | 对照词库打分 | prefill 只保留最后一个 token 再算 logits，省大量计算 |
| `Sampler` | 按概率抽签 | 除以 temperature，Gumbel-max 采样（比 multinomial 快）|

**为什么自己实现这些层（不用 HuggingFace）：**
需要掌控 KV Cache 写入时机、张量并行切分方式、prefill/decode 两种模式切换，HuggingFace 的标准实现不支持这些推理引擎专属的优化。
- [x] 模块 8：模型架构 + 工具（`models/qwen3.py` / `utils/`）
