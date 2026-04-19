import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

# OOM = Out Of Memory
# 推理里最常见的 OOM 根因：
# - KV cache 占用随“并发序列数 × 序列长度”线性增长 （最常见）
# - prefill 激活峰值 （长 prompt + 大 batch 的一次性峰值）
# - 内存碎片 + allocator 缓存导致“看起来还有空闲，但无法分配连续块” todo：来个sample

# python:
# event：multiprocessing 的 Event 是一个 跨进程同步原语 ，可以理解成一个共享的布尔标志位，同步的是“进程/CPU 控制流”
# - 初始状态： unset（False）
# - event.set() ：把标志置为 True，并唤醒所有在等待它的进程
# - event.wait() ：如果标志是 False 就阻塞；直到被 set 才返回
# - event.clear() ：把标志重新置为 False（让下一轮 wait 继续阻塞）

# dist.barrier() 是 分布式同步点 （也是一种 collective）：
# - 同一个 process group 里的所有 rank 都执行到 barrier 时，大家才一起继续往下走。
# - 如果某个 rank 没到 barrier，其他到达 barrier 的 rank 会一直等它。

# torch.cuda.synchronize()：阻塞当前进程，直到 当前 GPU 上此前提交的所有 CUDA 工作都完成
# - 避免销毁 NCCL process group 或释放 cudagraph 相关资源时，GPU 上还有未完成的工作导致
# - 很多 CUDA 操作是 异步 的：你在 Python 里调用了某个算子，函数可能很快返回，但 GPU kernel 仍在后台跑

# SharedMemory(...):
# - create：共享内存段的名字（字符串），相当于“全局标识符”。
# -  创建时 ：你给一个 name，让别的进程用同名来 attach。
# -  attach 时 ：只需要提供 name，就能连接到已存在的共享内存段。
# - create=True ：创建一个新的共享内存段（要求此 name 还不存在）。
# -  create=False （默认）：表示“我不创建，我要连接已有的”。
# - size: 共享内存的字节大小。仅在 create=True 时必须提供 （因为创建者要决定开多大）。

# warmup使用的api：
# torch.cuda.mem_get_info() ：读“系统层面”的显存 free/total（全局）
# 作用 ：返回当前 device 的 (free_bytes, total_bytes) ，是 CUDA runtime 报告的显存信息（更接近“系统视角”）
# 作用范围 ：
# - 和“当前进程”有关（因为其他进程也会占显存，会反映在 free 里）
# - 和“当前 GPU”有关（set_device 到哪张卡，就读哪张）
# torch.cuda.memory_stats()：返回一个 dict，里面很多 key
# - key: allocated_bytes.all.current 当前 PyTorch 认为“已分配给 tensor 使用”的字节数（更接近“用户真正占用”）。它不等于显存总占用，因为 PyTorch 还有缓存池、碎片等
# - key: allocated_bytes.all.peak 从上次 reset 以来， allocated_bytes.all.current 的历史峰值。帮助你估算“prefill 峰值会冲到多高”，用来给 KV cache 留余量
# - 作用范围 ：当前进程 + 当前 GPU；不包含其他进程的 allocator 统计
# torch.cuda.reset_peak_memory_stats() ：把 peak 重新计数
# - 把 allocated_bytes.all.peak 等“峰值统计”重置为当前值，从此重新开始统计峰值
# - 作用范围 ：当前进程 + 当前 GPU
# torch.cuda.empty_cache() ：清 PyTorch 的“显存缓存池”，不等于释放张量；作用范围 ：当前进程 + 当前 GPU
# - 作用 ：把 PyTorch CUDA allocator 里“当前没被 tensor 占用、但被缓存起来以便复用的显存块”尽量归还给 CUDA driver。
# - 它不会 ：
# -   释放还在被引用的 tensor 占用的显存，e.g. 模型权重是有 Python 对象引用着的（ self.model 里的参数 tensor），属于“仍然活着的 tensor”，不会被释放
# -   把 allocated_bytes.all.current 清到 0（除非你真的把 tensor 都删了并且没有引用）
# PyTorch caching allocator（CUDA 缓存分配器）可以理解成： PyTorch 在每个进程里自己维护的一套“显存内存池” ，用来加速频繁的 GPU 内存分配/释放。
# - PyTorch 会把释放掉的显存块先留在“缓存池”里，下次再需要类似大小的显存时直接复用，速度快很多。
# - 开多进程（像 nano-vllm 的 TP 多进程），每个进程都有各自的 caching allocator；它们不会共享同一个缓存池。
# - empty_cache() 做的就是：把“缓存池里当前空闲的块”尽量还给 CUDA driver，让系统层面 free 显存变多。

# torch.cuda.set_device(rank)： 解决的是： 用哪张 GPU
# - 设定“当前进程当前线程的默认 CUDA 设备”= cuda:{rank} 。
# - 影响：你后面写 .cuda() 、或不指定 device 的 CUDA 相关操作，会落到这张卡上。
# torch.set_default_device("cuda")：解决的是： 新建张量默认在 CPU 还是 GPU
# - 设定“PyTorch 新建 tensor/parameter 时默认放到哪个 device”。
# - 影响：你写 torch.empty(...) 这种没指定 device 的创建语句时，默认就会在 GPU 上创建，而不是 CPU。

# pin_memory=True 的作用，可以把 CPU 内存分两类理解：（llm说不是零拷贝）
# - 可分页内存（pageable） ：普通 malloc 出来的内存，OS 可以把它换页/移动。
# - 页锁定内存（pinned / page-locked） ：OS 保证这块内存不会被换页，物理地址稳定。（me：不会受虚拟页表影响）
# - pin_memory=True 的主要收益在于 CPU → GPU（H2D）拷贝 ：
# -  GPU 的 DMA 引擎要高效从 CPU 内存直接拷到显存，最好源地址是稳定的；pinned 内存满足这一点。
# -  如果源是普通 pageable 内存，CUDA 往往会先做一次“内部 staging”：把数据先拷到一块 pinned buffer，再从 pinned buffer DMA 到 GPU。这样等于多了一次 CPU 拷贝和额外开销。
# -  DMA走的PCIe总线，单机多卡最常见： CPU 内存 →(PCIe)→ GPU 显存 ，用 GPU/驱动的 DMA 引擎 做数据搬运（避免 CPU 逐字节复制）
# staging（中转缓冲）指的是：当源数据不满足高效 DMA 的条件时，系统会先把数据拷到一个“合适的中转区”，再从中转区搬到目标。
#     如果没有pin_memory=True，CUDA/框架可能会，比“直接 pinned → GPU”多了一次拷贝和额外同步开销。：
#     1. 分配一块 pinned staging buffer
#     2. 先 CPU→CPU 拷贝：pageable → pinned staging
#     3. 再 DMA：pinned staging → GPU
# torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)这个操作执行的是：
# - torch.tensor这时tensor是在内存中（基于torch.set_default_device）的pin_memory部分；然后.cuda是将其传入gpu（触发 H2D 拷贝），然后返回gpu tensor对象；使用non_blocking可以不用等待直接返回
# - 不用care异步返回结果是否执行完成，调度系统会保证“用到之前一定拷完”，同一个 CUDA stream 上，操作是按顺序执行的：先完成拷贝，再执行后续 kernel：
# -  PyTorch 默认会把这些 .cuda(non_blocking=True) 的拷贝排进当前 stream（或内部拷贝 stream，并通过事件做依赖），然后你紧接着用这些 tensor 去做 self.model(...) ，框架会确保依赖正确，否则结果会错。
# -  什么时候需要“显式同步”：要测时间（benchmark）时：需要 torch.cuda.synchronize() 才能得到真实 GPU 时间；所以exit需要等gpu操作执行完了再继续
# 为什么不把这些 tensor 一开始就“直接创建在 GPU”-降低后续维护成本和隐患，直接在GPU的缺点：
# - 额外占显存 （推理服务里很危险）
# - 引入 CPU/GPU device mismatch 的隐蔽 bug（你以为在 CPU，实际在 GPU）
# - 让一些本应在 CPU 上的轻量逻辑（比如调度数据结构、统计、日志）跑到 GPU
# - 工业界里这叫“ CPU 侧构建元数据，GPU 侧消费 ”，尤其在调度/分页 KV 这种逻辑重的场景很常见

# 整体定位
# - ModelRunner 是 执行层（execution） ：把 scheduler 选出的 Sequence 列表变成 GPU 张量输入，设置 attention/head 需要的上下文（context），调用模型 forward + 采样出 token，再把结果返回给 engine。
# - 同时它还承担 TP 多进程的控制面通信 ：rank0 把“本轮要执行的 method + 参数”通过共享内存广播给 rank>0；rank>0 常驻 loop 收命令执行。
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]): # rank0 传的是 list[Event] （给多个子进程），rank>0 传的是单个 Event
        self.config = config
        hf_config = config.hf_config # todo：这个是qwen带的配置？哪来的？
        self.block_size = config.kvcache_block_size # 256，KV block中的slot数量
        self.enforce_eager = config.enforce_eager # False，是否禁用 cudagraph（强制 eager）,todo: 作用？
        self.world_size = config.tensor_parallel_size # 1，TP size
        self.rank = rank # 当前进程 rank
        self.event = event # 通信同步对象

        # 建立 TP 通信组：所有 rank 加入同一个 process group。
        # 用固定端口 localhost TCP 做 rendezvous（非常简化，生产不会这么写死）。todo：没看懂，入参有什么
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)

        torch.cuda.set_device(rank) # rank == GPU id （0→cuda:0，1→cuda:1 ...）
        default_dtype = torch.get_default_dtype() # 保存原默认 dtype，后面恢复, 这个是精度
        torch.set_default_dtype(hf_config.torch_dtype) # 让模型参数按 HF config dtype 创建（比如 bf16/fp16）。
        torch.set_default_device("cuda") # 默认在 CUDA 上创建张量/参数（简化写法） todo: 影响和作用是什么？
        self.model = Qwen3ForCausalLM(hf_config) # 构建模型结构
        load_model(self.model, config.model) # 从目录加载 safetensors 权重到参数
        self.sampler = Sampler() # 创建 sampler（temperature 采样），todo：采样的作用？
        self.warmup_model() # warmup：跑一次最大形状的 prefill（见后面）以触发编译/缓存、稳定显存峰值
        self.allocate_kv_cache() # 根据显存预算计算 num_kvcache_blocks 并分配一大块 KV cache 张量，注入每层 attention（见后面）。todo：打印有多大
        
        if not self.enforce_eager: # 果不是 eager，就捕获 cudagraph（主要用于 decode 小 batch）
            self.capture_cudagraph()

        # todo：之前为什么需要设置到cuda以及dtype 
        # 把 default device/dtype 恢复回 CPU/原 dtype，避免后续误把 CPU 张量建到 GPU 上
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # 控制面/参数广播”， 跨进程共享的一块“CPU 内存” 。shm.buf 是一个 buffer（memoryview） ，本质是“字节数组视图”
                # 前 4 字节作为“header”存长度 n；每次写入的数据长度不同，否则读端不知道“这次该读多少字节”
                # 后面 [4 : 4+n] 才是这次真正的 payload
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20) # 创建（create=true）共享内存段，名字写死 "nanovllm" ，size=1MB（极简）
                dist.barrier() # 等其他 rank ready 再继续
            else:
                # 等rank 0先初始化结束
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close() # 关闭当前进程对这段共享内存的“句柄/映射”, close 不会删除共享内存段本身, 其他进程如果还 attach 着，仍然可以继续用
            dist.barrier() # 等其他rank都close
            if self.rank == 0:
                self.shm.unlink() # rank0 负责 unlink; 删除共享内存段（把它从系统里“注销/移除名字”）,新的进程无法再通过 name attach
        if not self.enforce_eager: # 如果用过 cudagraph，释放 graph 相关缓存。
            del self.graphs, self.graph_pool
        torch.cuda.synchronize() # 确保 CUDA 操作完成后再销毁通信组
        dist.destroy_process_group() # 销毁 NCCL 进程组。

    def loop(self):
        while True:
            method_name, args = self.read_shm() # 阻塞等待 event，读共享内存反序列化出 method + args（sequence中）
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self): # rank > 0才会调用
        assert self.world_size > 1 and self.rank > 0 # 多卡
        self.event.wait() # 等 rank0 通知
        n = int.from_bytes(self.shm.buf[0:4], "little") # 读前 4 字节得到 payload 长度 n；litte表示小端存储
        method_name, *args = pickle.loads(self.shm.buf[4:n+4]) # pickle.loads 反序列化出 [method_name, *args]，args 里如果含 Sequence ，会触发 Sequence.__setstate__
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0 # 只有rank 0调用
        data = pickle.dumps([method_name, *args]) # 序列化，这里会触发 Sequence.__getstate__ ：prefill 传全量 token_ids；decode 只传 last_token（优化跨进程拷贝）
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little") # 把长度写入前 4 字节，再写入 payload；litte表示小端存储
        self.shm.buf[4:n+4] = data
        for event in self.event: # set 所有 events，唤醒各 rank>0
            event.set()

    # 该项目只涉及exit和run
    # 主要是多卡时用于传播控制流
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args) # ！！！rank0先广播 控制流 到共享内存
        method = getattr(self, method_name, None) # 通过反射 getattr 找到同名方法并执行，todo：为什么不直接调用
        return method(*args)


    def warmup_model(self): # todo：用prefill去计算现存的峰值占用？
        torch.cuda.empty_cache() # 清理 PyTorch caching allocator 里“已经空闲、未被任何 tensor 引用”的缓存块（进程级），让它们还给 CUDA driver。
        torch.cuda.reset_peak_memory_stats() # 重置 peak 统计，后面 allocate_kv_cache 会用到 peak/current 来估算可用显存。
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len # 16384 prefill 总 token 上限；4096 一个seq最大的token上限（模型最大上下文长度上限）；
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs) # 构造一个“最坏情况的 prefill batch”：每条都长到 max_model_len，这时可以prefill多少seq。
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)] # 造一批假的 seq（token 全是 0），只为了跑一次最大形状
        self.run(seqs, True) # 跑一次 prefill，让模型/算子完成第一次编译/初始化（flash-attn、torch.compile、CUDA kernel caching 等），同时让显存峰值更接近真实运行。
        torch.cuda.empty_cache() # 清缓存，减少碎片，让后面的 KV cache 分配更稳定。

    # todo：查一下这些cuda的作用和用法
    def allocate_kv_cache(self):
        # 包含层数、head 数、dtype 等信息
        config = self.config
        hf_config = config.hf_config

        free, total = torch.cuda.mem_get_info() # 读 GPU 当前空闲/总显存（字节）
        used = total - free # 当前已用显存

        # todo：如果有个并行的线程会不会破环cuda.memory_stats的结果，下面这是啥？
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"] # 读 PyTorch allocator 的 peak
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"] # 读 PyTorch allocator 的 current

        # todo：看一下megatron是怎么分的！
        num_kv_heads = hf_config.num_key_value_heads // self.world_size # TP 时每张卡只持有一部分 KV heads（这里假设 KV heads 能整除 TP size）
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads) # todo：这是什么？一个head的dim？

        # todo：每个rank都有一个独立的kv cache?
        # - 2 表示 K 和 V
        # - num_hidden_layers ：每层都有 KV
        # - block_size ：一个 block 覆盖多少 token
        # - num_kv_heads * head_dim ：每 token 的 KV 向量大小（每层、每 rank）
        # - torch_dtype.itemsize ：每个元素字节数（fp16=2，bf16=2，fp32=4）
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize # 一个 KV block 在所有层上的总字节数

        # 不能把所有显存都给kv cache，需要留部分给模型和计算的中间结果
        # - peak + current 的直觉：peak 统计包含某些 warmup 时的峰值，当前已经回落，想扣掉峰值多出来的部分，避免过度分配导致 OOM。
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes # 估算在设定的显存利用率下，还能塞下多少 KV blocks
        assert config.num_kvcache_blocks > 0

        # todo：了解kv cache的计算
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"): # 遍历 self.model.modules() ，找到有 k_cache/v_cache 属性的模块（就是自定义 Attention），todo：没有的怎么办
                # 让该层 attention 的 module.k_cache/module.v_cache 指向 self.kv_cache 对应层的 view。
                module.k_cache = self.kv_cache[0, layer_id] # todo：了解作用，model计算中使用吗？
                module.v_cache = self.kv_cache[1, layer_id] # todo：了解作用，model计算中使用吗？
                layer_id += 1

    # 把每条 seq 的 block_table 打包成 batch 张量
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs) # 找到最长上下文
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs] # 整合seqs的物理block_id，不足的部分用 -1 pad 到同长度

        # todo: pin_memory的作用以及异步的作用，他怎么找到拷贝到哪个gpu上（又没有rank输入）？
        # 先在 CPU pinned memory 建 tensor，再异步拷到 GPU
        # pinned memory 能提升 H2D 拷贝效率， non_blocking=True 允许异步
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # 构造 varlen prefill batch + slot_mapping
    # - prefill 的本质：计算 prompt token 的 hidden states，写入 KV cache，为后续 decode 服务
    # - prefix cache 的本质：如果某些整块前缀已经存在 KV cache，就跳过这些 token 的计算与写入 （省时）
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = [] # 本轮要送进模型seqs的 token id
        positions = [] # 每个token的 position id（逻辑id）
        
        # FlashAttention varlen 需要的前缀和数组（告诉它每条序列在拼接后的起止位置），todo：FlashAttention varlen 是什么？
        cu_seqlens_q = [0] 
        cu_seqlens_k = [0]  
        
        # batch 内最大长度，用于 kernel 选择/launch
        max_seqlen_q = 0 
        max_seqlen_k = 0

        slot_mapping = [] # 每个“本轮要写 KV 的 token”应该写到 KV cache 的哪个物理 slot
        block_tables = None # 只有 prefix cache 时才需要 todo：这啥

        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:]) # 如果该seq的prefix cache 命中了一部分前缀（整块），那么这部分不需要再算 K/V，所以本轮只送未缓存部分 token 进模型
            positions.extend(list(range(seq.num_cached_tokens, seqlen))) # 对应seq需要计算token的 position ids（逻辑位置）
            seqlen_q = seqlen - seq.num_cached_tokens # ！！！本轮 query 的长度（要算的部分）
            seqlen_k = seqlen # key/value 的“逻辑长度”（因为注意力要看全上下文）

            # 用前缀和累加，告诉 varlen kernel 每条 seq 的kv计算的边界
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            
            # 更新seq需要计算的最大长度
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:    # warmup 的假 seq 没有分配 block_table，所以不构造 slot_mapping（否则会访问不存在的物理块）
                continue

            for i in range(seq.num_cached_blocks, seq.num_blocks): # 只为“新增/未缓存”的逻辑 blocks 写入 slot
                start = seq.block_table[i] * self.block_size # 物理的slot id
                if i != seq.num_blocks - 1: # 如果不是最后一个 block，写满整个 block（256 slots）
                    end = start + self.block_size
                else:                       # 如果是最后一个 block，只写到 last_block_num_tokens （因为最后一个 block 可能没满）
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end))) # 用于存放每个token的kv值对应全局的物理slot id

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # 需要进行prefix cache
            block_tables = self.prepare_block_tables(seqs)
        
        # todo：这是在干吗？计算的操作在哪？
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    # 构造 decode batch（每条 1 token）+ slot_mapping
    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = [] # 多的这个作用是什么，用于和kv cache中的值做atten？

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        
        # todo：这是在干吗？计算的操作在哪？
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    # 把 temperature 打包
    # 采样策略属于“解码策略|decode strategy”，不属于模型本体；同一个 logits 不同采样策略会生成不同结果
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # 这一步到底怎么执行forward的调度点：根据场景选择 eager 直接跑 或 CUDA Graph replay
    @torch.inference_mode() # 关闭 autograd（不建反向图），减少显存和开销。
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # prefill 形状变化大（varlen），不适合 cudagraph；
        # 用户强制不用图优化；
        # batch 很大时，走 eager 更稳妥（这里是经验阈值）
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 为什么 decode 更适合 graph replay；todo：graph replay是什么？
            # - decode 每步算子很多但每步工作量小，CPU launch 开销占比高。
            # - replay 可以显著减少这部分开销，提升 tokens/s。
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])


    # ！！！prefill和decode的输出都是[seq_len]，prefill输出首token，decode得到next token
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # todo: logits和最终的结果有什么区别，为什么还需呀sampler？ 
        logits = self.run_model(input_ids, positions, is_prefill) # logits 形状在 rank0 侧可以理解为 [bs, vocab]
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        
        reset_context() # 防止下一轮误用旧 context
        return token_ids

    # 预先录制 decode 的 CUDA 执行图，让后续 decode 小 batch 省掉大量 Python/CUDA launch 开销。
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
