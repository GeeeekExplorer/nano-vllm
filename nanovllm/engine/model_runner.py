import pickle # 用于序列化和反序列化对象，这里主要用于多进程通信时传递数据
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory # 因为 Python 的 GIL Global Interpreter Lock 限制，多 GPU 推理通常使用多进程实现

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context # 用于管理推理过程中的上下文信息（如 KV Cache 的位置映射）
from nanovllm.utils.loader import load_model


"""
prefill/decode的scheduler调度KVblock----->信息存全局context----->模型算的结果填到context中给的位置
"""
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event # 用于进程间同步的事件对象

        # 初始化分布式环境 (NCCL 后端)
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank) # tcp://localhost:2333是一个 TCP/IP 地址，表示进程组将通过网络进行初始化；其他初始化方式有共享文件系统、环境变量方式等
        torch.cuda.set_device(rank)
        
        default_dtype = torch.get_default_dtype() # 存个torch的default值
        torch.set_default_dtype(hf_config.torch_dtype) # 再set成hfconfig里面的dtype
        torch.set_default_device("cuda")
        
        # 初始化模型并加载权重
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        
        """
        在 nano-vllm（以及 vLLM 等高性能推理框架）中，将 Sampler（采样器）放在 ModelRunner 中而不是集成在 Model（如 Qwen3ForCausalLM）内部是因为：
        1. Model计算密集，而Sampler为决策，计算量小；Model权重恒定，而Sampler参数是动态的 -> 分开易于cuda graph优化等
        2. 分布式中，采样通常只需要在主进程（Rank 0）上进行，分开易于写代码逻辑
        """
        self.sampler = Sampler()
        self.warmup_model() # 预热模型（分配显存等）
        self.allocate_kv_cache() # 分配 KV Cache 显存
        
        # 如果不强制 Eager 模式，则捕获 CUDA Graph 以加速推理
        if not self.enforce_eager:
            self.capture_cudagraph()
            
        # 恢复默认设置，避免影响其他代码
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 多进程通信设置 (SharedMemory)
        # 是 CPU 端的系统内存，用于 进程间通信 (IPC)，而不是 GPU 芯片上的高速缓存
        if self.world_size > 1:
            if rank == 0:
                # 主进程创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier() # 等待所有进程到达
            else:
                dist.barrier()
                # 从进程连接共享内存并进入监听循环
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close() # 关闭当前进程的共享内存句柄
            dist.barrier() # 进程同步
            if self.rank == 0:
                self.shm.unlink() # 主进程销毁共享内存
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self): # 针对子进程
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self): # 针对子进程
        assert self.world_size > 1 and self.rank > 0
        self.event.wait() # 当前进程会在这里暂停执行，直到 self.event 被设置为 True，类似读锁但是是仅仅单向设置的
        n = int.from_bytes(self.shm.buf[0:4], "little") # 读取缓冲区的前 4 个字节，将这 4 个字节按照小端序（Little-endian）转换为一个整数 n
        method_name, *args = pickle.loads(self.shm.buf[4:n+4]) # 从第 4 个字节开始，截取长度为 n 的字节切片，这就是实际的数据内容，再反序列
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args): # 针对主进程
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args]) # 将方法名和参数打包成一个列表，使用 pickle 库将这个列表序列化为二进制字节流（bytes）。这是因为共享内存只能存储原始字节，不能直接存 Python 对象
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data # 和上面反过来
        for event in self.event:
            event.set() # 设置self.event为 True，类似写锁但仅是单向设置的

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0: # 把主进程要调用的方法名和参数写入共享内存，通知所有从进程（Worker）也去执行这个方法
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None) # getattr(object, name[, default]) 是 Python 的内置函数，用于动态地获取对象的属性或方法
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs) # 作最大seq数限制
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)] # 创建一些伪造的序列数据运行一次模型，确保 CUDA context 初始化完成
        self.run(seqs, True) # 目的是为了让 PyTorch 预先分配一些必要的显存，避免后续运行时抖动
        torch.cuda.empty_cache()

    """
    计算剩余显存，尽可能多地分配 KV Cache Block，并将这些 Block 绑定到模型的 Attention 层中，供后续推理使用
    """
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info() # 函数输出byte
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"] # 获取 PyTorch 内存分配器的统计信息：历史峰值已分配内存，通常发生在模型加载或预热（warmup）期间
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"] # torch当前已分配内存
        num_kv_heads = hf_config.num_key_value_heads // self.world_size # 如果是多卡并行（Tensor Parallelism），总的 KV Heads 会被均分到每张卡上
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads) # 计算每个注意力头的维度大小，通常等于 hidden_size / num_attention_heads
        """
        在内存管理（分配）层面，vLLM 采用了“垂直对齐”的策略，即同一个 Block ID 在所有层通用
        这代表当你为某个 Sequence 分配了 物理块 i 时，实际上是同时分配了模型每一层的物理块 i，这样使页表（Block Table）变得非常小且简单
        但是在flashattn中，会通过layerid切片出来每一层的kv来算每一层（利用了 PyTorch 的 View/Slice 机制（零拷贝））
        """
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize # 计算所有attn层的一个 kv cache Block 需要占用多少字节，要乘模型总层数；所以一个块的byte=总层数*每个block的token数*头数*每头hiddendim*一个type占的byte
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes # 注意是在total显存乘的gpu_memory_utilization
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim) # 实际申请显存，这是一个非常巨大的张量，占据了 GPU 剩余的大部分空间；作为属性存在modelrunner里面
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id] # self.kv_cache[0, layer_id] 获取第 layer_id 层的 Key Cache 的引用，而不复制
                module.v_cache = self.kv_cache[1, layer_id] # 01对应前面的2，layerid逐层把刚初始化的kvcache分下来
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]): # 把页表创好并存在GPU显存中，便于kernel读写
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs] # 为了将所有seq的页表打包成一个 Tensor，需要对齐维度，短的序列需要进行 Padding，约定 -1 代表无效块或空指针
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) # pin_memory=True：锁页内存：在 CPU 内存中锁定这块数据，不让操作系统将其交换到磁盘，这允许 GPU 通过 DMA 快速读取数据，加速 CPU 到 GPU 的传输；.cuda(non_blocking=True)：异步传输：将数据搬运到 GPU，且不阻塞 CPU 线程，CPU 可以立即去执行下一行代码
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]): # 为 Prefill 阶段准备输入数据
        input_ids = [] # 存储所有序列展平后的 Token ID
        positions = [] # 一个 Token 是它所属句子中的第几个词
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen))) # 如果 seq.num_cached_tokens > 0，说明这个请求命中了一些前缀缓存（比如多轮对话的上一轮历史）。不需要重新计算这些 Token，只需要把它们算作 Key/Value 的一部分即可
            seqlen_q = seqlen - seq.num_cached_tokens # 本次需要计算的长度
            seqlen_k = seqlen # 总上下文长度 (历史 + 新增)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table: # warmup 阶段可能没有 block_table，跳过
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks): # 遍历该seq分配到的所有block
                start = seq.block_table[i] * self.block_size # 计算该block在全局一维 KV Cache 中的物理起始位置
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens # 如果是最后一个块，它可能只填了一部分
                slot_mapping.extend(list(range(start, end))) # 将该block内所有 Token 对应的在全局kv的token index加入映射表
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]: # 如果总长度 (k) 大于新输入长度 (q)，说明存在历史 Token（即前缀）
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables) # 将这些计算好的元数据保存起来
        return input_ids, positions # 返回不在context里面的

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = [] # 存储当前 Batch 中每个序列最新生成的那个 Token ID
        positions = [] # 存储对应的位置索引
        slot_mapping = [] # 存储该 Token 的 KV Cache 应该写入的物理显存地址
        context_lens = [] # 存储每个序列当前的上下文总长度 (用于 Attention Mask)
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1) # 计算当前这个新 Token 的 KV Cache 应该存到哪里，注意这里 seq.last_block_num_tokens 已经包含了新 Token，所以要减 1 才是新 Token 的索引
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True) # pin_memory=True: 使用锁页内存，加速 CPU 到 GPU 的传输；non_blocking=True: 异步传输，不阻塞 CPU 继续执行后续代码
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs) # 因为 Decode 阶段需要查表去读取之前的 KV Cache，所以必须把页表传进去
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]): # 收集当前 Batch 中每个seq的Temperature
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool): # 决定是使用普通的 PyTorch 模式（Eager Mode）还是高性能的 CUDA Graph 模式来运行模型
        # 判断是否应该使用 Eager Mode
        # 1. is_prefill: Prefill 阶段输入长度变化大，不适合用静态图
        # 2. self.enforce_eager: 用户配置强制不使用 CUDA Graph
        # 3. input_ids.size(0) > 512: 如果 Batch Size 超过了我们录制的最大 Graph (512)，只能回退到普通模式
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA Graph Mode: 针对 Decode 阶段的优化
            bs = input_ids.size(0) # 求N*1 tensor 的 Batch Size
            context = get_context()
            
            # 找到第一个大于等于当前 bs 的档位。例如当前 bs=3，我们会选 bs=4 的 Graph
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            
            graph_vars = self.graph_vars # 把数据 copy 到录制时使用的那些静态 Tensor (graph_vars) 中
            
            # 将当前数据填入静态 buffer 的前 bs 个位置
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            
            # 填充 slot_mapping (KV Cache 物理地址)
            graph_vars["slot_mapping"].fill_(-1) # 先重置为无效值
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            
            graph_vars["context_lens"].zero_() # 填充 context_lens
            graph_vars["context_lens"][:bs] = context.context_lens
            
            # 填充 block_tables
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            graph.replay() # 重放 Graph (Replay)：这一步极快，因为它是一次性向 GPU 发送所有指令，没有 Python 解释器和 CPU 调度的开销
            return self.model.compute_logits(graph_vars["outputs"][:bs]) # 从静态输出 buffer 中取出前 bs 个结果，并计算 Logits

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]: # 外部调用的主入口，它串联了数据准备、模型执行和采样三个步骤
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs) # 根据阶段不同，调用不同的准备函数
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None # 只有主进程 (Rank 0) 需要做采样，其他进程只需要负责计算
        logits = self.run_model(input_ids, positions, is_prefill) # 得到 Logits (未归一化的概率分布)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None # 同样只有主进程需要，选出下一个token id
        reset_context() # 清除全局变量，防止污染下一次推理
        return token_ids


    """
    针对 Decode 阶段（通常是小 Batch、计算量小但启动开销大），使用 CUDA Graph 录制执行流。可以显著减少CPU 发射 Kernel 的开销（Kernel Launch Overhead），提高推理速度。
    为不同的 Batch Size 录制了不同的 Graph。
    """
    @torch.inference_mode() # 告诉 PyTorch 在这个函数执行期间不需要计算梯度，也不需要维护计算图
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512) # 确定 CUDA Graph 支持的最大 Batch Size（seq数量）
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size # 计算一个序列最大可能需要的 KV Cache Block 数量
        # CUDA Graph 录制的是对固定物理内存地址的操作指令，所以开maxbs的空间作为固定，在录制不同 Batch Size 的 Graph 时，会通过切片（Slicing）只使用其中bs那么大的一部分
        input_ids = torch.zeros(max_bs, dtype=torch.int64) # 模型的 Embedding 层（将 Token ID 转为向量）在 PyTorch 中通常要求输入的索引必须是 LongTensor (即 int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32) # 节省显存和带宽用int32
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) # 推理时，如果实际 Batch Size 是 3，我们会向上取整使用 Batch Size 为 4 的 Graph，多余的位置填补 Padding
        self.graphs = {} # self.graphs[bs] = graph 以 Batch Size 为 Key，存入字典
        self.graph_pool = None # 为了节省显存，不同 Graph 可以共享同一块私有显存池

        for bs in reversed(self.graph_bs): # reversed 的作用：从大到小录制。通常最大的 Batch Size 需要最大的显存池，先录制大的，可以让后续小的 Graph 复用这个已经分配好的大内存池，避免反复重新分配或碎片化。
            graph = torch.cuda.CUDAGraph() # 创建一个新的 CUDA Graph 对象实例
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs]) # 将静态分配的张量的切片（Slice）传递给全局上下文
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs]) # 预热，为了让 PyTorch 完成所有必要的懒加载初始化（Lazy Initialization），确保 CUDA 缓存分配到位，避免这些一次性开销被录制进 Graph 中
            with torch.cuda.graph(graph, self.graph_pool): # 进入录制上下文。在此期间，所有 CUDA Kernel 调用不会真正被 GPU 执行，而是被记录到 graph 对象中，with结束说明cpu把录制指令发完了
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None: # 如果是第一次录制（也就是最大的 Batch Size），获取它创建的内存池，赋值给 self.graph_pool，供后续较小的 Graph 复用
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph # 把这个生成好的 Python 对象句柄（Handle）存到字典里（不需要等待 GPU 录制完）
            torch.cuda.synchronize() # CUDA 的执行是异步的，sync确保cpu等gpu录制完
            reset_context() # 清理上下文，为下一次循环做准备

        # 将这些静态分配的张量保存到 self.graph_vars 中，在后续推理（run_model）时，不能创建新的 Tensor 传给 Graph，必须把数据拷贝到这些特定的静态 Tensor 中，然后重放 Graph
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
