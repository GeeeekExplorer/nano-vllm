# Model Runner类职责：
# 1. 加载模型、准备GPU的kv cache
# 2. 接收scheduler送来的sequence的batch
# 3. 计算prefill和decode（包含可选的CUDA Graph重放优化）
# 4. 采样得到token，并将结果写回主进程的执行单元
# 支持tensor parallel

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence    # 数列数据结构
from nanovllm.models.qwen3 import Qwen3ForCausalLM  # LLM模型
from nanovllm.layers.sampler import Sampler         # 采样器
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size  # 可以笼统地理解为分布式中参与计算的GPU总数量，也可以理解成进程的数量
        self.rank = rank   # 在这个GPU组内每个GPU的编号（0~world_size-1）,其中rank0称为主进程
        self.event = event # 用于多进程之间的同步控制

        # 每个GPU(进程)在初始化时都会调用init_process_group，将这world_size个进程组成一个集群
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)  # 将当前进程绑定对应的GPU
        default_dtype = torch.get_default_dtype()  # get当前默认的数据类型，如FP16，FP32等（改了之后可以恢复过来）
        torch.set_default_dtype(hf_config.torch_dtype) # set默认的浮点数类型
        torch.set_default_device("cuda")               # 将默认设备改成GPU
        self.model = Qwen3ForCausalLM(hf_config)       # 实例化模型（空框架）
        load_model(self.model, config.model)           # load预训练权重
        self.sampler = Sampler()                       # 实例化采样器
        self.warmup_model()                            # 执行模型预热（这是一个方法而非变量）
        self.allocate_kv_cache()                       # 分配kv cache缓存（方法）
        # 是否启用CUDA Graph优化
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")                # 恢复默认设备为CPU（之前转成GPU是为了warmup）
        torch.set_default_dtype(default_dtype)         # 恢复默认dtype

        # 若有多个GPU（进程） 则进程之间需要共享信息和同步状态
        if self.world_size > 1:
            if rank == 0: # 若该进程是主进程（负责调度，管理，通信初始化等）
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20) 
                # 创建一个名为nanovllm的shared memory块，大小1MB
                dist.barrier()  # 分布式同步屏障（等所有GPU进程都达到这一点之后，再继续执行）
                # 也就是说必须让主进程创建完shared memory之后，其他进程才能继续执行
            else:
                dist.barrier()  # 其他进程也要barrier
                self.shm = SharedMemory(name="nanovllm") # 连接到主进程刚刚创建的共享内存块
                self.loop() # ？？？
    # 清理函数，任务结束时释放资源，同步进程，销毁通信组
    def exit(self):
        if self.world_size > 1:
            self.shm.close()  # 当前进程不再连接该共享内存
            dist.barrier()    # 同步
            if self.rank == 0: # 若该进程为主进程，删除这个块
                self.shm.unlink()
        # 若使用了CUDA Graph加速，则需要释放相应资源（graphs, graph_pool）
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()  # GPU同步，确保所有GPU都完成了释放graph的操作
        dist.destroy_process_group() # 销毁分布式通信组（对应init_process_group）

    # 子进程的事件循环
    def loop(self):
        while True:
            method_name, args = self.read_shm() # 子进程不断等待主进程通过共享内存发送命令
            self.call(method_name, *args)       # 收到命令后，调用call执行对应的方法
            if method_name == "exit":           # 若命令时exit，则该子进程退出这个无限循环
                break
    
    # 子进程读取shared memory中主进程的命令
    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0 # 限制在多GPU且当前进程非主进程时才能调用
        self.event.wait()                            # 等待事件触发
        n = int.from_bytes(self.shm.buf[0:4], "little")  # 读取shared mem中前4个字节（内容有多长）
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 读取后面的所有内容（n的长度），并反序列化为python对象
        self.event.clear()                                      # 清除事件标志（防止重复触发）
        return method_name, args

    # 主进程写入命令到shared memory中
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0  # 限制在多GPU且当前进程为主进程时才能调用
        data = pickle.dumps([method_name, *args])      # 将命令和数据序列化为字节
        n = len(data)                                  
        self.shm.buf[0:4] = n.to_bytes(4, "little")    # 用前4个字节记录的[method_name, *args]序列化后的长度
        self.shm.buf[4:n+4] = data                     # 将整体写入shared memory
        for event in self.event:                       # 通知所有GPU进程，有任务来了
            event.set()

    # 主/子进程执行计算的接口
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:  # 若该进程时主进程，先将命令和数据写入shm，子进程不动
            self.write_shm(method_name, *args)   
        method = getattr(self, method_name, None)   # 
        return method(*args)  # 无论是主进程还是子进程，都会执行计算 method(*args)

    # 进行模型warmup
    def warmup_model(self):
        torch.cuda.empty_cache()                # 清空GPU缓存
        torch.cuda.reset_peak_memory_stats()    # 重置GPU的峰值显存统计
        # 从配置中读取一次batch的最大token数和模型能处理的最大序列长度
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 计算warmup时的序列的数量
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        # 创建一批全0的序列（序列长度为max_model_len）
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)  # 调用run方法让模型处理这些序列，true 代表prefill阶段
        torch.cuda.empty_cache()  # 再次清空GPU缓存

    # 计算每个kvcache block所需要的显存-->计算当前显存能申请的kvcache block数量-->创建kv cache的大tensor
    # -->分配给对应的模块
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()  # 获取当前GPU信息，共显存容量，和空闲的容量
        used = total - free                      # 已经使用了的显存容量 
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]  # 获取历史峰值显存
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"] # 当前已分配的显存
        # 计算每个GPU下 kv head的数量（tensor并行下 kv head被均分）
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 计算单个kv cache block 占用显存的大小
        # 2(1k1v)*num_hidden_layers（每层都要存）* block size(每个block存储多少个token的kv cache)
        # * 每个GPU的KV head数 * 每个head的隐藏维度 * 每个元素占用的字节数
        # num_hidden_layers：因为现在的大模型都是很多个（Attention+FFN）的串行堆叠，因此kv cache block需要存储每个
        # Transformer block的数据
        # kv_head:在传统的self attention中一个Q对应一个KV，但是目前的GQA等是多个Q对应一个KV，所以现在将以前的attention head叫做kv head了
        # 本质上就是多头注意力机制中的“多头”的概念
        # head_dim: 每个kv head的计算量（这个head有多少维）

        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        # 计算目前可用显存大小，然后除以每个block占用显存的大小==当前显存还能申请多少个block
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0 
        # 申请了 形状为[2, num_hidden_layers, num_kvcache_blocks, block_size, kv_heads, head_dim]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        # 遍历模型中的所有子模块（即transformer中的每一层layer, FFN, Attention）
        # 
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"): # 检查这个模块是否有属性k_cache,v_cache（一般只有Attention层有）
                module.k_cache = self.kv_cache[0, layer_id] # 将对应num_hidden_layer的K缓存分配给该模块
                module.v_cache = self.kv_cache[1, layer_id] # 同理
                layer_id += 1

    # 将一个batch的所有序列的block table拼成一个二维tensor，并将其搬到GPU
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs) # 计算这个batch的序列中，所需kv block的最大值
        # 进行padding，将所有序列的block table的长度一致（为max_len），不足的用-1填充，这样这个batch的block table就可以变成一个矩阵
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 将python list变成tensor，并把内存固定，然后异步的将数据从CPU拷到GPU
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # --------------------------------------------------------

    def prepare_prefill(self, seqs: list[Sequence]):
        # seqs是一个Sequence对象的list，现在要将这个batch中每个序列的信息都提取出来
        input_ids = []   # 存放当前batch需要送入模型的所有token内容
        positions = []   # 存放每个token的位置索引
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0  # 统计batch中query和key的最大长度（用于GPU tensor的分配）
        max_seqlen_k = 0
        slot_mapping = [] # 将token映射到kv cache的位置？
        block_tables = None  # 最终存放每个序列kvcache block的索引表
        # 对batch中的每个序列进行循环
        for seq in seqs:
            seqlen = len(seq)   # 该序列的token数
            # num_cached_tokens: 已经进行kv 缓存的token的数量
            input_ids.extend(seq[seq.num_cached_tokens:]) # 加入当前需要计算kv cache的token
            positions.extend(list(range(seq.num_cached_tokens, seqlen))) # 对应当前token的绝对位置？
            seqlen_q = seqlen - seq.num_cached_tokens   # 计算需要生成query的token数量
            seqlen_k = seqlen                           # key的数量
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q) # 用于FlashAttention的快速索引
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k) # 同上
            max_seqlen_q = max(seqlen_q, max_seqlen_q)  # 更新batch中query和key的最大长度
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        
        input_ids    = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions    = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()  # 禁用梯度计算（节省显存和计算）
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # 若数prefill阶段 or 不使用cuda_graph or bs太大（不适合cuda_graph）时，直接前向计算
        # 调用model((input_ids, positions))的得到hidden_states
        # 再调用model.compute_logits()得到logits
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        # 否则使用CUDA Graph优化
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
            graph.replay()  # 执行cuda graph
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # 整个模型推理的封装（输入序列到生成下一个token）
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 若为prefill阶段，则调用prepare_prefill, 否则调用prepare_decode
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 在主进程中准备采样的temp
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 进行推理生成每个序列的下一个token的logits
        logits = self.run_model(input_ids, positions, is_prefill)
        # 调用sampler对logits进行采样，得到新token
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context() # 清空全局的context
        return token_ids # 返回本次生成的token ID list

    # 使用CUDA Graph捕获模型前向推理过程
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
