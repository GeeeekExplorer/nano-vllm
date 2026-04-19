import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

# Knowledge：
# Model Parallel 是“大类”，常见还包括：
# - Pipeline Parallel（PP） ：按 layer 切到不同 GPU，靠 pipeline 跑。
# - Expert Parallel（MoE EP） ：按 expert 分到不同 GPU。
# - Tensor Parallel（TP） ：把同一层里的矩阵（权重/激活）按维度切分到不同 GPU，上下游靠 all-reduce / all-gather / reduce-scatter 等集合通信拼起来。

# NCCL(NVIDIA Collective Communication Library)s NVIDIA提供的GPU间搞性能通信库
# - 控制面使用shm/event/queue，数据面(tensor)使用NCCL
# -- 控制面 ：把“本轮要跑哪些 seq、它们的 block_table、slot_mapping 等元信息”发给别的进程。这在 nano-vllm 里确实像“rank0 广播指令”（用 shared memory pickle），属于“任务分发”。
# -- 计算面（算子级） ：为了完成 TP 线性层/embedding/head 等计算，必须做 all_reduce/all_gather 等。这些是 NCCL collective ，是数学正确性需要的。
# - todo：看一下modelrunner中的cpu和gpu的通信

# Collective communication 集合通信 = 一组 rank 共同参与的一次通信原语 。常见 collective 原语（最重要的几个）：
# - broadcast ：一个 rank 的数据发给所有 rank（这确实是“单点发给所有人”，但仍然属于 collective，因为所有 rank 必须一起参与这次广播的调用）
# - all-reduce ：每个 rank 都有一个张量，最后每个 rank 都得到 “按元素求和/求平均/最大值”等结果
# - all-gather ：每个 rank 都有一段分片，最后每个 rank 都拿到“所有分片拼起来”的完整张量
# - reduce-scatter ：先 all-reduce 再把结果切片分给各 rank（训练里很常见）
# - all-to-all ：每个 rank 给每个 rank 发不同的分片（MoE 常见）

# todo：这个工程需要加锁吗
class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = [] # 保存子进程对象，方便退出时 join。
        # 是 tensor parallel 分布式进程 rank，rank0是主控进程里的 ModelRunner负责接收 LLMEngine 调用、广播命令给其他 rank、汇总主流程，其他rank常驻 loop() 等待 rank0 下发命令
        self.events = [] # 保存跨进程同步事件（给 TP rank0 通知 rank>0 “来读共享内存”）
        # 多卡并行时，每张 GPU 对应一个进程/一个“参与者”，分布式库会给每个参与者一个编号，叫 rank （0,1,2,...） 
        # todo：spawn启动方式是什么（为啥modelrunner中没有__main__）? mp的语法是什么？
        ctx = mp.get_context("spawn") # 选择 spawn 启动方式（在 CUDA/多进程下更安全；也是跨平台常见选择）
        for i in range(1, config.tensor_parallel_size): # 启动 TP 的其他 rank, tensor_parallel_size = 1
            event = ctx.Event() # 每个子进程一个 Event
            process = ctx.Process(target=ModelRunner, args=(config, i, event)) # 创建子进程，子进程启动后会在 ModelRunner.__init__ 里进入 loop() 常驻等待命令
            process.start() # todo：启动进程？
            self.ps.append(process)
            self.events.append(event)
        # rank0 在主进程里直接创建一个 ModelRunner ，它既负责本 rank 的执行，也负责把命令/参数写入 shared memory 广播给其他 rank
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True) # 加载 tokenizer（给 add_request 编码、给最终 outputs 解码）
        config.eos = self.tokenizer.eos_token_id # 把 eos id 写回 config，供 scheduler 判断停止条件
        self.scheduler = Scheduler(config) # 创建调度器，内部也会创建 BlockManager （KV block 池）。
        # todo：atexit是什么？
        atexit.register(self.exit) # Python 进程退出时自动调用 exit() ，确保子进程回收。

    def exit(self):
        # todo: 是注册了什么方法吗，这是什么玩法？
        self.model_runner.call("exit")  # 通知各 rank 执行 ModelRunner.exit() （销毁 process group、释放 cudagraph pool、unlink shared memory 等
        del self.model_runner # 释放本进程 rank0 的 runner
        for p in self.ps:
            p.join()

    # todo：什么时候会出现prompt是str，什么时候是list[int]
    # 将选中的sequence进入waiting队列
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    # 一次 engine 迭代（调度 → 执行 → 更新状态 → 产出完成的序列）    
    def step(self):
        seqs, is_prefill = self.scheduler.schedule() # 返回本轮要跑的序列列表 seqs, is_prefill 表示本轮是 prefill 还是 decode（注意：这个实现一轮只做一种）
        token_ids = self.model_runner.call("run", seqs, is_prefill) # 实际 GPU forward + 采样发生在 ModelRunner.run；TP 时 call 还会把 seqs pickle 后广播给其他 rank
        self.scheduler.postprocess(seqs, token_ids) # 把（采样）输出的新 token append 到每个 seq，判断 eos/max_tokens，结束则释放 KV blocks，从 running 队列移除
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished] # 返回完成推理的seq的completion token ids
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs) # 这是给 tqdm 展示吞吐用的：prefill 用本轮处理的 token 数；decode 用本轮 seq 数（取负数作区分）
        return outputs, num_tokens

    def is_finished(self):
        # 判断一次offline batching是否结束
        return self.scheduler.is_finished() # scheduler waiting/running 都空就结束

    # 实现offline batching
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) # 打印进度条
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts) # 每条请求一个params， 如果只有一个params则复制到各个请求
        
        # todo：看一下v0和v1的add_request，zip的作用？
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        outputs = {} # 存最终 completion token ids
        prefill_throughput = decode_throughput = 0. # 用于展示的吞吐
        while not self.is_finished():
            t = perf_counter() # 计时，todo：为啥使用这个记时
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",   # todo：算的是TTFT吗？
                    "Decode": f"{int(decode_throughput)}tok/s",     # todo：算的是TPOT吗？
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1) # 进度条+1
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())] # 按 seq_id 排序，把 dict 还原成 list（保证输出顺序与输入一致
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs] # detokenize
        if use_tqdm:
            pbar.close()
        return outputs

    # todo：实现online serving