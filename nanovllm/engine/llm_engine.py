import atexit # 用来注册程序退出时要自动执行的清理函数（比如释放子进程、显存）
from dataclasses import fields # 用来拿 Config 里定义的所有字段名，方便从 kwargs 里筛选配置参数
from time import perf_counter # 计时函数，用来算吞吐量 tok/s
from tqdm.auto import tqdm # 进度条
from transformers import AutoTokenizer # 自动分词器
import torch.multiprocessing as mp # 用 PyTorch 的多进程接口来做 tensor parallel

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler # 决定每一步哪些序列进入模型
from nanovllm.engine.model_runner import ModelRunner

"""
LLMEngine:

管理多进程 ModelRunner
管理 Scheduler
提供 add_request / step / generate 这些高层接口给用户用
"""

class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)} # 用 fields(Config) 获取 Config 里所有字段（model, max_model_len, ...），取名字组成一个 set
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields} # 遍历 kwargs，只保留 key 落在 config_fields 里的那部分，形成一个 config_kwargs 字典
        config = Config(model, **config_kwargs)
        self.ps = [] # 保存子进程对象列表
        self.events = [] # 保存进程间同步用的 Event 列表
        ctx = mp.get_context("spawn") # 获取多进程上下文，启动方式为 "spawn"
        for i in range(1, config.tensor_parallel_size): # 其他 rank（1~N-1）用子进程来跑
            event = ctx.Event() # 创建一个进程间共享的 Event，用于 rank 0 和该子进程i之间同步
            process = ctx.Process(target=ModelRunner, args=(config, i, event)) # 创建一个子进程对象，target=ModelRunner：子进程启动后会调用 ModelRunner(config, rank=i, event) 构造函数，并在构造函数内部进入循环
            process.start()
            self.ps.append(process) # 收集一下信息
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events) # 这里传入的是 self.events（Event 列表），rank 0 会用共享内存 + 这些 events 跟各子进程通信
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True) # 用config里面tokenizer配置来加载tokenizer，use_fast=True 使用 fast tokenizer（Rust 实现，速度更快）
        config.eos = self.tokenizer.eos_token_id # 把 tokenizer 的 eos_token_id 写回 config
        self.scheduler = Scheduler(config) # scheduler 负责管理所有 Sequence，决定本 step 是 prefill 还是 decode，选择哪些 seq 进入模型等
        atexit.register(self.exit) # 注册一个退出钩子，当进程退出时自动调用 self.exit()

    def exit(self):
        self.model_runner.call("exit") # 告诉 rank 0 的 ModelRunner 执行它的 exit 方法，rank 0 再通过共享内存/Event 通知各 rank>0 的子进程退出
        del self.model_runner # 删除引用，触发析构
        for p in self.ps:
            p.join() # 等待每个子进程结束

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams): # 对外暴露的“添加一个生成请求”的接口，prompt 可以是字符串，也可以是已经 tokenized 的 list[int]
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt) # 如果是字符串，先用 tokenizer 编成 token id 列表
        seq = Sequence(prompt, sampling_params) # 用 prompt token 列表和采样参数构造一个 Sequence 对象
        self.scheduler.add(seq) # 然后把新序列放到scheduler里面等调度

    def step(self):
        seqs, is_prefill = self.scheduler.schedule() # 让 Scheduler 决定本 step 要送进模型的 seqs 和当前的阶段p or d
        token_ids = self.model_runner.call("run", seqs, is_prefill) # 内部会准备 input_ids、positions、KV cache、调用模型、采样出下一步 token；对 rank 0 来说，call("run", ...) 会同时驱动所有 rank0/子进程的模型一起跑
        self.scheduler.postprocess(seqs, token_ids) # 把这一步生成的 token_ids 交给调度器，调度器更新每个 Sequence 内部的 token、状态（是否结束）、KV cache block table 等
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished] # 从本次参与计算的 seqs 中，挑出已经完成的序列，对每个 finished seq，取它的 seq_id 和生成的 completion 部分 completion_token_ids
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs) # 统计吞吐量：prefill 阶段，一次性喂入多个序列的所有 prompt tokens，所以用 sum(len(seq) for seq in seqs)（总 token 数）；decode 阶段，每个 seq 一次只生成 1 个 token，此时 len(seqs) 就是这一步生成 token 的个数；前面加个负号区分 p / d（prefill 用正数，decode 用负数）
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished() # Scheduler 内部会根据队列里是否还有 WAITING / RUNNING 序列来决定是否所有seq生成完


    """
    用户侧的核心 API：
    一次传入多个 prompt（字符串列表或 token id 列表）
    可以传一个 SamplingParams（会广播给所有 prompt），也可以传一个列表（每个 prompt 自己的采样配置）
    use_tqdm 控制是否显示进度条
    """
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) # 创建一个进度条对象，总数是 prompt 个数（每完成一条 seq 更新 1）
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts) # 如果只有一个sampling_params则广播
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp) # 调 add_request 把 prompt 都塞进 Scheduler
        outputs = {} # 用字典记录最终结果，key 是 seq_id，value 是 completion token ids
        prefill_throughput = decode_throughput = 0. # 记录pd两种阶段的吞吐量（tok/s）
        while not self.is_finished(): # 主循环：只要还有没完成的序列，就继续 step
            t = perf_counter() # 记录当前时间，用来计算这一step用了多久
            output, num_tokens = self.step() # 执行一次调度+模型运行+采样，得到本步完成的若干序列及它们的 token_ids，以及本步处理的 token 数
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t) # pd速度
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids # 遍历本步刚刚完成的那些序列，把 seq_id -> completion token_ids 存到 outputs 字典
                if use_tqdm:
                    pbar.update(1) # 每完成一个 seq，进度条前进 1
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())] # 把 outputs 从字典按 seq_id 排序，变成 list，保证多个 prompt 的结果顺序与请求顺序一致
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs] # 把每个结果 token_ids 用 tokenizer 解码成文本 text，最终输出{"text": "...", "token_ids": [...]}
        if use_tqdm:
            pbar.close()
        return outputs
