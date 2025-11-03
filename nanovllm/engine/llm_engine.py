# 本文件是nano-vllm中最核心的逻辑
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


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # ----------------------------------
        self.ps = []       # 保存创建的子进程对象 
        self.events = []   # 保存同步事件对象，用于主进程与子进程通信
        ctx = mp.get_context("spawn")   # 指定使用spawn启动新进程（更安全，尤其是CUDA）
        # 创建子进程（rank1--rank(world_size-1)）
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()   # 创建一个事件，用于该进程与主进程进行同步
            # 每个子进程都运行一个ModelRunner，将配置，子进程序号和同步事件传入
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()              # 启动子进程
            self.ps.append(process)      # 保存进程
            self.events.append(event)    # 保存同步事件
        self.model_runner = ModelRunner(config, 0, self.events)  # 为主进程（rank0）也创建一个ModelRunner
        # 加载tokenizer，用于将文本转换成token
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

        config.eos = self.tokenizer.eos_token_id # 将eos的token id保存到配置中，用于生成结束判断
        self.scheduler = Scheduler(config) # 创建scheduler,管理生成任务队列和生成step的调度逻辑
        atexit.register(self.exit) # 注册退出函数，python退出时，自动调用LLMEngine.exit()

    def exit(self):
        self.model_runner.call("exit")  # 调用rank0的ModelRunner的exit方法
        # rank0的exit会通过共享内存和事件通知其他rank进程也退出，并释放资源
        del self.model_runner  # 释放modelrunner对象
        # 等到所有子进程结束
        for p in self.ps:
            p.join()

    # 将用户输入的prompt转成tokenID 列表，然后封装成sequence对象送入scheduler等待调度
    # 针对用户的一条prompt，generate时会循环调用这个函数
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):  # 若输入的prompt是字符串，则需要使用tokenizer的encode方法转成token ID列表
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)  # 将输入封装成Sequence对象
        self.scheduler.add(seq) # 将这个序列加入Scheduler，等待调度

    # 单轮生成循环
    def step(self):
        # 得到本轮需要处理的Sequence列表
        seqs, is_prefill = self.scheduler.schedule()  # 也要拿到这一轮是prefill还是decode
        # 调用ModelRunner.run，该方法会
        # 1. 处理输入的tokens（prefill或decode）
        # 2. 调用模型生成logits
        # 3. 使用采样器得到新的token
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 添加新生成的token到completed_token_ids,并更新is_finished的标志
        self.scheduler.postprocess(seqs, token_ids)
        # 从这一轮生成中筛选出已经完成的sequence
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 计算token数量用于性能统计：prefill返回总token数；decode返回-len(seqs)
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        # 为了让一个prompt都有对应的一个sampling params参数对象
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 循环拿到用户输入的一个batch的prompt，在进行tokenizer后装入scheduler
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        # 核心推理循环
        outputs = {} 
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():   # 调用self.scheduler.is_finished()，检查当前所有prompt都完成
            t = perf_counter()   # 开始计时
            # output: [(seq_id_0, [new_token_0], (seq_id_1, [new_token_2]))]
            output, num_tokens = self.step()   
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            # 这里都仅针对RUNNING队列
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # 首先将这个字典output按照key的大小顺序排列（从小到大），然后将dict转换成list
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 将 token id转换成文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
