from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs  # 单次调度时允许并行的序列数（最大的batch size）
        self.max_num_batched_tokens = config.max_num_batched_tokens # 按 token 计的批次大小上限（控制 token 数以避免超显存或超时）
        self.eos = config.eos    # 结束token id，用于判断序列是否完成
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size) # 管理KV Cache的Manager
        self.waiting: deque[Sequence] = deque() # 存放等待被调度的序列（双端序列）
        self.running: deque[Sequence] = deque() # 已分配资源并正在运行或等待进一步decode的序列

    def is_finished(self):
        # 当waiting队列与running队列都为空时，说明推理完成
        return not self.waiting and not self.running

    # 将新到的序列放到waiting队列的右端等到调度
    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []  # 本次调度需要执行（交给计算单元）的序列列表
        num_seqs = 0         # 初始化当前调度的序列数
        num_batched_tokens = 0 # 初始化当前调度的 “按token计的批次大小”
        while self.waiting and num_seqs < self.max_num_seqs:
        # 如果waiting队列中有序列且当前序列数小于最大值
            seq = self.waiting[0] # 取最左的序列
            # 若将新加入的序列的长度加入当前batch会超高token上限的话，则停止prefill
            # 若目前无法为这个序列分配kvcache，也停止prefill（资源不足）
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            # 若检查通过，就把该序列安排进本次的prefill
            num_seqs += 1  # 更新当前序列数
            self.block_manager.allocate(seq) # 为新添加的序列分类kvcache block
            num_batched_tokens += len(seq) - seq.num_cached_tokens # 更新当前token个数
            seq.status = SequenceStatus.RUNNING # 将该序列的状态设置为RUNNING
            self.waiting.popleft()     # 将该序列从waiting的最左端弹出，加入running队列最右端
            self.running.append(seq)
            scheduled_seqs.append(seq) # 将该序列添加到scheduled_seqs
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        # 在prefill结束之后，调度正在运行中的序列
        while self.running and num_seqs < self.max_num_seqs:
        # 当running不为空且当前序列数小于序列数上限时
            seq = self.running.popleft()  # 从running队列最左边取一个序列 FIFO

            # 检查当前序列是否能继续在其分配的kvcache块中追加一个token了
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop()) # 若running队列不为空，则pop最后一个序列（牺牲最晚进来的任务）
                else:
                    self.preempt(seq)   # 若running队列中已经空了（也就说所系统所有的资源都不够你这一个序列用了）
                    break               # 则暂停当前序列
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs   # 确保至少调度了一个序列
        self.running.extendleft(reversed(scheduled_seqs)) # 把刚刚调度的序列放回runing队列左侧
        return scheduled_seqs, False

    # 将序列的状态改回WAITING，并回收它的kvcache，并将其放在waiting队列的最左边等待下一次调度
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id) # 把新生成的token加入该序列的token列表中（更新上下文，让下轮decode能继续机基于新token生成下一个token）
            # 检查是否结束（模型输出的终止符号eos 或是 达到了最大的生成的长度）
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED  # 将当前序列的状态设置为FINISHIED
                self.block_manager.deallocate(seq)    # 释放该序列占用的kvcache block
                self.running.remove(seq)              # 从running队列中删除该序列
