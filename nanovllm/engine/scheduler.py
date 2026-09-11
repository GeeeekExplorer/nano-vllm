from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.prefill_reserve_blocks = config.prefill_reserve_blocks

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool, bool]:
        """生成一个可同时包含 decode 与 chunked prefill 的调度批。

        先为每个 running 序列保留一个 decode token，再以剩余 token 预算接纳
        waiting 请求的 prefill。返回值中的 ``has_prefill`` 决定执行路径：混合批
        必须复用 varlen prefill attention，而纯 decode 批仍可使用 CUDA Graph。
        """
        scheduled_seqs = []
        num_batched_tokens = 0
        has_prefill = False

        # 先保证已有请求的一个 decode token，避免持续到达的 prefill 长期阻塞它们。
        decode_seqs = []

        while self.running and len(decode_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                seq.should_sample = True
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
                num_batched_tokens += 1
        # 已入批的 decode 仍保持 running 身份，供 postprocess 更新和下一轮继续调度。
        self.running.extendleft(reversed(decode_seqs))
        scheduled_seqs.extend(decode_seqs)

        # prefill
        num_prefill = 0
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks, num_new_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                free = len(self.block_manager.free_block_ids)
                # decode 已先占用本轮增量块；剩余空闲块还必须满足 watermark 才接纳 prefill。
                if free - num_new_blocks < self.prefill_reserve_blocks:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 单步只允许一个被截断的 prefill，避免多个长请求耗尽剩余 token budget。
            if remaining < num_tokens and num_prefill > 0:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            num_prefill += 1
            has_prefill = True
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.should_sample = True
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            else:
                seq.should_sample = False
            scheduled_seqs.append(seq)

        # 没有可执行请求意味着资源状态无法前进，显式报错优于原来的空断言。
        if not scheduled_seqs:
            raise RuntimeError("scheduler cannot make progress: no sequence can be scheduled")

        # ``is_prefill`` 仅供统计展示；运行路径由 ``has_prefill`` 决定。
        is_prefill = all(s.is_prefill for s in scheduled_seqs)
        return scheduled_seqs, is_prefill, has_prefill

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], has_prefill: bool):
        """提交本轮 KV 写入，并仅为需要采样的序列追加输出 token。"""
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if has_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
