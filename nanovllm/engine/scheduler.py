from collections import deque
from time import perf_counter

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

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Chunked-prefill 混合连续批处理：一个 step 内同时调度 running 的 decode
        (每个 1 token) 和 waiting 的 prefill chunk。

        返回 (scheduled_seqs, has_prefill)。has_prefill 为 True 表示本批含 prefill chunk，
        需走 eager 统一 varlen 路径；为 False 表示纯 decode，可走 CUDA Graph。
        """
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        # ---- decode: 每个 running seq 推进 1 个 token（预算便宜，优先排）----
        running = self.running
        self.running = deque()
        while running and num_seqs < self.max_num_seqs:
            seq = running.popleft()
            while not self.block_manager.can_append(seq):
                if running:
                    self.preempt(running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                num_batched_tokens += 1
                self.running.append(seq)
                scheduled_seqs.append(seq)
        # 受 max_num_seqs 限制本步没处理到的 running seq 留在队列里
        self.running.extend(running)

        # ---- prefill: 用剩余 token 预算给 waiting 的 seq 分块 ----
        has_prefill = False
        while self.waiting and num_seqs < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break
            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            scheduled_tokens = min(num_tokens, remaining)
            if scheduled_tokens <= 0:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = scheduled_tokens
            num_batched_tokens += scheduled_tokens
            num_seqs += 1
            has_prefill = True
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        assert scheduled_seqs
        return scheduled_seqs, has_prefill

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            # 逐 seq 判定：仍在 prefill 中(本块非最后一块)的 seq 不产出 token，继续留在 waiting
            still_prefilling = seq.num_cached_tokens < seq.num_tokens
            seq.num_scheduled_tokens = 0
            if still_prefilling:
                continue
            seq.append_token(token_id)
            if seq.first_token_time is None:    # 首个 completion token 产出，记录 TTFT 时间点
                seq.first_token_time = perf_counter()
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
