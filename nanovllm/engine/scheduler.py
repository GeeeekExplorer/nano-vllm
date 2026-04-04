from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.chunked_prefill_size = config.chunked_prefill_size
        self.schedule_decode_next = False
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule_prefill(self) -> list[Sequence]:
        scheduled_seqs = []
        scheduled_seq_ids = set()
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if seq.seq_id in scheduled_seq_ids:
                break
            allocated = False
            if not seq.block_table:
                if not self.block_manager.can_allocate(seq):
                    break
                self.block_manager.allocate(seq)
                allocated = True
            num_remaining_prompt_tokens = seq.num_prompt_tokens - seq.num_cached_tokens
            if num_remaining_prompt_tokens <= 0:
                self.waiting.popleft()
                scheduled_seq_ids.add(seq.seq_id)
                self.running.append(seq)
                continue
            num_scheduled_prefill_tokens = num_remaining_prompt_tokens
            if self.enable_chunked_prefill:
                num_budget_tokens = self.max_num_batched_tokens - num_batched_tokens
                if num_budget_tokens <= 0:
                    break
                num_scheduled_prefill_tokens = min(
                    num_scheduled_prefill_tokens,
                    self.chunked_prefill_size,
                    num_budget_tokens,
                )
            if num_scheduled_prefill_tokens <= 0 or num_batched_tokens + num_scheduled_prefill_tokens > self.max_num_batched_tokens:
                if allocated:
                    self.block_manager.deallocate(seq)
                break
            num_seqs += 1
            num_batched_tokens += num_scheduled_prefill_tokens
            self.waiting.popleft()
            scheduled_seq_ids.add(seq.seq_id)
            seq.status = SequenceStatus.RUNNING
            seq.scheduled_prefill_tokens = num_scheduled_prefill_tokens
            scheduled_seqs.append(seq)
            if seq.num_cached_tokens + num_scheduled_prefill_tokens < seq.num_prompt_tokens:
                seq.status = SequenceStatus.WAITING
                self.waiting.append(seq)
            else:
                self.running.append(seq)
        return scheduled_seqs

    def schedule_decode(self) -> list[Sequence]:
        scheduled_seqs = []
        num_seqs = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            return []
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.enable_chunked_prefill and self.waiting and self.running:
            if self.schedule_decode_next:
                scheduled_seqs = self.schedule_decode()
                if scheduled_seqs:
                    self.schedule_decode_next = False
                    return scheduled_seqs, False
            scheduled_seqs = self.schedule_prefill()
            if scheduled_seqs:
                self.schedule_decode_next = True
                return scheduled_seqs, True
            scheduled_seqs = self.schedule_decode()
            if scheduled_seqs:
                self.schedule_decode_next = False
                return scheduled_seqs, False
        else:
            scheduled_seqs = self.schedule_prefill()
            if scheduled_seqs:
                return scheduled_seqs, True
            scheduled_seqs = self.schedule_decode()
            if scheduled_seqs:
                return scheduled_seqs, False
        raise RuntimeError("scheduler has no runnable sequences")

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill and seq.scheduled_prefill_tokens > 0:
                seq.num_cached_tokens += seq.scheduled_prefill_tokens
                seq.scheduled_prefill_tokens = 0
                if seq.num_cached_tokens < seq.num_prompt_tokens:
                    seq.status = SequenceStatus.WAITING
                    continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
