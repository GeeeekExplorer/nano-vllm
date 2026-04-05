from collections import deque
from dataclasses import dataclass

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass
class ScheduledItem:
    seq: Sequence
    num_query_tokens: int
    should_sample: bool
    is_decode: bool


@dataclass
class ScheduleBatch:
    items: list[ScheduledItem]


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.enable_continuous_batching = config.enable_continuous_batching
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

    def _promote_resumable_waiting(self, scheduled_seq_ids: set[int] | None = None) -> bool:
        for idx, seq in enumerate(self.waiting):
            if scheduled_seq_ids is not None and seq.seq_id in scheduled_seq_ids:
                continue
            if seq.block_table:
                self.waiting.rotate(-idx)
                return True
        return False

    def schedule(self) -> ScheduleBatch:
        if self.enable_continuous_batching:
            return self._schedule_continuous()
        return self._schedule_legacy()

    def _schedule_continuous(self) -> ScheduleBatch:
        scheduled_items = []
        scheduled_seq_ids = set()
        scheduled_running = []
        num_seqs = 0
        num_batched_tokens = 0

        # Decode is latency-sensitive. Schedule one token for active sequences first.
        while self.running and num_seqs < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            seq = self.running.popleft()
            if seq.seq_id in scheduled_seq_ids:
                continue
            if seq.num_computed_tokens >= len(seq):
                seq.num_computed_tokens = len(seq) - 1
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    seq = None
                    break
            if seq is None:
                break
            self.block_manager.may_append(seq)
            seq.status = SequenceStatus.RUNNING
            seq.scheduled_tokens = 1
            scheduled_items.append(ScheduledItem(seq, 1, True, True))
            scheduled_running.append(seq)
            scheduled_seq_ids.add(seq.seq_id)
            num_seqs += 1
            num_batched_tokens += 1

        # Round-robin decode fairness: scheduled sequences go to the tail.
        self.running.extend(scheduled_running)

        # Fill leftover budget with waiting/recompute chunks.
        while self.waiting and num_seqs < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            seq = self.waiting[0]
            if seq.seq_id in scheduled_seq_ids:
                break
            allocated = False
            if not seq.block_table:
                if not self.block_manager.can_allocate(seq):
                    if self._promote_resumable_waiting(scheduled_seq_ids):
                        continue
                    break
                self.block_manager.allocate(seq)
                allocated = True

            num_remaining_tokens = len(seq) - seq.num_computed_tokens
            if num_remaining_tokens <= 0:
                # Even with full prefix cache hits, one-token decode is still needed
                # to produce the next-token logits.
                seq.num_computed_tokens = len(seq) - 1
                num_remaining_tokens = 1

            num_budget_tokens = self.max_num_batched_tokens - num_batched_tokens
            if num_budget_tokens <= 0:
                break
            num_query_tokens = min(num_remaining_tokens, num_budget_tokens)
            if self.enable_chunked_prefill:
                num_query_tokens = min(num_query_tokens, self.chunked_prefill_size)

            if num_query_tokens <= 0:
                if allocated:
                    self.block_manager.deallocate(seq)
                break

            self.waiting.popleft()
            seq.scheduled_tokens = num_query_tokens
            should_sample = seq.num_computed_tokens + num_query_tokens >= len(seq)
            seq.status = SequenceStatus.RUNNING if should_sample else SequenceStatus.WAITING
            if should_sample:
                self.running.append(seq)
            else:
                self.waiting.append(seq)
            scheduled_items.append(ScheduledItem(seq, num_query_tokens, should_sample, False))
            scheduled_seq_ids.add(seq.seq_id)
            num_seqs += 1
            num_batched_tokens += num_query_tokens

        if not scheduled_items:
            raise RuntimeError("scheduler has no runnable sequences")
        return ScheduleBatch(scheduled_items)

    def _schedule_prefill_legacy(self) -> list[ScheduledItem]:
        scheduled_items = []
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
                    if self._promote_resumable_waiting(scheduled_seq_ids):
                        continue
                    break
                self.block_manager.allocate(seq)
                allocated = True

            num_remaining_prompt_tokens = seq.num_prompt_tokens - seq.num_computed_tokens
            if num_remaining_prompt_tokens <= 0:
                self.waiting.popleft()
                scheduled_seq_ids.add(seq.seq_id)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                continue

            num_query_tokens = num_remaining_prompt_tokens
            if self.enable_chunked_prefill:
                num_budget_tokens = self.max_num_batched_tokens - num_batched_tokens
                if num_budget_tokens <= 0:
                    break
                num_query_tokens = min(
                    num_query_tokens,
                    self.chunked_prefill_size,
                    num_budget_tokens,
                )
            if num_query_tokens <= 0 or num_batched_tokens + num_query_tokens > self.max_num_batched_tokens:
                if allocated:
                    self.block_manager.deallocate(seq)
                break

            num_seqs += 1
            num_batched_tokens += num_query_tokens
            self.waiting.popleft()
            scheduled_seq_ids.add(seq.seq_id)
            should_sample = seq.num_computed_tokens + num_query_tokens >= seq.num_prompt_tokens
            seq.scheduled_tokens = num_query_tokens
            seq.status = SequenceStatus.RUNNING if should_sample else SequenceStatus.WAITING
            scheduled_items.append(ScheduledItem(seq, num_query_tokens, should_sample, False))
            if should_sample:
                self.running.append(seq)
            else:
                self.waiting.append(seq)
        return scheduled_items

    def _schedule_decode_legacy(self) -> list[ScheduledItem]:
        scheduled_items = []
        num_seqs = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            if seq.num_computed_tokens >= len(seq):
                seq.num_computed_tokens = len(seq) - 1
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                seq.status = SequenceStatus.RUNNING
                seq.scheduled_tokens = 1
                scheduled_items.append(ScheduledItem(seq, 1, True, True))
        if not scheduled_items:
            return []
        self.running.extendleft(reversed([item.seq for item in scheduled_items]))
        return scheduled_items

    def _schedule_legacy(self) -> ScheduleBatch:
        if self.enable_chunked_prefill and self.waiting and self.running:
            if self.schedule_decode_next:
                scheduled_items = self._schedule_decode_legacy()
                if scheduled_items:
                    self.schedule_decode_next = False
                    return ScheduleBatch(scheduled_items)
            scheduled_items = self._schedule_prefill_legacy()
            if scheduled_items:
                self.schedule_decode_next = True
                return ScheduleBatch(scheduled_items)
            scheduled_items = self._schedule_decode_legacy()
            if scheduled_items:
                self.schedule_decode_next = False
                return ScheduleBatch(scheduled_items)
        else:
            scheduled_items = self._schedule_prefill_legacy()
            if scheduled_items:
                return ScheduleBatch(scheduled_items)
            scheduled_items = self._schedule_decode_legacy()
            if scheduled_items:
                return ScheduleBatch(scheduled_items)
        raise RuntimeError("scheduler has no runnable sequences")

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.scheduled_tokens = 0
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, batch: ScheduleBatch, token_ids: list[int]):
        for item, token_id in zip(batch.items, token_ids):
            seq = item.seq
            seq.num_computed_tokens += item.num_query_tokens
            seq.scheduled_tokens = 0
            if not item.should_sample:
                seq.status = SequenceStatus.WAITING
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
