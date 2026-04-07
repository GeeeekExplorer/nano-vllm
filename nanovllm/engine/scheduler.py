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
    prefix_cache_hit_tokens: int = 0
    recomputed_prefill_tokens: int = 0


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
        self.enable_cb_prefill_liveness = config.enable_cb_prefill_liveness
        self.cb_prefill_reserve_ratio = config.cb_prefill_reserve_ratio
        self.cb_prefill_min_tokens = config.cb_prefill_min_tokens
        self.cb_prefill_min_seqs = config.cb_prefill_min_seqs
        self.enable_resumable_priority = config.enable_resumable_priority
        self.resumable_priority_cached_tokens_weight = config.resumable_priority_cached_tokens_weight
        self.resumable_priority_remaining_prefill_tokens_weight = config.resumable_priority_remaining_prefill_tokens_weight
        self.resumable_priority_waiting_time_weight = config.resumable_priority_waiting_time_weight
        self.resumable_priority_preempt_count_weight = config.resumable_priority_preempt_count_weight
        self.schedule_decode_next = False
        self.schedule_tick = 0
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self._append_waiting(seq)

    def _append_waiting(self, seq: Sequence):
        seq.waiting_since = self.schedule_tick
        self.waiting.append(seq)

    def _appendleft_waiting(self, seq: Sequence):
        seq.waiting_since = self.schedule_tick
        self.waiting.appendleft(seq)

    def _estimate_cached_tokens(self, seq: Sequence) -> int:
        if seq.block_table:
            return seq.num_computed_tokens
        return self.block_manager.count_cached_tokens(seq)

    def _compute_resumable_priority_score(self, seq: Sequence) -> tuple[float, int, int, int]:
        cached_tokens = self._estimate_cached_tokens(seq)
        remaining_prefill_tokens = max(len(seq) - cached_tokens, 0)
        waiting_time = max(self.schedule_tick - seq.waiting_since, 0)
        score = (
            self.resumable_priority_cached_tokens_weight * cached_tokens
            - self.resumable_priority_remaining_prefill_tokens_weight * remaining_prefill_tokens
            + self.resumable_priority_waiting_time_weight * waiting_time
            + self.resumable_priority_preempt_count_weight * seq.preempt_count
        )
        return score, cached_tokens, remaining_prefill_tokens, waiting_time

    def _promote_priority_waiting(self, scheduled_seq_ids: set[int] | None = None) -> bool:
        best_idx = None
        best_key = None
        for idx, seq in enumerate(self.waiting):
            if scheduled_seq_ids is not None and seq.seq_id in scheduled_seq_ids:
                continue
            if not seq.block_table and not self.block_manager.can_allocate(seq):
                continue
            score, cached_tokens, remaining_prefill_tokens, waiting_time = self._compute_resumable_priority_score(seq)
            key = (
                score,
                waiting_time,
                cached_tokens,
                -remaining_prefill_tokens,
                seq.preempt_count,
                -idx,
            )
            if best_key is None or key > best_key:
                best_idx = idx
                best_key = key
        if best_idx is None:
            return False
        if best_idx != 0:
            self.waiting.rotate(-best_idx)
        return True

    def _promote_resumable_waiting(self, scheduled_seq_ids: set[int] | None = None) -> bool:
        for idx, seq in enumerate(self.waiting):
            if scheduled_seq_ids is not None and seq.seq_id in scheduled_seq_ids:
                continue
            if seq.block_table:
                self.waiting.rotate(-idx)
                return True
        return False

    def _compute_prefill_reserve(self) -> tuple[int, int]:
        if (
            not self.enable_cb_prefill_liveness
            or not self.running
            or not self.waiting
        ):
            return 0, 0

        reserve_tokens = max(
            self.cb_prefill_min_tokens,
            int(self.max_num_batched_tokens * self.cb_prefill_reserve_ratio),
        )
        if self.enable_chunked_prefill and self.chunked_prefill_size > 0 and reserve_tokens > 0:
            reserve_tokens = ((reserve_tokens + self.chunked_prefill_size - 1) // self.chunked_prefill_size) * self.chunked_prefill_size
        reserve_seqs = self.cb_prefill_min_seqs

        # Keep at least minimal decode room when both queues are non-empty.
        if self.max_num_batched_tokens <= 1:
            reserve_tokens = 0
        else:
            reserve_tokens = min(reserve_tokens, self.max_num_batched_tokens - 1)
            if (
                self.enable_chunked_prefill
                and self.chunked_prefill_size > 0
                and reserve_tokens > self.chunked_prefill_size
                and reserve_tokens % self.chunked_prefill_size != 0
            ):
                reserve_tokens -= reserve_tokens % self.chunked_prefill_size

        if self.max_num_seqs <= 1:
            reserve_seqs = 0
        else:
            reserve_seqs = min(reserve_seqs, self.max_num_seqs - 1)

        return max(reserve_tokens, 0), max(reserve_seqs, 0)

    def _schedule_decode_pass(
        self,
        scheduled_items: list[ScheduledItem],
        scheduled_seq_ids: set[int],
        decode_scheduled_running: list[Sequence],
        num_seqs: int,
        num_batched_tokens: int,
        seq_limit: int,
        token_limit: int,
    ) -> tuple[int, int]:
        def pop_preempt_victim() -> Sequence | None:
            # Never preempt sequences already scheduled in this step.
            # They may already be part of current ScheduleBatch.
            for i in range(len(self.running) - 1, -1, -1):
                victim = self.running[i]
                if victim.seq_id in scheduled_seq_ids:
                    continue
                del self.running[i]
                return victim
            return None

        deferred_running = []
        while self.running and num_seqs < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            if num_seqs >= seq_limit or num_batched_tokens >= token_limit:
                break
            seq = self.running.popleft()
            if seq.seq_id in scheduled_seq_ids:
                deferred_running.append(seq)
                continue
            if seq.num_computed_tokens >= len(seq):
                seq.num_computed_tokens = len(seq) - 1
            while not self.block_manager.can_append(seq):
                victim = pop_preempt_victim()
                if victim is None:
                    self.preempt(seq)
                    seq = None
                    break
                self.preempt(victim)
            if seq is None:
                break
            self.block_manager.may_append(seq)
            seq.status = SequenceStatus.RUNNING
            seq.scheduled_tokens = 1
            scheduled_items.append(ScheduledItem(seq, 1, True, True))
            decode_scheduled_running.append(seq)
            scheduled_seq_ids.add(seq.seq_id)
            num_seqs += 1
            num_batched_tokens += 1
        if deferred_running:
            self.running.extend(deferred_running)
        return num_seqs, num_batched_tokens

    def _schedule_prefill_pass(
        self,
        scheduled_items: list[ScheduledItem],
        scheduled_seq_ids: set[int],
        num_seqs: int,
        num_batched_tokens: int,
        target_prefill_tokens: int,
        target_prefill_seqs: int,
    ) -> tuple[int, int, int, int]:
        scheduled_prefill_tokens = 0
        scheduled_prefill_seqs = 0
        while self.waiting and num_seqs < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            if (
                scheduled_prefill_tokens >= target_prefill_tokens
                and scheduled_prefill_seqs >= target_prefill_seqs
            ):
                break

            if self.enable_resumable_priority:
                if not self._promote_priority_waiting(scheduled_seq_ids):
                    break
            seq = self.waiting[0]
            if seq.seq_id in scheduled_seq_ids:
                break
            allocated = False
            prefix_cache_hit_tokens = 0
            if not seq.block_table:
                if not self.block_manager.can_allocate(seq):
                    if self.enable_resumable_priority:
                        break
                    if self._promote_resumable_waiting(scheduled_seq_ids):
                        continue
                    break
                prefix_cache_hit_tokens = self.block_manager.allocate(seq)
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
            remain_target_tokens = max(target_prefill_tokens - scheduled_prefill_tokens, 0)
            if remain_target_tokens > 0:
                num_budget_tokens = min(num_budget_tokens, remain_target_tokens)
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
            recomputed_prefill_tokens = seq.count_recomputed_prefill_tokens(num_query_tokens)
            seq.scheduled_tokens = num_query_tokens
            should_sample = seq.num_computed_tokens + num_query_tokens >= len(seq)
            seq.status = SequenceStatus.RUNNING if should_sample else SequenceStatus.WAITING
            if should_sample:
                self.running.append(seq)
            else:
                self._append_waiting(seq)
            scheduled_items.append(
                ScheduledItem(
                    seq,
                    num_query_tokens,
                    should_sample,
                    False,
                    prefix_cache_hit_tokens=prefix_cache_hit_tokens,
                    recomputed_prefill_tokens=recomputed_prefill_tokens,
                )
            )
            scheduled_seq_ids.add(seq.seq_id)
            num_seqs += 1
            num_batched_tokens += num_query_tokens
            scheduled_prefill_tokens += num_query_tokens
            scheduled_prefill_seqs += 1
        return num_seqs, num_batched_tokens, scheduled_prefill_tokens, scheduled_prefill_seqs

    def schedule(self) -> ScheduleBatch:
        self.schedule_tick += 1
        if self.enable_continuous_batching:
            return self._schedule_continuous()
        return self._schedule_legacy()

    def _schedule_continuous(self) -> ScheduleBatch:
        scheduled_items = []
        scheduled_seq_ids = set()
        decode_scheduled_running = []
        num_seqs = 0
        num_batched_tokens = 0

        reserve_prefill_tokens, reserve_prefill_seqs = self._compute_prefill_reserve()

        # Pass 1: decode first, while reserving minimal quota for prefill liveness.
        decode_seq_limit = max(self.max_num_seqs - reserve_prefill_seqs, 0)
        decode_token_limit = max(self.max_num_batched_tokens - reserve_prefill_tokens, 0)
        num_seqs, num_batched_tokens = self._schedule_decode_pass(
            scheduled_items,
            scheduled_seq_ids,
            decode_scheduled_running,
            num_seqs,
            num_batched_tokens,
            decode_seq_limit,
            decode_token_limit,
        )

        # Pass 2: guarantee minimal prefill progress.
        if self.waiting and num_seqs < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            if reserve_prefill_tokens > 0 or reserve_prefill_seqs > 0:
                target_prefill_tokens = reserve_prefill_tokens
                target_prefill_seqs = reserve_prefill_seqs
            else:
                # No decode pressure: let prefill use all available budget.
                target_prefill_tokens = self.max_num_batched_tokens
                target_prefill_seqs = self.max_num_seqs
            num_seqs, num_batched_tokens, _, _ = self._schedule_prefill_pass(
                scheduled_items,
                scheduled_seq_ids,
                num_seqs,
                num_batched_tokens,
                target_prefill_tokens,
                target_prefill_seqs,
            )

        # Pass 3: return unused quota to decode.
        if self.running and num_seqs < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            num_seqs, num_batched_tokens = self._schedule_decode_pass(
                scheduled_items,
                scheduled_seq_ids,
                decode_scheduled_running,
                num_seqs,
                num_batched_tokens,
                self.max_num_seqs,
                self.max_num_batched_tokens,
            )

        # If decode drained out, avoid under-utilization by letting prefill backfill.
        if (not self.running) and self.waiting and num_seqs < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            num_seqs, num_batched_tokens, _, _ = self._schedule_prefill_pass(
                scheduled_items,
                scheduled_seq_ids,
                num_seqs,
                num_batched_tokens,
                self.max_num_batched_tokens,
                self.max_num_seqs,
            )

        # Round-robin decode fairness: scheduled decode sequences go to tail once.
        self.running.extend(decode_scheduled_running)

        if not scheduled_items:
            raise RuntimeError("scheduler has no runnable sequences")
        return ScheduleBatch(scheduled_items)

    def _schedule_prefill_legacy(self) -> list[ScheduledItem]:
        scheduled_items = []
        scheduled_seq_ids = set()
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            if self.enable_resumable_priority:
                if not self._promote_priority_waiting(scheduled_seq_ids):
                    break
            seq = self.waiting[0]
            if seq.seq_id in scheduled_seq_ids:
                break
            allocated = False
            prefix_cache_hit_tokens = 0
            if not seq.block_table:
                if not self.block_manager.can_allocate(seq):
                    if self.enable_resumable_priority:
                        break
                    if self._promote_resumable_waiting(scheduled_seq_ids):
                        continue
                    break
                prefix_cache_hit_tokens = self.block_manager.allocate(seq)
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
            recomputed_prefill_tokens = seq.count_recomputed_prefill_tokens(num_query_tokens)
            should_sample = seq.num_computed_tokens + num_query_tokens >= seq.num_prompt_tokens
            seq.scheduled_tokens = num_query_tokens
            seq.status = SequenceStatus.RUNNING if should_sample else SequenceStatus.WAITING
            scheduled_items.append(
                ScheduledItem(
                    seq,
                    num_query_tokens,
                    should_sample,
                    False,
                    prefix_cache_hit_tokens=prefix_cache_hit_tokens,
                    recomputed_prefill_tokens=recomputed_prefill_tokens,
                )
            )
            if should_sample:
                self.running.append(seq)
            else:
                self._append_waiting(seq)
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
        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)
        seq.status = SequenceStatus.WAITING
        seq.scheduled_tokens = 0
        seq.preempt_count += 1
        self.block_manager.deallocate(seq)
        self._appendleft_waiting(seq)

    def postprocess(self, batch: ScheduleBatch, token_ids: list[int]):
        for item, token_id in zip(batch.items, token_ids):
            seq = item.seq
            seq.num_computed_tokens += item.num_query_tokens
            seq.max_context_tokens_seen = max(seq.max_context_tokens_seen, seq.num_computed_tokens)
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
