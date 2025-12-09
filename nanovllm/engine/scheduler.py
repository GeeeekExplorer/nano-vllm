from collections import deque
from typing import Literal

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    Scheduler with explicit PD (Prefill/Decode) separation.

    Provides separate queues and scheduling methods for prefill and decode,
    enabling independent control flow for pipeline scheduling.
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

        # Explicit PD separation: separate queues for prefill and decode
        self.prefill_queue: deque[Sequence] = deque()  # Sequences waiting for prefill
        self.decode_queue: deque[Sequence] = deque()   # Sequences in decode phase

        # Legacy aliases for compatibility
        self.waiting = self.prefill_queue
        self.running = self.decode_queue

    def is_finished(self):
        return not self.prefill_queue and not self.decode_queue

    def add(self, seq: Sequence):
        self.prefill_queue.append(seq)

    # --- Explicit PD separation interfaces ---

    def ready_for_prefill(self) -> bool:
        """Check if there are sequences ready for prefill"""
        return len(self.prefill_queue) > 0

    def ready_for_decode(self) -> bool:
        """Check if there are sequences ready for decode"""
        return len(self.decode_queue) > 0

    def schedule_prefill(self) -> list[Sequence]:
        """
        Schedule sequences for prefill only.
        Returns list of sequences to prefill.
        """
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        while self.prefill_queue and num_seqs < self.max_num_seqs:
            seq = self.prefill_queue[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.prefill_queue.popleft()
            self.decode_queue.append(seq)
            scheduled_seqs.append(seq)

        # Track prefilled sequences for two-GPU mode
        if not hasattr(self, '_last_prefilled'):
            self._last_prefilled = []
        self._last_prefilled = scheduled_seqs.copy()

        return scheduled_seqs

    def schedule_decode(self) -> list[Sequence]:
        """
        Schedule sequences for decode only.
        Returns list of sequences to decode.
        """
        scheduled_seqs = []
        num_seqs = 0

        while self.decode_queue and num_seqs < self.max_num_seqs:
            seq = self.decode_queue.popleft()
            while not self.block_manager.can_append(seq):
                if self.decode_queue:
                    self.preempt(self.decode_queue.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        if scheduled_seqs:
            self.decode_queue.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        Legacy unified scheduling interface.
        Returns (scheduled_seqs, is_prefill).
        """
        # Try prefill first
        scheduled_seqs = self.schedule_prefill()
        if scheduled_seqs:
            return scheduled_seqs, True

        # Then decode
        scheduled_seqs = self.schedule_decode()
        assert scheduled_seqs
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.prefill_queue.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.decode_queue.remove(seq)

    def get_queue_stats(self) -> dict:
        """Get current queue statistics for debugging"""
        return {
            "prefill_queue_len": len(self.prefill_queue),
            "decode_queue_len": len(self.decode_queue),
        }

    def get_prefilled_sequences(self) -> list[Sequence]:
        """
        Get sequences that just completed prefill and need to be migrated to decode GPU.
        This is used in two-GPU PD separation mode.

        Returns:
            List of sequences that were just added to decode_queue in the last schedule_prefill call.
        """
        # In two-GPU mode, sequences that were just prefilled are at the end of decode_queue
        # We'll track this via a temporary list that gets populated during schedule_prefill
        if not hasattr(self, '_last_prefilled'):
            self._last_prefilled = []
        result = self._last_prefilled
        self._last_prefilled = []
        return result
