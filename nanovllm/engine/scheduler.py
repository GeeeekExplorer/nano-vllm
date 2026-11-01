from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config, draft_config: Config | None = None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.num_speculative_tokens = config.num_speculative_tokens
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.enable_prefix_cache,
            role="target",
        )
        self.draft_block_manager: BlockManager | None = None
        if draft_config is not None:
            self.draft_block_manager = BlockManager(
                draft_config.num_kvcache_blocks,
                config.kvcache_block_size,
                enable_prefix_cache=False,
                role="draft",
            )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _can_admit(self, seq: Sequence) -> int:
        """Cached-block count of the target allocation, or -1 when either pool is short."""
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if self.draft_block_manager is not None and self.draft_block_manager.can_allocate(seq) == -1:
            return -1
        return num_cached_blocks

    def _allocate(self, seq: Sequence, num_cached_blocks: int) -> None:
        self.block_manager.allocate(seq, num_cached_blocks)
        if self.draft_block_manager is not None:
            self.draft_block_manager.allocate(seq, 0)
            assert seq.draft_kv.num_cached_tokens == seq.num_cached_tokens   # prefix cache is off with a draft

    def _deallocate(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        if self.draft_block_manager is not None:
            self.draft_block_manager.deallocate(seq)

    def _can_reserve_round(self, seq: Sequence) -> bool:
        k = self.num_speculative_tokens
        if not self.block_manager.can_reserve(seq, len(seq) + k):
            return False
        return self.draft_block_manager is None or self.draft_block_manager.can_reserve(seq, len(seq) + k - 1)

    def _reserve_round(self, seq: Sequence) -> None:
        k = self.num_speculative_tokens
        self.block_manager.reserve(seq, len(seq) + k)           # verification writes positions len-1 .. len+k-1
        if self.draft_block_manager is not None:
            self.draft_block_manager.reserve(seq, len(seq) + k - 1)   # draft steps write up to len+k-2

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self._can_admit(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self._allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            if self.draft_block_manager is not None:
                seq.draft_kv.num_scheduled_tokens = seq.num_scheduled_tokens
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode: reserve the blocks one round will write before running it
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self._can_reserve_round(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self._reserve_round(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self._deallocate(seq)
        self.waiting.appendleft(seq)

    def _finish(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.FINISHED
        self._deallocate(seq)
        self.running.remove(seq)

    def _is_stop_token(self, seq: Sequence, token_id: int) -> bool:
        return (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and self.draft_block_manager is not None:
                seq.draft_kv.num_cached_tokens += seq.draft_kv.num_scheduled_tokens
                seq.draft_kv.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if self._is_stop_token(seq, token_id):
                self._finish(seq)

    def postprocess_speculative(self, seqs: list[Sequence], appended: list[list[int]]) -> None:
        """Commit the accepted drafts plus the tail token of one round and restore the KV invariants."""
        k = self.num_speculative_tokens
        for seq, token_ids in zip(seqs, appended):
            assert token_ids
            len_before = len(seq)
            for token_id in token_ids:
                seq.append_token(token_id)
                if self._is_stop_token(seq, token_id):
                    self._finish(seq)
                    break
            if seq.is_finished:
                continue
            seq.target_kv.num_cached_tokens = len(seq) - 1
            if self.draft_block_manager is not None:    # a model draft; DFlash's context K/V follows the target's
                seq.draft_kv.num_cached_tokens = min(len_before - 1 + k, len(seq) - 1)
