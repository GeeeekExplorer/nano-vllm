from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        # Queues for sequences waiting to be scheduled and currently running.
        # Sequences in the waiting queue are in the order of arrival, and sequences in the running queue are in the order of scheduling.
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """Returns True if there are no sequences in both waiting and running queues."""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """Adds a new sequence to the waiting queue."""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        Decides on each scheduling step which sequences to run in the next model inference step and whether it's a prefill step or a decode step.

        Always tries to schedule a prefill step, if there are sequences in the waiting queue until
        reaching the max number of batched tokens or until not being able to allocate blocks for the next sequence in the waiting queue.

        If no sequence can be scheduled for prefill, schedules sequences for decode.
        If the decode sequences get too long, preempts the longest running sequence until the new token can be appended to the current sequence.

        Returns the list of scheduled sequences and a boolean indicating whether it's a prefill step.
        """
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            # Take a look at the first sequence in the waiting queue and check if we can schedule it for prefill.
            seq = self.waiting[0]
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            # We schedule this sequence for prefill, if
            # 1) adding this sequence does not exceed the max number of batched tokens, and
            # 2) we can allocate blocks for this sequence in the block manager.

            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            # Update queues
            self.waiting.popleft()
            self.running.append(seq)

            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()

            # Preempt sequences until we can append a new token to the current sequence.
            while not self.block_manager.can_append(seq):
                # Try to preempt the longest running sequence (i.e. the one at the end of the running queue).
                # If there is no other sequence to preempt, we have to preempt the current sequence.
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs, "No sequence scheduled for decode"
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence) -> None:
        """
        Preempts the given sequence during decode scheduling by changing its status back to WAITING,
        deallocating its blocks, and moving it back to the waiting queue.
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        """
        To be called after each step of model inference with the generated token ids for the scheduled sequences.
        Updates the status of each sequence.
        If a sequence has finished, deallocates its blocks and removes it from the running queue.
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
