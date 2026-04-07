from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_computed_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.top_p = sampling_params.top_p
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.scheduled_tokens = 0
        self.preempt_count = 0
        self.waiting_since = 0
        self.max_context_tokens_seen = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        token_ids = getattr(self, "token_ids", None)
        if token_ids is not None:
            return token_ids[key]
        if isinstance(key, int):
            if key in (self.num_tokens - 1, -1):
                return self.last_token
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.num_tokens)
            if step == 1 and start == self.num_tokens - 1 and stop == self.num_tokens:
                return [self.last_token]
        raise RuntimeError("token ids are unavailable for the requested index")

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_computed_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def count_recomputed_prefill_tokens(self, num_query_tokens: int):
        if num_query_tokens <= 0 or self.num_computed_tokens >= self.max_context_tokens_seen:
            return 0, 0
        start = self.num_computed_tokens
        end = start + num_query_tokens
        total_recomputed_tokens = max(0, min(end, self.max_context_tokens_seen) - start)
        if total_recomputed_tokens == 0:
            return 0, 0

        prompt_seen_end = min(self.max_context_tokens_seen, self.num_prompt_tokens)
        recomputed_prompt_tokens = max(0, min(end, prompt_seen_end) - start)
        recomputed_decode_context_tokens = total_recomputed_tokens - recomputed_prompt_tokens
        return recomputed_prompt_tokens, recomputed_decode_context_tokens

    def __getstate__(self):
        needs_full_tokens = (
            self.scheduled_tokens != 1
            or self.num_computed_tokens != self.num_tokens - 1
            or not self.block_table
        )
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_computed_tokens,
            self.block_table,
            self.scheduled_tokens,
            needs_full_tokens,
            self.token_ids if needs_full_tokens else self.last_token,
        )

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_computed_tokens, self.block_table = state[:4]
        if len(state) == 7:
            self.scheduled_tokens = state[4]
            needs_full_tokens = state[5]
            payload = state[6]
            if needs_full_tokens:
                self.token_ids = payload
                self.last_token = payload[-1]
            else:
                self.last_token = payload
                if hasattr(self, "token_ids"):
                    del self.token_ids
            return

        # Backward compatibility for older pickled states.
        self.scheduled_tokens = state[4] if len(state) == 6 else 0
        payload = state[-1]
        if self.num_completion_tokens == 0:
            self.token_ids = payload
            self.last_token = payload[-1]
        else:
            self.last_token = payload
            if hasattr(self, "token_ids"):
                del self.token_ids
