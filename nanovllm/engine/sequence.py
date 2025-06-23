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

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams, prompt: str | list[int]):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.prompt = prompt
        if isinstance(token_ids, str):
            raise ValueError("token_ids must be a list of integers, not a string")
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)  # Convert to list if needed
            if not all(isinstance(x, int) for x in token_ids):
                raise ValueError("token_ids must be a list of integers")
        
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1] if token_ids else 0
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        # Initialize an empty block table - will be filled by BlockManager
        self.block_table = []
        self.sampling_params = sampling_params
        self.current_position = self.num_tokens  # Position for the NEXT token

        # Copy essential properties from sampling_params for convenience
        self.ignore_eos = self.sampling_params.ignore_eos
        self.max_tokens = self.sampling_params.max_tokens
        
        # Keep track of generated tokens separately for clear output
        self.generated_tokens = []

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

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
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    @property
    def position(self):
        """Current position for the next token to be generated"""
        return self.current_position

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """Add a new token to the sequence and update tracking"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.current_position += 1  # Increment position for next token
        self.generated_tokens.append(token_id)  # Track generated tokens separately

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, 
                self.block_table, self.token_ids, self.prompt, self.current_position, 
                self.generated_tokens)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:4]
        self.token_ids = state[4]
        self.prompt = state[5]
        self.current_position = state[6]
        self.generated_tokens = state[7] if len(state) > 7 else []
        self.last_token = self.token_ids[-1] if self.token_ids else 0
