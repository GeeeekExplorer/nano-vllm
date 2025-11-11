from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Optional

import torch

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        images=None,
        pixel_values=None,
        image_grid_thw=None,
        vision_counts: Optional[list[int]] = None,
        vision_placeholders: Optional[list[tuple[int, int]]] = None,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        # Multimodal metadata
        self.images = images
        self.pixel_values = pixel_values
        self.image_grid_thw = image_grid_thw
        self.vision_placeholders = vision_placeholders or []
        if vision_counts is not None:
            self.vision_counts = vision_counts
        elif self.vision_placeholders:
            self.vision_counts = [
                length for _, length in self.vision_placeholders
            ]
        else:
            self.vision_counts = []
        # Track how many visual tokens per placeholder have been copied into
        # the prompt so far.
        self.vision_consumed = [0] * len(self.vision_placeholders)
        # Cached outputs of the vision encoder; populated on first access and
        # released once every placeholder is consumed.
        self.cached_vision_tokens: Optional[list[torch.Tensor]] = None
        self.cached_deepstack_tokens: Optional[list[list[torch.Tensor]]] = None
        self.vision_offset = 0

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

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids
            if self.num_completion_tokens == 0
            else self.last_token,
        )

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
        ) = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
        # Reset multimodal caches when the sequence is restored.
        self.images = None
        self.pixel_values = None
        self.image_grid_thw = None
        self.vision_placeholders = []
        self.vision_counts = []
        self.vision_consumed = []
        self.cached_vision_tokens = None
        self.cached_deepstack_tokens = None
        self.vision_offset = 0
