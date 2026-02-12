"""Global context for the model forward pass computation.

Contains metadata about the sequences in the current batch.
"""

from dataclasses import dataclass
import torch


@dataclass
class Context:
    # Whether the current forward pass is for prefill or decode.
    is_prefill: bool = False

    # Cumulative sequence lengths for the queries (ndim=1), len = batch_size + 1 (first element is 0). Used for prefill and decode with prefix cache.
    cu_seqlens_q: torch.Tensor | None = None
    # Cumulative sequence lengths for the keys/values (ndim=1), len = batch_size + 1 (first element is 0). Used for prefill and decode with prefix cache.
    cu_seqlens_k: torch.Tensor | None = None

    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # Mapping between the actual cache slot indices in the allocated kv cache and the tokens in the input sequence.
    # Each element in the slot_mapping tensor corresponds to a token in the input sequence, and its value is the index of the cache slot in the kv cache where the key/value for that token is stored.
    slot_mapping: torch.Tensor | None = None  # (N_tokens,)
    # The actual sequence lengths for each sequence in the batch. Used for decode without prefix cache.
    context_lens: torch.Tensor | None = None  # (N_seqs,)

    # The block ids / idxes for per sequence. -1 is padding. Used for prefill and decode with prefix cache.
    # Assumes a kv_cache of shape (N_kv_cache_slots, block_size, num_kv_heads, head_dim), the block_ids index into the N_kv_cache_slots dimension of the kv cache.
    block_tables: torch.Tensor | None = None  # (N_seqs, max_num_blocks_per seq)


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
