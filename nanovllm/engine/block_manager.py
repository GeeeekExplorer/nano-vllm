from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        # Number of sequences sharing this block.
        self.ref_count = 0
        # Hash of the token ids of the full prefix
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # Block ids are from 0 to num_blocks-1 and contiguously map to the tokens in the physical KV cache.
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        """Computes the hash for a sequence of token ids.
        If a prefix hash is provided, it is used as the prefix for the hash computation.
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        # Ensure the block is not used by any sequence
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> None:
        # Ensure the block is not used by any sequence anymore
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """
        Used during prefill scheduling.

        Checks if there are enough free blocks to allocate for the given sequence.
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        Allocates blocks for the given sequence during prefill scheduling.

        The blocks are allocated in order. If there is a cache hit for a block (i.e. there is a block with the same hash and the same token ids),
        we reuse the block by increasing its ref count.
        If there is a cache miss for a block, we allocate a new block for it.

        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # Compute the hash for the current block, if it is a full block. Use previous block's hash as the prefix.
            # If it is not a full block, we do not compute the hash, i.e. set it  -1.
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            # Check if there is a block with the same hash and the same token ids.
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            # If cache miss, we need to allocate a new block for this block of the sequence.
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            # If cache hit, we can reuse the block with the same hash and the same token ids.
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # TODO@max: when does this case happen?
                    block = self._allocate_block(block_id)

            # Updated the block's hash and token ids if it is a cache miss, if it is a full block (i.e. len(token_ids) == block_size).
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        Deallocates a sequence.
        Used during preemption in decode scheduling and when a sequence finishes.
        """
        # TODO@max: Reverse order necessary?
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        Used during decode scheduling to check if we can append a new token to the sequence without preemption.

        Checks if we can append a new token to the sequence.
        Returns True if either we are still in the same block (i.e. len(seq) % block_size != 1),
        or we need to start a new block but there is at least one free block to allocate.
        """
        # If we are about to start a new block (i.e. len(seq) % block_size == 1), we need to allocate a new block for the new token.
        # If we are still in the same block, (i.e. len(seq) % block_size != 1), we can always append without allocation.
        need_new_block = len(seq) % self.block_size == 1
        # If we need a new block, check if there is any free block to allocate.
        return len(self.free_block_ids) >= need_new_block

    def may_append(self, seq: Sequence):
        """
        Used during decode scheduling after can_append returned True to actually allocate a new block if needed.

        Allocates a new block for the sequence if we are about to start a new block (i.e. len(seq) % block_size == 1).
        Otherwise: Computes the hash for the previous block if we have just filled it up with the last token (i.e. len(seq) % block_size == 0),
        and updates the block's hash in the block manager.
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        # If we are about to start a new block, we need to allocate a new block for the new token.
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        # Update the hash of the last block if we have just filled it up with the last token.
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            # use the hash of the previous full block as the prefix hash, if available, otherwise use -1 as the prefix hash.
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        # Clear and keep the hash of the last block cleared (i.e. set to -1)
        else:
            assert last_block.hash == -1
