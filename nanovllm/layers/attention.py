import torch
from torch import nn
import triton
import triton.language as tl

import torch.nn.functional as F
import math

from nanovllm.utils.context import get_context


import torch
from torch import nn
import torch.nn.functional as F
from typing import Any # To stand in for the 'context' object

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def store_kvcache_pytorch(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor
):
    """
    PyTorch-only replacement for the store_kvcache Triton kernel.
    
    Args:
        key: Shape (num_tokens, num_kv_heads, head_dim)
        value: Shape (num_tokens, num_kv_heads, head_dim)
        k_cache: Shape (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: Shape (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: Shape (num_tokens,)
    """
    num_tokens = key.shape[0]
    block_size = 256
    
    for i in range(num_tokens):
        slot_idx = slot_mapping[i].item()
        
        # Skip if slot index is -1 (invalid)
        if slot_idx == -1:
            continue
            
        # Calculate block index and offset within block
        block_idx = slot_idx // block_size
        offset_in_block = slot_idx % block_size
        
        # Copy key and value data to cache
        k_cache[block_idx, offset_in_block] = key[i]
        v_cache[block_idx, offset_in_block] = value[i]

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        block_size=256, # PagedAttention requires a block size
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size # Store block_size
        
        # GQA repetition factor
        self.num_q_per_kv = self.num_heads // self.num_kv_heads
        
        # Caches are initialized empty
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        # NOTE: We pass 'context' directly as an argument 
        # instead of using the global get_context()
        
        # 1. Store new K/V tokens into the cache
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 2. Perform Attention
        if context.is_prefill:
            # --- PREFILL LOGIC ---
            # Handles variable-length sequences (un-padded)
            # We must loop over the batch, as SDPA is batched.
            
            output_list = []
            batch_size = len(context.cu_seqlens_q) - 1
            
            for i in range(batch_size):
                # Get sequence boundaries from cumulative lengths
                q_start, q_end = context.cu_seqlens_q[i], context.cu_seqlens_q[i+1]
                
                # Slice Q for this sequence
                q_seq = q[q_start:q_end] # (seq_len_q, num_q_heads, head_dim)
                seq_len_q = q_seq.shape[0]

                if seq_len_q == 0:
                    continue # Skip empty sequences

                # Add batch dim for SDPA: (1, num_q_heads, seq_len_q, head_dim)
                q_b = q_seq.permute(1, 0, 2).unsqueeze(0)
                
                k_seq = v_seq = None
                
                if context.block_tables is not None:
                    # --- PagedAttention Prefill ---
                    # Gather K/V from paged cache for this sequence
                    
                    k_context_len = (context.cu_seqlens_k[i+1] - context.cu_seqlens_k[i]).item()
                    seq_len_k = k_context_len
                    
                    if seq_len_k == 0:
                        k_seq = torch.empty(0, self.num_kv_heads, self.head_dim, device=q.device, dtype=q.dtype)
                        v_seq = torch.empty(0, self.num_kv_heads, self.head_dim, device=q.device, dtype=q.dtype)
                    else:
                        token_indices = torch.arange(seq_len_k, device=q.device)
                        block_idx_for_token = token_indices // self.block_size
                        physical_block_nums = context.block_tables[i, block_idx_for_token]
                        offset_in_block = token_indices % self.block_size
                        slot_indices = physical_block_nums * self.block_size + offset_in_block
                        
                        # Gather and reshape from (S_k, N_kv*H) to (S_k, N_kv, H)
                        k_seq_flat = k_cache[physical_block_nums, offset_in_block] # LLM generated k_cache[slot_indices] which is incorrect?
                        v_seq_flat = v_cache[physical_block_nums, offset_in_block]
                        k_seq = k_seq_flat.view(seq_len_k, self.num_kv_heads, self.head_dim)
                        v_seq = v_seq_flat.view(seq_len_k, self.num_kv_heads, self.head_dim)
                
                else:
                    # --- Non-Paged Prefill ---
                    # Use the provided K/V tensors directly
                    k_start, k_end = context.cu_seqlens_k[i], context.cu_seqlens_k[i+1]
                    k_seq = k[k_start:k_end] # (seq_len_k, num_kv_heads, head_dim)
                    v_seq = v[k_start:k_end] # (seq_len_k, num_kv_heads, head_dim)
                    seq_len_k = k_seq.shape[0]

                # --- Common Prefill Attention Logic (for this sequence) ---
                
                # Handle GQA: Repeat K/V
                # (S_k, N_kv, H) -> (S_k, N_q, H)
                k_seq_rep = k_seq.unsqueeze(2).expand(-1, -1, self.num_q_per_kv, -1).reshape(seq_len_k, self.num_heads, self.head_dim)
                v_seq_rep = v_seq.unsqueeze(2).expand(-1, -1, self.num_q_per_kv, -1).reshape(seq_len_k, self.num_heads, self.head_dim)

                # Add batch dim for SDPA
                # (1, num_q_heads, seq_len_k, head_dim)
                k_b = k_seq_rep.permute(1, 0, 2).unsqueeze(0)
                v_b = v_seq_rep.permute(1, 0, 2).unsqueeze(0)
                
                # Run SDPA for this single sequence
                # Causal mask is applied automatically
                o_seq_b = F.scaled_dot_product_attention(
                    q_b, k_b, v_b, 
                    scale=self.scale, 
                    is_causal=True # Prefill is always causal
                )
                
                # (1, N_h, S_q, H) -> (S_q, N_h, H)
                o_seq = o_seq_b.squeeze(0).permute(1, 0, 2)
                output_list.append(o_seq)

            # Stack all sequence outputs back into a single un-padded tensor
            if not output_list:
                return torch.empty_like(q)
            
            o = torch.cat(output_list, dim=0)
        
        else:
            # --- DECODE LOGIC ---
            # Input q is (batch_size, num_q_heads, head_dim)
            # This is already batched, so we can do one batched SDPA call.
            
            batch_size = q.shape[0]

            # Reshape q for SDPA: (B, N_q, 1, H)
            q_b = q.unsqueeze(1).permute(0, 2, 1, 3)
            
            # --- PagedAttention Gather (Batched) ---
            # 1. Get max context len in batch
            max_seq_len_k = context.context_lens.max().item()
            
            # 2. Create token indices: (B, max_seq_len_k)
            token_indices = torch.arange(max_seq_len_k, device=q.device).unsqueeze(0).expand(batch_size, -1)
            
            # 3. Find block indices: (B, max_seq_len_k)
            block_idx_for_token = token_indices // self.block_size
            
            # 4. Ensure block indices are within bounds of block table
            max_block_idx = context.block_tables.shape[1] - 1
            block_idx_for_token = torch.clamp(block_idx_for_token, 0, max_block_idx)
            
            # 5. Find physical block numbers: (B, max_seq_len_k)
            # We use gather to select block numbers from the table
            physical_block_nums = torch.gather(context.block_tables, 1, block_idx_for_token)
            
            # 6. Find offsets in block: (B, max_seq_len_k)
            offset_in_block = token_indices % self.block_size
            
            # 7. Calculate final slot indices: (B, max_seq_len_k)
            slot_indices = physical_block_nums * self.block_size + offset_in_block
            
            # 8. Create padding mask: (B, max_seq_len_k)
            # True for valid tokens, False for padding
            valid_token_mask = (token_indices < context.context_lens.unsqueeze(1))
            
            # 9. Ensure slot indices are within cache bounds for valid tokens
            cache_size = k_cache.shape[0] * k_cache.shape[1]
            slot_indices = torch.clamp(slot_indices, 0, cache_size - 1)
            
            # 10. Set slot indices for padding tokens to 0 (to avoid OOB)
            # These will be masked out by attn_mask anyway.
            slot_indices[~valid_token_mask] = 0
            
            # 11. Flatten indices and gather from cache
            # Only gather valid tokens to avoid OOB errors
            valid_slot_indices = slot_indices[valid_token_mask]
            
            # The cache has shape [num_blocks, block_size, num_kv_heads, head_dim]
            # We need to calculate block indices and offsets within blocks
            block_indices = valid_slot_indices // self.block_size
            offsets_in_block = valid_slot_indices % self.block_size
            
            # Gather from the cache using the correct indexing
            k_valid_flat = k_cache[block_indices, offsets_in_block] # (num_valid_tokens, num_kv_heads, head_dim)
            v_valid_flat = v_cache[block_indices, offsets_in_block] # (num_valid_tokens, num_kv_heads, head_dim)
            
            # Flatten the KV heads and head_dim dimensions
            k_valid_flat = k_valid_flat.reshape(-1, self.num_kv_heads * self.head_dim)
            v_valid_flat = v_valid_flat.reshape(-1, self.num_kv_heads * self.head_dim)
            
            # 12. Create output tensors with proper padding
            # Initialize with zeros for padding
            k_past = torch.zeros(batch_size, max_seq_len_k, self.num_kv_heads * self.head_dim, 
                               device=k_cache.device, dtype=k_cache.dtype)
            v_past = torch.zeros(batch_size, max_seq_len_k, self.num_kv_heads * self.head_dim,
                               device=v_cache.device, dtype=v_cache.dtype)
            
            # Fill in the valid tokens - we need to flatten the mask to match the gathered data
            flat_valid_mask = valid_token_mask.view(-1)
            k_past2 = k_past.view(-1, self.num_kv_heads * self.head_dim)
            k_past2[flat_valid_mask] = k_valid_flat
            v_past2 = v_past.view(-1, self.num_kv_heads * self.head_dim)
            v_past2[flat_valid_mask] = v_valid_flat
            
            # Reshape to final shape (B, S_k, N_kv, H)
            k_past = k_past2.view(batch_size, max_seq_len_k, self.num_kv_heads, self.head_dim)
            v_past = v_past2.view(batch_size, max_seq_len_k, self.num_kv_heads, self.head_dim)
            
            # --- End Gather ---

            # Handle GQA: Repeat K/V
            # (B, S_k, N_kv, H) -> (B, S_k, N_q, H)
            k_past_rep = k_past.unsqueeze(3).expand(-1, -1, -1, self.num_q_per_kv, -1).reshape(batch_size, max_seq_len_k, self.num_heads, self.head_dim)
            v_past_rep = v_past.unsqueeze(3).expand(-1, -1, -1, self.num_q_per_kv, -1).reshape(batch_size, max_seq_len_k, self.num_heads, self.head_dim)
            
            # Permute for SDPA: (B, N_q, S_k, H)
            k_b = k_past_rep.permute(0, 2, 1, 3)
            v_b = v_past_rep.permute(0, 2, 1, 3)
            
            # Create attention mask from the valid token mask
            # SDPA needs mask where True means "mask out"
            # (B, S_k) -> (B, 1, 1, S_k)
            attn_mask = valid_token_mask #? LLM generate this line as attn_mask = ~valid_token_mask and generated rubbish content.
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2) 
            
            # Run Batched SDPA
            # q: (B, N_q, 1, H), k/v: (B, N_q, S_k, H), mask: (B, 1, 1, S_k)
            # is_causal=False because S_q=1 and we provide an explicit mask
            o_b = F.scaled_dot_product_attention(
                q_b, k_b, v_b, 
                attn_mask=attn_mask,
                scale=self.scale, 
                is_causal=False 
            )
            
            # Reshape output: (B, N_q, 1, H) -> (B, 1, N_q, H) -> (B, N_q, H)
            o = o_b.permute(0, 2, 1, 3).squeeze(1)

        return o
