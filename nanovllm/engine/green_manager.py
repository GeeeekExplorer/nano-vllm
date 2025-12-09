"""
Green Context Manager for GPU1 Attention/FFN pipeline (M2).

Manages SM-level resource allocation for the decode card's two-stage pipeline.
"""
import torch
import sys
import os
from typing import Optional, Tuple

# Import green context utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from green_ctx import split_device_green_ctx_by_sm_count


class GreenManager:
    """
    Manages Green Context resources for Attention/FFN pipeline on GPU1.

    This manager:
    - Splits GPU1's SMs between Attention and FFN stages
    - Creates separate CUDA streams for each stage
    - Provides rebalance capability for M3 (disabled in M2)
    """

    def __init__(
        self,
        device: torch.device,
        attention_sm: int = 16,
        ffn_sm: int = 16,
        enable_rebalance: bool = False,
    ):
        """
        Initialize GreenManager for GPU1.

        Args:
            device: CUDA device (should be decode_device_id from config)
            attention_sm: SM count for attention stage
            ffn_sm: SM count for FFN stage
            enable_rebalance: Enable dynamic rebalancing (M3 feature, disabled in M2)
        """
        self.device = device
        self.attention_sm = attention_sm
        self.ffn_sm = ffn_sm
        self.enable_rebalance = enable_rebalance
        self.enabled = False
        self.rebalance_count = 0

        # Try to initialize Green Context
        try:
            self._allocate_resources(attention_sm, ffn_sm)
            self.enabled = True
            print(f"[GreenManager] Initialized on {device}")
            print(f"  - Attention stage: {attention_sm} SMs")
            print(f"  - FFN stage: {ffn_sm} SMs")
        except RuntimeError as e:
            print(f"[GreenManager] Failed to initialize: {e}")
            print(f"[GreenManager] Falling back to sequential decode")
            self.attention_stream = torch.cuda.default_stream(device)
            self.ffn_stream = torch.cuda.default_stream(device)

    def _allocate_resources(self, attention_sm: int, ffn_sm: int):
        """Allocate SM resources using Green Context"""
        sm_counts = [attention_sm, ffn_sm]
        streams, resources = split_device_green_ctx_by_sm_count(self.device, sm_counts)

        self.attention_stream = streams[0]
        self.ffn_stream = streams[1]
        self.remaining_stream = streams[2]
        self.resources = resources

        # Log actual allocation
        actual_sm_counts = [r.sm.smCount for r in resources]
        print(f"[GreenManager] Actual SM allocation: Attn={actual_sm_counts[0]}, FFN={actual_sm_counts[1]}, Remaining={actual_sm_counts[2]}")

    def allocate(self, attention_sm: int, ffn_sm: int) -> Tuple[torch.cuda.Stream, torch.cuda.Stream]:
        """
        Get streams for attention and FFN stages.

        Args:
            attention_sm: Requested SM count for attention (ignored in M2, uses constructor value)
            ffn_sm: Requested SM count for FFN (ignored in M2, uses constructor value)

        Returns:
            (attention_stream, ffn_stream)
        """
        return self.attention_stream, self.ffn_stream

    def rebalance(self, attention_sm: int, ffn_sm: int) -> None:
        """
        Dynamically adjust SM allocation (M3 feature).

        In M2, this is a no-op. M3 will implement dynamic reallocation.

        Args:
            attention_sm: New SM count for attention
            ffn_sm: New SM count for FFN
        """
        if not self.enable_rebalance or not self.enabled:
            return

        # M3 will implement this
        print(f"[GreenManager] Rebalance requested: Attn={attention_sm}, FFN={ffn_sm} (not implemented in M2)")

    def stats(self) -> dict:
        """
        Get current statistics.

        Returns:
            Dictionary with current state and allocation info
        """
        return {
            "enabled": self.enabled,
            "attention_sm": self.attention_sm,
            "ffn_sm": self.ffn_sm,
            "rebalance_count": self.rebalance_count,
        }

    def get_attention_stream(self) -> torch.cuda.Stream:
        """Get CUDA stream for attention stage"""
        return self.attention_stream

    def get_ffn_stream(self) -> torch.cuda.Stream:
        """Get CUDA stream for FFN stage"""
        return self.ffn_stream

    def synchronize(self):
        """Synchronize all streams"""
        if self.enabled:
            self.attention_stream.synchronize()
            self.ffn_stream.synchronize()
        else:
            torch.cuda.synchronize(self.device)
