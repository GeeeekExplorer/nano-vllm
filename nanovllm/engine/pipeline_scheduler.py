"""
Pipeline Scheduler for Attention/FFN two-stage execution on GPU1 (M2).

Coordinates the execution of Attention and FFN stages on separate SM partitions
using Green Context streams.
"""
import torch
import time
from collections import defaultdict
from typing import Optional
from nanovllm.engine.green_manager import GreenManager


class PipelineScheduler:
    """
    Manages two-stage pipeline execution for decode on GPU1.

    Pipeline stages:
    1. Attention Stage: KV lookup + Attention computation
    2. FFN Stage: Feed-forward network + normalization

    These stages execute concurrently on different SM partitions using Green Context.
    """

    def __init__(
        self,
        model,
        green_manager: GreenManager,
        enable_profiling: bool = False,
    ):
        """
        Initialize PipelineScheduler.

        Args:
            model: Qwen3ForCausalLM model instance
            green_manager: GreenManager instance for SM allocation
            enable_profiling: Enable timing statistics
        """
        self.model = model
        self.green_manager = green_manager
        self.enable_profiling = enable_profiling

        # Statistics
        self.total_tokens = 0
        self.total_attention_time = 0.0
        self.total_ffn_time = 0.0
        self.total_pipeline_time = 0.0
        self.total_comm_tokens = 0
        self.total_comm_bytes = 0
        self.layer_comm_totals = defaultdict(
            lambda: {"attention_bytes": 0, "residual_bytes": 0, "total_bytes": 0}
        )
        self.last_comm_snapshot: Optional[dict] = None

    def decode_token(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute decode through the two-stage pipeline.

        Args:
            input_ids: Input token IDs [batch_size]
            positions: Token positions [batch_size]

        Returns:
            Hidden states tensor [batch_size, hidden_size]
        """
        if not self.green_manager.enabled:
            return self._execute_without_green_ctx(input_ids, positions)

        # Pipeline execution with profiling
        start_time = time.perf_counter() if self.enable_profiling else 0.0

        # Stage 1: Attention (on attention stream)
        attention_start = time.perf_counter() if self.enable_profiling else 0.0
        attention_event = torch.cuda.Event()

        with torch.cuda.stream(self.green_manager.get_attention_stream()):
            hidden_states, residual, residuals, comm_metadata = self.model.forward_attention_stage(
                input_ids, positions
            )
            attention_event.record()
        self._update_comm_counters(comm_metadata)

        if self.enable_profiling:
            self.green_manager.get_attention_stream().synchronize()
            attention_time = time.perf_counter() - attention_start

        # Stage 2: FFN (on FFN stream, waits for attention to complete)
        ffn_start = time.perf_counter() if self.enable_profiling else 0.0
        ffn_event = torch.cuda.Event()

        with torch.cuda.stream(self.green_manager.get_ffn_stream()):
            # Wait for attention stage to complete
            attention_event.wait()

            # Execute FFN stage
            hidden_states = self.model.forward_ffn_stage(
                hidden_states, residual, residuals
            )
            ffn_event.record()

        # Wait for pipeline completion
        ffn_event.synchronize()

        if self.enable_profiling:
            ffn_time = time.perf_counter() - ffn_start
            total_time = time.perf_counter() - start_time
            self._update_statistics(attention_time, ffn_time, total_time, input_ids.size(0))

        return hidden_states

    def _execute_without_green_ctx(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sequential fallback when Green Context is unavailable.

        Still routes through attention/FFN split so we can collect statistics.
        """
        start_time = time.perf_counter() if self.enable_profiling else 0.0

        attention_start = time.perf_counter() if self.enable_profiling else 0.0
        hidden_states, residual, residuals, comm_metadata = self.model.forward_attention_stage(
            input_ids, positions
        )
        self._update_comm_counters(comm_metadata)
        if self.enable_profiling:
            attention_time = time.perf_counter() - attention_start

        ffn_start = time.perf_counter() if self.enable_profiling else 0.0
        hidden_states = self.model.forward_ffn_stage(hidden_states, residual, residuals)
        if self.enable_profiling:
            ffn_time = time.perf_counter() - ffn_start
            total_time = time.perf_counter() - start_time
            self._update_statistics(attention_time, ffn_time, total_time, input_ids.size(0))

        return hidden_states

    def _update_statistics(
        self,
        attention_time: float,
        ffn_time: float,
        total_time: float,
        batch_size: int,
    ):
        """Update profiling statistics"""
        self.total_tokens += batch_size
        self.total_attention_time += attention_time
        self.total_ffn_time += ffn_time
        self.total_pipeline_time += total_time

    def get_statistics(self) -> dict:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with timing and throughput metrics
        """
        gm_stats = self.green_manager.stats()
        stats = {
            "enabled": self.green_manager.enabled,
            "total_tokens": self.total_tokens,
            "avg_attention_time": 0.0,
            "avg_ffn_time": 0.0,
            "avg_total_time": 0.0,
            "attention_sm": gm_stats.get("attention_sm", 0),
            "ffn_sm": gm_stats.get("ffn_sm", 0),
            "rebalance_count": gm_stats.get("rebalance_count", 0),
            "comm_tokens": self.total_comm_tokens,
            "avg_comm_bytes_per_token": (
                self.total_comm_bytes / self.total_comm_tokens
                if self.total_comm_tokens > 0
                else 0.0
            ),
            "layer_comm": self._format_layer_comm_statistics(),
            "last_comm_snapshot": self.last_comm_snapshot,
        }
        if self.total_tokens > 0:
            stats["avg_attention_time"] = self.total_attention_time / self.total_tokens
            stats["avg_ffn_time"] = self.total_ffn_time / self.total_tokens
            stats["avg_total_time"] = self.total_pipeline_time / self.total_tokens
        return stats

    def reset_statistics(self):
        """Reset profiling statistics"""
        self.total_tokens = 0
        self.total_attention_time = 0.0
        self.total_ffn_time = 0.0
        self.total_pipeline_time = 0.0
        self.total_comm_tokens = 0
        self.total_comm_bytes = 0
        self.layer_comm_totals = defaultdict(
            lambda: {"attention_bytes": 0, "residual_bytes": 0, "total_bytes": 0}
        )
        self.last_comm_snapshot = None

    def _update_comm_counters(self, metadata: Optional[dict]):
        """Track how much data attention→FFN separation needs to communicate."""
        if not metadata:
            return
        batch_size = int(metadata.get("batch_size", 0))
        total_bytes = int(metadata.get("total_bytes", 0))
        per_layer = metadata.get("per_layer", [])
        if batch_size <= 0:
            return
        self.total_comm_tokens += batch_size
        self.total_comm_bytes += total_bytes
        for entry in per_layer:
            layer_idx = entry.get("layer_idx")
            if layer_idx is None:
                continue
            totals = self.layer_comm_totals[layer_idx]
            totals["attention_bytes"] += int(entry.get("attention_bytes", 0))
            totals["residual_bytes"] += int(entry.get("residual_bytes", 0))
            totals["total_bytes"] += int(entry.get("total_bytes", 0))
        self.last_comm_snapshot = {
            "batch_size": batch_size,
            "hidden_size": metadata.get("hidden_size"),
            "dtype": metadata.get("dtype"),
            "total_bytes": total_bytes,
            "per_layer": per_layer,
        }

    def _format_layer_comm_statistics(self) -> list[dict]:
        """Return normalized per-layer communication stats."""
        if self.total_comm_tokens <= 0:
            return []
        normalized = []
        denom = self.total_comm_tokens
        for layer_idx in sorted(self.layer_comm_totals.keys()):
            totals = self.layer_comm_totals[layer_idx]
            normalized.append(
                {
                    "layer_idx": layer_idx,
                    "attention_bytes_per_token": totals["attention_bytes"] / denom,
                    "residual_bytes_per_token": totals["residual_bytes"] / denom,
                    "total_bytes_per_token": totals["total_bytes"] / denom,
                }
            )
        return normalized
