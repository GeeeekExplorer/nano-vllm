from dataclasses import dataclass

import numpy as np
import torch

from nanovllm.engine.sequence import Sequence


@dataclass(frozen=True, slots=True)
class DecodeBatchView:
    batch_size: int
    graph_bucket: int
    active_block_width: int
    padding_rows: int


class DecodeInputBatch:
    """Persistent CPU staging buffers for CUDA graph decode metadata."""

    def __init__(
        self,
        max_batch_size: int,
        max_num_blocks: int,
        block_size: int,
        *,
        pin_memory: bool = True,
        allocate_temperature_gpu: bool = True,
        device: torch.device | str = "cuda",
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if max_num_blocks < 1:
            raise ValueError("max_num_blocks must be positive")
        if block_size < 1:
            raise ValueError("block_size must be positive")

        self.max_batch_size = max_batch_size
        self.max_num_blocks = max_num_blocks
        self.block_size = block_size
        factory = {"device": "cpu", "pin_memory": pin_memory}
        self.input_ids_cpu = torch.empty(max_batch_size, dtype=torch.int64, **factory)
        self.positions_cpu = torch.empty(max_batch_size, dtype=torch.int64, **factory)
        self.slot_mapping_cpu = torch.empty(max_batch_size, dtype=torch.int32, **factory)
        self.context_lens_cpu = torch.empty(max_batch_size, dtype=torch.int32, **factory)
        self.block_tables_cpu = torch.empty(
            max_batch_size, max_num_blocks, dtype=torch.int32, **factory
        )
        self.temperatures_cpu = torch.empty(max_batch_size, dtype=torch.float32, **factory)
        self.temperatures_gpu = (
            torch.empty(max_batch_size, dtype=torch.float32, device=device)
            if allocate_temperature_gpu
            else None
        )

        # NumPy views share the pinned tensor storage and make bulk CPU writes cheap.
        self.input_ids_np = self.input_ids_cpu.numpy()
        self.positions_np = self.positions_cpu.numpy()
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.context_lens_np = self.context_lens_cpu.numpy()
        self.block_tables_np = self.block_tables_cpu.numpy()
        self.temperatures_np = self.temperatures_cpu.numpy()
        self.last_view: DecodeBatchView | None = None

    def prepare_decode(
        self,
        seqs: list[Sequence],
        graph_bucket: int,
        include_temperatures: bool,
    ) -> DecodeBatchView:
        batch_size = len(seqs)
        if batch_size < 1:
            raise ValueError("decode batch cannot be empty")
        if batch_size > graph_bucket or graph_bucket > self.max_batch_size:
            raise ValueError(
                f"invalid graph bucket {graph_bucket} for batch size {batch_size} "
                f"and capacity {self.max_batch_size}"
            )

        active_block_width = max(len(seq.block_table) for seq in seqs)
        if active_block_width < 1:
            raise ValueError("decode sequences must own at least one KV-cache block")
        if active_block_width > self.max_num_blocks:
            raise ValueError(
                f"block-table width {active_block_width} exceeds capacity {self.max_num_blocks}"
            )

        # Initialize the entire captured bucket so padded graph lanes cannot observe
        # metadata left by a previous, larger batch.
        self.input_ids_np[:graph_bucket] = 0
        self.positions_np[:graph_bucket] = 0
        self.slot_mapping_np[:graph_bucket] = -1
        self.context_lens_np[:graph_bucket] = 0
        for row, seq in enumerate(seqs):
            sequence_length = len(seq)
            self.input_ids_np[row] = seq.last_token
            self.positions_np[row] = sequence_length - 1
            self.context_lens_np[row] = sequence_length
            self.slot_mapping_np[row] = (
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
            if include_temperatures:
                self.temperatures_np[row] = seq.temperature

        active_tables = self.block_tables_np[:graph_bucket, :active_block_width]
        active_tables.fill(-1)
        for row, seq in enumerate(seqs):
            active_tables[row, : len(seq.block_table)] = seq.block_table

        view = DecodeBatchView(
            batch_size=batch_size,
            graph_bucket=graph_bucket,
            active_block_width=active_block_width,
            padding_rows=graph_bucket - batch_size,
        )
        self.last_view = view
        return view

    def copy_temperatures_to_gpu(self, batch_size: int) -> torch.Tensor:
        if self.temperatures_gpu is None:
            raise RuntimeError("temperature GPU storage was not allocated")
        if batch_size < 1 or batch_size > self.max_batch_size:
            raise ValueError(f"invalid temperature batch size {batch_size}")
        target = self.temperatures_gpu[:batch_size]
        target.copy_(self.temperatures_cpu[:batch_size], non_blocking=True)
        return target
