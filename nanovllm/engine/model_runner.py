"""
ModelRunner for the nano-vLLM engine.

This module defines the ModelRunner class, which is responsible for running the model inference.

The ModelRunner supports tensor parallelism by running multiple processes that each handle a part of the model. The main process (rank=0) also runs a ModelRunner, and it communicates with the other ModelRunners (rank > 0) through shared memory and events.

# Inter-process communication:

The setup for the ModelRunner processes and inter-process communication is done in the LLMEngine class, which creates the ModelRunner instances and manages the communication between them.
On a high level a process is spawned for each tensor parallel rank (i.e. GPU), and each process runs a ModelRunner instance.
The other processes (rank > 0) receive each an event and their rank as arguments, while the main process (rank=0) receives a list of events for communication with the other processes.

The main process (rank=0) runs the scheduler and calls the ModelRunner's call method to execute methods on the ModelRunner instances. If the world size is greater than 1, the call method of the main process (rank=0) writes the method name and arguments to shared memory and sets the events to notify the other processes. The other processes (rank > 0) wait for the event to be set, read the method name and arguments from shared memory, and then execute the corresponding method.

How do the different processes get the sequences to run the model on?
The main process (rank=0) sends the sequences to all other processes (rank > 0) through shared memory.

# Tensor parallelism:

See model implementation.
Only rank=0 returns the logits, other ranks return None.




"""

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        assert hf_config is not None
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event: Event | list[Event] = event

        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # SharedMemory (CPU) setup for inter-process communication, e.g. for the sequences
        if self.world_size > 1:
            # The main process (rank=0) will create the shared memory and the other processes will connect to it.
            if rank == 0:
                # Size of 2**20B ~ 1 MB should be sufficient for the method name and arguments.
                # 1 int object in Python takes 28 bytes, so we can fit around 2**20 / 28 ~ 37k integers in the shared memory, which should be enough for the method name and arguments.
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                # Others wait for the main process to create the shared memory and then connect to it.
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # Then they will wait for the main process to send the method name and arguments through shared memory and execute the corresponding method in the loop() method.
                self.loop()

    def exit(self):
        """Cleans up the ModelRunner process and shared memory."""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """Loop for the other processes (rank > 0) to wait for method calls from the main process (rank=0) and execute them."""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """Read method name and arguments from shared memory for the other processes (rank > 0)."""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """Write method name and arguments to shared memory for the main process (rank=0)."""
        assert self.world_size > 1 and self.rank == 0
        assert isinstance(self.event, list) and len(self.event) == self.world_size - 1
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """Call a method by name with the given arguments.
        If rank=0 and world_size > 1, notify other processes.
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """Warmup the model with max sequence length prefill to collect GPU memory statistics."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """Allocate the kv cache for all layers and heads based on the GPU memory statistics collected during warmup.
        Assign the corresponding slice of the allocated kv cache to each attention module in the model.

        The KV cache blocks are used by different sequences during prefill and decode,
        and the mapping between the actual cache slots in the allocated kv cache and the tokens in the input sequences
        is created in prepare_prefill() and prepare_decode().
        """
        config = self.config
        hf_config = config.hf_config
        # GPU memory info
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # Number of key-value heads per GPU
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        # size of one block of kv cache for all layers and all heads, in bytes
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )
        # calculate number of block to allocate for kv cache
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,  # Will be distributed over batch and sequence dim
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        # Set the kv cache for each attention module to the corresponding slice of the allocated kv cache.
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        """Creates a block table tensor of shape (N_seq, max_num_blocks).
        Padded by -1, where N_seq is the number of sequences in the batch and max_num_blocks is the maximum number of blocks among the sequences in the batch.
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(
        self, seqs: list[Sequence]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares the input_ids, positions, and the context for the prefill step or (decode with prefix cache) based on the given sequences.
        The context comprises the slot mapping for the kv cache and the cumulative sequence lengths for flash attention with block tables.
        In case of decode with prefix cache, the block tables for flash attention with block tables are also updated in the context.

        Returns:
        - input_ids (N_tok,): The input ids for the prefill step, which are the concatenated tokens of all sequences in the batch.
        - positions (N_tok,): The positions of the tokens in each sequence, which are the concatenated ranges of the lengths of each sequence in the batch.
        """
        # All concatenated input ids for the prefill step.
        input_ids: list[int] = []
        # Sequence positions per sequence, same length as input_ids, used for the rotary embedding.
        positions: list[int] = []
        # Cumulative sequence lengths for q and k, used for the flash attention with block tables.
        # The first element is 0, and the i-th element is the cumulative sequence length of the first i sequences in the batch.
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        # Maximum seqlens
        max_seqlen_q = 0
        max_seqlen_k = 0
        # Mapping between the actual cache slots in the (allocated) kv cache and the tokens in the input_ids.
        slot_mapping: list[int] = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq.token_ids[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        # In case we have a prefix cache (i.e. cu_seqlens_k[-1] > cu_seqlens_q[-1]),
        # we need to prepare the block tables for flash attention with block tables.
        # block_tables (N_seq, max_num_blocks): the block table for each sequence, where max_num_blocks is the maximum number of blocks among the sequences in the batch (padded by -1).
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        # Convert to tensors and move to GPU (non-blocking)
        # Tensors in CPU RAM are in pin memory, so that they can be transferred to GPU asynchronously with non_blocking=True.
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        # set the context for flash attention with block tables
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares the input_ids, positions, and the context for the decode step based on the given sequences.

        Returns:
        - input_ids (N_tok,): The input ids for the decode step, which are the last tokens of each sequence in the batch.
        - positions (N_tok,): The positions of the tokens in each sequence, which are the lengths of each sequence in the batch minus one (since the position is 0-indexed).

        """
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        context_lens: list[int] = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ) -> torch.Tensor:
        """
        Runs the model forward pass either in prefill or decode mode.
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = (
                context.block_tables
            )
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int] | None:
        """Runs one either prefill or decode inference step for the given sequences."""
        # Prepare prefill or decode step
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )

        # Enable different sampling paramters for each sequence
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # Run the model and get the logits. Only rank=0 will return the logits, other ranks will return None.
        logits = self.run_model(input_ids, positions, is_prefill)

        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
