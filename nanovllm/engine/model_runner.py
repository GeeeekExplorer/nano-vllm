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

    def __init__(self, config: Config, rank: int = 0, event: Event | list[Event] = None, device_id: int = None, is_decode_runner: bool = False):
        """
        Initialize ModelRunner.

        Args:
            config: Model configuration
            rank: Rank for tensor parallel mode
            event: Event for tensor parallel communication
            device_id: Override device ID (for two-GPU PD mode)
            is_decode_runner: Whether this is the decode runner (enables M2 pipeline)
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.is_decode_runner = is_decode_runner

        # Determine device: use device_id if specified, otherwise use rank
        self.device_id = device_id if device_id is not None else rank

        # Initialize distributed
        # Note: In two-GPU PD mode, LLMEngine initializes dist before creating runners
        if self.world_size > 1:
            # Multi-GPU tensor parallel mode
            if not dist.is_initialized():
                dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        else:
            # Single process mode (two-GPU PD or single-GPU)
            # Requires dist for Parallel layers (VocabParallelEmbedding, etc.)
            if not dist.is_initialized():
                dist.init_process_group("gloo", init_method="tcp://localhost:29500", world_size=1, rank=0)

        torch.cuda.set_device(self.device_id)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        # IMPORTANT: Set default device to specific GPU, not just "cuda"
        torch.set_default_device(f"cuda:{self.device_id}")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        # Explicitly move model to correct device (handles shared/cached modules like RoPE)
        self.model = self.model.to(f"cuda:{self.device_id}")
        self.sampler = Sampler()
        self.allocate_kv_cache()  # Must allocate before warmup!
        self.warmup_model() #预热模型
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu") # 恢复成CPU
        torch.set_default_dtype(default_dtype)

        # Initialize M2 pipeline (only for decode runner)
        self.pipeline_scheduler = None
        if is_decode_runner and config.enable_decode_pipeline:
            try:
                from nanovllm.engine.green_manager import GreenManager
                from nanovllm.engine.pipeline_scheduler import PipelineScheduler

                print(f"[ModelRunner] Initializing decode pipeline on GPU:{self.device_id}")
                print(f"  - Attention SMs: {config.decode_attention_sm}")
                print(f"  - FFN SMs: {config.decode_ffn_sm}")

                device = torch.device(f"cuda:{self.device_id}")
                green_manager = GreenManager(
                    device=device,
                    attention_sm=config.decode_attention_sm,
                    ffn_sm=config.decode_ffn_sm,
                    enable_rebalance=False,  # M3 feature
                )
                self.pipeline_scheduler = PipelineScheduler(
                    model=self.model,
                    green_manager=green_manager,
                    enable_profiling=config.decode_pipeline_profiling,
                )
                print(f"[ModelRunner] Decode pipeline initialized successfully")
            except Exception as e:
                print(f"[ModelRunner] Failed to initialize decode pipeline: {e}")
                print(f"[ModelRunner] Falling back to sequential decode")
                self.pipeline_scheduler = None

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        # Don't destroy process group here in two-GPU mode
        # LLMEngine will handle it after both runners exit
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # Allocate blocks for warmup sequences (needed since KV cache is now allocated before warmup)
        for seq in seqs:
            num_blocks = (max_model_len + self.block_size - 1) // self.block_size
            seq.block_table = list(range(num_blocks))  # Dummy block allocation for warmup
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize

        # Calculate available memory for KV cache. Fall back to actual free memory
        # if the heuristic based on total/used reports a negative budget (happens
        # when other processes already occupy most of the device).
        available_memory = int(total * config.gpu_memory_utilization - used - peak + current)
        fallback_available = int(free * config.gpu_memory_utilization)
        available_memory = max(available_memory, fallback_available)
        config.num_kvcache_blocks = available_memory // block_bytes

        # Debug info if allocation fails
        if config.num_kvcache_blocks <= 0:
            print(f"[ModelRunner GPU:{self.device_id}] KV cache allocation failed:")
            print(f"  Total GPU memory: {total / 1e9:.2f} GB")
            print(f"  Used memory: {used / 1e9:.2f} GB")
            print(f"  Free memory: {free / 1e9:.2f} GB")
            print(f"  Peak allocated: {peak / 1e9:.2f} GB")
            print(f"  Current allocated: {current / 1e9:.2f} GB")
            print(f"  Utilization target: {config.gpu_memory_utilization}")
            print(f"  Available for KV: {available_memory / 1e9:.2f} GB")
            print(f"  Block size: {block_bytes / 1e6:.2f} MB")
            raise RuntimeError(f"Cannot allocate KV cache: num_blocks={config.num_kvcache_blocks}. "
                             f"Try reducing gpu_memory_utilization or max_model_len.")
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim, device=f"cuda:{self.device_id}")
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def sync_kv_cache_to(self, target_device: int, block_indices: list[int]):
        """
        Synchronize specific KV cache blocks to target device.
        Used for two-GPU PD separation: GPU0 -> GPU1.

        Args:
            target_device: Target GPU device ID
            block_indices: List of block indices to synchronize
        """
        if not block_indices:
            return

        # Synchronize specified blocks to target device
        with torch.cuda.device(self.device_id):
            for block_idx in block_indices:
                # Copy K and V cache for this block to target device
                self.kv_cache[:, :, block_idx].data = self.kv_cache[:, :, block_idx].to(f"cuda:{target_device}", non_blocking=True)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(self.device_id, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(self.device_id, non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(self.device_id, non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(self.device_id, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(self.device_id, non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(self.device_id, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(self.device_id, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(self.device_id, non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(self.device_id, non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(self.device_id, non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_prefill(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """
        Execute prefill stage only.
        This is the entry point for prefill in PD-separated mode.
        """
        return self.model.compute_logits(self.model(input_ids, positions))

    @torch.inference_mode()
    def run_decode_core(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """
        Execute decode stage only.
        This is the single entry point for decode in PD-separated mode.
        Supports M2 pipeline (Attention/FFN on separate SMs).
        """
        # Use pipeline if enabled (M2)
        if self.pipeline_scheduler is not None:
            return self.model.compute_logits(
                self.pipeline_scheduler.decode_token(input_ids, positions)
            )

        # Standard decode path with CUDA graph optimization
        if self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """Legacy unified interface for compatibility"""
        if is_prefill:
            return self.run_prefill(input_ids, positions)
        else:
            return self.run_decode_core(input_ids, positions)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def get_pipeline_statistics(self) -> dict:
        """Get M2 pipeline statistics if enabled"""
        if self.pipeline_scheduler is not None:
            return self.pipeline_scheduler.get_statistics()
        return {"enabled": False}

    def print_pipeline_statistics(self):
        """Print M2 pipeline statistics"""
        stats = self.get_pipeline_statistics()
        enabled = stats.get("enabled", False)

        print("\n[M2 Pipeline Statistics - GPU1 Decode Card]")
        if enabled:
            print(f"  Attention SMs: {stats['attention_sm']}")
            print(f"  FFN SMs: {stats['ffn_sm']}")
        else:
            print("  Green Context unavailable → running sequential decode")
            print(f"  Requested Attention SMs: {stats.get('attention_sm', 'N/A')}")
            print(f"  Requested FFN SMs: {stats.get('ffn_sm', 'N/A')}")

        if stats.get("total_tokens", 0) > 0:
            print(f"  Total tokens profiled: {stats['total_tokens']}")
            print(f"  Avg attention time: {stats['avg_attention_time']*1000:.2f} ms")
            print(f"  Avg FFN time: {stats['avg_ffn_time']*1000:.2f} ms")
            print(f"  Avg total time: {stats['avg_total_time']*1000:.2f} ms")
        else:
            print("  Timing profile: disabled (set decode_pipeline_profiling=True to enable)")

        if stats.get("comm_tokens", 0) > 0:
            print(f"  Tokens sampled for comm stats: {stats['comm_tokens']}")
            avg_comm_mb = stats["avg_comm_bytes_per_token"] / (1024**2)
            print(f"  Avg A→F payload: {avg_comm_mb:.4f} MiB/token")
            layer_comm = stats.get("layer_comm", [])
            if layer_comm:
                print("  Layer payload per token:")
                for entry in layer_comm:
                    attn_mb = entry["attention_bytes_per_token"] / (1024**2)
                    residual_mb = entry["residual_bytes_per_token"] / (1024**2)
                    total_mb = entry["total_bytes_per_token"] / (1024**2)
                    print(
                        f"    Layer {entry['layer_idx']:02d}: {total_mb:.4f} MiB "
                        f"(Attn {attn_mb:.4f} + Residual {residual_mb:.4f})"
                    )
            snapshot = stats.get("last_comm_snapshot")
            if snapshot:
                total_bytes = snapshot.get("total_bytes", 0)
                total_mb = total_bytes / (1024**2)
                print(
                    "  Last batch payload: "
                    f"{total_mb:.4f} MiB for batch_size={snapshot.get('batch_size', 'N/A')} "
                    f"(dtype={snapshot.get('dtype', 'unknown')})"
                )
        else:
            print("  Communication stats: no tokens processed yet")

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
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
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
