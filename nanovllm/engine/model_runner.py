import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.batch_metadata import decode_metadata, padded_block_tables, prefill_metadata
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.runtime import DeviceRuntime
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.spec_sampler import sampling_tensors
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

MAX_GRAPH_BATCH_SIZE = 512
EAGER_DECODE_BATCH_THRESHOLD = 512
# RMSNorm and RoPE are torch.compile'd methods shared by every layer of the target and the draft, called with
# 2-D and 3-D inputs, 32 or 8 heads, and every batch length. Past the default limit of 8 recompiles, dynamo runs
# new shapes uncompiled: slower, and numerically unlike the compiled kernels inside the CUDA graphs.
torch._dynamo.config.recompile_limit = 64


def blocks_for_budget(budget_bytes: int, block_bytes: int) -> int:
    num_blocks = budget_bytes // block_bytes
    if num_blocks == 0:
        raise ValueError("KV budget cannot hold one complete block")
    return num_blocks


class ModelRunner:
    """Runs one model. `kv_role` selects which KVState of a Sequence its KV cache belongs to."""

    def __init__(self, config: Config, rank: int, event: Event | list[Event],
                 *, runtime: DeviceRuntime | None = None, defer_cache: bool = False, kv_role: str = "target"):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.kv_role = kv_role
        # The target of a speculative engine owns one zeroed block that padding rows of the
        # verification graph read; no sequence is ever assigned it.
        self.has_pad_block = kv_role == "target" and config.num_speculative_tokens > 0
        self.pad_block_id = -1

        if runtime is not None and (self.world_size != 1 or config.kv_cache_memory_bytes is None):
            raise ValueError("Shared runners require TP=1 and an explicit KV budget")
        if defer_cache and runtime is None:
            raise ValueError("Deferred initialization requires a shared runtime")
        self._owns_runtime = runtime is None
        self.runtime = runtime if runtime is not None else DeviceRuntime(rank, self.world_size)
        self._exited = False
        self._cache_initialized = False
        self.runtime.attach(self)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.generator = torch.Generator(device=self.device)
        if config.seed is not None:
            self.generator.manual_seed(config.seed)
        default_dtype = torch.get_default_dtype()
        default_device = torch.get_default_device()
        try:
            torch.set_default_dtype(hf_config.dtype)
            torch.set_default_device("cuda")
            self.model = Qwen3ForCausalLM(hf_config)
            load_model(self.model, config.model)
            self.sampler = Sampler()
            if not defer_cache:
                self.initialize_cache()
            if self.world_size > 1:
                if rank == 0:
                    self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                    dist.barrier()
                else:
                    dist.barrier()
                    self.shm = SharedMemory(name="nanovllm")
        finally:
            torch.set_default_device(default_device)
            torch.set_default_dtype(default_dtype)
        if self.world_size > 1 and rank > 0:
            self.loop()

    def initialize_cache(self):
        assert not self._exited and not self._cache_initialized
        default_dtype = torch.get_default_dtype()
        default_device = torch.get_default_device()
        try:
            torch.set_default_dtype(self.config.hf_config.dtype)
            torch.set_default_device("cuda")
            self.warmup_model()
            self.allocate_kv_cache()
            if not self.enforce_eager:
                self.capture_cudagraph()
            self._cache_initialized = True
        finally:
            reset_context()
            torch.set_default_device(default_device)
            torch.set_default_dtype(default_dtype)

    def exit(self):
        if self._exited:
            return
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager and self._cache_initialized:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        self.runtime.detach(self)
        self._exited = True
        if self._owns_runtime:
            self.runtime.close()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
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
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.kv(self.kv_role).num_scheduled_tokens = seq_len
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
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        if config.kv_cache_memory_bytes is None:
            config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        else:
            config.num_kvcache_blocks = blocks_for_budget(config.kv_cache_memory_bytes, block_bytes)
        assert config.num_kvcache_blocks > 0
        num_blocks = config.num_kvcache_blocks + int(self.has_pad_block)
        if config.kv_cache_memory_bytes is not None and num_blocks * block_bytes > free:
            raise MemoryError("Explicit KV allocation exceeds current free device memory")
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, num_blocks, self.block_size, num_kv_heads, head_dim)
        if self.has_pad_block:
            self.pad_block_id = config.num_kvcache_blocks
            self.kv_cache[:, :, self.pad_block_id].zero_()
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def _device_tensor(self, values: list, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, pin_memory=True).cuda(non_blocking=True)

    def prepare_block_tables(self, seqs: list[Sequence]):
        return self._device_tensor(padded_block_tables(seqs, self.kv_role), torch.int32)

    def prepare_prefill(self, seqs: list[Sequence]):
        metadata = prefill_metadata(seqs, self.kv_role, self.block_size)
        block_tables = self.prepare_block_tables(seqs) if metadata.has_cached_prefix else None
        input_ids = self._device_tensor(metadata.input_ids, torch.int64)
        positions = self._device_tensor(metadata.positions, torch.int64)
        cu_seqlens_q = self._device_tensor(metadata.cu_seqlens_q, torch.int32)
        cu_seqlens_k = self._device_tensor(metadata.cu_seqlens_k, torch.int32)
        slot_mapping = self._device_tensor(metadata.slot_mapping, torch.int32)
        set_context(True, cu_seqlens_q, cu_seqlens_k, metadata.max_seqlen_q, metadata.max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        metadata = decode_metadata(seqs, self.kv_role, self.block_size, num_steps=1)
        input_ids = self._device_tensor(metadata.input_ids, torch.int64)
        positions = self._device_tensor(metadata.positions, torch.int64)
        slot_mapping = self._device_tensor(metadata.slot_mapping[0], torch.int32)
        context_lens = self._device_tensor(metadata.context_lens, torch.int32)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool,
                  *, need_logits: bool = True, aux_out: torch.Tensor | None = None):
        """aux_out: the buffer a feature-based draft's aux features go into (Qwen3Model.forward); eager paths only."""
        if is_prefill or self.enforce_eager or input_ids.size(0) > EAGER_DECODE_BATCH_THRESHOLD:
            hidden = self.model(input_ids, positions, aux_out)
            return self.model.compute_logits(hidden) if need_logits else hidden
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
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            hidden = graph_vars["outputs"][:bs]
            return self.model.compute_logits(hidden) if need_logits else hidden

    def run(self, seqs: list[Sequence], is_prefill: bool, aux_out: torch.Tensor | None = None) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        sampling = sampling_tensors(seqs, self.device) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill, aux_out=aux_out)
        token_ids = self.sampler(logits, sampling, self.generator).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, MAX_GRAPH_BATCH_SIZE)
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
