import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

try:
    from nanovllm.models.qwen3_vl import load_qwen3_vl_model
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    print("[ModelRunner] multimodal module unavailable")

from nanovllm.models.qwen3 import Qwen3ForCausalLM


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group(
            "nccl",
            "tcp://localhost:2333",
            world_size=self.world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype is None and hasattr(hf_config, "text_config"):
            torch_dtype = getattr(hf_config.text_config, "torch_dtype", None)
        if isinstance(torch_dtype, str):
            resolved_dtype = getattr(torch, torch_dtype, None)
            if resolved_dtype is None:
                alias_map = {
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                    "float16": torch.float16,
                }
                resolved_dtype = alias_map.get(torch_dtype.lower())
            torch_dtype = resolved_dtype
        if torch_dtype is None:
            torch_dtype = torch.float16
        torch.set_default_dtype(torch_dtype)
        torch.set_default_device("cuda")

        # Multimodal support is optional; fall back to text-only runner when
        # the extended Qwen3-VL stack is not available.
        self.is_multimodal = (
            getattr(config, "is_multimodal", False) and MULTIMODAL_AVAILABLE
        )
        if self.is_multimodal:
            self.model = load_qwen3_vl_model(config.model, config)
        else:
            text_config = getattr(hf_config, "text_config", hf_config)
            self.model = Qwen3ForCausalLM(text_config)
            load_model(self.model, config.model)

        embed_module = getattr(self.model, "language_model", self.model)
        if hasattr(embed_module, "model"):
            embed_module = embed_module.model
        # Keep a reference dtype so that cached vision embeddings can be copied
        # back to the GPU without hitting dtype mismatches.
        self.model_dtype = embed_module.embed_tokens.weight.dtype
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(
                    name="nanovllm",
                    create=True,
                    size=2**20,
                )
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
        dist.destroy_process_group()

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
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(
            max_num_batched_tokens // max_model_len,
            self.config.max_num_seqs,
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = text_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        )
        dtype = getattr(
            text_config,
            "torch_dtype",
            getattr(hf_config, "torch_dtype", torch.float16),
        )
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.float16)
        block_bytes = (
            2
            * text_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * dtype.itemsize
        )
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            text_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        block_tables = (
            torch.tensor(block_tables, dtype=torch.int32, pin_memory=True)
            .cuda(non_blocking=True)
        )
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
            if not seq.block_table:    # warmup
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
        input_ids = (
            torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)
            .cuda(non_blocking=True)
        )
        positions = (
            torch.tensor(positions, dtype=torch.int64, pin_memory=True)
            .cuda(non_blocking=True)
        )
        cu_seqlens_q = (
            torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True)
            .cuda(non_blocking=True)
        )
        cu_seqlens_k = (
            torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True)
            .cuda(non_blocking=True)
        )
        slot_mapping = (
            torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True)
            .cuda(non_blocking=True)
        )
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

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size
                + seq.last_block_num_tokens
                - 1
            )
        input_ids = (
            torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)
            .cuda(non_blocking=True)
        )
        positions = (
            torch.tensor(positions, dtype=torch.int64, pin_memory=True)
            .cuda(non_blocking=True)
        )
        slot_mapping = (
            torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True)
            .cuda(non_blocking=True)
        )
        context_lens = (
            torch.tensor(context_lens, dtype=torch.int32, pin_memory=True)
            .cuda(non_blocking=True)
        )
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = (
            torch.tensor(temperatures, dtype=torch.float32, pin_memory=True)
            .cuda(non_blocking=True)
        )
        return temperatures

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        sequence_lengths: list[int] | None = None,
        vision_slices_per_seq: list[list[dict]] | None = None,
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            model_kwargs = {}
            if self.is_multimodal:
                # Prefill can stream only part of the visual tokens. Pass
                # slice metadata so the forward pass knows which cached chunks
                # to use.
                model_kwargs["sequence_lengths"] = sequence_lengths
                model_kwargs["vision_slices_per_seq"] = vision_slices_per_seq
            outputs = self.model(input_ids, positions, **model_kwargs)
            return self.model.compute_logits(outputs)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph_idx = next(x for x in self.graph_bs if x >= bs)
            graph = self.graphs[graph_idx]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = (
                context.block_tables
            )
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)

        # Track how many freshly decoded tokens each sequence contributes; the
        # model uses these lengths to align partial vision slices with text.
        sequence_lengths = (
            [len(seq) - seq.num_cached_tokens for seq in seqs]
            if is_prefill
            else None
        )
        vision_slices_per_seq = None

        if is_prefill and self.is_multimodal:
            vision_slices_per_seq = []
            has_slices = False

            for seq in seqs:
                # Cache the full vision tower output once; subsequent prefill
                # steps only read the portions still needed for this sequence.
                self._ensure_vision_cache(seq)
                slices_for_seq: list[dict] = []
                window_start = seq.num_cached_tokens
                window_end = len(seq)

                for placeholder_idx, (offset, length) in enumerate(
                    seq.vision_placeholders
                ):
                    if placeholder_idx >= len(seq.vision_counts):
                        continue
                    consumed = seq.vision_consumed[placeholder_idx]
                    total_len = length
                    if consumed >= total_len:
                        continue

                    range_start = offset
                    range_end = offset + total_len

                    overlap_start = max(range_start, window_start)
                    overlap_end = min(range_end, window_end)
                    if overlap_end <= overlap_start:
                        continue

                    slice_offset = max(consumed, overlap_start - range_start)
                    remaining = total_len - slice_offset
                    overlap_available = overlap_end - overlap_start
                    take = min(remaining, overlap_available)
                    if take <= 0:
                        continue

                    target_offset = overlap_start - window_start

                    chunk_tokens = seq.cached_vision_tokens[placeholder_idx]
                    token_slice = (
                        chunk_tokens[slice_offset:slice_offset + take]
                        .to(
                            device="cuda",
                            dtype=self.model_dtype,
                            non_blocking=True,
                        )
                        .contiguous()
                    )

                    deepstack_slice: list[torch.Tensor] | None = None
                    if seq.cached_deepstack_tokens:
                        deepstack_slice = []
                        for layer_tokens in seq.cached_deepstack_tokens:
                            if placeholder_idx >= len(layer_tokens):
                                deepstack_slice.append(None)
                                continue
                            layer_slice = (
                                layer_tokens[placeholder_idx][
                                    slice_offset:slice_offset + take
                                ]
                                .to(
                                    device="cuda",
                                    dtype=self.model_dtype,
                                    non_blocking=True,
                                )
                                .contiguous()
                            )
                            deepstack_slice.append(layer_slice)

                    slices_for_seq.append(
                        {
                            "tokens": token_slice,
                            "deepstack": deepstack_slice,
                            "length": take,
                            "target_offset": target_offset,
                            "placeholder_idx": placeholder_idx,
                        }
                    )
                    has_slices = True

                vision_slices_per_seq.append(slices_for_seq)

            if not has_slices:
                vision_slices_per_seq = None

        def _advance_vision_offsets():
            if not is_prefill or not self.is_multimodal:
                return
            if vision_slices_per_seq is None:
                return
            for seq, slices in zip(seqs, vision_slices_per_seq):
                for slice_info in slices:
                    length = slice_info["length"]
                    placeholder_idx = slice_info["placeholder_idx"]
                    if placeholder_idx < len(seq.vision_consumed):
                        span = seq.vision_placeholders[placeholder_idx][1]
                        seq.vision_consumed[placeholder_idx] += length
                        seq.vision_consumed[placeholder_idx] = min(
                            seq.vision_consumed[placeholder_idx],
                            span,
                        )
                if seq.vision_placeholders:
                    # Once every placeholder has been consumed we can drop the
                    # cached tensors to release CPU memory.
                    all_consumed = all(
                        seq.vision_consumed[idx] >= span
                        for idx, (_, span) in enumerate(
                            seq.vision_placeholders
                        )
                    )
                else:
                    all_consumed = True
                if all_consumed:
                    seq.cached_vision_tokens = None
                    seq.cached_deepstack_tokens = None

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(
            input_ids,
            positions,
            is_prefill,
            sequence_lengths=sequence_lengths,
            vision_slices_per_seq=vision_slices_per_seq,
        )
        _advance_vision_offsets()
        if self.rank == 0:
            token_ids = self.sampler(logits, temperatures).tolist()
        else:
            token_ids = None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (
            config.max_model_len + self.block_size - 1
        ) // self.block_size
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
            warmup_out = self.model(input_ids[:bs], positions[:bs])
            outputs[:bs] = warmup_out  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                capture_out = self.model(input_ids[:bs], positions[:bs])
                outputs[:bs] = capture_out  # capture
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

    def _ensure_vision_cache(self, seq: Sequence):
        if seq.cached_vision_tokens is not None:
            return
        if seq.pixel_values is None or seq.image_grid_thw is None:
            seq.cached_vision_tokens = []
            seq.cached_deepstack_tokens = []
            return

        # Run the vision encoder once on the GPU and stash the outputs on CPU.
        # Later prefill iterations reuse these tensors without recomputing the
        # expensive 3D convolutions.
        pixel = seq.pixel_values.to(
            device="cuda",
            dtype=self.model_dtype,
            non_blocking=True,
        ).contiguous()
        grid = seq.image_grid_thw.to(
            device="cuda",
            dtype=torch.int32,
            non_blocking=True,
        ).contiguous()

        image_embeds, deepstack_features = self.model.visual(pixel, grid)
        seq.cached_vision_tokens = [emb.detach().cpu() for emb in image_embeds]
        if deepstack_features:
            cached_deepstack = []
            for layer_tokens in deepstack_features:
                cached_layer = [feat.detach().cpu() for feat in layer_tokens]
                cached_deepstack.append(cached_layer)
            seq.cached_deepstack_tokens = cached_deepstack
        else:
            seq.cached_deepstack_tokens = []

        seq.pixel_values = None
        seq.image_grid_thw = None
