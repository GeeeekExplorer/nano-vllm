import os
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

try:
    from nanovllm.models.qwen3_vl import load_qwen3_vl_model
    QWEN3_VL_AVAILABLE = True
except ImportError:
    QWEN3_VL_AVAILABLE = False
    load_qwen3_vl_model = None
try:
    from nanovllm.models.qwen3_5 import load_qwen3_5_model
    QWEN3_5_AVAILABLE = True
except ImportError:
    QWEN3_5_AVAILABLE = False
    load_qwen3_5_model = None
MULTIMODAL_AVAILABLE = QWEN3_VL_AVAILABLE or QWEN3_5_AVAILABLE

try:
    from nanovllm.models.qwen3_next import Qwen3NextForCausalLM
    QWEN3_NEXT_AVAILABLE = True
except ImportError:
    QWEN3_NEXT_AVAILABLE = False
    Qwen3NextForCausalLM = None

if not MULTIMODAL_AVAILABLE:
    print("[ModelRunner] 多模态模块不可用")


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda", rank) if self.use_cuda else torch.device("cpu")

        # Even with world_size=1, layers call dist.get_world_size()/get_rank().
        # Initialize a process group with a CPU-capable backend when CUDA isn't available.
        backend = "nccl" if (self.use_cuda and dist.is_nccl_available()) else "gloo"
        dist.init_process_group(
            backend,
            "tcp://localhost:2333",
            world_size=self.world_size,
            rank=rank,
        )

        if self.use_cuda:
            torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype is None and hasattr(hf_config, "text_config"):
            torch_dtype = getattr(hf_config.text_config, "torch_dtype", None)
        if isinstance(torch_dtype, str):
            resolved = getattr(torch, torch_dtype, None)
            if resolved is None:
                alias_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
                resolved = alias_map.get(torch_dtype.lower())
            torch_dtype = resolved
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        torch.set_default_dtype(torch_dtype)
        torch.set_default_device(self.device.type)

        # 根据 model_type 区分多模态 vs 纯文本，以及 qwen3_vl vs qwen3_5 / qwen3_next vs qwen3
        model_type = getattr(hf_config, "model_type", None) or ""
        self.is_multimodal = (
            getattr(config, "is_multimodal", False) and MULTIMODAL_AVAILABLE
        )
        if self.is_multimodal:
            # 多模态：按 model_type 选择 qwen3_5 或 qwen3_vl
            if model_type in ("qwen3_5", "qwen3_5_moe") and QWEN3_5_AVAILABLE and load_qwen3_5_model:
                print("[ModelRunner] 加载 Qwen3.5 多模态模型 (model_type=%s)" % model_type)
                self.model = load_qwen3_5_model(config.model, config)
            elif QWEN3_VL_AVAILABLE and load_qwen3_vl_model:
                print("[ModelRunner] 加载 Qwen3-VL 多模态模型 (model_type=%s)" % model_type)
                self.model = load_qwen3_vl_model(config.model, config)
            else:
                raise RuntimeError(
                    "多模态已开启但无法加载: model_type=%s, QWEN3_5_AVAILABLE=%s, QWEN3_VL_AVAILABLE=%s"
                    % (model_type, QWEN3_5_AVAILABLE, QWEN3_VL_AVAILABLE)
                )
        else:
            # 纯文本：按 text_config.model_type 选择 Qwen3Next 或 Qwen3
            text_config = getattr(hf_config, "text_config", hf_config)
            text_model_type = getattr(text_config, "model_type", None) or model_type
            if text_model_type == "qwen3_next" and QWEN3_NEXT_AVAILABLE and Qwen3NextForCausalLM is not None:
                print("[ModelRunner] 加载纯文本模型 Qwen3Next (model_type={})".format(text_model_type))
                print("[ModelRunner] 正在初始化模型结构...")
                self.model = Qwen3NextForCausalLM(text_config)
                print("[ModelRunner] 模型结构初始化完成，正在加载权重...")
                load_model(self.model, config.model)
                print("[ModelRunner] 权重加载完成")
            else:
                print("[ModelRunner] 加载纯文本模型 Qwen3 (model_type={})".format(text_model_type or "qwen3"))
                print("[ModelRunner] 正在初始化模型结构...")
                self.model = Qwen3ForCausalLM(text_config)
                print("[ModelRunner] 模型结构初始化完成，正在加载权重...")
                load_model(self.model, config.model)
                print("[ModelRunner] 权重加载完成")

        embed_module = getattr(self.model, "language_model", self.model)
        if hasattr(embed_module, "model"):
            embed_module = embed_module.model
        self.model_dtype = embed_module.embed_tokens.weight.dtype
        # GatedDeltaNet now supports CUDA graph decode via persistent state buffers
        # (_graph_conv_state / _graph_recurrent_state) with in-place updates that
        # are correctly captured and replayed by the CUDA graph.  No need to force
        # eager decode for Qwen3.5 anymore.

        self.sampler = Sampler()
        self.warmup_model()
        # Reset GatedDeltaNet states after warmup to avoid polluting real sequences
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
            for layer in self.model.language_model.model.layers:
                if hasattr(layer, 'linear_attn') and layer.linear_attn is not None:
                    layer.linear_attn.reset_state()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                try:
                    try:
                        self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                    except FileExistsError:
                        # 共享内存已存在，先尝试清理再创建
                        try:
                            existing_shm = SharedMemory(name="nanovllm")
                            existing_shm.unlink()
                            self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                        except Exception:
                            # 如果清理失败，直接打开现有的
                            self.shm = SharedMemory(name="nanovllm")
                    # 确保 barrier 前所有初始化都成功
                    if dist.is_initialized():
                        dist.barrier()
                except Exception as e:
                    print(f"[R{rank}] Failed to initialize shared memory: {e}")
                    import traceback
                    traceback.print_exc()
                    # 尝试清理并重新抛出异常
                    if dist.is_initialized():
                        try:
                            dist.destroy_process_group()
                        except Exception:
                            pass
                    raise
            else:
                try:
                    if dist.is_initialized():
                        dist.barrier()
                    self.shm = SharedMemory(name="nanovllm")
                    self.loop()
                except Exception as e:
                    print(f"[R{rank}] Failed to initialize worker: {e}")
                    import traceback
                    traceback.print_exc()
                    if dist.is_initialized():
                        try:
                            dist.destroy_process_group()
                        except Exception:
                            pass
                    raise

    def exit(self):
        if self.world_size > 1:
            if hasattr(self, "shm"):
                try:
                    self.shm.close()
                except Exception:
                    pass  # 忽略关闭时的异常
            if dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass  # 忽略 barrier 时的异常
            if self.rank == 0 and hasattr(self, "shm"):
                try:
                    self.shm.unlink()
                except (FileNotFoundError, Exception):
                    pass  # 已经被清理或不存在
        if not self.enforce_eager and hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        if self.use_cuda:
            torch.cuda.synchronize()
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass  # Process group 可能已经被销毁

    def loop(self):
        while True:
            try:
                method_name, args = self.read_shm()
                try:
                    self.call(method_name, *args)
                except Exception as e:
                    # 捕获执行时的异常，记录但不崩溃
                    print(f"[R{self.rank}] Error executing {method_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    # 继续循环，等待下一个指令
                if method_name == "exit":
                    break
            except Exception as e:
                # 捕获 read_shm 或其他顶层异常
                print(f"[R{self.rank}] Fatal error in loop: {e}")
                import traceback
                traceback.print_exc()
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        try:
            n = int.from_bytes(self.shm.buf[0:4], "little")
            # 验证数据长度合理性
            if n < 0 or n > len(self.shm.buf) - 4:
                raise ValueError(f"Invalid data length: {n}")
            method_name, *args = pickle.loads(self.shm.buf[4:n+4])
            self.event.clear()
            return method_name, args
        except Exception as e:
            self.event.clear()  # 确保清除 event，避免死锁
            print(f"[R{self.rank}] Error reading shared memory: {e}")
            raise

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
        try:
            method = getattr(self, method_name, None)
            if method is None:
                raise AttributeError(f"Method '{method_name}' not found")
            return method(*args)
        except Exception as e:
            # 记录异常但不让进程崩溃
            print(f"[R{self.rank}] Error in call({method_name}): {e}")
            import traceback
            traceback.print_exc()
            # 如果是 rank 0，需要通知 workers 退出，避免他们永远等待
            if self.world_size > 1 and self.rank == 0:
                try:
                    # 尝试发送 exit 信号给 workers
                    self.write_shm("exit")
                except Exception:
                    pass  # 如果连 exit 都发不出去，说明系统已经严重损坏
            raise  # 重新抛出异常，让上层处理

    def warmup_model(self):
        if self.use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        if self.use_cuda:
            torch.cuda.empty_cache()

    def _reset_gdn_states(self):
        """Reset Qwen3.5/Qwen3Next linear-attn recurrent states if present."""
        if hasattr(self.model, "language_model") and hasattr(
            self.model.language_model, "model"
        ):
            for layer in self.model.language_model.model.layers:
                if hasattr(layer, "linear_attn") and layer.linear_attn is not None:
                    if hasattr(layer.linear_attn, "reset_state"):
                        layer.linear_attn.reset_state()

    def cleanup_seq_states(self, seq_ids: list[int]):
        """Remove GDN state_cache entries for finished sequences."""
        if hasattr(self.model, "language_model") and hasattr(
            self.model.language_model, "model"
        ):
            for layer in self.model.language_model.model.layers:
                if hasattr(layer, "linear_attn") and layer.linear_attn is not None:
                    state_cache = getattr(layer.linear_attn, "state_cache", None)
                    if state_cache is not None:
                        for sid in seq_ids:
                            state_cache.pop(sid, None)
                    counter = getattr(layer.linear_attn, "_decode_step_counter", None)
                    if counter is not None:
                        for sid in seq_ids:
                            counter.pop(sid, None)

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        num_kv_heads = text_config.num_key_value_heads // self.world_size
        head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
        dtype = getattr(getattr(hf_config, "text_config", hf_config), "torch_dtype", getattr(hf_config, "torch_dtype", torch.bfloat16))
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.bfloat16)
        
        # 对于混合架构（如 Qwen3-Next），只计算需要 KV cache 的层
        # 只有 full_attention 层需要 KV cache，linear_attention (Mamba/GatedDeltaNet) 不需要
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is not None:
            num_attention_layers = sum(1 for lt in layer_types if lt == "full_attention")
        else:
            num_attention_layers = text_config.num_hidden_layers
        
        block_bytes = 2 * num_attention_layers * self.block_size * num_kv_heads * head_dim * dtype.itemsize
        if self.use_cuda:
            free, total = torch.cuda.mem_get_info()
            used = total - free
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            # Use simpler formula: only account for currently used memory
            available = int(total * config.gpu_memory_utilization - used)
            config.num_kvcache_blocks = available // block_bytes
        else:
            # CPU debug path: allocate enough blocks to cover max_model_len.
            if config.num_kvcache_blocks == -1:
                config.num_kvcache_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        assert config.num_kvcache_blocks > 0, (
            "Insufficient KV cache blocks "
            f"(num_kvcache_blocks={config.num_kvcache_blocks}, block_bytes={block_bytes}, "
            f"num_attention_layers={num_attention_layers}, total_layers={text_config.num_hidden_layers})"
        )

        self.kv_cache = torch.empty(
            2,
            num_attention_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=self.device,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                assert layer_id < num_attention_layers, (
                    f"More attention layers found ({layer_id + 1}) than expected ({num_attention_layers}). "
                    f"This may indicate a mismatch between layer_types configuration and actual model structure."
                )
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        
        # 验证分配的层数是否正确
        assert layer_id == num_attention_layers, (
            f"KV cache allocation mismatch: expected {num_attention_layers} attention layers, "
            f"but found {layer_id} layers with k_cache/v_cache attributes."
        )

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, device=self.device)
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
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        positions = torch.tensor(positions, dtype=torch.int64, device=self.device)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
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
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        positions = torch.tensor(positions, dtype=torch.int64, device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, device=self.device)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, device=self.device)
        return temperatures

    def _gather_gdn_states(self, sequence_ids: list[int], bs: int):
        """Copy per-sequence states from state_cache into graph buffers before replay."""
        if not hasattr(self.model, "language_model") or not hasattr(
            self.model.language_model, "model"
        ):
            return
        for layer in self.model.language_model.model.layers:
            if hasattr(layer, "linear_attn") and layer.linear_attn is not None:
                gdn = layer.linear_attn
                for i, seq_id in enumerate(sequence_ids):
                    conv_state, recurrent_state = gdn.state_cache.get(seq_id, (None, None))
                    if conv_state is not None:
                        # conv_state eager shape: (1, conv_dim, kernel_size-1)
                        gdn._graph_conv_state[i].copy_(conv_state.squeeze(0))
                    else:
                        gdn._graph_conv_state[i].zero_()
                    if recurrent_state is not None:
                        # recurrent_state eager shape: (1, H, V, K)
                        gdn._graph_recurrent_state[i].copy_(recurrent_state.squeeze(0))
                    else:
                        gdn._graph_recurrent_state[i].zero_()

    def _scatter_gdn_states(self, sequence_ids: list[int], bs: int):
        """Copy updated graph buffers back into state_cache after replay."""
        if not hasattr(self.model, "language_model") or not hasattr(
            self.model.language_model, "model"
        ):
            return
        for layer in self.model.language_model.model.layers:
            if hasattr(layer, "linear_attn") and layer.linear_attn is not None:
                gdn = layer.linear_attn
                for i, seq_id in enumerate(sequence_ids):
                    gdn.state_cache[seq_id] = (
                        gdn._graph_conv_state[i : i + 1].clone(),
                        gdn._graph_recurrent_state[i : i + 1].clone(),
                    )

    def _debug_prefill_logits(self, hidden_states: torch.Tensor, logits: torch.Tensor):
        """Print last-token hidden norm and logits top-5 when NANOVLLM_DEBUG_LOGITS=1."""
        if self.rank != 0 or hidden_states is None or logits is None:
            return
        ctx = get_context()
        if not getattr(ctx, "is_prefill", False) or not hasattr(ctx, "cu_seqlens_q"):
            return
        last_indices = ctx.cu_seqlens_q[1:] - 1
        last_h = hidden_states[last_indices]
        norm_val = last_h.norm().item()
        has_nan = torch.isnan(hidden_states).any().item() or torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        top5 = torch.topk(logits[0].float().cpu(), min(5, logits.shape[-1]))
        print("[NANOVLLM_DEBUG_LOGITS] last_token_hidden_norm=%.6f has_nan=%s has_inf=%s" % (norm_val, has_nan, has_inf))
        print("[NANOVLLM_DEBUG_LOGITS] argmax_id=%s top5_ids=%s top5_vals=%s" % (
            logits[0].argmax().item(),
            top5.indices.tolist(),
            [round(v, 4) for v in top5.values.tolist()],
        ))

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        sequence_lengths: list[int] | None = None,
        sequence_ids: list[int] | None = None,
        vision_slices_per_seq: list[list[dict]] | None = None,
        image_grid_thw_per_seq: list | None = None,
    ):
        if (
            is_prefill
            or self.enforce_eager
            or input_ids.size(0) > 512
        ):
            model_kwargs = {}
            # HF-style: pass sequence_lengths for vision alignment and GDN per-sequence isolation
            if self.is_multimodal:
                if sequence_lengths is not None:
                    model_kwargs["sequence_lengths"] = sequence_lengths
                if sequence_ids is not None:
                    model_kwargs["sequence_ids"] = sequence_ids
                if vision_slices_per_seq is not None:
                    model_kwargs["vision_slices_per_seq"] = vision_slices_per_seq
                if image_grid_thw_per_seq is not None:
                    model_kwargs["image_grid_thw_per_seq"] = image_grid_thw_per_seq
            elif (
                Qwen3NextForCausalLM is not None
                and isinstance(self.model, Qwen3NextForCausalLM)
                and sequence_lengths is not None
                and len(sequence_lengths) > 1
            ):
                # Pure text Qwen3.5/Qwen3-Next batch: GDN needs per-seq isolation
                model_kwargs["sequence_lengths"] = sequence_lengths
                if sequence_ids is not None:
                    model_kwargs["sequence_ids"] = sequence_ids
            outputs = self.model(input_ids, positions, **model_kwargs)
            # Prefill: do NOT index here. ParallelLMHead.forward() already does
            # x[context.cu_seqlens_q[1:]-1] to take the last token per sequence.
            # Indexing here would shrink outputs to (num_seqs, hidden), then
            # lm_head would index again with full-sequence indices -> OOB.
            logits = self.model.compute_logits(outputs)
            if os.environ.get("NANOVLLM_DEBUG_LOGITS"):
                self._debug_prefill_logits(outputs, logits)
            return logits

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
        # Gather per-sequence GDN states into graph buffers before replay.
        if sequence_ids is not None:
            self._gather_gdn_states(sequence_ids, bs)
        graph.replay()
        # Scatter updated GDN states back to per-sequence state_cache after replay.
        if sequence_ids is not None:
            self._scatter_gdn_states(sequence_ids, bs)
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)

        # HF-style: per-sequence GDN state isolation.
        # Use actual seq_id from each Sequence so GDN state_cache keys are
        # stable across batch composition changes (e.g. when a sequence finishes
        # and the remaining sequences are re-batched with different indices).
        sequence_ids = [seq.seq_id for seq in seqs]
        if is_prefill:
            sequence_lengths = [len(seq) - seq.num_cached_tokens for seq in seqs]
        else:
            sequence_lengths = [1] * len(seqs)  # decode: 1 token per sequence
        vision_slices_per_seq = None
        image_grid_thw_per_seq = None
        if is_prefill and self.is_multimodal:
            image_grid_thw_per_seq = [getattr(seq, "image_grid_thw", None) for seq in seqs]
            image_token_id = getattr(self.config.hf_config, "image_token_id", None)
            if image_token_id is not None:
                seq_stats = []
                for seq in seqs:
                    seq_stats.append(
                        {
                            "seq_id": seq.seq_id,
                            "len": len(seq),
                            "cached": seq.num_cached_tokens,
                            "placeholder_count": seq.token_ids.count(image_token_id),
                            "vision_counts": seq.vision_counts,
                            "vision_consumed": seq.vision_consumed,
                        }
                    )
                # print("[ModelRunner] prefill seq stats", seq_stats)

            vision_slices_per_seq = []
            has_slices = False
            for seq in seqs:
                self._ensure_vision_cache(seq)
                slices_for_seq: list[dict] = []
                window_start = seq.num_cached_tokens
                window_end = len(seq)

                for placeholder_idx, (offset, length) in enumerate(seq.vision_placeholders):
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
                    token_slice = chunk_tokens[
                        slice_offset : slice_offset + take
                    ].to(
                        device="cuda",
                        dtype=self.model_dtype,
                        non_blocking=True,
                    ).contiguous()

                    # debug_mean = float(token_slice.float().abs().mean().item())
                    # print(
                    #     "[ModelRunner] slice",
                    #     {
                    #         "seq_id": seq.seq_id,
                    #         "placeholder": placeholder_idx,
                    #         "slice_offset": slice_offset,
                    #         "length": take,
                    #         "target_offset": target_offset,
                    #         "mean_abs": round(debug_mean, 6),
                    #     },
                    # )

                    slices_for_seq.append(
                        {
                            "tokens": token_slice,
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
                            seq.vision_consumed[placeholder_idx], span
                        )
                if seq.vision_placeholders:
                    all_consumed = all(
                        seq.vision_consumed[idx] >= span
                        for idx, (_, span) in enumerate(seq.vision_placeholders)
                    )
                else:
                    all_consumed = True
                if all_consumed:
                    seq.cached_vision_tokens = None

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # vLLM-style: batch handling is done via cu_seqlens in global context
        logits = self.run_model(
            input_ids,
            positions,
            is_prefill,
            sequence_lengths=sequence_lengths,
            sequence_ids=sequence_ids,
            vision_slices_per_seq=vision_slices_per_seq,
            image_grid_thw_per_seq=image_grid_thw_per_seq,
        )
        _advance_vision_offsets()
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        if not self.use_cuda:
            # CPU mode: CUDA graph replay isn't applicable.
            return
        config = self.config
        hf_config = config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)

        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (
            config.max_model_len + self.block_size - 1
        ) // self.block_size

        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(
            max_bs, max_num_blocks, dtype=torch.int32
        )

        hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is None:
            # Fallback: infer from token embedding dim.
            embed_module = getattr(
                self.model, "language_model", self.model
            )
            if hasattr(embed_module, "model"):
                embed_module = embed_module.model
            hidden_size = int(
                embed_module.embed_tokens.weight.shape[-1]
            )

        outputs = torch.zeros(max_bs, hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(
            range(16, max_bs + 1, 16)
        )
        # Only capture graphs that fit the allocated dummy buffers.
        # When max_bs is small (e.g. max_num_seqs=1), we must not capture
        # larger bs values because input_ids[:bs] will still have length
        # max_bs, but sequence_lengths would have length bs, causing empty
        # sequence slices inside Qwen3.5 linear_attention.
        self.graph_bs = [x for x in self.graph_bs if x <= max_bs]
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            # Capture uses dummy tokens; ensure it doesn't leak recurrent states
            # into subsequent captures or real inference.
            self._reset_gdn_states()
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            # Decode-only: each "sequence" in the batch is exactly one token.
            # Qwen3.5/GDN uses `sequence_lengths/sequence_ids` to provide per-seq
            # recurrent state isolation; if we omit it here, CUDA graph replay
            # will use the wrong recurrence semantics and can produce repeated
            # tokens / garbled outputs.
            extra_kwargs = {}
            if Qwen3NextForCausalLM is not None and isinstance(self.model, Qwen3NextForCausalLM):
                extra_kwargs["sequence_lengths"] = [1] * bs
                extra_kwargs["use_graph"] = True
            outputs[:bs] = self.model(
                input_ids[:bs],
                positions[:bs],
                **extra_kwargs,
            )  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(
                    input_ids[:bs],
                    positions[:bs],
                    **extra_kwargs,
                )  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            if self.use_cuda:
                torch.cuda.synchronize()
            reset_context()

        # Final cleanup before serving real requests.
        self._reset_gdn_states()

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
            return
        pixel = seq.pixel_values.to(
            device=self.device,
            dtype=self.model_dtype,
            non_blocking=True,
        ).contiguous()
        grid = seq.image_grid_thw.to(
            device=self.device,
            dtype=torch.int32,
            non_blocking=True,
        ).contiguous()
        image_tokens = self.model.visual(pixel, grid)
        seq.cached_vision_tokens = [
            image_tokens[i].detach().cpu() for i in range(image_tokens.size(0))
        ]
        seq.pixel_values = None
        seq.image_grid_thw = None
