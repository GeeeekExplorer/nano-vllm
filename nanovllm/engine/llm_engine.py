import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.ps = []
        self.events = []
        self.prefill_runner = None
        self.decode_runner = None
        self.model_runner = None

        # Two-GPU PD separation mode
        if config.enable_two_gpu_pd:
            assert config.tensor_parallel_size == 1, "Two-GPU PD mode requires tensor_parallel_size=1"
            print(f"[LLMEngine] Initializing Two-GPU PD separation mode")
            print(f"  - Prefill GPU: {config.prefill_device_id}")
            print(f"  - Decode GPU: {config.decode_device_id}")

            # Create two ModelRunner instances
            self.prefill_runner = ModelRunner(config, rank=0, event=None, device_id=config.prefill_device_id, is_decode_runner=False)
            self.decode_runner = ModelRunner(config, rank=0, event=None, device_id=config.decode_device_id, is_decode_runner=True)

        # Standard tensor parallel mode
        else:
            ctx = mp.get_context("spawn") # 建立多进程上下文 spawn 摸索确保每个进程有独立的环境
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
            self.model_runner = ModelRunner(config, 0, self.events)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if self.config.enable_two_gpu_pd:
            if self.prefill_runner is not None:
                self.prefill_runner.exit()
            if self.decode_runner is not None:
                self.decode_runner.exit()
            self.prefill_runner = None
            self.decode_runner = None
            # Clean up distributed process group in two-GPU mode
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        else:
            if self.model_runner is not None:
                self.model_runner.call("exit")
                self.model_runner = None
            for p in self.ps:
                p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt) # 文本转token ID
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        if self.config.enable_two_gpu_pd:
            return self._step_two_gpu()
        else:
            return self._step_single_gpu()

    def _step_single_gpu(self):
        """Original single-GPU or tensor-parallel step logic"""
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill) #执行推理
        self.scheduler.postprocess(seqs, token_ids) #更新序列状态
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def _step_two_gpu(self):
        """Two-GPU PD separation step logic"""
        outputs = []
        num_tokens = 0

        # Step 1: Try to run prefill on GPU0
        if self.scheduler.ready_for_prefill():
            prefill_seqs = self.scheduler.schedule_prefill()
            if prefill_seqs:
                # Run prefill on GPU0
                token_ids = self.prefill_runner.run(prefill_seqs, is_prefill=True)

                # Get block indices that need to be synced to GPU1
                block_indices = []
                for seq in prefill_seqs:
                    block_indices.extend(seq.block_table)

                # Sync KV cache from GPU0 to GPU1
                if block_indices:
                    # Copy KV cache blocks to decode GPU
                    import torch
                    with torch.cuda.device(self.config.prefill_device_id):
                        for block_idx in set(block_indices):  # Use set to avoid duplicates
                            # Direct P2P copy
                            self.decode_runner.kv_cache[:, :, block_idx].copy_(
                                self.prefill_runner.kv_cache[:, :, block_idx]
                            )

                num_tokens += sum(len(seq) for seq in prefill_seqs)

        # Step 2: Run decode on GPU1
        if self.scheduler.ready_for_decode():
            decode_seqs = self.scheduler.schedule_decode()
            if decode_seqs:
                # Run decode on GPU1
                token_ids = self.decode_runner.run(decode_seqs, is_prefill=False)

                # Postprocess decode results
                self.scheduler.postprocess(decode_seqs, token_ids)
                outputs.extend([(seq.seq_id, seq.completion_token_ids) for seq in decode_seqs if seq.is_finished])
                num_tokens -= len(decode_seqs)  # Negative for decode tokens

        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()  #高精度计时
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
