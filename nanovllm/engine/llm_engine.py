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
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
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
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
        self.ps.clear()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def step(self, return_metadata: bool = False):
        batch = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", batch)
        self.scheduler.postprocess(batch, token_ids)
        outputs = [(item.seq.seq_id, item.seq.completion_token_ids) for item in batch.items if item.seq.is_finished]
        num_prefill_tokens = sum(item.num_query_tokens for item in batch.items if not item.is_decode)
        num_decode_tokens = sum(item.num_query_tokens for item in batch.items if item.is_decode)
        num_total_tokens = num_prefill_tokens + num_decode_tokens
        prefix_cache_hit_tokens = sum(item.prefix_cache_hit_tokens for item in batch.items)
        recomputed_prefill_tokens = sum(item.recomputed_prefill_tokens for item in batch.items)
        if self.scheduler.enable_continuous_batching:
            num_tokens = num_total_tokens
        else:
            # Keep the legacy signed convention in non-CB mode.
            num_tokens = num_prefill_tokens if num_decode_tokens == 0 else -num_decode_tokens
        if return_metadata:
            has_prefill = any(not item.is_decode for item in batch.items)
            has_decode = any(item.is_decode for item in batch.items)
            legacy_is_prefill = has_prefill and not has_decode
            metadata = {
                "seq_ids": [item.seq.seq_id for item in batch.items],
                "scheduled_query_tokens": [item.num_query_tokens for item in batch.items],
                "should_sample": [item.should_sample for item in batch.items],
                "is_decode": [item.is_decode for item in batch.items],
                "num_prefill_tokens": num_prefill_tokens,
                "num_decode_tokens": num_decode_tokens,
                "num_total_tokens": num_total_tokens,
                "prefix_cache_hit_tokens": prefix_cache_hit_tokens,
                "recomputed_prefill_tokens": recomputed_prefill_tokens,
                "is_prefill": legacy_is_prefill if not self.scheduler.enable_continuous_batching else None,
                "scheduled_prefill_tokens": [item.num_query_tokens if not item.is_decode else 0 for item in batch.items],
                "scheduled_prefix_cache_hit_tokens": [item.prefix_cache_hit_tokens for item in batch.items],
                "scheduled_recomputed_prefill_tokens": [item.recomputed_prefill_tokens for item in batch.items],
            }
            return outputs, num_tokens, metadata
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
            t = perf_counter()
            output, _, metadata = self.step(return_metadata=True)
            if use_tqdm:
                dt = perf_counter() - t
                if metadata["num_prefill_tokens"] > 0:
                    prefill_throughput = metadata["num_prefill_tokens"] / dt
                if metadata["num_decode_tokens"] > 0:
                    decode_throughput = metadata["num_decode_tokens"] / dt
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
