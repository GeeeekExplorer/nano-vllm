import atexit
from time import perf_counter
from typing import Any
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from tokenizers import Tokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    def __init__(self, config: Config) -> None:

        # Setup ModelRunner processes and inter-process communication for tensor parallelism.
        self.ps = []
        self.events = []
        # Set global start method to "spawn" for multiprocessing to avoid issues with CUDA in forked processes.
        ctx = mp.get_context("spawn")
        # The main process (rank=0) will also run a ModelRunner, so we only need to
        # spawn tensor_parallel_size - 1 processes and start with rank=1.
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)

        self.tokenizer: Tokenizer = AutoTokenizer.from_pretrained(
            config.model, use_fast=True
        )
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

        # Register the exit function to clean up the model runner processes when the program exits.
        atexit.register(self.exit)

    def exit(self):
        """Clean up the model runner processes."""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        Adds a generation request to the scheduler.
        Decodes the prompt to a list of token_ids if it's a string.
        """
        if isinstance(prompt, str):
            token_ids: list[int] = self.tokenizer.encode(prompt)
        else:
            token_ids = prompt
        seq = Sequence(token_ids, sampling_params)
        self.scheduler.add(seq)

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """
        Performs one step of scheduling and model inference, i.e. one step of batched prefill or one step of batched decode.

        Returns a list of (seq_id, completion_token_ids) for finished sequences in this step
        and the number of tokens processed in this step (positive for prefill, negative for decode).
        """
        # Schedule sequences for the next model inference step
        seqs, is_prefill = self.scheduler.schedule()

        # Perform one step of model inference with the scheduled sequences
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # Postprocess the generated token ids and update the status of each sequence.
        self.scheduler.postprocess(seqs, token_ids)

        # Collect the generated token ids for finished sequences in this step.
        finished_seq_ids_and_compl_tokens = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        # Number of tokens processed in this step. Positive for prefill, negative for decode.
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return finished_seq_ids_and_compl_tokens, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict[str, Any]]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            assert isinstance(prompt, (str, list))
            self.add_request(prompt, sp)

        # Store the finished requests' generated token ids here. Key is seq_id and value is the list of generated token ids (excluding prompt tokens).
        outputs: dict[int, list[int]] = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        output_seqs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        output_seqs_text = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in output_seqs
        ]
        if use_tqdm:
            pbar.close()
        return output_seqs_text
