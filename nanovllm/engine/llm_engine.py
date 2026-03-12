import atexit
from dataclasses import fields
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 1. Build Config from kwargs. Use fields to return metadata about each
        # dataclass field. Can be not only the field name, e.g. each element in
        # that list contains infor like field name, type, default value, etc.
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # Init worker processes (do the work) and their sync events (coordinate the work).
        self.ps = []
        # ctx.Event() provides a shared SIGNAL only that process can choose to
        # wait for, check, or react to.
        self.events = []
        # spawn starts a fresh Python interpreter and imports everything from
        # scratch, which is required for CUDA. fork and forkserver won't work
        # because they inherit CUDA context from the parent process, which
        # causes errors.
        ctx = mp.get_context("spawn")
        # 2. Spawn worker processes for tensor parallelism.
        # FYI, tensor parallelism is split the computation of a single neural
        # network layer across multiple GPUs, so they work together on the same
        # model at the same time.
        # FYI, sum partial outputs -> all-reduce
        #      concatenate partial outputs -> all-gather
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            # Start a separate worker process, and in that process, call this
            # callable (ModelRunner) with these arguments (config, i, event)
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            # worker process has to start first, due to under the
            # ModelRunner.__init__(), the dist.init_process_group() is a
            # collective operation, it blocks until all ranks have called it. So
            # the flow is:
            # (1) Worker processes (rank 1, 2, ...) are spawned and
            # start executing ModelRunner.__init__(). They each call
            # dist.init_process_group() and block, waiting for all other ranks
            # to join.
            # (2) Rank 0 calls ModelRunner.init(), now all ranks have
            # joined, so the barrier is released and all ranks processed
            # together.
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 3. Create rank-0 ModelRunner (loads model, warmup forward pass,
        #    allocates KV cache, captures CUDA graphs)
        self.model_runner = ModelRunner(config, 0, self.events)
        # 4. Load tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, 
                                                       use_fast=True # Use fast tokenizer for better performance.
        )
        config.eos = self.tokenizer.eos_token_id

        # 5. Create Scheduler (needs to know num_kvcache_blocks, which was computed by ModelRunner)
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            # .join() blocks the caller until the target process exits.
            # It actually only waits until child processes finish, not terminate them immediately.
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt) # e.g. "Hello, world" --> [15496, 11, 703]
        seq = Sequence(prompt, sampling_params)    # Create a Sequence object.
        self.scheduler.add(seq)                    # Push to the waiting queue in the Scheduler.

    def step(self):
        seqs, is_prefill = self.scheduler.schedule() # (a) What to run?
        # prefill: process all tokens at once, produces logits for the last position, samples 1 token.
        # subsequent decodes: process only 1 token at a time (the last generated token)
        token_ids = self.model_runner.call("run", seqs, is_prefill) # (b) Run the model.
        # If the token is NOT EOS, and we have not yet hit max_tokens, the
        # sequence stays running and will be scheduled for decode on the next step.
        self.scheduler.postprocess(seqs, token_ids) # (c) Update sequence state.
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished] # Collect done finished.
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]], # prompts can be raw text or already-tokenized list of token ids.
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[str]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 1. ENqueue all requests.
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        # 2. Run the engine loop until all requests finish.
        outputs = {}
        while not self.is_finished():
            output, num_tokens = self.step() # <-- this is the core loop.
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 3. Decode token IDs back to text.
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
