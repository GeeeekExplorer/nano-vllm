import argparse
from statistics import median
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="local directory with Qwen3 config, tokenizer, and safetensors")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--input-step", type=int, default=0,
                        help="subtract this many prompt tokens from each successive request")
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if min(args.batch_size, args.input_len, args.output_len, args.repeat) < 1:
        raise ValueError("all lengths and batch size must be positive")
    input_lens = [args.input_len - i * args.input_step for i in range(args.batch_size)]
    if args.input_step < 0 or min(input_lens) < 1:
        raise ValueError("input-step must be nonnegative and every prompt must have at least one token")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be between 0 and 1")

    llm = LLM(
        args.model,
        device="mps",
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=min(16384, max(256, sum(input_lens))),
        max_model_len=args.max_model_len or args.input_len + args.output_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    try:
        vocab = llm.model_runner.config.hf_config.vocab_size
        if args.input_len + args.output_len > llm.model_runner.config.max_model_len:
            raise ValueError("input and output lengths exceed the model's maximum position")
        if vocab <= 5:
            raise ValueError("model vocabulary is too small for benchmark prompts")
        def make_prompts(offset):
            return [
                [4 + ((i * args.input_len + j + offset) % (vocab - 4)) for j in range(length)]
                for i, length in enumerate(input_lens)
            ]

        warmup_prompts = make_prompts(0)
        llm.generate(warmup_prompts, SamplingParams(max_tokens=args.output_len, ignore_eos=True), use_tqdm=False)

        params = SamplingParams(max_tokens=args.output_len, ignore_eos=True)
        timings = []
        for repetition in range(args.repeat):
            prompts = make_prompts(7 + repetition * args.batch_size * args.input_len)
            for prompt in prompts:
                llm.add_request(prompt, params)
            torch.mps.synchronize()
            started = perf_counter()
            prefill_time = decode_time = 0.0
            prefill_tokens = decode_tokens = 0
            while not llm.is_finished():
                step_start = perf_counter()
                _, count = llm.step()
                elapsed = perf_counter() - step_start
                if count > 0:
                    prefill_time += elapsed
                    prefill_tokens += count
                else:
                    decode_time += elapsed
                    decode_tokens -= count
            timings.append((prefill_time, decode_time, perf_counter() - started))
        prefill_time, decode_time, total_time = (median(values) for values in zip(*timings))
        total_output = args.batch_size * args.output_len
        print(f"model: {args.model}")
        print(f"device: MPS, dtype: {llm.model_runner.config.hf_config.dtype}")
        print(f"batch: {args.batch_size}, input: {input_lens if args.input_step else args.input_len}, "
              f"output: {args.output_len}, repeats: {args.repeat} (median)")
        print(f"prefill: {prefill_tokens} tokens / {prefill_time:.3f} s = {prefill_tokens / prefill_time:.1f} tok/s")
        if decode_time:
            print(f"decode: {decode_tokens} tokens / {decode_time:.3f} s = {decode_tokens / decode_time:.1f} tok/s")
        print(f"end-to-end: {total_output} output tokens / {total_time:.3f} s = {total_output / total_time:.1f} tok/s")
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
