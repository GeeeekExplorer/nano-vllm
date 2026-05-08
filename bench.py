import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.config import Config

# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024
    enforce_eager = False
    print(f"use cuda graph:{not enforce_eager}")
    path = os.path.expanduser("/YOUR/MODEL/PATH")
    llm = LLM(path, enforce_eager=enforce_eager, max_model_len=max(4096, max_ouput_len+max_input_len))

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # prompt_token_ids = [[randint(0, 10000) for _ in range(max_input_len)] for _ in range(num_seqs)]
    # sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_ouput_len) for _ in range(num_seqs)]

    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
