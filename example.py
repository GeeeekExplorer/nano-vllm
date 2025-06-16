import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import argparse

def main(path, rank, smname):
    path = os.path.expanduser(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=rank, smname=smname)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
    
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Use 'rank' and 'path' parameters to control multi-rank inference and model path.")
    parser.add_argument('--rank', default=1, type=int)
    parser.add_argument('--path', default="/home/data/Qwen3-8B", type=str)
    # --smname: Shared memory name used for inter-process communication.
    # If the program crashes or is terminated unexpectedly, the shared memory segment
    # may remain and block future runs with the same name.
    # Use this option to specify a unique name per run or clean it up manually if needed.
    parser.add_argument('--smname', default="nanovllm", type=str)

    args = parser.parse_args()
    main(args.path, args.rank, args.smname)
