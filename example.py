import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    


    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "write a hello world progrm in python",
        "list all prime numbers within 100",
        "What is the capital of India",
        "write a program in python to add 2 numbers",
        
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + " /no_think"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
