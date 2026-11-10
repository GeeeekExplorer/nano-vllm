import argparse
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(Path(__file__).resolve().parent / "Qwen3-0.6B"))
    parser.add_argument("--device", choices=("auto", "cuda", "mps"), default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float)
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    memory_utilization = args.gpu_memory_utilization
    if memory_utilization is None:
        memory_utilization = 0.64 if device == "mps" else 0.9
    llm = LLM(args.model, device=device, enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=memory_utilization)

    try:
        sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
        prompts = [
            "introduce yourself",
            "list all prime numbers within 100",
        ]
        prompts = [
            llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
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
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
