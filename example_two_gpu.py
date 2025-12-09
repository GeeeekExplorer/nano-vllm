#!/usr/bin/env python3
"""
Example demonstrating Two-GPU PD (Prefill/Decode) separation mode.

This implements M1 milestone from DECODE_PIPELINE_PLAN.md:
- GPU0: Dedicated to Prefill (large batch encoding)
- GPU1: Dedicated to Decode (incremental generation)
- KV cache synchronization between GPUs
"""
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    print("=" * 80)
    print("Two-GPU PD Separation Example (M1)")
    print("=" * 80)

    # Model path
    model_path = os.path.expanduser("./Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create LLM with two-GPU PD separation enabled
    print("\n[1/3] Initializing LLM with two-GPU PD separation...")
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        # Two-GPU configuration
        enable_two_gpu_pd=True,       # Enable two-GPU mode
        prefill_device_id=4,          # GPU4 for prefill (use free GPU)
        decode_device_id=5,           # GPU5 for decode (use free GPU)
    )

    # Prepare prompts
    prompts = [
        "What is the capital of France?",
        "Write a short poem about AI",
        "Explain quantum computing in simple terms",
        "What are the benefits of renewable energy?",
    ]

    # Apply chat template
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    sampling_params = SamplingParams(temperature=0.7, max_tokens=64)

    print("\n[2/3] Running generation with two-GPU PD separation...")
    print(f"  - Prefill will run on GPU:0")
    print(f"  - Decode will run on GPU:1")
    print(f"  - KV cache will be synced between GPUs")

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Display results
    print("\n[3/3] Generation completed!")
    print("=" * 80)
    print("Generated Outputs")
    print("=" * 80)
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\n[{i+1}] Prompt: {prompt[:60]}...")
        print(f"    Output: {output['text'][:150]}...")

    print("\n" + "=" * 80)
    print("Two-GPU PD separation demo completed successfully!")
    print("=" * 80)
    print("\nArchitecture:")
    print("  GPU0 (Prefill): Processes prompts, generates initial KV cache")
    print("  GPU1 (Decode):  Generates tokens incrementally using synced KV cache")
    print("\nNext steps (M2 & M3):")
    print("  - M2: Add Attention/FFN pipeline parallelism on GPU1")
    print("  - M3: Implement dynamic SM scheduling and back-pressure")


if __name__ == "__main__":
    main()
