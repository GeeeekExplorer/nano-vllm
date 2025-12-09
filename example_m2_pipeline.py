#!/usr/bin/env python3
"""
Example demonstrating M2: GPU1 Attention/FFN Pipeline.

This builds on M1 (two-GPU PD separation) and adds:
- Attention and FFN stages run on separate SM partitions on GPU1
- Green Context for SM-level resource allocation
- Pipeline execution with fixed SM ratio (16:16)
"""
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    print("=" * 80)
    print("M2: GPU1 Attention/FFN Pipeline Example")
    print("=" * 80)

    # Model path
    model_path = os.path.expanduser("./Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create LLM with M1 + M2 enabled
    print("\n[1/3] Initializing LLM with M1 + M2...")
    print("  M1: Two-GPU PD separation (GPU0=Prefill, GPU1=Decode)")
    print("  M2: Attention/FFN pipeline on GPU1")
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        # M1: Two-GPU PD separation
        enable_two_gpu_pd=True,
        prefill_device_id=4,            # GPU4 for prefill (use free GPU)
        decode_device_id=5,             # GPU5 for decode (use free GPU)
        # M2: Decode pipeline on GPU5
        enable_decode_pipeline=True,    # Enable Attention/FFN pipeline
        decode_attention_sm=16,         # 16 SMs for Attention
        decode_ffn_sm=16,               # 16 SMs for FFN
        decode_pipeline_profiling=True, # Enable performance profiling
    )

    # Prepare prompts
    prompts = [
        "Explain the concept of neural networks",
        "What are the main benefits of renewable energy?",
        "Describe the process of photosynthesis",
        "How does machine learning differ from traditional programming?",
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

    sampling_params = SamplingParams(temperature=0.7, max_tokens=96)

    print("\n[2/3] Running generation with M2 pipeline...")
    print("  - GPU0: Prefill (prompt encoding)")
    print("  - GPU1: Decode with Attention/FFN pipeline")
    print("    * Attention stage: 16 SMs")
    print("    * FFN stage: 16 SMs")
    print("    * Stages execute concurrently on different tokens")

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Display results
    print("\n[3/3] Generation completed!")
    print("=" * 80)
    print("Generated Outputs")
    print("=" * 80)
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\n[{i+1}] Prompt: {prompt[:50]}...")
        print(f"    Output: {output['text'][:120]}...")

    # Show pipeline statistics
    print("\n" + "=" * 80)
    print("M2 Pipeline Performance")
    print("=" * 80)
    if hasattr(llm, 'decode_runner') and llm.decode_runner:
        llm.decode_runner.print_pipeline_statistics()

    print("\n" + "=" * 80)
    print("M2 Pipeline Architecture")
    print("=" * 80)
    print("GPU0 (Prefill):")
    print("  └─ Processes prompts sequentially")
    print("  └─ Generates KV cache")
    print("  └─ Syncs KV to GPU1")
    print("\nGPU1 (Decode with Pipeline):")
    print("  ├─ Attention Stage (16 SMs)")
    print("  │  └─ KV lookup + Attention computation")
    print("  └─ FFN Stage (16 SMs)")
    print("     └─ Feed-forward + Normalization + Logits")
    print("\nPipeline Benefits:")
    print("  • Attention and FFN execute concurrently")
    print("  • Better GPU utilization on decode card")
    print("  • Reduced decode latency per token")

    print("\n" + "=" * 80)
    print("Next: M3 - Dynamic scheduling and rebalancing")
    print("=" * 80)


if __name__ == "__main__":
    main()
