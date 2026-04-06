import os
from pathlib import Path
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def get_model_path():
    """Auto-detect model path based on environment."""
    # Check environment variable first
    if "MODEL_DIR" in os.environ:
        model_dir = Path(os.environ["MODEL_DIR"]) / "Qwen3-0.6B"
        if model_dir.exists():
            return str(model_dir)
    
    # Check common locations
    common_paths = [
        Path.home() / "huggingface" / "Qwen3-0.6B",
        Path("/content/models/Qwen3-0.6B"),  # Colab
        Path("./models/Qwen3-0.6B"),  # Local
    ]
    
    for path in common_paths:
        if path.exists():
            return str(path)
    
    # Default fallback
    return str(Path.home() / "huggingface" / "Qwen3-0.6B")


def main():
    path = get_model_path()
    print(f"Using model from: {path}")
    print("")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)

    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        # kv_cache_dtype="default",  # Explicitly use default (optional, this is the default)
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
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
    
    # Generate outputs
    print("Generating responses...")
    print("=" * 60)
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
        print(f"\nQuery {i}:")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
        print("=" * 60)


if __name__ == "__main__":
    main()
