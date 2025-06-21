import os
from pathlib import Path

from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def ensure_weights(repo_id: str, local_dir: Path):
    """
    Download the repo from Hugging Face into local_dir if no .bin weights are present.
    Uses resume_download=True so it only grabs missing files.
    """
    # Check for any .bin files in the target folder
    if not local_dir.exists() or not any(local_dir.glob("*.bin")):
        print(f"⏬ Downloading weights for {repo_id} into {local_dir} …")
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            cache_dir=str(local_dir),
            resume_download=True,
        )
    else:
        print(f"✅ Weights already present in {local_dir}, skipping download.")


def main():
    repo_id = "Qwen/Qwen3-0.6B"
    local_path = Path.home() / "huggingface" / "Qwen3-0.6B"

    # 1) download if needed
    ensure_weights(repo_id, local_path)

    # 2) load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    llm = LLM(str(local_path), enforce_eager=True, tensor_parallel_size=1)

    # 3) prepare & generate
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    raw_prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for p in raw_prompts
    ]
    outputs = llm.generate(chat_prompts, sampling_params)

    # 4) print results
    for prompt, output in zip(chat_prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
