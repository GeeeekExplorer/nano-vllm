"""Run nano-vLLM as a one-shot Modal job.

Usage:
    pip install modal
    modal setup
    modal run modal_app.py --prompt "Explain KV caching in one sentence."
"""

from pathlib import Path

import modal


APP_NAME = "nano-vllm"
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_ROOT = Path("/models")
MODEL_VOLUME = modal.Volume.from_name("nano-vllm-models", create_if_missing=True)

# FlashAttention 2 requires Ampere-or-newer hardware. A10 is the low-cost
# compatible fallback when L4 capacity is unavailable; T4 is unsupported here.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.51.0",
        "tqdm",
        "xxhash",
        "ninja",
        "wheel",
    )
    .pip_install(
        "flash-attn==2.7.4.post1",
        extra_options="--no-build-isolation",
        env={"MAX_JOBS": "4"},
    )
    .add_local_python_source("nanovllm")
)

app = modal.App(APP_NAME)


def model_path(model_id: str) -> Path:
    """Map a Hugging Face repository ID to a safe, stable directory name."""
    return MODEL_ROOT / model_id.replace("/", "--")


@app.function(image=image, volumes={MODEL_ROOT.as_posix(): MODEL_VOLUME}, timeout=20 * 60)
def download_model(model_id: str = DEFAULT_MODEL_ID) -> str:
    """Download weights once; later invocations reuse the Modal Volume."""
    from huggingface_hub import snapshot_download

    destination = model_path(model_id)
    if not (destination / "config.json").is_file():
        snapshot_download(repo_id=model_id, local_dir=destination)
    return str(destination)


@app.function(
    image=image,
    gpu="A10",
    volumes={MODEL_ROOT.as_posix(): MODEL_VOLUME},
    timeout=20 * 60,
)
def generate(
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_tokens: int = 128,
    temperature: float = 0.6,
) -> str:
    """Generate one completion on a single L4 GPU."""
    from transformers import AutoTokenizer

    from nanovllm import LLM, SamplingParams

    path = model_path(model_id)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Model {model_id!r} is not in the Modal Volume. "
            "Run the local entrypoint so it can download the model first."
        )

    tokenizer = AutoTokenizer.from_pretrained(path)
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    llm = LLM(
        str(path),
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.6,
    )
    output = llm.generate(
        [formatted_prompt],
        SamplingParams(temperature=temperature, max_tokens=max_tokens),
        use_tqdm=False,
    )[0]
    return output["text"]


@app.local_entrypoint()
def main(
    prompt: str = "Explain KV caching in one sentence.",
    model_id: str = DEFAULT_MODEL_ID,
    max_tokens: int = 128,
    temperature: float = 0.6,
) -> None:
    """Download the selected model if needed, then run exactly one GPU job."""
    print(f"Ensuring {model_id} is available in the Modal Volume...")
    download_model.remote(model_id)
    print(
        generate.remote(
            prompt=prompt,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )
