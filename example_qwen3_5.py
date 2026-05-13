import argparse
import io
import os
import urllib.request

from PIL import Image
from nanovllm import LLM, SamplingParams
from transformers import AutoProcessor


def get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def download_image(url: str) -> Image.Image:
    """Download an image from URL and return as PIL Image."""
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    return Image.open(io.BytesIO(payload)).convert("RGB")


def parse_args() -> argparse.Namespace:
    default_max_tokens = get_env_int("out_seq_length", 2048)
    default_temperature = get_env_float("temperature", 0.6)
    parser = argparse.ArgumentParser(
        description="nano-vllm Qwen3.5 multimodal example (text and/or image+text)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model directory (same as AutoProcessor.from_pretrained).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="LLM max_model_len (KV / context budget).",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=8192,
        help="Scheduler max tokens per forward step (all sequences).",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=8,
        help="Max concurrent sequences in one batch.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Disable CUDA graph capture (enforce eager mode).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=default_max_tokens,
        help="Sampling max_tokens (env: out_seq_length).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=default_temperature,
        help="Sampling temperature (env: temperature).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("text", "multimodal", "all"),
        default="text",
        help="Which mode to run: text-only batch, multimodal, or both.",
    )
    return parser.parse_args()


def example_text(llm: LLM, processor: AutoProcessor, sampling_params: SamplingParams) -> None:
    print("=" * 50)
    print("Example 1: Text-only prompts")
    print("=" * 50)

    text_prompts = [
        "Give me a short introduction to large language models.",
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    chat_prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in text_prompts
    ]
    outputs = llm.generate(chat_prompts, sampling_params)

    for prompt, output in zip(chat_prompts, outputs):
        print(f"\nPrompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


def example_multimodal(
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
) -> None:
    print("\n" + "=" * 50)
    print("Example 2: Multimodal prompts with images")
    print("=" * 50)

    image_urls = [
        "http://images.cocodataset.org/val2017/000000000285.jpg",
        "http://images.cocodataset.org/val2017/000000000632.jpg",
    ]

    requests = []
    for url in image_urls:
        image = download_image(url)
        prompt_text = (
            "Please describe the scene in the image and highlight the primary objects."
        )

        chat_prompt = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        requests.append({
            "text": chat_prompt,
            "images": [image],
        })

    sampling_params_list = [sampling_params] * len(requests)
    outputs = llm.generate_multimodal(
        requests,
        sampling_params_list,
        processor,
        use_tqdm=True,
    )

    for req, output in zip(requests, outputs):
        print(f"\nPrompt: {req['text'][:100]}...")
        print(f"Completion: {output['text']!r}")


def main() -> None:
    args = parse_args()
    path = os.path.expanduser(args.model)

    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        is_multimodal=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )

    processor = AutoProcessor.from_pretrained(path)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.mode in ("text", "all"):
        example_text(llm, processor, sampling_params)
    if args.mode in ("multimodal", "all"):
        example_multimodal(llm, processor, sampling_params)


if __name__ == "__main__":
    main()
