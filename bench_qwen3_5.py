"""Benchmark script for nano-vllm Qwen3.5: multimodal (image+text) or pure text.

Text mode mirrors nano-vllm/bench.py: random prompt token ids, batched llm.generate().
Defaults to CUDA graph decode for maximum throughput.
"""

import argparse
import io
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterable

import torch
from PIL import Image

from nanovllm import LLM, SamplingParams


DEFAULT_IMAGE_URLS: tuple[str, ...] = (
    "http://images.cocodataset.org/val2017/000000000285.jpg",
    "http://images.cocodataset.org/val2017/000000000632.jpg",
    "http://images.cocodataset.org/val2017/000000000724.jpg",
    "http://images.cocodataset.org/val2017/000000000776.jpg",
    "http://images.cocodataset.org/val2017/000000001000.jpg",
    "http://images.cocodataset.org/val2017/000000001268.jpg",
    "http://images.cocodataset.org/val2017/000000006012.jpg",
    "http://images.cocodataset.org/val2017/000000190236.jpg",
    "http://images.cocodataset.org/val2017/000000331352.jpg",
    "http://images.cocodataset.org/val2017/000000517069.jpg",
)

DEFAULT_PROMPT = (
    "Please describe the scene in the image and highlight the primary objects."
)


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
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    return Image.open(io.BytesIO(payload)).convert("RGB")


@dataclass
class BenchmarkResult:
    num_requests: int
    total_prompt_tokens: int
    total_generated_tokens: int
    latency: float

    @property
    def tok_per_sec(self) -> float:
        if self.latency <= 0:
            return 0.0
        return self.total_generated_tokens / self.latency


def build_requests(
    processor,
    image_urls: Iterable[str],
    prompt: str,
) -> list[dict]:
    requests = []
    for url in image_urls:
        image = download_image(url)
        chat_prompt = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        requests.append(
            {
                "text": chat_prompt,
                "images": [image],
                "meta": {"url": url},
            }
        )
    return requests


def run_benchmark_text(
    model_path: str,
    num_seqs: int,
    min_prompt_len: int,
    max_prompt_len: int,
    min_output_len: int,
    max_output_len: int,
    temperature: float,
    seed: int,
    ignore_eos: bool,
    enforce_eager: bool,
) -> BenchmarkResult:
    """Pure-text throughput test (same spirit as nano-vllm/bench.py)."""
    from transformers import AutoTokenizer

    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    hi = min(10_000, max(vocab_size - 1, 1))

    prompt_token_ids = [
        [random.randint(0, hi) for _ in range(random.randint(min_prompt_len, max_prompt_len))]
        for _ in range(num_seqs)
    ]
    sampling_params_list = [
        SamplingParams(
            temperature=temperature,
            max_tokens=random.randint(min_output_len, max_output_len),
            ignore_eos=ignore_eos,
        )
        for _ in range(num_seqs)
    ]

    max_prompt_toks = max(len(p) for p in prompt_token_ids)
    max_out_toks = max(sp.max_tokens for sp in sampling_params_list)
    max_model_len = max(2048, max_prompt_toks + max_out_toks + 32)
    max_num_batched_tokens = max(16384, num_seqs * max_prompt_len + 4096)

    if not enforce_eager:
        os.environ["NANOVLLM_ALLOW_QWEN35_DECODE_CUDAGRAPH"] = "1"

    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        is_multimodal=False,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max(16, num_seqs),
    )

    # Warm-up
    llm.generate(
        ["Benchmark: "],
        SamplingParams(temperature=temperature, max_tokens=8),
        use_tqdm=False,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate(
        prompt_token_ids,
        sampling_params_list,
        use_tqdm=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - start

    total_generated = sum(len(item["token_ids"]) for item in outputs)
    total_prompt_tokens = sum(len(p) for p in prompt_token_ids)

    return BenchmarkResult(
        num_requests=num_seqs,
        total_prompt_tokens=total_prompt_tokens,
        total_generated_tokens=total_generated,
        latency=latency,
    )


def run_benchmark_multimodal(
    model_path: str,
    max_new_tokens: int,
    temperature: float,
    image_urls: Iterable[str],
    enforce_eager: bool = False,
) -> BenchmarkResult:
    from transformers import AutoProcessor

    urls = list(image_urls)
    num_urls = len(urls)

    if not enforce_eager:
        os.environ["NANOVLLM_ALLOW_QWEN35_DECODE_CUDAGRAPH"] = "1"

    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        is_multimodal=True,
        max_model_len=2048,
        max_num_batched_tokens=16384,
        max_num_seqs=max(16, num_urls),
    )

    processor = AutoProcessor.from_pretrained(model_path)

    requests = build_requests(processor, urls, DEFAULT_PROMPT)
    num_requests = len(requests)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    sampling_params_list = [sampling_params] * num_requests

    # Warm-up with a single request
    llm.generate_multimodal(
        [requests[0]],
        sampling_params_list[:1],
        processor,
        use_tqdm=False,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate_multimodal(
        requests,
        sampling_params_list,
        processor,
        use_tqdm=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - start

    total_generated = sum(len(item["token_ids"]) for item in outputs)

    prompt_token_lengths = 0
    for req in requests:
        encoded = processor(
            text=[req["text"]],
            images=req["images"],
            return_tensors="pt",
        )
        prompt_token_lengths += encoded["input_ids"].shape[-1]

    return BenchmarkResult(
        num_requests=num_requests,
        total_prompt_tokens=prompt_token_lengths,
        total_generated_tokens=total_generated,
        latency=latency,
    )


def parse_args() -> argparse.Namespace:
    default_out_seq = get_env_int("out_seq_length", 2048)
    default_temperature = get_env_float("temperature", 0.2)

    parser = argparse.ArgumentParser(
        description="Benchmark nano-vllm Qwen3.5 (multimodal or pure text)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the Qwen3.5 model directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("multimodal", "text"),
        default="text",
        help=(
            "multimodal: image+COCO URLs via generate_multimodal; "
            "text: random prompt ids + llm.generate (like nano-vllm/bench.py)."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=default_out_seq,
        help="(multimodal) Maximum new tokens per request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=default_temperature,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=len(DEFAULT_IMAGE_URLS),
        help="(multimodal) Number of images (picked from a fixed COCO subset).",
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=64,
        help="(text) Number of concurrent random prompts.",
    )
    parser.add_argument(
        "--min-prompt-len",
        type=int,
        default=100,
        help="(text) Inclusive lower bound for random prompt length.",
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=1024,
        help="(text) Inclusive upper bound for random prompt length.",
    )
    parser.add_argument(
        "--min-output-len",
        type=int,
        default=100,
        help="(text) Inclusive lower bound for max_tokens per sequence.",
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=1024,
        help="(text) Inclusive upper bound for max_tokens per sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="(text) RNG seed for reproducible random prompts/sampling params.",
    )
    parser.add_argument(
        "--ignore-eos",
        dest="ignore_eos",
        action="store_true",
        help="(text) Force decoding until max_tokens (higher sustained tok/s).",
    )
    parser.add_argument(
        "--no-ignore-eos",
        dest="ignore_eos",
        action="store_false",
        help="(text) Stop at EOS.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Disable CUDA graph capture (enforce eager mode).",
    )
    parser.set_defaults(ignore_eos=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model = os.path.expanduser(args.model)

    if args.mode == "text":
        if args.num_seqs < 1:
            raise ValueError("num-seqs must be positive.")
        if args.min_prompt_len > args.max_prompt_len:
            raise ValueError("min-prompt-len must be <= max-prompt-len.")
        if args.min_output_len > args.max_output_len:
            raise ValueError("min-output-len must be <= max-output-len.")
        result = run_benchmark_text(
            model_path=args.model,
            num_seqs=args.num_seqs,
            min_prompt_len=args.min_prompt_len,
            max_prompt_len=args.max_prompt_len,
            min_output_len=args.min_output_len,
            max_output_len=args.max_output_len,
            temperature=args.temperature,
            seed=args.seed,
            ignore_eos=args.ignore_eos,
            enforce_eager=args.enforce_eager,
        )
        print("=== nano-vllm Qwen3.5 text benchmark ===")
        print(
            "Mode              : random prompt token ids, llm.generate() "
            "(see nano-vllm/bench.py)"
        )
    else:
        image_urls = DEFAULT_IMAGE_URLS[: args.num_images]
        if not image_urls:
            raise ValueError("num-images must be positive.")
        result = run_benchmark_multimodal(
            model_path=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            image_urls=image_urls,
            enforce_eager=args.enforce_eager,
        )
        print("=== nano-vllm Qwen3.5 multimodal benchmark ===")
        print(
            "Mode              : one generate_multimodal() over all requests "
            "(batched in engine)"
        )

    print(f"Requests          : {result.num_requests}")
    print(f"Prompt tokens     : {result.total_prompt_tokens}")
    print(f"Generated tokens  : {result.total_generated_tokens}")
    print(f"Latency           : {result.latency:.2f}s")
    print(f"Throughput        : {result.tok_per_sec:.2f} tok/s")


if __name__ == "__main__":
    main()
