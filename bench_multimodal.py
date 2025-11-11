"""Benchmark script for nano-vllm multimodal inference."""

from __future__ import annotations

import argparse
import io
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterable

from PIL import Image

from nanovllm import LLM, SamplingParams


DEFAULT_IMAGE_URLS: tuple[str, ...] = (
    # A subset of COCO validation images with diverse scenes.
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


def run_benchmark(
    model_path: str,
    max_new_tokens: int,
    temperature: float,
    image_urls: Iterable[str],
) -> BenchmarkResult:
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        is_multimodal=True,
    )

    # Imported lazily to avoid the dependency when this benchmark is unused.
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoProcessor,
    )

    processor = AutoProcessor.from_pretrained(model_path)

    requests = build_requests(processor, image_urls, DEFAULT_PROMPT)
    num_requests = len(requests)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    sampling_params_list = [sampling_params] * num_requests

    # Warm-up with a single request to exclude one-time costs.
    llm.generate_multimodal(
        [requests[0]],
        sampling_params_list[0],
        processor,
        use_tqdm=False,
    )

    start = time.perf_counter()
    outputs = llm.generate_multimodal(
        requests,
        sampling_params_list,
        processor,
        use_tqdm=False,
    )
    latency = time.perf_counter() - start

    total_generated = sum(len(item["token_ids"]) for item in outputs)

    # nano-vllm does not report prompt token lengths; estimate them here.
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
    parser = argparse.ArgumentParser(
        description="Benchmark nano-vllm multimodal inference."
    )
    default_out_seq = get_env_int("out_seq_length", 64)
    default_temperature = get_env_float("temperature", 0.2)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Path to the Qwen3-VL model directory "
            "(the same path used by AutoProcessor)."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=default_out_seq,
        help="Maximum number of tokens to generate per request.",
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
        help=(
            "Number of images to benchmark "
            "(picked from a fixed COCO subset)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_urls = DEFAULT_IMAGE_URLS[: args.num_images]
    if not image_urls:
        raise ValueError("num-images must be positive.")

    result = run_benchmark(
        model_path=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        image_urls=image_urls,
    )

    print("=== nano-vllm multimodal benchmark ===")
    print(f"Requests          : {result.num_requests}")
    print(f"Prompt tokens     : {result.total_prompt_tokens}")
    print(f"Generated tokens  : {result.total_generated_tokens}")
    print(f"Latency           : {result.latency:.2f}s")
    print(f"Throughput        : {result.tok_per_sec:.2f} tok/s")
    if os.getenv("top_p") or os.getenv("top_k") or os.getenv("repetition_penalty"):
        print(
            "Note: Sampling parameters such as top_p/top_k/repetition_penalty "
            "are not currently supported by nano-vllm and were ignored."
        )


if __name__ == "__main__":
    main()

