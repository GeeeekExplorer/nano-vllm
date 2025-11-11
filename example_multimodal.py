"""Minimal multimodal inference example using Qwen3-VL."""

from __future__ import annotations

import io
import os
import urllib.request

from PIL import Image
from transformers import AutoProcessor

from nanovllm import LLM, SamplingParams


DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000000285.jpg"
DEFAULT_PROMPT = (
    "Please describe the scene in the image and highlight the main objects."
)


def download_image(url: str) -> Image.Image:
    with urllib.request.urlopen(url) as response:
        data = response.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def main() -> None:
    model_path = os.path.expanduser("~/huggingface/Qwen3-VL-2B-Instruct/")
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        is_multimodal=True,
    )

    processor = AutoProcessor.from_pretrained(model_path)
    image = download_image(DEFAULT_IMAGE_URL)
    chat_prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": DEFAULT_PROMPT},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    request = {"text": chat_prompt, "images": [image]}
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
    outputs = llm.generate_multimodal(
        [request],
        sampling_params,
        processor,
        use_tqdm=False,
    )

    print("Prompt:", DEFAULT_PROMPT)
    print("Image URL:", DEFAULT_IMAGE_URL)
    print("Completion:", outputs[0]["text"])


if __name__ == "__main__":
    main()

