import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        # Load tokenizer without use_fast to match HF behavior (same tokenization as tokenizer([text], return_tensors="pt")).
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def _expand_vision_placeholders(
        self,
        input_ids: list[int],
        image_grid_thw: torch.Tensor,
    ) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        hf_config = self.model_runner.config.hf_config
        vision_config = hf_config.vision_config
        merge_size = vision_config.spatial_merge_size

        image_token_id = getattr(hf_config, "image_token_id", None)
        vision_start_token_id = getattr(hf_config, "vision_start_token_id", None)
        vision_end_token_id = getattr(hf_config, "vision_end_token_id", None)

        if None in (image_token_id, vision_start_token_id, vision_end_token_id):
            raise ValueError("缺少视觉占位符相关的 token id 配置")

        if image_grid_thw.dim() != 2 or image_grid_thw.size(-1) != 3:
            raise ValueError("image_grid_thw 形状不正确，期望为 [num_images, 3]")

        grids = image_grid_thw.tolist()
        expected_counts = [
            int(t * h * w // (merge_size**2)) for t, h, w in grids
        ]

        original_count = input_ids.count(image_token_id)
        new_input_ids: list[int] = []
        i = 0
        image_idx = 0
        total_images = len(expected_counts)
        length = len(input_ids)

        placeholder_ranges: list[tuple[int, int]] = []

        while i < length:
            token = input_ids[i]
            if token == vision_start_token_id and image_idx < total_images:
                new_input_ids.append(token)
                i += 1
                # Skip original contents until matching vision_end_token_id
                while i < length and input_ids[i] != vision_end_token_id:
                    i += 1
                if i == length:
                    raise ValueError("vision_start_token 后未找到匹配的 vision_end_token")

                required = expected_counts[image_idx]
                start_offset = len(new_input_ids)
                new_input_ids.extend([image_token_id] * required)
                new_input_ids.append(vision_end_token_id)
                placeholder_ranges.append((start_offset, required))
                i += 1  # Skip the original vision_end token
                image_idx += 1
            else:
                new_input_ids.append(token)
                i += 1

        if image_idx != total_images:
            raise ValueError(
                f"存在 {total_images - image_idx} 个图像没有匹配的占位符"
            )

        # new_count = new_input_ids.count(image_token_id)
        # print(
        #     "[LLMEngine] expand placeholders:",
        #     {
        #         "original_count": original_count,
        #         "expected_counts": expected_counts,
        #         "new_count": new_count,
        #     },
        # )

        return new_input_ids, expected_counts, placeholder_ranges

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        images=None,
        pixel_values=None,
        image_grid_thw=None,
        vision_counts=None,
        vision_placeholders=None,
    ):
        if isinstance(prompt, str):
            # Match HF: chat template already contains <|im_start|>, <|im_end|>, etc.;
            # do not add extra BOS/special tokens so tokenization matches tokenizer([text], return_tensors="pt").
            prompt = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Robustness: clamp the requested generation length so that
        # (prompt_len + max_tokens) never exceeds `config.max_model_len`.
        # This avoids cudagraph capture shape mismatches (e.g. block_tables
        # width 8 vs 9) when users pass `max_tokens` larger than the
        # remaining context budget.
        max_model_len = self.model_runner.config.max_model_len
        input_len = len(prompt)
        if input_len > max_model_len:
            raise ValueError(
                f"Input length ({input_len}) exceeds model's maximum "
                f"context length ({max_model_len})."
            )
        remaining_budget = max_model_len - input_len
        if remaining_budget < 1:
            raise ValueError(
                f"Remaining generation budget is {remaining_budget} "
                f"(input_len={input_len}, max_model_len={max_model_len}). "
                f"Please reduce the prompt length."
            )
        if sampling_params.max_tokens > remaining_budget:
            sampling_params = SamplingParams(
                temperature=sampling_params.temperature,
                max_tokens=remaining_budget,
                ignore_eos=sampling_params.ignore_eos,
            )
        seq = Sequence(
            prompt,
            sampling_params,
            images=images,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            vision_counts=vision_counts,
            vision_placeholders=vision_placeholders,
        )
        # if self.model_runner.config.hf_config is not None:
        #     image_token_id = getattr(self.model_runner.config.hf_config, "image_token_id", None)
        #     if image_token_id is not None:
        #         placeholder_count = seq.token_ids.count(image_token_id)
        #         print(
        #             "[LLMEngine] sequence placeholder count",
        #             {"seq_id": seq.seq_id, "placeholder_count": placeholder_count, "vision_counts": vision_counts},
        #         )
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        # Clean up GDN recurrent/conv states for finished sequences to avoid
        # memory leaks and stale state reuse when seq_ids are recycled.
        finished_seqs = [seq for seq in seqs if seq.is_finished]
        if finished_seqs:
            self.model_runner.call("cleanup_seq_states", [seq.seq_id for seq in finished_seqs])
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]
        num_tokens = (
            sum(len(seq) for seq in seqs)
            if is_prefill
            else -len(seqs)
        )
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(
                total=len(prompts),
                desc="Generating",
                dynamic_ncols=True,
            )
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [
            outputs[seq_id]
            for seq_id in sorted(outputs.keys())
        ]
        outputs = [
            {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
            }
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs

    def generate_multimodal(
        self,
        requests: list[dict],
        sampling_params: SamplingParams | list[SamplingParams],
        processor,
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(
                total=len(requests),
                desc="Generating",
                dynamic_ncols=True,
            )

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(requests)

        for request, sp in zip(requests, sampling_params):
            messages = request.get("messages")
            text = request.get("text")
            images = request.get("images")

            if text is None:
                if messages is None:
                    raise ValueError(
                        "multimodal request requires 'text' or 'messages'"
                    )

                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                if images is None:
                    extracted_images = []
                    for message in messages:
                        for content in message.get("content", []):
                            is_image = content.get("type") == "image"
                            has_payload = "image" in content
                            if is_image and has_payload:
                                extracted_images.append(content["image"])
                    images = extracted_images if extracted_images else None

            if images is not None and not isinstance(images, (list, tuple)):
                images = [images]

            processor_kwargs = {
                "text": [text],
                "return_tensors": "pt",
                "padding": True,
            }
            if images:
                processor_kwargs["images"] = images

            processor_outputs = processor(**processor_kwargs)

            input_ids = processor_outputs["input_ids"][0].tolist()
            pixel_values = processor_outputs.get("pixel_values")
            image_grid_thw = processor_outputs.get("image_grid_thw")

            vision_counts = []
            vision_placeholders = []
            if image_grid_thw is not None:
                expanded_input_ids, vision_counts, vision_placeholders = self._expand_vision_placeholders(
                    input_ids,
                    image_grid_thw.squeeze(0) if image_grid_thw.dim() == 3 else image_grid_thw,
                )
                input_ids = expanded_input_ids

            if pixel_values is not None:
                pixel_values = pixel_values.contiguous().cpu()

            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.contiguous().cpu()

            self.add_request(
                input_ids,
                sp,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                vision_counts=vision_counts,
                vision_placeholders=vision_placeholders,
            )

        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        outputs = [
            outputs[seq_id]
            for seq_id in sorted(outputs.keys())
        ]
        results = [
            {
                "text": self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                "token_ids": token_ids,
            }
            for token_ids in outputs
        ]

        if use_tqdm:
            pbar.close()

        return results
