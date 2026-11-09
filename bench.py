import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path


MAX_MODEL_LEN = 4096


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark offline batch generation")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--num-seqs", type=positive_int, default=256)
    parser.add_argument("--min-input-len", type=positive_int, default=100)
    parser.add_argument("--max-input-len", type=positive_int, default=1024)
    parser.add_argument("--min-output-len", type=positive_int, default=100)
    parser.add_argument("--max-output-len", type=positive_int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=positive_int, default=1)
    parser.add_argument("--output", type=Path, help="write JSON results to this path")
    args = parser.parse_args(argv)
    if args.min_input_len > args.max_input_len or args.min_output_len > args.max_output_len:
        parser.error("minimum length must not exceed maximum length")
    if args.max_input_len + args.max_output_len > MAX_MODEL_LEN:
        parser.error(f"maximum input and output lengths must sum to at most {MAX_MODEL_LEN}")
    return args


def make_workload(args, rng, sampling_params):
    prompts = [[rng.randint(0, 10000) for _ in range(rng.randint(args.min_input_len, args.max_input_len))]
               for _ in range(args.num_seqs)]
    params = [sampling_params(temperature=0.6, ignore_eos=True,
                              max_tokens=rng.randint(args.min_output_len, args.max_output_len))
              for _ in range(args.num_seqs)]
    return prompts, params


def metadata():
    # nanovllm has already imported torch; do not require it for --help or tests.
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    result = {"torch": torch.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        result["device"] = torch.cuda.get_device_name()
    return result


def main(argv=None):
    args = parse_args(argv)
    from nanovllm import LLM, SamplingParams

    model = os.path.expanduser(args.model)
    llm = LLM(model, enforce_eager=False, max_model_len=MAX_MODEL_LEN)
    rng = random.Random(args.seed)
    llm.generate(["Benchmark: "], SamplingParams())  # warmup, excluded from results
    runs = []
    for index in range(args.runs):
        prompts, params = make_workload(args, rng, SamplingParams)
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        seconds = time.perf_counter() - start
        tokens = sum(len(output["token_ids"]) for output in outputs)
        runs.append({"seconds": seconds, "generated_tokens": tokens})
        print(f"Run {index + 1}: Total: {tokens}tok, Time: {seconds:.2f}s, "
              f"Throughput: {tokens / seconds:.2f}tok/s")

    latencies = sorted(run["seconds"] for run in runs)
    total_seconds = sum(latencies)
    total_tokens = sum(run["generated_tokens"] for run in runs)
    report = {
        "config": {"model": model, "num_seqs": args.num_seqs,
                   "min_input_len": args.min_input_len, "max_input_len": args.max_input_len,
                   "min_output_len": args.min_output_len, "max_output_len": args.max_output_len,
                   "seed": args.seed, "runs": args.runs, "max_model_len": MAX_MODEL_LEN,
                   "temperature": 0.6, "ignore_eos": True, "enforce_eager": False},
        "runs": runs,
        "summary": {"median_batch_seconds": statistics.median(latencies),
                    "p95_batch_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1],
                    "total_seconds": total_seconds, "total_generated_tokens": total_tokens,
                    "throughput_tokens_per_second": total_tokens / total_seconds},
        "metadata": metadata(),
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
