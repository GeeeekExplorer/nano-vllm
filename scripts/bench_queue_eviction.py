"""Paired single-GPU benchmark with a benchmark-only physical KV capacity limit."""
import argparse
import ast
import atexit
import hashlib
import inspect
import json
import random
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine import model_runner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def limit_kv_capacity(blocks):
    """Replace only the profiling-derived capacity in the original allocator."""
    original = model_runner.ModelRunner.allocate_kv_cache
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    matches = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr == "num_kvcache_blocks"
            for t in node.targets
        ):
            node.value = ast.Constant(blocks)
            matches += 1
    assert matches == 1, "KV allocator changed; review this benchmark override"
    scope = vars(model_runner).copy()
    exec(compile(ast.fix_missing_locations(tree), "<benchmark-kv-capacity>", "exec"), scope)
    model_runner.ModelRunner.allocate_kv_cache = scope["allocate_kv_cache"]
    return original


def make_trace(tokenizer, unique, seed, waves):
    rng = random.Random(seed)
    families = {}
    trace = []
    for wave in range(waves):
        prompts = []
        for j in range(8):
            index = wave * 8 + j
            family = index if unique else rng.choices(range(8), weights=[4, 3, 2, 1, 1, 1, 1, 1])[0]
            if family not in families:
                text = f"Document {family}: This technical document describes system {family}. "
                text += "Memory allocation, request scheduling, cache management and performance analysis. "
                ids = tokenizer.encode(text, add_special_tokens=False)
                families[family] = (ids * (4096 // len(ids) + 1))[:4096]
            suffix = [1000 + ((index * 137 + i * 17) % 19000) for i in range(64)]
            prompts.append(families[family] + suffix)
        trace.append(prompts)
    return trace


def run(engine, trace, lookahead, output_tokens):
    assert engine.scheduler.is_finished()
    engine.model_runner.config.prefix_cache_lookahead = lookahead
    engine.scheduler = Scheduler(engine.model_runner.config)
    torch.manual_seed(20260919)
    params = SamplingParams(temperature=.8, max_tokens=output_tokens, ignore_eos=True)
    metrics = dict(lookahead=lookahead, scheduler_seconds=0., prefill_tokens=0,
                   input_tokens=sum(len(p) for wave in trace for p in wave),
                   output_tokens=0, ttft_ms=[], requests=0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for wave in trace:
        requests = []
        first = {}
        arrived = time.perf_counter()
        for prompt in wave:
            request = Sequence(prompt, params)
            requests.append(request)
            engine.scheduler.add(request)
        while not engine.scheduler.is_finished():
            before = time.perf_counter()
            seqs, prefill = engine.scheduler.schedule()
            metrics["scheduler_seconds"] += time.perf_counter() - before
            if prefill:
                metrics["prefill_tokens"] += sum(s.num_scheduled_tokens for s in seqs)
            lengths = [s.num_completion_tokens for s in seqs]
            tokens = engine.model_runner.call("run", seqs, prefill)
            engine.scheduler.postprocess(seqs, tokens, prefill)
            now = time.perf_counter()
            for seq, previous in zip(seqs, lengths):
                if seq.num_completion_tokens > previous:
                    first.setdefault(seq.seq_id, (now - arrived) * 1000)
        assert all(s.num_completion_tokens == output_tokens for s in requests)
        metrics["ttft_ms"].extend(first[s.seq_id] for s in requests)
        metrics["requests"] += len(requests)
        metrics["output_tokens"] += sum(s.num_completion_tokens for s in requests)
    torch.cuda.synchronize()
    metrics["wall_seconds"] = time.perf_counter() - start
    metrics["output_tok_s"] = metrics["output_tokens"] / metrics["wall_seconds"]
    metrics["p95_ttft_ms"] = float(np.percentile(metrics["ttft_ms"], 95))
    metrics["cache_hit_fraction"] = 1 - metrics["prefill_tokens"] / metrics["input_tokens"]
    manager = engine.scheduler.block_manager
    assert not manager.used_block_ids
    assert len(manager.free_block_ids) == len(manager.blocks)
    assert all(b.ref_count == 0 for b in manager.blocks)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", choices=["pressure", "ample", "unique"], required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite results; choose a new output path")
    if min(args.pairs, args.waves, args.output_tokens) < 1:
        parser.error("pairs, waves and output-tokens must be positive")
    blocks = 160 if args.scenario == "ample" else 68
    original_allocator = limit_kv_capacity(blocks)
    try:
        engine = LLM(args.model, max_num_seqs=8, max_num_batched_tokens=8192,
                     max_model_len=8192, gpu_memory_utilization=.8, enforce_eager=False)
    finally:
        model_runner.ModelRunner.allocate_kv_cache = original_allocator
    atexit.unregister(engine.exit)
    try:
        trace = make_trace(engine.tokenizer, args.scenario == "unique", args.seed, args.waves)
        result = dict(scenario=args.scenario, seed=args.seed, pairs=args.pairs,
                      physical_blocks=blocks, block_size=256, model=args.model,
                      kv_bytes=engine.model_runner.kv_cache.numel() * engine.model_runner.kv_cache.element_size(),
                      gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      output_tokens_per_request=args.output_tokens, cuda_graph=True,
                      trace_sha256=hashlib.sha256(json.dumps(trace).encode()).hexdigest(),
                      warmups=[], runs=[])
        for window in (0, 16):
            result["warmups"].append(run(engine, trace, window, args.output_tokens))
        for pair in range(args.pairs):
            for window in ((0, 16) if pair % 2 == 0 else (16, 0)):
                metrics = run(engine, trace, window, args.output_tokens)
                metrics["pair"] = pair
                result["runs"].append(metrics)
                args.output.write_text(json.dumps(result, indent=2))
                print(json.dumps({k: v for k, v in metrics.items() if k != "ttft_ms"}), flush=True)
        result["complete"] = True
        args.output.write_text(json.dumps(result, indent=2))
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
