"""Targeted full-vocabulary logit diagnostic; not a performance benchmark."""
import argparse
import atexit
import json
from pathlib import Path

from bench_queue_eviction import limit_kv_capacity
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine import model_runner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


class Capture(torch.nn.Module):
    def forward(self, logits, temperatures):
        self.logits = logits.detach().float().cpu()
        return torch.full((logits.shape[0],), 1000, device=logits.device, dtype=torch.long)


def execute(engine, prompts, count):
    requests = [Sequence(p, SamplingParams(max_tokens=count, ignore_eos=True)) for p in prompts]
    for request in requests:
        engine.scheduler.add(request)
    rows = {}
    while not engine.scheduler.is_finished():
        seqs, prefill = engine.scheduler.schedule()
        tokens = engine.model_runner.call("run", seqs, prefill)
        for seq, logits in zip(seqs, engine.model_runner.sampler.logits):
            rows[(requests.index(seq), seq.num_completion_tokens)] = logits
        engine.scheduler.postprocess(seqs, tokens, prefill)
    assert len(rows) == len(prompts) * count
    assert all(torch.isfinite(row).all() for row in rows.values())
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite results")
    original = limit_kv_capacity(102)
    try:
        engine = LLM(args.model, max_num_seqs=8, max_num_batched_tokens=8192,
                     max_model_len=8192, gpu_memory_utilization=.8, enforce_eager=False)
    finally:
        model_runner.ModelRunner.allocate_kv_cache = original
    atexit.unregister(engine.exit)
    engine.model_runner.sampler = Capture()
    try:
        def prompt(family, index):
            text = f"Document {family}: This technical document describes system {family}. "
            text += "Memory allocation, request scheduling, cache management and performance analysis. "
            ids = engine.tokenizer.encode(text, add_special_tokens=False)
            prefix = (ids * (4096 // len(ids) + 1))[:4096]
            return prefix + [1000 + ((index * 137 + i * 17) % 19000) for i in range(64)]

        results = {}
        for name, window, blocks in (("baseline", 0, 68), ("queue", 16, 68),
                                     ("cache_hit_control", 0, 102)):
            config = engine.model_runner.config
            config.num_kvcache_blocks = blocks
            config.prefix_cache_lookahead = window
            engine.scheduler = Scheduler(config)
            engine.model_runner.kv_cache.fill_(float("nan"))
            for family in range(4):
                execute(engine, [prompt(family, 1000 + family)], 1)
            results[name] = execute(engine, [prompt(f, 2000 + f) for f in (4, 0, 1, 2)], 16)
        report = {"physical_blocks": 102, "teacher_forced_token": 1000, "comparisons": {}}
        for first, second in (("baseline", "queue"), ("baseline", "cache_hit_control"),
                              ("cache_hit_control", "queue")):
            reference, candidate = results[first], results[second]
            assert reference.keys() == candidate.keys()
            differences = [(candidate[k] - reference[k]) for k in reference]
            report["comparisons"][first + "__" + second] = {
                "rows": len(reference),
                "max_abs": max(d.abs().max().item() for d in differences),
                "max_relative_l2": max((d.norm() / reference[k].norm().clamp_min(1e-8)).item()
                                       for k, d in zip(reference, differences)),
                "argmax_matches": sum(int(reference[k].argmax() == candidate[k].argmax()) for k in reference),
            }
        report["passed"] = report["comparisons"]["cache_hit_control__queue"]["max_abs"] == 0
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        assert report["passed"], "Queue policy differed from baseline with matching cache hits"
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
