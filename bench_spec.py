"""Speculative decoding benchmarks on nano-vllm.

  --mode closed-loop  protocol v2: keep C requests in the engine, adding one as each finishes, until the panel is
                      used up (bench-v2 §1.1). Sampling, KV budgets, warm-up and the line's draft come from the protocol.
  --mode offline      v1 (default): one generate() over the 48-prompt panel, so the v1 results stay reproducible.

One invocation = one configuration = one JSON result with a manifest, per-step timings, per-request
records and acceptance statistics. tools/summarize_bench.py turns results into metrics.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import TypedDict

import torch

from nanovllm.config import BLOCK_DRAFTS

ROOT = Path(__file__).resolve().parent
GIB = 1 << 30
WARMUP_PROMPTS = 4
SCHEMA = "bench-v2"
RESULT_KEYS = ("schema", "protocol_sha256", "manifest", "gpu_state_start", "gpu_state_end", "config",
               "wall_seconds", "steps", "phases", "requests", "speculative_stats")
QUERY_FIELDS = ("name", "power.limit", "clocks.max.sm", "clocks.max.memory",
                "ecc.mode.current", "temperature.gpu", "clocks_event_reasons.active")


class PanelEntry(TypedDict):
    id: str
    category: str
    prompt_token_ids: list[int]


class RequestRecord(TypedDict):
    id: str
    category: str
    prompt_len: int
    submit_step: int                  # engine.num_steps when add_request ran
    deliveries: list[list[int]]       # [[step, n_tokens], ...] in step order
    finish_step: int                  # the step that finished the request
    token_ids: list[int]


class ClosedLoopRecord(TypedDict):
    requests: list[RequestRecord]     # in panel order


def load_panel(path: Path, concurrency: int, take: str) -> list[PanelEntry]:
    """take "full_cycles": L·⌈max(L, 4C)/L⌉ entries, whole cycles of the L-entry panel; "prefix": max(48, 4C), cycling."""
    panel = json.loads(Path(path).read_text())
    count = {"full_cycles": len(panel) * math.ceil(max(len(panel), 4 * concurrency) / len(panel)),
             "prefix": max(48, 4 * concurrency)}[take]
    return [panel[index % len(panel)] for index in range(count)]


def kv_budget_bytes(panel: list[PanelEntry], concurrency: int, max_tokens: int, k: int,
                    block_bytes: int, block_size: int = 256) -> int:
    """C × ⌈(longest prompt + max_tokens + k) / block_size⌉ blocks: with C requests in flight nothing is preempted."""
    longest = max(len(entry["prompt_token_ids"]) for entry in panel)
    return concurrency * math.ceil((longest + max_tokens + k) / block_size) * block_bytes


def kv_block_bytes(model_dir: str, block_size: int = 256) -> int:
    """One KV block of a bf16 model: K and V of every layer for block_size tokens, as ModelRunner.allocate_kv_cache sizes it."""
    config = json.loads((Path(model_dir) / "config.json").read_text())
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    return 2 * config["num_hidden_layers"] * block_size * config["num_key_value_heads"] * head_dim * 2


def engine_kwargs(protocol: dict, line: dict, panel: list[PanelEntry], concurrency: int, seed: int,
                  models_root: str) -> tuple[str, str | None, dict]:
    """Target directory, draft directory and LLM keyword arguments of one protocol line. Every line gets the target
    KV budget of the largest k; a model draft gets the same block count in its own pool, a block draft's KV mirrors
    the target's blocks. The protocol names block drafts by their draft_type (dflash, dspark)."""
    max_tokens = protocol["sampling"]["max_tokens"]
    budget_k = max(spec["k"] for spec in protocol["lines"].values())
    target = f"{models_root}/{protocol['models']['target']}"
    kwargs = dict(max_model_len=protocol["max_model_len"], max_num_seqs=concurrency,
                  max_num_batched_tokens=protocol["max_num_batched_tokens"], enable_prefix_cache=False,
                  kv_cache_memory_bytes=kv_budget_bytes(panel, concurrency, max_tokens, budget_k, kv_block_bytes(target)),
                  seed=seed, record_step_timings=True)
    draft = None
    if line["drafter"] is not None:
        draft = f"{models_root}/{protocol['models']['drafts'][line['drafter']]}"
        kwargs.update(draft_model=draft, num_speculative_tokens=line["k"])
        if line["drafter"] in BLOCK_DRAFTS:
            kwargs.update(draft_type=line["drafter"])
        else:
            kwargs.update(draft_kv_cache_memory_bytes=kv_budget_bytes(panel, concurrency, max_tokens, budget_k,
                                                                      kv_block_bytes(draft)))
    return target, draft, kwargs


def run_closed_loop(llm, panel: list[PanelEntry], params, concurrency: int) -> ClosedLoopRecord:
    """Before every step() top the engine up to C requests; after it, record each in-flight request's new tokens
    as one delivery. llm only needs add_request / step / num_steps, so the tests drive a fake engine."""
    records, in_flight = [], []    # in_flight items: [record, sequence, tokens delivered so far]
    next_entry = 0
    while in_flight or next_entry < len(panel):
        while len(in_flight) < concurrency and next_entry < len(panel):
            entry = panel[next_entry]
            record = RequestRecord(id=entry["id"], category=entry["category"], prompt_len=len(entry["prompt_token_ids"]),
                                   submit_step=llm.num_steps, deliveries=[], finish_step=-1, token_ids=[])
            in_flight.append([record, llm.add_request(entry["prompt_token_ids"], params), 0])
            records.append(record)
            next_entry += 1
        step = llm.num_steps
        llm.step()
        for item in in_flight:
            record, seq, delivered = item
            count = seq.num_completion_tokens    # an int; completion_token_ids would copy a list per request per step
            if count > delivered:
                record["deliveries"].append([step, count - delivered])
                item[2] = count
            if seq.is_finished:
                record["finish_step"] = step
                record["token_ids"] = seq.completion_token_ids
        in_flight = [item for item in in_flight if not item[1].is_finished]
    return ClosedLoopRecord(requests=records)


def gpu_state(run=subprocess.run) -> dict:
    """QUERY_FIELDS of the first GPU and the node name; recorded before and after a run."""
    query = run(["nvidia-smi", f"--query-gpu={','.join(QUERY_FIELDS)}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True).stdout.splitlines()[0]
    node = run(["hostname"], capture_output=True, text=True, check=True).stdout.strip()
    return {**dict(zip(QUERY_FIELDS, (value.strip() for value in query.split(",")))), "node": node}


def assert_gpu(state: dict, expected_name: str) -> None:
    if state["name"] != expected_name:
        print(f"wrong GPU: {state['name']!r}; the protocol expects {expected_name!r}")
        raise SystemExit(3)


def installed_packages(distributions=importlib.metadata.distributions) -> tuple[str, str]:
    """Sorted `name==version` lines of every installed distribution, and their SHA256. The envs have no pip."""
    text = "".join(sorted(f"{dist.metadata['Name']}=={dist.version}\n" for dist in distributions()))
    return text, hashlib.sha256(text.encode()).hexdigest()


def build_result(**parts) -> dict:
    return {"schema": SCHEMA, **{key: parts[key] for key in RESULT_KEYS[1:]}}


def manifest(commit: str) -> dict:
    import flash_attn
    import triton
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip()
    return {"gpu": torch.cuda.get_device_name(), "driver": driver, "torch": torch.__version__, "cuda": torch.version.cuda,
            "flash_attn": flash_attn.__version__, "triton": triton.__version__, "python": platform.python_version(),
            "commit": commit}


def parse_closed_loop_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks" / "protocol-v2.json")
    parser.add_argument("--panel", required=True, help="a key of the protocol's panels (D48, K128)")
    parser.add_argument("--line", required=True, help="a key of the protocol's lines (k0, model_k3, ...)")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=1, help="2 for the timing-noise repeat")
    parser.add_argument("--models-root", default="/projects/nanovllm/models")
    parser.add_argument("--commit", required=True, help="git commit of the synced tree (the sync step refuses a dirty tree)")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def closed_loop(args: argparse.Namespace) -> None:
    from nanovllm import LLM, SamplingParams

    protocol_text = args.protocol.read_text()
    protocol = json.loads(protocol_text)
    state_start = gpu_state()
    assert_gpu(state_start, protocol["gpu"]["name"])
    panel_spec = protocol["panels"][args.panel]
    panel_path = ROOT / panel_spec["file"]
    panel = load_panel(panel_path, args.concurrency, panel_spec["take"])
    line = protocol["lines"][args.line]
    sampling = protocol["sampling"]
    target, draft, kwargs = engine_kwargs(protocol, line, panel, args.concurrency, args.seed, args.models_root)
    llm = LLM(target, **kwargs)
    rec = dict(temperature=sampling["temperature"], top_k=sampling["top_k"], top_p=sampling["top_p"],
               ignore_eos=sampling["ignore_eos"])
    warmup = [panel[index % len(panel)] for index in range(args.concurrency)]
    run_closed_loop(llm, warmup, SamplingParams(max_tokens=protocol["warmup"]["max_tokens"], **rec), args.concurrency)
    llm.start_measurement()
    wall_start = time.perf_counter()
    record = run_closed_loop(llm, panel, SamplingParams(max_tokens=sampling["max_tokens"], **rec), args.concurrency)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    steps, phases = llm.step_timings(), llm.phase_timings()
    speculative_stats = dict(llm.speculative.stats) if llm.speculative is not None else None
    state_end = gpu_state()
    packages, packages_sha256 = installed_packages()
    config = {"mode": "closed-loop", "panel": args.panel, "line": args.line, "drafter": line["drafter"],
              "num_speculative_tokens": line["k"], "concurrency": args.concurrency, "seed": args.seed,
              "repeat": args.repeat, "num_requests": len(panel), "sampling": {**rec, "max_tokens": sampling["max_tokens"]},
              "target": target, "draft": draft,
              **{key: kwargs.get(key) for key in ("max_model_len", "max_num_batched_tokens", "kv_cache_memory_bytes",
                                                  "draft_kv_cache_memory_bytes", "draft_type")}}
    result = build_result(
        protocol_sha256=hashlib.sha256(protocol_text.encode()).hexdigest(),
        manifest={**manifest(args.commit), "node": state_start["node"], "packages_sha256": packages_sha256,
                  "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest()},
        gpu_state_start=state_start, gpu_state_end=state_end, config=config, wall_seconds=wall_seconds,
        steps=steps, phases=phases, requests=record["requests"], speculative_stats=speculative_stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result))
    args.output.with_name("packages.txt").write_text(packages)
    tokens = sum(len(request["token_ids"]) for request in record["requests"])
    print(f"{args.output}: {tokens} tokens in {wall_seconds:.1f}s wall")
    llm.exit()


def parse_offline_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--target-kv-gib", type=float, default=10)
    parser.add_argument("--draft-kv-gib", type=float, default=6)
    parser.add_argument("--seed", type=int, default=17011)
    parser.add_argument("--panel", default="benchmarks/inputs/panel-48.json")
    parser.add_argument("--commit", required=True, help="git commit of the synced tree (the sync step refuses a dirty tree)")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def offline(args: argparse.Namespace) -> None:
    from nanovllm import LLM, SamplingParams

    kwargs = dict(max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs, enable_prefix_cache=False,
                  kv_cache_memory_bytes=int(args.target_kv_gib * GIB), seed=args.seed, record_step_timings=True)
    if args.k:
        kwargs.update(draft_model=args.draft, num_speculative_tokens=args.k,
                      draft_kv_cache_memory_bytes=int(args.draft_kv_gib * GIB))
    llm = LLM(args.target, **kwargs)
    panel = json.loads(Path(args.panel).read_text())
    prompts = [entry["prompt_token_ids"] for entry in panel]
    params = SamplingParams(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, max_tokens=args.max_tokens)

    llm.generate(prompts[:WARMUP_PROMPTS], params, use_tqdm=False)          # warm-up: excluded
    if llm.speculative is not None:
        llm.speculative.stats.clear()
    wall_start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    wall_seconds = time.perf_counter() - wall_start
    result = {
        "manifest": manifest(args.commit),
        "config": {"target": args.target, "draft": args.draft, "num_speculative_tokens": args.k,
                   "max_num_seqs": args.max_num_seqs, "temperature": args.temperature, "top_k": args.top_k,
                   "top_p": args.top_p, "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
                   "seed": args.seed, "panel": args.panel, "num_prompts": len(prompts)},
        "wall_seconds": wall_seconds,
        "step_timings": llm.step_timings(),
        "phase_timings": llm.phase_timings(),
        "outputs": [{"id": entry["id"], "token_ids": output["token_ids"], "first_token_step": output["first_token_step"],
                     "finish_step": output["finish_step"]} for entry, output in zip(panel, outputs)],
        "speculative_stats": dict(llm.speculative.stats) if llm.speculative is not None else None,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result))
    completion_tokens = sum(len(output["token_ids"]) for output in outputs)
    print(f"{args.output}: {completion_tokens} tokens in {wall_seconds:.1f}s wall")
    llm.exit()


def main(argv: list[str] | None = None) -> None:
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--mode", choices=["offline", "closed-loop"], default="offline")
    known, rest = mode_parser.parse_known_args(argv)
    if known.mode == "closed-loop":
        closed_loop(parse_closed_loop_args(rest))
    else:
        offline(parse_offline_args(rest))


if __name__ == "__main__":
    main()
