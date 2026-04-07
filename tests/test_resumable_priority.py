from __future__ import annotations

import hashlib
import sys
import types
from collections import deque
from itertools import count
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


class _FakeArray:

    def __init__(self, values):
        self.values = tuple(int(v) for v in values)

    def tobytes(self):
        return b",".join(str(v).encode("ascii") for v in self.values)


class _FakeXXHash64:

    def __init__(self):
        self._buffer = bytearray()

    def update(self, data):
        self._buffer.extend(data)

    def intdigest(self):
        digest = hashlib.blake2b(bytes(self._buffer), digest_size=8).digest()
        return int.from_bytes(digest, "little")


def _load_module(name: str, relative_path: str):
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / relative_path)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    source = "from __future__ import annotations\n" + source
    code = compile(source, module.__file__, "exec")
    exec(code, module.__dict__)
    return module


def _ensure_test_modules_loaded():
    if "numpy" not in sys.modules:
        numpy_mod = types.ModuleType("numpy")
        numpy_mod.array = lambda values: _FakeArray(values)
        sys.modules["numpy"] = numpy_mod

    if "xxhash" not in sys.modules:
        xxhash_mod = types.ModuleType("xxhash")
        xxhash_mod.xxh64 = _FakeXXHash64
        sys.modules["xxhash"] = xxhash_mod

    if "nanovllm" not in sys.modules:
        nanovllm_pkg = types.ModuleType("nanovllm")
        nanovllm_pkg.__path__ = [str(ROOT / "nanovllm")]
        sys.modules["nanovllm"] = nanovllm_pkg

    if "nanovllm.engine" not in sys.modules:
        engine_pkg = types.ModuleType("nanovllm.engine")
        engine_pkg.__path__ = [str(ROOT / "nanovllm" / "engine")]
        sys.modules["nanovllm.engine"] = engine_pkg

    if "nanovllm.config" not in sys.modules:
        config_mod = types.ModuleType("nanovllm.config")
        config_mod.Config = object
        sys.modules["nanovllm.config"] = config_mod

    sampling_params_mod = _load_module("nanovllm.sampling_params", "nanovllm/sampling_params.py")
    sequence_mod = _load_module("nanovllm.engine.sequence", "nanovllm/engine/sequence.py")
    _load_module("nanovllm.engine.block_manager", "nanovllm/engine/block_manager.py")
    scheduler_mod = _load_module("nanovllm.engine.scheduler", "nanovllm/engine/scheduler.py")
    return sampling_params_mod, sequence_mod, scheduler_mod


SamplingParams, Sequence, Scheduler = None, None, None
_sampling_params_mod, _sequence_mod, _scheduler_mod = _ensure_test_modules_loaded()
SamplingParams = _sampling_params_mod.SamplingParams
Sequence = _sequence_mod.Sequence
Scheduler = _scheduler_mod.Scheduler


def reset_sequence_state():
    Sequence.block_size = 4
    Sequence.counter = count()


def make_scheduler(enable_resumable_priority: bool, **overrides) -> Scheduler:
    config = SimpleNamespace(
        max_num_seqs=1,
        max_num_batched_tokens=8,
        enable_continuous_batching=True,
        enable_chunked_prefill=True,
        chunked_prefill_size=8,
        enable_cb_prefill_liveness=False,
        cb_prefill_reserve_ratio=0.0,
        cb_prefill_min_tokens=0,
        cb_prefill_min_seqs=0,
        enable_resumable_priority=enable_resumable_priority,
        resumable_priority_cached_tokens_weight=1.0,
        resumable_priority_remaining_prefill_tokens_weight=1.0,
        resumable_priority_waiting_time_weight=1.0,
        resumable_priority_preempt_count_weight=1.0,
        eos=-1,
        num_kvcache_blocks=4,
        kvcache_block_size=4,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return Scheduler(config)


def test_resumable_priority_prefers_high_score_recovery_seq():
    reset_sequence_state()

    def build_case(enable_resumable_priority: bool):
        scheduler = make_scheduler(
            enable_resumable_priority,
            max_num_batched_tokens=2,
            chunked_prefill_size=2,
            num_kvcache_blocks=6,
        )
        recovered = Sequence([1] * 8, SamplingParams(max_tokens=4, ignore_eos=True))
        scheduler.add(recovered)
        batch = scheduler.schedule()
        scheduler.postprocess(batch, [7])

        recovered.preempt_count = 4
        newcomer = Sequence([2, 2], SamplingParams(max_tokens=4, ignore_eos=True))
        scheduler.add(newcomer)
        scheduler.waiting = deque([newcomer, recovered])
        return scheduler, newcomer, recovered

    scheduler_off, newcomer_off, _ = build_case(False)
    batch_off = scheduler_off.schedule()
    assert batch_off.items[0].seq.seq_id == newcomer_off.seq_id

    scheduler_on, _, recovered_on = build_case(True)
    batch_on = scheduler_on.schedule()
    assert batch_on.items[0].seq.seq_id == recovered_on.seq_id


def test_resume_metrics_track_prefix_hits_and_recompute():
    reset_sequence_state()
    scheduler = make_scheduler(False)
    seq = Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=4, ignore_eos=True))
    scheduler.add(seq)

    first_batch = scheduler.schedule()
    scheduler.postprocess(first_batch, [6])
    scheduler.preempt(seq)

    resumed_batch = scheduler.schedule()
    resumed_item = resumed_batch.items[0]
    assert resumed_item.prefix_cache_hit_tokens == 4
    assert resumed_item.recomputed_prefill_tokens == 1


def test_resumable_priority_reduces_recomputed_prefill_after_cache_eviction():
    reset_sequence_state()

    def run_scenario(enable_resumable_priority: bool) -> tuple[int, int]:
        scheduler = make_scheduler(enable_resumable_priority, num_kvcache_blocks=3)
        victim = Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=8, ignore_eos=True))
        scheduler.add(victim)

        first_batch = scheduler.schedule()
        scheduler.postprocess(first_batch, [6])
        scheduler.preempt(victim)

        evictors = []
        for base in (11, 21, 31):
            evictor = Sequence(
                [base, base + 1, base + 2, base + 3],
                SamplingParams(max_tokens=4, ignore_eos=False),
            )
            scheduler.add(evictor)
            evictors.append(evictor)
        scheduler.waiting = deque([*evictors, victim])

        total_recomputed_prefill_tokens = 0
        total_prefix_cache_hit_tokens = 0
        while True:
            batch = scheduler.schedule()
            item = batch.items[0]
            total_recomputed_prefill_tokens += item.recomputed_prefill_tokens
            total_prefix_cache_hit_tokens += item.prefix_cache_hit_tokens
            token_id = -1 if item.seq in evictors else 7
            scheduler.postprocess(batch, [token_id])
            if item.seq.seq_id == victim.seq_id:
                return total_recomputed_prefill_tokens, total_prefix_cache_hit_tokens

    off_recomputed, off_cache_hits = run_scenario(False)
    on_recomputed, on_cache_hits = run_scenario(True)

    assert off_recomputed == 5
    assert on_recomputed == 1
    assert off_recomputed > on_recomputed
    assert off_cache_hits == 0
    assert on_cache_hits == 4
