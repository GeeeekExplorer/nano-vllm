# Queue-aware prefix cache eviction

Free cached blocks may be reused until their physical storage is allocated again.
The original FIFO free queue does not consider whether waiting requests will soon
reuse those prefixes. Under cache pressure, a new allocation can evict the prefix
of the next request and cause avoidable prefill computation.

`prefix_cache_lookahead=16` enables a soft eviction preference; the default `0`
preserves FIFO selection. The window counts waiting queue entries, including the
allocating request when it is still at the head. Lookup skips that request and
requests with existing block tables. It follows the same chained hashes and token
checks as admission, stops at the first miss, and excludes the final block as
the existing cache-reuse path does.

Lookup is lazy: an allocation starting with uncached blocks does not inspect the
queue until it reaches a cached FIFO candidate. Matching IDs are collected once
per `allocate` or `may_append` call. This read-only query does not change reference
counts, cache metadata or request tables. The allocator chooses the first free
ID outside the retained set, preserving the relative order of remaining IDs.
If all candidates are retained, it uses the original FIFO head. Blocks are never
pinned, admission order is unchanged, and the set is local to the allocation.
The existing path still invalidates overwritten hash mappings and updates refs.

The window bounds request count, not total prefix tokens. Hashing long prefixes
adds CPU work. Selecting an alternative scans/removes from a deque; repeated
selection can be quadratic in free-queue length in the worst case, including
all-retained fallback. This implementation favors a small opt-in change over a
new cache index. Empty or unique-prefix workloads should not be expected to gain.

## Reproduce

Install nano-vLLM's normal dependencies and `pytest`, then run from this checkout:

```bash
python -m pytest tests/test_queue_eviction.py -q
python scripts/check_queue_eviction.py --model /path/to/Qwen3-0.6B --output /tmp/queue-check.json
for scenario in pressure ample unique; do
  python scripts/bench_queue_eviction.py --model /path/to/Qwen3-0.6B \
    --scenario "$scenario" --output "/tmp/queue-$scenario.json"
done
```

Use new output paths on subsequent runs. The benchmark temporarily replaces only
the profiling-derived block count in `ModelRunner.allocate_kv_cache` before model
initialization, so both policies have the same **physical** KV tensor capacity.
This override is benchmark-only; no capacity override is added to the public API.

Each scenario uses one loaded model, an initial full-trace warmup for each policy,
then five alternating-order pairs with fresh scheduler/cache metadata each run.
Baseline is this implementation with lookahead `0`; candidate uses `16`.
There are 64 requests in eight waves, 4096-token family prefixes plus 64 unique
suffix tokens, and 32 output tokens per request. Pressure and unique scenarios
use 68 blocks; ample uses 160. Repeated families use a fixed skewed distribution
and seed 20260919; unique uses a distinct first block per request.

Throughput includes scheduling, model execution and postprocessing but excludes
model startup, tokenization and warmup. TTFT is measured inside the offline engine
from wave arrival, not through an HTTP server. `scheduler_seconds` covers
`schedule()` only; hashing in `postprocess()` is included in throughput but not
that CPU metric. The reported cache-hit fraction is inferred from scheduled
prefill tokens; with preemption/recomputation (e.g. much longer outputs), interpret
that quantity as avoided prefill work rather than a literal cache-hit rate.

The separate correctness diagnostic poisons KV storage with NaNs between arms,
primes four prefix families, and compares full-vocabulary logits for four requests
with 16 identical teacher-forced continuation tokens each. It checks finite
values and requires bitwise agreement with a disabled-policy control that has
enough logical cache capacity to preserve the same prefixes. Differences from
the memory-constrained disabled-policy baseline are reported separately because
different prefill/cache-hit shapes can change floating-point results. This is a
targeted single-model check, not exhaustive numerical or model-quality coverage.

## Local results

RTX 5080 (16 GB), Qwen3-0.6B BF16, TP=1, CUDA graphs enabled, WSL Ubuntu 24.04,
PyTorch 2.11.0+cu128, FlashAttention 2.8.3.post1, Triton 3.6.0. Base commit:
`bb823b3e06983d71485a8e1f23715ebd87d98ef8`. Arithmetic means of five measured runs
per policy after warmup (same fixed trace in each pair):

| Scenario | FIFO output tokens/s | Queue output tokens/s | Change | FIFO / queue prefill tokens |
|---|---:|---:|---:|---:|
| Repeated prefixes, 68 blocks | 394.05 | 442.40 | +12.27% | 140032 / 107776 |
| Repeated prefixes, 160 blocks | 814.97 | 813.37 | -0.20% | 36864 / 36864 |
| Unique prefixes, 68 blocks | 265.81 | 265.85 | +0.01% | 266240 / 266240 |

In the pressure case, scheduled prefill tokens decreased 23.03%; inferred prefix
hit fraction increased from 47.40% to 59.52%. Mean per-run p95 engine TTFT moved
from 571.16 to 556.03 ms (-2.65%). Time in `schedule()` increased from 15.48 to
33.97 ms per 64-request run. The tradeoff is more scheduler work for less prefill.

All 14 CPU tests passed, including a 1000-operation randomized accounting test.
The targeted GPU diagnostic passed: queue vs matching-cache-hit control was
bitwise identical across 64 full-vocabulary logit rows. Against pressure FIFO,
max absolute logit difference was 0.21875 and max relative L2 was 0.02958;
argmax agreed on all 64 rows. The same differences occurred between the two FIFO
controls, so bitwise equality to the pressure FIFO arm is not claimed.
An additional pressure run (16 requests, 256 output tokens each, one pair plus
warmups) completed for both policies, crossing decode block boundaries; all
requests reached the requested length and all block references were released.

These are synthetic results from one model, GPU and seed, not a general throughput
guarantee. No multi-GPU, large-cache scaling, all-retained performance benchmark,
or production serving validation has been performed. The sub-percent control
differences should be treated as noise, not evidence of a speedup or regression.
