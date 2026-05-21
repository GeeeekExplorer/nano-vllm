# Nano-vLLM Optimization Roadmap

This is an engineering implementation roadmap for optimizing Nano-vLLM without losing the main value of the project: a small, readable vLLM-style inference engine. Each phase should leave the project runnable, measurable, and easier to compare against the previous phase.

## Ground Rules

- Optimize only after recording a baseline.
- Keep correctness checks beside performance checks.
- Make one optimization at a time and measure it in isolation.
- Prefer changes that preserve the current architecture: `LLMEngine -> Scheduler -> ModelRunner -> Qwen3/layers`.
- Do not hide regressions behind benchmark noise. Run each benchmark multiple times and compare medians.
- Keep the default path simple. Put experimental behavior behind explicit config flags.

## Baseline Commands

Use these commands as the starting point once the environment is working on Linux or WSL2 with an NVIDIA GPU:

```bash
python example.py
python bench.py
```

Recommended benchmark variants:

```bash
python bench.py
```

Then edit `bench.py` locally for controlled experiments:

- small batch: `num_seqs = 16`
- medium batch: `num_seqs = 64`
- large batch: `num_seqs = 256`
- short prompts: input/output length `32..128`
- long prompts: input/output length `512..1024`
- eager mode: `enforce_eager=True`
- CUDA graph mode: `enforce_eager=False`

Track at least:

- total output tokens
- wall time
- output tokens per second
- prefill throughput
- decode throughput
- peak GPU memory
- number of preemptions
- prefix-cache hit rate

## Phase 1: Establish Reproducible Benchmarks

**Target files:**

- `bench.py`
- optional new file: `benchmarks/benchmark_matrix.py`
- optional new file: `benchmarks/README.md`

**Goal:** make performance comparable before and after every change.

**Steps:**

1. Add CLI arguments to `bench.py` for `num_seqs`, `max_input_len`, `max_output_len`, `model`, `enforce_eager`, `seed`, and `tensor_parallel_size`.
2. Print model path, GPU name, dtype, tensor parallel size, eager mode, sequence count, input length range, and output length range before each run.
3. Run one warmup request before timing, as the current benchmark already does.
4. Report median throughput across `N` repeated timed runs.
5. Add a benchmark matrix script that runs a small fixed set of scenarios.
6. Save benchmark results as JSON lines so changes can be compared over time.

**Success signal:**

- Two consecutive benchmark runs with the same settings produce similar throughput.
- Benchmark output includes enough metadata to explain performance differences.

**Regression checks:**

- `python example.py` still produces text.
- `python bench.py --num-seqs 16 --max-input-len 128 --max-output-len 128` completes.

## Phase 2: Add Correctness Smoke Tests

**Target files:**

- optional new file: `tests/test_sequence.py`
- optional new file: `tests/test_block_manager.py`
- optional new file: `tests/test_scheduler.py`
- optional new file: `tests/test_sampling_params.py`
- `pyproject.toml`

**Goal:** protect scheduling and KV-cache behavior before optimizing.

**Steps:**

1. Add `pytest` as a development dependency or document how to install it.
2. Test `Sequence` initialization, token append, prompt/completion slicing, and block splitting.
3. Test `BlockManager` allocation, deallocation, ref counts, and prefix-cache hash reuse.
4. Test scheduler transitions from `WAITING` to `RUNNING` to `FINISHED`.
5. Test chunked prefill when a sequence exceeds `max_num_batched_tokens`.
6. Test `SamplingParams` rejects greedy temperature values as it does today.

**Success signal:**

- CPU-only unit tests cover the pure Python scheduling and block-management path.
- Optimizations can be made without requiring a GPU for every correctness check.

**Regression checks:**

```bash
pytest tests/test_sequence.py tests/test_block_manager.py tests/test_scheduler.py -q
```

## Phase 3: Add Runtime Instrumentation

**Target files:**

- `nanovllm/engine/llm_engine.py`
- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/model_runner.py`
- optional new file: `nanovllm/engine/stats.py`

**Goal:** expose where time and memory go before changing core behavior.

**Steps:**

1. Add an internal stats object with counters for prefill steps, decode steps, scheduled tokens, preemptions, allocated blocks, freed blocks, prefix-cache hits, and prefix-cache misses.
2. Time scheduler execution separately from model execution in `LLMEngine.step`.
3. Time `prepare_prefill`, `prepare_decode`, `run_model`, and sampling inside `ModelRunner.run`.
4. Track peak memory after warmup, after KV-cache allocation, and after generation.
5. Print stats only when an explicit flag is enabled, for example `enable_stats=True`.
6. Add benchmark output fields for the new stats.

**Success signal:**

- A benchmark run explains whether the bottleneck is scheduling, tensor preparation, model execution, sampling, or memory pressure.

**Regression checks:**

- Stats disabled keeps current user-facing output nearly unchanged.
- Stats enabled does not change generated token IDs for a fixed seed and temperature behavior.

## Phase 4: Optimize Scheduler Behavior

**Target files:**

- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/sequence.py`
- `tests/test_scheduler.py`

**Goal:** improve batching efficiency and reduce unnecessary preemption.

**Steps:**

1. Measure average number of sequences per prefill batch and decode batch.
2. Measure how often chunked prefill schedules only one long request while other requests wait.
3. Add tests for mixed prompt lengths where one long prefill should not starve many short requests.
4. Consider a configurable chunked prefill limit separate from `max_num_batched_tokens`.
5. Prefer scheduling more short prefills when doing so improves batch occupancy without causing KV-cache pressure.
6. Track preemptions and verify they decrease or remain justified under memory pressure.

**Success signal:**

- Better throughput for mixed prompt-length workloads.
- No increase in failed allocations or pathological preemption loops.

**Regression checks:**

- Existing prefill/decode ordering remains correct.
- Finished outputs are still returned sorted by original sequence ID.

## Phase 5: Optimize KV-Cache Block Management

**Target files:**

- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/sequence.py`
- `tests/test_block_manager.py`

**Goal:** reduce CPU overhead in allocation, deallocation, and prefix-cache lookup.

**Steps:**

1. Measure time spent in `can_allocate`, `allocate`, `deallocate`, `can_append`, and `hash_blocks`.
2. Add tests for prefix-cache reuse across prompts with shared full blocks.
3. Avoid repeated list slicing in `Sequence.block` if profiling shows it is material.
4. Cache per-block token hashes on the sequence where safe.
5. Review `free_block_ids.remove(block_id)` in prefix-cache reuse because deque removal is linear.
6. Consider maintaining an auxiliary free-block set for O(1) membership checks.
7. Keep ref-count semantics explicit and tested before changing allocation policy.

**Success signal:**

- Lower scheduler-side CPU time for workloads with many sequences.
- Prefix-cache hit rate is measurable and stable.

**Regression checks:**

- Blocks are never freed while still referenced.
- Hash collisions remain guarded by token equality checks.
- Prefix-cache reuse never changes generated outputs.

## Phase 6: Optimize Tensor Preparation

**Target files:**

- `nanovllm/engine/model_runner.py`

**Goal:** reduce Python overhead and host-to-device transfer cost in `prepare_prefill` and `prepare_decode`.

**Steps:**

1. Profile `prepare_prefill` and `prepare_decode` separately.
2. Replace repeated Python list extension hot paths only when profiling confirms they matter.
3. Reuse pinned CPU buffers for common decode batch sizes.
4. Reuse GPU tensors for decode metadata where shapes are stable.
5. Avoid reallocating `block_tables` when the maximum table width has not changed.
6. Keep fallback allocation paths for unusual batch sizes.

**Success signal:**

- Decode throughput improves for small and medium batches.
- Tensor preparation becomes a smaller percentage of total step time.

**Regression checks:**

- `slot_mapping`, `context_lens`, and `block_tables` match the old implementation for representative sequences.
- CUDA graph replay receives correctly padded metadata.

## Phase 7: Improve CUDA Graph Coverage

**Target files:**

- `nanovllm/engine/model_runner.py`

**Goal:** increase decode steps that can use captured CUDA graphs.

**Steps:**

1. Record how often decode batch sizes hit existing graph buckets.
2. Review the current graph sizes: `[1, 2, 4, 8] + range(16, max_bs + 1, 16)`.
3. Add graph buckets only if real workloads frequently round up too far.
4. Measure memory cost of additional graph captures.
5. Keep eager fallback for prefill and large decode batches.
6. Test `enforce_eager=True` and `enforce_eager=False` in benchmarks.

**Success signal:**

- More decode steps use CUDA graph replay.
- Decode throughput improves without unacceptable graph memory growth.

**Regression checks:**

- CUDA graph outputs match eager outputs for the same cached state.
- Graph metadata reset logic still clears stale `slot_mapping`, `context_lens`, and `block_tables`.

## Phase 8: Optimize Attention and KV-Cache Writes

**Target files:**

- `nanovllm/layers/attention.py`
- `nanovllm/engine/model_runner.py`

**Goal:** reduce overhead around storing K/V and invoking FlashAttention.

**Steps:**

1. Profile the Triton `store_kvcache_kernel`.
2. Verify memory layout assumptions for `key`, `value`, `k_cache`, and `v_cache`.
3. Benchmark prefill with and without prefix-cache block tables.
4. Benchmark decode with different batch sizes and context lengths.
5. Consider fusing or specializing KV-cache writes only after profiling proves the kernel is material.
6. Keep assertions or debug checks for strides in development paths.

**Success signal:**

- Attention time decreases or remains stable while tensor preparation and scheduling improve.
- No new correctness failures under long-context decode.

**Regression checks:**

- Prefill with prefix cache still attends over cached and new tokens correctly.
- Decode still reads the right cached K/V blocks for each sequence.

## Phase 9: Optimize Logits and Sampling

**Target files:**

- `nanovllm/layers/embed_head.py`
- `nanovllm/layers/sampler.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/sampling_params.py`

**Goal:** reduce overhead after model forward, especially for large vocab logits.

**Steps:**

1. Measure time spent in `compute_logits` and `Sampler.forward`.
2. Confirm that prefill only computes logits for last prompt tokens, as `ParallelLMHead` currently does.
3. Add support for deterministic greedy sampling only if needed, because current `SamplingParams` explicitly rejects near-zero temperature.
4. Consider top-k or top-p only after the baseline sampler is measured; extra features can slow the simple path.
5. For tensor parallel mode, measure `dist.gather` cost in `ParallelLMHead`.
6. Keep rank-0-only sampling behavior clear.

**Success signal:**

- Sampling and logits become a smaller fraction of decode step time.
- Any new sampling option is guarded by tests and benchmarked separately.

**Regression checks:**

- Temperature sampling remains valid.
- Tensor-parallel logits are assembled in the correct vocab order.

## Phase 10: Optimize Tensor Parallel Communication

**Target files:**

- `nanovllm/layers/linear.py`
- `nanovllm/layers/embed_head.py`
- `nanovllm/engine/model_runner.py`

**Goal:** reduce distributed communication overhead when `tensor_parallel_size > 1`.

**Steps:**

1. Benchmark tensor parallel size 1 versus 2 or more on matching hardware.
2. Measure `all_reduce` cost in `RowParallelLinear`.
3. Measure `all_reduce` cost in `VocabParallelEmbedding`.
4. Measure `gather` cost in `ParallelLMHead`.
5. Avoid optimizing multi-GPU communication before single-GPU performance is stable.
6. Consider overlapping communication only if profiling shows communication dominates.

**Success signal:**

- Tensor parallel mode improves capacity or throughput for models that need it.
- Single-GPU performance remains unchanged.

**Regression checks:**

- Weight sharding still loads the correct shard for Q, K, V, MLP gate/up, embeddings, and LM head.
- Rank 0 and worker ranks exit cleanly.

## Phase 11: Improve Weight Loading

**Target files:**

- `nanovllm/utils/loader.py`
- `nanovllm/layers/linear.py`
- `nanovllm/layers/embed_head.py`

**Goal:** make startup faster and failures easier to debug.

**Steps:**

1. Time model loading separately from warmup and KV-cache allocation.
2. Print missing or unexpected weight names with enough context when loading fails.
3. Avoid repeated lookup work in packed-module mapping if profiling shows startup overhead matters.
4. Keep support for safetensors because the current loader assumes it.
5. Add a strict mode that reports unconsumed parameters and unconsumed checkpoint tensors.

**Success signal:**

- Startup failures are easier to diagnose.
- Load time is measured and does not regress accidentally.

**Regression checks:**

- Qwen3-0.6B loads successfully.
- Tied embeddings still share storage when `config.tie_word_embeddings` is true.

## Phase 12: Tune Memory Policy

**Target files:**

- `nanovllm/config.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/scheduler.py`

**Goal:** make memory use predictable and reduce avoidable preemption.

**Steps:**

1. Log total GPU memory, free memory, peak warmup memory, and computed KV-cache block count.
2. Benchmark different `gpu_memory_utilization` values.
3. Benchmark different `kvcache_block_size` values if FlashAttention constraints allow them.
4. Track preemptions per generated token.
5. Add a clear error message when no KV-cache blocks can be allocated.
6. Consider a minimum reserved-memory setting if CUDA graph capture or runtime allocations need headroom.

**Success signal:**

- Users can choose memory settings without trial-and-error failures.
- Preemption frequency decreases under realistic loads.

**Regression checks:**

- Long prompts still fail cleanly when they exceed available KV-cache capacity.
- Prefix caching still works with the chosen block size.

## Phase 13: Add End-to-End Regression Matrix

**Target files:**

- `bench.py`
- optional new file: `benchmarks/benchmark_matrix.py`
- optional new file: `benchmarks/results/`
- optional new file: `docs/benchmarking.md`

**Goal:** make performance regressions visible before merging changes.

**Steps:**

1. Define a small benchmark matrix that can run in a few minutes.
2. Include at least one short-prompt decode-heavy scenario.
3. Include at least one long-prompt prefill-heavy scenario.
4. Include at least one mixed-length scenario.
5. Include eager and CUDA graph modes.
6. Store results in a machine-readable format.
7. Document how to compare a new run against a baseline.

**Success signal:**

- Every optimization PR can include before/after numbers.
- Results identify whether gains came from prefill, decode, memory, or scheduling.

**Regression checks:**

- Benchmark scripts fail clearly when CUDA, model files, or dependencies are missing.
- Benchmark scripts do not silently compare different models or settings.

## Phase 14: Documentation and Configuration Cleanup

**Target files:**

- `README.md`
- `example.py`
- `bench.py`
- `pyproject.toml`
- optional new file: `docs/benchmarking.md`
- optional new file: `docs/optimization-notes.md`

**Goal:** make optimized behavior understandable and repeatable.

**Steps:**

1. Document supported platform assumptions: Linux/WSL2, NVIDIA CUDA, FlashAttention, Triton, NCCL.
2. Document benchmark commands and expected environment variables.
3. Document important config knobs: `max_num_batched_tokens`, `max_num_seqs`, `max_model_len`, `gpu_memory_utilization`, `tensor_parallel_size`, `enforce_eager`, and `kvcache_block_size`.
4. Add troubleshooting notes for `flash-attn` build isolation and CUDA toolkit issues.
5. Keep README quick-start short; put deeper benchmark details in docs.

**Success signal:**

- A new engineer can reproduce benchmark numbers from a fresh checkout.
- Users can distinguish install problems from runtime performance problems.

**Regression checks:**

- README examples still match the public API.
- Benchmark docs match actual CLI flags.

## Recommended Implementation Order

1. Phase 1: Establish reproducible benchmarks.
2. Phase 2: Add correctness smoke tests.
3. Phase 3: Add runtime instrumentation.
4. Phase 5: Optimize KV-cache block management.
5. Phase 6: Optimize tensor preparation.
6. Phase 4: Optimize scheduler behavior.
7. Phase 7: Improve CUDA graph coverage.
8. Phase 9: Optimize logits and sampling.
9. Phase 8: Optimize attention and KV-cache writes.
10. Phase 12: Tune memory policy.
11. Phase 10: Optimize tensor parallel communication.
12. Phase 11: Improve weight loading.
13. Phase 13: Add end-to-end regression matrix.
14. Phase 14: Documentation and configuration cleanup.

This order intentionally starts with measurement and pure Python correctness checks. The lower-level GPU work should come after the project can prove whether a change helped.

## Suggested Commit Strategy

Use small commits that each leave the repo runnable:

```bash
git commit -m "bench: add configurable benchmark arguments"
git commit -m "test: cover sequence and block manager behavior"
git commit -m "perf: add engine runtime stats"
git commit -m "perf: reduce decode tensor preparation overhead"
```

Avoid commits that mix benchmark changes, behavior changes, and kernel changes. Those are hard to review and harder to benchmark.

## Stop Conditions

Pause optimization and investigate before continuing if:

- correctness tests fail
- generated output becomes empty or malformed
- throughput changes by more than expected but profiling does not explain why
- GPU memory rises without a known reason
- preemption count increases sharply
- prefix-cache hit rate drops unexpectedly
- tensor parallel workers fail to exit cleanly

The right loop is always:

```text
measure -> change one thing -> verify correctness -> benchmark -> record result
```
