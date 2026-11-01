# Using speculative decoding

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3-8B",
    draft_model="/path/to/Qwen3-0.6B",
    num_speculative_tokens=3,
    enable_prefix_cache=False,           # required with a draft model
    kv_cache_memory_bytes=10 << 30,      # explicit budgets: two models share the GPU
    draft_kv_cache_memory_bytes=6 << 30,
    max_num_seqs=8,
    seed=0,                              # optional: reproducible sampling
)
params = SamplingParams(temperature=0.7, top_k=20, top_p=0.8, max_tokens=256)
try:
    outputs = llm.generate(["Explain KV caching."], params)
    print(outputs[0]["text"])
finally:
    llm.exit()
```

For ordinary decoding leave out both `draft_model` and `num_speculative_tokens` (a `k > 0` without a draft is rejected); everything else is unchanged. Greedy
decoding is `temperature=0`. `top_p` requires `top_k` (the nucleus is computed over the
top-k candidates).

## A DFlash draft

```python
llm = LLM(
    "/path/to/Qwen3-8B",
    draft_model="/path/to/dflash_qwen3_8b_block7",   # deepseek-ai/dflash_qwen3_8b_block7
    draft_type="dflash",
    enable_prefix_cache=False,
    kv_cache_memory_bytes=10 << 30,
    max_num_seqs=8,
)
```

`num_speculative_tokens` defaults to the draft's block size (7) and cannot differ from it,
and there is no `draft_kv_cache_memory_bytes`: the draft keeps its context K/V in the
target's KV blocks. A round runs the draft once on a block of the last committed token
plus 6 mask tokens (non-causal attention over context K/V projected from five target
layers), verifies the 7 drafts as below, writes the context K/V of all 8 verified
positions, and accepts with the same rule. Round-entry invariant: the draft's context
covers `len-1` positions, like the target's KV.

## A DSpark draft

```python
llm = LLM(
    "/path/to/Qwen3-8B",
    draft_model="/path/to/dspark_qwen3_8b_block7",   # deepseek-ai/dspark_qwen3_8b_block7
    draft_type="dspark",
    enable_prefix_cache=False,
    kv_cache_memory_bytes=10 << 30,
    max_num_seqs=8,
)
```

The block, its KV and the verification are DFlash's. `draft_type` must match the
checkpoint: a DSpark checkpoint has a Markov head, a DFlash one does not. The 7 drafts are
drawn left to right: row j's logits get the Markov bias of the token drawn at row j−1
(the last committed token for row 0), and the acceptance rule gets each row's distribution
after its bias. The checkpoint's confidence head is loaded but not used, so every round
verifies all 7 drafts. When every request truncates with top-k, each row is drawn among its
top `max_top_k + 64` logits, which matches the whole-vocabulary rule unless more than 64
tokens tie at the cut value. Sampling graphs are captured for the first 4 sampling settings
(top-k width, top-p on or off, whether every request truncates); a batch with any other
setting samples eagerly, with the same drafts, about 0.4–0.5 ms more per round at C = 1.

## What a round does

For every sequence in a decode batch: the draft recomputes at most one missing KV entry
(one decode-graph step), proposes `k` tokens (one decode-graph step each), the target
scores `[last token, d1..dk]` in one forward through the paged varlen-attention path (a
CUDA graph per padded batch size), rejection sampling accepts a prefix and draws one more
token, and a single `[B, k+2]` copy brings the result to the host. Round-entry invariants:
target KV covers `len-1` positions, draft KV covers `len-2` or `len-1`. The design is in
`docs/design/batched-speculative-decoding.md`.

## Reproducibility

One `torch.Generator` per engine, seeded from `seed`. A run is reproducible given the
seed and the same batch composition; which requests share a batch changes the draws.
The measured greedy outputs differ across batch sizes and between speculative and
ordinary decoding. Most first divergences occur at small top-2 logit margins; the
per-divergence results and identical-KV probe are in `docs/v1/RESULTS.md`, gate G3 and its control.

## Limits (v1)

TP = 1; prefix cache off with a draft; no n-gram proposals; a batch is all-greedy or
all-random; `prompt + max_tokens + k <= max_model_len`.

## Timing

`record_step_timings=True` records a pair of CUDA events per engine step and one event
per phase of every speculative round; `llm.step_timings()` and `llm.phase_timings()`
return them after `generate()`. `bench_spec.py`, `tools/summarize_bench.py` and
`tools/v1/report_matrix.py` build throughput, per-round costs and the break-even check
from them.

## Gates

`tools/v1/gate_smoke.py`, `tools/gate_verify_logits.py` (G1: verification logits vs decode
logits), `tools/gate_sampler_distribution.py` (G2: sampler distribution),
`tools/v1/gate_greedy_equivalence.py` (G3: greedy match rate with per-divergence logit
margins) and `tools/v1/gate_divergence_probe.py` (decode vs verify logits on identical KV
at G3 divergence points) run on a CUDA machine with both models.
