# tools

Measurement, gates and analysis for bench-v2 (`benchmarks/protocol-v2.json`). `v1/` keeps the scripts of the September release, which its documents still use.

| Script | What it does |
|---|---|
| `summarize_bench.py` | Metrics of one `bench_spec.py` result: Q_steady, Q_full, TTFT, ITL, round time, tokens per round, MBU |
| `report_v2.py` | The protocol's verdicts over a set of results: timing noise, R against k0, line against line, crossover, invalid runs |
| `stall_stats.py` | How much of the steady window prefill steps take, and the throughput they cost |
| `itl_stats.py` | Inter-token latency, split by whether a prefill step fell inside the interval |
| `request_stats.py` | What a request sees: TTFT, gaps between deliveries and round time (p50, p99), tokens per round by prompt category and over requests, and the two figures of `docs/RESULTS.md` |
| `plot_speedup.py` | The README figure: speedup over plain decoding by concurrency, median and range over seeds |
| `gain_factors.py` | One line's paired gain over another, split into tokens per round, round time, prefill share and the rest |
| `bench_vllm.py` | The vLLM baselines on the same panel and GPU |
| `fetch_panels.py` | Downloads the D48 development panel |
| `build_panel_k128.py` | Builds the confirmation panels (K128, then K128b) from SPEED-Bench `throughput_1k` |
| `gate_verify_logits.py` | G1-wide: verification logits against step-by-step decoding, judged against the decode noise floor |
| `gate_sampler_distribution.py` | G2-real: acceptance rate and output distribution on a real target/draft pair |
| `gate_task_accuracy.py` | G5: GSM8K accuracy of a line against k0 |
| `gate_dflash.py` | Block-draft (DFlash, DSpark) port checks against DeepSpec: block logits and DSpark's Markov-corrected rows, tokens-per-round alignment, CUDA graphs against eager |
| `ref_dflash_deepspec.py` | Runs DeepSpec's own DFlash or DSpark inference for those checks |
| `v1/report_matrix.py` | Tables of the September release's `docs/v1/RESULTS.md` |
| `v1/decompose_timings.py` | Per-round and per-step cost decomposition of archived September timings |
| `v1/plot_results.py` | The README figures of the September release |
| `v1/gate_smoke.py` | Smoke test: the speculative engine and k = 0 both generate |
| `v1/gate_greedy_equivalence.py` | G3: greedy k = 0 against greedy speculation |
| `v1/gate_divergence_probe.py` | Decode-path against verify-path logits at G3's divergence points |
