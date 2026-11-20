# vLLM serving: engineering report

> Template. Sections marked TODO are filled in after the first full run from `results/SUMMARY.md` and `plots/`. No numbers here until they come from a real run. Every "Why" below is a **hypothesis to confirm against results**, not a finding.

## Setup

- Model: `Qwen/Qwen2.5-1.5B-Instruct` (bf16). Smoke test: `Qwen/Qwen2.5-0.5B-Instruct`.
- Engine: vLLM 0.34.0, V1 engine (continuous batching always on), served via `vllm serve`, driven by `vllm bench serve`.
- GPU: TODO (L4 / A100-40GB), Modal. Price used: TODO ($/hr, checked on TODO).
- Workload default: 512 input / 128 output tokens, random dataset, seed 42, `--ignore-eos`, 10 discarded warm-up requests.
- SLO: p95 TTFT <= 500 ms, p95 TPOT <= 50 ms, zero failed requests.
- Run id(s): TODO.

## 01 Baseline latency

**What we measured:** Single-user (concurrency 1) TTFT, TPOT, ITL and end-to-end latency on the baseline config.

**Result (TODO):** TODO

**Why (mechanism).** *Hypothesis - confirm against results:* with one request in flight the GPU is lightly loaded, so this is the latency floor. TTFT is dominated by prefill, which is compute-bound and scales with input length. TPOT is dominated by decode, which is memory-bandwidth-bound: every generated token re-reads all model weights (and the KV cache) from HBM.

**Interview takeaway:** TODO

## 02 Concurrency sweep

**What we measured:** Output tokens/s, requests/s, TTFT and TPOT (p50/p95) at `--max-concurrency` 1, 4, 8, 16, 32, 64.

**Result (TODO):** TODO (throughput curve, where latency crosses the SLO).

**Why (mechanism).** *Hypothesis - confirm against results:* batching amortizes the weight reads: one pass over the weights serves every sequence in the batch, so throughput rises almost linearly while decode is memory-bound. Per-token latency grows slowly at first, then faster as the step becomes compute-bound and as prefills of new requests interleave with decode steps. Past saturation, extra concurrency only adds queueing (TTFT up, throughput flat). KV cache capacity can cap the useful batch earlier.

**Interview takeaway:** TODO

## 03 Prefill vs decode

**What we measured:** TTFT and end-to-end latency for input {128, 1024, 4096} x output {64, 512} at concurrency 8.

**Result (TODO):** TODO

**Why (mechanism).** *Hypothesis - confirm against results:* prefill is compute-bound and its cost scales with input length, so TTFT should grow with input length and be roughly independent of output length. Decode is memory-bandwidth-bound per token, so end-to-end latency should grow roughly linearly with output length (about output_len x TPOT) and TPOT should be mostly flat in input length until the KV cache read becomes noticeable.

**Interview takeaway:** TODO

## 04a max_num_seqs (batching width)

**What we measured:** `--max-num-seqs` 8 / 32 / 256 at concurrency 64.

**Result (TODO):** TODO

**Why (mechanism).** *Hypothesis - confirm against results:* a small cap limits how many sequences share each weight read, so throughput drops and the excess requests queue (TTFT rises sharply). TPOT per request may look better at a small cap because each step is lighter. V1 batches continuously regardless; this experiment shows the cost of limiting the batch width.

**Interview takeaway:** TODO

## 04b max_num_batched_tokens (chunked prefill budget)

**What we measured:** `--max-num-batched-tokens` 512 / 2048 / 8192 with 4096-token inputs at concurrency 16.

**Result (TODO):** TODO

**Why (mechanism).** *Hypothesis - confirm against results:* the budget bounds how many tokens a scheduler step processes. A small budget splits long prompts into more chunks, so TTFT for long prompts rises but decode steps are interrupted less (smoother ITL/TPOT tail). A large budget prefills fast (lower TTFT) but a long prefill chunk delays co-scheduled decode tokens (worse ITL tail).

**Interview takeaway:** TODO

## 04c Prefix caching

**What we measured:** prefix caching on vs off on a workload with a shared 1024-token prefix + 128-token unique suffix at concurrency 16; prefix cache hit rate.

**Result (TODO):** TODO

**Why (mechanism).** *Hypothesis - confirm against results:* with caching on, requests that share a prefix reuse its KV blocks instead of recomputing them, so prefill work drops to roughly the unique suffix: TTFT falls and throughput rises, and the hit rate should approach the shared fraction (about 1024/1152). With caching off every request pays the full prefill. Note the warm-up requests also populate the cache.

**Interview takeaway:** TODO

## 04d KV cache capacity

**What we measured:** a fixed 2 GiB KV cache (`--kv-cache-memory-bytes`, ~75k tokens) vs the default (~90% of GPU memory) at concurrency 64 with 1024-token inputs and 512-token outputs (~98k tokens of demand); KV cache usage and preemption count. A byte budget keeps the experiment GPU-independent, unlike `--gpu-memory-utilization`.

**Result (TODO):** TODO (check whether preemptions actually occurred; see Limitations).

**Why (mechanism).** *Hypothesis - confirm against results:* the KV cache is whatever memory is left after weights and activations. A smaller budget means fewer cache blocks and lower max concurrency. When the cache fills, the scheduler preempts running sequences (their KV is dropped and recomputed later), which shows up as preemptions, TTFT/TPOT spikes and lower throughput.

**Interview takeaway:** TODO

## 05 Cost / performance

**What we measured:** (analysis only) $/1M output tokens vs p95 TTFT across 02 and 04 runs, filtered by the SLO.

**Result (TODO):** TODO (best config under SLO, its concurrency, throughput and $/1M tokens).

**Why (mechanism).** *Hypothesis - confirm against results:* cost per token falls as batch size rises (fixed GPU price, higher throughput) until the SLO binds; the cheapest SLO-compliant point sits just before latency crosses the limit. A pricier GPU wins only if its throughput gain exceeds its price ratio.

**Interview takeaway:** TODO

## Bottlenecks

TODO - which resource limited each regime (compute in prefill, memory bandwidth in decode, KV cache capacity, scheduler limits), backed by GPU utilization / KV usage / waiting-queue telemetry.

## Trade-offs

TODO - latency vs throughput vs cost; chunk size vs TTFT/ITL; cache memory vs weights/headroom.

## Limitations

- Single model, single GPU, one seed; random token prompts (not natural text), so no real-world prefix structure except in 04c.
- Cost counts GPU time only (Modal bills CPU and memory separately), at the price checked on the date above; container cold start is excluded.
- 04d sizes demand (~98k tokens) above a 2 GiB budget (~75k tokens, from ~28 KiB/token for this model). If no preemptions show up, report that and check the per-token KV estimate.
- Telemetry sampling (0.5 s) includes the warm-up window; counters are deltas over the run.
- Results depend on vLLM 0.34.0 defaults and kernels; do not extrapolate to other versions or models.
