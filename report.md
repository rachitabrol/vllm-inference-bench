# vLLM serving: engineering report

## Summary

- Single-user decode runs at about 77% of the memory-bandwidth roofline (TPOT 13.4 ms against a 10.3 ms floor).
- Raising concurrency from 1 to 16 users increases throughput 11× while TPOT rises 22%. Cost falls from $3.03 to $0.27 per 1M output tokens.
- The latency SLO is first violated by p95 TTFT between 16 and 32 users, due to prefill queueing. KV-cache usage remains below 7%.
- Prefix caching reduces median TTFT by 70% and raises throughput by 54% on shared-prefix traffic.
- A KV cache too small for the workload causes preemptions, 12% lower throughput and a 2× worse p95 TTFT.
- Lowest cost within the SLO: concurrency 16 at $0.27 per 1M output tokens on an L4.

All figures are medians of 3 repeats, each repeat in its own GPU container.

- **Experiments 01–03:** run `44748004817-1-L4` (per-workload seeds; see the [methodology note](#methodology-note-prefix-cache-contamination-in-the-first-run)).
- **Experiments 04a–04d:** run `44708914259-1-L4`.

Combined outputs are in `results/summary_agg.csv`, `results/SUMMARY.md` and `plots/`. The first single-shot run (`44703863719-1-L4`) is retained in `results/raw/` for provenance only. Its 02 and 03 results are affected by prefix-cache contamination and are not used.

## Setup

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`, bf16 (about 3.1 GB of weights).
- **Engine:** vLLM 0.34.0, V1 engine (continuous batching always enabled), prefix caching enabled in the baseline. `vllm serve` is driven by `vllm bench serve`.
- **GPU:** NVIDIA L4 (24 GB, about 300 GB/s memory bandwidth) on Modal at $0.80/hr (price checked 2026-11-20).
- **Default workload:** 512 input / 128 output tokens, random-token prompts with a distinct seed per workload, `--ignore-eos`, `--temperature 0`, 10 warm-up requests.
- **SLO:** p95 TTFT ≤ 500 ms, p95 TPOT ≤ 50 ms, zero failed requests.
- **Reliability:** no failed requests in any measured run.
- **Repeatability:** throughput CV across all 23 points is at most 3.3% (median 0.7%). p95 TTFT CV is at most 10.2% (median 2.9%). Differences larger than about 5% are therefore outside run-to-run variation.

## 01 Baseline latency

**Measurement:** concurrency 1, 50 prompts.

**Result:** TTFT p50 38 ms and p95 43 ms. TPOT 13.4 ms (about 75 tokens/s for a single user). E2E p95 1.75 s for 128 output tokens.

**Explanation:**
- Decode is memory-bandwidth-bound. Each generated token reads all weights once, so the floor is 3.1 GB / 300 GB/s ≈ 10.3 ms. The measured 13.4 ms is about 77% of that roofline. The remainder is attributable to kernel launch overhead, attention over the KV cache and sampling.
- A 512-token prefill takes about 38 ms because all prompt tokens are processed in a single pass over the weights. Prefill is therefore compute-bound rather than bandwidth-bound.

**Conclusion:** at batch size 1 the GPU is limited by memory bandwidth. TPOT can be estimated from model size divided by bandwidth; the measurement was within about 30% of that floor.

## 02 Concurrency sweep

**Measurement:** the same workload at maximum concurrency 1, 4, 8, 16, 32 and 64 on one server, with distinct prompts at each step.

| Concurrency | Output tok/s | TTFT p95 (ms) | TPOT p50 (ms) | ITL p99 (ms) | $/1M output tok | SLO met |
|---|---|---|---|---|---|---|
| 1 | 73 | 40 | 13.4 | 15 | 3.03 | Yes |
| 4 | 269 | 150 | 13.9 | 17 | 0.83 | Yes |
| 8 | 496 | 254 | 14.9 | 37 | 0.45 | Yes |
| 16 | 833 | 427 | 16.4 | 99 | 0.27 | Yes |
| 32 | 1231 | 809 | 21.5 | 129 | 0.18 | No |
| 64 | 1804 | 1721 | 32.0 | 138 | 0.12 | No |

**Result:**
- From 1 to 16 users, throughput increases 11× while TPOT rises from 13.4 to 16.4 ms (+22%). From 1 to 64 users, throughput increases 25×.
- The SLO is violated between 16 and 32 users, on p95 TTFT (427 to 809 ms). p95 TPOT remains well below its 50 ms limit throughout.
- Memory is not the constraint. KV-cache usage peaks at 6.9% at concurrency 64.

**Explanation:**
- **Batching amortizes weight reads.** One pass over the weights produces a token for every sequence in the batch, so at small batch sizes additional users add little per-step cost.
- **Prefill becomes the bottleneck.** As concurrency rises, more new prompts arrive per second. Their prefills queue, which appears as TTFT, and they interleave with decode steps, which appears as ITL p99 spikes (15 ms at 1 user, roughly 100–140 ms at 16 and above) when a decode token waits behind another request's prefill chunk.
- **The tail degrades first.** Median TTFT changes little from 32 to 64 users (370 to 413 ms) while p95 doubles. Some requests land behind a burst of prefills.

**Conclusion:** concurrency is the most effective cost lever, reducing cost 11× from 1 to 16 users. The first constraint to bind is tail TTFT from prefill queueing, not average latency or memory.

## 03 Prefill and decode

**Measurement:** input {128, 1024, 4096} × output {64, 512} at concurrency 8.

| Input → output | TTFT p50 (ms) | TPOT p50 (ms) | ITL p95 (ms) | E2E p95 (s) |
|---|---|---|---|---|
| 128 → 64 | 59 | 14.0 | 15 | 0.95 |
| 1024 → 64 | 204 | 18.9 | 67 | 1.46 |
| 4096 → 64 | 853 | 37.7 | 146 | 3.37 |
| 128 → 512 | 67 | 14.0 | 14 | 7.24 |
| 1024 → 512 | 207 | 15.3 | 15 | 8.10 |
| 4096 → 512 | 855 | 20.0 | 18 | 11.23 |

**Result:**
- **Input length determines TTFT.** Output length has no effect: median TTFT is 59, 204 and 853 ms for 128, 1024 and 4096 input tokens, with either 64 or 512 outputs. 4096-token prompts exceed the 500 ms TTFT limit even at concurrency 8.
- **Output length determines E2E.** Going from 64 to 512 output tokens raises E2E by 3.3–7.6×. E2E ≈ TTFT + output length × TPOT; for example, 67 ms + 512 × 14 ms ≈ 7.2 s.
- **Long prompts slow concurrent decoding, mainly when outputs are short.** At 4096 → 64, TPOT is 37.7 ms with ITL p95 146 ms. At 4096 → 512, TPOT is 20.0 ms with ITL p95 18 ms.

**Explanation:**
- **Prefill.** Prefill work grows with prompt length (linearly for the MLP layers, quadratically for attention), so TTFT follows input length.
- **Decode.** Decode produces one token per sequence per step, so E2E follows output length.
- **The decode slowdown depends on the prompt-to-output ratio.** A short-output request finishes quickly and is replaced by a new one, which brings another 4096-token prefill. With 64-token outputs, prefills account for a large share of GPU steps and decode tokens frequently wait behind them. With 512-token outputs the same prefills are spread over 8× more decode steps, which dilutes the interference.

**Conclusion:** prefill determines TTFT, decode determines E2E, and the effect of prefill on decode depends on the ratio of prompt tokens to generated tokens. Traffic with long prompts and short answers, such as retrieval-augmented generation, is the least favourable case.

## 04a Batch width (`max_num_seqs`)

**Measurement:** `--max-num-seqs` 8 / 32 / 256 (baseline) at concurrency 64.

| max_num_seqs | Output tok/s | TTFT p50 | TPOT p50 |
|---|---|---|---|
| 8 | 504 | 14.3 s | 14.9 ms |
| 32 | 1345 | 3.2 s | 21.9 ms |
| 256 | 1851 | 0.39 s | 31.3 ms |

**Result:** limiting the batch to 8 sequences reduces throughput 3.7× and raises median TTFT to 14 seconds, while giving the fastest per-user token rate (14.9 ms against 31.3 ms).

**Explanation:** the V1 engine always batches continuously, but `max_num_seqs` limits how many sequences share each step. With a limit of 8, 56 of 64 requests wait in the queue, so their TTFT is queueing time. The 8 running sequences get light, fast decode steps. A wider batch spreads each weight read over more sequences, which increases throughput but makes each step heavier and per-user tokens slower.

**Conclusion:** batch width trades per-user token speed against queueing delay. For an interactive SLO, queueing delay is the larger cost of a narrow batch.

## 04b Chunked-prefill budget (`max_num_batched_tokens`)

**Measurement:** per-step token budget of 512 / 2048 (baseline) / 8192, with 4096-token prompts at concurrency 16.

| Budget | Output tok/s | TTFT p50 | TPOT p50 | ITL p95 | ITL p99 |
|---|---|---|---|---|---|
| 512 | 285 | 642 ms | 51.0 ms | 57 ms | 59 ms |
| 2048 | 300 | 768 ms | 47.0 ms | 147 ms | 151 ms |
| 8192 | 289 | 1938 ms | 39.6 ms | 27 ms | 560 ms |

**Result:** throughput is flat (285–300 tok/s), so this setting does not change throughput. It changes the shape of the latency distribution:
- **512:** lowest TTFT. Inter-token latency is consistently about 57 ms, with no large stalls.
- **2048:** frequent medium stalls of about 150 ms.
- **8192:** highest TTFT and the lowest p95 ITL (27 ms), but a p99 of 560 ms, the largest stall of the three.

No configuration meets the SLO; 4096-token prompts at concurrency 16 exceed what this GPU and model can serve within it.

**Explanation:**
- **Equal work, different scheduling.** The budget determines how a long prefill is divided across engine steps. Total work is unchanged, so throughput is unchanged. What changes is how that work lands on concurrent decodes.
- **512.** Many small chunks. Every step carries a small amount of prefill, so decode is never stalled for long, but every token is somewhat slower (higher median TPOT).
- **8192.** A whole 4096-token prompt fits in one step, so most steps are pure decode and fast. When a prefill step occurs, every decoding sequence waits for all of it. These stalls are rare enough to be invisible at p95 (27 ms) and dominate p99 (560 ms).
- **2048.** Two medium chunks per prompt, so medium-sized stalls affect more than 5% of tokens.
- **TTFT.** Larger budgets let the scheduler admit several large prefills in one step, which delays later arrivals.

**Conclusion:** the chunked-prefill budget does not change total work; it changes who waits and how often. Small chunks give smooth, predictable streaming. Large chunks give the best p95 and the worst p99. Reporting several percentiles is necessary: at p95 alone, the 8192 configuration would appear to be the best.

## 04c Prefix caching

**Measurement:** caching enabled (baseline) against disabled. Every request shares a 1024-token prefix and adds a unique 128-token suffix, at concurrency 16.

**Result:**
- **TTFT:** p50 decreases from 331 ms to 98 ms (−70%), and p95 from 507 ms to 161 ms.
- **Throughput:** increases 54% (646 to 996 tok/s). Cost decreases from $0.34 to $0.22 per 1M output tokens.
- **Hit rate:** 0.89, equal to the shared fraction 1024 / 1152 = 0.889.
- **SLO:** with caching disabled, p95 TTFT is 507 ms and narrowly misses the limit. With caching enabled it passes with a wide margin.

**Explanation:** vLLM hashes KV blocks by token content. When a request's prefix blocks are already cached, prefill is skipped for those tokens and only the 128-token suffix is computed. This removes about 89% of the prefill computation, which lowers TTFT and frees compute for decoding (TPOT 21.7 to 14.9 ms).

**Conclusion:** for traffic with shared prompts (system prompts, retrieval templates, multi-turn chat), prefix caching gives a substantial improvement with no configuration cost. The measured hit rate matched the theoretical shared fraction. Prefix caching also affected the first benchmark run through unintended prompt reuse, described in the methodology note below.

## 04d KV-cache capacity

**Measurement:** a fixed 2 GiB KV cache (`--kv-cache-memory-bytes`, about 75k tokens at roughly 28 KiB per token) against the default (about 90% of GPU memory), at concurrency 64 with 1024 input / 512 output tokens (about 98k tokens of demand).

| KV budget | KV peak | Preemptions | Output tok/s | TTFT p95 | TPOT p95 | E2E p95 |
|---|---|---|---|---|---|---|
| Default (≈90% of GPU memory) | 17% | 0 | 1905 | 3.3 s | 32.9 ms | 20.0 s |
| 2 GiB | 100% | 70 | 1670 | 7.0 s | 45.7 ms | 26.1 s |

**Result:**
- The 2 GiB cache reached 100% usage and produced 70 preemptions (median; all 3 repeats were consistent).
- Throughput decreased 12%, p95 TTFT approximately doubled, and p95 E2E increased 31%.
- With default settings, the L4 used only 17% of its KV cache at this load.

**Explanation:**
- **Block exhaustion.** Each running sequence needs KV blocks for all of its tokens so far. When no free block remains for the next decode step, the scheduler preempts a running sequence: it frees that sequence's blocks and later recomputes its entire prefix. This is repeated prefill work (lower throughput) and additional waiting for the preempted request (worse tail).
- **Capacity estimate.** The per-token estimate (2 × 28 layers × 2 KV heads × 128 head dimension × 2 bytes ≈ 28 KiB) correctly predicted that about 98k tokens of demand would not fit in 2 GiB.

**Conclusion:** KV-cache capacity is the hard limit on concurrency. Preemption appears as recomputation: lower throughput and a considerably worse tail. This model has roughly 6× headroom on an L4 at the tested load; a larger model would reach this limit at much lower concurrency.

## 05 Cost and performance

**Result:**
- **Lowest cost within the SLO, general traffic:** baseline at concurrency 16: 833 tok/s, p95 TTFT 427 ms, p95 TPOT 18 ms, **$0.27 per 1M output tokens** on an L4. Concurrency 32 would cost $0.18 but exceeds the TTFT limit (809 ms).
- **Shared-prefix traffic:** with prefix caching, concurrency 16 reaches 996 tok/s at **$0.22 per 1M output tokens** with p95 TTFT of 161 ms, which leaves TTFT headroom for higher concurrency.
- **Single user:** $3.03 per 1M output tokens, 11× the cost at the SLO-bound optimum.

**Explanation:** the GPU is billed per hour regardless of utilization, so cost per token is price divided by throughput. Cost decreases as concurrency rises until the latency SLO becomes binding. The optimum is the last point before p95 TTFT crosses the limit; the objective is maximum throughput subject to the SLO, not maximum throughput.

**Conclusion:** for this model on an L4, limit each replica to about 16 concurrent requests to hold a 500 ms p95 TTFT, and scale horizontally beyond that. For prompt-heavy traffic, enable prefix caching before adding GPUs.

## Methodology note: prefix-cache contamination in the first run

**Issue.** The first version of the benchmark used the same random seed (42) for every workload. Workloads within an experiment run consecutively on one server, so experiments 02 and 03 repeatedly sent identical prompts. With prefix caching enabled, a large share of prefill was served from cache. The measured hit rates were 0.88 for concurrency 4–32 and 0.69 for concurrency 64 in experiment 02 (0.04 at concurrency 1), and up to 0.94 in experiment 03 (1024 → 512), against 0.02–0.18 for most other points.

**Detection.** Repeated runs showed that container-to-container variation was small (CV ≤ 3.3%). The same configuration and workload (baseline, 512/128, 64 users) nevertheless produced 2438 tok/s in experiment 02 and 1851 tok/s in 04a, consistently across all three repeats. Run-to-run noise could not explain the gap, and the prefix-cache hit-rate telemetry identified the cause.

**Fix.** The seed is now `42 + crc32(workload_id)`. Workloads on the same server never share prompts, while the same workload under different configurations still receives identical prompts, so comparisons across configurations remain fair. After the fix, experiments 02 and 04a agree within 2.6% and hit rates are about 4% (the warm-up re-sends the first prompt).

**Impact on the first run:**
- **Concurrency limit.** The contaminated experiment 02 indicated that concurrency 32 met the SLO (p95 TTFT 191 ms, $0.14 per 1M tokens). The corrected p95 TTFT is 809 ms. The best SLO-compliant configuration is concurrency 16 at $0.27 per 1M tokens, so the contaminated data understated cost by about a factor of two.
- **Spurious anomaly in 03.** The contaminated run showed 1024 → 512 with a TTFT about 4.5× lower than 1024 → 64. This was a cache effect and does not appear in the corrected data.

**Practice adopted.** Check cache hit rates in every benchmark of a system with caching. Treat a result that is stable across repeats but inconsistent with another experiment as a methodology error until shown otherwise.

## Bottlenecks

- **Single user:** memory bandwidth (TPOT is about 77% of the weights-over-bandwidth floor).
- **Rising concurrency:** prefill queueing and prefill/decode interference. p95 TTFT crosses the limit between 16 and 32 users, while KV-cache usage stays below 7%.
- **Long prompts:** prefill compute. 4096-token prompts exceed the TTFT limit even at concurrency 8.
- **Constrained KV cache:** capacity, leading to preemption and recomputation (04d).

## Trade-offs

- **Throughput and tail latency:** going from 16 to 32 users gives 48% more throughput but 1.9× worse p95 TTFT.
- **Batch width:** a narrow batch gives fast tokens and long queues; a wide batch gives the opposite (04a).
- **Chunk budget:** a small budget gives smooth streaming; a large one gives better p95 and much worse p99. Throughput is unchanged (04b).
- **Cache memory:** a smaller KV budget frees memory, for example for a larger model or more replicas per GPU, at the cost of preemptions under load.

## Limitations

- **GPU utilization.** The nvidia-smi utilization figure decreases as concurrency rises, because the sampling window includes warm-up and high-concurrency runs are short. It also reports that a kernel was running, not SM occupancy. No conclusions are drawn from it.
- **Memory figures.** vLLM reserves about 90% of GPU memory at startup (about 20 GB peak in every default run), so `gpu_mem_peak` does not vary with load. KV-cache usage is the informative measure.
- **Workload.** Prompts are random tokens with no natural-language structure. One model and one GPU type were tested. `--ignore-eos` fixes output lengths. Load is closed-loop (fixed concurrency) rather than a Poisson arrival process.
- **Warm-up.** The warm-up re-sends the first measured prompt, so one request per workload is a cache hit (about 4% hit rate).
- **Cost.** GPU time only, at the listed price. Modal also bills CPU and memory. Container cold start (about 2–3 minutes) and time spent waiting for an L4 are excluded.
- **Version.** Results depend on the defaults and kernels of vLLM 0.34.0.
