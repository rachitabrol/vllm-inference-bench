# vLLM serving: engineering report

All numbers are medians of 3 repeats, each repeat in its own GPU container.
- **01–03:** run `44748004817-1-L4` (per-workload seeds, see "Lesson" below).
- **04a–04d:** run `44708914259-1-L4`.

Combined outputs are in `results/summary_agg.csv`, `results/SUMMARY.md` and `plots/`. The first single-shot run (`44703863719-1-L4`) is kept in `results/raw/` for history only; its 02/03 numbers are wrong.

## Setup

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`, bf16 (~3.1 GB of weights).
- **Engine:** vLLM 0.34.0, V1 engine (continuous batching always on), prefix caching on in the baseline. `vllm serve` is driven by `vllm bench serve`.
- **GPU:** NVIDIA L4 (24 GB, ~300 GB/s memory bandwidth) on Modal at $0.80/hr (price checked 2026-11-20).
- **Workload default:** 512 input / 128 output tokens, random-token prompts with a seed per workload, `--ignore-eos`, `--temperature 0`, 10 warm-up requests.
- **SLO:** p95 TTFT ≤ 500 ms, p95 TPOT ≤ 50 ms, zero failed requests.
- **Reliability:** 0 failed requests in every measured run.
- **Repeatability:** throughput CV across all 23 points is at most 3.3% (median 0.7%). p95 TTFT CV is at most 10.2% (median 2.9%). Differences larger than ~5% are real.

## 01 Baseline latency

**Measured:** concurrency 1, 50 prompts.

**Result:** TTFT p50 38 ms / p95 43 ms. TPOT 13.4 ms (~75 tokens/s for one user). E2E p95 1.75 s for 128 output tokens.

**Why:** decode is memory-bandwidth-bound.
- Each new token reads every weight once: 3.1 GB / 300 GB/s ≈ 10.3 ms is the floor. We measured 13.4 ms, about 77% of the bandwidth roofline. The gap is kernel launch overhead, attention over the KV cache, and sampling.
- A 512-token prefill takes only ~38 ms because prefill processes all prompt tokens in one pass over the weights. That makes prefill compute-bound, not bandwidth-bound.

**Takeaway:** At batch 1 the GPU is a memory-bandwidth machine. I can predict TPOT from model size ÷ bandwidth, and the measurement was within ~25% of that floor.

## 02 Concurrency sweep

**Measured:** the same workload at max concurrency 1, 4, 8, 16, 32 and 64, all on one server, each step with its own prompts.

| conc | out tok/s | TTFT p95 (ms) | TPOT p50 (ms) | ITL p99 (ms) | $/1M out tok | SLO |
|---|---|---|---|---|---|---|
| 1 | 73 | 40 | 13.4 | 15 | 3.03 | yes |
| 4 | 269 | 150 | 13.9 | 17 | 0.83 | yes |
| 8 | 496 | 254 | 14.9 | 37 | 0.45 | yes |
| 16 | 833 | 427 | 16.4 | 99 | 0.27 | yes |
| 32 | 1231 | 809 | 21.5 | 129 | 0.18 | **no** |
| 64 | 1804 | 1721 | 32.0 | 138 | 0.12 | **no** |

**Result:**
- **Throughput scales almost for free at first.** From 1 to 16 users, throughput rose 11× while TPOT only went from 13.4 to 16.4 ms (+23%). From 1 to 64 it rose 25×.
- **The SLO breaks between 16 and 32 users, on p95 TTFT** (427 → 809 ms). p95 TPOT stays far under its 50 ms limit throughout.
- **Memory is not the limit.** KV cache usage peaked at only 6.9% at concurrency 64.

**Why:**
- **Batching amortizes the weight reads.** One pass over the weights produces a token for every sequence in the batch, so at small batch sizes extra users cost almost nothing.
- **Prefill is the bottleneck past that point.** As concurrency rises, more new prompts arrive per second. Their prefills queue and also interleave with decode steps. Queueing shows up in TTFT. Interleaving shows up as ITL p99 spikes (15 ms at 1 user, ~100–140 ms at 16+), where a decode token waits behind someone's prefill chunk.
- **The tail goes first.** Median TTFT barely moves from 32 to 64 users (370 → 413 ms) while p95 doubles. Unlucky requests land behind a burst of prefills.

**Takeaway:** Concurrency is the cheapest lever: 11× cost reduction from 1 to 16 users. The first thing to break is tail TTFT, driven by prefill queueing. It's not average latency and not memory.

## 03 Prefill vs decode

**Measured:** input {128, 1024, 4096} × output {64, 512} at concurrency 8.

| in → out | TTFT p50 (ms) | TPOT p50 (ms) | ITL p95 (ms) | E2E p95 (s) |
|---|---|---|---|---|
| 128 → 64 | 59 | 14.0 | 15 | 0.95 |
| 1024 → 64 | 204 | 18.9 | 67 | 1.46 |
| 4096 → 64 | 853 | 37.7 | 146 | 3.37 |
| 128 → 512 | 67 | 14.0 | 14 | 7.24 |
| 1024 → 512 | 207 | 15.3 | 15 | 8.10 |
| 4096 → 512 | 855 | 20.0 | 18 | 11.23 |

**Result:**
- **Input length sets TTFT,** and output length has no effect on it: 59 / 204 / 853 ms for 128 / 1024 / 4096 input tokens, the same with 64 or 512 outputs. 4096-token prompts miss the 500 ms SLO even at concurrency 8.
- **Output length sets E2E.** Going from 64 to 512 output tokens raises E2E 3.3–7.6×. E2E ≈ TTFT + output_len × TPOT; for example, 67 ms + 512 × 14 ms ≈ 7.2 s.
- **Long prompts slow everyone's decode, but only when outputs are short.** At 4096 → 64, TPOT is 37.7 ms with ITL p95 146 ms. At 4096 → 512, it's 20.0 ms with ITL p95 18 ms.

**Why:**
- **Prefill.** Prefill work grows with prompt length (linear for the MLPs, quadratic for attention), so TTFT tracks input length.
- **Decode.** Decode produces one token per step per sequence, so E2E tracks output length.
- **The decode slowdown is a ratio effect.** A short-output request finishes quickly and is replaced by a new one, which brings a new 4096-token prefill. With 64-token outputs, prefills are a large share of all GPU steps, and decode tokens keep waiting behind them. With 512-token outputs, the same prefills are spread across 8× more decode steps, so the interference is diluted.

**Takeaway:** Prefill sets TTFT, decode sets E2E, and how much prefill hurts decode depends on the ratio of prompt tokens to generated tokens. RAG-style traffic with long prompts and short answers is the worst case.

## 04a max_num_seqs (batch width)

**Measured:** `--max-num-seqs` 8 / 32 / 256 (baseline) at concurrency 64.

| max_num_seqs | out tok/s | TTFT p50 | TPOT p50 |
|---|---|---|---|
| 8 | 504 | 14.3 s | 14.9 ms |
| 32 | 1345 | 3.2 s | 21.9 ms |
| 256 | 1851 | 0.39 s | 31.3 ms |

**Result:** capping the batch at 8 cut throughput 3.7× and pushed median TTFT to 14 seconds, while giving the fastest tokens per user (14.9 ms vs 31.3 ms).

**Why:** V1 always batches continuously, but `max_num_seqs` limits how many sequences share each step. With a cap of 8, 56 of 64 requests sit in the waiting queue, so their TTFT is pure queueing time. The 8 that are running get light, fast decode steps. A wider batch spreads each weight read over more sequences: more throughput, but a heavier step and slower tokens for each user.

**Takeaway:** Batch width trades per-user token speed against queueing. Too narrow and users wait in line; for an interactive SLO the queue is the bigger problem.

## 04b max_num_batched_tokens (chunked prefill budget)

**Measured:** token budget per step of 512 / 2048 (baseline) / 8192, with 4096-token prompts at concurrency 16.

| budget | out tok/s | TTFT p50 | TPOT p50 | ITL p95 | ITL p99 |
|---|---|---|---|---|---|
| 512 | 285 | 642 ms | 51.1 ms | 57 ms | 59 ms |
| 2048 | 300 | 768 ms | 47.0 ms | 147 ms | 151 ms |
| 8192 | 289 | 1938 ms | 39.6 ms | 27 ms | **560 ms** |

**Result:** throughput is flat (285–300 tok/s), so this knob doesn't buy throughput. It changes the *shape* of the latency:
- **512:** best TTFT. Inter-token latency is consistently ~57 ms with no spikes.
- **2048:** frequent medium stalls of ~150 ms.
- **8192:** worst TTFT and the best p95 ITL (27 ms), but a **560 ms p99**, the worst stall of the three.

No config meets the SLO; 4096-token prompts at concurrency 16 are too heavy for this GPU and model.

**Why:**
- **Same work, different scheduling.** The chunk budget decides how a long prefill is sliced across engine steps. Total work is the same, so throughput doesn't change. What changes is how the cost lands on concurrent decodes.
- **512.** Many small chunks. Every step carries a little prefill, so decode is never badly stalled, but every token is a bit slower (high median TPOT).
- **8192.** A whole 4096-token prompt fits in one step, so most steps are pure decode and fast. When a prefill step does happen, every decoding sequence waits for all of it: rare but huge stalls. They are invisible at p95 (27 ms) and dominant at p99 (560 ms).
- **2048.** Two medium chunks per prompt, so medium stalls land on more than 5% of tokens.
- **TTFT.** Larger budgets let the scheduler admit several big prefills at once, which delays later arrivals.

**Takeaway:** The chunked-prefill budget doesn't change total work, it changes who waits and how often. Small chunks give smooth, predictable streaming. Big chunks look great at p95 and terrible at p99. This is why I report more than one percentile: the 8192 config would have looked best if I'd stopped at p95.

## 04c Prefix caching

**Measured:** caching on (baseline) vs off. Every request shares a 1024-token prefix plus a 128-token unique suffix, at concurrency 16.

**Result:**
- **TTFT:** p50 fell from 331 ms to 98 ms (−70%), and p95 from 507 ms to 161 ms.
- **Throughput:** rose 54% (646 → 996 tok/s), and cost fell from $0.34 to $0.22 per 1M output tokens.
- **Hit rate:** 0.89, matching the shared fraction 1024/1152 = 0.889.
- **SLO:** caching off narrowly fails it (p95 TTFT 507 ms); caching on passes easily.

**Why:** vLLM hashes KV blocks by their token content. When a request's prefix blocks are already cached, it skips prefill for those tokens and computes only the 128-token suffix. That removes ~89% of prefill compute, so TTFT drops and the freed compute goes to decode (TPOT 21.7 → 14.9 ms).

**Takeaway:** For shared-prompt traffic (system prompts, RAG templates, multi-turn chat), prefix caching is nearly free performance. The hit rate matched the theoretical shared fraction. It was also strong enough to fool my own benchmark (see Lesson).

## 04d KV cache capacity

**Measured:** a fixed 2 GiB KV cache (`--kv-cache-memory-bytes`, ~75k tokens at ~28 KiB/token) vs the default (~90% of GPU memory), at concurrency 64 with 1024 in / 512 out (~98k tokens of demand).

| KV budget | KV peak | preemptions | out tok/s | TTFT p95 | TPOT p95 | E2E p95 |
|---|---|---|---|---|---|---|
| default (~15 GB) | 17% | 0 | 1905 | 3.3 s | 32.9 ms | 20.0 s |
| 2 GiB | 100% | 70 | 1670 | 7.0 s | 45.7 ms | 26.1 s |

**Result:**
- The 2 GiB cache filled to 100% and caused 70 preemptions (the median; all 3 repeats were consistent).
- Throughput fell 12%, p95 TTFT doubled, and p95 E2E rose 31%.
- At default settings the L4 used only 17% of its KV cache at this load.

**Why:**
- **Running out of blocks.** Every running sequence needs KV blocks for all of its tokens so far. When no free block is left for the next decode step, the scheduler preempts a running sequence: it frees that sequence's blocks and later recomputes its whole prefix. That is wasted prefill work (lower throughput) plus extra waiting for the evicted request (worse tail).
- **The prediction held.** My per-token estimate (2 × 28 layers × 2 KV heads × 128 head dim × 2 bytes ≈ 28 KiB) correctly predicted that 98k tokens would not fit in 2 GiB.

**Takeaway:** KV cache is the real capacity limit for concurrency. I sized this experiment from first principles to force exhaustion. Preemption shows up as recompute: lower throughput and a much worse tail. This model on an L4 has ~6× headroom, so a bigger model would hit this wall far sooner.

## 05 Cost / performance

**Result:**
- **Best under the SLO, general traffic:** baseline at **concurrency 16**, 833 tok/s, p95 TTFT 427 ms, p95 TPOT 18 ms, **$0.27 per 1M output tokens** on an L4. Concurrency 32 would cost $0.18 but fails p95 TTFT (809 ms).
- **Shared-prefix traffic:** with prefix caching, concurrency 16 reaches 996 tok/s at **$0.22 per 1M** with p95 TTFT 161 ms, leaving TTFT headroom to push concurrency higher.
- **Single user:** $3.03 per 1M, 11× more than the SLO-bound optimum.

**Why:** the GPU costs the same per hour whether it's busy or not, so cost per token is just price ÷ throughput. Cost keeps falling as concurrency rises until the latency SLO binds. The optimum is the last point before the TTFT tail crosses the limit, so the target is "max throughput subject to an SLO", not "max throughput".

**Takeaway:** For this model on an L4, cap each replica at ~16 concurrent requests for a 500 ms p95 TTFT and scale horizontally beyond that. For prompt-heavy traffic, turn on prefix caching before buying more GPUs.

## Lesson: my benchmark was fooled by prefix caching

The first version gave every workload the same random seed (42). Workloads in one experiment run back-to-back on the same server, so 02 and 03 kept sending identical prompts. With prefix caching on, 69–94% of prefill came from cache.

**How it was caught:** repeated runs showed container noise was tiny (CV ≤ 3.3%). Yet the identical config and workload (baseline, 512/128, 64 users) gave 2478 tok/s in 02 and 1851 in 04a, consistently across all 3 repeats. Noise couldn't explain that, and the prefix-cache hit-rate telemetry (0.69–0.88 vs 0.03) pinpointed it.

**Fix:** seed = 42 + crc32(workload id). Workloads on one server never share prompts, while the same workload under different configs still gets identical prompts, so comparisons stay fair. After the fix, 02 and 04a agree within 2.6% and hit rates are ~4% (the warm-up re-sends the first prompt).

**What it got wrong:**
- **Concurrency limit:** the contaminated 02 said concurrency 32 met the SLO (p95 TTFT 191 ms, $0.14/1M). The truth is 809 ms. The best SLO-compliant config is concurrency 16 at $0.27/1M, so the bug made the system look about 2× cheaper than it is.
- **Explanation:** it also produced a fake TTFT anomaly in 03 (1024 → 512 looked 4× faster than 1024 → 64), which I had initially explained with a plausible but wrong story about request timing.

**Takeaway:** check cache hit rates in every benchmark of a caching system, and treat a result that survives repeats but contradicts another experiment as a bug until proven otherwise.

## Bottlenecks

- **Single user:** memory bandwidth (TPOT ≈ 77% of the weights ÷ bandwidth floor).
- **Rising concurrency:** prefill queueing and prefill/decode interference. p95 TTFT breaks at 16–32 users; KV cache stayed under 7%.
- **Long prompts:** prefill compute. 4096-token prompts miss the TTFT SLO even at concurrency 8.
- **Constrained KV cache:** capacity, which leads to preemption and recompute (04d).

## Trade-offs

- **Throughput vs tail latency:** 16 → 32 users gives +48% throughput but 1.9× worse p95 TTFT.
- **Batch width:** narrow gives fast tokens and long queues; wide gives the opposite (04a).
- **Chunk budget:** small gives smooth streaming; large gives better p95 but terrible p99. Throughput is unchanged (04b).
- **Cache memory:** a smaller KV budget frees memory (e.g. for a bigger model or more replicas per GPU) at the cost of preemptions under load.

## Limitations

- **GPU utilization is not trustworthy here.** nvidia-smi utilization *falls* as concurrency rises because the sampling window includes warm-up and high-concurrency runs are short. It also measures "a kernel was running", not SM occupancy. I don't draw conclusions from it.
- **Memory numbers are pre-allocated.** vLLM reserves ~90% of GPU memory at startup (~20 GB peak in every default run), so `gpu_mem_peak` doesn't vary with load. KV cache usage % is the meaningful signal.
- **Workload.** Random-token prompts (no natural-text structure); one model and one GPU type; `--ignore-eos` forces exact output lengths. Closed-loop load (fixed concurrency), not a Poisson arrival rate.
- **Warm-up.** vLLM's warm-up re-sends the first measured prompt, so one request per workload is a cache hit (~4% hit rate).
- **Cost.** GPU time only (Modal also bills CPU and memory) at the listed price. Container cold start (~2–3 min) and the time spent waiting for an L4 are excluded.
- **Version.** Results depend on vLLM 0.34.0 defaults and kernels.
