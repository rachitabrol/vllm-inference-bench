# vLLM serving: engineering report

> **Correction pending (2026-11-20).** A 3-repeat rerun (`44708914259-1-L4`) showed run-to-run noise is tiny (throughput CV ≤ 3.3%), so the 23% gap noted under Limitations is *not* noise. It's a measurement bug: every workload used seed 42, so workloads run back-to-back on one server sent identical prompts and prefix caching served 69–94% of prefill from cache in **02 and 03**. Numbers in 02, 03 and the 05 pick below are inflated; 01 and 04a–04d were unaffected (≤ 5% hit rate, from warm-up). Fixed with per-workload seeds; 01–03 are being re-run.

All numbers below come from run `44703863719-1-L4` (`results/summary.csv`, `results/SUMMARY.md`, `plots/`). Single run per point; see Limitations for how much to trust small differences.

## Setup

- Model: `Qwen/Qwen2.5-1.5B-Instruct`, bf16 (~3.1 GB of weights).
- Engine: vLLM 0.34.0, V1 engine (continuous batching always on), `vllm serve` driven by `vllm bench serve`.
- GPU: NVIDIA L4 (24 GB, ~300 GB/s memory bandwidth) on Modal at $0.80/hr (checked 2026-11-20).
- Workload default: 512 input / 128 output tokens, random dataset, seed 42, `--ignore-eos`, `--temperature 0`, 10 discarded warm-up requests.
- SLO: p95 TTFT ≤ 500 ms, p95 TPOT ≤ 50 ms, zero failed requests.
- Every run: 0 failed requests.

## 01 Baseline latency

**Measured:** concurrency 1, 50 prompts.

**Result:** TTFT p50 36 ms / p95 39 ms. TPOT 13.4 ms (≈ 75 tokens/s for one user). E2E p95 1.75 s for 128 output tokens.

**Why:** decode is memory-bandwidth-bound. Each new token reads every weight once: 3.1 GB / 300 GB/s ≈ 10.3 ms is the theoretical floor. We measured 13.4 ms, about 77% of the bandwidth roofline. The remaining gap is kernel launch overhead, attention over the KV cache, and sampling. A 512-token prefill takes only ~36 ms because prefill processes all prompt tokens in parallel (one weight read, many tokens), which makes it compute-bound rather than bandwidth-bound.

**Interview takeaway:** "At batch 1 the GPU is a memory-bandwidth machine. I can predict TPOT from model size ÷ bandwidth, and my measurement was within ~25% of that floor."

## 02 Concurrency sweep

**Measured:** same workload at max concurrency 1, 4, 8, 16, 32, 64.

| conc | out tok/s | TTFT p95 (ms) | TPOT p50 (ms) | $/1M out tok | SLO |
|---|---|---|---|---|---|
| 1 | 73 | 48 | 13.5 | 3.04 | yes |
| 8 | 545 | 89 | 14.2 | 0.41 | yes |
| 32 | 1603 | 191 | 17.6 | 0.14 | yes |
| 64 | 2438 | 832 | 22.0 | 0.09 | **no** |

**Result:** throughput scaled ~7.5× from 1→8 (almost perfectly linear) and ~33× from 1→64, while TPOT p50 only rose from 13.5 to 22 ms. The SLO breaks between 32 and 64: p95 TTFT jumps from 191 to 832 ms.

**Why:** batching amortizes the weight reads. One pass over the weights produces a token for every sequence in the batch, so up to ~16 concurrent requests each user barely notices (TPOT +10%) while total throughput climbs ~14×. Past that, each step does more compute and new prompts' prefills interleave with decode steps, so per-token latency starts to rise. At 64, prefill of new arrivals queues behind a busy engine, and that shows up first in the TTFT tail (p95), not the median. Memory was not the limit: KV cache usage peaked at only ~7% at concurrency 64.

**Interview takeaway:** "Concurrency is the cheapest lever: 22× lower cost per token from 1 to 32 users. The constraint that binds first is tail TTFT, not average latency and not memory."

## 03 Prefill vs decode

**Measured:** input {128, 1024, 4096} × output {64, 512} at concurrency 8.

| in → out | TTFT p50 (ms) | TPOT p50 (ms) | E2E p95 (s) |
|---|---|---|---|
| 128 → 64 | 54 | 13.8 | 0.95 |
| 1024 → 64 | 237 | 17.5 | 1.39 |
| 4096 → 64 | 708 | 34.3 | 3.16 |
| 128 → 512 | 48 | 13.8 | 7.13 |
| 1024 → 512 | 52 | 14.7 | 7.58 |
| 4096 → 512 | 807 | 19.8 | 11.11 |

**Result:**
- **Input length drives TTFT.** At 4096 input tokens, TTFT is 700–800 ms, over the SLO even at concurrency 8.
- **Output length drives E2E.** Going from 64 to 512 output tokens raises E2E 3.5–7.5×, roughly output_len × TPOT on top of TTFT.
- **Long prompts slow other users' decode.** At 4096→64, TPOT p50 is 34 ms and ITL p95 is 142 ms, versus ~14 ms for short prompts.
- **Odd one out.** 1024→512 has a *lower* TTFT (52 ms) than 1024→64 (237 ms).

**Why:**
- **Prefill.** Prefill work grows with prompt length (linear for the MLPs, quadratic for attention), so TTFT tracks input length.
- **Decode.** Decode emits one token per step per sequence, so E2E tracks output length.
- **Interference.** When a 4096-token prefill is chunked into the same steps as other sequences' decode, those decode tokens wait. That is the TPOT/ITL hit.
- **The odd case.** With short outputs, all 8 requests finish at about the same time and the next 8 arrive together, so their prefills collide. With long outputs, completions are staggered, so a new prompt usually prefills nearly alone. This is a workload-shape effect, not a property of 1024-token prompts.

**Interview takeaway:** "Prefill sets TTFT, decode sets E2E, and long prefills hurt everyone else's decode. That interference is exactly what chunked prefill (04b) is meant to manage."

## 04a max_num_seqs (batch width)

**Measured:** `--max-num-seqs` 8 / 32 / 256 (baseline) at concurrency 64.

| max_num_seqs | out tok/s | TTFT p50 | TPOT p50 |
|---|---|---|---|
| 8 | 504 | 14.4 s | 14.9 ms |
| 32 | 1343 | 3.2 s | 22.2 ms |
| 256 | 1869 | 0.38 s | 31.0 ms |

**Result:** capping the batch at 8 cut throughput 3.7× and pushed median TTFT to 14 seconds. The small cap did give the best per-token latency (14.9 ms vs 31 ms).

**Why:** V1 always batches continuously, but `max_num_seqs` limits how many sequences can share each step. With a cap of 8, 56 of the 64 requests sit in the waiting queue, so TTFT becomes pure queueing time. The running 8 get fast, light decode steps. A wider batch spreads each weight read over more sequences, giving higher throughput but a heavier step and slower tokens for each user.

**Interview takeaway:** "Batch width trades per-user token speed against queueing. Too narrow and users wait in line; for an interactive SLO the queue is the bigger problem."

## 04b max_num_batched_tokens (chunked prefill budget)

**Measured:** token budget per step of 512 / 2048 (baseline) / 8192, with 4096-token prompts at concurrency 16.

| budget | out tok/s | TTFT p50 | TPOT p50 | ITL p95 |
|---|---|---|---|---|
| 512 | 288 | 636 ms | 50.2 ms | 57 ms |
| 2048 | 297 | 875 ms | 47.0 ms | 147 ms |
| 8192 | 291 | 1914 ms | 39.8 ms | 28 ms |

**Result:** throughput was the same within noise (~3%), so this knob doesn't buy throughput here. What it changes is how latency is distributed:
- **512:** best TTFT, worst TPOT.
- **8192:** worst TTFT, best TPOT and best p95 ITL.
- **2048:** a bad p95 ITL. No config met the SLO, because 4096-token prompts at concurrency 16 are too heavy for an L4 with this model.

**Why (best explanation, worth re-running to confirm):**
- **512.** Long prompts are split into many small chunks. Every step stays light, so decode is never badly stalled, but prefill work is spread across many steps that all carry some extra load, which raises median TPOT.
- **8192.** A whole 4096-token prompt fits in one step. Prefills finish in a few big steps, so most decode steps are pure decode and fast (low p95 ITL). The stalls become rarer but larger, and p95 hides them. Admitting big prefills in bursts also lengthens the queue for later arrivals (high TTFT).
- **2048.** Each prompt takes two medium chunks, so moderate stalls land on more than 5% of tokens, which shows up in p95 ITL.

p99/max ITL would separate "rare big stalls" from "frequent medium stalls". That is the next measurement to add.

**Interview takeaway:** "The chunked-prefill budget doesn't change total work, it changes who waits. Small chunks protect time-to-first-token, big chunks protect steady decode. Choose based on which SLO you're defending, and look at more than one percentile."

## 04c Prefix caching

**Measured:** caching on (baseline) vs off; every request shares a 1024-token prefix plus a 128-token unique suffix, concurrency 16.

**Result:**
- TTFT p50 fell from 430 ms to 109 ms (−75%), and p95 from 590 ms to 174 ms.
- Throughput rose 61% (613 → 990 tok/s), and cost per token fell by the same factor ($0.36 → $0.23 per 1M).
- Prefix cache hit rate was 0.89, almost exactly the shared fraction 1024/1152 = 0.889.
- Caching on is the only config here that meets the SLO.

**Why:** with caching, vLLM hashes KV blocks by their token content. A request whose prefix blocks are already cached skips prefill for those tokens and only computes the 128-token suffix. That removes ~89% of prefill compute, so TTFT drops and freed compute goes to decode. Without caching, every request recomputes the same 1024-token prefix.

**Interview takeaway:** "For workloads with shared prompts (system prompts, RAG templates, multi-turn chat), prefix caching is close to free performance: the hit rate matched the theoretical shared fraction, and it was the difference between passing and failing the SLO."

## 04d KV cache capacity

**Measured:** a fixed 2 GiB KV cache (`--kv-cache-memory-bytes`, ~75k tokens at ~28 KiB/token) vs the default (~90% of GPU memory), at concurrency 64 with 1024 in / 512 out (~98k tokens of demand).

| KV budget | KV peak | preemptions | out tok/s | TTFT p95 | TPOT p95 | E2E p95 |
|---|---|---|---|---|---|---|
| default (~15 GB) | 17% | 0 | 1871 | 3.4 s | 33.5 ms | 20.3 s |
| 2 GiB | 100% | 70 | 1670 | 7.0 s | 45.7 ms | 26.1 s |

**Result:**
- The small cache filled to 100% and caused 70 preemptions.
- Throughput fell 11%, p95 TTFT doubled, and p95 E2E rose 29%.
- With the default budget, the L4 used only 17% of its KV cache even at this load.

**Why:**
- **Running out of blocks.** Each running sequence needs KV blocks for every token so far. When no free block is left for the next decode step, the scheduler preempts a running sequence: it frees that sequence's blocks and later recomputes its whole prefix. That is wasted prefill work (lower throughput) plus extra waiting for the victim (tail latency).
- **Validating the setup.** My per-token estimate (2 × 28 layers × 2 KV heads × 128 head dim × 2 bytes ≈ 28 KiB) correctly predicted that 98k tokens would not fit in 2 GiB.

**Interview takeaway:** "KV cache is the real capacity limit for concurrency. I sized the experiment from first principles to force exhaustion, and preemption shows up as recompute: lower throughput and a much worse tail. On this GPU and model there's ~6× headroom, so a bigger model would hit this wall much sooner."

## 05 Cost / performance

**Result:** the cheapest config that meets the SLO is the **baseline at concurrency 32**: 1603 output tok/s, p95 TTFT 191 ms, p95 TPOT 17.9 ms, **$0.139 per 1M output tokens** on an L4. Concurrency 64 is 35% cheaper ($0.091) but fails the TTFT SLO (832 ms p95). The single-user point costs $3.04, 22× more.

**Why:** the GPU costs the same per hour whether it's busy or not, so cost per token is just price ÷ throughput. Cost keeps falling as concurrency rises until the latency SLO binds. The optimum sits just before the TTFT tail crosses the limit, which is why you pick "max throughput subject to an SLO" rather than "max throughput".

**Interview takeaway:** "My recommendation for this model on an L4 is to cap concurrency around 32 per replica and scale horizontally beyond that. For prompt-heavy traffic, turn on prefix caching first, since it moved a failing config to passing."

## Bottlenecks

- **Single user:** memory bandwidth (TPOT ≈ 77% of the weights ÷ bandwidth floor).
- **High concurrency, short prompts:** scheduling and prefill/decode interference. TTFT tail grows first; KV cache stayed under 7% used.
- **Long prompts:** prefill compute. 4096-token prompts miss the TTFT SLO even at concurrency 8.
- **Constrained KV cache:** capacity, which leads to preemption and recompute (04d).

## Trade-offs

- **Throughput vs tail latency:** 32 → 64 users gives +52% throughput but 4.4× worse p95 TTFT.
- **Batch width:** narrow gives fast tokens but long queues; wide gives the opposite (04a).
- **Chunk budget:** small protects TTFT, large protects steady decode; throughput unchanged (04b).
- **Cache memory:** a smaller KV budget frees memory (e.g. for a bigger model or more replicas per GPU) at the cost of preemptions under load.

## Limitations

- **One run per point.** The identical config and workload (baseline, 512/128, concurrency 64) gave 2438 tok/s in 02 and 1869 tok/s in 04a: a 23% spread between two L4 containers. Treat differences under ~25% across experiments as noise. Comparisons *within* one experiment share a run window but still ran on separate containers. Next step: 3 repeats per point, report the median.
- **GPU utilization is not trustworthy here.** nvidia-smi utilization *falls* as concurrency rises because the sampling window includes warm-up and the high-concurrency runs are short (13 s). It also measures "a kernel was running", not SM occupancy. I don't draw conclusions from it.
- **Memory numbers are pre-allocated.** vLLM reserves ~90% of GPU memory at startup (~20 GB peak in every default run), so `gpu_mem_peak` doesn't vary with load. KV cache usage % is the meaningful signal.
- **Workload.** Random-token prompts (no natural-text structure); a single model and GPU type; `--ignore-eos` forces exact output lengths.
- **Cost.** GPU time only (Modal also bills CPU and memory) at the listed price; container cold start (~2–3 min) is excluded.
- **Version.** Results depend on vLLM 0.34.0 defaults and kernels.
