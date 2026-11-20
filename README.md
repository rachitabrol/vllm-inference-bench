# vllm-inference-bench

How does vLLM behave under load, and what does it cost? Benchmarks `Qwen/Qwen2.5-1.5B-Instruct` served with vLLM 0.34.0 on a cloud GPU, one engine setting at a time, then turns the numbers into latency curves and $ per 1M output tokens.

GitHub runners have no GPU. Actions only orchestrates; Modal runs the GPU work. Your laptop does nothing.

```
GitHub Actions (CPU runner)                  Modal (GPU container, one per server config)
───────────────────────────                  ────────────────────────────────────────────
workflow_dispatch                            run_config(job)
  └─ modal run benchmark/modal_app.py ─────►   1. vllm serve  <config flags>
       fan out: run_config.map(jobs)           2. wait for /health
                                               3. vllm bench serve  x N workloads
                                               4. sample nvidia-smi + /metrics @ 0.5s
  ◄──────── one JSON per workload ─────────    5. return results, stop server
analysis/analyze.py
  ├─ results/summary.csv, SUMMARY.md  ─► $GITHUB_STEP_SUMMARY
  ├─ plots/*.png
  └─ upload-artifact + commit results/ and plots/ (non-smoke only)

HF weights cached in Modal Volume `vllm-bench-hf-cache`.
```

## Owner setup (once, ~5 min)

1. Create an empty public repo `rachitabrol/vllm-inference-bench` and push this code.
2. modal.com: sign up, then Settings > API Tokens > new token.
3. Repo > Settings > Secrets and variables > Actions: add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
4. Actions tab > "benchmark" > Run workflow with `smoke=true`. Check the logs/summary: it prints the head of `vllm serve --help` and `vllm bench serve --help` so you can eyeball the flags.
5. Then run with `smoke=false`, `gpu=L4`. Optionally rerun on `A100-40GB` for the final numbers.

## Run it

**Actions UI:** Actions > benchmark > Run workflow. Inputs: `experiments` (`all`, or `01,02`, or `04`), `gpu` (`L4` | `A100-40GB` | `A10G` | `H100`), `smoke` (default true), `repeats` (default 3; each repeat runs in its own container; smoke forces 1).

**Locally with Modal** (needs `pip install -r requirements.txt` and `modal setup`):

```bash
modal run benchmark/modal_app.py --smoke                                  # cents, validates the pipeline
modal run benchmark/modal_app.py --experiments all --gpu L4 --no-smoke --repeats 3   # full run
python analysis/analyze.py results/raw/<run_id>                          # summary.csv, SUMMARY.md, plots/
```

**Dev checks (no GPU, no Modal account):** `pip install -r requirements-dev.txt && ruff check . && pytest -q`

## Experiments

Default workload: 512 input / 128 output tokens, random dataset, seed 42, `--ignore-eos` (exact output length), 10 discarded warm-up requests, `num_prompts = max(200, 4 x concurrency)`. Every variant config differs from `baseline` in exactly one flag. Each (experiment, config) gets a fresh server in a fresh container.

| ID | Question | What varies | Workload |
|----|----------|-------------|----------|
| 01 | Single-user latency profile | baseline | concurrency 1, 50 prompts |
| 02 | Throughput vs latency curve | `--max-concurrency` 1,4,8,16,32,64 | baseline server |
| 03 | Prefill vs decode cost | input {128,1024,4096} x output {64,512} | concurrency 8 |
| 04a | Continuous-batching width | `--max-num-seqs` 8 / 32 / 256 | concurrency 64 |
| 04b | Chunked-prefill budget | `--max-num-batched-tokens` 512 / 2048 / 8192 | 4096-token inputs, concurrency 16 |
| 04c | Prefix caching | `--enable-prefix-caching` vs `--no-enable-prefix-caching` | shared 1024-token prefix + 128 suffix, concurrency 16 |
| 04d | KV cache capacity | `--kv-cache-memory-bytes` 2 GiB vs default (≈90% of GPU) | 1024 in / 512 out, concurrency 64 |
| 05 | Cheapest config that meets the SLO | analysis only | uses 02 + 04 |

SLO (`configs/slo.yaml`): p95 TTFT <= 500 ms, p95 TPOT <= 50 ms, zero failed requests.
Cost: `$/1M output tokens = gpu_price_per_hr / (output_tok_per_s x 3600) x 1e6`. Prices in `configs/gpu_prices.yaml` (Modal, checked 2026-11-20; GPU time only).

vLLM V1 always batches continuously and that cannot be turned off; 04a shows its effect by capping the batch width.

## Layout

```
benchmark/   modal_app.py (image, volume, run_config, entrypoint)  server.py  workloads.py  telemetry.py  config.py
configs/     baseline.yaml + one-flag variants, experiments.yaml, slo.yaml, gpu_prices.yaml
analysis/    analyze.py, test_analyze.py, fixtures/ (FAKE data for tests only)
results/     raw/<run_id>/*.json, summary.csv, SUMMARY.md   (empty until the first real run)
plots/       *.png                                           (empty until the first real run)
report.md    engineering write-up (template; filled after real results)
```

## Key findings

From run `44703863719-1-L4` (Qwen2.5-1.5B, vLLM 0.34.0, one L4). Details and caveats in [report.md](report.md).

- **Batch 1 is bandwidth-bound:** TPOT 13.4 ms vs a 10.3 ms floor (3.1 GB weights / 300 GB/s).
- **Concurrency is the big lever:** 73 → 1603 tok/s from 1 to 32 users with TPOT only 13.5 → 17.6 ms; cost per 1M output tokens $3.04 → $0.14.
- **Tail TTFT binds first:** at 64 users p95 TTFT jumps to 832 ms (SLO 500 ms) while KV cache is only ~7% used.
- **Prefix caching:** −75% TTFT, +61% throughput on a shared-prefix workload; hit rate 0.89 = the shared fraction.
- **KV exhaustion:** a 2 GiB cache under ~98k tokens of demand caused 70 preemptions, 2× p95 TTFT, −11% throughput.
- **Best under SLO:** baseline at concurrency 32 → **$0.139 / 1M output tokens** on an L4.
- **Noise:** identical configs on two containers differed by 23%. Single runs; small differences are not significant.

## How this was built

I designed the experiments and own the results and analysis. Claude (Anthropic) was used as a coding assistant for scaffolding the harness, configs, tests and workflows; every benchmark number in `results/` comes from real runs on Modal GPUs.
