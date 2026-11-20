# vLLM Inference Benchmark — Implementation Plan

Goal: portfolio project. Benchmark vLLM serving on a cloud GPU, explain results in interviews.
Owner: Rachit. Repo: `rachitabrol/vllm-inference-bench` (to be created by owner, empty).

## Architecture

```
GitHub Actions (CPU runner, free)          Modal (GPU, $30/mo free credits)
──────────────────────────────            ─────────────────────────────────
workflow_dispatch (inputs)        ──►      run_config(cfg) on A100/L4
  modal run bench/modal_app.py              1. start `vllm serve` (subprocess) w/ cfg flags
                                            2. wait for /health
                                            3. run workloads with `vllm bench serve`
                                            4. sample nvidia-smi + /metrics during run
                                            5. return JSON results
  ◄── results JSON (one per config)
  analyze.py → results/summary.csv, plots/*.png, results/SUMMARY.md
  SUMMARY.md → $GITHUB_STEP_SUMMARY (report visible in Actions UI)
  upload-artifact + commit results/ plots/ back to repo
```

Key point: GitHub runners have no GPU. Actions only orchestrates. Modal does GPU work. Owner's laptop does nothing.

## Decisions (fixed — don't relitigate)

- **Model**: `Qwen/Qwen2.5-1.5B-Instruct` (ungated, no HF token). Smoke-test model: `Qwen/Qwen2.5-0.5B-Instruct`.
- **GPU**: workflow input, default `L4` (cheap). Option `A100-40GB` for final run. Price table in `configs/gpu_prices.yaml` (Modal per-hour: T4 0.59, L4 ~0.80, A10G ~1.10, A100-40GB 2.10, H100 3.95 — verify on modal.com/pricing and note date checked).
- **dtype**: bf16 (L4/A100 support it).
- **vLLM version**: pin latest stable from PyPI at build time. Flag names drift between versions — verify every CLI flag with `vllm serve --help` / `vllm bench serve --help` inside the image (smoke test does this).
- **Load generator**: vLLM's built-in `vllm bench serve` (OpenAI-compatible endpoint, `--save-result` JSON). Gives TTFT, TPOT, ITL, E2EL (mean/median/p90/p95/p99), request throughput, output tok/s. Don't hand-roll a client. Thin Python wrapper builds the command.
- **Datasets**: `--dataset-name random` with fixed `--seed 42`, `--random-input-len`, `--random-output-len`, `--ignore-eos` (forces exact output length → fair comparisons). Prefix caching test: `--dataset-name prefix_repetition` if available in pinned version, else `random` with `--random-prefix-len`.
- **Extra telemetry** (background thread during each run, 0.5s interval):
  - `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits` → peak/mean memory, mean util.
  - scrape `http://localhost:8000/metrics`: any gauge matching `cache_usage` (KV cache %), `num_requests_waiting`, `num_preemptions` counter, prefix cache hit/query counters. Match by regex; metric names vary by version. Record whatever exists, don't crash if missing.
- **Fresh server per config.** Server flags = config. Workloads within a config reuse the server.
- **Parallelism**: run configs via `run_config.map(...)` across containers (cap ≤ 8 concurrent). Each container is an isolated GPU → fair.
- **Weights cache**: Modal Volume `vllm-bench-hf-cache` mounted at HF cache dir. Avoid re-downloading.
- **Timeouts**: Modal function `timeout=1800`. Workflow `timeout-minutes: 90`.
- **Warmup**: before each measured run, send ~10 warmup requests (discard). Note in report.

## SLO (latency limit for experiment 05)

Defined in `configs/slo.yaml`:
- p95 TTFT ≤ 500 ms
- p95 TPOT ≤ 50 ms
- error rate = 0

## Experiments

Workload default: input 512, output 128, `num-prompts` 200 (≥ 4× concurrency), seed 42.

| ID | What | Server config | Workload sweep |
|----|------|---------------|----------------|
| 01 baseline | single-user latency profile | `baseline` | concurrency 1, 50 prompts |
| 02 concurrency | throughput vs latency curve | `baseline` | `--max-concurrency` 1, 4, 8, 16, 32, 64 (same workload each) |
| 03 prefill/decode | input vs output length effects | `baseline` | input {128, 1024, 4096} × output {64, 512}, concurrency 8 |
| 04a max_num_seqs | batch size cap (continuous batching width) | `max_seqs_8`, `max_seqs_32`, `baseline` (default 256) | concurrency 64 |
| 04b chunked prefill budget | `--max-num-batched-tokens` 512 vs 2048 vs 8192 | 3 configs | long input (4096) mixed w/ concurrency 16 |
| 04c prefix caching | `--enable-prefix-caching` vs `--no-enable-prefix-caching` | 2 configs | shared-prefix workload (prefix 1024, unique suffix 128, output 128), concurrency 16 |
| 04d KV cache capacity | `--kv-cache-memory-bytes` 2 GiB vs default (gpu util 0.9) | 2 configs | concurrency 64, input 1024, output 512 → ~98k tokens demand vs ~75k capacity → preemptions |
| 05 cost/perf | analysis only, no new runs | — | uses 02 + 04 results |

Rule: each 04 config differs from `baseline` in exactly ONE flag. Baseline flags written explicitly (don't rely on vLLM defaults, they change).

Note for report: vLLM V1 always does continuous batching — can't turn off. Experiment 04a shows its effect by capping the batch width (`max_num_seqs`), which is the honest way to frame it.

## Configs (`configs/*.yaml`)

```yaml
# configs/baseline.yaml
name: baseline
model: Qwen/Qwen2.5-1.5B-Instruct
dtype: bfloat16
max_model_len: 8192
gpu_memory_utilization: 0.90
max_num_seqs: 256
max_num_batched_tokens: 2048
enable_prefix_caching: true
seed: 42
```

Variants inherit baseline via `base: baseline` + overrides. Loader merges. `experiments.yaml` maps experiment ID → list of (config, workloads).

## Repo layout

```
README.md               architecture diagram, setup, how to run, key findings (filled after run)
report.md               engineering report — written AFTER real results exist (template w/ TODOs now)
PLAN.md                 this file
requirements.txt        modal, pyyaml, pandas, matplotlib (runner side)
configs/
  baseline.yaml, *.yaml variants
  experiments.yaml      experiment matrix
  slo.yaml
  gpu_prices.yaml
benchmark/
  modal_app.py          Modal image, volume, run_config(), local_entrypoint(experiments, gpu, smoke)
  server.py             start/stop vllm serve, health wait, log capture
  workloads.py          builds `vllm bench serve` commands from workload spec
  telemetry.py          nvidia-smi + /metrics sampler thread
  config.py             yaml load + inheritance merge + CLI flag rendering
analysis/
  analyze.py            raw JSON → summary.csv, plots, SUMMARY.md, SLO pass/fail, $/1M tokens
  test_analyze.py       pytest on fixture JSON (no GPU)
  fixtures/             tiny fake result JSON
results/
  raw/<run_id>/<exp>_<config>_<workload>.json
  summary.csv
  SUMMARY.md
plots/
  02_throughput_vs_concurrency.png
  02_latency_vs_concurrency.png   (p50/p95 TTFT + TPOT)
  03_ttft_vs_input_len.png
  03_e2e_vs_output_len.png
  04_*_comparison.png             (bar charts baseline vs variant)
  05_cost_vs_latency.png          (scatter: $/1M output tok vs p95 TTFT, SLO line drawn)
.github/workflows/
  benchmark.yml         workflow_dispatch → Modal → analyze → commit + step summary
  ci.yml                on push/PR: ruff + pytest analysis (no GPU, no secrets)
```

## Metrics per run (row in summary.csv)

run_id, experiment, config, gpu, vllm_version, model, input_len, output_len, concurrency, num_prompts,
completed, failed, duration_s, request_throughput, output_tok_per_s, total_tok_per_s,
ttft_{mean,p50,p95,p99}_ms, tpot_{mean,p50,p95,p99}_ms, itl_{p50,p95}_ms, e2el_{p50,p95}_ms,
gpu_mem_peak_mib, gpu_util_mean, kv_cache_usage_peak, preemptions, prefix_cache_hit_rate,
gpu_price_per_hr, usd_per_1m_output_tokens, meets_slo

Cost formula: `usd_per_1m_output_tokens = gpu_price_per_hr / (output_tok_per_s * 3600) * 1e6`.
05 = among rows that meet SLO, pick max output_tok_per_s (= min $/1M tokens). Report that config + concurrency.

## GitHub Actions — benchmark.yml

- Trigger: `workflow_dispatch` inputs:
  - `experiments`: string, default `all` (comma list e.g. `01,02`)
  - `gpu`: choice `L4` | `A100-40GB` | `A10G` | `H100`, default `L4`
  - `smoke`: boolean, default `true` → 0.5B model, 20 prompts, concurrency {1,4}, only exp 01+02. Costs cents. Validates pipeline + prints `--help` of pinned vLLM into logs.
- `permissions: contents: write`
- `concurrency: group: benchmark` (no overlapping paid runs)
- Steps: checkout → setup-python 3.12 → pip install -r requirements.txt → `modal run benchmark/modal_app.py --experiments ... --gpu ... --smoke ...` with env `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` from secrets → `python analysis/analyze.py results/raw/<run_id>` → `cat results/SUMMARY.md >> $GITHUB_STEP_SUMMARY` → upload-artifact (results + plots) → commit & push if not smoke (`git config user.name github-actions[bot]`).
- Fail loudly if secrets missing.

## Owner setup (manual, ~5 min)

1. Create empty repo `rachitabrol/vllm-inference-bench` (public = portfolio-visible).
2. modal.com → sign up → Settings → API Tokens → new token.
3. Repo → Settings → Secrets → Actions: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.
4. Actions tab → "benchmark" → Run workflow → smoke=true first. Then smoke=false, gpu=L4.

## Budget estimate

~15 server configs × (cold start ~2–3 min + runs ~5–10 min) ≈ 2–3 GPU-hours. L4 ≈ $2–3, A100 ≈ $5–6. Well under $30.

## Acceptance criteria

- `ruff check .` clean, `pytest analysis` passes locally without GPU or Modal creds.
- `python -c "import ast; ..."` / `modal` import of modal_app works without creds (no side effects at import).
- Config loader renders exact `vllm serve` flags; unit test asserts each 04 variant differs from baseline in exactly one flag.
- analyze.py produces all plots + SUMMARY.md from fixtures.
- README has setup + run instructions; report.md is a template with section headers + TODO markers, NO invented numbers.
- No fabricated results anywhere in repo. results/ starts empty (`.gitkeep`).
