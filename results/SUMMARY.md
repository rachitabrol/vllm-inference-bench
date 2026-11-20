# Benchmark summary

- Runs: 44748004817-1-L4
- Repeats: n = 3..3 runs per point (tables show medians; throughput as median (min–max) when n>1)
- GPU: L4  |  vLLM: 0.34.0
- Model(s): Qwen/Qwen2.5-1.5B-Instruct
- SLO: p95 TTFT <= 500 ms, p95 TPOT <= 50 ms, error rate <= 0
- GPU prices checked: 2026-11-20. Cost = price/hr / (output tok/s * 3600) * 1e6 (GPU time only).
- Warm-up: 10 discarded requests per run (`--num-warmups`). Telemetry window includes warm-up.

## 01 Baseline single-user latency

| config | n | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 1 | 512 | 128 | 74 (73–74) | 0.57 | 38 | 43 | 13.4 | 13.4 | 1748 | 0 | 3.022 | yes |

## 02 Concurrency sweep

| config | n | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 1 | 512 | 128 | 73 (73–73) | 0.57 | 37 | 40 | 13.4 | 13.5 | 1750 | 0 | 3.025 | yes |
| baseline | 3 | 4 | 512 | 128 | 269 (267–270) | 2.10 | 119 | 150 | 13.9 | 14.6 | 1916 | 0 | 0.827 | yes |
| baseline | 3 | 8 | 512 | 128 | 496 (488–497) | 3.88 | 198 | 254 | 14.9 | 15.7 | 2074 | 0 | 0.448 | yes |
| baseline | 3 | 16 | 512 | 128 | 833 (832–837) | 6.51 | 294 | 427 | 16.4 | 18.1 | 2476 | 0 | 0.267 | yes |
| baseline | 3 | 32 | 512 | 128 | 1231 (1221–1233) | 9.61 | 370 | 809 | 21.5 | 23.3 | 3518 | 0 | 0.181 | NO |
| baseline | 3 | 64 | 512 | 128 | 1804 (1758–1812) | 14.10 | 413 | 1721 | 32.0 | 33.7 | 5780 | 0 | 0.123 | NO |

## 03 Prefill vs decode (input/output length)

| config | n | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | ITL p99 | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 8 | 128 | 64 | 542 (536–543) | 8.47 | 59 | 87 | 14.0 | 14.3 | 953 | 0 | 21.9 | 0.410 | yes |
| baseline | 3 | 8 | 128 | 512 | 568 (568–570) | 1.11 | 67 | 88 | 14.0 | 14.0 | 7240 | 0 | 17.6 | 0.391 | yes |
| baseline | 3 | 8 | 1024 | 64 | 367 (355–367) | 5.73 | 204 | 395 | 18.9 | 20.7 | 1464 | 0 | 72.4 | 0.606 | yes |
| baseline | 3 | 8 | 1024 | 512 | 510 (507–511) | 1.00 | 207 | 397 | 15.3 | 15.5 | 8100 | 0 | 22.7 | 0.436 | yes |
| baseline | 3 | 8 | 4096 | 64 | 159 (153–160) | 2.49 | 853 | 992 | 37.7 | 43.4 | 3370 | 0 | 154.0 | 1.396 | NO |
| baseline | 3 | 8 | 4096 | 512 | 370 (365–370) | 0.72 | 855 | 915 | 20.0 | 20.7 | 11230 | 0 | 145.0 | 0.600 | NO |

## 05 Cost / performance under SLO

Best config meeting the SLO (max output tok/s = min $/1M tokens): **baseline** at concurrency **16** (experiment 02, L4): 833 output tok/s (median of n=3), p95 TTFT 427 ms, p95 TPOT 18.1 ms, **$0.267 per 1M output tokens**.

## Run-to-run variance

One row per (experiment, config, workload); each repeat ran in its own container. CV = sample std / mean.

| exp | config | workload | n | tok/s median | min | max | CV % | p95 TTFT CV % |
|---|---|---|---|---|---|---|---|---|
| 01 | baseline | in512_out128_c1 | 3 | 74 | 73 | 74 | 0.2 | 3.8 |
| 02 | baseline | in512_out128_c1 | 3 | 73 | 73 | 73 | 0.0 | 3.4 |
| 02 | baseline | in512_out128_c4 | 3 | 269 | 267 | 270 | 0.6 | 4.2 |
| 02 | baseline | in512_out128_c8 | 3 | 496 | 488 | 497 | 1.1 | 6.0 |
| 02 | baseline | in512_out128_c16 | 3 | 833 | 832 | 837 | 0.3 | 2.6 |
| 02 | baseline | in512_out128_c32 | 3 | 1231 | 1221 | 1233 | 0.5 | 2.4 |
| 02 | baseline | in512_out128_c64 | 3 | 1804 | 1758 | 1812 | 1.6 | 2.8 |
| 03 | baseline | in128_out64_c8 | 3 | 542 | 536 | 543 | 0.7 | 2.9 |
| 03 | baseline | in128_out512_c8 | 3 | 568 | 568 | 570 | 0.2 | 3.7 |
| 03 | baseline | in1024_out64_c8 | 3 | 367 | 355 | 367 | 1.9 | 4.4 |
| 03 | baseline | in1024_out512_c8 | 3 | 510 | 507 | 511 | 0.5 | 3.9 |
| 03 | baseline | in4096_out64_c8 | 3 | 159 | 153 | 160 | 2.5 | 10.2 |
| 03 | baseline | in4096_out512_c8 | 3 | 370 | 365 | 370 | 0.8 | 3.5 |

Throughput CV across 13 groups: max 2.5%, median 0.6%. p95 TTFT CV: max 10.2%, median 3.7%.

### Same config + workload in different experiments

No identical config + workload appears in more than one experiment.


