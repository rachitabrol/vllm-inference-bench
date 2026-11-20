# Benchmark summary

- Runs: 44708914259-1-L4, 44748004817-1-L4
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

## 04a max_num_seqs

| config | n | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | GPU util % | KV peak % | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| max_seqs_8 | 3 | 64 | 512 | 128 | 504 (503–508) | 3.94 | 14344 | 14426 | 14.9 | 15.6 | 16269 | 0 | 73 | 1 | 0.441 | NO |
| max_seqs_32 | 3 | 64 | 512 | 128 | 1345 (1330–1347) | 10.51 | 3243 | 3675 | 21.9 | 22.9 | 6425 | 0 | 51 | 3 | 0.165 | NO |
| baseline | 3 | 64 | 512 | 128 | 1851 (1850–1885) | 14.46 | 389 | 1657 | 31.3 | 32.7 | 5615 | 0 | 47 | 7 | 0.120 | NO |

## 04b max_num_batched_tokens

| config | n | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | ITL p95 | ITL p99 | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| batched_tokens_512 | 3 | 16 | 4096 | 128 | 285 (282–287) | 2.23 | 642 | 1635 | 51.0 | 51.6 | 7447 | 0 | 56.6 | 59.2 | 0.780 | NO |
| baseline | 3 | 16 | 4096 | 128 | 300 (284–302) | 2.35 | 768 | 1493 | 47.0 | 49.3 | 7382 | 0 | 147.1 | 151.1 | 0.740 | NO |
| batched_tokens_8192 | 3 | 16 | 4096 | 128 | 289 (284–290) | 2.26 | 1938 | 2672 | 39.6 | 51.2 | 7832 | 0 | 27.0 | 560.2 | 0.768 | NO |

## 04c Prefix caching

| config | n | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | prefix hit rate | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no_prefix_cache | 3 | 16 | 1152 | 128 | 646 (645–647) | 5.05 | 331 | 507 | 21.7 | 23.2 | 3202 | 0 | - | 0.344 | NO |
| baseline | 3 | 16 | 1152 | 128 | 996 (989–997) | 7.78 | 98 | 161 | 14.9 | 15.2 | 2008 | 0 | 0.89 | 0.223 | yes |

## 04d KV cache capacity (kv_cache_memory_bytes)

| config | n | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | KV peak % | preemptions | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 64 | 1024 | 512 | 1905 (1848–1907) | 3.72 | 569 | 3330 | 32.6 | 32.9 | 19965 | 0 | 17 | 0 | 0.117 | NO |
| kv_cache_2gib | 3 | 64 | 1024 | 512 | 1670 (1631–1672) | 3.26 | 1656 | 6969 | 30.6 | 45.7 | 26143 | 0 | 100 | 70 | 0.133 | NO |

## 05 Cost / performance under SLO

Best config meeting the SLO (max output tok/s = min $/1M tokens): **baseline** at concurrency **16** (experiment 04c, L4): 996 output tok/s (median of n=3), p95 TTFT 161 ms, p95 TPOT 15.2 ms, **$0.223 per 1M output tokens**.

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
| 04a | baseline | in512_out128_c64 | 3 | 1851 | 1850 | 1885 | 1.1 | 2.6 |
| 04a | max_seqs_32 | in512_out128_c64 | 3 | 1345 | 1330 | 1347 | 0.7 | 0.6 |
| 04a | max_seqs_8 | in512_out128_c64 | 3 | 504 | 503 | 508 | 0.6 | 0.5 |
| 04b | baseline | in4096_out128_c16 | 3 | 300 | 284 | 302 | 3.3 | 1.5 |
| 04b | batched_tokens_512 | in4096_out128_c16 | 3 | 285 | 282 | 287 | 0.8 | 0.5 |
| 04b | batched_tokens_8192 | in4096_out128_c16 | 3 | 289 | 284 | 290 | 1.0 | 5.5 |
| 04c | baseline | prefix1024x1_sfx128_out128_c16 | 3 | 996 | 989 | 997 | 0.4 | 5.3 |
| 04c | no_prefix_cache | prefix1024x1_sfx128_out128_c16 | 3 | 646 | 645 | 647 | 0.2 | 1.5 |
| 04d | baseline | in1024_out512_c64 | 3 | 1905 | 1848 | 1907 | 1.8 | 1.7 |
| 04d | kv_cache_2gib | in1024_out512_c64 | 3 | 1670 | 1631 | 1672 | 1.4 | 0.9 |

Throughput CV across 23 groups: max 3.3%, median 0.7%. p95 TTFT CV: max 10.2%, median 2.9%.

### Same config + workload in different experiments

- `baseline` in512_out128_c64 in experiments 02, 04a (per-experiment medians 02: 1804 tok/s; 04a: 1851 tok/s; gap 2.6%). Pooled n=6: median 1831, min 1758, max 1885 tok/s, CV 2.4%, p95 TTFT CV 3.9%.

