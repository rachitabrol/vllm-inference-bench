# Benchmark summary

- Runs: 44708914259-1-L4
- Repeats: n = 3..3 runs per point (tables show medians; throughput as median (min–max) when n>1)
- GPU: L4  |  vLLM: 0.34.0
- Model(s): Qwen/Qwen2.5-1.5B-Instruct
- SLO: p95 TTFT <= 500 ms, p95 TPOT <= 50 ms, error rate <= 0
- GPU prices checked: 2026-11-20. Cost = price/hr / (output tok/s * 3600) * 1e6 (GPU time only).
- Warm-up: 10 discarded requests per run (`--num-warmups`). Telemetry window includes warm-up.

## 01 Baseline single-user latency

| config | n | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 1 | 512 | 128 | 73 (73–74) | 0.57 | 38 | 39 | 13.4 | 13.5 | 1752 | 0 | 3.031 | yes |

## 02 Concurrency sweep

| config | n | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 1 | 512 | 128 | 73 (73–73) | 0.57 | 39 | 40 | 13.5 | 13.5 | 1755 | 0 | 3.040 | yes |
| baseline | 3 | 4 | 512 | 128 | 284 (283–284) | 2.22 | 45 | 49 | 13.8 | 13.9 | 1812 | 0 | 0.783 | yes |
| baseline | 3 | 8 | 512 | 128 | 550 (550–551) | 4.30 | 49 | 70 | 14.2 | 14.3 | 1871 | 0 | 0.404 | yes |
| baseline | 3 | 16 | 512 | 128 | 999 (996–1001) | 7.80 | 68 | 120 | 15.0 | 15.2 | 2005 | 0 | 0.223 | yes |
| baseline | 3 | 32 | 512 | 128 | 1608 (1604–1610) | 12.56 | 111 | 209 | 17.5 | 17.8 | 2385 | 0 | 0.138 | yes |
| baseline | 3 | 64 | 512 | 128 | 2478 (2458–2478) | 19.36 | 244 | 926 | 21.8 | 31.7 | 4311 | 0 | 0.090 | NO |

## 03 Prefill vs decode (input/output length)

| config | n | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | ITL p99 | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 8 | 128 | 64 | 539 (537–541) | 8.43 | 64 | 89 | 13.9 | 14.4 | 961 | 0 | 22.7 | 0.412 | yes |
| baseline | 3 | 8 | 128 | 512 | 572 (569–573) | 1.12 | 48 | 64 | 13.9 | 13.9 | 7174 | 0 | 17.4 | 0.388 | yes |
| baseline | 3 | 8 | 1024 | 64 | 378 (369–378) | 5.91 | 287 | 351 | 17.0 | 20.1 | 1364 | 0 | 112.7 | 0.588 | yes |
| baseline | 3 | 8 | 1024 | 512 | 539 (537–539) | 1.05 | 54 | 72 | 14.8 | 14.8 | 7628 | 0 | 18.2 | 0.412 | yes |
| baseline | 3 | 8 | 4096 | 64 | 169 (169–174) | 2.65 | 757 | 988 | 35.6 | 44.1 | 3356 | 0 | 158.3 | 1.312 | NO |
| baseline | 3 | 8 | 4096 | 512 | 366 (365–369) | 0.71 | 894 | 948 | 20.3 | 20.9 | 11267 | 0 | 149.8 | 0.607 | NO |

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

Best config meeting the SLO (max output tok/s = min $/1M tokens): **baseline** at concurrency **32** (experiment 02, L4): 1608 output tok/s (median of n=3), p95 TTFT 209 ms, p95 TPOT 17.8 ms, **$0.138 per 1M output tokens**.

## Run-to-run variance

One row per (experiment, config, workload); each repeat ran in its own container. CV = sample std / mean.

| exp | config | workload | n | tok/s median | min | max | CV % | p95 TTFT CV % |
|---|---|---|---|---|---|---|---|---|
| 01 | baseline | in512_out128_c1 | 3 | 73 | 73 | 74 | 0.7 | 10.3 |
| 02 | baseline | in512_out128_c1 | 3 | 73 | 73 | 73 | 0.2 | 4.1 |
| 02 | baseline | in512_out128_c4 | 3 | 284 | 283 | 284 | 0.2 | 11.1 |
| 02 | baseline | in512_out128_c8 | 3 | 550 | 550 | 551 | 0.1 | 2.6 |
| 02 | baseline | in512_out128_c16 | 3 | 999 | 996 | 1001 | 0.2 | 3.0 |
| 02 | baseline | in512_out128_c32 | 3 | 1608 | 1604 | 1610 | 0.2 | 23.6 |
| 02 | baseline | in512_out128_c64 | 3 | 2478 | 2458 | 2478 | 0.5 | 19.2 |
| 03 | baseline | in128_out64_c8 | 3 | 539 | 537 | 541 | 0.3 | 5.4 |
| 03 | baseline | in128_out512_c8 | 3 | 572 | 569 | 573 | 0.4 | 6.3 |
| 03 | baseline | in1024_out64_c8 | 3 | 378 | 369 | 378 | 1.4 | 9.4 |
| 03 | baseline | in1024_out512_c8 | 3 | 539 | 537 | 539 | 0.2 | 5.7 |
| 03 | baseline | in4096_out64_c8 | 3 | 169 | 169 | 174 | 1.7 | 6.0 |
| 03 | baseline | in4096_out512_c8 | 3 | 366 | 365 | 369 | 0.6 | 8.1 |
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

Throughput CV across 23 groups: max 3.3%, median 0.6%. p95 TTFT CV: max 23.6%, median 5.3%.

### Same config + workload in different experiments

- `baseline` in512_out128_c64 in experiments 02, 04a (per-experiment medians 02: 2478 tok/s; 04a: 1851 tok/s; gap 33.8%). Pooled n=6: median 2171, min 1850, max 2478 tok/s, CV 15.4%, p95 TTFT CV 29.3%.

