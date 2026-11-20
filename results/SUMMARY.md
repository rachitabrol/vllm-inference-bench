# Benchmark summary

- Runs: 44703863719-1-L4
- GPU: L4  |  vLLM: 0.34.0
- Model(s): Qwen/Qwen2.5-1.5B-Instruct
- SLO: p95 TTFT <= 500 ms, p95 TPOT <= 50 ms, error rate <= 0
- GPU prices checked: 2026-11-20. Cost = price/hr / (output tok/s * 3600) * 1e6 (GPU time only).
- Warm-up: 10 discarded requests per run (`--num-warmups`). Telemetry window includes warm-up.

## 01 Baseline single-user latency

| config | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 512 | 128 | 73 | 0.57 | 36 | 39 | 13.4 | 13.5 | 1747 | 0 | 3.024 | yes |

## 02 Concurrency sweep

| config | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 512 | 128 | 73 | 0.57 | 39 | 48 | 13.5 | 13.5 | 1773 | 0 | 3.037 | yes |
| baseline | 4 | 512 | 128 | 275 | 2.15 | 45 | 107 | 13.9 | 16.5 | 2146 | 0 | 0.807 | yes |
| baseline | 8 | 512 | 128 | 545 | 4.26 | 51 | 89 | 14.2 | 14.7 | 1963 | 0 | 0.407 | yes |
| baseline | 16 | 512 | 128 | 999 | 7.80 | 64 | 120 | 15.0 | 15.2 | 2001 | 0 | 0.223 | yes |
| baseline | 32 | 512 | 128 | 1603 | 12.52 | 132 | 191 | 17.6 | 17.9 | 2401 | 0 | 0.139 | yes |
| baseline | 64 | 512 | 128 | 2438 | 19.05 | 250 | 832 | 22.0 | 32.1 | 4312 | 0 | 0.091 | NO |

## 03 Prefill vs decode (input/output length)

| config | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 8 | 128 | 64 | 547 | 8.55 | 54 | 94 | 13.8 | 14.1 | 954 | 0 | 0.406 | yes |
| baseline | 8 | 128 | 512 | 576 | 1.13 | 48 | 67 | 13.8 | 13.9 | 7134 | 0 | 0.385 | yes |
| baseline | 8 | 1024 | 64 | 382 | 5.96 | 237 | 401 | 17.5 | 19.9 | 1394 | 0 | 0.582 | yes |
| baseline | 8 | 1024 | 512 | 541 | 1.06 | 52 | 77 | 14.7 | 14.7 | 7584 | 0 | 0.411 | yes |
| baseline | 8 | 4096 | 64 | 178 | 2.78 | 708 | 975 | 34.3 | 42.1 | 3162 | 0 | 1.251 | NO |
| baseline | 8 | 4096 | 512 | 375 | 0.73 | 807 | 846 | 19.8 | 20.5 | 11108 | 0 | 0.592 | NO |

## 04a max_num_seqs

| config | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | GPU util % | KV peak % | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| max_seqs_8 | 64 | 512 | 128 | 504 | 3.94 | 14365 | 14431 | 14.9 | 15.6 | 16266 | 0 | 78 | 1 | 0.441 | NO |
| max_seqs_32 | 64 | 512 | 128 | 1343 | 10.49 | 3241 | 3685 | 22.2 | 22.8 | 6447 | 0 | 57 | 3 | 0.165 | NO |
| baseline | 64 | 512 | 128 | 1869 | 14.60 | 382 | 1635 | 31.0 | 32.7 | 5526 | 0 | 50 | 7 | 0.119 | NO |

## 04b max_num_batched_tokens

| config | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | ITL p95 | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| batched_tokens_512 | 16 | 4096 | 128 | 288 | 2.25 | 636 | 1617 | 50.2 | 52.0 | 7333 | 0 | 56.6 | 0.771 | NO |
| baseline | 16 | 4096 | 128 | 297 | 2.32 | 875 | 1531 | 47.0 | 49.5 | 7342 | 0 | 147.5 | 0.749 | NO |
| batched_tokens_8192 | 16 | 4096 | 128 | 291 | 2.28 | 1914 | 2638 | 39.8 | 51.2 | 7374 | 0 | 28.0 | 0.763 | NO |

## 04c Prefix caching

| config | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | prefix hit rate | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no_prefix_cache | 16 | 1152 | 128 | 613 | 4.79 | 430 | 590 | 22.1 | 24.3 | 3389 | 0 | - | 0.362 | NO |
| baseline | 16 | 1152 | 128 | 990 | 7.73 | 109 | 174 | 14.8 | 15.3 | 2020 | 0 | 0.89 | 0.225 | yes |

## 04d KV cache capacity (kv_cache_memory_bytes)

| config | conc | in | out | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | failed | KV peak % | preemptions | $/1M out | SLO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 64 | 1024 | 512 | 1871 | 3.65 | 586 | 3414 | 33.2 | 33.5 | 20348 | 0 | 17 | 0 | 0.119 | NO |
| kv_cache_2gib | 64 | 1024 | 512 | 1670 | 3.26 | 1653 | 6961 | 30.6 | 45.7 | 26142 | 0 | 100 | 70 | 0.133 | NO |

## 05 Cost / performance under SLO

Best config meeting the SLO (max output tok/s = min $/1M tokens): **baseline** at concurrency **32** (experiment 02, L4): 1603 output tok/s, p95 TTFT 191 ms, p95 TPOT 17.9 ms, **$0.139 per 1M output tokens**.

