"""Build `vllm bench serve` commands. Stdlib only.

Every flag used here was confirmed in vllm 0.34.0 (vllm/benchmarks/serve.py and
vllm/benchmarks/datasets/datasets.py). `REQUIRED_BENCH_FLAGS` is re-checked against
`vllm bench serve --help` inside the container before any GPU time is spent.
"""

from __future__ import annotations

from typing import Any

PERCENTILE_METRICS = "ttft,tpot,itl,e2el"
METRIC_PERCENTILES = "50,90,95,99"

# Flags that may be emitted by build_bench_command (used for the in-container sanity check).
REQUIRED_BENCH_FLAGS = [
    "--base-url",
    "--model",
    "--dataset-name",
    "--num-prompts",
    "--seed",
    "--max-concurrency",
    "--num-warmups",
    "--ignore-eos",
    "--temperature",
    "--save-result",
    "--result-dir",
    "--result-filename",
    "--percentile-metrics",
    "--metric-percentiles",
    "--disable-tqdm",
    "--random-input-len",
    "--random-output-len",
    "--random-prefix-len",
    "--prefix-repetition-prefix-len",
    "--prefix-repetition-suffix-len",
    "--prefix-repetition-num-prefixes",
    "--prefix-repetition-output-len",
]

REQUIRED_SERVE_FLAGS = [
    "--host",
    "--port",
    "--dtype",
    "--max-model-len",
    "--gpu-memory-utilization",
    "--kv-cache-memory-bytes",
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--enable-prefix-caching",
    "--enable-chunked-prefill",
    "--seed",
]


def build_bench_command(
    model: str,
    workload: dict[str, Any],
    result_dir: str,
    result_filename: str,
    port: int = 8000,
) -> list[str]:
    """Return the argv for one measured run.

    `--ignore-eos` is passed explicitly so every request generates exactly output_len tokens,
    regardless of dataset. `--temperature 0` pins greedy decoding (vLLM bench
    no longer sends it by default), so runs are reproducible.
    Warm-up requests are sent by vLLM itself (`--num-warmups`) and are
    not part of the measured window.
    """
    ds = workload["dataset"]
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model",
        model,
        "--dataset-name",
        ds,
        "--num-prompts",
        str(workload["num_prompts"]),
        "--seed",
        str(workload["seed"]),
        "--max-concurrency",
        str(workload["concurrency"]),
        "--num-warmups",
        str(workload["warmup_requests"]),
        "--ignore-eos",
        "--temperature",
        "0",
    ]
    if ds == "random":
        cmd += [
            "--random-input-len",
            str(workload["input_len"]),
            "--random-output-len",
            str(workload["output_len"]),
        ]
        if workload.get("prefix_len"):
            cmd += ["--random-prefix-len", str(workload["prefix_len"])]
    elif ds == "prefix_repetition":
        cmd += [
            "--prefix-repetition-prefix-len",
            str(workload["prefix_len"]),
            "--prefix-repetition-suffix-len",
            str(workload["suffix_len"]),
            "--prefix-repetition-num-prefixes",
            str(workload["num_prefixes"]),
            "--prefix-repetition-output-len",
            str(workload["output_len"]),
        ]
    else:
        raise ValueError(f"unsupported dataset '{ds}'")
    cmd += [
        "--percentile-metrics",
        PERCENTILE_METRICS,
        "--metric-percentiles",
        METRIC_PERCENTILES,
        "--disable-tqdm",
        "--save-result",
        "--result-dir",
        result_dir,
        "--result-filename",
        result_filename,
    ]
    return cmd


def build_serve_command(model: str, serve_flags: list[str], port: int = 8000) -> list[str]:
    return ["vllm", "serve", model, "--host", "127.0.0.1", "--port", str(port), *serve_flags]


def missing_flags(help_text: str, flags: list[str]) -> list[str]:
    """Flags (from `flags`) that do not appear in a --help dump."""
    return [f for f in flags if f not in help_text]
