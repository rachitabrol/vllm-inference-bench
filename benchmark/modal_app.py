"""Modal app: one GPU container per (experiment, config) job.

Local side (`modal run benchmark/modal_app.py --experiments all --gpu L4 --smoke`) resolves the
experiment matrix, fans jobs out with `run_config.with_options(gpu=...).map(...)`, and writes one
JSON per workload under results/raw/<run_id>/. Importing this module has no side effects and needs
no Modal credentials.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

from benchmark import config as cfgmod
from benchmark.server import VllmServer
from benchmark.telemetry import Telemetry
from benchmark.workloads import (
    REQUIRED_BENCH_FLAGS,
    REQUIRED_SERVE_FLAGS,
    build_bench_command,
    build_serve_command,
    missing_flags,
)

# Pinned: latest stable on PyPI when this repo was written (2026-11-20). Bumping it can rename
# CLI flags; the smoke run prints `--help` and run_config() verifies every flag before serving.
VLLM_VERSION = "0.34.0"
GPU_CHOICES = ["T4", "L4", "A10G", "A100-40GB", "H100"]
HF_CACHE_DIR = "/root/.cache/huggingface"
PORT = 8000
MAX_PARALLEL_CONTAINERS = 8
FUNCTION_TIMEOUT_S = 1800

app = modal.App("vllm-inference-bench")

hf_cache = modal.Volume.from_name("vllm-bench-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "pyyaml", "huggingface_hub")
    .env({"HF_HOME": HF_CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "0", "VLLM_LOGGING_LEVEL": "INFO"})
    .add_local_python_source("benchmark")
)


def _run(cmd: list[str], timeout: int = 300) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    return (p.stdout or "") + (p.stderr or "")


def _help(cmd: list[str]) -> str:
    """Full --help text. vLLM prints only group names for plain --help; `--help=all` lists every flag."""
    out = _run([*cmd, "--help=all"])
    return out if len(out) > 500 else _run([*cmd, "--help"])


@app.function(
    image=image,
    gpu="L4",  # default only; the local entrypoint overrides it via .with_options(gpu=...)
    timeout=FUNCTION_TIMEOUT_S,
    volumes={HF_CACHE_DIR: hf_cache},
    single_use_containers=True,  # fresh container (and GPU state) per job
    max_containers=MAX_PARALLEL_CONTAINERS,
)
def run_config(job: dict[str, Any], run_id: str, gpu: str, print_help: bool = False) -> dict[str, Any]:
    """Start `vllm serve` for job['config'], run every workload with `vllm bench serve`, return JSON."""
    import importlib.metadata as md

    vllm_version = md.version("vllm")
    out_dir = Path("/tmp/bench")
    out_dir.mkdir(parents=True, exist_ok=True)

    serve_help = _help(["vllm", "serve"])
    bench_help = _help(["vllm", "bench", "serve"])
    if print_help:
        print(f"===== vllm {vllm_version}: `vllm serve --help` (head) =====")
        print("\n".join(serve_help.splitlines()[:150]))
        print(f"===== vllm {vllm_version}: `vllm bench serve --help` (head) =====")
        print("\n".join(bench_help.splitlines()[:150]))

    # Fail before paying for model load if a flag we rely on does not exist in this vLLM.
    bad = missing_flags(serve_help, REQUIRED_SERVE_FLAGS) + missing_flags(bench_help, REQUIRED_BENCH_FLAGS)
    if bad:
        raise RuntimeError(f"flags not found in vllm {vllm_version} --help: {bad}")

    model = job["model"]
    serve_cmd = build_serve_command(model, job["serve_flags"], PORT)
    print("SERVE:", " ".join(serve_cmd))
    server = VllmServer(serve_cmd, out_dir / "server.log", PORT)
    records: list[dict[str, Any]] = []
    t_start = time.time()
    try:
        server.start()
        startup_s = server.wait_healthy(timeout_s=900)
        print(f"server healthy after {startup_s:.0f}s")
        hf_cache.commit()  # persist downloaded weights for later containers

        for wl in job["workloads"]:
            rep = job.get("repeat", 0)
            fname = cfgmod.result_filename(job["experiment"], job["config_name"], wl["id"], rep)
            cmd = build_bench_command(model, wl, str(out_dir), fname, PORT)
            print("BENCH:", " ".join(cmd))
            t0 = time.time()
            with Telemetry(PORT, 0.5) as tel:
                p = subprocess.run(cmd, capture_output=True, text=True, check=False)
            print(p.stdout[-3000:])
            if p.returncode != 0:
                print(p.stderr[-3000:])
                records.append(
                    {
                        "workload": wl,
                        "filename": fname,
                        "error": f"bench exit {p.returncode}",
                        "stderr_tail": p.stderr[-2000:],
                    }
                )
                continue
            bench = json.loads((out_dir / fname).read_text())
            records.append(
                {
                    "workload": wl,
                    "filename": fname,
                    "bench_command": cmd,
                    "wall_s": time.time() - t0,
                    "bench": bench,
                    "telemetry": tel.summary,
                }
            )
    finally:
        log_tail = server.log_tail(80)
        server.stop()

    return {
        "run_id": run_id,
        "experiment": job["experiment"],
        "config_name": job["config_name"],
        "config": job["config"],
        "serve_command": serve_cmd,
        "gpu": gpu,
        "vllm_version": vllm_version,
        "model": model,
        "smoke": job.get("smoke", False),
        "repeat": job.get("repeat", 0),
        "total_s": time.time() - t_start,
        "server_log_tail": log_tail,
        "help": {"serve": serve_help, "bench": bench_help} if print_help else None,
        "records": records,
    }


def _write_results(run_dir: Path, result: dict[str, Any]) -> int:
    """One JSON per workload: {meta, bench, telemetry}. Returns number of files written."""
    n = 0
    meta_base = {
        "run_id": result["run_id"],
        "experiment": result["experiment"],
        "config": result["config_name"],
        "config_values": result["config"],
        "gpu": result["gpu"],
        "vllm_version": result["vllm_version"],
        "model": result["model"],
        "smoke": result["smoke"],
        "repeat": result.get("repeat", 0),
        "serve_command": result["serve_command"],
    }
    for rec in result["records"]:
        if "bench" not in rec:
            (run_dir / f"FAILED_{rec['filename']}").write_text(json.dumps({**meta_base, **rec}, indent=2))
            continue
        doc = {
            "meta": {**meta_base, "workload": rec["workload"], "bench_command": rec["bench_command"]},
            "bench": rec["bench"],
            "telemetry": rec["telemetry"],
        }
        (run_dir / rec["filename"]).write_text(json.dumps(doc, indent=2))
        n += 1
    log = run_dir / "logs"
    log.mkdir(exist_ok=True)
    tail_name = f"{result['experiment']}_{result['config_name']}_r{result.get('repeat', 0)}_server_tail.log"
    (log / tail_name).write_text(result["server_log_tail"])
    return n


@app.local_entrypoint()
def main(
    experiments: str = "all", gpu: str = "L4", smoke: bool = False, run_id: str = "", repeats: int = 3
) -> None:
    if gpu not in GPU_CHOICES:
        raise SystemExit(f"--gpu must be one of {GPU_CHOICES}")
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + f"-{gpu}" + ("-smoke" if smoke else "")
    jobs = cfgmod.expand_repeats(cfgmod.build_jobs(experiments, smoke=smoke), repeats, smoke=smoke)
    n_bench = sum(len(j["workloads"]) for j in jobs)
    n_rep = 1 if smoke else repeats
    print(f"run_id={run_id} gpu={gpu} smoke={smoke} repeats={n_rep}: {len(jobs)} servers, {n_bench} runs")

    run_dir = Path("results/raw") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # GPU chosen at runtime: with_options returns a copy of the Function on a different GPU.
    # volumes/timeout/max_containers are repeated on purpose: with_options() may replace (not merge) them.
    fn = run_config.with_options(
        gpu=gpu,
        volumes={HF_CACHE_DIR: hf_cache},
        timeout=FUNCTION_TIMEOUT_S,
        max_containers=MAX_PARALLEL_CONTAINERS,
    )
    help_flags = [smoke and i == 0 for i in range(len(jobs))]  # dump --help once, from the first job
    written = 0
    failures = 0
    for result in fn.map(
        jobs, [run_id] * len(jobs), [gpu] * len(jobs), help_flags, order_outputs=False, return_exceptions=True
    ):
        if isinstance(result, BaseException):
            failures += 1
            print(f"JOB FAILED: {result!r}")
            continue
        if result.get("help"):
            for kind, text in result["help"].items():
                (run_dir / f"help_{kind}.txt").write_text(text)
        written += _write_results(run_dir, result)
    print(f"wrote {written} result files to {run_dir} ({failures} job failures)")
    info = {"run_id": run_id, "gpu": gpu, "smoke": smoke, "experiments": experiments, "repeats": n_rep}
    (run_dir / "RUN_INFO.json").write_text(json.dumps(info, indent=2))
    if failures or written == 0:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    print("Run with: modal run benchmark/modal_app.py --help", os.getcwd())
