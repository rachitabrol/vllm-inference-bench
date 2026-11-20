"""Raw benchmark JSON -> summary.csv, plots/*.png, SUMMARY.md.

Input : directory (or directories) of JSON files written by benchmark/modal_app.py. Each file is
        {"meta": {...}, "bench": <`vllm bench serve --save-result` JSON>, "telemetry": {...}}.
Output: <out>/summary.csv, <out>/SUMMARY.md, <plots>/*.png

Keys read from the vLLM bench JSON (confirmed in vllm 0.34.0 vllm/benchmarks/serve.py):
  completed, failed, duration, request_throughput, output_throughput, total_token_throughput,
  total_input_tokens, total_output_tokens,
  mean_/median_/p{50,90,95,99}_ + {ttft,tpot,itl,e2el}_ms
  (p-keys exist only for the percentiles requested with --metric-percentiles).

Usage: python analysis/analyze.py results/raw/<run_id> [--out results] [--plots plots]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# ---- consistent plot style: one accent color per series ----------------------------------------
INK = "#1f2430"
GRID = "#d9dce3"
SLO_COLOR = "#c0392b"
NEUTRAL = "#8a93a3"  # baseline / "does not meet SLO"
SERIES = ["#2563eb", "#e07a1f", "#1b9e77", "#9b59b6", "#b8860b", "#d6336c"]

COLUMNS = [
    "run_id", "experiment", "config", "gpu", "vllm_version", "model", "dataset",
    "input_len", "output_len", "concurrency", "num_prompts",
    "completed", "failed", "duration_s", "request_throughput", "output_tok_per_s", "total_tok_per_s",
    "ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
    "tpot_mean_ms", "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms",
    "itl_p50_ms", "itl_p95_ms", "e2el_p50_ms", "e2el_p95_ms",
    "gpu_mem_peak_mib", "gpu_util_mean", "kv_cache_usage_peak", "preemptions", "prefix_cache_hit_rate",
    "gpu_price_per_hr", "usd_per_1m_output_tokens", "meets_slo",
    "workload_id",
]  # fmt: skip

# experiment -> (config key that varies, human title)
VARIANT_KEY = {
    "04a": ("max_num_seqs", "max_num_seqs"),
    "04b": ("max_num_batched_tokens", "max_num_batched_tokens"),
    "04c": ("enable_prefix_caching", "prefix caching"),
    "04d": ("gpu_memory_utilization", "gpu_memory_utilization"),
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
        }
    )


# ---- loading -----------------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Every *.json under the given dirs/files that has meta + bench keys."""
    files: list[Path] = []
    for p in paths:
        files += sorted(p.rglob("*.json")) if p.is_dir() else [p]
    recs = []
    for f in files:
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict) and "meta" in doc and "bench" in doc:
            doc["_path"] = str(f)
            recs.append(doc)
    return recs


def _num(d: dict[str, Any], *keys: str) -> float:
    """First present numeric key, else NaN. (vLLM names p50 'median_*' and also 'p50_*' when requested.)"""
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return math.nan


def error_rate(completed: float, failed: float) -> float:
    total = completed + failed
    return failed / total if total > 0 else math.nan


def usd_per_1m_output_tokens(price_per_hr: float, output_tok_per_s: float) -> float:
    """price_per_hr / (output_tok_per_s * 3600) * 1e6"""
    if not output_tok_per_s or math.isnan(output_tok_per_s) or math.isnan(price_per_hr):
        return math.nan
    return price_per_hr / (output_tok_per_s * 3600.0) * 1e6


def meets_slo(ttft_p95: float, tpot_p95: float, err_rate: float, slo: dict[str, Any]) -> bool:
    """True iff p95 TTFT, p95 TPOT and error rate are all within the SLO. Missing (NaN) values fail."""
    vals = (ttft_p95, tpot_p95, err_rate)
    if any(math.isnan(v) for v in vals):
        return False
    return (
        ttft_p95 <= slo["p95_ttft_ms"]
        and tpot_p95 <= slo["p95_tpot_ms"]
        and err_rate <= slo.get("max_error_rate", 0.0)
    )


def build_row(rec: dict[str, Any], prices: dict[str, float], slo: dict[str, Any]) -> dict[str, Any]:
    m, b, t = rec["meta"], rec["bench"], rec.get("telemetry") or {}
    wl = m.get("workload", {})
    completed, failed = _num(b, "completed"), _num(b, "failed")
    if math.isnan(failed):
        failed = 0.0
    tput = _num(b, "output_throughput")
    price = float(prices.get(m["gpu"], math.nan))
    ttft95, tpot95 = _num(b, "p95_ttft_ms"), _num(b, "p95_tpot_ms")
    return {
        "run_id": m.get("run_id"),
        "experiment": m["experiment"],
        "config": m["config"],
        "gpu": m["gpu"],
        "vllm_version": m.get("vllm_version"),
        "model": m.get("model"),
        "dataset": wl.get("dataset"),
        "input_len": wl.get("input_len"),
        "output_len": wl.get("output_len"),
        "concurrency": wl.get("concurrency"),
        "num_prompts": wl.get("num_prompts"),
        "completed": completed,
        "failed": failed,
        "duration_s": _num(b, "duration"),
        "request_throughput": _num(b, "request_throughput"),
        "output_tok_per_s": tput,
        "total_tok_per_s": _num(b, "total_token_throughput"),
        "ttft_mean_ms": _num(b, "mean_ttft_ms"),
        "ttft_p50_ms": _num(b, "median_ttft_ms", "p50_ttft_ms"),
        "ttft_p95_ms": ttft95,
        "ttft_p99_ms": _num(b, "p99_ttft_ms"),
        "tpot_mean_ms": _num(b, "mean_tpot_ms"),
        "tpot_p50_ms": _num(b, "median_tpot_ms", "p50_tpot_ms"),
        "tpot_p95_ms": tpot95,
        "tpot_p99_ms": _num(b, "p99_tpot_ms"),
        "itl_p50_ms": _num(b, "median_itl_ms", "p50_itl_ms"),
        "itl_p95_ms": _num(b, "p95_itl_ms"),
        "e2el_p50_ms": _num(b, "median_e2el_ms", "p50_e2el_ms"),
        "e2el_p95_ms": _num(b, "p95_e2el_ms"),
        "gpu_mem_peak_mib": t.get("gpu_mem_peak_mib"),
        "gpu_util_mean": t.get("gpu_util_mean"),
        "kv_cache_usage_peak": t.get("kv_cache_usage_peak"),
        "preemptions": t.get("preemptions"),
        "prefix_cache_hit_rate": t.get("prefix_cache_hit_rate"),
        "gpu_price_per_hr": price,
        "usd_per_1m_output_tokens": usd_per_1m_output_tokens(price, tput),
        "meets_slo": meets_slo(ttft95, tpot95, error_rate(completed, failed), slo),
        "workload_id": wl.get("id"),
        "_config_values": m.get("config_values") or {},
        "_fixture": bool(rec.get("_fixture") or m.get("_fixture")),
    }


def build_dataframe(
    recs: list[dict[str, Any]], prices: dict[str, float], slo: dict[str, Any]
) -> pd.DataFrame:
    rows = [build_row(r, prices, slo) for r in recs]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # keep the latest run for each (experiment, config, workload, gpu)
    df = df.sort_values("run_id").drop_duplicates(["experiment", "config", "workload_id", "gpu"], keep="last")
    return df.sort_values(["experiment", "config", "input_len", "output_len", "concurrency"]).reset_index(
        drop=True
    )


# ---- experiment 05 -----------------------------------------------------------------------------


def pick_best_under_slo(df: pd.DataFrame) -> pd.Series | None:
    """05: among 02 + 04* rows that meet the SLO, the one with max output_tok_per_s (= min $/1M tokens
    on a fixed GPU price)."""
    pool = df[df["experiment"].isin(["02"]) | df["experiment"].str.startswith("04")]
    ok = pool[pool["meets_slo"]]
    if ok.empty:
        return None
    return ok.loc[ok["output_tok_per_s"].idxmax()]


# ---- plotting ----------------------------------------------------------------------------------


def _finish(fig: plt.Figure, path: Path, fixture: bool) -> None:
    if fixture:
        fig.text(0.5, 0.5, "FIXTURE DATA - NOT REAL RESULTS", ha="center", va="center", fontsize=26,
                 color="#c0392b", alpha=0.18, rotation=20)  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _slo_line(ax: plt.Axes, y: float, label: str, horizontal: bool = True) -> None:
    kw = {"color": SLO_COLOR, "linestyle": "--", "linewidth": 1.4, "zorder": 1}
    (ax.axhline if horizontal else ax.axvline)(y, **kw)
    if horizontal:
        ax.set_ylim(top=max(ax.get_ylim()[1], y * 1.15))  # keep the SLO line and its label in view
        ax.annotate(label, xy=(0.99, y), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    color=SLO_COLOR, fontsize=8)  # fmt: skip
    else:
        ax.annotate(label, xy=(y, 0.99), xycoords=("data", "axes fraction"), ha="right", va="top",
                    color=SLO_COLOR, fontsize=8, rotation=90)  # fmt: skip


def plot_02(df: pd.DataFrame, slo: dict, plots: Path, fixture: bool) -> list[Path]:
    d = df[df["experiment"] == "02"].sort_values("concurrency")
    if d.empty:
        return []
    made = []
    x = d["concurrency"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, d["output_tok_per_s"], marker="o", color=SERIES[0])
    axes[0].set_ylabel("Output throughput (tokens/s)")
    axes[1].plot(x, d["request_throughput"], marker="o", color=SERIES[1])
    axes[1].set_ylabel("Request throughput (requests/s)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(x))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Max concurrency (requests in flight)")
        ax.set_ylim(bottom=0)
    fig.suptitle("Experiment 02: throughput vs concurrency", fontweight="bold", color=INK)
    p = plots / "02_throughput_vs_concurrency.png"
    _finish(fig, p, fixture)
    made.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, name, lim in (
        (axes[0], "ttft", "TTFT", slo["p95_ttft_ms"]),
        (axes[1], "tpot", "TPOT", slo["p95_tpot_ms"]),
    ):
        ax.plot(x, d[f"{metric}_p50_ms"], marker="o", color=SERIES[0], label="p50")
        ax.plot(x, d[f"{metric}_p95_ms"], marker="s", color=SERIES[1], label="p95")
        _slo_line(ax, lim, f"SLO: p95 {name} <= {lim:g} ms")
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(x))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Max concurrency (requests in flight)")
        ax.set_ylabel(f"{name} (ms)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{name} vs concurrency")
        ax.legend(loc="lower right")
    fig.suptitle("Experiment 02: latency vs concurrency", fontweight="bold", color=INK)
    p = plots / "02_latency_vs_concurrency.png"
    _finish(fig, p, fixture)
    made.append(p)
    return made


def plot_03(df: pd.DataFrame, slo: dict, plots: Path, fixture: bool) -> list[Path]:
    d = df[df["experiment"] == "03"]
    if d.empty:
        return []
    made = []

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (out_len, g) in enumerate(sorted(d.groupby("output_len"))):
        g = g.sort_values("input_len")
        ax.plot(g["input_len"], g["ttft_p50_ms"], marker="o", color=SERIES[i % len(SERIES)],
                label=f"output {int(out_len)} tokens")  # fmt: skip
    _slo_line(ax, slo["p95_ttft_ms"], f"SLO: p95 TTFT <= {slo['p95_ttft_ms']:g} ms")
    ax.set_xscale("log", base=2)
    xs = sorted(d["input_len"].unique())
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Input length (tokens)")
    ax.set_ylabel("p50 TTFT (ms)")
    ax.set_ylim(bottom=0)
    ax.set_title("Experiment 03: TTFT vs input length (concurrency 8)")
    ax.legend(loc="upper left")
    p = plots / "03_ttft_vs_input_len.png"
    _finish(fig, p, fixture)
    made.append(p)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (in_len, g) in enumerate(sorted(d.groupby("input_len"))):
        g = g.sort_values("output_len")
        ax.plot(g["output_len"], g["e2el_p50_ms"] / 1000.0, marker="o", color=SERIES[i % len(SERIES)],
                label=f"input {int(in_len)} tokens")  # fmt: skip
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(d["output_len"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Output length (tokens)")
    ax.set_ylabel("p50 end-to-end latency (s)")
    ax.set_ylim(bottom=0)
    ax.set_title("Experiment 03: end-to-end latency vs output length (concurrency 8)")
    ax.legend(loc="upper left")
    p = plots / "03_e2e_vs_output_len.png"
    _finish(fig, p, fixture)
    made.append(p)
    return made


def _variant_label(row: pd.Series, exp: str) -> str:
    key, _ = VARIANT_KEY[exp]
    val = row["_config_values"].get(key)
    if val is None:
        return row["config"]
    if isinstance(val, bool):
        val = "on" if val else "off"
    return f"{key}\n= {val}" if key != "enable_prefix_caching" else f"prefix caching\n{val}"


def plot_04(df: pd.DataFrame, slo: dict, plots: Path, fixture: bool) -> list[Path]:
    names = {
        "04a": "04a_max_num_seqs_comparison.png",
        "04b": "04b_batched_tokens_comparison.png",
        "04c": "04c_prefix_caching_comparison.png",
        "04d": "04d_kv_capacity_comparison.png",
    }
    extra = {  # 4th panel per experiment
        "04a": ("gpu_util_mean", "Mean GPU utilization (%)", None),
        "04b": ("itl_p95_ms", "p95 inter-token latency (ms)", None),
        "04c": ("prefix_cache_hit_rate", "Prefix cache hit rate (fraction)", None),
        "04d": ("preemptions", "Preemptions during run (count)", None),
    }
    made = []
    for exp, fname in names.items():
        d = df[df["experiment"] == exp]
        if d.empty:
            continue
        key, _title = VARIANT_KEY[exp]
        d = d.copy()
        d["_sort"] = d["_config_values"].map(lambda c, k=key: c.get(k, 0))
        d = d.sort_values("_sort")
        labels = [_variant_label(r, exp) for _, r in d.iterrows()]
        colors = [NEUTRAL if c == "baseline" else SERIES[0] for c in d["config"]]
        panels = [
            ("output_tok_per_s", "Output throughput (tokens/s)", None),
            ("ttft_p95_ms", "p95 TTFT (ms)", (slo["p95_ttft_ms"], f"SLO {slo['p95_ttft_ms']:g} ms")),
            ("tpot_p95_ms", "p95 TPOT (ms)", (slo["p95_tpot_ms"], f"SLO {slo['p95_tpot_ms']:g} ms")),
            extra[exp][:2] + (None,),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        for ax, (col, ylabel, slo_line) in zip(axes, panels, strict=True):
            vals = pd.to_numeric(d[col], errors="coerce").fillna(0)
            ax.bar(range(len(d)), vals, color=colors, width=0.6)
            ax.set_xticks(range(len(d)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(ylabel)
            ax.grid(axis="x", visible=False)
            if slo_line:
                _slo_line(ax, slo_line[0], slo_line[1])
        fig.suptitle(
            f"Experiment {exp}: {d['concurrency'].iloc[0]:g} concurrent requests, "
            f"{VARIANT_KEY[exp][1]} (grey = baseline)",
            fontweight="bold",
            color=INK,
        )
        p = plots / fname
        _finish(fig, p, fixture)
        made.append(p)
    return made


def plot_05(df: pd.DataFrame, slo: dict, plots: Path, fixture: bool) -> list[Path]:
    pool = df[df["experiment"].isin(["02"]) | df["experiment"].str.startswith("04")]
    pool = pool.dropna(subset=["usd_per_1m_output_tokens", "ttft_p95_ms"])
    if pool.empty:
        return []
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ok, bad = pool[pool["meets_slo"]], pool[~pool["meets_slo"]]
    ax.scatter(bad["ttft_p95_ms"], bad["usd_per_1m_output_tokens"], color=NEUTRAL, s=36, label="violates SLO")
    ax.scatter(ok["ttft_p95_ms"], ok["usd_per_1m_output_tokens"], color=SERIES[0], s=42, label="meets SLO")
    best = pick_best_under_slo(df)
    if best is not None:
        ax.scatter([best["ttft_p95_ms"]], [best["usd_per_1m_output_tokens"]], s=170, facecolors="none",
                   edgecolors=SERIES[1], linewidths=2, label="best under SLO")  # fmt: skip
    _slo_line(ax, slo["p95_ttft_ms"], f"SLO: p95 TTFT <= {slo['p95_ttft_ms']:g} ms", horizontal=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p95 TTFT (ms, log scale)")
    ax.set_ylabel("Cost per 1M output tokens (USD, log scale)")
    gpus = ", ".join(sorted(pool["gpu"].unique()))
    ax.set_title(f"Experiment 05: cost vs latency ({gpus}; runs from 02 and 04)")
    ax.legend(loc="upper right")
    p = plots / "05_cost_vs_latency.png"
    _finish(fig, p, fixture)
    return [p]


# ---- markdown ----------------------------------------------------------------------------------


def _fmt(v: Any, nd: int = 1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(df: pd.DataFrame, cols: list[tuple[str, str, int]]) -> str:
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], nd) for c, _, nd in cols) + " |")
    return "\n".join(lines)


def build_summary_md(df: pd.DataFrame, slo: dict, prices_checked: str | None) -> str:
    fixture = bool(df["_fixture"].any())
    gpus = ", ".join(sorted(df["gpu"].unique()))
    versions = ", ".join(sorted(map(str, df["vllm_version"].dropna().unique())))
    out: list[str] = ["# Benchmark summary", ""]
    if fixture:
        out += ["> **FIXTURE DATA - NOT REAL RESULTS.** Generated from test fixtures.", ""]
    out += [
        f"- Runs: {', '.join(sorted(map(str, df['run_id'].dropna().unique())))}",
        f"- GPU: {gpus}  |  vLLM: {versions}",
        f"- Model(s): {', '.join(sorted(map(str, df['model'].dropna().unique())))}",
        f"- SLO: p95 TTFT <= {slo['p95_ttft_ms']:g} ms, p95 TPOT <= {slo['p95_tpot_ms']:g} ms, "
        f"error rate <= {slo.get('max_error_rate', 0):g}",
        f"- GPU prices checked: {prices_checked or 'unknown'}. "
        "Cost = price/hr / (output tok/s * 3600) * 1e6 (GPU time only).",
        "- Warm-up: 10 discarded requests per run (`--num-warmups`). Telemetry window includes warm-up.",
        "",
    ]
    common = [
        ("config", "config", 0), ("concurrency", "conc", 0), ("input_len", "in", 0), ("output_len", "out", 0),
        ("output_tok_per_s", "out tok/s", 0), ("request_throughput", "req/s", 2),
        ("ttft_p50_ms", "TTFT p50", 0), ("ttft_p95_ms", "TTFT p95", 0),
        ("tpot_p50_ms", "TPOT p50", 1), ("tpot_p95_ms", "TPOT p95", 1),
        ("e2el_p95_ms", "E2E p95", 0), ("failed", "failed", 0),
        ("usd_per_1m_output_tokens", "$/1M out", 3), ("meets_slo", "SLO", 0),
    ]  # fmt: skip
    extra = {
        "04a": [("gpu_util_mean", "GPU util %", 0), ("kv_cache_usage_peak", "KV peak %", 0)],
        "04b": [("itl_p95_ms", "ITL p95", 1)],
        "04c": [("prefix_cache_hit_rate", "prefix hit rate", 2)],
        "04d": [("kv_cache_usage_peak", "KV peak %", 0), ("preemptions", "preemptions", 0)],
    }
    titles = {
        "01": "01 Baseline single-user latency",
        "02": "02 Concurrency sweep",
        "03": "03 Prefill vs decode (input/output length)",
        "04a": "04a max_num_seqs",
        "04b": "04b max_num_batched_tokens",
        "04c": "04c Prefix caching",
        "04d": "04d KV cache capacity (gpu_memory_utilization)",
    }
    for exp in sorted(df["experiment"].unique()):
        d = df[df["experiment"] == exp]
        if exp in VARIANT_KEY:
            vk = VARIANT_KEY[exp][0]
            d = d.assign(_k=d["_config_values"].map(lambda c, k=vk: c.get(k, 0))).sort_values("_k")
        cols = common[:-2] + extra.get(exp, []) + common[-2:]
        out += [f"## {titles.get(exp, exp)}", "", md_table(d, cols), ""]

    out += ["## 05 Cost / performance under SLO", ""]
    best = pick_best_under_slo(df)
    if best is None:
        out += ["No run in 02 + 04 met the SLO. Nothing to recommend; relax the SLO or lower load.", ""]
    else:
        out += [
            f"Best config meeting the SLO (max output tok/s = min $/1M tokens): "
            f"**{best['config']}** at concurrency **{int(best['concurrency'])}** "
            f"(experiment {best['experiment']}, {best['gpu']}): "
            f"{best['output_tok_per_s']:.0f} output tok/s, p95 TTFT {best['ttft_p95_ms']:.0f} ms, "
            f"p95 TPOT {best['tpot_p95_ms']:.1f} ms, "
            f"**${best['usd_per_1m_output_tokens']:.3f} per 1M output tokens**.",
            "",
        ]
    return "\n".join(out)


# ---- main --------------------------------------------------------------------------------------


def run(
    inputs: list[Path],
    out_dir: Path,
    plots_dir: Path,
    slo_path: Path | None = None,
    prices_path: Path | None = None,
) -> pd.DataFrame:
    slo = load_yaml(slo_path or REPO / "configs" / "slo.yaml")
    prices_doc = load_yaml(prices_path or REPO / "configs" / "gpu_prices.yaml")
    prices = prices_doc["gpu_usd_per_hour"]
    recs = load_records(inputs)
    if not recs:
        raise SystemExit(f"no result JSON (with meta+bench) found in {[str(p) for p in inputs]}")
    df = build_dataframe(recs, prices, slo)
    fixture = bool(df["_fixture"].any())

    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(out_dir / "summary.csv", index=False)
    for fn in (plot_02, plot_03, plot_04, plot_05):
        fn(df, slo, plots_dir, fixture)
    (out_dir / "SUMMARY.md").write_text(build_summary_md(df, slo, prices_doc.get("checked")) + "\n")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="results/raw/<run_id> dir(s) or JSON file(s)")
    ap.add_argument("--out", type=Path, default=REPO / "results")
    ap.add_argument("--plots", type=Path, default=REPO / "plots")
    ap.add_argument("--slo", type=Path, default=None)
    ap.add_argument("--prices", type=Path, default=None)
    a = ap.parse_args()
    df = run(a.inputs, a.out, a.plots, a.slo, a.prices)
    print(f"{len(df)} rows -> {a.out}/summary.csv, {a.out}/SUMMARY.md, plots in {a.plots}/")


if __name__ == "__main__":
    main()
