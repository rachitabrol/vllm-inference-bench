"""Background sampler: nvidia-smi + vLLM /metrics every 0.5 s. Stdlib only.

Metric names vary between vLLM versions, so /metrics lines are matched by regex and anything
missing is reported as None instead of raising. In vllm 0.34.0 the relevant names are
vllm:kv_cache_usage_perc, vllm:num_requests_waiting, vllm:num_preemptions(_total),
vllm:prefix_cache_queries(_total), vllm:prefix_cache_hits(_total).
"""

from __future__ import annotations

import re
import subprocess
import threading
import urllib.request
from typing import Any, Self

NVSMI_CMD = [
    "nvidia-smi",
    "--query-gpu=memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
]

# (key, regex on the metric name, kind). Regexes are anchored to avoid e.g. *_by_reason, external_*.
_GAUGES = {
    "kv_cache_usage": re.compile(r"^vllm:.*cache_usage.*$"),
    "num_waiting": re.compile(r"^vllm:num_requests_waiting$"),
}
_COUNTERS = {
    "preemptions": re.compile(r"^vllm:num_preemptions(_total)?$"),
    "prefix_cache_queries": re.compile(r"^vllm:prefix_cache_queries(_total)?$"),
    "prefix_cache_hits": re.compile(r"^vllm:prefix_cache_hits(_total)?$"),
}
_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|[+-]?Inf)\s*$")


def parse_nvidia_smi(text: str) -> tuple[float, float, float] | None:
    """First GPU line -> (mem_used_mib, mem_total_mib, util_pct). None if unparsable."""
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                return float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                return None
    return None


def parse_metrics(text: str) -> dict[str, float]:
    """Reduce a Prometheus text exposition to one value per telemetry key (gauges: max, counters: sum)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, _labels, raw = m.groups()
        try:
            val = float(raw)
        except ValueError:
            continue
        for key, rx in _GAUGES.items():  # gauges: max across series
            if rx.match(name):
                out[key] = max(out.get(key, val), val)
        for key, rx in _COUNTERS.items():  # counters: sum across series
            if rx.match(name):
                out[key] = out.get(key, 0.0) + val
    return out


def summarize(
    gpu_samples: list[tuple[float, float, float]],
    metric_samples: list[dict[str, float]],
) -> dict[str, Any]:
    """Collapse raw samples into the telemetry record stored next to each result."""

    def col(i: int) -> list[float]:
        return [s[i] for s in gpu_samples]

    def series(key: str) -> list[float]:
        return [m[key] for m in metric_samples if key in m]

    kv = series("kv_cache_usage")
    wait = series("num_waiting")
    tel: dict[str, Any] = {
        "gpu_samples": len(gpu_samples),
        "metric_samples": len(metric_samples),
        "gpu_mem_peak_mib": max(col(0)) if gpu_samples else None,
        "gpu_mem_mean_mib": sum(col(0)) / len(gpu_samples) if gpu_samples else None,
        "gpu_mem_total_mib": gpu_samples[0][1] if gpu_samples else None,
        "gpu_util_mean": sum(col(2)) / len(gpu_samples) if gpu_samples else None,
        "gpu_util_peak": max(col(2)) if gpu_samples else None,
        # vLLM reports a 0-1 fraction; stored here as percent
        "kv_cache_usage_peak": 100.0 * max(kv) if kv else None,
        "kv_cache_usage_mean": 100.0 * sum(kv) / len(kv) if kv else None,
        "num_waiting_peak": max(wait) if wait else None,
        "preemptions": None,
        "prefix_cache_queries": None,
        "prefix_cache_hits": None,
        "prefix_cache_hit_rate": None,
    }
    for key in ("preemptions", "prefix_cache_queries", "prefix_cache_hits"):
        s = series(key)
        if len(s) >= 2:  # counters: delta over the sampling window
            tel[key] = max(s[-1] - s[0], 0.0)
    q, h = tel["prefix_cache_queries"], tel["prefix_cache_hits"]
    if q is not None and h is not None and q > 0:
        tel["prefix_cache_hit_rate"] = h / q
    tel["metrics_seen"] = sorted({k for m in metric_samples for k in m})
    return tel


class Telemetry:
    """Context manager: `with Telemetry(port) as t: ...; t.summary`."""

    def __init__(self, port: int = 8000, interval_s: float = 0.5):
        self.port = port
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu: list[tuple[float, float, float]] = []
        self._metrics: list[dict[str, float]] = []
        self.summary: dict[str, Any] = {}

    def _sample_once(self) -> None:
        try:
            out = subprocess.run(NVSMI_CMD, capture_output=True, text=True, timeout=5, check=False).stdout
            g = parse_nvidia_smi(out)
            if g:
                self._gpu.append(g)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/metrics", timeout=2) as r:
                self._metrics.append(parse_metrics(r.read().decode("utf-8", "replace")))
        except OSError:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._sample_once()  # final counter snapshot
        self.summary = summarize(self._gpu, self._metrics)
        return self.summary

    def __enter__(self) -> Self:
        self._sample_once()  # baseline counter snapshot before the run
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
