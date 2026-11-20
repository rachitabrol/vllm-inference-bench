"""Config loading (with `base:` inheritance), `vllm serve` flag rendering, and job building.

Pure Python + PyYAML. Safe to import with no GPU, no Modal credentials and no vLLM installed.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

# config key -> `vllm serve` flag. Order here = order on the command line.
# Every flag below was confirmed in vllm 0.34.0 (vllm/engine/arg_utils.py).
# Booleans are rendered as --flag / --no-flag (argparse.BooleanOptionalAction in vLLM).
SERVE_FLAGS: dict[str, str] = {
    "dtype": "--dtype",
    "max_model_len": "--max-model-len",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "kv_cache_memory_bytes": "--kv-cache-memory-bytes",
    "max_num_seqs": "--max-num-seqs",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "enable_prefix_caching": "--enable-prefix-caching",
    "enable_chunked_prefill": "--enable-chunked-prefill",
    "seed": "--seed",
}
# Engine flags that baseline deliberately leaves unset (vLLM then sizes the KV cache from
# gpu_memory_utilization). Only variants set them.
OPTIONAL_SERVE_KEYS = {"kv_cache_memory_bytes"}
# keys that are not engine flags
META_KEYS = {"name", "base", "model"}


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_config(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load configs/<name>.yaml, resolving `base:` inheritance. Returns a flat dict (no `base`)."""
    config_dir = config_dir or CONFIG_DIR
    chain: list[str] = []
    merged: dict[str, Any] = {}
    cur: str | None = name
    layers: list[dict[str, Any]] = []
    while cur is not None:
        if cur in chain:
            raise ValueError(f"config inheritance cycle: {' -> '.join([*chain, cur])}")
        chain.append(cur)
        path = config_dir / f"{cur}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"config '{cur}' not found at {path}")
        layer = _read_yaml(path)
        layers.append(layer)
        cur = layer.get("base")
    for layer in reversed(layers):  # root base first, child last
        merged.update(layer)
    merged.pop("base", None)
    merged["name"] = name
    unknown = set(merged) - set(SERVE_FLAGS) - META_KEYS
    if unknown:
        raise ValueError(f"config '{name}' has unknown keys {sorted(unknown)}")
    if "model" not in merged:
        raise ValueError(f"config '{name}' has no model")
    return merged


def render_serve_flags(cfg: dict[str, Any]) -> list[str]:
    """Render engine args as `vllm serve` CLI flags (without the model or host/port)."""
    out: list[str] = []
    for key, flag in SERVE_FLAGS.items():
        if key not in cfg:
            continue
        val = cfg[key]
        if isinstance(val, bool):
            out.append(flag if val else "--no-" + flag.removeprefix("--"))
        else:
            out += [flag, str(val)]
    return out


def flag_dict(cfg: dict[str, Any]) -> dict[str, str]:
    """Flags as {flag-name: rendered-value} so two configs can be diffed. Booleans -> 'true'/'false'."""
    d: dict[str, str] = {}
    for key, flag in SERVE_FLAGS.items():
        if key in cfg:
            v = cfg[key]
            d[flag] = str(v).lower() if isinstance(v, bool) else str(v)
    return d


def diff_flags(a: dict[str, Any], b: dict[str, Any]) -> dict[str, tuple[str | None, str | None]]:
    """Engine flags that differ between two configs: {flag: (a_value, b_value)}."""
    fa, fb = flag_dict(a), flag_dict(b)
    return {k: (fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)}


# ---------------------------------------------------------------- workloads / jobs


@dataclass(frozen=True)
class Workload:
    """One `vllm bench serve` invocation."""

    dataset: str = "random"  # "random" | "prefix_repetition"
    input_len: int = 512  # for prefix_repetition this is prefix_len + suffix_len
    output_len: int = 128
    concurrency: int = 1
    num_prompts: int = 200
    seed: int = 42
    warmup_requests: int = 10
    prefix_len: int = 0  # random: --random-prefix-len ; prefix_repetition: shared prefix tokens
    suffix_len: int = 0  # prefix_repetition only
    num_prefixes: int = 1  # prefix_repetition only

    @property
    def id(self) -> str:
        if self.dataset == "prefix_repetition":
            return (
                f"prefix{self.prefix_len}x{self.num_prefixes}_sfx{self.suffix_len}"
                f"_out{self.output_len}_c{self.concurrency}"
            )
        pre = f"_pre{self.prefix_len}" if self.prefix_len else ""
        return f"in{self.input_len}_out{self.output_len}{pre}_c{self.concurrency}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "id": self.id}


def _expand_workloads(entries: list[dict[str, Any]], defaults: dict[str, Any]) -> list[Workload]:
    out: list[Workload] = []
    for entry in entries:
        entry = dict(entry)
        sweep = entry.pop("sweep", {}) or {}
        keys = list(sweep)
        for combo in itertools.product(*(sweep[k] for k in keys)) if keys else [()]:
            w = {**defaults, **entry, **dict(zip(keys, combo, strict=True))}
            out.append(_make_workload(w, defaults))
    return out


def _make_workload(w: dict[str, Any], defaults: dict[str, Any]) -> Workload:
    conc = int(w.get("concurrency", 1))
    if "num_prompts" not in w:
        w["num_prompts"] = max(
            int(defaults.get("min_prompts", 200)),
            int(defaults.get("prompts_per_concurrency", 4)) * conc,
        )
    fields = set(Workload.__dataclass_fields__)
    extra = set(w) - fields - {"min_prompts", "prompts_per_concurrency"}
    if extra:
        raise ValueError(f"unknown workload keys {sorted(extra)}")
    kwargs = {k: v for k, v in w.items() if k in fields}
    if kwargs.get("dataset") == "prefix_repetition":
        kwargs["input_len"] = int(kwargs.get("prefix_len", 0)) + int(kwargs.get("suffix_len", 0))
    return Workload(**kwargs)


def load_experiments(config_dir: Path | None = None) -> dict[str, Any]:
    return _read_yaml((config_dir or CONFIG_DIR) / "experiments.yaml")


def select_experiments(spec: str, available: list[str], smoke: bool, smoke_ids: list[str]) -> list[str]:
    """Parse the `experiments` input ('all' or '01,02,04a'). Smoke mode forces the smoke subset."""
    if smoke:
        return list(smoke_ids)
    spec = spec.strip()
    if spec in ("", "all"):
        return list(available)
    chosen = [s.strip() for s in spec.split(",") if s.strip()]
    # allow prefix selection: "04" -> 04a..04d
    resolved: list[str] = []
    for c in chosen:
        matches = [a for a in available if a == c or (len(c) == 2 and a.startswith(c))]
        if not matches:
            raise ValueError(f"unknown experiment '{c}'; available: {available}")
        resolved += [m for m in matches if m not in resolved]
    return resolved


def build_jobs(
    spec: str = "all",
    smoke: bool = False,
    config_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Resolve the experiment matrix into a list of JSON-serialisable jobs.

    One job = one (experiment, config) pair = one fresh container + one fresh `vllm serve`,
    running that experiment's workloads sequentially against the same server.
    """
    exp_file = load_experiments(config_dir)
    defaults = exp_file.get("defaults", {})
    exps = exp_file["experiments"]
    smoke_cfg = exp_file.get("smoke", {})
    ids = select_experiments(spec, list(exps), smoke, smoke_cfg.get("experiments", ["01", "02"]))

    jobs: list[dict[str, Any]] = []
    for exp_id in ids:
        exp = exps[exp_id]
        workloads = _expand_workloads(exp["workloads"], defaults)
        if smoke:
            workloads = _smoke_workloads(workloads, smoke_cfg)
        for cfg_name in exp["configs"]:
            cfg = load_config(cfg_name, config_dir)
            if smoke:
                cfg["model"] = smoke_cfg["model"]
            jobs.append(
                {
                    "experiment": exp_id,
                    "title": exp.get("title", ""),
                    "config_name": cfg_name,
                    "config": cfg,
                    "model": cfg["model"],
                    "serve_flags": render_serve_flags(cfg),
                    "workloads": [w.to_dict() for w in workloads],
                    "smoke": smoke,
                }
            )
    return jobs


def _smoke_workloads(workloads: list[Workload], smoke_cfg: dict[str, Any]) -> list[Workload]:
    """Smoke: tiny prompt count, only concurrency in smoke_cfg['concurrency'] (deduplicated)."""
    allowed = set(smoke_cfg.get("concurrency", [1, 4]))
    n = int(smoke_cfg.get("num_prompts", 20))
    out: list[Workload] = []
    seen: set[str] = set()
    for w in workloads:
        if w.concurrency not in allowed:
            continue
        d = {**asdict(w), "num_prompts": n}
        sw = Workload(**d)
        if sw.id not in seen:
            seen.add(sw.id)
            out.append(sw)
    return out
