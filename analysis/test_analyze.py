"""analyze.py end-to-end on fixture JSON (no GPU, no Modal). Fixture numbers are fake."""

import json
import math
from pathlib import Path

import analyze as A
import pandas as pd
import pytest
import yaml

FIX = Path(__file__).resolve().parent / "fixtures"
REPO = Path(__file__).resolve().parent.parent
SLO = yaml.safe_load((REPO / "configs" / "slo.yaml").read_text())
PRICES = yaml.safe_load((REPO / "configs" / "gpu_prices.yaml").read_text())["gpu_usd_per_hour"]

EXPECTED_PLOTS = [
    "02_throughput_vs_concurrency.png",
    "02_latency_vs_concurrency.png",
    "03_ttft_vs_input_len.png",
    "03_e2e_vs_output_len.png",
    "04a_max_num_seqs_comparison.png",
    "04b_batched_tokens_comparison.png",
    "04c_prefix_caching_comparison.png",
    "04d_kv_capacity_comparison.png",
    "05_cost_vs_latency.png",
]


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("analysis")
    df = A.run([FIX], d / "results", d / "plots")
    return d, df


def test_fixtures_are_marked_fake():
    files = sorted(FIX.glob("*.json"))
    assert files
    for f in files:
        assert json.loads(f.read_text())["_fixture"] is True


def test_outputs_created(out):
    d, df = out
    assert (d / "results" / "summary.csv").stat().st_size > 0
    assert (d / "results" / "SUMMARY.md").stat().st_size > 0
    for name in EXPECTED_PLOTS:
        assert (d / "plots" / name).stat().st_size > 1000, name
    csv = pd.read_csv(d / "results" / "summary.csv")
    assert len(csv) == len(list(FIX.glob("*.json")))
    for col in (
        "usd_per_1m_output_tokens",
        "meets_slo",
        "ttft_p95_ms",
        "preemptions",
        "prefix_cache_hit_rate",
    ):
        assert col in csv.columns


def test_summary_md_flags_fixture_data(out):
    d, _ = out
    md = (d / "results" / "SUMMARY.md").read_text()
    assert "FIXTURE DATA" in md
    assert "## 05 Cost / performance under SLO" in md


def test_cost_math_matches_formula_for_every_row(out):
    _, df = out
    for _, r in df.iterrows():
        expected = PRICES["L4"] / (r["output_tok_per_s"] * 3600) * 1e6
        assert r["usd_per_1m_output_tokens"] == pytest.approx(expected, rel=1e-9)


def test_cost_known_value():
    # $0.80/hr at 1000 output tok/s = 3.6M tok/hr -> $0.80 / 3.6 = $0.2222 per 1M
    assert A.usd_per_1m_output_tokens(0.80, 1000.0) == pytest.approx(0.8 / 3.6)
    assert math.isnan(A.usd_per_1m_output_tokens(0.80, 0.0))


def test_slo_boundaries():
    ok = {"p95_ttft_ms": 500, "p95_tpot_ms": 50, "max_error_rate": 0.0}
    assert A.meets_slo(500, 50, 0.0, ok)  # limits are inclusive
    assert not A.meets_slo(500.1, 50, 0.0, ok)
    assert not A.meets_slo(100, 50.1, 0.0, ok)
    assert not A.meets_slo(100, 10, 0.01, ok)
    assert not A.meets_slo(math.nan, 10, 0.0, ok)  # missing data never passes


def test_slo_column_matches_raw_json(out):
    _, df = out
    for f in FIX.glob("*.json"):
        doc = json.loads(f.read_text())
        b, m = doc["bench"], doc["meta"]
        expect = b["p95_ttft_ms"] <= 500 and b["p95_tpot_ms"] <= 50 and b["failed"] == 0
        row = df[
            (df.experiment == m["experiment"])
            & (df.config == m["config"])
            & (df.workload_id == m["workload"]["id"])
        ]
        assert len(row) == 1
        assert bool(row.iloc[0]["meets_slo"]) is expect, f.name


def test_failed_requests_break_slo():
    rec = json.loads((FIX / "02_baseline_in512_out128_c4.json").read_text())
    rec["bench"]["failed"] = 1
    assert A.build_row(rec, PRICES, SLO)["meets_slo"] is False
    rec["bench"]["failed"] = 0
    assert A.build_row(rec, PRICES, SLO)["meets_slo"] is True


def test_best_under_slo_is_max_throughput_among_passing(out):
    _, df = out
    best = A.pick_best_under_slo(df)
    pool = df[df.experiment.isin(["02"]) | df.experiment.str.startswith("04")]
    passing = pool[pool.meets_slo]
    assert best is not None and best["meets_slo"]
    assert best["output_tok_per_s"] == passing["output_tok_per_s"].max()
    # max throughput under a fixed GPU price == min $/1M tokens
    assert best["usd_per_1m_output_tokens"] == passing["usd_per_1m_output_tokens"].min()


def test_no_passing_rows_gives_none():
    df = pd.DataFrame({"experiment": ["02"], "meets_slo": [False], "output_tok_per_s": [1.0]})
    assert A.pick_best_under_slo(df) is None


def test_missing_optional_telemetry_does_not_crash():
    rec = json.loads((FIX / "02_baseline_in512_out128_c4.json").read_text())
    rec["telemetry"] = {}
    row = A.build_row(rec, PRICES, SLO)
    assert row["preemptions"] is None
