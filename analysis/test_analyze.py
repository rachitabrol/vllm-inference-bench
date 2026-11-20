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
            & (df.repeat == m.get("repeat", 0))
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


# ---- repeats ------------------------------------------------------------------------------------


def _agg(df):
    return A.aggregate(df, SLO)


def _group(agg, exp, config, wid):
    g = agg[(agg.experiment == exp) & (agg.config == config) & (agg.workload_id == wid)]
    assert len(g) == 1
    return g.iloc[0]


def test_summary_csv_has_repeat_and_itl_p99(out):
    d, df = out
    csv = pd.read_csv(d / "results" / "summary.csv", dtype={"experiment": str})
    assert {"repeat", "itl_p99_ms"} <= set(csv.columns)
    assert csv["itl_p99_ms"].notna().all()
    sel = csv[(csv.experiment == "02") & (csv.workload_id == "in512_out128_c64")]
    assert sorted(sel["repeat"]) == [0, 1, 2]


def test_old_json_without_repeat_is_repeat_zero_n1(out):
    _, df = out
    rec = json.loads((FIX / "02_baseline_in512_out128_c4.json").read_text())
    assert "repeat" not in rec["meta"]
    assert A.build_row(rec, PRICES, SLO)["repeat"] == 0
    g = _group(_agg(df), "02", "baseline", "in512_out128_c4")
    assert g["n"] == 1
    assert math.isnan(g["output_tok_per_s_cv_pct"])
    assert g["output_tok_per_s_min"] == g["output_tok_per_s_max"] == g["output_tok_per_s"]


def test_grouping_median_and_cv_math(out):
    _, df = out
    runs = df[(df.experiment == "02") & (df.config == "baseline") & (df.workload_id == "in512_out128_c64")]
    assert len(runs) == 3
    t = runs["output_tok_per_s"].tolist()
    g = _group(_agg(df), "02", "baseline", "in512_out128_c64")
    mean = sum(t) / 3
    std = math.sqrt(sum((x - mean) ** 2 for x in t) / 2)
    assert g["n"] == 3
    assert g["output_tok_per_s"] == pytest.approx(sorted(t)[1])
    assert g["output_tok_per_s_min"] == pytest.approx(min(t))
    assert g["output_tok_per_s_max"] == pytest.approx(max(t))
    assert g["output_tok_per_s_cv_pct"] == pytest.approx(std / mean * 100)
    assert g["ttft_p95_ms"] == pytest.approx(runs["ttft_p95_ms"].median())
    assert g["itl_p99_ms"] == pytest.approx(runs["itl_p99_ms"].median())
    # cost + SLO come from the medians
    assert g["usd_per_1m_output_tokens"] == pytest.approx(PRICES["L4"] / (g["output_tok_per_s"] * 3600) * 1e6)


def test_cv_pct_helper():
    assert A.cv_pct([10.0, 10.0]) == 0.0
    assert A.cv_pct([8.0, 12.0]) == pytest.approx(math.sqrt(8) / 10 * 100)
    assert math.isnan(A.cv_pct([5.0]))


def test_aggregate_row_count_and_agg_csv(out):
    d, df = out
    agg = pd.read_csv(d / "results" / "summary_agg.csv")
    assert len(agg) == len(df.groupby(["experiment", "config", "workload_id"]))
    cols = {"n", "output_tok_per_s_cv_pct", "ttft_p95_ms_min", "ttft_p95_ms_max", "itl_p99_ms"}
    assert cols <= set(agg.columns)
    assert agg["n"].max() == 3


def test_slo_uses_medians():
    base = json.loads((FIX / "02_baseline_in512_out128_c4.json").read_text())
    rows = []
    for i, ttft in enumerate([100.0, 100.0, 900.0]):  # median passes, one outlier fails
        r = json.loads(json.dumps(base))
        r["meta"]["repeat"] = i
        r["bench"]["p95_ttft_ms"] = ttft
        r["bench"]["p95_tpot_ms"] = 10.0
        r["bench"]["failed"] = 0
        rows.append(r)
    df = A.build_dataframe(rows, PRICES, SLO)
    assert list(df.meets_slo) == [True, True, False]
    assert bool(_agg(df).iloc[0]["meets_slo"]) is True


def test_summary_md_n_column_variance_and_itl_p99(out):
    d, _ = out
    md = (d / "results" / "SUMMARY.md").read_text()
    assert "| n |" in md
    assert "## Run-to-run variance" in md
    assert "Throughput CV across" in md and "max" in md and "median" in md
    assert "\u2013" in md  # median (min-max) throughput text for n>1
    sec03 = md.split("## 03")[1].split("\n## ")[0]
    sec04b = md.split("## 04b")[1].split("\n## ")[0]
    assert "ITL p99" in sec03 and "ITL p99" in sec04b


def test_cross_experiment_identical_point_is_pooled(out):
    out_dir, df = out
    (c,) = [x for x in A.cross_experiment_groups(df) if x["workload_id"] == "in512_out128_c64"]
    assert c["experiments"] == ["02", "04a"]
    assert c["n"] == 5  # 3 runs in 02 + 2 runs in 04a
    t = df[(df.config == "baseline") & (df.workload_id == "in512_out128_c64")]["output_tok_per_s"]
    assert len(t) == 5
    assert c["median"] == pytest.approx(t.median())
    assert c["cv_pct"] == pytest.approx(A.cv_pct(t))
    assert c["min"] == pytest.approx(t.min()) and c["max"] == pytest.approx(t.max())
    md = (out_dir / "results" / "SUMMARY.md").read_text()
    assert "Pooled n=5" in md


def test_01_and_02_c1_not_pooled(out):
    # same config + concurrency but different num_prompts -> not the identical workload
    _, df = out
    assert all(x["workload_id"] != "in512_out128_c1" for x in A.cross_experiment_groups(df))


def test_error_bars_only_when_repeats(out, tmp_path):
    _, df = out
    agg = _agg(df)
    assert A._yerr(agg[agg.experiment == "02"], "output_tok_per_s") is not None
    only_single = agg[(agg.experiment == "03")]
    assert A._yerr(only_single, "output_tok_per_s") is None
    ye = A._yerr(agg[agg.experiment == "02"].sort_values("concurrency"), "output_tok_per_s")
    assert all(v >= 0 for side in ye for v in side)


def test_old_style_dataset_plots_without_error_bars(tmp_path):
    recs = [json.loads(f.read_text()) for f in FIX.glob("*.json")]
    recs = [r for r in recs if "repeat" not in r["meta"]]  # the original single-shot fixtures
    assert len(recs) > 15
    d = tmp_path / "raw"
    d.mkdir()
    for i, r in enumerate(recs):
        (d / f"{i}.json").write_text(json.dumps(r))
    df = A.run([d], tmp_path / "res", tmp_path / "pl")
    assert (df["repeat"] == 0).all()
    assert (tmp_path / "pl" / "02_throughput_vs_concurrency.png").stat().st_size > 1000
    agg = pd.read_csv(tmp_path / "res" / "summary_agg.csv")
    assert (agg["n"] == 1).all()
    tables = (tmp_path / "res" / "SUMMARY.md").read_text().split("## 01")[1]
    assert "\u2013" not in tables
