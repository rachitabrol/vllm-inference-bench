"""Config inheritance, flag rendering, the one-flag-difference rule, and job building."""

from pathlib import Path

import pytest
import yaml

from benchmark import config as C

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
VARIANTS = sorted(
    p.stem for p in CONFIGS.glob("*.yaml") if p.stem not in {"baseline", "experiments", "slo", "gpu_prices"}
)


def test_baseline_is_fully_explicit():
    cfg = C.load_config("baseline")
    for key in set(C.SERVE_FLAGS) - C.OPTIONAL_SERVE_KEYS:
        assert key in cfg, f"baseline must set {key} explicitly (vLLM defaults drift)"
    assert cfg["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg["dtype"] == "bfloat16"


def test_baseline_flag_rendering_exact():
    flags = C.render_serve_flags(C.load_config("baseline"))
    assert flags == [
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.9",
        "--max-num-seqs", "256",
        "--max-num-batched-tokens", "2048",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--seed", "42",
    ]  # fmt: skip


def test_inheritance_merges_and_overrides():
    v = C.load_config("max_seqs_8")
    b = C.load_config("baseline")
    assert v["max_num_seqs"] == 8
    assert v["name"] == "max_seqs_8"
    assert "base" not in v
    for k in b:
        if k not in ("max_num_seqs", "name"):
            assert v[k] == b[k]


def test_boolean_flag_renders_no_form():
    flags = C.render_serve_flags(C.load_config("no_prefix_cache"))
    assert "--no-enable-prefix-caching" in flags
    assert "--enable-prefix-caching" not in flags


@pytest.mark.parametrize("name", VARIANTS)
def test_each_variant_differs_from_baseline_in_exactly_one_flag(name):
    diff = C.diff_flags(C.load_config("baseline"), C.load_config(name))
    assert len(diff) == 1, f"{name} differs in {diff}"


def test_variant_list_is_what_we_expect():
    assert set(VARIANTS) == {
        "max_seqs_8", "max_seqs_32", "batched_tokens_512", "batched_tokens_8192",
        "no_prefix_cache", "kv_cache_2gib",
    }  # fmt: skip


def test_all_experiment_configs_exist_and_are_baseline_or_one_flag_variants():
    base = C.load_config("baseline")
    for exp_id, exp in C.load_experiments()["experiments"].items():
        for name in exp["configs"]:
            cfg = C.load_config(name)
            assert len(C.diff_flags(base, cfg)) <= 1, f"{exp_id}/{name}"


def test_cycle_and_unknown_key_detection(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"name": "a", "base": "b", "model": "m"}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"name": "b", "base": "a", "model": "m"}))
    with pytest.raises(ValueError, match="cycle"):
        C.load_config("a", tmp_path)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({"name": "c", "model": "m", "max_num_seq": 4}))
    with pytest.raises(ValueError, match="unknown keys"):
        C.load_config("c", tmp_path)


def test_build_jobs_all_counts():
    jobs = C.build_jobs("all")
    assert len(jobs) == 13  # 01:1 02:1 03:1 04a:3 04b:3 04c:2 04d:2
    by_exp = {}
    for j in jobs:
        by_exp.setdefault(j["experiment"], []).append(j)
    assert len(by_exp["02"][0]["workloads"]) == 6
    assert len(by_exp["03"][0]["workloads"]) == 6
    for j in jobs:
        for w in j["workloads"]:
            assert w["num_prompts"] >= 4 * w["concurrency"]


def test_build_jobs_selection_and_prefix():
    assert {j["experiment"] for j in C.build_jobs("01,02")} == {"01", "02"}
    assert {j["experiment"] for j in C.build_jobs("04")} == {"04a", "04b", "04c", "04d"}
    with pytest.raises(ValueError):
        C.build_jobs("99")


def test_smoke_jobs():
    jobs = C.build_jobs("all", smoke=True)  # smoke ignores the selection: only 01 + 02
    assert {j["experiment"] for j in jobs} == {"01", "02"}
    for j in jobs:
        assert j["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert all(w["num_prompts"] == 20 for w in j["workloads"])
        assert {w["concurrency"] for w in j["workloads"]} <= {1, 4}


def test_prefix_workload_fields():
    (job,) = [j for j in C.build_jobs("04c") if j["config_name"] == "baseline"]
    (w,) = job["workloads"]
    assert w["dataset"] == "prefix_repetition"
    assert (w["prefix_len"], w["suffix_len"], w["output_len"], w["concurrency"]) == (1024, 128, 128, 16)
    assert w["input_len"] == 1152


def test_expand_repeats_one_container_per_repeat():
    jobs = C.build_jobs("01,02")
    out = C.expand_repeats(jobs, 3)
    assert len(out) == 3 * len(jobs)
    for j in jobs:
        same = [x for x in out if (x["experiment"], x["config_name"]) == (j["experiment"], j["config_name"])]
        reps = [x["repeat"] for x in same]
        assert sorted(reps) == [0, 1, 2]
    assert all("repeat" not in j for j in jobs)  # inputs not mutated
    assert len(C.build_jobs("all")) * 3 == len(C.expand_repeats(C.build_jobs("all"), 3)) == 39


def test_smoke_forces_one_repeat():
    jobs = C.build_jobs("all", smoke=True)
    out = C.expand_repeats(jobs, 5, smoke=True)
    assert len(out) == len(jobs) and {j["repeat"] for j in out} == {0}


def test_repeats_must_be_positive():
    with pytest.raises(ValueError):
        C.expand_repeats(C.build_jobs("01"), 0)


def test_result_filename_suffix():
    name = C.result_filename("02", "baseline", "in512_out128_c64", 2)
    assert name == "02_baseline_in512_out128_c64_r2.json"
    assert C.result_filename("02", "baseline", "in512_out128_c64").endswith("_r0.json")


def test_workloads_on_one_server_never_share_a_seed():
    # Same seed on the same server = identical prompts = prefix-cache hits that inflate later runs.
    for exp_id, exp in C.load_experiments()["experiments"].items():
        wls = C._expand_workloads(exp["workloads"], C.load_experiments()["defaults"])
        seeds = [w.seed for w in wls]
        assert len(set(seeds)) == len(seeds), f"{exp_id} reuses a seed: {seeds}"


def test_workload_seed_is_stable_and_shared_across_configs():
    assert C.workload_seed(42, "in512_out128_c64") == C.workload_seed(42, "in512_out128_c64")
    assert C.workload_seed(42, "in512_out128_c64") != C.workload_seed(42, "in512_out128_c32")
