from benchmark import config as C
from benchmark import telemetry as T
from benchmark import workloads as W


def _job(exp, cfg):
    return next(j for j in C.build_jobs(exp) if j["config_name"] == cfg)


def test_random_bench_command():
    wl = _job("02", "baseline")["workloads"][0]
    cmd = W.build_bench_command("Qwen/Qwen2.5-1.5B-Instruct", wl, "/tmp/r", "x.json", 8000)
    assert cmd[:3] == ["vllm", "bench", "serve"]
    s = " ".join(cmd)
    for frag in (
        "--dataset-name random", "--random-input-len 512", "--random-output-len 128", f"--seed {wl['seed']}",
        "--max-concurrency 1", "--num-warmups 10", "--ignore-eos", "--save-result",
        "--result-dir /tmp/r", "--result-filename x.json", "--num-prompts 200",
        "--percentile-metrics ttft,tpot,itl,e2el", "--metric-percentiles 50,90,95,99",
    ):  # fmt: skip
        assert frag in s, frag
    assert "--prefix-repetition-prefix-len" not in s
    assert wl["seed"] != 42  # per-workload seed, not the shared base


def test_prefix_repetition_bench_command():
    wl = _job("04c", "baseline")["workloads"][0]
    s = " ".join(W.build_bench_command("m", wl, "/tmp/r", "x.json"))
    assert "--dataset-name prefix_repetition" in s
    assert "--prefix-repetition-prefix-len 1024" in s
    assert "--prefix-repetition-suffix-len 128" in s
    assert "--prefix-repetition-num-prefixes 1" in s
    assert "--prefix-repetition-output-len 128" in s
    assert "--random-input-len" not in s


def test_every_emitted_bench_flag_is_declared_required():
    declared = set(W.REQUIRED_BENCH_FLAGS)
    for job in C.build_jobs("all"):
        for wl in job["workloads"]:
            flags = {a for a in W.build_bench_command("m", wl, "/d", "f.json") if a.startswith("--")}
            assert flags <= declared, flags - declared


def test_serve_command_and_flag_check():
    job = _job("01", "baseline")
    cmd = W.build_serve_command(job["model"], job["serve_flags"], 8000)
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen2.5-1.5B-Instruct"]
    declared = set(W.REQUIRED_SERVE_FLAGS)
    assert {a for a in cmd if a.startswith("--")} <= declared | {"--no-enable-prefix-caching"}
    assert W.missing_flags("--dtype --seed", ["--dtype", "--nope"]) == ["--nope"]


PROM = """\
# HELP vllm:kv_cache_usage_perc KV cache usage
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:num_requests_waiting{engine="0"} 3.0
vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 3.0
vllm:num_preemptions_total{engine="0"} 5.0
vllm:request_num_preemptions_sum{engine="0"} 99.0
vllm:prefix_cache_queries_total{engine="0"} 1000.0
vllm:prefix_cache_hits_total{engine="0"} 800.0
vllm:external_prefix_cache_hits_total{engine="0"} 7.0
"""


def test_parse_metrics_matches_only_intended_series():
    m = T.parse_metrics(PROM)
    assert m == {
        "kv_cache_usage": 0.25, "num_waiting": 3.0, "preemptions": 5.0,
        "prefix_cache_queries": 1000.0, "prefix_cache_hits": 800.0,
    }  # fmt: skip


def test_parse_nvidia_smi():
    assert T.parse_nvidia_smi("1234, 23034, 87\n") == (1234.0, 23034.0, 87.0)
    assert T.parse_nvidia_smi("garbage") is None


def test_summarize_deltas_and_missing():
    gpu = [(1000.0, 24000.0, 10.0), (3000.0, 24000.0, 90.0)]
    ms = [
        {"kv_cache_usage": 0.1, "preemptions": 2.0, "prefix_cache_queries": 100.0, "prefix_cache_hits": 10.0},
        {
            "kv_cache_usage": 0.5,
            "preemptions": 6.0,
            "prefix_cache_queries": 300.0,
            "prefix_cache_hits": 110.0,
        },
    ]
    s = T.summarize(gpu, ms)
    assert s["gpu_mem_peak_mib"] == 3000.0 and s["gpu_util_mean"] == 50.0
    assert s["kv_cache_usage_peak"] == 50.0
    assert s["preemptions"] == 4.0
    assert s["prefix_cache_hit_rate"] == 0.5
    empty = T.summarize([], [])  # nothing sampled: must not crash
    assert empty["gpu_mem_peak_mib"] is None and empty["preemptions"] is None
