"""CPU tests of run receipts (receipt/): the vLLM log parsers on real captured lines, collection parsing,
check and diff, measurement methods, the exclusive window and the report's receipt columns.

The fixtures under receipt/fixtures/ are real captures from a ring of eight DGX Sparks on 2026-10-11, with
LAN and control-network addresses replaced by documentation addresses (192.0.2.0/24, 198.51.100.0/24) and
host names by ``spark-p<position>``:

- ``collect-qwen-tp2/``: one ``receipt collect`` of qwen38-flash-next-tp2 (csf checkpoint) on positions 6
  and 7 serving FP8 target KV in 32-token blocks with the draft's ``kv_cache_dtype`` ``auto``; the raw reads
  of both ranks, the status entry, ``/metrics`` (cache and request gauges only) and ``/v1/models``.
- ``logs/``: the selected startup lines (``docker logs --timestamps``) of single server starts:
  ``qwen-tp2-fp8-kv-draft-auto.log`` (the same configuration as above) and ``qwen-tp2-bf16-kv-draft-auto.log``
  (BF16 target KV in 16-token blocks, draft ``auto``); ``glm-tp2-compile-arm.log`` (glm53-flash-nvfp4-spark-tp2
  installed with a ``--compilation-config`` that leaves out ``"mode":0``) and ``glm-tp2-profile-arm.log`` (the
  profile's own compilation configuration); ``deepseek-tp4-*-config-lines.txt``: the ``non-default args`` lines a
  campaign captured for deepseek-v41-flash-tp4 as installed and with ``--async-scheduling`` added.
- ``window/``: rank 0 access-log lines of one measurement window with one client, and access plus engine lines
  of a window in which two benchmark clients on one machine overlapped.
- ``bench/``: llm-inference-bench result metadata: the owner's temperature-1 matrix (0.7.6), the serving A/B
  runner's temperature-0 decode (0.4.29), a run with the tool's own defaults, and an arm's run with the
  owner's grid, temperature and token limit but the tool's default warm-up and coding-peak temperature.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from performance.harnesses.serving_ab import cli, measure, report
from performance.harnesses.serving_ab.receipt import collect, compare, method, pyrepr, resolved, vllm_log, window
from performance.harnesses.serving_ab.receipt.__main__ import main as receipt_main

FIXTURES = Path(__file__).parent / "receipt" / "fixtures"
LOGS = FIXTURES / "logs"


def log(name: str) -> str:
    return (LOGS / name).read_text(encoding="utf-8")


def receipt_of(text: str, **extra) -> dict:
    return resolved.build([{"rank": 0, "log": text, **extra}])


# Parsers ------------------------------------------------------------------------------------------------------

def test_repr_parser_reads_vllm_reprs_without_evaluating_them():
    text = ("{'mode': <CompilationMode.NONE: 0>, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, "
            "'spec': SpeculativeConfig(method='mtp', num_spec_tokens=3), 'dtype': torch.bfloat16, 'sizes': [1, 2], "
            "'ops': {'+a', '-b'}, 'empty': {}, 'f': 0.85, 'neg': -1e-3, 'n': None, 'quote': 'it\\'s', "
            "'shape': <DynamicShapesType.BACKED: 'backed'>, 'fn': <function f at 0x7f>, 't': (1,)}")
    value = pyrepr.parse(text)
    assert value["mode"] == {"enum": "CompilationMode.NONE", "value": 0}
    assert value["cudagraph_mode"] == {"enum": "CUDAGraphMode.FULL_AND_PIECEWISE", "value": [2, 1]}
    assert value["spec"] == {"__type__": "SpeculativeConfig", "method": "mtp", "num_spec_tokens": 3}
    assert value["dtype"] == "torch.bfloat16" and value["ops"] == ["+a", "-b"] and value["empty"] == {}
    assert value["f"] == 0.85 and value["neg"] == -0.001 and value["n"] is None and value["quote"] == "it's"
    assert value["shape"] == {"enum": "DynamicShapesType.BACKED", "value": "backed"}
    assert value["fn"] == {"repr": "<function f at 0x7f>"} and value["t"] == [1]
    with pytest.raises(pyrepr.ReprError):
        pyrepr.parse("{'a': 1")
    with pytest.raises(pyrepr.ReprError):
        pyrepr.parse("{'a': 1} trailing")


def test_a_truncated_argument_line_keeps_every_item_completed_before_the_cut():
    line = next(x for x in log("qwen-tp2-fp8-kv-draft-auto.log").splitlines() if "non-default args" in x)
    cut = line[:line.index("'speculative_config'") + 60]
    scan = vllm_log.scan(cut)
    assert scan["truncated"] == ["non_default_args"]
    args = scan["non_default_args"]
    assert args["kv_cache_dtype"] == "fp8" and args["block_size"] == 32 and args["async_scheduling"] is True
    assert "compilation_config" not in args
    record = receipt_of(cut)
    # Keys after the cut are unknown, not vLLM's defaults.
    assert record["sources"]["scheduler.prefill_schedule_interval"] == "not recorded"
    assert record["settings"]["target.kv_cache_dtype"] == "fp8"


def test_the_engine_line_splits_into_vllm_config_fields_and_keeps_plain_text_values():
    line = next(x for x in log("qwen-tp2-fp8-kv-draft-auto.log").splitlines() if "with config:" in x)
    config = vllm_log.scan(line)["engine_config"]
    assert config["version"] == "0.1.dev21553+gab86b7073.d20261001"
    assert config["kv_cache_dtype"] == "fp8" and config["dtype"] == "torch.bfloat16"
    assert config["served_model_name"] == "Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD-TP2"
    assert config["speculative_config"]["num_spec_tokens"] == 3
    assert config["compilation_config"]["mode"] == {"enum": "CompilationMode.NONE", "value": 0}
    assert config["kernel_config"]["linear_backend"] == "b12x"
    assert list(config)[1:4] == ["model", "speculative_config", "tokenizer"]


def test_scan_reads_the_single_purpose_lines_of_a_start():
    scan = vllm_log.scan(log("qwen-tp2-fp8-kv-draft-auto.log"))
    assert scan["attention_block_size"] == 2848 and scan["kv_cache_tokens"] == 2534058
    assert [head["format"] for head in scan["lm_heads"]] == ["MXFP8", "NVFP4"]  # target head, then the draft's
    assert scan["moe_kernel"] == {"NvFp4": "B12X"} and scan["gemm_kernel"]["MXFP8"] == "B12xMxfp8LinearKernel"
    assert scan["auto_env"] == {"VLLM_USE_BREAKABLE_CUDAGRAPH": "1"} and scan["breakable_cudagraph"] is True
    assert scan["sircl_receipt_lines"]["tp:0"]["nccl"] == "none"
    assert scan["occurrences"]["non_default_args"] == 1 and "truncated" not in scan


# Resolved settings, check and diff -----------------------------------------------------------------------------

def test_resolved_settings_state_the_target_and_draft_kv_and_their_effective_dtypes():
    fp8 = receipt_of(log("qwen-tp2-fp8-kv-draft-auto.log"))
    bf16 = receipt_of(log("qwen-tp2-bf16-kv-draft-auto.log"))
    assert fp8["settings"]["target.kv_cache_dtype"] == "fp8" and fp8["settings"]["target.block_size"] == 32
    assert bf16["settings"]["target.kv_cache_dtype"] == "bfloat16" and bf16["settings"]["target.block_size"] == 16
    for record in (fp8, bf16):
        assert record["settings"]["speculative.kv_cache_dtype"] == "auto"
        assert record["derived"]["speculative.kv_cache_dtype_effective"] == "bfloat16"
        assert record["sources"]["target.kv_cache_dtype"] == "engine config line"
    assert compare.differences(fp8["settings"], bf16["settings"]) == {
        "target.block_size": (32, 16), "target.kv_cache_dtype": ("fp8", "bfloat16")}
    assert fp8["observed"]["kv.block_size"] == 2848 and bf16["observed"]["kv.block_size"] == 1504


def without_draft_kv(text: str) -> str:
    """The BF16 start's lines as the control arm installed them: no kv_cache_dtype in --speculative-config."""
    return text.replace(", 'kv_cache_dtype': 'auto'}", "}")


def test_an_arm_that_changed_two_settings_besides_the_declared_one_fails_its_check():
    arm = receipt_of(log("qwen-tp2-fp8-kv-draft-auto.log"))
    control = receipt_of(without_draft_kv(log("qwen-tp2-bf16-kv-draft-auto.log")))
    assert control["settings"]["speculative.kv_cache_dtype"] is None
    declared = {"schema": compare.SCHEMA, "arm": "WD", "control": "W", "changes": {"speculative.kv_cache_dtype": "auto"}}
    result = compare.check(arm, control, declared)
    assert not result["passed"]
    assert {(p["kind"], p["key"]) for p in result["problems"]} == {
        ("undeclared", "target.kv_cache_dtype"), ("undeclared", "target.block_size")}
    # Unset, the draft inherits the control's BF16 target KV; "auto" is the model dtype, BF16: no effective change.
    assert any("speculative.kv_cache_dtype_effective at \"bfloat16\"" in w for w in result["warnings"])
    assert result["consequences"]["observed"]["kv.block_size"] == ["2848", "1504"]
    full = dict(declared, changes={"speculative.kv_cache_dtype": "auto", "target.kv_cache_dtype": "fp8",
                                   "target.block_size": {"from": 16, "to": 32}})
    assert compare.check(arm, control, full)["passed"]
    wrong_from = dict(declared, changes={**full["changes"], "target.block_size": {"from": 8, "to": 32}})
    assert [p["kind"] for p in compare.check(arm, control, wrong_from)["problems"]] == ["control"]


def test_a_declared_change_that_did_not_take_effect_fails_the_check():
    # The compile arm left "mode":0 out of --compilation-config to turn torch.compile on; vLLM turned on
    # breakable CUDA graphs for the architecture, which set the mode to NONE again.
    arm, control = receipt_of(log("glm-tp2-compile-arm.log")), receipt_of(log("glm-tp2-profile-arm.log"))
    assert compare.diff(arm, control)["settings"] == {}
    assert arm["observed"]["breakable_cudagraph"] is True
    assert arm["observed"]["auto_env.VLLM_USE_BREAKABLE_CUDAGRAPH"] == "1"
    result = compare.check(arm, control, {"changes": {"compilation.mode": "VLLM_COMPILE"}})
    assert not result["passed"] and [p["kind"] for p in result["problems"]] == ["no-effect"]
    assert "resolved CompilationMode.NONE (0)" in result["problems"][0]["text"]
    assert compare.check(arm, control, {"changes": {"compilation.*": None}})["problems"][0]["kind"] == "no-effect"


def test_an_unset_async_scheduling_resolves_by_vllms_rule_so_adding_the_flag_changes_nothing():
    profile = receipt_of(log("deepseek-tp4-profile-config-lines.txt"))
    added = receipt_of(log("deepseek-tp4-async-config-lines.txt"))
    assert profile["settings"]["scheduler.async_scheduling"] is True
    assert profile["sources"]["scheduler.async_scheduling"].startswith("inferred: vLLM turns unset async scheduling on")
    assert added["sources"]["scheduler.async_scheduling"] == "argument"
    result = compare.check(added, profile, {"changes": {"scheduler.async_scheduling": True}})
    assert [p["kind"] for p in result["problems"]] == ["no-effect"]
    assert "the control already has it" in result["problems"][0]["text"]
    assert resolved.async_rule(None, "ngram", "mp", False)[0] is False
    assert resolved.async_rule(None, "qwen3_next_mtp", "mp", False)[0] is True
    assert resolved.async_rule(True, "mtp", "mp", True)[0] is False


def test_matching_accepts_an_enum_by_name_or_value_and_scalars_by_text():
    mode = {"enum": "CompilationMode.NONE", "value": 0}
    assert all(compare.matches(mode, v) for v in ("CompilationMode.NONE", "NONE", 0))
    assert not compare.matches(mode, "VLLM_COMPILE")
    assert compare.matches(16, "16") and compare.matches(True, "true") and not compare.matches([1], "1")
    a, b = {"x": 1, "env.A": "1", "y": [1]}, {"x": 1, "env.A": "2"}
    assert compare.differences(a, b) == {"env.A": ("1", "2"), "y": ([1], compare.MISSING)}
    assert compare.differences(a, b, ignore=["env.*"]) == {"y": ([1], compare.MISSING)}


def test_cli_check_and_diff_exit_nonzero_on_a_failed_check_and_on_differences(tmp_path, capsys):
    arm, control = tmp_path / "arm", tmp_path / "control"
    for path, text in ((arm, log("qwen-tp2-fp8-kv-draft-auto.log")),
                       (control, without_draft_kv(log("qwen-tp2-bf16-kv-draft-auto.log")))):
        path.mkdir()
        (path / "resolved.json").write_text(json.dumps(receipt_of(text)), encoding="utf-8")
    declaration = tmp_path / "arm.json"
    declaration.write_text(json.dumps({"schema": compare.SCHEMA, "arm": "WD", "control": "W",
                                       "changes": {"speculative.kv_cache_dtype": "auto"}}), encoding="utf-8")
    assert receipt_main(["check", str(arm), str(control), "--declare", str(declaration),
                         "--output", str(arm / "check.json")]) == 1
    out = capsys.readouterr().out
    assert "FAIL target.kv_cache_dtype differs and is not declared" in out and "FAIL target.block_size" in out
    assert json.loads((arm / "check.json").read_text())["passed"] is False
    assert receipt_main(["diff", str(arm), str(control)]) == 1
    assert receipt_main(["diff", str(arm), str(arm / "resolved.json")]) == 0
    declaration.write_text(json.dumps({"changes": {}}), encoding="utf-8")
    assert receipt_main(["check", str(arm), str(control), "--declare", str(declaration)]) == 2


# Collection ---------------------------------------------------------------------------------------------------

def test_a_collection_rebuilds_offline_from_its_raw_reads():
    document = collect.rebuild(FIXTURES / "collect-qwen-tp2")
    settings = document["settings"]
    assert document["deployment"]["profile"] == "qwen38-flash-next-tp2"
    assert settings["deployment.checkpoint"] == "csf" and settings["image.id"].startswith("sha256:d52737a109e0")
    assert settings["target.kv_cache_dtype"] == "fp8" and settings["speculative.kv_cache_dtype"] == "auto"
    # The vLLM process's environment holds what the toolchain entrypoint adds to the container's.
    assert settings["env.QWEN_MTP_TORCH_PREFILL"] == "1"
    assert "QWEN_MTP_TORCH_PREFILL" in document["ranks"][0]["environment_set_at_start"]
    assert not any(k.endswith(("_IFNAME", "_HOST_IP", "_DIR")) for k in settings if k.startswith("env."))
    assert settings["sircl.tp.nccl"] == "none" and document["sources"]["sircl.tp.nccl"] == "log: SIRCL receipt line"
    assert document["observed"]["kv.block_size"] == 2848  # /metrics vllm:cache_config_info
    assert document["observed"]["clocks"][0]["applications_mhz"] == 2418
    assert document["rank_mismatches"] == []
    assert document["ranks"][1]["settings"]["env.SIRCL_NCCL"] == "never"
    assert document["digest"] == resolved.digest(settings)
    assert document["served_models"] == ["Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD-TP2"]


def test_ranks_agree_on_session_settings_whatever_their_place_in_the_group():
    line = ("(Worker pid=1) INFO 10-11 22:11:43 [adapter.py:487] SIRCL receipt group=tp:0 global_rank={r} rank={r} "
            "world=4 layout=ring:8 positions=0,1,2,3 nccl=none pynccl=skipped session=ring lanes=2 hcas=dev{r} "
            "capacity=131072 p2p=peers={peers},slots=8x524288,windows=65536 state=ready")
    document = resolved.build([{"rank": r, "log": line.format(r=r, peers=1 if r in (0, 3) else 2)} for r in range(4)])
    assert document["settings"]["sircl.tp.p2p"] == "slots=8x524288,windows=65536"
    assert document["settings"]["sircl.tp.capacity"] == 131072 and "sircl.tp.hcas" not in document["settings"]
    assert document["rank_mismatches"] == []


def test_the_rank_read_script_reads_only_and_never_the_full_environment():
    script = collect.rank_script("sr-x-r0")
    assert "Config.Env" not in script and "docker exec \"$C\" env | grep -E" in script
    assert "[v]llm.entrypoints" in script
    for verb in ("docker rm", "docker stop", "docker restart", "docker kill", "docker run", "mkdir", "> /"):
        assert verb not in script
    parts = collect.sections(f"x\n{collect.MARK} log\na\nb\n{collect.MARK} receipt r.json\n{{}}\n{collect.MARK} end\n")
    assert parts["log"] == "a\nb" and parts["receipts"] == {"r.json": "{}"} and "end" in parts


def test_the_runner_writes_a_receipt_from_the_logs_it_saved(tmp_path):
    (tmp_path / "log-r0.txt").write_text(log("qwen-tp2-bf16-kv-draft-auto.log"), encoding="utf-8")
    (tmp_path / "inspect-r0.json").write_text(json.dumps([{"Image": "sha256:" + "ab" * 32, "Config": {
        "Env": ["VLLM_MXFP8_LM_HEAD=1", "VLLM_HOST_IP=192.0.2.5", "HF_TOKEN=x"]}}]), encoding="utf-8")
    cli.write_receipt(tmp_path)
    document = json.loads((tmp_path / "resolved.json").read_text())
    assert document["settings"]["target.kv_cache_dtype"] == "bfloat16"
    assert document["settings"]["env.VLLM_MXFP8_LM_HEAD"] == "1"
    assert "env.HF_TOKEN" not in document["settings"] and "env.VLLM_HOST_IP" not in document["settings"]


# Measurement methods ------------------------------------------------------------------------------------------

def bench(name: str) -> dict:
    return json.loads((FIXTURES / "bench" / name).read_text(encoding="utf-8"))


def test_method_profiles_identify_the_owner_matrix_and_the_runner_method():
    assert method.identify(bench("owner-temp1-matrix.json")) == "owner-temp1-matrix"
    assert method.identify(bench("serving-ab-temp0.json")) == "serving-ab-temp0"
    assert method.identify(bench("tool-defaults.json")) is None
    problems = method.check(bench("tool-defaults.json"), "owner-temp1-matrix")
    assert any(p.startswith("temperature:") and "the server's default" in p for p in problems)
    assert any(p.startswith("max_tokens:") and "8192" in p for p in problems)
    assert {p.split(":")[0] for p in problems} == {"temperature", "max_tokens", "context_lengths", "concurrency_levels",
                                                     "duration_per_test", "decode_warmup_seconds", "coding_peak",
                                                     "coding_peak_temperature"}
    # A run with the owner's grid, temperature and token limit still differs in its warm-up and in the
    # coding peak's temperature, which the run left to the server.
    assert method.check(bench("temp1-arm-run.json"), "owner-temp1-matrix") == [
        "decode_warmup_seconds: method owner-temp1-matrix has 5.0, the result records 3.0",
        "coding_peak_temperature: method owner-temp1-matrix has 1.0, the result records the server's default"]
    assert {p.split(":")[0] for p in method.check(bench("serving-ab-temp0.json"), "owner-temp1-matrix")} >= {
        "version", "temperature", "max_tokens", "prefill_mode"}


def test_cli_method_exits_nonzero_when_the_result_differs_from_the_expected_method(capsys):
    assert receipt_main(["method", str(FIXTURES / "bench" / "owner-temp1-matrix.json"),
                         "--expect", "owner-temp1-matrix"]) == 0
    assert receipt_main(["method", str(FIXTURES / "bench" / "serving-ab-temp0.json"),
                         "--expect", "owner-temp1-matrix"]) == 1
    assert "DIFFERS" in capsys.readouterr().out


def test_the_runner_decode_keeps_its_arguments_by_default_and_runs_the_owner_method_by_name():
    default = measure.decode_method("phase1")
    assert default["method"] == "serving-ab-temp0"
    assert default["bench_args"] == ["--skip-prefill", "--contexts", "0,32k", "--concurrency", "1,2,4,8",
                                     "--token-targeting", "exact", "--temperature", "0", "--max-tokens", "1024",
                                     "--duration", "30", "--decode-warmup-seconds", "10",
                                     "--cell-warmup-timeout-seconds", "900"]
    owner = measure.decode_method("phase1", "owner-temp1-matrix")
    assert owner["method"] == "owner-temp1-matrix" and "--coding-peak" in owner["bench_args"]
    assert owner["bench_args"][owner["bench_args"].index("--decode-warmup-seconds") + 1] == "5"
    assert owner["bench_args"][owner["bench_args"].index("--coding-peak-temperature") + 1] == "1"
    assert "--skip-prefill" not in owner["bench_args"]
    assert owner["bench_args"][owner["bench_args"].index("--contexts") + 1] == "8k,16k,32k,64k,128k"
    custom = measure.decode_method("phase1", temperature=1.0)
    assert custom["method"] == "custom" and custom["bench_args"][custom["bench_args"].index("--temperature") + 1] == "1"
    assert measure.decode_method("phase1", max_tokens=2048)["settings"]["max_tokens"] == 2048


# Exclusive window ---------------------------------------------------------------------------------------------

def window_log(name: str) -> str:
    return (FIXTURES / "window" / name).read_text(encoding="utf-8")


def test_a_window_with_one_client_is_exclusive_and_another_clients_request_breaks_it():
    text = window_log("qwen-tp2-arm-window.log")
    entries = window.parse_access_log(text)
    record = window.summarize(entries, clients=["192.0.2.114"], since="2026-10-11T20:55:08Z",
                              until="2026-10-11T20:58:00Z")
    assert record["exclusive"] and record["outside_requests"] == 0
    sources = {(r["address"], r["path"]): r["source"] for r in record["rows"]}
    assert sources[("192.0.2.193", "/metrics")] == "monitoring" and sources[("127.0.0.1", "/health")] == "monitoring"
    assert sources[("192.0.2.114", "/v1/chat/completions")] == "declared client"
    assert not window.summarize(entries, clients=["192.0.2.114"], strict=True)["exclusive"]
    intruder = {"time": "2026-10-11T20:56:00.0Z", "address": "192.0.2.50", "method": "POST",
                "path": "/v1/chat/completions", "status": 200}
    broken = window.summarize(entries + [intruder], clients=["192.0.2.114"], since="2026-10-11T20:55:08Z",
                              until="2026-10-11T20:58:00Z")
    assert not broken["exclusive"] and broken["outside"][0]["address"] == "192.0.2.50"
    outside_window = window.summarize(entries + [dict(intruder, time="2026-10-11T21:30:00Z")],
                                      clients=["192.0.2.114"], since="2026-10-11T20:55:08Z",
                                      until="2026-10-11T20:58:00Z")
    assert outside_window["exclusive"]


def test_two_clients_on_one_machine_share_an_address_and_only_the_engine_peak_can_show_them():
    text = window_log("qwen-tp2-two-clients-one-host.log")
    entries, stats = window.parse_access_log(text), window.parse_engine_stats(text)
    assert {e["address"] for e in entries if e["method"] == "POST"} == {"192.0.2.114"}
    record = window.summarize(entries, clients=["192.0.2.114"], engine_stats=stats, max_concurrency=8)
    assert record["exclusive"] and record["engine_peak"]["requests"] == 8  # within the benchmark's own 8 streams
    assert not window.summarize(entries, clients=["192.0.2.114"], engine_stats=stats, max_concurrency=4)["exclusive"]
    assert window.format_window(record).endswith("EXCLUSIVE")


def test_preflight_refuses_a_serving_deployment_and_another_owners_reservation():
    idle = "vllm:num_requests_running{engine=\"0\"} 0.0\nvllm:num_requests_waiting{engine=\"0\"} 0.0\n"
    busy = idle.replace("running{engine=\"0\"} 0.0", "running{engine=\"0\"} 2.0")
    now = 1_791_760_000
    locks = ("@@LOCK /tmp/ring8-hw.lock 1791756273\nqwen-campaign 1791756273\n"
             "@@LOCK /tmp/ring8-hw-6-7.lock 1791759000\nowner: kv campaign\nsparks: 6-7 (spark-p6, spark-p7)\n"
             "started: 2026-10-11T17:57:35-05:00\n")

    def run(command):
        return locks if "ring8-hw" in command else ""

    readings = iter([idle, idle, busy])
    refused = window.preflight(positions=[6, 7], read_metrics=lambda: next(readings), run_on_node_a=run,
                               sources=window.sources_from(["hwlock", "setlock"]), samples=3, sleep=lambda s: None,
                               now=lambda: now)
    assert not refused["clear"]
    assert refused["problems"][0].startswith("the deployment is serving: 2 running")
    assert any("/tmp/ring8-hw-6-7.lock reserves [6, 7] for 'kv campaign'" in p for p in refused["problems"])
    stale = next(r for r in refused["reservations"] if r["where"] == "/tmp/ring8-hw.lock")
    assert stale["stale"] and stale["positions"] is None  # NAME EPOCH older than 2,700 s: abandoned
    own = window.preflight(positions=[6, 7], read_metrics=lambda: idle, run_on_node_a=run, owner="kv campaign",
                           sources=window.sources_from(["hwlock"]), samples=2, sleep=lambda s: None, now=lambda: now)
    assert own["clear"]
    elsewhere = window.preflight(positions=[4, 5], read_metrics=lambda: idle, run_on_node_a=run,
                                 sources=window.sources_from(["hwlock"]), samples=1, now=lambda: now)
    assert elsewhere["clear"]
    reserved = window.preflight(positions=[4], read_metrics=lambda: idle,
                                run_on_node_a=lambda c: '[{"positions": [4, 5], "owner": "reserve-tool"}]',
                                sources=window.sources_from(["command:sparkring reserve list --json"]), samples=1)
    assert not reserved["clear"] and "reserve-tool" in reserved["problems"][0]
    with pytest.raises(ValueError):
        window.sources_from(["nonsense"])


# Report -------------------------------------------------------------------------------------------------------

def write_start(root: Path, label: str, steps: float, bench_doc: dict, *, settings_log: str | None = None,
                check: dict | None = None) -> None:
    d = root / label
    d.mkdir()
    (d / "ready.json").write_text(json.dumps({"ready_seconds": 180}))
    (d / "decode.json").write_text(json.dumps({**bench_doc, "results": [
        {"context_tokens": 0, "concurrency": 1, "aggregate_tps": steps * 2.5, "server_steps_per_s": steps,
         "server_spec_accept_length": 2.5}]}))
    (d / "prefill.jsonl").write_text(json.dumps({"target_tokens": 8192, "ttft_seconds": 1.0}) + "\n")
    (d / "fingerprint.json").write_text(json.dumps({"fingerprints": {"math": {"tokens": ["1"]}}}))
    (d / "logprobs.json").write_text(json.dumps({"positions": [{"1": {"logprob": -0.5, "rank": 1}}]}))
    if settings_log:
        (d / "resolved.json").write_text(json.dumps(receipt_of(settings_log)))
    if check:
        (d / "check.json").write_text(json.dumps(check))


def test_the_report_shows_each_starts_receipt_and_refuses_ratios_across_methods(tmp_path):
    runner = bench("serving-ab-temp0.json")
    write_start(tmp_path, "S1", 25.0, runner, settings_log=log("qwen-tp2-fp8-kv-draft-auto.log"))
    write_start(tmp_path, "N1", 20.0, runner, settings_log=log("qwen-tp2-bf16-kv-draft-auto.log"))
    doc = report.summary(tmp_path, ["S1", "N1"])
    assert not doc["comparisons"]["S"]["refused"]
    text = report.tables(doc)
    assert "| S1 | S | serving-ab-temp0 |" in text and "| fp8 | 32 | auto | on | NONE | FULL_AND_PIECEWISE | 16 | 8192 |" in text
    assert "| 0k | 1 | 25.00 (25.00-25.00) | 20.00 (20.00-20.00) | 1.250 |" in text
    assert "method serving-ab-temp0: temperature 0, at most 1024 tokens, 30 s cells" in text

    other = tmp_path / "other"
    other.mkdir()
    write_start(other, "S1", 25.0, bench("owner-temp1-matrix.json"))
    write_start(other, "N1", 20.0, runner)
    text = report.tables(report.summary(other, ["S1", "N1"]))
    assert "| 0k | 1 | 25.00 (25.00-25.00) | 20.00 (20.00-20.00) | REFUSED |" in text
    assert "REFUSED: arm S against N: measured differently (S1 owner-temp1-matrix, N1 serving-ab-temp0" in text


def test_the_report_refuses_a_ratio_whose_arm_failed_its_check(tmp_path):
    runner = bench("serving-ab-temp0.json")
    failed = {"passed": False, "control": "N", "declared": {"speculative.kv_cache_dtype": "auto"},
              "problems": [{"text": "target.kv_cache_dtype differs and is not declared"}]}
    write_start(tmp_path, "S1", 25.0, runner, check=failed)
    write_start(tmp_path, "N1", 20.0, runner)
    doc = report.summary(tmp_path, ["S1", "N1"])
    assert doc["comparisons"]["S"]["refused"]
    text = report.tables(doc)
    assert "REFUSED: arm S against N: S1 failed its arm check: target.kv_cache_dtype differs and is not declared" in text
    assert "speculative.kv_cache_dtype=auto vs N | FAILED |" in text
