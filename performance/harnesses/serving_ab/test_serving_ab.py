"""CPU tests of the serving A/B runner: arm rendering, arm checks, set locks and the report."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from performance.harnesses.serving_ab import cli, report, setlock, spec, verify

IMAGE = "sha256:" + "ab" * 32


def base(rank: int) -> list[str]:
    return ["docker", "create", "--name", f"profile-r{rank}", "--entrypoint", "python3", "--gpus", "all",
            "--mount", "type=bind,src=/placeholder/model,dst=/models/target,readonly",
            "--mount", "type=bind,src=/serving-ab-cache,dst=/cache",
            "--env", "NCCL_ALGO=Ring", "--env", "NCCL_DEBUG=WARN", "--env", "NCCL_IB_HCA==a:1,b:1",
            "--env", "SIRCL_ENABLED=0", "--env", "SPARKRING_TRANSPORT_PROFILE=tp2-rocenante-adaptive",
            "--env", "VLLM_ENABLE_ROCE_ALLREDUCE=1", "--env", "VLLM_HOST_IP=192.0.2.200",
            "--env", "VLLM_PLUGINS=b12x_loader", IMAGE, "/entry.py", "serve", "--node-rank", str(rank)]


def bundle_doc(nccl: str, hosts: list[str]) -> dict:
    ranks = []
    for r, host in enumerate(hosts):
        env = ({"SIRCL_MODE": "custom", "SIRCL_NCCL": "never", "SIRCL_FUSED_NORM": "0",
                "VLLM_ENABLE_ROCE_ALLREDUCE": "0", "SPARKRING_TRANSPORT_PROFILE": "",
                "SPARKRING_TRANSPORT_MANIFEST_SHA256": "", "SPARK_TP4_ENABLED": "0"}
               if nccl == "none" else {"NCCL_IB_HCA": f"=dev{r}", "NCCL_ALGO": "Ring", "NCCL_SKIP_TREE_CONNECT": "1"})
        ranks.append({"rank": r, "lan_address": host, "environment": env, "pythonpath_prepend": "/sircl/src",
                      "vllm_plugins_add": "sircl", "vllm_arguments": [],
                      "mounts": [{"option": "type=bind,src=/tmp/run,dst=/sircl/run"}]})
    return {"nccl": nccl, "ranks": ranks}


HOSTS = ["192.0.2.10", "192.0.2.11"]
COMMON = dict(docker=["sudo", "-n", "docker"], run="t", models=["/m0", "/m1"],
              caches={"S": "/c/s", "S+": "/c/sp", "N": "/c/n", "P": "/c/p"}, host_ips=HOSTS,
              bundle=bundle_doc("none", HOSTS), bundle_auto=bundle_doc("all", HOSTS), ring_size=8, positions=[0, 1])


def test_every_arm_shares_the_site_substitutions():
    for arm in ("S", "S+", "N"):
        tokens = spec.render_arm(arm, [base(0), base(1)], **COMMON, seccomp="/r/loader-seccomp-1.json")[1]
        assert tokens[tokens.index("--security-opt") + 1] == "seccomp=/r/loader-seccomp-1.json"
        assert tokens[:7] == ["sudo", "-n", "docker", "run", "-d", "--name", f"ab-t-{spec.slug(arm)}-r1"]
        assert "type=bind,src=/m1,dst=/models/target,readonly" in tokens
        assert f"type=bind,src={COMMON['caches'][arm]},dst=/cache" in tokens
        env = spec.environment(tokens)
        assert env["VLLM_HOST_IP"] == "192.0.2.11" and env["NCCL_DEBUG"] == "INFO" and env["NCCL_DEBUG_SUBSYS"] == "INIT"


def test_sircl_arms_take_the_bundle_part_and_s_plus_adds_fused_norm():
    s = spec.environment(spec.render_arm("S", [base(0), base(1)], **COMMON)[0])
    sp = spec.environment(spec.render_arm("S+", [base(0), base(1)], **COMMON, extra_env={"SIRCL_REDUCE_LINK_BLOCKS": "1"},
                                          roce_slot=True)[0])
    assert s["SIRCL_MODE"] == "custom" and s["VLLM_PLUGINS"] == "b12x_loader,sircl" and s["PYTHONPATH"] == "/sircl/src"
    assert s["VLLM_ENABLE_ROCE_ALLREDUCE"] == "0" and s["SIRCL_FUSED_NORM"] == "0"
    assert sp["SIRCL_FUSED_NORM"] == "1" and sp["SIRCL_REDUCE_LINK_BLOCKS"] == "1"
    assert sp["VLLM_ENABLE_ROCE_ALLREDUCE"] == "1"


def test_nccl_arm_takes_the_auto_devices_and_drops_sircl():
    tokens = spec.render_arm("N", [base(0), base(1)], **COMMON)[1]
    env = spec.environment(tokens)
    assert env["NCCL_IB_HCA"] == "=dev1" and env["NCCL_SKIP_TREE_CONNECT"] == "1" and env["NCCL_ALGO"] == "Ring"
    assert env["VLLM_ENABLE_ROCE_ALLREDUCE"] == "0" and env["SPARKRING_TRANSPORT_PROFILE"] == ""
    assert env["SPARK_TP4_ENABLED"] == "0" and env["VLLM_PLUGINS"] == "b12x_loader"
    assert [k for k in env if k.startswith("SIRCL_")] == ["SIRCL_ENABLED"]


def test_the_s_n_diff_names_only_the_transport_parts():
    s = spec.render_arm("S", [base(0), base(1)], **COMMON)[0]
    n = spec.render_arm("N", [base(0), base(1)], **COMMON)[0]
    d = spec.diff(s, n)
    assert d["other_tokens_equal"]
    assert set(d["environment"]) == {"NCCL_IB_HCA", "NCCL_SKIP_TREE_CONNECT", "PYTHONPATH", "SIRCL_MODE", "SIRCL_NCCL",
                                     "SIRCL_FUSED_NORM", "VLLM_PLUGINS"}
    assert d["mounts_only_first"] == ["type=bind,src=/c/s,dst=/cache", "type=bind,src=/tmp/run,dst=/sircl/run"]


def test_refusals():
    with pytest.raises(spec.SpecError, match="NCCL may run"):
        spec.render_arm("N", [base(0), base(1)], **{**COMMON, "bundle_auto": bundle_doc("none", HOSTS)})
    with pytest.raises(spec.SpecError, match="prepared transport"):
        spec.render_arm("P", [base(0), base(1)], **COMMON)
    assert spec.prepared_allowed(4, [2, 3]) and spec.prepared_allowed(2, [0, 1]) and not spec.prepared_allowed(8, [0, 1])
    with pytest.raises(spec.SpecError, match="unknown arm"):
        spec.render_arm("X", [base(0)], **COMMON)


def test_order_labels():
    assert cli.order_of("W:S+,W:N,S+,N,N,S+") == [("W-S+", "S+", True), ("W-N", "N", True), ("S+1", "S+", False),
                                                  ("N1", "N", False), ("N2", "N", False), ("S+2", "S+", False)]
    assert cli.positions_of("0-3") == [0, 1, 2, 3] and cli.positions_of("2,3") == [2, 3]


S_LOG = ("INFO Loading plugin sircl\nSIRCL shim mhc_prefill_shard installed\nSIRCL shim worker_regimes installed\n"
         "INFO SIRCL receipt group=tp:0 global_rank=0 rank=0 nccl=none pynccl=skipped session=ring lanes=2\n")
DCP_LOG = (S_LOG + "SIRCL shim dcp_all_to_all installed\n"
           "INFO SIRCL receipt group=dcp:0 global_rank=0 rank=0 nccl=none pynccl=skipped session=chain lanes=2\n")
N_LOG = ("x NCCL INFO NET/IB : Using [0]a:1/RoCE\nx NCCL INFO ncclCommInitRank comm 0x1 rank 0 nranks 2 - Init COMPLETE\n"
         "INFO Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0' out of potential backends: []\n")


def test_arm_checks():
    s = verify.check("S", [S_LOG, S_LOG])
    assert s["passed"] and s["ranks"][0]["shims"] == ["mhc_prefill_shard", "worker_regimes"]
    assert not verify.check("S", [S_LOG + "x NCCL INFO comm\n"])["passed"]
    assert verify.check("N", [N_LOG])["passed"] and not verify.check("N", [S_LOG])["passed"]
    sp = verify.check("S+", [S_LOG], [{"fused_norm": "off"}])
    assert not sp["passed"] and "fused_norm" in sp["problems"][0]
    assert verify.check("S+", [S_LOG], [{"fused_norm": "on", "fused_norm_detail": {"provider": "vllm_c"}}])["passed"]


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="needs a POSIX shell and paths")
def test_set_locks_exclude_overlapping_sets_and_the_global_lock(tmp_path):
    g, s = (tmp_path / "global").as_posix(), (tmp_path / "sets").as_posix()

    def sh(script):
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.strip()
    assert sh(setlock.acquire_script("a", [0, 1], global_dir=g, sets_dir=s)) == "ACQUIRED"
    assert sh(setlock.acquire_script("b", [1, 2], global_dir=g, sets_dir=s)).startswith("BUSY")
    assert not (tmp_path / "sets" / "2").exists()          # no partial set left behind
    assert sh(setlock.acquire_script("c", [2, 3], global_dir=g, sets_dir=s)) == "ACQUIRED"
    sh(setlock.release_script("b", [0, 1], sets_dir=s))     # another owner's release changes nothing
    assert (tmp_path / "sets" / "0").exists()
    sh(setlock.release_script("a", [0, 1], sets_dir=s))
    assert not (tmp_path / "sets" / "0").exists()
    (tmp_path / "global").mkdir()
    (tmp_path / "global" / "owner").write_text("other 9999999999\n")
    assert sh(setlock.acquire_script("d", [0], global_dir=g, sets_dir=s)) == "BUSY global other"
    (tmp_path / "global" / "owner").write_text("other 1\n")  # stale
    assert sh(setlock.acquire_script("d", [0], global_dir=g, sets_dir=s)) == "ACQUIRED"


def write_start(root: Path, label: str, steps: float, ttft: float, tokens: list[str]):
    d = root / label
    d.mkdir()
    (d / "ready.json").write_text(json.dumps({"ready_seconds": 180}))
    (d / "decode.json").write_text(json.dumps({"results": [
        {"context_tokens": 0, "concurrency": 1, "aggregate_tps": steps * 2.7, "server_steps_per_s": steps,
         "server_spec_accept_length": 2.7}]}))
    (d / "prefill.jsonl").write_text(json.dumps({"metadata": {}}) + "\n" + json.dumps(
        {"target_tokens": 8192, "ttft_seconds": ttft, "cached_tokens_reported": 0}) + "\n")
    (d / "fingerprint.json").write_text(json.dumps({"fingerprints": {"math": {"tokens": tokens}}}))
    (d / "logprobs.json").write_text(json.dumps({"positions": [{"1": {"logprob": -0.5, "rank": 1}}]}))


def test_report_tables(tmp_path):
    write_start(tmp_path, "S+1", 14.0, 4.1, ["3", "9", "1"])
    write_start(tmp_path, "N1", 13.0, 4.2, ["3", "9", "2"])
    doc = report.summary(tmp_path, ["S+1", "N1"])
    assert doc["arms"] == ["S+", "N"]
    assert doc["decode"][0]["steps_per_s"]["S+"]["median"] == 14.0
    assert doc["outputs"][0]["fingerprints"]["math"] == "diverges at token 2"
    text = report.tables(doc)
    assert "| 0k | 1 | 14.00 (14.00-14.00) | 13.00 (13.00-13.00) | 1.077 |" in text


def test_fused_calls_come_from_the_receipt_decisions():
    receipt = {"fused_norm": "on", "fused_norm_detail": {"provider": "vllm_c"},
               "decisions": [{"collective": "all_reduce", "method": "direct", "calls": 9},
                             {"collective": "all_reduce", "method": "fused_rms_norm", "calls": 120}]}
    assert verify.fused_calls(receipt) == {"fused_norm": "on", "provider": "vllm_c", "fused_calls": 120}


def test_decode_context_checks_need_a_dcp_session_and_the_all_to_all_shim():
    assert verify.check("S", [DCP_LOG], dcp=2)["passed"]
    problems = verify.check("S", [S_LOG], dcp=2)["problems"]
    assert any("decode-context-parallel receipt" in p for p in problems)
    assert any("dcp_all_to_all" in p for p in problems)


def test_set_arg_replaces_or_appends_after_the_image():
    tokens = base(0)
    assert spec.set_arg(tokens, "--node-rank", "3") == "0" and spec.arg(tokens, "--node-rank") == "3"
    assert spec.set_arg(tokens, "--decode-context-parallel-size", "2") is None
    assert tokens[-2:] == ["--decode-context-parallel-size", "2"]


def test_memory_settles_after_dropping_caches_and_refuses_leftovers_and_short_sparks(tmp_path, monkeypatch):
    from performance.harnesses.serving_ab import remote

    sparks = tuple(remote.Spark(p, f"spark{p}", f"host{p}", f"192.0.2.{p}", ("docker",)) for p in (0, 1))
    site = remote.Site("site.json", "lan0", "/srv/run", sparks)
    tokens = base(0)
    spec.set_arg(tokens, "--gpu-memory-utilization", "0.25")
    plan = {"positions": [0, 1], "run_id": "run", "commands": {"S": [tokens, tokens]}}
    calls, available, left = [], iter([100, 200, 300, 300, 300, 300]), {"count": "0"}

    def fake_run(spark, command, **_):
        calls.append((spark.position, command))
        if "docker ps" in command:
            return left["count"] + "\n"
        if "meminfo" in command:
            return f"1048576 {next(available) * 1024}\n"
        return ""
    monkeypatch.setattr(remote, "run", fake_run)
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    rows = cli.settle_memory(plan, site, "S", tmp_path)
    assert [r["available_gib"] for r in rows] == [0.29, 0.29] and rows[0]["vllm_asks_gib"] == 0.25
    assert sum("drop_caches" in c for _, c in calls) == 2
    assert "memory before arm S (settled" in (tmp_path / "campaign.log").read_text()
    left["count"] = "2"
    with pytest.raises(RuntimeError, match="containers of run run are left"):
        cli.settle_memory(plan, site, "S", tmp_path)
    left["count"] = "0"
    spec.set_arg(tokens, "--gpu-memory-utilization", "0.9")
    available = iter([100] * 6)
    with pytest.raises(RuntimeError, match="below what vLLM asks on Spark 0"):
        cli.settle_memory(plan, site, "S", tmp_path)


def test_decode_cells_without_speculation_take_one_token_per_request_and_step():
    plain = {"aggregate_tps": 60.0, "server_steps_per_s": 0.0, "server_spec_accept_length": 0.0,
             "server_spec_drafts": 0, "server_gen_throughput": 59.0, "avg_running_reqs": 4}
    assert report.decode_cell(plain) == (60.0, 60.0, 1.0)
    assert report.arm_of("W-S+") == "S+" and report.arm_of("N2") == "N"


def test_overrides_change_every_rank_and_record_the_profile_value(tmp_path):
    bases = [base(0), base(1)]
    (tmp_path / "spec.json").write_text('{"method":"mtp","num_speculative_tokens":3}')
    changed, applied = cli.overrides(bases, [f"--speculative-config=@{tmp_path / 'spec.json'}", "--node-rank=5",
                                             "--async-scheduling"],
                                     ["VLLM_B12X_KDA_PREFILL_COALESCING=0"])
    assert all(spec.arg(t, "--speculative-config") == '{"method":"mtp","num_speculative_tokens":3}' for t in bases)
    assert spec.environment(bases[1])["VLLM_B12X_KDA_PREFILL_COALESCING"] == "0"
    assert changed[0].endswith("(the profile's: unset)") and changed[1].endswith("(the profile's: 0)")
    assert applied["environment"] == {"VLLM_B12X_KDA_PREFILL_COALESCING": "0"}
    assert all(t[-1] == "--async-scheduling" for t in bases) and changed[2] == "--async-scheduling (the profile's: unset)"


def test_choose_model_takes_a_listed_checkpoint_or_another_profiles_model():
    profile, _ = cli.profile_config("qwen38-flash-next-tp2")
    chosen, deviations = cli.choose_model(profile, "qad-step5500-mxfp8-attention", None)
    assert chosen["model"]["revision"].startswith("648b194a") and deviations[0].startswith("checkpoint qad-step5500")
    glm, _ = cli.profile_config("glm53-flash-nvfp4-spark-tp4")
    csf, deviations = cli.choose_model(glm, None, "glm53-flash-csf-tp8")
    assert csf["model"]["revision"].startswith("dec48abd") and csf["vllm_args"] == glm["vllm_args"]
    assert "of glm53-flash-csf-tp8" in deviations[0]


def test_sircl_arm_bundles_take_the_profiles_sircl_settings_from_the_repository(monkeypatch):
    """The SIRCL arms' bundles read the SIRCL switches the profile pins (bundle --profile), as the installer does."""
    from types import SimpleNamespace
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps({"nccl": "none", "ranks": []}), "")
    monkeypatch.setattr(cli.subprocess, "run", run)
    args = SimpleNamespace(capacity=None, dispatch=None, tuning_table=[], profile="glm53-nvfp4-tp8")
    cli.bundle("site.json", list(range(8)), "run-s", "never", args, dcp=4)
    cli.bundle("site.json", list(range(8)), "run-auto", "auto", args)
    never, auto = seen
    assert never[never.index("--repository") + 1] == str(cli.ROOT) and never[never.index("--profile") + 1] == (
        "glm53-nvfp4-tp8")
    assert "--profile" not in auto and "--fused-norm" not in never


def test_the_nccl_arm_removes_the_sircl_plugin_a_profile_lists():
    bases = [base(0), base(1)]
    for tokens in bases:
        spec.set_env(tokens, "VLLM_PLUGINS", "b12x_loader,sparkring_status,sircl,glm_dsa_indexer_split")
    env = spec.environment(spec.render_arm("N", bases, **COMMON)[0])
    assert env["VLLM_PLUGINS"] == "b12x_loader,sparkring_status,glm_dsa_indexer_split"
    # The SIRCL arms keep it once.
    s = spec.environment(spec.render_arm("S", [base(0), base(1)], **COMMON)[0])
    assert s["VLLM_PLUGINS"].split(",").count("sircl") == 1


def test_arm_n_is_refused_where_a_decode_context_parallel_group_is_not_cabled(monkeypatch):
    from types import SimpleNamespace
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps({"nccl": "ring", "ranks": []}), "")
    monkeypatch.setattr(cli.subprocess, "run", run)
    args = SimpleNamespace(site="site.json", capacity=None, dispatch=None, tuning_table=[], profile="glm53-nvfp4-tp8")
    # Four consecutive Sparks of a ring of eight: the last and first share no cable.
    assert "positions [0, 1, 2, 3]" in cli.nccl_dcp_problem(8, list(range(8)), 4)
    with pytest.raises(SystemExit, match="arm N: decode-context parallelism 4"):
        cli.nccl_bundle(args, 8, list(range(8)), "run", 4)
    assert not seen
    assert cli.nccl_dcp_problem(8, list(range(8)), 2) is None and cli.nccl_dcp_problem(8, list(range(8)), 1) is None
    # The auto bundle takes the same decode-context parallelism as the SIRCL arms' bundles.
    cli.nccl_bundle(args, 8, list(range(8)), "run", 2)
    cli.nccl_bundle(args, 8, [0, 1], "run", 1)
    with_dcp, without = seen
    assert with_dcp[with_dcp.index("--dcp-size") + 1] == "2" and with_dcp[with_dcp.index("--nccl") + 1] == "auto"
    assert "--dcp-size" not in without


def test_the_plan_states_that_arm_s_runs_a_fused_norm_the_profile_pins():
    glm, _ = cli.profile_config("glm53-nvfp4-tp8")
    assert cli.fused_norm_notes(glm["environment"], ["S", "S+"]) == [
        "arm S runs SIRCL_FUSED_NORM=1, which the profile pins; arms S and S+ run the same SIRCL switches"]
    assert cli.fused_norm_notes(glm["environment"], ["S+", "N"]) == []
    qwen, _ = cli.profile_config("qwen38-flash-next-tp2")
    assert cli.fused_norm_notes(qwen["environment"], ["S", "S+"]) == []
