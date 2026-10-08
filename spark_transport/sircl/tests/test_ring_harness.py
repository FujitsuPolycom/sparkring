"""The ring harness offline: site checks, plans, commands, counters and result merging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparkring_sircl import env, routes
from sparkring_sircl.ring import cli, counters, plan, remote, summary, worker
from sparkring_sircl.ring.site import Site, SiteError


def _site_document(size: int = 8) -> dict:
    return {
        "schema": "sircl-ring-site/v1", "image": "sha256:0123456789ab", "lan_interface": "lan0",
        "control_port": 29650, "remote_dir": "/tmp/sircl-ring",
        "ring": [{"name": f"spark{i}", "ssh": f"op@192.0.2.{10 + i}", "lan_address": f"192.0.2.{10 + i}"}
                 for i in range(size)],
    }


def _site(size: int = 8) -> Site:
    return Site.from_json(_site_document(size))


def test_site_validation():
    example = json.loads((Path(plan.PACKAGE) / "ring" / "site.example.json").read_text())
    with pytest.raises(SiteError, match="placeholder"):
        Site.from_json(example)
    document = {"schema": "sircl-ring-site/v1", "image": "img", "lan_interface": "lan0", "control_port": 29650,
                "remote_dir": "/tmp/x", "ring": [{"ssh": "a@h1", "lan_address": "192.0.2.1"},
                                                  {"ssh": "a@h2", "lan_address": "192.0.2.1"}]}
    with pytest.raises(SiteError, match="appears twice"):
        Site.from_json(document)
    document["ring"][1]["ssh"] = "a@h2; rm -rf /"
    with pytest.raises(SiteError):
        Site.from_json(document)


def test_builtin_configurations_on_an_eight_ring():
    site = _site()
    pairs = plan.build_plan(site, "pairs", "run1", digest="d" * 16)
    assert [g.positions for g in pairs.groups] == [(0, 1), (2, 3), (4, 5), (6, 7)]
    assert all(g.max_relays == 0 and g.relay_load == 0 for g in pairs.groups)
    path4 = plan.build_plan(site, "path4", "run1", digest="d" * 16)
    (group,) = path4.groups
    assert group.max_relays == 2 and group.relay_load == 1
    assert group.route_texts[0] == "1=rocep1s0f0/roceP2p1s0f0,2=rocep1s0f0/roceP2p1s0f0,3=rocep1s0f0/roceP2p1s0f0"
    assert group.route_texts[3] == "0=rocep1s0f1/roceP2p1s0f1,1=rocep1s0f1/roceP2p1s0f1,2=rocep1s0f1/roceP2p1s0f1"
    two = plan.build_plan(site, "two-tp4", "run1", digest="d" * 16)
    assert [g.global_ranks for g in two.groups] == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert two.ranks[4].host == "spark4" and two.ranks[4].group_rank == 0
    ring = plan.build_plan(site, "ring8", "run1", digest="d" * 16)
    (whole,) = ring.groups
    assert whole.max_relays == 3 and whole.relay_load == 3
    assert whole.route_texts == tuple(routes.derive_routes(routes.Layout.parse("ring:8")).route_text(r)
                                      for r in range(8))
    assert any("all-gather sizes from" in warning for warning in whole.warnings)
    assert ring.capacity == 131072 and ring.gather_capacity == 155648


def test_groups_must_be_consecutive_or_the_whole_ring():
    site = _site()
    with pytest.raises(plan.PlanError, match="consecutive"):
        plan.build_plan(site, "custom", "run1", groups=[(0, 2, 4, 6)], digest="d" * 16)
    with pytest.raises(plan.PlanError, match="two groups"):
        plan.build_plan(site, "custom", "run1", groups=[(0, 1), (1, 2)], digest="d" * 16)
    wrapped = plan.build_plan(site, "custom", "run1", groups=plan.parse_groups("6-7,0-1"), digest="d" * 16)
    assert wrapped.groups[0].layout.startswith("cables=6.port0-7.port1,7.port0-0.port1,0.port0-1.port1")
    with pytest.raises(plan.PlanError, match="relays"):
        plan.build_plan(site, "custom", "run1", groups=[(0, 1, 2, 3, 4, 5)], digest="d" * 16)


def test_container_commands_quote_and_label():
    built = plan.build_plan(_site(), "pairs", "run1", digest="abcdef0123456789")
    command = remote.docker_run(built, built.ranks[0])
    for part in ("--gpus device=0", "--network host", "--ulimit memlock=-1", "--label sircl-ring=run1",
                 "PYTHONPATH=/sircl/src/abcdef0123456789", "GLOO_SOCKET_IFNAME=lan0", "sha256:0123456789ab",
                 "sparkring_sircl.ring.worker --plan /sircl/runs/run1/pairs/plan.json --global-rank 0"):
        assert part in command
    assert "label=sircl-ring=run1" in remote.remove_harness_containers("run1")
    assert built.to_json()["rendezvous"] == "tcp://192.0.2.10:29650"


def test_plan_and_print_contact_nothing(monkeypatch, tmp_path, capsys):
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps({
        "schema": "sircl-ring-site/v1", "image": "img", "lan_interface": "lan0", "control_port": 29650,
        "remote_dir": "/tmp/sircl-ring",
        "ring": [{"name": f"s{i}", "ssh": f"op@192.0.2.{i + 1}", "lan_address": f"192.0.2.{i + 1}"} for i in range(8)],
    }))

    def refuse(*_args, **_kwargs):
        raise AssertionError("contacted a host")

    monkeypatch.setattr(remote, "ssh", refuse)
    assert cli.main(["plan", "--site", str(site_file)]) == 0
    assert cli.main(["run", "--site", str(site_file), "--config", "ring8", "--print"]) == 0
    text = capsys.readouterr().out
    assert "configuration two-tp4" in text and "SIRCL_PEER_ROUTES=" in text and "via Sparks [1, 2]" in text


def test_counters_from_a_sysfs_tree(tmp_path):
    for device in ("rocep1s0f0", "roceP2p1s0f0"):
        hw = tmp_path / device / "ports" / "1" / "hw_counters"
        hw.mkdir(parents=True)
        (hw / "out_of_sequence").write_text("5\n")
        (hw / "rx_write_requests").write_text("100\n")
        (tmp_path / device / "device" / "net" / f"net-{device}").mkdir(parents=True)
    before = counters.snapshot(["rocep1s0f0", "roceP2p1s0f0"], root=tmp_path)
    (tmp_path / "rocep1s0f0" / "ports" / "1" / "hw_counters" / "out_of_sequence").write_text("9\n")
    (tmp_path / "rocep1s0f0" / "ports" / "1" / "hw_counters" / "rx_write_requests").write_text("150\n")
    after = counters.snapshot(["rocep1s0f0", "roceP2p1s0f0"], root=tmp_path)
    changes = counters.delta(before, after)
    assert changes == {"rocep1s0f0": {"hw_counters/out_of_sequence": 4, "hw_counters/rx_write_requests": 50}}
    assert counters.key_deltas(changes) == {"rocep1s0f0:hw_counters/out_of_sequence": 4}
    assert before["rocep1s0f0"]["netdev"] == "net-rocep1s0f0"
    parsed = counters.parse_ethtool_text("NIC statistics:\n     rx_out_of_buffer: 7\n     tx_hairpin_drops: 2\n")
    assert parsed == {"rx_out_of_buffer": 7, "tx_hairpin_drops": 2}
    assert counters.is_key("tx_hairpin_drops") and counters.is_key("rx_out_of_buffer")


def _rank_result(rank, group, host, runs, error=None, code=0):
    return {"global_rank": rank, "group": group, "host": host, "runs": runs, "error": error, "exit_code": code}


def test_results_merge_into_slowest_rank_percentiles():
    built = plan.build_plan(_site(), "pairs", "run1", digest="d" * 16).to_json()
    run = {"collective": "all_reduce", "mode": "eager", "shape": [8], "dim": 0, "bytes": 16, "correct": True,
           "mismatched_calls": 0, "checked": 4, "p50_us": 10.0, "counters": {}}
    results = []
    for rank in range(8):
        times = [10.0 + rank, 20.0, 30.0 + rank]
        results.append(_rank_result(rank, rank // 2, f"spark{rank}", [{**run, "times_us": times}]))
    merged = summary.merge(built, results)
    assert merged["status"] == "passed"
    first = merged["cases"][0]
    assert first["slowest_p50_us"] == 20.0 and first["slowest_p99_us"] == 31.0
    results[3]["runs"][0] = {**results[3]["runs"][0], "correct": False, "mismatched_calls": 1,
                             "counters": {"rocep1s0f0:hw_counters/out_of_sequence": 3}}
    results[5] = None
    merged = summary.merge(built, results)
    assert merged["status"] == "failed"
    assert any("rank 5" in p for p in merged["problems"])
    assert any("calls differ" in p for p in merged["problems"])
    assert "NO" in summary.table(merged)


def _lan_preflight_output(position: int, extra: str = "") -> str:
    lines = ["docker\t27.3.1", "image\tsha256:abc", "gpu\t0, NVIDIA GB10",
             f"lan\t192.0.2.{10 + position}/24", f"address:lan0\t192.0.2.{10 + position}/24",
             f"address:wlP9s9\t192.0.2.{60 + position}/24",
             f"address:enx00e04c68000{position}\t192.0.2.{70 + position}/24"]
    for role in routes.ROLES:
        lines.append(f"device:{role.device}\t4: ACTIVE {role.netdev}")
    return "\n".join(lines) + "\n" + extra


def test_preflight_ignores_non_fabric_interfaces_on_the_lan(monkeypatch):
    document = _site_document()
    for index, entry in enumerate(document["ring"]):
        entry["lan_address"] = f"192.0.2.{10 + index}"
    site = Site.from_json(document)
    positions = {host.ssh: index for index, host in enumerate(site.ring)}
    extra: dict[int, str] = {}

    def fake_ssh(target, command, *, timeout=60, input_bytes=None, binary="ssh"):
        if "info --format" in command:
            position = positions[target]
            return remote.Result(0, _lan_preflight_output(position, extra.get(position, "")), "")
        return remote.Result(0, "", "")

    monkeypatch.setattr(remote, "ssh", fake_ssh)
    plans = [plan.build_plan(site, "pairs", "r1", digest="d" * 16)]
    _, lines = cli.preflight(site, plans, force=False)
    assert not any("lies in the subnet" in line for line in lines)
    extra[2] = "address:enP2p1s0f0np0\t192.0.2.200/24\n"      # a fabric interface on the LAN subnet
    _, lines = cli.preflight(site, plans, force=False)
    flagged = [line for line in lines if "lies in the subnet" in line]
    assert len(flagged) == 1 and flagged[0].startswith("BLOCKER: spark2:")
    assert "fabric interface enP2p1s0f0np0" in flagged[0]


def test_fabric_netdevs_include_reported_names():
    values = {"device:rocep1s0f0": "4: ACTIVE fab0", "device:roceP2p1s0f0": "missing"}
    names = cli.fabric_netdevs(values)
    assert "fab0" in names and {role.netdev for role in routes.ROLES} <= names
    assert "wlP9s9" not in names and "missing" not in names


def test_large_cases_and_placement_options_reach_the_plan(tmp_path, monkeypatch, capsys):
    options = plan.Options(large=True, cpu_policy="none", forward_window=0)
    built = plan.build_plan(_site(), "path4", "run1", options=options, digest="d" * 16)
    assert built.capacity == plan.LARGE_CAPACITY and built.gather_capacity == plan.LARGE_CAPACITY
    document = built.to_json()
    assert document["options"]["large"] and document["options"]["cpu_policy"] == "none"
    assert list(document["options"]["large_allreduce_sizes"]) == list(plan.LARGE_ALLREDUCE_SIZES)
    text = plan.render_text(built)
    assert "CPU placement none" in text and "forward windows off" in text and "all_reduce_large above" in text
    assert "no forward window" in text
    default = plan.build_plan(_site(), "path4", "run1", digest="d" * 16)
    assert default.capacity == 131072
    windows = {(d["rank"], d["peer"], d["lane"]): d["forward_window"] for d in default.groups[0].lanes_detail}
    assert windows[(0, 1, 0)] == 0 and windows[(0, 3, 1)] == 131072
    assert "forward window 131072 bytes" in plan.render_text(default)
    with pytest.raises(plan.PlanError, match="CPU policy"):
        plan.Options(cpu_policy="fastest")
    with pytest.raises(plan.PlanError, match="multiples of 16"):
        plan.Options(large_allreduce_sizes=(1000,))
    monkeypatch.setattr(remote, "ssh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("contacted a host")))
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    assert cli.main(["run", "--site", str(site_file), "--config", "path4", "--print", "--large",
                     "--cpu-policy", "none", "--forward-window", "65536"]) == 0
    assert "forward windows up to 65536 bytes" in capsys.readouterr().out


def test_bandwidths_follow_the_bus_conventions():
    from sparkring_sircl.ring import worker

    reduce = worker.bandwidths("all_reduce_large", 64 << 20, 4, 6710.886)
    assert reduce == {"algbw_gbps": 10.0, "busbw_gbps": 15.0}
    gather = worker.bandwidths("all_gather_large", 1 << 20, 4, 419.4304)
    assert gather == {"algbw_gbps": 10.0, "busbw_gbps": 7.5}
    assert worker.bandwidths("transport", 1000, 4, 1.0)["busbw_gbps"] == 3.0
    assert worker.bandwidths("all_reduce", 16, 1, 1.0) == {} and worker.bandwidths("all_reduce", 16, 4, None) == {}
    built = plan.build_plan(_site(), "pairs", "run1", digest="d" * 16).to_json()
    run = {"collective": "all_reduce_large", "mode": "graph", "shape": [1 << 25], "dim": 0, "bytes": 64 << 20,
           "correct": True, "mismatched_calls": 0, "checked": 2, "counters": {}, "algorithm": "pieces of 2097152",
           "placement": {"main_cpu_start": 5, "main_cpu_end": 5, "proxy_cpu": 19, "proxy_migrations": 0}}
    results = [_rank_result(rank, rank // 2, f"spark{rank}", [{**run, "times_us": [6710.886] * 3}])
               for rank in range(8)]
    merged = summary.merge(built, results)
    case = merged["cases"][0]
    assert case["algbw_gbps"] == 10.0 and case["busbw_gbps"] == 10.0     # pairs: 2 (W - 1) / W = 1
    assert case["rank_placement"][0]["proxy_cpu"] == 19 and case["algorithm"] == "pieces of 2097152"
    assert "10.00" in summary.table(merged)


def test_the_path4_large_configuration_measures_large_all_reduces(tmp_path, monkeypatch, capsys):
    options = plan.configuration_options("path4-large", plan.Options())
    built = plan.build_plan(_site(), "path4-large", "run1", options=options, digest="d" * 16)
    assert [group.positions for group in built.groups] == [(0, 1, 2, 3)]
    assert built.options.large and built.options.large_only
    assert built.options.large_allreduce_sizes == plan.PATH4_LARGE_SIZES == (1 << 20, 8 << 20, 32 << 20, 64 << 20)
    assert built.options.large_allgather_sizes == () and built.options.targets == ((64 << 20, 8.0, 5.0),)
    assert plan.configuration_options("path4", plan.Options()) == plan.Options()
    text = plan.render_text(built)
    assert "target: 67108864 bytes all-reduced in at most 8 ms (stretch 5 ms)" in text
    assert "large cases only" in text and "300 s during setup and warm-up, 20 s for the timed cases" in text
    with pytest.raises(plan.PlanError, match="wait limits"):
        plan.Options(serving_wait_s=30.0, startup_wait_s=10.0)
    monkeypatch.setattr(remote, "ssh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("contacted a host")))
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    assert cli.main(["run", "--site", str(site_file), "--config", "path4-large", "--print",
                     "--large-piece", "4194304", "--serving-wait", "5"]) == 0
    printed = capsys.readouterr().out
    assert "large-message pieces 4194304 bytes" in printed and "5 s for the timed cases" in printed
    assert "large all-reduce schedules ['pieces', 'chain', 'ring']; chain chunks [262144, 524288, 1048576]" in printed
    assert "chain target: 67108864 bytes all-reduced in at most 4 ms (stretch 3.3 ms)" in printed
    assert cli.main(["plan", "--site", str(site_file), "--config", "path4-large", "--json", "--host-cap-gbps", "20",
                     "--host-recv-gbps", "25"]) == 0
    options = json.loads(capsys.readouterr().out)["options"]
    assert (options["host_send_gbps"], options["host_recv_gbps"]) == (20.0, 25.0)
    assert cli.main(["plan", "--site", str(site_file), "--config", "path4-large,two-tp4-large", "--json",
                     "--large-schedules", "chain", "--chain-chunks", "131072,262144"]) == 0
    printed, documents = capsys.readouterr().out, []
    while printed.strip():
        document, end = json.JSONDecoder().raw_decode(printed.strip())
        documents.append(document)
        printed = printed.strip()[end:]
    assert [document["configuration"] for document in documents] == ["path4-large", "two-tp4-large"]
    for document in documents:
        assert document["options"]["large_schedules"] == ["chain"]
        assert document["options"]["chain_chunks"] == [131072, 262144]
    slot_default = next(variable.default for variable in env.VARIABLES if variable.name == "SIRCL_CHAIN_SLOT_BYTES")
    assert plan.CHAIN_SLOT_BYTES == int(slot_default)
    with pytest.raises(plan.PlanError, match="up to the chain slot"):
        plan.Options(chain_chunks=(2 << 20,))
    run = {"collective": "all_reduce_large", "mode": "graph", "shape": [1 << 25], "dim": 0, "bytes": 64 << 20,
           "correct": True, "mismatched_calls": 0, "checked": 2, "counters": {}, "target_ms": 8.0,
           "stretch_ms": 5.0}
    assert built.options.scatter_schedules == ("pieces", "chain", "ring")
    assert built.groups[0].ring.startswith("ring over [0, 1, 2, 3]: rank 3 reaches rank 0 through the relays of Sparks "
                                           "[1, 2], one ring lane per relay hairpin queue")
    assert ("all_reduce", 64 << 20, 4.5, 4.3) in built.options.ring_targets
    assert "bounds: a Spark's NIC host interface sends 24 GB/s and receives 26.8 GB/s" in plan.render_text(built)
    chain_run = {"collective": "all_reduce_large", "mode": "eager", "shape": [1 << 25], "dim": 0,
                 "bytes": 64 << 20, "correct": True, "mismatched_calls": 0, "checked": 1, "counters": {},
                 "algorithm": "chain, chunks of 524288", "traffic": "chain", "chain_order": [0, 1, 2, 3],
                 "times_us": [6000.0] * 3}
    merged = summary.merge(built.to_json(), [_rank_result(rank, 0, f"spark{rank}", [chain_run])
                                             for rank in range(4)])
    case = merged["cases"][0]
    assert case["bound_ms"] == pytest.approx(5.5924, abs=1e-3) and case["of_bound"] == pytest.approx(0.932, abs=1e-3)
    assert "[bound 5.59 ms, 93 % of it]" in summary.table(merged)
    # The chain reduce-scatter is receive-bound (1.25M into a middle rank): 3.13 ms at 26.8 GB/s, 3.5 ms when
    # a plan names one host rate for both directions.
    scatter_run = {**chain_run, "collective": "reduce_scatter", "algorithm": "chain reduce-scatter"}
    for options, bound in (({}, 3.1302), ({"host_cap_gbps": 24.0}, 3.4953)):
        document = built.to_json()
        document["options"] = {key: value for key, value in document["options"].items()
                               if key not in ("host_send_gbps", "host_recv_gbps")} | options
        merged = summary.merge(document, [_rank_result(rank, 0, f"spark{rank}", [scatter_run]) for rank in range(4)])
        assert merged["cases"][0]["bound_ms"] == pytest.approx(bound, abs=1e-3)
    for times, verdict in (([7000.0] * 3, "target met"), ([4800.0] * 3, "stretch met"),
                           ([9000.0] * 3, "target missed")):
        results = [_rank_result(rank, 0, f"spark{rank}", [{**run, "times_us": times}]) for rank in range(4)]
        merged = summary.merge(built.to_json(), results)
        assert merged["cases"][0]["target_met"] == (times[0] <= 8000.0)
        assert verdict in summary.table(merged)


def test_ring_large_and_session_variables(tmp_path, capsys):
    options = plan.configuration_options("ring8-large", plan.Options(session_env=(("SIRCL_CHAIN_SLOTS", "8"),)))
    built = plan.build_plan(_site(), "ring8-large", "run1", options=options, digest="d" * 16)
    assert [group.positions for group in built.groups] == [tuple(range(8))]
    assert built.options.large_only and built.options.large_allreduce_sizes == plan.RING_LARGE_SIZES
    assert built.options.large_schedules == ("pieces", "chain", "ring") and built.options.chain_chunks
    assert "session variables on every rank: SIRCL_CHAIN_SLOTS=8" in plan.render_text(built)
    assert json.loads(json.dumps(built.to_json()))["options"]["session_env"] == [["SIRCL_CHAIN_SLOTS", "8"]]
    for pairs, message in (((("SIRCL_LARGE_SCHEDULE", "chain"),), "--large-schedules"),
                           ((("SIRCL_NOT_A_VARIABLE", "1"),), "not a documented"),
                           ((("SIRCL_CHAIN_SLOTS", "8;x"),), "characters other than")):
        with pytest.raises(plan.PlanError, match=message):
            plan.Options(session_env=pairs)
    with pytest.raises(plan.PlanError, match="up to the chain slot of 1048576"):
        plan.Options(chain_chunks=(2 << 20,))
    assert plan.Options(chain_chunks=(2 << 20,), session_env=(("SIRCL_CHAIN_SLOT_BYTES", str(2 << 20)),))
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring-large", "--session-env",
                     "SIRCL_CHAIN_BLOCKS=8", "--session-env", "SIRCL_CHAIN_SLOTS=8"]) == 0
    assert "SIRCL_CHAIN_BLOCKS=8 SIRCL_CHAIN_SLOTS=8" in capsys.readouterr().out
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring-large", "--session-env", "BAD"]) == 2


def test_nccl_baseline_plans_on_pairs_and_cycles_only(tmp_path, capsys):
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    assert cli.main(["plan", "--site", str(site_file), "--groups", "0-1", "--name", "pair", "--large", "--json",
                     "--large-sizes", "8192,134217728", "--large-gather-sizes", "8192,1048576",
                     "--baseline", "nccl"]) == 0
    options = json.loads(capsys.readouterr().out)["options"]
    assert options["baseline"] == "nccl" and options["large_allgather_sizes"] == [8192, 1048576]
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring8-large", "--baseline", "nccl"]) == 0
    assert "NCCL baseline" in capsys.readouterr().out
    assert cli.main(["plan", "--site", str(site_file), "--groups", "0-3", "--name", "path4", "--baseline",
                     "nccl"]) == 2
    assert "pairs and whole cycles; group (0, 1, 2, 3) is a path" in capsys.readouterr().err
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring8-large", "--baseline", "nccl",
                     "--nccl-library", "lib/libnccl.so.2"]) == 2
    assert "absolute path" in capsys.readouterr().err


def test_nccl_environment_and_kinds():
    from sparkring_sircl.ring import nccl

    assert nccl.baseline_kind("path", 2) == "pair" and nccl.baseline_kind("cycle", 8) == "cycle"
    assert nccl.baseline_kind("path", 4) is None
    values = nccl.environment("enP7s7", None)
    assert values["NCCL_IB_HCA"] == "=rocep1s0f0:1,rocep1s0f1:1,roceP2p1s0f0:1,roceP2p1s0f1:1"
    assert values["NCCL_SOCKET_IFNAME"] == "enP7s7" and values["NCCL_IB_GID_INDEX"] == "3"
    assert values["LD_PRELOAD"] == "/opt/sparkring/toolchain/nccl/lib/libnccl.so.2"
    for name, value in (("NCCL_ALGO", "Ring"), ("NCCL_NET", "IB"), ("NCCL_SWITCHLESS_RING_ONLY", "1"),
                        ("NCCL_IB_EXTENDED_IPV4_GIDS", "1"), ("NCCL_MIN_NCHANNELS", "4"), ("NCCL_CUMEM_ENABLE", "0")):
        assert values[name] == value
    overridden = nccl.environment("lan0", 5, "/x/libnccl.so.2", [("NCCL_MAX_NCHANNELS", "8"), ("NCCL_X", "1")])
    assert overridden["NCCL_IB_GID_INDEX"] == "5" and overridden["LD_PRELOAD"] == "/x/libnccl.so.2"
    assert overridden["NCCL_MAX_NCHANNELS"] == "8" and overridden["NCCL_X"] == "1"
    with pytest.raises(nccl.NcclError):
        nccl.check_override("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(nccl.NcclError):
        nccl.check_override("NCCL_DEBUG", "INFO; rm")
    with pytest.raises(nccl.NcclError, match="cannot load NCCL"):
        nccl.Library("/nonexistent/libnccl.so.2")
    log = ("host:1:1 [0] NCCL INFO NET/IB : Using [0]rocep1s0f0:1/RoCE [1]rocep1s0f1:1/RoCE\n"
           "host:1:1 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0\n")
    assert nccl.transport(log) == "NET/IB" and nccl.transport("") is None
    assert nccl.transport(log + "NCCL INFO NET/Socket : Using [0]enP7s7\n") == "NET/Socket"
    maps = ("7f00-7f01 r-xp 0 0:1 5 /opt/sparkring/toolchain/nccl/lib/libnccl.so.2\n"
            "7f02-7f03 r--p 0 0:1 5 /opt/sparkring/toolchain/nccl/lib/libnccl.so.2\n7f04-7f05 rw-p 0 0:0 0\n")
    assert nccl.loaded_libraries(maps) == ["/opt/sparkring/toolchain/nccl/lib/libnccl.so.2"]


def test_nccl_environment_reaches_the_plan_and_the_containers(tmp_path, capsys):
    from sparkring_sircl.ring import remote

    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    line = ["plan", "--site", str(site_file), "--config", "ring8-large", "--baseline", "nccl", "--nccl-env",
            "NCCL_MAX_NCHANNELS=8"]
    assert cli.main(line) == 0
    text = capsys.readouterr().out
    assert "NCCL_IB_HCA==rocep1s0f0:1,rocep1s0f1:1,roceP2p1s0f0:1,roceP2p1s0f1:1" in text
    assert "NCCL_MAX_NCHANNELS=8" in text and "NCCL_SOCKET_IFNAME=lan0" in text
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring8-large", "--nccl-env", "NCCL_X=1"]) == 2
    assert "applies to the NCCL baseline" in capsys.readouterr().err
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring8-large", "--baseline", "nccl",
                     "--nccl-env", "PATH=/x"]) == 2
    capsys.readouterr()
    options = plan.Options(baseline="nccl", nccl_env=(("NCCL_MAX_NCHANNELS", "8"),))
    built = plan.build_plan(_site(), "ring8-large", "run1", digest="d" * 16,
                            options=plan.configuration_options("ring8-large", options))
    command = remote.docker_run(built, built.ranks[3])
    assert "LD_PRELOAD=/opt/sparkring/toolchain/nccl/lib/libnccl.so.2" in command
    assert "NCCL_MAX_NCHANNELS=8" in command and "NCCL_DEBUG=INFO" in command
    assert "NCCL_DEBUG_FILE=/sircl/runs/run1/ring8-large/nccl-rank-3.log" in command
    assert "NCCL_" not in remote.docker_run(plan.build_plan(_site(), "ring8-large", "run1", digest="d" * 16),
                                            built.ranks[3])


def test_a_slow_8k_pair_all_reduce_marks_nccl_rows_degraded():
    built = plan.build_plan(_site(), "pairs", "run1", digest="d" * 16).to_json()
    base = {"dim": 0, "correct": True, "mismatched_calls": 0, "checked": 1, "counters": {}, "mode": "eager",
            "shape": [4096], "bytes": 8192}
    runs = [{**base, "collective": "all_reduce", "times_us": [20.0], "algorithm": "oneshot"},
            {**base, "collective": "nccl_all_reduce", "times_us": [600.0], "algorithm": "NCCL 2.32.3 NET/Socket",
             "baseline": "nccl"}]
    results = [_rank_result(rank, rank // 2, f"spark{rank}", runs) for rank in range(8)]
    merged = summary.merge(built, results)
    assert merged["cases"][0]["nccl_degraded"] and merged["cases"][1]["nccl_degraded"]
    text = summary.table(merged)
    assert "[NCCL degraded]" in text and "warning: group 0: NCCL degraded" in text
    fast = [{**run, "times_us": [25.0]} if run["collective"] == "nccl_all_reduce" else run for run in runs]
    merged = summary.merge(built, [_rank_result(rank, rank // 2, f"spark{rank}", fast) for rank in range(8)])
    assert not merged["warnings"] and "nccl_degraded" not in merged["cases"][1]


def test_summarize_and_trace_accept_a_run_folder(tmp_path, capsys):
    built = plan.build_plan(_site(), "pairs", "run1", digest="d" * 16).to_json()
    folder = tmp_path / "run1" / "pairs"
    folder.mkdir(parents=True)
    (folder / "plan.json").write_text(json.dumps(built))
    run = {"collective": "all_reduce", "mode": "eager", "shape": [8], "dim": 0, "bytes": 16, "correct": True,
           "mismatched_calls": 0, "checked": 1, "counters": {}, "times_us": [10.0]}
    for rank in range(8):
        (folder / f"rank-{rank}.json").write_text(json.dumps(_rank_result(rank, rank // 2, f"spark{rank}", [run])))
    assert cli.main(["summarize", "--results", str(tmp_path / "run1")]) == 0
    assert "configuration pairs" in capsys.readouterr().out
    assert cli.main(["trace", "--results", str(folder)]) == 0
    capsys.readouterr()
    assert cli.main(["trace", "--results", str(tmp_path)]) == 2
    assert "holds no plan.json" in capsys.readouterr().err


def test_nccl_rows_pair_with_sircl_rows_in_the_summary():
    built = plan.build_plan(_site(), "pairs", "run1", digest="d" * 16).to_json()
    base = {"dim": 0, "correct": True, "mismatched_calls": 0, "checked": 1, "counters": {}}
    runs = [{**base, "collective": "all_reduce_large", "mode": "graph", "shape": [4194304], "bytes": 8388608,
             "times_us": [400.0], "algorithm": "ring"},
            {**base, "collective": "nccl_all_reduce", "mode": "graph", "shape": [4194304], "bytes": 8388608,
             "times_us": [500.0], "algorithm": "NCCL 2.32.3", "baseline": "nccl"},
            {**base, "collective": "all_gather_large", "mode": "graph", "shape": [512, 2048], "dim": 1,
             "bytes": 2097152, "times_us": [90.0]},
            {**base, "collective": "nccl_all_gather", "mode": "graph", "shape": [512, 2048], "bytes": 2097152,
             "times_us": [100.0], "algorithm": "NCCL 2.32.3", "baseline": "nccl"}]
    results = [_rank_result(rank, rank // 2, f"spark{rank}", runs) for rank in range(8)]
    merged = summary.merge(built, results)
    first = merged["cases"][0]
    assert first["nccl_p50_us"] == 500.0 and first["vs_nccl"] == 1.25
    assert "vs_nccl" not in merged["cases"][2]
    text = summary.table(merged)
    assert "vs NCCL" in text.splitlines()[1] and "1.25x" in text


def test_tune_plans_its_candidates_and_quick_sizes(tmp_path, capsys, monkeypatch):
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring8", "--tune", "--quick", "--json"]) == 0
    options = json.loads(capsys.readouterr().out)["options"]
    assert options["tune"] and options["large_only"] and options["tune_sizes"][0] == 4096
    assert options["tune_sizes"] == [4096 << shift for shift in range(0, 16, 2)]
    assert options["eager_iterations"] == 40 and options["large_iterations"] == 8
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring8", "--tune", "--tune-collectives",
                     "all_reduce", "--tune-sizes", "8192,4096"]) == 2
    assert "increasing" in capsys.readouterr().err
    assert cli.main(["plan", "--site", str(site_file), "--groups", "0-1", "--name", "pair", "--tune",
                     "--tune-collectives", "all_reduce,all_gather", "--baseline", "nccl"]) == 0
    assert "tune: all_reduce, all_gather at 16 sizes" in capsys.readouterr().out
    # tune --print prints the launch plan and contacts nothing.
    def refuse(*args, **kwargs):
        raise AssertionError("tune --print contacted a Spark")

    monkeypatch.setattr(remote, "ssh", refuse)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["tune", "--site", str(site_file), "--config", "ring8", "--quick", "--print"]) == 0
    assert "tune: all_reduce, all_gather, reduce_scatter, all_to_all at 8 sizes" in capsys.readouterr().out
    assert not (tmp_path / "sircl-ring-results").exists()


def test_tune_candidates_follow_the_session(monkeypatch):
    class Session:
        max_size = 2 << 20
        max_gather_bytes = 16 << 20
        chain_available = True
        chain_slot_bytes = 1 << 20
        link_available = True
        ring_available = True
        _available = {"oneshot": True, "twoshot": True, "swing": False}

    harness = worker.Harness.__new__(worker.Harness)
    harness.session, harness.world, harness.swing_through, harness.scatter_through = Session(), 8, None, "session"
    harness.options = {"tune_grids": [8, 32], "tune_pieces": [524288, 2097152], "tune_staggers": [0, 1],
                       "tune_large_from": 262144}
    small = harness.tune_candidates("all_reduce", 65536)
    assert [choice for choice, *_ in small] == [{"algorithm": "oneshot", "grid": 8}, {"algorithm": "oneshot", "grid": 32},
                                               {"algorithm": "twoshot", "grid": 8}, {"algorithm": "twoshot", "grid": 32}]
    assert small[2][1] == "all_reduce_twoshot" and small[2][3] == {"grid": 8}
    large = harness.tune_candidates("all_reduce", 8 << 20)
    labels = [json.dumps(choice, sort_keys=True) for choice, *_ in large]
    assert '{"grid": 8, "schedule": "pieces"}' in labels and '{"piece": 524288, "schedule": "chain"}' in labels
    assert '{"piece": 2097152, "schedule": "chain"}' not in labels    # above the chain slot
    assert '{"gather_stagger": 0, "piece": 2097152, "schedule": "ring", "stagger": 1}' in labels
    assert '{"gather_stagger": 1, "piece": 524288, "schedule": "ring", "stagger": 0}' in labels
    assert all(name == "all_reduce_large" for _, name, *_ in large)
    gathers = harness.tune_candidates("all_gather", 1 << 20)
    assert {choice.get("schedule") for choice, *_ in gathers} == {"pieces", "chain", "ring"}
    assert {(choice.get("gather_stagger"), variant.get("gather_stagger")) for choice, _, _, variant in gathers
            if choice.get("schedule") == "ring"} == {(0, 0), (1, 1)}
    assert [c for c, *_ in harness.tune_candidates("all_to_all", 65536)] == [{"grid": 8}, {"grid": 32}]
    assert harness.tune_candidates("reduce_scatter", 65536 + 16) == []
    harness.scatter_through = None
    assert harness.tune_candidates("reduce_scatter", 65536) == []


def test_tune_results_become_a_tuning_table(tmp_path, capsys):
    from sparkring_sircl import tuning

    built = plan.build_plan(_site(), "ring8", "run1", digest="d" * 16).to_json()
    base = {"dim": 0, "correct": True, "mismatched_calls": 0, "checked": 1, "counters": {}, "mode": "graph"}
    runs = []
    for size, one, two in ((8192, 20.0, 30.0), (65536, 40.0, 35.0), (262144, 90.0, 60.0)):
        for algorithm, micros in (("oneshot", one), ("twoshot", two)):
            runs.append({**base, "collective": f"all_reduce_{algorithm}", "shape": [size // 2], "bytes": size,
                         "times_us": [micros], "tune": {"collective": "all_reduce",
                                                        "choice": {"algorithm": algorithm, "grid": 8}}})
    folder = tmp_path / "run1" / "ring8"
    folder.mkdir(parents=True)
    (folder / "plan.json").write_text(json.dumps(built))
    for rank in range(8):
        (folder / f"rank-{rank}.json").write_text(json.dumps(_rank_result(rank, 0, f"spark{rank}", runs)))
    assert cli.main(["tune-table", "--results", str(tmp_path / "run1")]) == 0
    out = capsys.readouterr().out
    assert "tuning table of group 0" in out
    table = tuning.Table.load(folder / "tuning-group0.json")
    assert table.key["shape"] == "cycle:8" and table.key["world"] == 8 and table.key["image"] == "sha256:0123456789ab"
    assert table.decide("all_reduce", 8192, "graph").algorithm == "oneshot"
    assert table.decide("all_reduce", 262144, "graph").algorithm == "twoshot"
    assert table.decide("all_reduce", 4096, "graph") is None and table.decide("all_reduce", 8192, "eager") is None


def _tuning_table(tmp_path, name, built_group, *, grid=8):
    from sparkring_sircl import routes, tuning

    layout = routes.Layout.parse(built_group.layout)
    key = dict(tuning.facts(layout.identity(), layout.world, built_group.lanes, built_group.max_relays), image="img")
    rows = [{"collective": "all_reduce", "mode": "graph", "bytes": 8192, "choice": {"algorithm": "oneshot", "grid": grid},
             "p50_us": 9.0}]
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(tuning.build_document(key, rows)))
    return path, tuning.Table.load(path).hash


def test_runs_stage_tuning_tables_by_group(tmp_path, capsys):
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    two = plan.build_plan(_site(), "two-tp4", "run1", digest="d" * 16)
    ring = plan.build_plan(_site(), "ring8", "run1", digest="d" * 16)
    path4, path4_hash = _tuning_table(tmp_path, "path4", two.groups[0])
    ring8, ring8_hash = _tuning_table(tmp_path, "ring8", ring.groups[0])
    built = plan.build_plan(_site(), "two-tp4", "run1", digest="d" * 16,
                            options=plan.Options(tuning_tables=(str(path4), str(ring8))))
    assert [group.tuning_table for group in built.groups] == [path4_hash, path4_hash]
    assert built.tuning_tables == ((path4_hash, str(path4)), (ring8_hash, str(ring8)))
    command = remote.docker_run(built, built.ranks[0])
    assert (f"SIRCL_TUNING_TABLE=/sircl/runs/run1/two-tp4/tuning-{path4_hash}.json,"
            f"/sircl/runs/run1/two-tp4/tuning-{ring8_hash}.json") in command
    assert f"tuning table {path4_hash}" in plan.render_text(built)
    assert "SIRCL_TUNING_TABLE" not in remote.docker_run(two, two.ranks[0])
    # Every named table must match a group of some planned configuration.
    assert cli.main(["plan", "--site", str(site_file), "--config", "two-tp4", "--tuning-table", str(path4)]) == 0
    capsys.readouterr()
    assert cli.main(["plan", "--site", str(site_file), "--config", "two-tp4", "--tuning-table", str(path4),
                     "--tuning-table", str(ring8)]) == 2
    assert "no session of the planned configurations matches" in capsys.readouterr().err
    assert cli.main(["plan", "--site", str(site_file), "--config", "two-tp4,ring8", "--tuning-table", str(path4),
                     "--tuning-table", str(ring8)]) == 0
    other, _ = _tuning_table(tmp_path, "path4-other", two.groups[0], grid=16)
    with pytest.raises(plan.PlanError, match="several different tuning tables"):
        plan.build_plan(_site(), "two-tp4", "run1", digest="d" * 16,
                        options=plan.Options(tuning_tables=(str(path4), str(other))))
    with pytest.raises(plan.PlanError, match="tune run"):
        plan.Options(tune=True, tuning_tables=(str(path4),))
    with pytest.raises(plan.PlanError, match="--tuning-table"):
        plan.Options(session_env=(("SIRCL_TUNING_TABLE", "/x.json"),))


def test_summary_reports_each_groups_tuning_decisions(tmp_path):
    two = plan.build_plan(_site(), "two-tp4", "run1", digest="d" * 16)
    path4, digest = _tuning_table(tmp_path, "path4", two.groups[0])
    built = plan.build_plan(_site(), "two-tp4", "run1", digest="d" * 16,
                            options=plan.Options(tuning_tables=(str(path4),))).to_json()
    run = {"collective": "all_reduce", "mode": "graph", "shape": [4096], "dim": 0, "bytes": 8192, "correct": True,
           "mismatched_calls": 0, "checked": 4, "p50_us": 10.0, "counters": {}, "times_us": [10.0]}
    results = []
    for rank in range(8):
        result = _rank_result(rank, rank // 4, f"spark{rank}", [run])
        result["tuning"] = {"table": digest, "decisions": {"all_reduce/graph/oneshot grid 8": 12}, "unusable": {}}
        results.append(result)
    merged = summary.merge(built, results)
    assert merged["status"] == "passed"
    assert merged["tuning"][0] == {"group": 0, "planned": digest, "table": digest,
                                   "decisions": {"all_reduce/graph/oneshot grid 8": 12}, "unusable": {}}
    assert f"tuning, group 1: table {digest}; rank 0's decisions: all_reduce/graph/oneshot grid 8 x12" in summary.table(merged)
    results[5]["tuning"] = None
    merged = summary.merge(built, results)
    assert merged["status"] == "failed"
    assert any(problem.startswith("group 1: the plan names tuning table") for problem in merged["problems"])


def test_link_case_options_need_cases_to_apply_to(tmp_path, capsys):
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    base = ["plan", "--site", str(site_file), "--config", "ring8-large", "--json"]
    assert cli.main(base + ["--gather-schedules", "chain,ring", "--gather-link-chunks", "524288"]) == 2
    error = capsys.readouterr().err
    assert "--gather-schedules applies to no case" in error and "--chain-gather-sizes" in error
    assert cli.main(base + ["--scatter-schedules", "ring"]) == 2
    assert "--reduce-scatter-sizes" in capsys.readouterr().err
    assert cli.main(base + ["--large-schedules", "chain", "--reduce-link-chunks", "524288"]) == 2
    assert "--reduce-link-chunks applies to no case" in capsys.readouterr().err
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring8", "--chain-gather-sizes", "2097152"]) == 2
    assert "pass --large" in capsys.readouterr().err
    line = base + ["--chain-gather-sizes", "2097152,16777216", "--gather-schedules", "chain,ring",
                   "--gather-link-chunks", "1048576,2097152", "--reduce-scatter-sizes", "8388608",
                   "--scatter-schedules", "ring", "--session-env", "SIRCL_LINK_SLOT_BYTES=2097152"]
    assert cli.main(line) == 0
    options = json.loads(capsys.readouterr().out)["options"]
    assert options["chain_gather_sizes"] == [2097152, 16777216] and options["gather_link_chunks"] == [1048576, 2097152]
    assert options["large_reduce_scatter_sizes"] == [8388608] and options["scatter_schedules"] == ["ring"]
    harness = worker.Harness.__new__(worker.Harness)
    harness.options = options
    harness.session = type("Session", (), {"max_size": 2 << 20})()
    gathers = [variant for collective, shape, dim, variant in harness.large_cases()
               if collective == "all_gather_large" and shape == (16777216 // 8192, 4096)]
    assert gathers == [{"gather_schedule": schedule, "link_chunk": chunk} for schedule in ("chain", "ring")
                       for chunk in (1048576, 2097152)]


def test_forced_variants_and_tune_pruning():
    assert not worker.forced(None) and not worker.forced({"schedule": "auto"})
    assert not worker.forced({"post_order": "rank"}) and not worker.forced({"gather_schedule": "auto", "grid": None})
    assert worker.forced({"schedule": "chain", "chunk": None}) and worker.forced({"grid": 8})
    assert worker.forced({"scatter_schedule": "ring", "link_chunk": 262144, "stagger": 0})
    assert worker.tuning_family({"backend": "nccl"}) == "nccl" and worker.tuning_family({"schedule": "ring"}) == "ring"
    assert worker.tuning_family({"algorithm": "twoshot", "grid": 8}) == "twoshot" and worker.tuning_family({"grid": 4}) == "grid"


def test_tune_plan_lists_families_and_prunes_from_four_mib(tmp_path, capsys):
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring8", "--tune", "--baseline", "nccl"]) == 0
    text = capsys.readouterr().out
    assert "all_reduce: one-shot, two-shot, pieces, chain, ring, NCCL" in text
    assert "from 4194304 bytes on, its ratio not falling" in text
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring8", "--tune", "--tune-prune-from", "0",
                     "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["options"]["tune_prune_from"] == 0


def test_tune_coverage_names_families_missing_at_the_largest_size():
    built = plan.build_plan(_site(), "ring8", "run1", digest="d" * 16).to_json()
    base = {"dim": 0, "correct": True, "mismatched_calls": 0, "checked": 1, "counters": {}, "mode": "eager"}
    runs = []
    for size, choices in ((1 << 20, ({"schedule": "pieces", "grid": 8}, {"schedule": "chain", "piece": 524288},
                                    {"backend": "nccl"})),
                          (64 << 20, ({"schedule": "pieces", "grid": 8},))):
        for choice in choices:
            runs.append({**base, "collective": "all_reduce_large", "shape": [size // 2], "bytes": size,
                         "times_us": [100.0], "tune": {"collective": "all_reduce", "choice": choice}})
    results = []
    for rank in range(8):
        result = _rank_result(rank, 0, f"spark{rank}", runs)
        result["tune_families"] = {"all_reduce": ["chain", "nccl", "pieces", "ring"]}
        results.append(result)
    merged = summary.merge(built, results)
    warnings = [warning for warning in merged["warnings"] if warning.startswith("tune coverage")]
    assert "tune coverage, group 0 all_reduce eager: no chain measurement at 67108864 bytes, last measured at " \
           "1048576 bytes (pruned)" in warnings
    assert any("no ring measurement at 67108864 bytes, never measured" in warning for warning in warnings)
    assert any("no nccl measurement" in warning for warning in warnings)


def test_default_run_ids_name_the_process(tmp_path, capsys):
    import os

    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring8", "--json"]) == 0
    run_id = json.loads(capsys.readouterr().out)["run_id"]
    assert run_id.endswith(f"-{os.getpid()}") and len(run_id.split("-")) == 3


def test_eager_profile_and_adapter_path_reach_the_plan(tmp_path, capsys):
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    line = ["plan", "--site", str(site_file), "--groups", "0-1", "--name", "pair", "--eager-profile", "--eager-path",
            "adapter"]
    assert cli.main(line + ["--json"]) == 0
    options = json.loads(capsys.readouterr().out)["options"]
    assert options["eager_profile"] and options["eager_path"] == "adapter"
    assert cli.main(line) == 0
    assert "eager path: adapter; eager rows of all_reduce" in capsys.readouterr().out
    with pytest.raises(plan.PlanError, match="--eager-profile"):
        plan.Options(session_env=(("SIRCL_CALL_PROFILE", "10"),))


def test_large_blocks_sweep_reaches_the_plan(tmp_path, capsys):
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    line = ["plan", "--site", str(site_file), "--config", "ring8-latency", "--json", "--latency-sizes",
            "196608,393216", "--large-blocks", "4,8,16,32"]
    assert cli.main(line) == 0
    assert json.loads(capsys.readouterr().out)["options"]["large_blocks"] == [4, 8, 16, 32]
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring8-latency", "--large-blocks", "4,8"]) == 0
    assert "grid caps of two-shot and large-message launches [4, 8]" in capsys.readouterr().out
    for bad in ("6", "64", "4,4", "0"):
        assert cli.main(["plan", "--site", str(site_file), "--config", "ring8-latency", "--large-blocks", bad]) == 2
        assert "grid caps are distinct powers of two up to 32" in capsys.readouterr().err
    assert cli.main(["plan", "--site", str(site_file), "--config", "dcp4", "--large-blocks", "64",
                     "--session-env", "SIRCL_LARGE_BLOCKS=64"]) == 0
    capsys.readouterr()


def test_large_blocks_sweep_runs_each_grid_case_once_per_cap(monkeypatch):
    calls, caps = [], []

    class Session:
        post_order_peers = ()

        def enter_serving(self):
            pass

        def set_large_blocks(self, cap):
            caps.append(cap)

        def stats(self):
            return {}

        def close(self):
            pass

    class Dist:
        def barrier(self):
            pass

    harness = worker.Harness.__new__(worker.Harness)
    harness.options = {"large_blocks": [4, 32], "large_only": True, "latency_sizes": [393216],
                       "post_orders": ["rank"]}
    harness.session, harness.tp_session, harness.dist, harness.devices = Session(), None, Dist(), []
    harness.result = {"runs": []}
    harness.latency_orders = ("rank",)
    harness._built_order_peers = ()
    monkeypatch.setattr(worker.counters, "snapshot", lambda devices: {})
    monkeypatch.setattr(worker.counters, "delta", lambda before, after: {})
    monkeypatch.setattr(worker.counters, "key_deltas", lambda delta: {})
    monkeypatch.setattr(worker.Harness, "large_cases", lambda self: [])
    monkeypatch.setattr(worker.Harness, "dcp_cases", lambda self: [])
    monkeypatch.setattr(worker.Harness, "swing_cases", lambda self: [])

    def run_case(self, collective, mode, shape, dim, case, large=False, variant=None):
        calls.append((collective, mode, caps[-1]))
        return {"collective": collective, "mode": mode}

    monkeypatch.setattr(worker.Harness, "run_case", run_case)
    harness.run()
    assert caps == [4, 32]
    assert calls == [("all_reduce_oneshot", "eager", 4), ("all_reduce_twoshot", "eager", 4),
                     ("all_reduce_oneshot", "graph", 4), ("all_reduce_twoshot", "graph", 4),
                     ("all_reduce_twoshot", "eager", 32), ("all_reduce_twoshot", "graph", 32)]
    tags = [(run["collective"], run.get("large_blocks")) for run in harness.result["runs"]]
    assert tags[:2] == [("all_reduce_oneshot", None), ("all_reduce_twoshot", 4)]
    assert tags[-1] == ("all_reduce_twoshot", 32)


def test_grid_caps_reach_the_summary_and_split_the_crossover():
    built = plan.build_plan(_site(), "pairs", "run1", digest="d" * 16).to_json()
    base = {"mode": "graph", "dim": 0, "correct": True, "mismatched_calls": 0, "checked": 1, "counters": {},
            "post_order": "rank"}
    runs = [{**base, "collective": "all_reduce_oneshot", "algorithm": "oneshot", "shape": [98304],
             "bytes": 196608, "times_us": [60.0]}]
    for cap, two in ((4, 50.0), (32, 70.0)):
        runs.append({**base, "collective": "all_reduce_twoshot", "algorithm": "twoshot", "shape": [98304],
                     "bytes": 196608, "times_us": [two], "large_blocks": cap})
    results = [_rank_result(rank, rank // 2, f"spark{rank}", runs) for rank in range(8)]
    merged = summary.merge(built, results)
    assert [case.get("large_blocks") for case in merged["cases"][:3]] == [None, 4, 32]
    found = {entry["large_blocks"]: entry for entry in merged["crossover"] if entry["group"] == 0}
    assert found[4]["first_twoshot_bytes"] == 196608 and found[32]["first_twoshot_bytes"] is None
    text = summary.table(merged)
    assert "[grid cap 4]" in text and "grid cap 32: one-shot faster at every size" in text


def test_forward_window_waits_reach_the_summary():
    built = plan.build_plan(_site(), "pairs", "run1", digest="d" * 16).to_json()
    run = {"collective": "all_reduce", "mode": "graph", "shape": [196608], "dim": 0, "bytes": 393216,
           "correct": True, "mismatched_calls": 0, "checked": 1, "counters": {}, "times_us": [130.0, 131.0],
           "algorithm": "twoshot"}
    results = [_rank_result(rank, rank // 2, f"spark{rank}",
                            [{**run, "forward": {"waits_per_call": 2.0 * (rank % 2), "wait_us_per_call": 30.0 * (rank % 2),
                                                 "proven_kib_per_call": 4.0}}]) for rank in range(8)]
    merged = summary.merge(built, results)
    case = merged["cases"][0]
    assert case["forward_wait_us_per_call"] == 30.0 and case["forward_waits_per_call"] == 2.0
    assert case["forward_proven_kib_per_call"] == 4.0
    assert "forward-window waits 2 per call on the slowest rank, 30 us" in summary.table(merged)


def test_large_sizes_replace_the_configurations(tmp_path, capsys):
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    line = ["plan", "--site", str(site_file), "--config", "ring8-large", "--json", "--large-sizes",
            "196608,393216,589824", "--large-schedules", "pieces,ring", "--reduce-link-chunks", "8192,24576",
            "--session-env", "SIRCL_LINK_SLOTS=12"]
    assert cli.main(line) == 0
    options = json.loads(capsys.readouterr().out)["options"]
    assert options["large_allreduce_sizes"] == [196608, 393216, 589824]
    assert options["large_schedules"] == ["pieces", "ring"] and options["reduce_link_chunks"] == [8192, 24576]
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring8-large", "--large-sizes", "100"]) == 2
    assert "multiples of 16" in capsys.readouterr().err


def test_per_collective_link_pieces(tmp_path, capsys):
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    # The target sweep: ring all-gather and all-reduce in 1 MiB pieces, chain reduce-scatter in 256 KiB.
    line = ["plan", "--site", str(site_file), "--config", "path4-large", "--json", "--large-schedules", "ring",
            "--gather-schedules", "ring", "--scatter-schedules", "chain", "--gather-link-chunks", "1048576",
            "--reduce-link-chunks", "1048576", "--scatter-link-chunks", "262144",
            "--session-env", "SIRCL_LINK_SLOT_BYTES=1048576"]
    assert cli.main(line) == 0
    document = json.loads(capsys.readouterr().out)
    options = document["options"]
    assert (options["gather_link_chunks"], options["scatter_link_chunks"], options["reduce_link_chunks"]) == (
        [1048576], [262144], [1048576])
    harness = worker.Harness.__new__(worker.Harness)
    harness.options = options
    cases = harness.large_cases()
    pieces = {(collective, (variant or {}).get("link_chunk")) for collective, _, _, variant in cases
              if variant and "link_chunk" in variant}
    assert pieces == {("all_reduce_large", 1048576), ("all_gather_large", 1048576), ("reduce_scatter", 262144)}
    assert worker.link_collective("reduce_scatter") == "scatter" and worker.link_collective("all_reduce") is None
    # A collective without a list of its own sweeps the common pieces; a piece set through --session-env
    # is that collective's sweep.
    assert cli.main(["plan", "--site", str(site_file), "--config", "path4-large", "--json",
                     "--session-env", "SIRCL_SCATTER_LINK_CHUNK_BYTES=262144"]) == 0
    options = json.loads(capsys.readouterr().out)["options"]
    harness.options = options
    assert harness._sweep("scatter") == (262144,) and harness._sweep("gather") == tuple(plan.LINK_CHUNK_SWEEP)
    assert cli.main(["plan", "--site", str(site_file), "--config", "path4-large"] + line[6:]) == 0
    text = capsys.readouterr().out
    assert "all-gather link pieces [1048576]" in text and "reduce-scatter link pieces [262144]" in text
    # Pieces above the link slot need the slot.
    assert cli.main(["plan", "--site", str(site_file), "--config", "path4-large", "--gather-link-chunks",
                     "1048576"]) == 2
    assert "SIRCL_LINK_SLOT_BYTES" in capsys.readouterr().err


def test_crossover_configuration_and_ring_rows_without_the_ring_minimum():
    configured = plan.configuration_options("path4-crossover", plan.Options())
    built = plan.build_plan(_site(), "path4-crossover", "run1", options=configured, digest="d" * 16)
    assert [group.positions for group in built.groups] == [(0, 1, 2, 3)]
    options = json.loads(json.dumps(built.to_json()["options"]))
    assert options["large_allreduce_sizes"] == list(plan.CROSSOVER_SIZES) == [256 << 10, 512 << 10, 1 << 20, 2 << 20,
                                                                                4 << 20]
    assert options["chain_gather_sizes"] == [size // 4 for size in plan.CROSSOVER_SIZES]
    assert options["large_reduce_scatter_sizes"] == list(plan.CROSSOVER_SIZES)
    harness = worker.Harness.__new__(worker.Harness)
    harness.options = options
    harness.session = type("Capacity", (), {"max_size": 2 << 20})()
    variants = [variant for _, _, _, variant in harness.large_cases()]
    assert {(v.get("schedule"), v.get("gather_schedule"), v.get("scatter_schedule")) for v in variants if v} >= {
        ("ring", None, None), (None, "ring", None), (None, None, "ring")}

    # A ring row runs at its size whatever the session's ring minimums, which come back afterwards.
    defaults = {"reduce": 4 << 20, "gather": 8 << 20, "scatter": 4 << 20}

    class Session:
        large_schedule, chain_chunk_bytes, gather_schedule, scatter_schedule = "auto", 524288, "auto", "auto"
        link_chunk_bytes = 524288

        def __init__(self):
            self.ring_mins = dict(defaults)

        def set_chain_chunk_bytes(self, value):
            self.chain_chunk_bytes = value

        def set_link_chunk_bytes(self, value, collective=None):
            self.link_chunk_bytes = value

        def ring_min_for(self, collective):
            return self.ring_mins[collective]

        def set_ring_min_bytes(self, value, collective=None):
            for kind in self.ring_mins if collective is None else (collective,):
                self.ring_mins[kind] = defaults[kind] if value is None else value

    seen = []
    harness.session = Session()
    harness._run_case = lambda *args: seen.append(dict(harness.session.ring_mins)) or {}
    harness.run_case("all_reduce_large", "eager", (1 << 18,), 0, 1, large=True, variant={"schedule": "ring"})
    harness.run_case("all_reduce_large", "eager", (1 << 18,), 0, 2, large=True, variant={"schedule": "chain"})
    harness.run_case("reduce_scatter", "eager", (32, 4096), 0, 3, large=True, variant={"scatter_schedule": "ring"})
    zero = {"reduce": 0, "gather": 0, "scatter": 0}
    assert seen == [zero, defaults, zero] and harness.session.ring_mins == defaults
