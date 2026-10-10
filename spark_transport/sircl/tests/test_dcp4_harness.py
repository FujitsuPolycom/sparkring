"""The ring harness's dcp4 configuration offline: plan, world session, cases, preflight and summary."""

from __future__ import annotations

import json

import pytest

from sparkring_sircl import routes
from sparkring_sircl.ring import cli, plan, remote, summary, worker
from sparkring_sircl.ring.site import Site


def _site_document(size: int = 8) -> dict:
    return {
        "schema": "sircl-ring-site/v1", "image": "sha256:0123456789ab", "lan_interface": "lan0",
        "control_port": 29650, "remote_dir": "/tmp/sircl-ring",
        "ring": [{"name": f"spark{i}", "ssh": f"op@192.0.2.{10 + i}", "lan_address": f"192.0.2.{10 + i}"}
                 for i in range(size)],
    }


def _dcp4(**overrides) -> plan.ConfigurationPlan:
    options = plan.configuration_options("dcp4", plan.Options(**overrides))
    return plan.build_plan(Site.from_json(_site_document()), "dcp4", "run1", options=options, digest="d" * 16)


def test_dcp4_holds_two_dcp_groups_inside_a_world_session_over_the_ring():
    built = _dcp4()
    assert [group.positions for group in built.groups] == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert [group.global_ranks for group in built.groups] == [(0, 1, 2, 3), (4, 5, 6, 7)]
    ring = plan.group_layout(8, tuple(range(8)))
    assert built.world_layout == plan.layout_text(ring)
    derived = routes.derive_routes(routes.Layout.parse(built.world_layout), 2)
    assert [rank.world_peer_routes for rank in built.ranks] == [derived.route_text(r) for r in range(8)]
    # The group sessions keep the path routes of two-tp4.
    two = plan.build_plan(Site.from_json(_site_document()), "two-tp4", "run1", digest="d" * 16)
    assert [rank.peer_routes for rank in built.ranks] == [rank.peer_routes for rank in two.ranks]
    options = built.options
    assert options.dcp_decode_rows == plan.DCP_DECODE_ROWS == (1, 2, 4, 8, 16, 32, 64, 128)
    assert options.dcp_prefill_rows == plan.DCP_PREFILL_ROWS == (8192,)
    assert options.dcp_gathers and options.world_session and options.large_only
    assert options.large_allreduce_sizes == () and options.large_allgather_sizes == ()
    document = built.to_json()
    assert document["world_layout"] == built.world_layout
    assert document["ranks"][5]["world_peer_routes"] == derived.route_text(5)
    text = plan.render_text(built)
    assert "world session (tensor parallel over every rank" in text
    assert "DCP exchanges at decode rows [1, 2, 4, 8, 16, 32, 64, 128] (eager and graph) and prefill rows [8192]" in text
    assert "[W, rows, 8, 514] BF16" in text and "[rows, 8, 576] BF16" in text and "16384 bytes per row" in text
    assert "large: all-reduce" not in text


def test_other_configurations_hold_no_world_session():
    built = plan.build_plan(Site.from_json(_site_document()), "two-tp4", "run1", digest="d" * 16)
    assert built.world_layout == "" and all(rank.world_peer_routes == "" for rank in built.ranks)
    assert built.to_json()["world_layout"] == ""


def test_dcp4_refusals():
    for size in (7, 9):
        with pytest.raises(plan.PlanError, match="exactly eight"):
            plan.builtin_groups("dcp4", size)
    with pytest.raises(plan.PlanError, match="transport-only"):
        plan.configuration_options("dcp4", plan.Options(transport_only=True))
    with pytest.raises(plan.PlanError, match="positive"):
        plan.Options(dcp_decode_rows=(0,))
    with pytest.raises(plan.PlanError, match="every Spark of the ring"):
        options = plan.Options(world_session=True)
        plan.build_plan(Site.from_json(_site_document()), "path4", "run1", options=options, digest="d" * 16)


def test_dcp4_row_counts_from_the_command_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(remote, "ssh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("contacted a host")))
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    assert cli.main(["plan", "--site", str(site_file), "--config", "dcp4", "--json",
                     "--dcp-decode-rows", "1,32", "--dcp-prefill-rows", "none"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["options"]["dcp_decode_rows"] == [1, 32] and document["options"]["dcp_prefill_rows"] == []
    assert cli.main(["plan", "--site", str(site_file), "--config", "dcp4", "--dcp-decode-rows", "x"]) == 2
    assert "row counts 'x'" in capsys.readouterr().err


def _harness(options: dict, world: int = 4, tp: bool = True) -> worker.Harness:
    harness = worker.Harness.__new__(worker.Harness)
    harness.options, harness.world, harness.tp_session = options, world, (object() if tp else None)
    return harness


def test_dcp_cases_follow_the_a2a_combine_layout():
    options = plan.configuration_options("dcp4", plan.Options(dcp_decode_rows=(1, 32), dcp_prefill_rows=(8192,)))
    cases = _harness(json.loads(json.dumps(plan.build_plan(
        Site.from_json(_site_document()), "dcp4", "r", options=options, digest="d" * 16).to_json()["options"])))
    found = cases.dcp_cases()
    assert found[:4] == [
        ("all_to_all", (4, 8 * 514), 0, 1, True),
        ("all_gather_large", (1, 8 * 576), 1, 1, True),
        ("all_gather_large", (1, 8192), 1, 1, True),
        ("tp_all_reduce", (1, 6144), 0, 1, True),
    ]
    prefill = [case for case in found if case[3] == 8192]
    assert [case[0] for case in prefill] == ["all_to_all", "all_gather_large", "all_gather_large"]
    assert prefill[0][1] == (4 * 8192, 8 * 514) and not any(case[4] for case in prefill)
    # The a2a message is vLLM's [W, rows, heads, 514] BF16 buffer: 8,224 bytes per row and peer.
    assert 4 * 8192 * 8 * 514 * 2 == 269_484_032
    no_world = _harness({"dcp_decode_rows": [1], "dcp_heads": 8}, tp=False).dcp_cases()
    assert [case[0] for case in no_world] == ["all_to_all"]


def test_bandwidths_and_summary_of_world_session_cases():
    assert worker.bandwidths("all_to_all", 4_000_000, 4, 1000.0) == {"algbw_gbps": 4.0, "busbw_gbps": 3.0}
    assert worker.bandwidths("tp_all_reduce", 8_000_000, 8, 1000.0) == {"algbw_gbps": 8.0, "busbw_gbps": 14.0}
    built = _dcp4().to_json()
    run = {"collective": "tp_all_reduce", "mode": "graph", "shape": [1, 6144], "dim": 0, "bytes": 8_000_000,
           "correct": True, "mismatched_calls": 0, "checked": 4, "counters": {}, "world": 8,
           "times_us": [1000.0, 1000.0, 1000.0]}
    results = [{"global_rank": rank, "group": rank // 4, "host": f"spark{rank}", "runs": [run], "error": None,
                "exit_code": 0} for rank in range(8)]
    merged = summary.merge(built, results)
    assert merged["status"] == "passed"
    # Bus bandwidth over the world of eight, not the group of four.
    assert [case["busbw_gbps"] for case in merged["cases"]] == [14.0, 14.0]


def _preflight_output(position: int) -> str:
    lines = ["docker\t27.3.1", "image\tsha256:abc", "gpu\t0, NVIDIA GB10", f"lan\t192.0.2.{10 + position}/24"]
    for index, role in enumerate(routes.ROLES):
        lines.append(f"device:{role.device}\t4: ACTIVE {role.netdev}")
        lines.append(f"address:{role.netdev}\t10.{position}.{index}.1/24")
    return "\n".join(lines) + "\n"


def test_preflight_checks_the_world_session_lanes(monkeypatch):
    site = Site.from_json(_site_document())
    positions = {host.ssh: index for index, host in enumerate(site.ring)}

    def fake_ssh(target, command, *, timeout=60, input_bytes=None, binary="ssh"):
        if "info --format" in command:
            return remote.Result(0, _preflight_output(positions[target]), "")
        return remote.Result(0, "", "")    # no route resolves

    monkeypatch.setattr(remote, "ssh", fake_ssh)
    ok, lines = cli.preflight(site, [_dcp4()], force=False)
    assert not ok
    world = [line for line in lines if "needs" in line and "world session rank" in line]
    groups = [line for line in lines if "needs" in line and "world session" not in line]
    assert len(world) == 8 * 7 * 2 and len(groups) == 2 * 4 * 3 * 2
