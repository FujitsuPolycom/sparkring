"""Offline regressions for setup admission and installer presentation."""
import copy
import json
from types import SimpleNamespace

import pytest

from runtime.host import activity, controller, node, placement, slices, topology
from runtime.host.test_appliance import nodes


def test_repeated_setup_preserves_firewall_owner_but_refuses_different_routes(tmp_path):
    found = nodes(2)
    plan = topology.build_spec(found, found[0]["node_id"])
    config = topology.persistent_config(plan, 0)
    node.save(tmp_path, "/etc/sparkring/node.json", {"node_id": config["node_id"]})
    node.save(tmp_path, "/etc/sparkring/fabric.json", config)
    repeated = {**config, "cluster_id": "f" * 64}
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    node.configure(repeated, root=tmp_path, collect=lambda _: found[0]["facts"], run=run)
    assert node.read(tmp_path, "/etc/sparkring/fabric.json")["cluster_id"] == config["cluster_id"]
    calls.clear()
    changed = copy.deepcopy(repeated)
    changed["interfaces"][0]["address"] = "198.18.99.1/24"
    with pytest.raises(ValueError, match="another approved configuration"):
        node.configure(changed, root=tmp_path, collect=lambda _: found[0]["facts"], run=run)
    assert calls == []


def test_setup_conflict_is_refused_before_changes_or_receipt(tmp_path):
    found = nodes(2)
    plan = topology.build_spec(found, found[0]["node_id"])
    changed = topology.persistent_config(plan, 1)
    changed["owner"] = "another-ring"
    def invoke(host, argv, **kwargs):
        if argv[:4] == ["sudo", "-n", "python3", "-I"]:
            return json.dumps({"config": changed if host == plan["spec"]["hosts"][1]["host"] else None})
        assert argv[-1] == "unit-overrides"
        return "{}"
    with pytest.raises(ValueError, match="another approved configuration"):
        controller.apply(plan, tmp_path / "setup", invoke=invoke,
                         run=lambda *a: pytest.fail("network changed"),
                         update_workers=lambda: pytest.fail("workers updated"))
    assert not (tmp_path / "setup").exists()


@pytest.mark.parametrize("state,expected", [(None, (6, 7)),
    ({"operation": "down", "complete": True}, (6, 7)),
    ({"operation": "up", "complete": True}, (4, 5, 6, 7)),
    ({"operation": "down", "complete": False}, (4, 5, 6, 7))])
def test_pair_reserves_only_deployments_it_can_stop(tmp_path, state, expected):
    directory = tmp_path / "deployment"
    node.save(directory, "deployment.lock.json", {})
    if state:
        node.save(directory, "state.json", state)
    placement.record(tmp_path, (4, 5, 6, 7), directory)
    assert slices.affected(tmp_path, (6, 7)) == expected


def test_unfinished_switch_still_reserves_rollback_group(tmp_path):
    node.save(placement.slot_directory(tmp_path, (4, 5, 6, 7)), "transaction.json",
              {"complete": False, "state": "recovering-previous"})
    assert slices.affected(tmp_path, (6, 7)) == (4, 5, 6, 7)


@pytest.mark.parametrize("operation", ["up", "down", "restart"])
def test_lifecycle_does_not_contend_for_install_downloads(tmp_path, monkeypatch, operation):
    monkeypatch.setattr(slices, "active", lambda _: [{"positions": [4, 5], "operation": "install profile"}])
    result = activity.assess(tmp_path, (6, 7), operation=operation, quiet_path=tmp_path / "quiet")
    assert result["grade"] == "clear"
    assert activity.assess(tmp_path, (4, 5), operation=operation)["grade"] == "blocked"
    assert activity.assess(tmp_path, (6, 7), operation="install")["grade"] == "may-be-affected"


@pytest.mark.parametrize("operation", ["up", "down", "restart", "status"])
def test_lifecycle_help_has_one_operation(operation, capsys):
    with pytest.raises(SystemExit) as error:
        controller.lifecycle([operation, "--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage: sparkring " + operation in help_text
    assert "{up,down,restart,status}" not in help_text


@pytest.mark.parametrize("mode,verb", [("on", "off"), ("off", "on"), ("unrecorded", "on")])
def test_ready_card_recovery_targets_the_installed_group(mode, verb):
    from runtime.host import colour, install_workflow
    result = {"example_request": "curl example", "recovery": mode,
              "commands": {"stop": "sudo sparkring down --on 6,7 --execute"}}
    lines = [colour.plain(line) for line in install_workflow.summary_lines(result)]
    assert any(f"sudo sparkring recover {verb} --on 6,7" in line for line in lines)


@pytest.mark.parametrize("live", [False, True])
def test_status_headline_reflects_three_deployments(live):
    views = [{"deployment": {"profile": f"model-{i}", "state": {"operation": "up", "complete": True}},
              "placement": [2 * i, 2 * i + 1]} for i in range(3)]
    if live:
        for view in views:
            view["model"] = {"state": "serving"}
    headline, action = controller.status_headline({"state": "network-configured", "next_action": "sparkring models"}, views)
    assert headline == ("Serving: model-0 (0-1), model-1 (2-3), model-2 (4-5)" if live else
                        "3 started deployments (saved state)")
    assert action == ("sparkring status" if live else "sudo sparkring status --refresh")
    views[0]["model"] = {"state": "stopped"}
    assert controller.status_headline({"state": "network-configured"}, views)[0] == (
        "Serving: model-1 (2-3), model-2 (4-5)" if live else "2 started deployments (saved state)")


def test_repeated_setup_refuses_before_worker_update_and_can_retry(tmp_path, monkeypatch):
    from runtime.host import single_uplink
    found = nodes(2)
    plan = topology.build_spec(found, found[0]["node_id"], preserve_control=True)
    node.save(tmp_path, "cluster.json", {"name": "sparkring", "plan": plan})
    monkeypatch.setattr(controller, "STATE", tmp_path)
    monkeypatch.setattr(single_uplink.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(single_uplink.distribution, "installed", lambda root: True)
    monkeypatch.setattr(single_uplink, "identity_key", lambda _: (tmp_path / "key", "public key"))
    monkeypatch.setattr(single_uplink, "record_reason", lambda _: None)
    monkeypatch.setattr(controller, "collect", lambda targets: found)
    monkeypatch.setattr(single_uplink.node, "read", lambda *args: {"node_id": found[0]["node_id"]})
    changed = topology.persistent_config(plan, 1)
    changed["owner"] = "different-ring"
    calls = []
    def invoke(host, argv):
        if argv[2] == "python3":
            return json.dumps({"config": changed if host.endswith("11") else None})
        assert argv[-1] == "unit-overrides"
        return "{}"
    monkeypatch.setattr(controller.discovery, "ssh", invoke)
    monkeypatch.setattr(single_uplink, "match_revisions", lambda targets, nodes, directory: calls.append("update") or nodes)
    monkeypatch.setattr(controller, "apply", lambda plan, directory, **kw: calls.append("apply") or plan)
    monkeypatch.setattr(controller, "summarize", lambda *args, **kw: None)
    monkeypatch.setattr(single_uplink.fabric_bandwidth, "after_setup", lambda *args: None)
    monkeypatch.setattr(single_uplink.fabric, "finish_setup", lambda *args, **kw: None)
    monkeypatch.setattr(controller.dashboard_admin, "after_setup", lambda: None)
    with pytest.raises(ValueError, match="another approved configuration"):
        single_uplink.main(["--yes"])
    assert calls == []
    assert not (tmp_path / "setups").exists()
    changed.clear()
    changed.update(topology.persistent_config(plan, 1))
    changed["cluster_id"] = "f" * 64
    assert single_uplink.main(["--yes"]) == 0
    assert calls == ["update", "apply"]


def test_down_ignores_network_advisories_and_does_not_sample_traffic(tmp_path, monkeypatch):
    from runtime.host import dashboard
    dashboard.quiet_hold(tmp_path / "quiet", "benchmark", ttl=60)
    node.save(tmp_path, "cluster.json", {})
    monkeypatch.setattr(placement, "layout_of", lambda _: {"shape": "cycle", "size": 8})
    monkeypatch.setattr(placement, "running", lambda _: pytest.fail("down sampled serving neighbours"))
    result = activity.assess(tmp_path, (6, 7), operation="down", quiet_path=tmp_path / "quiet")
    assert result["grade"] == "clear"
    assert result["observations"] == []


def test_configuration_preflight_refuses_a_corrupt_saved_plan_id():
    found = nodes(2)
    plan = topology.build_spec(found, found[0]["node_id"])
    config = topology.persistent_config(plan, 0)
    with pytest.raises(ValueError, match="Invalid approved fabric state"):
        node.check_configuration(config, {**config, "cluster_id": "corrupt"})


@pytest.fixture
def named_setup(tmp_path, monkeypatch):
    from runtime.host import single_uplink
    found = nodes(2)
    plan = topology.build_spec(found, found[0]["node_id"], name="ring8c", preserve_control=True)
    node.save(tmp_path, "cluster.json", {"name": "ring8c", "plan": plan})
    monkeypatch.setattr(controller, "STATE", tmp_path)
    monkeypatch.setattr(single_uplink.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(single_uplink.distribution, "installed", lambda root: True)
    monkeypatch.setattr(single_uplink.distribution, "identity", lambda root: found[0]["revision"])
    monkeypatch.setattr(single_uplink, "identity_key", lambda _: (tmp_path / "key", "public key"))
    monkeypatch.setattr(single_uplink, "record_reason", lambda _: None)
    monkeypatch.setattr(controller, "collect", lambda targets: found)
    monkeypatch.setattr(single_uplink.node, "read", lambda *args: {"node_id": found[0]["node_id"]})
    monkeypatch.setattr(controller, "preflight_configuration", lambda plan: None)
    monkeypatch.setattr(controller, "refuse_unit_overrides", lambda plan: None)
    calls = []
    monkeypatch.setattr(single_uplink, "match_revisions", lambda targets, nodes, directory: calls.append("update") or nodes)
    monkeypatch.setattr(controller, "apply", lambda plan, directory, **kw: calls.append(plan["spec"]["owner"]) or plan)
    monkeypatch.setattr(single_uplink.fabric, "finish_setup", lambda *args, **kw: None)
    return tmp_path, calls


@pytest.mark.parametrize("name,expected", [(None, "ring8c"), ("ring8c", "ring8c"), ("renamed", "renamed")])
@pytest.mark.parametrize("mode", ["--plan", "--yes"])
def test_setup_keeps_recorded_name_and_shows_explicit_rename(named_setup, capsys, name, expected, mode):
    from runtime.common import installer
    from runtime.host import single_uplink
    root, calls = named_setup
    assert single_uplink.main([mode, *(["--name", name] if name else [])]) == 0
    output = capsys.readouterr().out
    assert "Cluster name: " + expected in output
    assert ("Rename cluster: ring8c -> renamed" in output) == (name == "renamed")
    plan = installer.read(next((root / "setups").glob("*/plan.json")))
    assert plan["spec"]["owner"] == expected
    assert installer.read(root / "cluster.json")["name"] == (expected if mode == "--yes" else "ring8c")
    assert calls == (["update", expected] if mode == "--yes" else [])


def test_setup_rename_requires_consent_before_worker_updates(named_setup, monkeypatch, capsys):
    from runtime.host import single_uplink
    root, calls = named_setup
    monkeypatch.setattr(single_uplink.sys.stdin, "isatty", lambda: True)
    questions = []
    monkeypatch.setattr("builtins.input", lambda question: questions.append(question) or "n")
    with pytest.raises(ValueError, match="Cancelled"):
        single_uplink.main(["--name", "renamed"])
    assert "Rename cluster" in questions[0]
    assert calls == []
    assert json.loads((root / "cluster.json").read_text())["name"] == "ring8c"
    assert "Rename cluster: ring8c -> renamed" in capsys.readouterr().out


@pytest.mark.parametrize("name,expected", [(None, "ring8c"), ("ring8c", "ring8c"), ("renamed", "renamed")])
@pytest.mark.parametrize("mode", ["--plan", "--apply"])
def test_setup_with_explicit_nodes_keeps_name(named_setup, capsys, name, expected, mode):
    root, calls = named_setup
    targets = [arg for found in nodes(2) for arg in ("--node", found["facts"]["ssh_target"])]
    assert controller.setup([*targets, mode, "--yes", "--skip-enroll", "--output", str(root / "setup"),
                             *(["--name", name] if name else [])]) == 0
    assert json.loads((root / "cluster.json").read_text())["name"] == (expected if mode == "--apply" else "ring8c")
    output = capsys.readouterr().out
    assert "Cluster name: " + expected in output
    assert ("Rename cluster: ring8c -> renamed" in output) == (name == "renamed")
    assert calls == ([expected] if mode == "--apply" else [])


def test_setup_without_name_prefers_record_over_env(named_setup):
    from runtime.host import single_uplink
    root, _ = named_setup
    env = root / "preferences.env"
    env.write_text("SPARKRING_NAME=another\n")
    assert single_uplink._arguments(["--env", str(env)])[0].name == "ring8c"
    assert single_uplink._arguments(["--env", str(env), "--name=explicit"])[0].name == "explicit"
    (root / "cluster.json").unlink()
    assert single_uplink._arguments(["--env", str(env)])[0].name == "another"


def test_explicit_node_setup_refuses_another_cluster_before_apply(named_setup):
    root, calls = named_setup
    path = root / "cluster.json"
    recorded = json.loads(path.read_text())
    recorded["plan"]["spec"]["hosts"][0]["node_id"] = "another-node"
    path.write_text(json.dumps(recorded))
    targets = [arg for found in nodes(2) for arg in ("--node", found["facts"]["ssh_target"])]
    with pytest.raises(ValueError, match="Controller already records another cluster"):
        controller.setup([*targets, "--apply", "--yes", "--skip-enroll", "--output", str(root / "setup")])
    assert calls == []
    assert not (root / "setup").exists()


def test_status_headline_names_only_live_profiles_and_positions():
    views = [{"deployment": {"profile": profile}, "group": {"positions": positions},
              "model": {"state": state}} for profile, positions, state in (
                  ("deepseek-v41-flash-sparkcache-tp4", [0, 1, 2, 3], "serving"),
                  ("qwen38-flash-next-step5500-sparkcache-tp4", [4, 5, 6, 7], "serving"),
                  ("stopped-profile", [4, 5], "stopped"))]
    headline, action = controller.status_headline({"state": "network-configured"}, views)
    assert headline == ("Serving: deepseek-v41-flash-sparkcache-tp4 (0-3), "
                        "qwen38-flash-next-step5500-sparkcache-tp4 (4-7)")
    assert action == "sparkring status"
