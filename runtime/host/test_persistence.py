"""Lifecycle boundary regressions for root services and control-preserving setup."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.host import controller, topology
from runtime.host.test_appliance import configured, nodes
from scripts.deploy_network_run import _guard, _network_local


def test_control_network_modifies_original_uuid_without_bouncing_ipv6():
    found = nodes(2)
    plan = topology.build_spec(found, found[0]["node_id"], reset=True, fabric_cidr="198.18.32.0/21", preserve_control=True)
    for host in plan["network"]["hosts"]:
        commands = [c["argv"] for c in host["apply"]]
        assert any("modify" in c for c in commands)
        assert any("reapply" in c for c in commands)
        assert all("up" not in c and "down" not in c and "add" not in c for c in commands)
        assert all("ipv6.addr-gen-mode" not in c and "connection.uuid" not in c for c in commands)
        assert host["rollback"]


def test_reset_address_choice_survives_driver_rediscovery(tmp_path):
    found = nodes()
    found[0]["facts"]["rdma"][0]["devlink"]["parameters"]["hairpin_num_queues"]["value"] = 0
    initial = topology.build_spec(found, found[0]["node_id"], reset=True, fabric_cidr="198.18.40.0/21", preserve_control=True)
    loaded = copy.deepcopy(found)
    loaded[0]["facts"]["rdma"][0]["devlink"]["parameters"]["hairpin_num_queues"]["value"] = 4
    post_reload = topology.build_spec(loaded, loaded[0]["node_id"], reset=True, fabric_cidr="198.18.40.0/21", preserve_control=True)
    ready = configured(post_reload)
    observations = iter((loaded, ready))
    reviews = []

    def runner(host, argv, timeout):
        is_check = "'check'" in argv[-1]
        return {"returncode": 0, "stderr": "", "stdout": '{"checked":true}' if is_check else '{"complete":true}'}

    result = controller.apply(initial, tmp_path, run=runner, invoke=lambda *a, **k: "{}",
                              inspect_nodes=lambda _: next(observations), allow_driver_reload=True, review=reviews.append)
    assert len(reviews) == 1
    assert result["spec"]["hosts"][0]["data_interfaces"][0]["address"] == "198.18.40.1/24"
    assert result["reset_requested"]


def test_gpu_compute_blocks_fabric_mutation_before_journal_creation(tmp_path):
    found = nodes(2, blank=True)
    plan = topology.build_spec(found, found[0]["node_id"])
    host = plan["spec"]["hosts"][0]
    host["backup_dir"] = str(tmp_path / "backup")
    facts = found[0]["facts"]
    payload = {"host": host, "request": {}, "guard": _guard(facts, host), "commands": [{}], "require_idle_gpu": True}
    with pytest.raises(ValueError, match="GPU compute"):
        _network_local(payload, "check", collect=lambda _: facts,
                       run=lambda *a, **k: SimpleNamespace(returncode=0, stdout="1234\n", stderr=""))
    assert not (tmp_path / "backup").exists()


def test_saved_model_api_address_uses_node_a_ethernet():
    from runtime.common import installer
    found = nodes(2)
    cluster = {"name": "home", "api_address": "192.0.2.55", "plan": topology.build_spec(found, found[0]["node_id"])}
    raw = controller.model_site(cluster, "qwen")
    lock = installer.make_lock(installer.DEFAULTS["qwen38", 2], raw, "a" * 40, "b" * 64)
    assert installer.connection(lock)["api_url"].startswith("http://192.0.2.55:")


def test_administrative_services_do_not_start_models_or_rewrite_ssh_policy():
    root = Path(__file__).resolve().parents[2] / "packaging/debian"
    postinst = (root / "postinst").read_text()
    assert "node initialize" in postinst and "sparkring up" not in postinst
    assert "sparkring-fabric" not in postinst
    assert "sshd_config" not in postinst
    for unit in root.glob("*.service"):
        assert "installer_host" not in unit.read_text()
    assert "weights" in (root / "postrm").read_text()


def test_plan_only_named_model_does_not_construct_runner(tmp_path, monkeypatch, capsys):
    from runtime.common import installer
    from scripts import installer_runner
    monkeypatch.setattr(controller, "STATE", tmp_path)
    (tmp_path / "active.json").write_text(json.dumps({"path": str(tmp_path / "model")}))
    monkeypatch.setattr(installer, "apply", lambda *a, **kw: {"profile": "fixture", "hosts": ["a", "b"], "phases": ["read"]})
    monkeypatch.setattr(installer_runner, "Runner", lambda *a: pytest.fail("Plan attempted SSH runner"))
    assert controller.lifecycle(["up", "--plan"]) == 0
    assert "--execute" in capsys.readouterr().out
