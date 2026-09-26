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
    # Reinstalling restores the units prerm recorded; a recorded fabric unit is
    # enabled for the next boot but never started by the package, so package
    # installation does not change the data network.
    fabric_rules = [line for line in postinst.splitlines() if "sparkring-fabric" in line]
    assert fabric_rules and all("systemctl enable \"$unit\"" in line for line in fabric_rules)
    assert not any(word in line for line in fabric_rules for word in ("--now", "start", "restart"))
    assert "/var/lib/sparkring/package-enabled-units" in postinst
    assert "/var/lib/sparkring/package-enabled-units" in (root / "prerm").read_text()
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


UP_PROFILE = "qwen38-flash-next-tp2"


@pytest.fixture
def plain_up(tmp_path, monkeypatch):
    """`sparkring up <profile>` on a recorded two-Spark cluster whose rank operations are simulated.

    SSH from the controller fails the test: `sparkring up` neither surveys nor
    probes the Sparks. Every rank operation runs through the plain
    installer_runner.Runner, which records it with the row it acted on.
    """
    from runtime.common import distribution
    from runtime.host import discovery, node
    from runtime.host.test_fabric_ssh import cluster
    from scripts import installer_runner
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    node.save(controller.STATE, "cluster.json", cluster(2))
    monkeypatch.setattr(distribution, "identity", lambda _: "a" * 40)
    monkeypatch.setattr(distribution, "bundle", lambda root, dest: dest.write_bytes(b"retained source"))
    monkeypatch.setattr(discovery, "ssh", lambda *a, **k: pytest.fail("sparkring up contacted a Spark: " + repr(a[1])))
    calls = []

    def call(self, target, argv, timeout):
        row = self.lock["site"]["ranks"][int(argv[2])]
        calls.append({"runner": type(self), "operation": argv[1], "rank": row["rank"], "model": row["model"],
                      "reuse": row["reuse_verified_model"]})
        return {"returncode": 0, "stdout": "ok", "stderr": "", "uncertain": False}
    monkeypatch.setattr(installer_runner.Runner, "_call", call)
    return calls


def test_sparkring_up_uses_the_cluster_checkpoint_directory_through_a_plain_runner(plain_up):
    from runtime.common import installer, setup
    from scripts import installer_runner
    assert controller.lifecycle(["up", UP_PROFILE, "--execute"]) == 0
    directory = installer.checkpoint_directory("test", setup.selection(UP_PROFILE))
    assert directory == ("/srv/sparkring/test/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/"
                         "629bc3218833a38b475b719f34aa571666f4a03e")
    lock = installer.load(controller.STATE / "deployments" / UP_PROFILE)
    assert [(row["model"], row["reuse_verified_model"]) for row in lock["site"]["ranks"]] == [(directory, False)] * 2
    # The plain runner's model operation adopts what that directory holds and
    # downloads the rest on each rank; no survey or asset probe precedes it. The
    # ranks' model actions run in parallel, so they may finish in either order.
    model = [call for call in plain_up if call["operation"] == "model"]
    assert sorted(call["rank"] for call in model) == [0, 1]
    assert all(call["runner"] is installer_runner.Runner and call["model"] == directory and not call["reuse"]
               for call in plain_up)


def test_sparkring_up_model_path_is_served_in_place_and_never_written(plain_up):
    from runtime.common import installer
    named = "/data/qwen-copy"
    assert controller.lifecycle(["up", UP_PROFILE, "--model-path", named, "--execute"]) == 0
    lock = installer.load(controller.STATE / "deployments" / UP_PROFILE)
    # reuse_verified_model means "verify; never write": the rank operations
    # verify the named copy and never create, link, fetch or repair under it.
    assert [(row["model"], row["reuse_verified_model"]) for row in lock["site"]["ranks"]] == [(named, True)] * 2
    assert all(call["model"] == named and call["reuse"] for call in plain_up)
    assert sorted(call["rank"] for call in plain_up if call["operation"] == "model") == [0, 1]
    # The in-place decision belongs to that deployment; naming another copy
    # needs another instance.
    with pytest.raises(ValueError, match="another model path"):
        controller.lifecycle(["up", UP_PROFILE, "--model-path", "/data/other", "--plan"])


def test_adoption_records_facts_without_running_network_commands(tmp_path, monkeypatch):
    from runtime.host import node
    found = nodes(2)
    plan = topology.build_spec(found, found[0]["node_id"])
    config = topology.persistent_config(plan, 0)
    config.update(ownership="observed", routes=[], forwarding=[])
    node.save(tmp_path, "/etc/sparkring/node.json", {"node_id": config["node_id"]})
    monkeypatch.setattr(node, "call", lambda *a, **k: pytest.fail("Adoption changed host services/network"))
    assert node.adopt(config, root=tmp_path, collect=lambda _: found[0]["facts"])["network_changed"] is False
    with pytest.raises(ValueError, match="existing service"):
        node.restore(config)
