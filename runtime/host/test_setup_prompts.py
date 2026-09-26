"""Setup asks once, recognizes known Sparks without new logins and records host keys it trusts."""
import argparse
import json

import pytest

from runtime.host import bootstrap, controller, single_uplink, topology


@pytest.mark.parametrize("answer,approved", [("", True), ("Y", True), ("yes", True), ("n", False), ("no", False)])
def test_default_yes_approval_accepts_enter(monkeypatch, answer, approved):
    monkeypatch.setattr(controller.sys.stdin, "isatty", lambda: True)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or answer)
    if approved:
        controller.confirm("Proceed?", default=True)
    else:
        with pytest.raises(ValueError, match="Cancelled"):
            controller.confirm("Proceed?", default=True)
    assert prompts == ["Proceed? [Y/n]: "]


def test_ordinary_confirmation_still_defaults_to_no(monkeypatch):
    monkeypatch.setattr(controller.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    with pytest.raises(ValueError, match="Cancelled"):
        controller.confirm("Stop these containers?")


def test_setup_summary_is_one_question(monkeypatch, capsys):
    monkeypatch.setattr(controller.sys.stdin, "isatty", lambda: True)
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "")
    args = argparse.Namespace(ssh_user="cody", ssh_port=22, no_share_internet=False)
    single_uplink.approve(args, fresh=True, follow="then install qwen38-flash-next-tp2 and start it")
    out = capsys.readouterr().out
    assert asked == ["Proceed? [Y/n]: "]
    assert "sign in as cody" in out and "host key on first contact" in out and "qwen38-flash-next-tp2" in out
    # A fresh setup does not know the ring size, so the one question also covers the ConnectX restarts.
    assert "about 8 seconds" in out and "now and at every boot" in out
    assert out.index("ConnectX hairpin setting") < out.index("qwen38-flash-next-tp2")


def test_yes_prints_the_approved_scope_without_asking(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("--yes asked a question"))
    args = argparse.Namespace(ssh_user="cody", ssh_port=22, no_share_internet=False)
    single_uplink.announce(args, fresh=False, follow="then install qwen38-flash-next-qad-tp4 and start it", four=True)
    out = capsys.readouterr().out
    assert out.startswith("Approved with --yes:\n")
    assert "\n".join(single_uplink.HAIRPIN_SCOPE) in out and "qwen38-flash-next-qad-tp4" in out


def test_hairpin_scope_is_listed_for_fresh_and_four_spark_setups_only(tmp_path):
    args = argparse.Namespace(ssh_user="cody", ssh_port=22, no_share_internet=False)
    assert single_uplink.ring_state(tmp_path) == (True, True)
    (tmp_path / "enrolled.json").write_text('{"targets": ["root@192.0.2.10", "root@192.0.2.11"]}')
    fresh, four = single_uplink.ring_state(tmp_path)
    assert (fresh, four) == (False, False)
    assert not set(single_uplink.HAIRPIN_SCOPE) & set(single_uplink.scope_lines(args, fresh=fresh, four=four))
    (tmp_path / "enrolled.json").write_text('{"targets": ["a", "b", "c", "d"]}')
    assert single_uplink.ring_state(tmp_path) == (False, True)


def test_setup_summary_prints_each_ranks_hairpin_line(capsys):
    from runtime.host.test_hairpin_ring import needing, ring_plan
    plan = needing(ring_plan(), [3])
    controller.summarize(plan)
    out = capsys.readouterr().out
    assert "  rank 0: root@192.0.2.10  none  driver: none\n" in out
    assert "    ConnectX hairpin: in effect; applied at every boot" in out
    assert "    ConnectX hairpin: restart 4 functions after addressing (1024 -> 8192), about 8 s link loss each" in out


def test_direct_setup_holds_the_installation_lock(tmp_path, monkeypatch, capsys):
    from runtime.common import process_lock
    monkeypatch.setattr(controller, "STATE", tmp_path)
    monkeypatch.setattr(controller, "setup", lambda argv: pytest.fail("setup ran without the lock"))
    with process_lock.hold(tmp_path / "install.lock"):
        assert controller.main(["setup", "--node", "root@192.0.2.10", "--node", "root@192.0.2.11", "--apply"]) == 2
    assert "Another operation is active" in capsys.readouterr().err


def test_setup_prints_needs_input_details(tmp_path, monkeypatch, capsys):
    from runtime.host.install_errors import NeedsInput
    monkeypatch.setattr(controller, "STATE", tmp_path)

    def setup(argv):
        raise NeedsInput("The ConnectX hairpin setting must be applied on 1 Spark, and 1 Spark is in use.",
                         field="driver", details={"lines": ["rank 1 (spark1):", "  - sparkring-mesh.service is active"]})
    monkeypatch.setattr(controller, "setup", setup)
    assert controller.main(["setup", "--node", "root@192.0.2.10", "--plan"]) == 2
    err = capsys.readouterr().err
    assert "SparkRing: The ConnectX hairpin setting must be applied" in err
    assert "  rank 1 (spark1):\n    - sparkring-mesh.service is active" in err


def test_approved_logins_record_new_host_keys_in_a_listed_file(tmp_path):
    route = [{"user": "cody", "address": "fe80::2", "interface": "port0", "port": 22}]
    approved = bootstrap.ssh_argv(route, tmp_path, interactive=True, trust_new=True)
    assert "StrictHostKeyChecking=accept-new" in approved and "HashKnownHosts=no" in approved
    assert f"UserKnownHostsFile={tmp_path / 'known_hosts'} ~/.ssh/known_hosts" in approved
    assert "StrictHostKeyChecking=ask" in bootstrap.ssh_argv(route, tmp_path, interactive=True)
    assert "StrictHostKeyChecking=yes" in bootstrap.ssh_argv(route, tmp_path)


def socket_direct_pair():
    """Two Sparks, one cable, two PCIe functions per port; each sees both remote functions."""
    def spark(name, prefix):
        return {"id": name, "hostname": name, "architecture": "aarch64", "routes": [], "os": {},
                "functions": [{"netdev": "p0", "mac": prefix + ":01", "addresses": [f"fe80::{name}1"]},
                              {"netdev": "p0b", "mac": prefix + ":02", "addresses": [f"fe80::{name}2"]}]}
    a, b = spark("a", "02:00:00:00:0a"), spark("b", "02:00:00:00:0b")
    for here, there in ((a, b), (b, a)):
        here["neighbors"] = [{"dev": dev, "dst": f["addresses"][0], "lladdr": f["mac"]}
                             for dev in ("p0", "p0b") for f in there["functions"]]
    return a, b


def test_discovery_signs_in_once_per_spark_on_socket_direct_links():
    a, b = socket_direct_pair()
    logins = []

    class Transport:
        def login(self, route):
            logins.append(route[-1]["address"])

        def inventory(self, route):
            return b if route else a

    result = bootstrap.discover(Transport())
    assert logins == ["fe80::b1"]
    assert result["head"] == "a" and [n["id"] for n in result["nodes"]] == ["a", "b"] and len(result["edges"]) == 1


# Setup entry points on four Sparks.

@pytest.fixture
def four_sparks(tmp_path, monkeypatch):
    """Four enrolled Sparks for sparkring setup --node; SSH answers and every host read are simulated."""
    from runtime.host.test_appliance import nodes
    from runtime.host.test_hairpin_ring import document
    found = nodes(4)
    plan = topology.build_spec(found, found[0]["node_id"])
    for rank, current in enumerate(found):
        current["hairpin"] = document(plan, rank, approved=False, armed=False)
    monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    monkeypatch.setattr(controller.node, "read", lambda root, name: {"node_id": found[0]["node_id"]})
    monkeypatch.setattr(controller, "collect", lambda targets: found)
    monkeypatch.setattr(controller.distribution, "identity", lambda root: "a" * 40)
    calls = []

    def ssh(host, argv, *, data=None, **options):
        calls.append((host, list(argv)))
        if "native-mesh" in argv:
            rank = int(argv[-1])
            return json.dumps({"mesh": {"reference": {"site_path": "/etc/sparkring/managed-mesh/site.json",
                                                      "site_sha256": "c" * 64, "plan_sha256": "d" * 64},
                                        "host_ip": f"192.0.2.{110 + rank}"}})
        return "{}"
    monkeypatch.setattr(controller.discovery, "ssh", ssh)
    targets = [argument for current in found for argument in ("--node", current["facts"]["ssh_target"])]
    return found, plan, calls, targets


def test_setup_with_nodes_passes_its_approval_to_the_hairpin_step(four_sparks, monkeypatch, capsys, tmp_path):
    from runtime.host.test_hairpin_ring import document
    from scripts import hairpin_setting
    found, plan, _, targets = four_sparks
    found[3]["hairpin"] = document(plan, 3, hairpin_setting.DEFAULT, approved=False, armed=False)
    received = []
    monkeypatch.setattr(controller, "apply", lambda value, directory, **options: received.append(options) or value)
    assert controller.setup([*targets, "--apply", "--yes", "--skip-enroll", "--output", str(tmp_path / "setup")]) == 0
    assert [options["approved"] for options in received] == [True]
    assert ("    ConnectX hairpin: restart 4 functions after addressing (1024 -> 8192), about 8 s link loss each"
            in capsys.readouterr().out)


def test_adoption_on_four_sparks_records_the_setting_without_restarting(four_sparks, monkeypatch, capsys, tmp_path):
    from runtime.host import hairpin_ring
    from runtime.host.test_hairpin_ring import document
    from scripts import hairpin_setting
    found, plan, calls, targets = four_sparks
    found[3]["hairpin"] = document(plan, 3, hairpin_setting.DEFAULT, approved=False, armed=False)
    monkeypatch.setattr(controller.sys.stdin, "isatty", lambda: True)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "y")
    ensured = []

    def ensure(value, **options):
        ensured.append((sum(1 for _, argv in calls if argv[3:5] == ["node", "adopt"]), options))
        options["record"].update(state="partial", ranks=[
            {"rank": rank, "after": hairpin_ring.KEPT if rank < 3 else hairpin_ring.RESTART} for rank in range(4)])
        return value
    monkeypatch.setattr(controller.hairpin_ring, "ensure", ensure)
    assert controller.setup([*targets, "--adopt", "--apply", "--skip-enroll", "--output", str(tmp_path / "adopt")]) == 0
    assert prompts == ["Record this verified existing fabric without network changes, and record the ConnectX hairpin "
                       "setting that is in effect and apply it at every boot (no driver restart)? [y/N]: "]
    # The hairpin step runs once, after node adopt on every Spark, and never restarts a function.
    [(adopted, options)] = ensured
    assert adopted == 4 and options["approved"] is True and options["restart"] is False
    receipt = json.loads((tmp_path / "adopt/setup.json").read_text())
    assert receipt["hairpin"] == "partial" and receipt["network_changed"] is False
    out = capsys.readouterr().out.splitlines()
    assert not any("restart 4 functions after addressing" in line for line in out)
    assert any(line.startswith("    ConnectX hairpin: not in effect (") and line.endswith(
        "adoption does not restart it; sudo sparkring hairpin applies it afterwards") for line in out)
    assert ("Existing fabric verified. No link or route changes; sparkring-hairpin.service applies the ConnectX "
            "hairpin setting at every boot on ranks 0-2.") in out


def test_setup_accepts_the_driver_reload_flag_and_explains_it(tmp_path, monkeypatch, capsys):
    from runtime.host.test_appliance import nodes
    found = nodes(2)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(found))
    assert controller.setup(["--inventory", str(inventory), "--head-id", found[0]["node_id"], "--plan",
                             "--allow-driver-reload", "--output", str(tmp_path / "plan")]) == 0
    assert "--allow-driver-reload: " + controller.ALLOW_DRIVER_RELOAD in capsys.readouterr().out
    monkeypatch.setattr(single_uplink.os, "geteuid", lambda: 1000, raising=False)
    with pytest.raises(ValueError, match="Run sudo sparkring setup"):
        single_uplink.main(["--allow-driver-reload"])
    assert "--allow-driver-reload: " + controller.ALLOW_DRIVER_RELOAD in capsys.readouterr().out


def test_fresh_setup_plan_lists_the_hairpin_step_when_it_finds_four_sparks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    monkeypatch.setattr(single_uplink.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(single_uplink.distribution, "installed", lambda root: True)
    # Planning never reads the controller's public key.
    monkeypatch.setattr(single_uplink, "identity_key", lambda directory: (tmp_path / "key", "controller public key"))

    class Transport:
        def __init__(self, *args, **kwargs):
            pass

        def trusted(self):
            return []
    monkeypatch.setattr(single_uplink.bootstrap, "SSH", Transport)
    for count in (4, 2):
        found = {"head": "n0", "nodes": [{"id": f"n{index}", "hostname": f"spark{index}"} for index in range(count)],
                 "routes": {}, "edges": []}
        monkeypatch.setattr(single_uplink.bootstrap, "discover", lambda transport, found=found, **options: found)
        assert single_uplink.main(["--plan"]) == 0
        out = capsys.readouterr().out
        listed = "Setup of these Sparks also includes this step:\n" + "\n".join(single_uplink.HAIRPIN_SCOPE) in out
        assert listed is (count == 4)
