"""Setup asks once, recognizes known Sparks without new logins and records host keys it trusts."""
import argparse

import pytest

from runtime.host import bootstrap, controller, single_uplink


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
