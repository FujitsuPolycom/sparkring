"""Offline tests for verify_ssh_mesh.py."""

from __future__ import annotations

import subprocess

import pytest

from sparkring_site import load_site
from verify_ssh_mesh import (
    Ssh,
    all_adjacent_hops,
    classify_failure,
    fanout_hops,
    split_ssh_target,
    verify,
)


def example_site():
    return load_site("scripts/config/site.example.yaml")


def test_split_ssh_target():
    assert split_ssh_target("operator@node0") == ("operator", "node0")
    with pytest.raises(ValueError):
        split_ssh_target("node0")


def test_fanout_is_redundant_four_hop_tree():
    hops = fanout_hops(example_site())
    assert [(hop.source_rank, hop.destination_rank) for hop in hops] == [
        (0, 1), (0, 3), (1, 2), (3, 2),
    ]


def test_all_adjacent_has_eight_directed_hops():
    hops = all_adjacent_hops(example_site())
    assert len(hops) == 8
    assert len({(hop.source_rank, hop.destination_rank) for hop in hops}) == 8


@pytest.mark.parametrize(("message", "expected"), [
    ("Host key verification failed.", "host_key"),
    ("Permission denied (publickey).", "authorization"),
    ("ssh: connect to host x port 22: No route to host", "unreachable"),
    ("Could not resolve hostname x", "name_resolution"),
    ("something else", "ssh_error"),
])
def test_classify_failure(message, expected):
    assert classify_failure(message, 255)[0] == expected


class FakeSsh(Ssh):
    def __init__(self, responses):
        super().__init__(5)
        self.responses = list(responses)
        self.calls = []

    def run(self, target, command):
        self.calls.append((target, command))
        if self.responses:
            return self.responses.pop(0)
        return subprocess.CompletedProcess([], 0, "", "")


def test_verify_reports_exact_failed_direction():
    # Four management probes pass; first direct hop lacks authorization.
    responses = [subprocess.CompletedProcess([], 0, "", "") for _ in range(4)]
    responses.append(subprocess.CompletedProcess(
        [], 255, "", "Permission denied (publickey)."
    ))
    fake = FakeSsh(responses)
    results = verify(example_site(), "fanout", False, fake)
    failed = [result for result in results if not result.ok]
    assert len(failed) == 1
    assert failed[0].source_rank == 0
    assert failed[0].destination_rank == 1
    assert failed[0].status == "authorization"


def test_fix_moves_only_public_material_and_reverifies():
    def ok(stdout=""):
        return subprocess.CompletedProcess([], 0, stdout, "")

    denied = subprocess.CompletedProcess(
        [], 255, "", "Permission denied (publickey)."
    )
    user_pub = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICody source"
    host_pub = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHost root@node"
    # management x4, failed hop, source pub, install auth, host pub,
    # install known_hosts, successful re-probe; remaining three hops pass.
    fake = FakeSsh(
        [ok(), ok(), ok(), ok(), denied, ok(user_pub), ok(), ok(host_pub),
         ok(), ok(), ok(), ok(), ok()]
    )
    results = verify(example_site(), "fanout", True, fake)
    assert all(result.ok for result in results)
    assert any(result.repaired for result in results)
    commands = "\n".join(command for _, command in fake.calls)
    assert "id_ed25519.pub" in commands
    assert "/etc/ssh/ssh_host_ed25519_key.pub" in commands
    assert "id_ed25519\"" in commands  # generated locally if absent
    assert "cat \"$HOME/.ssh/id_ed25519\"" not in commands
