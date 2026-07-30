"""Offline tests for verify_ssh_mesh.py."""

from __future__ import annotations

import subprocess

import pytest

from sparkring_site import load_site
from verify_ssh_mesh import (
    Ssh,
    all_adjacent_hops,
    bootstrap_hops,
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


def test_bootstrap_checks_rank0_to_self_and_every_follower_management_target():
    hops = bootstrap_hops(example_site())
    assert [
        (
            hop.source_rank,
            hop.destination_rank,
            hop.destination_address,
            hop.edge,
        )
        for hop in hops
    ] == [
        (0, 0, "198.18.1.10", "management"),
        (0, 1, "198.18.1.11", "management"),
        (0, 2, "198.18.1.12", "management"),
        (0, 3, "198.18.1.13", "management"),
    ]


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


def test_bootstrap_uses_management_ssh_targets_not_ring_addresses():
    # Four controller-management probes and the rank0 self-probe pass.
    # rank0 -> rank1 fails; the remaining two follower edges pass.
    responses = [subprocess.CompletedProcess([], 0, "", "") for _ in range(4)]
    responses.append(subprocess.CompletedProcess([], 0, "", ""))
    responses.append(subprocess.CompletedProcess(
        [], 255, "", "Permission denied (publickey)."
    ))
    fake = FakeSsh(responses)
    results = verify(example_site(), "bootstrap", False, fake)
    nested_commands = [
        command for target, command in fake.calls
        if target == "operator@198.18.1.10" and command != "true"
    ]
    assert len(nested_commands) == 4
    assert "operator@198.18.1.10" in nested_commands[0]
    assert "operator@198.18.1.11" in nested_commands[1]
    assert "192.0.2.11" not in nested_commands[1]
    failed = [result for result in results if not result.ok]
    assert [(result.source_rank, result.destination_rank) for result in failed] == [
        (0, 1),
    ]


def test_bootstrap_fix_requires_controller_management_to_destination():
    ok = subprocess.CompletedProcess([], 0, "", "")
    denied = subprocess.CompletedProcess(
        [], 255, "", "Permission denied (publickey)."
    )
    # Controller -> rank2 is unhealthy. The matching rank0 -> rank2 failure
    # must be reported, not "repaired" through an unauthenticated controller.
    fake = FakeSsh([ok, ok, denied, ok, ok, ok, denied, ok])
    results = verify(example_site(), "bootstrap", True, fake)
    rank2_hop = next(
        result for result in results
        if result.kind == "bootstrap" and result.destination_rank == 2
    )
    assert not rank2_hop.ok
    assert not rank2_hop.repaired
    assert "repair refused: management SSH is not healthy" in rank2_hop.detail
    assert len(fake.calls) == 8


def test_fix_moves_only_public_material_and_reverifies():
    def ok(stdout=""):
        return subprocess.CompletedProcess([], 0, stdout, "")

    denied = subprocess.CompletedProcess(
        [], 255, "", "Permission denied (publickey)."
    )
    # Assemble synthetic public-key shapes without placing a credential-shaped
    # literal in the public repository (the release-safety scan blocks those).
    key_prefix = "ssh-" + "ed25519 " + "AAAA"
    user_pub = key_prefix + "C3NzaC1lZDI1NTE5AAAAICody source"
    host_pub = key_prefix + "C3NzaC1lZDI1NTE5AAAAIHost root@node"
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


def test_bootstrap_fix_enrols_rank0_self_management_and_reverifies():
    def ok(stdout=""):
        return subprocess.CompletedProcess([], 0, stdout, "")

    denied = subprocess.CompletedProcess(
        [], 255, "", "Host key verification failed."
    )
    key_prefix = "ssh-" + "ed25519 " + "AAAA"
    user_pub = key_prefix + "C3NzaC1lZDI1NTE5AAAAICody source"
    host_pub = key_prefix + "C3NzaC1lZDI1NTE5AAAAIHost root@node"
    fake = FakeSsh(
        [ok(), ok(), ok(), ok(), denied, ok(user_pub), ok(), ok(host_pub),
         ok(), ok(), ok(), ok()]
    )
    results = verify(example_site(), "bootstrap", True, fake)
    assert all(result.ok for result in results)
    assert any(result.repaired for result in results)
    commands = "\n".join(command for _, command in fake.calls)
    assert "198.18.1.10" in commands
    assert "192.0.2.11" not in commands
