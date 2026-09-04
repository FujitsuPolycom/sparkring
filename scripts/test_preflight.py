#!/usr/bin/env python3
"""Tests for the read-only SparkRing preflight checker.

Entirely offline: no GPU, no cluster, no ssh, no network.  Every remote
interaction goes through a fake runner, and every probe transcript is
synthesised from the site configuration so the parser and the evaluator are
exercised against exactly the record format the real script emits.

Run with::

    python -m pytest scripts/test_preflight.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import preflight  # noqa: E402
from preflight import (  # noqa: E402
    CHECK_DESCRIPTIONS,
    CHECK_IDS,
    CommandResult,
    ReadOnlyViolation,
    assert_read_only,
    build_evidence,
    build_probe_script,
    check_rank,
    evaluate_rank,
    exit_code_for,
    parse_probe_output,
    render_text,
    run_preflight,
    summarise,
)
from sparkring_site import ipv4_mapped_gid, load_site, validate_site  # noqa: E402
from test_sparkring_site import six_ring_document  # noqa: E402

EXAMPLE_PATH = (
    Path(__file__).resolve().parent / "config" / "exl3-r7-site.example.yaml"
)


@pytest.fixture(scope="session")
def site():
    return load_site(EXAMPLE_PATH)


# ==========================================================================
# Synthetic probe transcripts
# ==========================================================================


def healthy_lines(site, rank) -> list[str]:
    """A transcript in which every check on ``rank`` passes."""
    management = rank.management
    lines = [
        preflight.PROBE_SENTINEL,
        f"IPROW 2: {management.interface}    inet {management.address}/24 "
        f"brd 198.18.1.255 scope global {management.interface}\\"
        "       valid_lft forever preferred_lft forever",
    ]
    for index, port in enumerate(rank.ring_ports):
        lines.append(
            f"IF {port.interface} up {site.topology.mtu} "
            f"{site.topology.link_speed_mbps}"
        )
        lines.append(
            f"IPROW {3 + index}: {port.interface}    inet {port.cidr} "
            f"brd 192.0.2.255 scope global {port.interface}\\"
            "       valid_lft forever preferred_lft forever"
        )
        lines.append(f"RDMA_STATE {port.rdma_key} 4: ACTIVE")
        lines.append(f"RDMA_LINK {port.rdma_key} Ethernet")
        gid_key = f"{port.rdma_key}:{port.roce_gid_index}"
        lines.append(f"GID {gid_key} {ipv4_mapped_gid(port.address)}")
        lines.append(f"GID_TYPE {gid_key} RoCE v2")
        lines.append(f"PING {port.edge} ok")
    for index in range(len(rank.transport_peers)):
        lines.append(f"PEER {index} ok")
    for index, artifact in enumerate(site.artifacts):
        mode = "exec" if artifact.executable else "noexec"
        lines.append(f"ART {index} {artifact.sha256} present {mode}")
    lines.append("SSROW LISTEN 0     4096   0.0.0.0:22    0.0.0.0:*")
    lines.append("SSROW LISTEN 0     4096   [::]:22       [::]:*")
    for index, (_label, path, minimum) in enumerate(
        site.paths.remote_space_targets()
    ):
        available_kib = minimum // 1024 + 4096
        lines.append(f"DIR {index} yes")
        lines.append(f"DFPATH {index}")
        lines.append(
            "DFROW Filesystem 1024-blocks Used Available Capacity Mounted on"
        )
        lines.append(
            f"DFROW /dev/nvme0n1p1 4000000000 1000 {available_kib} 1% {path}"
        )
    lines.append(f"IMAGE_ID {site.runtime.container_image_digest}")
    lines.append(
        f"IMAGE_REPODIGESTS {site.runtime.container_image}"
        f"@{site.runtime.container_image_digest}"
    )
    return lines


def healthy_transcript(site, rank) -> str:
    return "\n".join(healthy_lines(site, rank)) + "\n"


def memory_site():
    document = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))
    document["preflight"]["memory"] = {
        "minimum_available_bytes": 103079215104,
        "contiguous_block_bytes": 33554432,
        "minimum_contiguous_blocks": 200,
    }
    return validate_site(document)


def memory_lines(
    *, available_kib: int, order12: int, order13: int, order14: int = 0
) -> list[str]:
    return [
        "MEM_PAGE_SIZE 4096",
        f"MEM_AVAILABLE_KIB {available_kib}",
        "BUDDY Normal "
        + " ".join(
            ["0"] * 12 + [str(order12), str(order13), str(order14)]
        ),
        "VMSTAT compact_stall 2700000",
        "VMSTAT compact_fail 1350000",
        "VMSTAT compact_success 1350000",
    ]


def healthy_fabric_transcript(site, rank) -> str:
    allowed = (
        preflight.PROBE_SENTINEL,
        "IPROW ",
        "IF ",
        "RDMA_STATE ",
        "RDMA_LINK ",
        "GID ",
        "GID_TYPE ",
        "PING ",
        "PEER ",
    )
    return "\n".join(
        line for line in healthy_lines(site, rank)
        if line == allowed[0] or line.startswith(allowed[1:])
    ) + "\n"


class FakeRunner:
    """Stand-in for ssh.  Refuses anything the read-only guard rejects."""

    def __init__(self, responses: dict[str, CommandResult]) -> None:
        self.responses = responses
        self.commands: list[tuple[str, str]] = []

    def run(self, target: str, command: str) -> CommandResult:
        assert_read_only(command)
        self.commands.append((target, command))
        return self.responses.get(
            target, CommandResult(ok=False, detail="no such fake host")
        )


def healthy_runner(site) -> FakeRunner:
    return FakeRunner({
        rank.ssh_target: CommandResult(
            ok=True, stdout=healthy_transcript(site, rank)
        )
        for rank in site.ranks
    })


# ==========================================================================
# Read-only guard
# ==========================================================================


MUTATING_COMMANDS = [
    "docker stop my-container",
    "docker start my-container",
    "docker rm -f my-container",
    "docker run --rm alpine true",
    "docker exec my-container true",
    "docker pull registry.example/x:1",
    "rm -rf /var/lib/cache",
    "mkdir -p /var/lib/sparkring/jit-cache",
    "touch /tmp/marker",
    "mv /tmp/a /tmp/b",
    "cp /tmp/a /tmp/b",
    "chmod +x /opt/bin/probe",
    "chown root /opt/bin/probe",
    "ip link set eth1 mtu 9000",
    "ip addr add 192.0.2.10/24 dev eth1",
    "ip route add 192.0.2.0/24 dev eth1",
    "ethtool -s eth1 speed 200000",
    "systemctl restart docker",
    "modprobe mlx5_core",
    "sysctl -w net.ipv4.ip_forward=1",
    "kill 1234",
    "pkill -f vllm",
    "reboot",
    "apt-get install -y ethtool",
    "sed -i 's/a/b/' /etc/hosts",
    "cat /proc/meminfo > /tmp/mem.txt",
    "echo hello >> /tmp/log",
    "dd if=/dev/zero of=/tmp/blob bs=1M count=1",
    "sha256sum /opt/x | tee /tmp/hash",
    "bash -c 'echo x > /dev/tcp/198.18.1.10/8000'",
    "crontab -l | crontab -",
    "nmcli con up eth0",
]

READ_ONLY_COMMANDS = [
    "cat /sys/class/net/eth1/mtu",
    "cat /sys/class/net/eth1/operstate 2>/dev/null",
    "ip -o -4 addr 2>/dev/null | sed 's/^/IPROW /'",
    "ss -ltnH 2>/dev/null | sed 's/^/SSROW /'",
    "df -Pk '/var/lib/sparkring/jit-cache' 2>/dev/null",
    "sha256sum -- '/opt/sparkring/lib/lib.so' 2>/dev/null | cut -d' ' -f1",
    "test -x '/opt/sparkring/bin/probe'",
    "docker image inspect 'registry.example/x:1' --format '{{.Id}}'",
    "docker inspect my-container",
    "docker ps",
    "ping -c 2 -W 2 -M do -s 8972 192.0.2.11 >/dev/null 2>&1",
    "ethtool eth1",
    "nvidia-smi --query-gpu=name --format=csv",
]


@pytest.mark.parametrize("command", MUTATING_COMMANDS)
def test_read_only_guard_rejects_mutating_commands(command):
    with pytest.raises(ReadOnlyViolation):
        assert_read_only(command)


@pytest.mark.parametrize("command", READ_ONLY_COMMANDS)
def test_read_only_guard_allows_read_only_commands(command):
    assert_read_only(command)


def test_generated_probe_script_is_read_only(site):
    for rank in site.ranks:
        script = build_probe_script(site, rank)
        assert_read_only(script)


def test_generated_probe_script_never_mentions_mutating_docker_verbs(site):
    for rank in site.ranks:
        script = build_probe_script(site, rank)
        for verb in ("docker start", "docker stop", "docker run",
                     "docker rm", "docker exec", "docker pull"):
            assert verb not in script


def test_fabric_probe_omits_unresolved_deployment_surfaces(site):
    script = build_probe_script(site, site.rank(0), scope="fabric")
    assert "RDMA_STATE" in script
    assert "GID_TYPE" in script
    assert "PING " in script
    assert "PEER " in script
    for deployment_probe in (
        "sha256sum",
        "ss -ltnH",
        "DFPATH",
        "docker image inspect",
    ):
        assert deployment_probe not in script


def test_runner_protocol_rejects_a_mutating_command():
    runner = FakeRunner({})
    with pytest.raises(ReadOnlyViolation):
        runner.run("operator@198.18.1.10", "docker stop x")


# ==========================================================================
# Probe script construction
# ==========================================================================


def test_probe_script_contains_the_readiness_sentinel(site):
    script = build_probe_script(site, site.rank(0))
    assert script.splitlines()[2] == f'echo "{preflight.PROBE_SENTINEL}"'


def test_probe_script_pings_the_far_end_of_each_edge_with_the_mtu(site):
    payload = site.topology.jumbo_payload_bytes
    for rank in site.ranks:
        script = build_probe_script(site, rank)
        for port in rank.ring_ports:
            assert (
                f"ping -c 2 -W 2 -M do -s {payload} {port.peer_address}"
                in script
            )


def test_probe_script_reads_the_configured_gid_index(site):
    rank = site.rank(0)
    script = build_probe_script(site, rank)
    for port in rank.ring_ports:
        base = (
            f"/sys/class/infiniband/{port.rdma_device}"
            f"/ports/{port.rdma_port}"
        )
        assert f"{base}/gids/{port.roce_gid_index}" in script
        assert f"{base}/gid_attrs/types/{port.roce_gid_index}" in script


def test_probe_script_covers_every_pinned_artifact(site):
    script = build_probe_script(site, site.rank(0))
    for artifact in site.artifacts:
        assert f"sha256sum -- '{artifact.path}'" in script


def test_probe_script_quotes_remote_paths(site):
    script = build_probe_script(site, site.rank(0))
    for _label, path, _minimum in site.paths.remote_space_targets():
        assert f"df -Pk '{path}'" in script


def test_probe_script_omits_the_socket_listing_when_no_ports_required(site):
    stripped = load_site(EXAMPLE_PATH)
    object.__setattr__(stripped.preflight, "required_free_ports", ())
    script = build_probe_script(stripped, stripped.rank(0))
    assert "ss -ltnH" not in script


def test_probe_script_is_deterministic(site):
    first = build_probe_script(site, site.rank(1))
    second = build_probe_script(site, site.rank(1))
    assert first == second


def test_memory_probe_reads_page_size_buddyinfo_and_compaction_counters():
    site = memory_site()
    script = build_probe_script(site, site.rank(0))

    assert "getconf PAGESIZE" in script
    assert "/proc/meminfo" in script
    assert "/proc/buddyinfo" in script
    assert "/proc/vmstat" in script
    assert (
        "awk '$1 == \"MemAvailable:\" "
        "{print \"MEM_AVAILABLE_KIB\", $2}' /proc/meminfo"
    ) in script
    assert_read_only(script)


def test_probe_script_differs_per_rank(site):
    assert build_probe_script(site, site.rank(0)) != build_probe_script(
        site, site.rank(1)
    )


# ==========================================================================
# Parsing
# ==========================================================================


def test_parse_healthy_transcript(site):
    rank = site.rank(0)
    state = parse_probe_output(healthy_transcript(site, rank))
    assert state.ready
    assert state.interfaces_holding(str(rank.management.address)) == [
        rank.management.interface
    ]
    for port in rank.ring_ports:
        assert state.interfaces[port.interface]["mtu"] == str(
            site.topology.mtu
        )
        assert state.addresses[port.interface] == {port.cidr}
        assert "ACTIVE" in state.rdma_state[port.rdma_key]
        assert state.rdma_link[port.rdma_key] == "Ethernet"
        gid_key = f"{port.rdma_key}:{port.roce_gid_index}"
        assert state.gids[gid_key] == ipv4_mapped_gid(port.address)
        assert state.gid_types[gid_key] == "RoCE v2"
        assert state.pings[port.edge] == "ok"
    assert state.listening_ports == {22}
    assert state.image_id == site.runtime.container_image_digest
    assert state.image_repo_digests


def test_parse_ignores_the_df_header_row(site):
    state = parse_probe_output(healthy_transcript(site, site.rank(0)))
    for index, (_label, _path, minimum) in enumerate(
        site.paths.remote_space_targets()
    ):
        assert state.available_kib[index] * 1024 > minimum


def test_parse_empty_output_is_not_ready():
    assert parse_probe_output("").ready is False


def test_parse_tolerates_unknown_and_short_records():
    state = parse_probe_output(
        f"{preflight.PROBE_SENTINEL}\nWAT\nIF\nIPROW\nGID\n"
    )
    assert state.ready
    assert state.interfaces == {}


def test_parse_extracts_ports_from_ipv6_socket_rows():
    state = parse_probe_output(
        "SSROW LISTEN 0 4096 [::]:29500 [::]:*\n"
        "SSROW LISTEN 0 4096 0.0.0.0:8000 0.0.0.0:*\n"
    )
    assert state.listening_ports == {29500, 8000}


def test_parse_accumulates_normal_zone_buddy_orders_across_nodes():
    state = parse_probe_output(
        "MEM_PAGE_SIZE 4096\n"
        "MEM_AVAILABLE_KIB 120000000\n"
        "BUDDY Normal 0 0 0 0 0 0 0 0 0 0 0 0 40 10\n"
        "BUDDY Normal 0 0 0 0 0 0 0 0 0 0 0 0 24 2\n"
        "BUDDY DMA 0 0 0 0 0 0 0 0 0 0 0 0 999 999\n"
        "VMSTAT compact_stall 2700000\n"
        "VMSTAT compact_fail 1350000\n"
    )

    assert state.memory_page_size == 4096
    assert state.memory_available_kib == 120000000
    assert state.buddy_orders["Normal"][12:] == [64, 12]
    assert state.vmstat["compact_fail"] == 1350000


# ==========================================================================
# Evaluation: happy path
# ==========================================================================


def test_every_check_passes_on_a_healthy_transcript(site):
    for rank in site.ranks:
        state = parse_probe_output(healthy_transcript(site, rank))
        results = evaluate_rank(site, rank, state)
        failures = [result for result in results if not result.passed]
        assert not failures, [
            (result.check_id, result.subject, result.detail)
            for result in failures
        ]


def test_memory_headroom_accepts_rebooted_gb10_geometry():
    site = memory_site()
    rank = site.rank(0)
    lines = healthy_lines(site, rank) + memory_lines(
        available_kib=120_000_000,
        order12=1417,
        order13=782,
    )

    results = evaluate_rank(site, rank, parse_probe_output("\n".join(lines)))
    by_id = {result.check_id: result for result in results}

    assert by_id["HOST.MEMORY_AVAILABLE"].passed
    assert by_id["HOST.MEMORY_CONTIGUITY"].passed
    assert "equivalent_32MiB_blocks=782" in (
        by_id["HOST.MEMORY_CONTIGUITY"].detail
    )


def test_memory_headroom_rejects_highly_fragmented_gb10_geometry():
    site = memory_site()
    rank = site.rank(0)
    lines = healthy_lines(site, rank) + memory_lines(
        available_kib=120_000_000,
        order12=27,
        order13=0,
    )

    results = evaluate_rank(site, rank, parse_probe_output("\n".join(lines)))
    by_id = {result.check_id: result for result in results}

    assert by_id["HOST.MEMORY_AVAILABLE"].passed
    assert not by_id["HOST.MEMORY_CONTIGUITY"].passed
    assert "equivalent_32MiB_blocks=0" in (
        by_id["HOST.MEMORY_CONTIGUITY"].detail
    )
    assert "compact_fail=1350000" in by_id["HOST.MEMORY_CONTIGUITY"].detail


def test_memory_headroom_rejects_82_equivalent_32mib_blocks():
    site = memory_site()
    rank = site.rank(0)
    lines = healthy_lines(site, rank) + memory_lines(
        available_kib=120_000_000,
        order12=1_000,
        order13=82,
    )

    results = evaluate_rank(site, rank, parse_probe_output("\n".join(lines)))
    contiguous = next(
        result for result in results
        if result.check_id == "HOST.MEMORY_CONTIGUITY"
    )

    assert not contiguous.passed
    assert "equivalent_32MiB_blocks=82, want >= 200" in contiguous.detail


def test_memory_headroom_derives_buddy_order_from_64k_pages():
    site = memory_site()
    rank = site.rank(0)
    lines = healthy_lines(site, rank) + [
        "MEM_PAGE_SIZE 65536",
        "MEM_AVAILABLE_KIB 120000000",
        "BUDDY Normal 0 0 0 0 0 0 0 0 64 250",
        "VMSTAT compact_stall 20",
        "VMSTAT compact_fail 5",
        "VMSTAT compact_success 15",
    ]

    results = evaluate_rank(site, rank, parse_probe_output("\n".join(lines)))
    contiguous = next(
        result for result in results
        if result.check_id == "HOST.MEMORY_CONTIGUITY"
    )

    assert contiguous.passed
    assert "target_order=9" in contiguous.detail
    assert "equivalent_32MiB_blocks=250" in contiguous.detail


def test_memory_headroom_rejects_missing_kernel_evidence():
    site = memory_site()
    rank = site.rank(0)

    results = evaluate_rank(
        site,
        rank,
        parse_probe_output(healthy_transcript(site, rank)),
    )
    by_id = {result.check_id: result for result in results}

    assert not by_id["HOST.MEMORY_AVAILABLE"].passed
    assert not by_id["HOST.MEMORY_CONTIGUITY"].passed
    assert "unavailable" in by_id["HOST.MEMORY_CONTIGUITY"].detail


def test_evaluation_emits_only_documented_check_ids(site):
    rank = site.rank(0)
    state = parse_probe_output(healthy_transcript(site, rank))
    emitted = {result.check_id for result in evaluate_rank(site, rank, state)}
    assert emitted <= set(CHECK_IDS)
    assert emitted, "evaluation produced no checks"


def test_healthy_evaluation_covers_the_expected_check_ids(site):
    rank = site.rank(0)
    state = parse_probe_output(healthy_transcript(site, rank))
    emitted = {result.check_id for result in evaluate_rank(site, rank, state)}
    expected = set(CHECK_IDS)
    if not site.artifacts:
        expected -= {
            "ARTIFACT.PRESENT",
            "ARTIFACT.SHA256",
            "ARTIFACT.EXECUTABLE",
        }
    if site.preflight.memory is None:
        expected -= {
            "HOST.MEMORY_AVAILABLE",
            "HOST.MEMORY_CONTIGUITY",
        }
    assert emitted == expected


def test_check_id_table_has_no_duplicates_and_all_have_descriptions():
    assert len(CHECK_IDS) == len(set(CHECK_IDS))
    for check_id in CHECK_IDS:
        assert CHECK_DESCRIPTIONS[check_id].strip()


def test_every_result_names_its_rank(site):
    for rank in site.ranks:
        state = parse_probe_output(healthy_transcript(site, rank))
        for result in evaluate_rank(site, rank, state):
            assert result.rank == rank.id


# ==========================================================================
# Evaluation: degraded transcripts
# ==========================================================================


def _replace(lines, predicate, replacement):
    return [
        replacement(line) if predicate(line) else line for line in lines
    ]


def _drop(lines, predicate):
    return [line for line in lines if not predicate(line)]


DEGRADED_CASES = [
    (
        "link-down",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith(f"IF {rank.ring_ports[0].interface} "),
            lambda line: line.replace(" up ", " down ", 1),
        ),
        "RING.LINK_UP",
    ),
    (
        "mtu-not-jumbo",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith(f"IF {rank.ring_ports[0].interface} "),
            lambda line: line.replace(f" {site.topology.mtu} ", " 1500 ", 1),
        ),
        "RING.MTU",
    ),
    (
        "link-trained-down",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith(f"IF {rank.ring_ports[0].interface} "),
            lambda line: line.replace(
                str(site.topology.link_speed_mbps), "100000"
            ),
        ),
        "RING.LINK_SPEED",
    ),
    (
        "ring-address-missing",
        lambda lines, site, rank: _drop(
            lines,
            lambda line: line.startswith("IPROW")
            and f" {rank.ring_ports[0].cidr} " in line,
        ),
        "RING.ADDRESS",
    ),
    (
        "ring-address-extra",
        lambda lines, site, rank: lines + [
            f"IPROW 9: {rank.ring_ports[0].interface}    inet 10.9.9.9/24 "
            f"scope global {rank.ring_ports[0].interface}\\ valid_lft forever"
        ],
        "RING.ADDRESS",
    ),
    (
        "rdma-port-down",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("RDMA_STATE"),
            lambda line: "RDMA_STATE "
            + line.split()[1] + " 1: DOWN",
        ),
        "RING.RDMA_PORT_ACTIVE",
    ),
    (
        "rdma-link-layer-infiniband",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("RDMA_LINK"),
            lambda line: line.replace("Ethernet", "InfiniBand"),
        ),
        "RING.RDMA_LINK_LAYER",
    ),
    (
        "gid-index-shifted",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("GID "),
            lambda line: " ".join(line.split()[:2])
            + " fe80:0000:0000:0000:0000:0000:0000:0001",
        ),
        "RING.ROCE_GID",
    ),
    (
        "gid-is-rocev1",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("GID_TYPE"),
            lambda line: line.replace("RoCE v2", "RoCE v1"),
        ),
        "RING.ROCE_GID",
    ),
    (
        "gid-missing",
        lambda lines, site, rank: _drop(
            lines, lambda line: line.startswith("GID ")
        ),
        "RING.ROCE_GID",
    ),
    (
        "jumbo-path-fragments",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("PING "),
            lambda line: line.replace(" ok", " fail"),
        ),
        "RING.JUMBO_PING",
    ),
    (
        "control-channel-unreachable",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("PEER "),
            lambda line: line.replace(" ok", " fail"),
        ),
        "PEER.CONTROL_CHANNEL",
    ),
    (
        "artifact-missing",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("ART 0 "),
            lambda line: "ART 0 - absent noexec",
        ),
        "ARTIFACT.PRESENT",
    ),
    (
        "artifact-hash-drifted",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("ART 0 "),
            lambda line: f"ART 0 {'9' * 64} present noexec",
        ),
        "ARTIFACT.SHA256",
    ),
    (
        "artifact-not-executable",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("ART 2 "),
            lambda line: line.replace(" exec", " noexec"),
        ),
        "ARTIFACT.EXECUTABLE",
    ),
    (
        "required-port-already-bound",
        lambda lines, site, rank: lines + [
            "SSROW LISTEN 0 4096 0.0.0.0:8000 0.0.0.0:*"
        ],
        "PORT.FREE",
    ),
    (
        "cache-directory-missing",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("DIR 1 "),
            lambda line: "DIR 1 no",
        ),
        "DISK.PATH_PRESENT",
    ),
    (
        "cache-disk-full",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("DFROW /dev/nvme0n1p1"),
            lambda line: "DFROW /dev/nvme0n1p1 4000000000 1000 1024 99% /x",
        ),
        "DISK.FREE",
    ),
    (
        "df-unreadable",
        lambda lines, site, rank: _drop(
            lines, lambda line: line.startswith("DFROW /dev/")
        ),
        "DISK.FREE",
    ),
    (
        "image-absent",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("IMAGE_ID"),
            lambda line: "IMAGE_ID -",
        ),
        "IMAGE.PRESENT",
    ),
    (
        "image-digest-drifted",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("IMAGE_"),
            lambda line: line.split()[0] + " sha256:" + "7" * 64,
        ),
        "IMAGE.DIGEST",
    ),
    (
        "management-address-not-held",
        lambda lines, site, rank: _drop(
            lines,
            lambda line: line.startswith("IPROW")
            and f" {rank.management.address}/" in line,
        ),
        "MGMT.ADDRESS_PRESENT",
    ),
    (
        "management-address-on-the-wrong-nic",
        lambda lines, site, rank: _replace(
            lines,
            lambda line: line.startswith("IPROW")
            and f" {rank.management.address}/" in line,
            lambda line: line.replace(rank.management.interface, "wlan0"),
        ),
        "MGMT.INTERFACE_MATCH",
    ),
]


@pytest.mark.parametrize(
    "case_id,transform,expected_check_id",
    DEGRADED_CASES,
    ids=[case[0] for case in DEGRADED_CASES],
)
def test_degraded_transcript_fails_the_right_check(
    site, case_id, transform, expected_check_id
):
    if expected_check_id.startswith("ARTIFACT.") and not site.artifacts:
        pytest.skip("the supported R7 image carries no loose host artifacts")
    rank = site.rank(0)
    lines = transform(healthy_lines(site, rank), site, rank)
    state = parse_probe_output("\n".join(lines) + "\n")
    results = evaluate_rank(site, rank, state)
    failed = {result.check_id for result in results if not result.passed}
    assert expected_check_id in failed, (
        f"{case_id}: expected {expected_check_id} to fail, failures were "
        f"{sorted(failed)}"
    )


def test_a_single_degradation_does_not_fail_unrelated_checks(site):
    rank = site.rank(0)
    lines = _replace(
        healthy_lines(site, rank),
        lambda line: line.startswith("PEER "),
        lambda line: line.replace(" ok", " fail"),
    )
    state = parse_probe_output("\n".join(lines) + "\n")
    failed = {
        result.check_id
        for result in evaluate_rank(site, rank, state)
        if not result.passed
    }
    assert failed == {"PEER.CONTROL_CHANNEL"}


# ==========================================================================
# Rank probing, aggregation and exit codes
# ==========================================================================


def test_check_rank_passes_with_a_healthy_fake(site):
    runner = healthy_runner(site)
    results = check_rank(site, site.rank(0), runner)
    assert all(result.passed for result in results)
    assert len(runner.commands) == 1
    assert runner.commands[0][0] == site.rank(0).ssh_target


def test_check_rank_reports_a_single_ssh_failure_when_unreachable(site):
    runner = FakeRunner({})
    results = check_rank(site, site.rank(2), runner)
    assert [result.check_id for result in results] == ["SSH.REACHABLE"]
    assert results[0].passed is False
    assert results[0].rank == 2
    assert "no such fake host" in results[0].detail


def test_check_rank_reports_ssh_failure_when_probe_output_is_empty(site):
    runner = FakeRunner({
        site.rank(0).ssh_target: CommandResult(ok=True, stdout="")
    })
    results = check_rank(site, site.rank(0), runner)
    assert [result.check_id for result in results] == ["SSH.REACHABLE"]
    assert "no output" in results[0].detail


def test_run_preflight_covers_every_rank(site):
    results = run_preflight(site, healthy_runner(site))
    assert {result.rank for result in results} == {0, 1, 2, 3}
    assert exit_code_for(results) == 0


def test_run_preflight_covers_all_six_ring_ranks():
    document = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))
    six_site = validate_site(six_ring_document(document))
    runner = healthy_runner(six_site)

    results = run_preflight(six_site, runner, scope="fabric")
    evidence = build_evidence(six_site, results, scope="fabric")

    assert {result.rank for result in results} == set(range(6))
    assert len(runner.commands) == 6
    assert exit_code_for(results) == 0
    assert [entry["rank"] for entry in evidence["ranks"]] == list(range(6))


def test_fabric_scope_emits_only_connectivity_and_ring_checks(site):
    runner = FakeRunner({
        rank.ssh_target: CommandResult(
            ok=True,
            stdout=healthy_fabric_transcript(site, rank),
        )
        for rank in site.ranks
    })
    results = run_preflight(site, runner, scope="fabric")
    assert results
    assert {
        result.check_id.partition(".")[0] for result in results
    } == {"SSH", "MGMT", "RING", "PEER"}
    assert exit_code_for(results) == 0


def test_fabric_scope_fails_closed_on_wrong_rocev2_gid(site):
    responses = {}
    for rank in site.ranks:
        transcript = healthy_fabric_transcript(site, rank)
        if rank.id == 2:
            expected = ipv4_mapped_gid(rank.ring_ports[0].address)
            transcript = transcript.replace(expected, "0000:0000:0000:0000")
        responses[rank.ssh_target] = CommandResult(
            ok=True,
            stdout=transcript,
        )
    results = run_preflight(site, FakeRunner(responses), scope="fabric")
    assert exit_code_for(results) == 1
    failed = [result for result in results if not result.passed]
    assert any(
        result.rank == 2 and result.check_id == "RING.ROCE_GID"
        for result in failed
    )


def test_run_preflight_can_be_limited_to_selected_ranks(site):
    results = run_preflight(site, healthy_runner(site), ranks=[1, 3])
    assert {result.rank for result in results} == {1, 3}


def test_run_preflight_with_no_selected_ranks_returns_nothing(site):
    assert run_preflight(site, healthy_runner(site), ranks=[]) == []


def test_exit_code_is_one_when_any_rank_fails(site):
    responses = {
        rank.ssh_target: CommandResult(
            ok=True, stdout=healthy_transcript(site, rank)
        )
        for rank in site.ranks
    }
    del responses[site.rank(2).ssh_target]
    results = run_preflight(site, FakeRunner(responses))
    assert exit_code_for(results) == 1
    summary = summarise(results)
    assert summary.failed_ranks == (2,)
    assert summary.failed_check_ids == ("SSH.REACHABLE",)


def test_exit_code_is_one_when_nothing_ran():
    assert exit_code_for([]) == 1


def test_summarise_counts_pass_and_fail(site):
    results = run_preflight(site, healthy_runner(site))
    summary = summarise(results)
    assert summary.total == summary.passed
    assert summary.failed == 0
    assert summary.ok is True
    assert summary.failed_check_ids == ()


def test_run_preflight_never_issues_a_mutating_command(site):
    runner = healthy_runner(site)
    run_preflight(site, runner)
    assert len(runner.commands) == 4
    for _target, command in runner.commands:
        assert_read_only(command)


# ==========================================================================
# Evidence and rendering
# ==========================================================================


def test_evidence_shape_on_success(site):
    results = run_preflight(site, healthy_runner(site))
    evidence = build_evidence(
        site, results, generated_at="2026-01-01T00:00:00Z"
    )
    assert evidence["schema"] == preflight.EVIDENCE_SCHEMA
    assert evidence["generated_at"] == "2026-01-01T00:00:00Z"
    assert evidence["read_only"] is True
    assert evidence["passed"] is True
    assert evidence["site"]["name"] == site.name
    assert evidence["site"]["source"] == site.source
    assert evidence["totals"]["failed"] == 0
    assert evidence["totals"]["checks"] == len(results)
    assert evidence["failed_check_ids"] == []
    assert evidence["failed_ranks"] == []
    assert evidence["known_check_ids"] == list(CHECK_IDS)
    assert evidence["diagnostics"]["schema"] == "sparkring-diagnostics/v1"
    assert evidence["diagnostics"]["passed"] is True
    assert evidence["diagnostics"]["generated_at"] == evidence["generated_at"]
    assert [entry["rank"] for entry in evidence["ranks"]] == [0, 1, 2, 3]
    for entry in evidence["ranks"]:
        for check in entry["checks"]:
            assert set(check) == {
                "check_id", "rank", "subject", "passed", "detail"
            }


def test_evidence_is_json_serialisable_and_stable(site):
    results = run_preflight(site, healthy_runner(site))
    evidence = build_evidence(
        site, results, generated_at="2026-01-01T00:00:00Z"
    )
    first = json.dumps(evidence, indent=2, sort_keys=True)
    assert json.loads(first) == evidence


def test_evidence_records_failures(site):
    responses = {
        rank.ssh_target: CommandResult(
            ok=True, stdout=healthy_transcript(site, rank)
        )
        for rank in site.ranks
    }
    del responses[site.rank(1).ssh_target]
    results = run_preflight(site, FakeRunner(responses))
    evidence = build_evidence(site, results)
    assert evidence["passed"] is False
    assert evidence["failed_ranks"] == [1]
    assert "SSH.REACHABLE" in evidence["failed_check_ids"]


def test_evidence_carries_placeholder_warnings(site):
    results = run_preflight(site, healthy_runner(site))
    evidence = build_evidence(site, results)
    assert evidence["placeholder_warnings"]


def test_render_text_reports_pass(site):
    results = run_preflight(site, healthy_runner(site))
    text = render_text(site, results, warnings=[])
    assert "preflight: PASS" in text
    assert "[FAIL]" not in text
    for rank in site.ranks:
        assert f"rank {rank.id} ({rank.ssh_target})" in text


def test_render_text_reports_fail_with_check_ids(site):
    responses = {
        rank.ssh_target: CommandResult(
            ok=True, stdout=healthy_transcript(site, rank)
        )
        for rank in site.ranks
    }
    del responses[site.rank(3).ssh_target]
    results = run_preflight(site, FakeRunner(responses))
    text = render_text(site, results, warnings=[])
    assert "preflight: FAIL" in text
    assert "failing check ids: SSH.REACHABLE" in text
    assert "failing ranks: 3" in text


def test_render_text_is_ascii_only(site):
    results = run_preflight(site, healthy_runner(site))
    render_text(site, results).encode("ascii")


def test_default_evidence_path_uses_the_configured_directory(site):
    path = preflight.default_evidence_path(site, timestamp="20260101T000000Z")
    assert path.name == "preflight-20260101T000000Z.json"
    assert Path(site.paths.evidence_dir) in path.parents


# ==========================================================================
# CLI
# ==========================================================================


def test_cli_list_checks(capsys):
    assert preflight.main(["--list-checks"]) == 0
    output = capsys.readouterr().out
    for check_id in CHECK_IDS:
        assert check_id in output


def test_cli_print_plan_does_not_contact_anything(capsys):
    assert preflight.main(
        ["--site", str(EXAMPLE_PATH), "--print-plan"]
    ) == 0
    output = capsys.readouterr().out
    assert output.count("# ---- rank") == 4
    assert_read_only(output.replace("# ---- rank", ""))


def test_cli_print_plan_honours_rank_selection(capsys):
    assert preflight.main(
        ["--site", str(EXAMPLE_PATH), "--print-plan", "--rank", "2"]
    ) == 0
    output = capsys.readouterr().out
    assert output.count("# ---- rank") == 1
    assert "rank 2" in output


def test_cli_rejects_an_invalid_site(tmp_path, capsys):
    target = tmp_path / "site.yaml"
    target.write_text("schema_version: 1\n", encoding="utf-8")
    assert preflight.main(["--site", str(target)]) == 1
    assert "INVALID SITE CONFIG" in capsys.readouterr().err


def test_cli_reports_a_missing_site_file(tmp_path, capsys):
    assert preflight.main(["--site", str(tmp_path / "nope.yaml")]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_runs_offline_against_a_fake_runner(site, tmp_path,
                                                monkeypatch, capsys):
    """Full CLI path with ssh replaced - proves exit code and evidence file."""
    runner = healthy_runner(site)
    monkeypatch.setattr(
        preflight, "SshRunner", lambda *args, **kwargs: runner
    )
    evidence_path = tmp_path / "evidence.json"
    status = preflight.main([
        "--site", str(EXAMPLE_PATH), "--json", str(evidence_path),
    ])
    assert status == 0
    captured = capsys.readouterr().out
    assert "preflight: PASS" in captured
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["totals"]["failed"] == 0


def test_cli_fabric_scope_omits_deployment_probes(
    site, monkeypatch, capsys
):
    runner = healthy_runner(site)
    monkeypatch.setattr(
        preflight, "SshRunner", lambda *args, **kwargs: runner
    )
    assert preflight.main([
        "--site", str(EXAMPLE_PATH),
        "--scope", "fabric",
        "--no-evidence",
    ]) == 0
    assert "scope=fabric" in capsys.readouterr().out
    assert len(runner.commands) == 4
    for _target, command in runner.commands:
        assert "RDMA_STATE" in command
        assert "docker image inspect" not in command
        assert "sha256sum" not in command
        assert "df -Pk" not in command
        assert "ss -ltnH" not in command


def test_cli_strict_placeholders_fails_even_when_checks_pass(
    site, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        preflight, "SshRunner",
        lambda *args, **kwargs: healthy_runner(site),
    )
    status = preflight.main([
        "--site", str(EXAMPLE_PATH),
        "--json", str(tmp_path / "evidence.json"),
        "--strict-placeholders",
    ])
    assert status == 1
    assert "placeholder" in capsys.readouterr().err


def test_cli_no_evidence_writes_nothing(site, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        preflight, "SshRunner",
        lambda *args, **kwargs: healthy_runner(site),
    )
    monkeypatch.chdir(tmp_path)
    assert preflight.main([
        "--site", str(EXAMPLE_PATH), "--no-evidence",
    ]) == 0
    assert not (tmp_path / "evidence").exists()
    assert "evidence=" not in capsys.readouterr().out
