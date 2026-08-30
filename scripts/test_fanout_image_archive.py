from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import fanout_image_archive as fanout


DIGEST = "a" * 64
IMAGE_ID = "sha256:" + "b" * 64
ROOT = Path(__file__).resolve().parents[1]


def _site() -> SimpleNamespace:
    edges = (
        (0, 1, "r0-r1", "10.0.1.10", "10.0.1.11"),
        (1, 2, "r1-r2", "10.0.2.11", "10.0.2.12"),
        (2, 3, "r2-r3", "10.0.3.12", "10.0.3.13"),
        (3, 0, "r3-r0", "10.0.4.13", "10.0.4.10"),
    )
    ports: dict[int, list[SimpleNamespace]] = {rank: [] for rank in range(4)}
    for first, second, edge, first_address, second_address in edges:
        ports[first].append(
            SimpleNamespace(
                edge=edge,
                address=first_address,
                peer_rank=second,
                peer_address=second_address,
            )
        )
        ports[second].append(
            SimpleNamespace(
                edge=edge,
                address=second_address,
                peer_rank=first,
                peer_address=first_address,
            )
        )
    return SimpleNamespace(
        topology=SimpleNamespace(link_speed_mbps=200000, mtu=9000),
        ranks=tuple(
            SimpleNamespace(
                id=rank,
                ssh_target=f"operator@management-rank-{rank}.example",
                ring_ports=tuple(ports[rank]),
                neighbour_ranks=tuple(
                    sorted(port.peer_rank for port in ports[rank])
                ),
            )
            for rank in range(4)
        )
    )


def _plan() -> tuple[SimpleNamespace, fanout.ArchivePaths, dict]:
    site = _site()
    paths = fanout.archive_paths("/var/lib/sparkring/images", "runtime.tar.zst")
    plan = fanout.plan_document(
        site,
        url="https://images.example/runtime.tar.zst",
        expected_sha256=DIGEST,
        paths=paths,
        seed_rank=0,
        first_hop_rank=1,
        create_only=True,
        image=None,
        expected_image_id=None,
        connect_timeout=15,
    )
    return site, paths, plan


def _completed(argv: tuple[str, ...], output: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(argv, 0, output, "")


def test_topology_builds_one_direct_chain_without_management_addresses() -> None:
    hops = fanout.fabric_chain(_site(), seed_rank=0, first_hop_rank=1)
    assert [(hop.source_rank, hop.destination_rank) for hop in hops] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    assert [hop.edge for hop in hops] == ["r0-r1", "r1-r2", "r2-r3"]
    assert all(hop.destination_address.startswith("10.0.") for hop in hops)


def test_command_plan_uses_resumable_rsync_and_binds_direct_source() -> None:
    _site_config, paths, plan = _plan()
    transfers = [
        action for action in plan["actions"] if action["kind"] == "direct-rsync-if-missing"
    ]
    assert len(transfers) == 3
    first_command = " ".join(transfers[0]["command"])
    assert "--partial" in first_command
    assert "--append-verify" in first_command
    assert "10.0.1.10" in first_command
    assert "10.0.1.11" in first_command
    assert paths.partial in first_command
    assert "management-rank-1.example" not in first_command
    assert not any(action["kind"] == "load-image" for action in plan["actions"])
    assert plan["topology"] == {"link_speed_mbps": 200000, "mtu": 9000}


def test_verify_mode_reports_exact_evidence_for_every_rank() -> None:
    site, paths, _plan_document = _plan()

    def runner(argv, **_kwargs):
        return _completed(argv, f"EXACT {DIGEST} 8192\n")

    receipt = fanout.verify_cluster(
        site,
        paths,
        DIGEST,
        timeout=60,
        connect_timeout=15,
        runner=runner,
    )
    assert receipt["schema"] == fanout.VERIFY_SCHEMA
    assert receipt["status"] == "verified"
    assert [rank["rank"] for rank in receipt["ranks"]] == [0, 1, 2, 3]


def test_verify_mode_rejects_one_wrong_digest() -> None:
    site, paths, _plan_document = _plan()
    calls = 0

    def runner(argv, **_kwargs):
        nonlocal calls
        calls += 1
        digest = DIGEST if calls != 3 else "c" * 64
        state = "EXACT" if calls != 3 else "CONFLICT"
        return _completed(argv, f"{state} {digest} 8192\n")

    with pytest.raises(fanout.FanoutError, match="did not prove"):
        fanout.verify_cluster(
            site,
            paths,
            DIGEST,
            timeout=60,
            connect_timeout=15,
            runner=runner,
        )


def test_wrong_digest_probe_is_conflicting_evidence() -> None:
    probe = fanout.parse_probe(f"CONFLICT {'c' * 64} 12345\n")
    assert probe.state == "conflict"
    assert probe.sha256 == "c" * 64
    assert probe.bytes == 12345


def test_existing_exact_archive_skips_download_and_every_hop() -> None:
    site, paths, plan = _plan()
    calls = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        return _completed(argv, f"EXACT {DIGEST} 8192\n")

    receipt = fanout.execute_fanout(
        site,
        plan,
        paths,
        timeout=60,
        connect_timeout=15,
        runner=runner,
    )
    assert len(calls) == 4
    assert all(rank["source"] == "existing" for rank in receipt["ranks"])
    assert all(hop["status"] == "reused" for hop in receipt["hops"])


def test_existing_conflicting_archive_is_never_overwritten() -> None:
    site, paths, plan = _plan()

    def runner(argv, **_kwargs):
        return _completed(argv, f"CONFLICT {'d' * 64} 8192\n")

    with pytest.raises(fanout.FanoutError, match="conflicting archive"):
        fanout.execute_fanout(
            site,
            plan,
            paths,
            timeout=60,
            connect_timeout=15,
            runner=runner,
        )


def test_interrupted_hop_reports_resumable_partial() -> None:
    site, paths, plan = _plan()
    probe_count = 0

    def runner(argv, **_kwargs):
        nonlocal probe_count
        command = " ".join(argv)
        if "rsync" in command:
            raise subprocess.TimeoutExpired(argv, 60)
        if "MISSING" in command or "sha256sum" in command:
            probe_count += 1
            if probe_count == 1:
                return _completed(argv, f"EXACT {DIGEST} 8192\n")
            return _completed(argv, "MISSING\n")
        return _completed(argv)

    with pytest.raises(fanout.FanoutError, match="partial file is resumable"):
        fanout.execute_fanout(
            site,
            plan,
            paths,
            timeout=60,
            connect_timeout=15,
            runner=runner,
        )


def test_nonzero_remote_command_names_the_stopped_action() -> None:
    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 23, "", "fabric unavailable")

    with pytest.raises(
        fanout.FanoutError,
        match="hop 2 direct rsync did not complete: fabric unavailable",
    ):
        fanout._run(
            ("ssh", "rank-1", "rsync"),
            timeout=60,
            runner=runner,
            action="hop 2 direct rsync",
        )


def test_archive_commands_verify_sha_before_immutable_promotion() -> None:
    paths = fanout.archive_paths("/var/lib/sparkring/images", "runtime.tar.zst")
    download = fanout._download_script(
        paths,
        "https://images.example/runtime.tar.zst",
        DIGEST,
    )
    promote = fanout._promote_script(paths, DIGEST)

    for command in (download, promote):
        assert "sha256sum" in command
        assert f'test "$actual" = {DIGEST}' in command
        assert f"ln -- {paths.partial} {paths.final}" in command


def test_source_url_rejects_embedded_authority_and_request_state() -> None:
    for url in (
        "https://user:secret@images.example/runtime.tar.zst",
        "https://images.example/runtime.tar.zst?token=secret",
        "https://images.example/runtime.tar.zst#fragment",
    ):
        with pytest.raises(fanout.FanoutError, match="must not contain"):
            fanout.source_url(url)


def test_path_and_image_identity_inputs_are_bounded() -> None:
    with pytest.raises(fanout.FanoutError, match="too broad"):
        fanout.archive_paths("/", "runtime.tar.zst")
    with pytest.raises(fanout.FanoutError, match="safe filename"):
        fanout.archive_paths("/var/lib/images", "../runtime.tar.zst")
    site, paths, _document = _plan()
    plan = fanout.plan_document(
        site,
        url="https://images.example/runtime.tar.zst",
        expected_sha256=DIGEST,
        paths=paths,
        seed_rank=0,
        first_hop_rank=1,
        create_only=False,
        image="registry.example/runtime@sha256:" + "e" * 64,
        expected_image_id=IMAGE_ID,
        connect_timeout=15,
    )
    assert plan["expected_image_id"] == IMAGE_ID


def test_image_import_tags_an_archive_saved_by_image_id() -> None:
    paths = fanout.archive_paths("/var/lib/sparkring/images", "runtime.tar.zst")
    image = "registry.example/runtime:verified"
    script = fanout._load_script(paths, image, IMAGE_ID)

    assert f"docker image inspect {IMAGE_ID}" in script
    assert f"test \"$loaded\" = {IMAGE_ID}" in script
    assert f"docker image tag {IMAGE_ID} {image}" in script


def test_operator_documentation_link_resolves() -> None:
    document = ROOT / "docs/DIRECT_FABRIC_IMAGE_ARCHIVE_FANOUT.md"
    assert document.is_file()
    config_readme = (ROOT / "scripts/config/README.md").read_text(
        encoding="utf-8"
    )
    assert "../../docs/DIRECT_FABRIC_IMAGE_ARCHIVE_FANOUT.md" in config_readme


def test_cli_help_names_all_modes_and_confirmation_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/fanout_image_archive.py"), "--help"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    for option in (
        "--verify",
        "--execute",
        "--create-only",
        "--confirmation",
        "--timeout",
        "--connect-timeout",
        "--output",
    ):
        assert option in completed.stdout
