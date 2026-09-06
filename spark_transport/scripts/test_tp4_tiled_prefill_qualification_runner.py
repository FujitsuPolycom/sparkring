"""Offline safety and matrix tests for the tiled-prefill PowerShell runners."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPTS = Path(__file__).resolve().parent
PROBE_RUNNER = SCRIPTS / "run_tp4_tiled_prefill_probe.ps1"
MATRIX_RUNNER = SCRIPTS / "run_tp4_tiled_prefill_qualification.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def invoke(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    if POWERSHELL is None:
        pytest.skip("PowerShell is unavailable")
    environment = os.environ.copy()
    environment.pop("SPARKRING_TARGETS", None)
    environment.pop("SPARKRING_RANK_HOSTS", None)
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *arguments,
        ],
        check=False,
        capture_output=True,
        encoding="utf-8",
        env=environment,
        timeout=30,
    )


def invoke_with_topology(
    script: Path,
    *,
    targets: str,
    rank_hosts: str,
    arguments: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    if POWERSHELL is None:
        pytest.skip("PowerShell is unavailable")
    environment = os.environ.copy()
    environment["SPARKRING_TARGETS"] = targets
    environment["SPARKRING_RANK_HOSTS"] = rank_hosts
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *arguments,
        ],
        check=False,
        capture_output=True,
        encoding="utf-8",
        env=environment,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("arm_id", "query_rows", "timing_mode", "expected_exit"),
    [
        ("q40_isolated", 40, "isolated", 0),
        ("q512_steady", 512, "steady", 0),
        ("q1024_isolated", 1024, "isolated", 0),
        ("q4096_steady", 4096, "steady", 0),
        ("q513_boundary_correctness", 513, "correctness", 0),
        ("q1025_boundary_correctness", 1025, "correctness", 0),
        ("q4096_backpressure_edge0", 4096, "steady", 0),
        ("q4096_backpressure_edge1", 4096, "steady", 0),
        ("q40_poison_unexpected_generation", 40, "poison", 42),
        ("q512_poison_credit_regression", 512, "poison", 42),
    ],
)
def test_probe_runner_defaults_to_offline_plan_without_remote_inputs(
    arm_id: str,
    query_rows: int,
    timing_mode: str,
    expected_exit: int,
) -> None:
    result = invoke(PROBE_RUNNER, "-ArmId", arm_id)

    assert result.returncode == 0, result.stderr
    assert f"tiled_prefill_arm={arm_id} action=plan" in result.stdout
    assert f"q={query_rows} " in result.stdout
    assert f"timing_mode={timing_mode} " in result.stdout
    assert f"expected_exit={expected_exit} " in result.stdout
    assert "remote_contact=false" in result.stdout
    assert "serving_integration=false" in result.stdout
    assert "protocol_nodes_status=standalone-native-executor-bound" in result.stdout
    assert "registered_tile_storage_bytes_per_edge=8389632" in result.stdout


def test_q40_plan_attests_partial_active_tail() -> None:
    result = invoke(PROBE_RUNNER, "-ArmId", "q40_isolated")

    assert result.returncode == 0, result.stderr
    assert "tiles=1" in result.stdout
    assert "tail_active_bytes=491520" in result.stdout


def test_q513_plan_records_rank_order_endpoint_tuples() -> None:
    result = invoke_with_topology(
        PROBE_RUNNER,
        targets=(
            "operator@192.0.2.10,operator@192.0.2.11,"
            "operator@192.0.2.12,operator@192.0.2.13"
        ),
        rank_hosts=(
            "192.0.2.10,192.0.2.11,"
            "192.0.2.12,192.0.2.13"
        ),
        arguments=("-ArmId", "q513_boundary_correctness"),
    )

    assert result.returncode == 0, result.stderr
    expected = (
        "rank=0 target=operator@192.0.2.10 local=192.0.2.10 "
        "edge0_peer=192.0.2.11 edge0_role=client edge0_port=12500 "
        "edge0_device=rocep1s0f0 edge1_peer=192.0.2.13 "
        "edge1_role=client edge1_port=12501 edge1_device=rocep1s0f1"
    )
    assert expected in result.stdout
    assert "rank=1" in result.stdout and "edge0_role=server" in result.stdout
    assert "rank=2" in result.stdout and "edge1_role=server" in result.stdout
    assert "rank=3" in result.stdout and "edge1_role=server" in result.stdout
    assert "remote_contact=false" in result.stdout


def test_plan_rejects_rank_host_order_that_disagrees_with_ssh_targets() -> None:
    result = invoke_with_topology(
        PROBE_RUNNER,
        targets=(
            "operator@192.0.2.10,operator@192.0.2.11,"
            "operator@192.0.2.12,operator@192.0.2.13"
        ),
        rank_hosts=(
            "192.0.2.11,192.0.2.10,"
            "192.0.2.13,192.0.2.12"
        ),
        arguments=("-ArmId", "q513_boundary_correctness"),
    )

    assert result.returncode != 0
    assert "rank-order host address" in (result.stdout + result.stderr)


def test_execute_requires_four_rank_inputs_before_remote_contact() -> None:
    result = invoke(
        PROBE_RUNNER,
        "-ArmId",
        "q40_isolated",
        "-Execute",
        "-Image",
        "offline-test-image",
    )

    assert result.returncode != 0
    assert "SPARKRING_TARGETS" in (result.stdout + result.stderr)
    assert "ssh" not in result.stdout.lower()


def test_invalid_arm_is_rejected_by_parameter_binding() -> None:
    result = invoke(PROBE_RUNNER, "-ArmId", "q513_steady")

    assert result.returncode != 0
    assert "q4096_steady" in result.stderr
    assert "q513_steady" in result.stderr


@pytest.mark.parametrize(
    ("suite", "expected_count"),
    [
        ("all", 18),
        ("timing", 8),
        ("glm", 4),
        ("boundary", 2),
        ("backpressure", 2),
        ("poison", 2),
    ],
)
def test_matrix_suites_are_bounded_offline_plans(
    suite: str, expected_count: int
) -> None:
    result = invoke(MATRIX_RUNNER, "-Suite", suite)

    assert result.returncode == 0, result.stderr
    arms = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("tiled_prefill_matrix_arm=")
    ]
    assert len(arms) == expected_count
    assert all("action=plan" in line for line in arms)
    assert f"suite={suite} arms={expected_count}" in result.stdout
    assert "remote_contact=false" in result.stdout
    assert "serving_integration=false" in result.stdout


def test_all_matrix_contains_every_required_shape_and_stress_arm() -> None:
    result = invoke(MATRIX_RUNNER)

    assert result.returncode == 0, result.stderr
    for query_rows in (40, 512, 1024, 4096):
        assert f"q{query_rows}_isolated" in result.stdout
        assert f"q{query_rows}_steady" in result.stdout
    assert "q4096_backpressure_edge0" in result.stdout
    assert "q4096_backpressure_edge1" in result.stdout
    assert "q40_poison_unexpected_generation" in result.stdout
    assert "q512_poison_credit_regression" in result.stdout
    assert "q513_boundary_correctness" in result.stdout
    assert "q1025_boundary_correctness" in result.stdout


def test_execute_path_is_receipt_gated_and_four_rank_symmetric() -> None:
    source = PROBE_RUNNER.read_text(encoding="utf-8")

    assert "if (-not $Execute)" in source
    assert source.index("if (-not $Execute)") < source.index(
        "function Invoke-NodeSsh"
    )
    assert "UTF8Encoding($false)" in source
    assert "identical_sha256=true" in source
    assert "rankReceipts.Count -ne 1" in source
    assert '"validate",' in source
    assert '"--arm-id",' in source
    assert '"--receipts",' in source
    assert '$expectedState = "exited:$($arm.expected_exit_code)"' in source
    for rank in range(4):
        assert f"Rank = {rank}" in source
    assert "$expectedRegisteredBytesPerEdge = 8389632L" in source
    assert '"registered_tile_storage_bytes_per_edge="' in source
    assert "sparkring_exl3_launcher" not in source
    assert "acceptance_gate" not in source
