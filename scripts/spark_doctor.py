#!/usr/bin/env python3
"""Read-only acceptance checks for one blank DGX Spark."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Protocol, Sequence

try:
    from .sparkring_diagnostics import CheckStatus, DiagnosticCheck, build_receipt
except ImportError:
    from sparkring_diagnostics import CheckStatus, DiagnosticCheck, build_receipt


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class Runner(Protocol):
    def run(self, argv: Sequence[str]) -> CommandResult: ...


class LocalRunner:
    def run(self, argv: Sequence[str]) -> CommandResult:
        completed = subprocess.run(
            list(argv), capture_output=True, text=True, check=False
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _command_check(
    runner: Runner,
    *,
    check_id: str,
    command: Sequence[str],
    summary: str,
    contains: str | None = None,
) -> DiagnosticCheck:
    if shutil.which(command[0]) is None and isinstance(runner, LocalRunner):
        return DiagnosticCheck(
            check_id,
            CheckStatus.FAIL,
            "host",
            command[0],
            summary,
            f"command not found: {command[0]}",
            source="spark-doctor",
        )
    result = runner.run(command)
    output = (result.stdout + "\n" + result.stderr).strip()
    passed = result.returncode == 0 and (contains is None or contains in output)
    return DiagnosticCheck(
        check_id,
        CheckStatus.PASS if passed else CheckStatus.FAIL,
        "host",
        " ".join(command),
        summary,
        output[:2000] or f"exit={result.returncode}",
        source="spark-doctor",
    )


def run_host_checks(
    runner: Runner | None = None,
    *,
    require_telemetry_disabled: bool = False,
) -> list[DiagnosticCheck]:
    runner = runner or LocalRunner()
    checks = [
        _command_check(
            runner,
            check_id="HOST.DGX_RELEASE",
            command=("sh", "-c", "test -r /etc/dgx-release && cat /etc/dgx-release"),
            summary="DGX release metadata is readable.",
        ),
        _command_check(
            runner,
            check_id="HOST.GPU",
            command=("nvidia-smi", "-L"),
            summary="The NVIDIA GPU driver inventories a GPU.",
            contains="GPU 0:",
        ),
        _command_check(
            runner,
            check_id="HOST.DOCKER",
            command=("docker", "version", "--format", "{{.Server.Version}}"),
            summary="The Docker daemon answers.",
        ),
        _command_check(
            runner,
            check_id="HOST.NVIDIA_CONTAINER_TOOLKIT",
            command=("nvidia-ctk", "--version"),
            summary="NVIDIA Container Toolkit is installed.",
        ),
        _command_check(
            runner,
            check_id="HOST.CONNECTX7_PCI",
            command=("sh", "-c", "lspci | grep -i 'Mellanox.*ConnectX-7'"),
            summary="PCI inventory contains ConnectX-7.",
        ),
        _command_check(
            runner,
            check_id="HOST.CONNECTX7_RDMA_MAP",
            command=("ibdev2netdev",),
            summary="ConnectX-7 RDMA devices map to Linux interfaces.",
            contains="rocep1s0f0",
        ),
        _command_check(
            runner,
            check_id="HOST.SYSTEMD",
            command=("sh", "-c", "test -z \"$(systemctl --failed --no-legend)\""),
            summary="No systemd unit is failed.",
        ),
        _command_check(
            runner,
            check_id="HOST.ROOT_FREE",
            command=(
                "sh",
                "-c",
                "test $(df -PB1 / | awk 'NR==2 {print $4}') -ge 21474836480",
            ),
            summary="Root filesystem has at least 20 GiB free.",
        ),
    ]
    telemetry = runner.run(
        ("systemctl", "is-enabled", "nvidia-dgx-telemetry.service")
    )
    telemetry_state = telemetry.stdout.strip() or telemetry.stderr.strip() or "unknown"
    disabled = telemetry_state in {"disabled", "masked", "not-found"}
    checks.append(
        DiagnosticCheck(
            "HOST.NVIDIA_TELEMETRY",
            (
                CheckStatus.PASS
                if disabled or not require_telemetry_disabled
                else CheckStatus.FAIL
            ),
            "host",
            "nvidia-dgx-telemetry.service",
            (
                "NVIDIA telemetry state was observed."
                if not require_telemetry_disabled
                else "NVIDIA telemetry is disabled or masked."
            ),
            f"is-enabled={telemetry_state}",
            source="spark-doctor",
        )
    )
    return checks


def render_text(checks: Sequence[DiagnosticCheck]) -> str:
    lines = ["DGX Spark host diagnosis (read-only)"]
    for check in checks:
        lines.append(
            f"[{check.status.value.upper():7}] {check.check_id}: {check.summary}"
        )
        lines.append(f"          {check.evidence.splitlines()[0]}")
    receipt = build_receipt(checks, source="spark-doctor")
    lines.append("host diagnosis: " + ("PASS" if receipt["passed"] else "FAIL"))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose one DGX Spark")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-telemetry-disabled", action="store_true")
    args = parser.parse_args(argv)
    checks = run_host_checks(
        require_telemetry_disabled=args.require_telemetry_disabled
    )
    receipt = build_receipt(checks, source="spark-doctor")
    print(json.dumps(receipt, indent=2, sort_keys=True) if args.json else render_text(checks))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
