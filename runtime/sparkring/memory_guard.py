#!/usr/bin/env python3
"""Stop opted-in model containers before host memory exhaustion breaks control access."""

from __future__ import annotations

import argparse
import dataclasses
import math
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path


GIB = 1024**3
RunCommand = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)


def read_mem_available(path: Path = Path("/proc/meminfo")) -> int:
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                break
            return int(fields[1]) * 1024
    raise RuntimeError(f"MemAvailable is missing or malformed in {path}")


def guarded_container_ids(run: RunCommand = run_command) -> tuple[str, ...]:
    result = run(
        (
            "docker",
            "ps",
            "--quiet",
            "--filter",
            "label=org.sparkring.memory-guard=true",
        )
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker ps failed: {result.stderr.strip()}")
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def _running(container_id: str, run: RunCommand) -> bool:
    result = run(
        (
            "docker",
            "inspect",
            "--format",
            "{{.State.Running}}",
            container_id,
        )
    )
    # Keep an uncertain container in the escalation set. Only a successful
    # stopped-state response proves that KILL is unnecessary.
    return not (result.returncode == 0 and result.stdout.strip() == "false")


def terminate_guarded_containers(
    *,
    grace_seconds: float,
    run: RunCommand = run_command,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, ...]:
    container_ids = guarded_container_ids(run)
    if not container_ids:
        return ()

    result = run(("docker", "kill", "--signal=TERM", *container_ids))
    if result.returncode != 0:
        print(f"memory-guard: TERM failed: {result.stderr.strip()}", file=sys.stderr)

    deadline = time.monotonic() + grace_seconds
    remaining = container_ids
    while remaining and time.monotonic() < deadline:
        sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        remaining = tuple(item for item in remaining if _running(item, run))

    if remaining:
        result = run(("docker", "kill", "--signal=KILL", *remaining))
        if result.returncode != 0:
            print(f"memory-guard: KILL failed: {result.stderr.strip()}", file=sys.stderr)
    return container_ids


@dataclasses.dataclass
class GuardState:
    low_samples: int = 0

    def observe(
        self,
        available_bytes: int,
        *,
        floor_bytes: int,
        required_samples: int,
    ) -> bool:
        if available_bytes >= floor_bytes:
            self.low_samples = 0
            return False
        self.low_samples += 1
        if self.low_samples < required_samples:
            return False
        self.low_samples = 0
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available-floor-bytes", type=int, default=8 * GIB)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--consecutive-samples", type=int, default=2)
    parser.add_argument("--term-grace-seconds", type=float, default=5.0)
    parser.add_argument("--trip-cooldown-seconds", type=float, default=10.0)
    args = parser.parse_args()
    for option, value in (
        ("--poll-seconds", args.poll_seconds),
        ("--term-grace-seconds", args.term_grace_seconds),
        ("--trip-cooldown-seconds", args.trip_cooldown_seconds),
    ):
        if not math.isfinite(value):
            parser.error(f"{option} must be finite")
    if args.available_floor_bytes <= 0:
        parser.error("--available-floor-bytes must be positive")
    if args.poll_seconds <= 0 or args.consecutive_samples <= 0:
        parser.error("poll interval and consecutive samples must be positive")
    if args.term_grace_seconds < 0 or args.trip_cooldown_seconds < 0:
        parser.error("grace and cooldown durations cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    state = GuardState()
    print(
        "memory-guard: active "
        f"floor_bytes={args.available_floor_bytes} "
        f"samples={args.consecutive_samples}",
        flush=True,
    )
    while True:
        try:
            available = read_mem_available()
            if state.observe(
                available,
                floor_bytes=args.available_floor_bytes,
                required_samples=args.consecutive_samples,
            ):
                print(
                    f"memory-guard: tripped available_bytes={available}",
                    file=sys.stderr,
                    flush=True,
                )
                stopped = terminate_guarded_containers(
                    grace_seconds=args.term_grace_seconds
                )
                print(
                    "memory-guard: signalled containers="
                    + (",".join(stopped) if stopped else "none"),
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(args.trip_cooldown_seconds)
        except Exception as error:
            print(f"memory-guard: probe error: {error}", file=sys.stderr, flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
