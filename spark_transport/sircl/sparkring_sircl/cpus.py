"""CPU placement of SIRCL processes on hosts with performance and efficiency cores.

A GB10 Spark has ten Cortex-X925 performance cores and ten Cortex-A725
efficiency cores. Eager collectives are bound by the host thread that launches
them, and graph replays of small messages nearly so; a launching thread or a
progress thread scheduled on an efficiency core slows every collective of its
group, because the other ranks wait for it. A progress thread that shares a
core with a spinning launching thread is worse: each must wait for the other's
scheduler slice, which stalls ops for milliseconds. Placement therefore pins
the launching process to performance cores and gives the native progress
thread (``SIRCL_PROGRESS_CPU``) CPUs that the launching threads never use.

:func:`classify` finds the class of the fastest cores from the first source
that distinguishes core types: the Arm part number of every CPU in
``/proc/cpuinfo`` (Cortex-X925 ``0xd85`` above Cortex-A725 ``0xd87``), then
``cpu_capacity`` in sysfs, then ``cpuinfo_max_freq``. Capacities and
frequencies within ``TOLERANCE`` of the largest value belong to the fastest
class, because cores of one type can differ slightly (on a GB10 one X925
reports a higher capacity than its siblings). :func:`plan` chooses the CPUs;
:func:`apply` sets the calling thread's affinity, which threads it creates
later inherit. Linux only; elsewhere every function reports no preference.
"""

from __future__ import annotations

import ctypes
import dataclasses
import os
from collections.abc import Iterable
from pathlib import Path

SYSFS = Path("/sys/devices/system/cpu")
CPUINFO = Path("/proc/cpuinfo")
# Relative speed tiers of Arm Cortex cores by MIDR part number (implementer 0x41).
ARM_PART_TIERS = {
    0xD44: 3, 0xD48: 3, 0xD4E: 3, 0xD82: 3, 0xD85: 3,   # Cortex-X1, X2, X3, X4, X925
    0xD47: 2, 0xD4D: 2, 0xD81: 2, 0xD87: 2,             # Cortex-A710, A715, A720, A725
    0xD46: 1, 0xD80: 1,                                 # Cortex-A510, A520
}
# Capacities or frequencies at least (1 - TOLERANCE) times the largest are one class.
TOLERANCE = 0.15
POLICIES = ("performance", "none")


def parse_cpu_list(text: str) -> list[int]:
    """``0-3,8,10-11`` as a sorted list."""
    cpus: set[int] = set()
    for item in text.strip().split(","):
        item = item.strip()
        if not item:
            continue
        first, _, last = item.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return sorted(cpus)


def format_cpu_list(cpus: Iterable[int]) -> str:
    """The inverse of :func:`parse_cpu_list`, with ranges."""
    ordered = sorted(set(cpus))
    parts: list[str] = []
    start = previous = None
    for cpu in ordered:
        if previous is not None and cpu == previous + 1:
            previous = cpu
            continue
        if start is not None:
            parts.append(f"{start}" if start == previous else f"{start}-{previous}")
        start = previous = cpu
    if start is not None:
        parts.append(f"{start}" if start == previous else f"{start}-{previous}")
    return ",".join(parts)


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    text = _read(path)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def online_cpus(root: Path = SYSFS) -> list[int]:
    text = _read(root / "online")
    if text:
        return parse_cpu_list(text)
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []


def _part_numbers(cpuinfo: Path) -> dict[int, int]:
    parts: dict[int, int] = {}
    current = None
    text = _read(cpuinfo) or ""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "processor":
            try:
                current = int(value.strip())
            except ValueError:
                current = None
        elif key == "CPU part" and current is not None:
            try:
                parts[current] = int(value.strip(), 0)
            except ValueError:
                pass
    return parts


def classify(root: Path = SYSFS, cpuinfo: Path = CPUINFO) -> tuple[list[int], list[int], str]:
    """``(online CPUs, CPUs of the fastest class, source)``; the source is ``uniform`` when no
    source distinguishes classes (every online CPU is then in the fastest class)."""
    cpus = online_cpus(root)
    if not cpus:
        return [], [], "uniform"
    parts = _part_numbers(cpuinfo)
    if all(cpu in parts for cpu in cpus):
        tiers = {cpu: ARM_PART_TIERS.get(parts[cpu], 2) for cpu in cpus}
        if len(set(tiers.values())) > 1:
            best = max(tiers.values())
            return cpus, sorted(cpu for cpu, tier in tiers.items() if tier == best), "cpu_part"
    for source, relative in (("cpu_capacity", "cpu_capacity"), ("cpuinfo_max_freq", "cpufreq/cpuinfo_max_freq")):
        values = {cpu: _read_int(root / f"cpu{cpu}" / relative) for cpu in cpus}
        if any(value is None or value <= 0 for value in values.values()):
            continue
        top = max(values.values())
        fastest = sorted(cpu for cpu, value in values.items() if value >= top * (1.0 - TOLERANCE))
        if len(fastest) < len(cpus):
            return cpus, fastest, source
    return cpus, list(cpus), "uniform"


@dataclasses.dataclass(frozen=True)
class Placement:
    """The CPUs a process uses: its own threads on ``main``, the progress thread on ``progress``.

    ``progress`` never overlaps ``main``. It is one performance core when two or
    more are available; otherwise every allowed CPU outside ``main`` (the
    progress thread then floats there); empty only when the process may use no
    other CPU, and then the progress thread is left unpinned.
    """

    policy: str
    source: str
    performance: tuple[int, ...]
    main: tuple[int, ...]
    progress: tuple[int, ...]

    @property
    def progress_cpu_list(self) -> str | None:
        """``SIRCL_PROGRESS_CPU`` for the session, or None to leave the progress thread unpinned."""
        return format_cpu_list(self.progress) if self.progress else None

    @property
    def dedicated_progress_core(self) -> bool:
        return len(self.progress) == 1 and self.progress[0] in self.performance

    def to_json(self) -> dict[str, object]:
        return {"policy": self.policy, "source": self.source,
                "performance_cpus": format_cpu_list(self.performance),
                "main_cpus": format_cpu_list(self.main),
                "progress_cpus": self.progress_cpu_list,
                "dedicated_progress_core": self.dedicated_progress_core}


def plan(policy: str = "performance", allowed: Iterable[int] | None = None, *, root: Path = SYSFS,
         cpuinfo: Path = CPUINFO) -> Placement:
    """The fastest-class CPUs among ``allowed`` (default: this process's affinity) for the
    launching threads, and a CPU set of its own for the progress thread."""
    if policy not in POLICIES:
        raise ValueError(f"CPU policy must be one of {', '.join(POLICIES)}, got {policy!r}")
    if allowed is None:
        allowed = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else ()
    allowed = sorted(set(allowed))
    if policy == "none" or not allowed:
        return Placement(policy, "none", tuple(allowed), tuple(allowed), ())
    _, fastest, source = classify(root, cpuinfo)
    fast = set(fastest)
    performance = [cpu for cpu in allowed if cpu in fast] or allowed
    if len(performance) >= 2:
        return Placement(policy, source, tuple(performance), tuple(performance[:-1]), (performance[-1],))
    rest = tuple(cpu for cpu in allowed if cpu not in performance)
    return Placement(policy, source, tuple(performance), tuple(performance), rest)


def apply(placement: Placement) -> None:
    """Pin the calling thread (and the threads it creates later) to ``placement.main``."""
    if placement.policy != "none" and placement.main and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(placement.main))


def current_cpu() -> int:
    """The CPU the calling thread runs on, or -1 when unknown."""
    try:
        libc = ctypes.CDLL(None)
        return int(libc.sched_getcpu())
    except (OSError, AttributeError, TypeError):
        return -1
