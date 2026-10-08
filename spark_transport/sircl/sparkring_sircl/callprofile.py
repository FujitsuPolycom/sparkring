"""Host timestamps of a session's eager calls (``SIRCL_CALL_PROFILE``), torch-free.

A session with ``SIRCL_CALL_PROFILE=<calls>`` keeps the newest ``<calls>`` eager calls of ``all_reduce``,
``all_gather``, ``all_reduce_large`` and ``all_gather_large``. Each call records ``time.perf_counter_ns()``
at the marks its method passes: ``entry`` (the call began), ``checked`` (health, eligibility and output
checks done), ``dispatched`` (algorithm chosen, launcher found, output allocated), ``launch_begin`` and
``launch_end`` (around the compiled kernel launcher's call, after the stream ordering), ``launched`` (the
launch helper returned: event record, copies) and ``exit`` (the final health check done). A CUDA graph
capture records nothing. :meth:`CallProfile.summary` gives, per operation, its path (the algorithm or
schedule) and size, the medians in microseconds of the stages between consecutive marks:

- ``checks``: entry to checked; ``dispatch``: checked to dispatched; ``order``: dispatched to
  launch_begin; ``launch``: the launcher's call; ``after``: launch_end to launched; ``finish``: launched to
  exit; ``total``: entry to exit;
- ``gap``: the previous profiled call's exit to this call's entry (the caller's own time between calls);
- ``gpu``: with ``SIRCL_CALL_PROFILE_GPU=1``, the device time between CUDA events recorded just before and
  after the launch (the kernel, including its waits for the peers).

``SIRCL_CALL_PROFILE_FILE=<prefix>`` makes the session write the summary to ``<prefix>.rank<rank>.json``
each time it has recorded ``<calls>`` more calls and when it closes.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

MARKS = ("entry", "checked", "dispatched", "launch_begin", "launch_end", "launched", "exit")
STAGES = (("checks", "entry", "checked"), ("dispatch", "checked", "dispatched"),
          ("order", "dispatched", "launch_begin"), ("launch", "launch_begin", "launch_end"),
          ("after", "launch_end", "launched"), ("finish", "launched", "exit"), ("total", "entry", "exit"))


class Call:
    """One profiled call: its operation and the time of each mark passed."""

    __slots__ = ("op", "marks", "detail", "nbytes", "gap_ns", "events")

    def __init__(self, op: str, entry_ns: int) -> None:
        self.op = op
        self.marks = {"entry": entry_ns}
        self.detail = ""
        self.nbytes = 0
        self.gap_ns: Optional[int] = None
        self.events: Optional[tuple[Any, Any]] = None

    def mark(self, name: str) -> None:
        self.marks[name] = time.perf_counter_ns()


class CallProfile:
    """The newest ``keep`` profiled calls of one session."""

    def __init__(self, keep: int, *, path: str = "", rank: int = 0, gpu: bool = False) -> None:
        if keep <= 0:
            raise ValueError("a call profile keeps at least one call")
        self.keep = int(keep)
        self.path = path
        self.rank = int(rank)
        self.gpu = bool(gpu)
        self.calls: deque[Call] = deque(maxlen=self.keep)
        self.recorded = 0
        self._last_exit: Optional[int] = None

    def begin(self, op: str) -> Call:
        return Call(op, time.perf_counter_ns())

    def end(self, call: Call, detail: str, nbytes: int) -> None:
        """Finish ``call`` (``detail``: its algorithm or schedule) and keep it."""
        call.marks["exit"] = time.perf_counter_ns()
        call.detail = str(detail)
        call.nbytes = int(nbytes)
        if self._last_exit is not None:
            call.gap_ns = call.marks["entry"] - self._last_exit
        self._last_exit = call.marks["exit"]
        self.calls.append(call)
        self.recorded += 1
        if self.path and self.recorded % self.keep == 0:
            self.dump()

    def summary(self, elapsed: Optional[Callable[[Any, Any], float]] = None) -> dict[str, Any]:
        """Medians in microseconds per operation, path and size (``elapsed(start, end)``: milliseconds
        between two CUDA events, for the ``gpu`` stage)."""
        groups: dict[tuple[str, str, int], list[Call]] = {}
        for call in self.calls:
            groups.setdefault((call.op, call.detail, call.nbytes), []).append(call)
        rows = []
        for (op, detail, nbytes), calls in sorted(groups.items()):
            row: dict[str, Any] = {"op": op, "path": detail, "bytes": nbytes, "calls": len(calls)}
            for stage, first, last in STAGES:
                values = [call.marks[last] - call.marks[first] for call in calls
                          if first in call.marks and last in call.marks]
                if values:
                    row[stage] = round(statistics.median(values) / 1e3, 2)
            gaps = [call.gap_ns for call in calls if call.gap_ns is not None]
            if gaps:
                row["gap"] = round(statistics.median(gaps) / 1e3, 2)
            if elapsed is not None:
                gpu = [elapsed(*call.events) for call in calls if call.events is not None]
                if gpu:
                    row["gpu"] = round(statistics.median(gpu) * 1e3, 2)
            rows.append(row)
        return {"rank": self.rank, "kept": len(self.calls), "recorded": self.recorded, "rows": rows}

    def reset(self) -> None:
        self.calls.clear()
        self._last_exit = None

    def dump(self, elapsed: Optional[Callable[[Any, Any], float]] = None) -> Optional[Path]:
        """Write the summary to ``<path>.rank<rank>.json``; None without a path."""
        if not self.path:
            return None
        target = Path(f"{self.path}.rank{self.rank}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.summary(elapsed), indent=1) + "\n", encoding="utf-8")
        return target


def render(summary: dict[str, Any]) -> list[str]:
    """One line per row of a summary: the stages in microseconds."""
    lines = []
    for row in summary.get("rows", ()):
        stages = ", ".join(f"{name} {row[name]:g}" for name, _, _ in STAGES if name in row)
        extra = "".join(f", {name} {row[name]:g}" for name in ("gap", "gpu") if name in row)
        lines.append(f"{row['op']} {row['bytes']} bytes ({row['path']}, {row['calls']} calls): {stages}{extra} us")
    return lines


__all__ = ["Call", "CallProfile", "MARKS", "STAGES", "render"]
