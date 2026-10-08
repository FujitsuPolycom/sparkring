"""Measured choices for a session's collectives: the tuning table (torch-free).

A tuning table (schema ``sircl-tuning-table/v1``) belongs to one group shape and build. Its key names
the shape (``pair``, ``path:<n>``, ``cycle:<n>``, or ``strided:<kind>:<fabric size>:<offsets>`` for a
group that spans part of a larger fabric), the group's size, lane count and most relays on any lane,
the hash of the native layer's source (``native``), the hash of the kernel sources (``kernels``) and the
SIRCL version (``sircl``); ``image`` names the serving image it was measured in. It holds:

- ``measurements``: the median time of the slowest rank of every candidate the ring harness's ``tune``
  command ran, per collective (``all_reduce``, ``all_gather``, ``reduce_scatter``, ``all_to_all``),
  message size in bytes (the all-reduce's message, the all-gather's shard, the reduce-scatter's input,
  the all-to-all's input, per rank) and mode (``eager`` or ``graph``, CUDA graph replay);
- ``decisions``: per collective and mode, size intervals, each with the fastest SIRCL candidate
  (``choice``) and whether NCCL was faster there (``nccl``). An interval starts at ``from`` bytes and
  runs to the next one; the last runs on without end, and sizes below the first have no decision (the
  session's rules apply).

A candidate (:class:`Choice`) names its backend (``sircl`` or ``nccl``) and, for SIRCL, the algorithm of
an all-reduce within the capacity (``oneshot``, ``twoshot``, ``swing``), or the schedule of a larger
message, all-gather or reduce-scatter (``pieces``: two-shot pieces, tiles or scatter ops; ``chain``;
``ring``), the launch grid cap (``grid``), the link piece or chain chunk (``piece``), the ring
reduce-scatter's stagger (``stagger``, link 2) and the ring all-gather's stagger (``gather_stagger``,
link 3).

:func:`build_document` derives the decisions from the measurements: at a measured size the measured
fastest candidate; between two measured sizes, every candidate measured at both by a cost model fitted
to its own measurements, ``a + b * bytes + c * items`` (a fixed latency, a time per byte of the per-rank
volume, and a time per link item, chain chunk or two-shot piece; ``b`` and ``c`` not negative), scaled
so that it passes through the candidate's measured times. The decisions depend only on the document,
so every rank that loads the same table makes the same choice for the same collective, size and mode;
sessions record the table's hash in their setup agreement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

SCHEMA = "sircl-tuning-table/v1"
COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
MODES = ("eager", "graph")
BACKENDS = ("sircl", "nccl")
ALGORITHMS = ("oneshot", "twoshot", "swing")
SCHEDULES = ("pieces", "chain", "ring")
# Key fields a session checks against its own facts; ``image`` is recorded but cannot be checked inside a
# session.
KEY_FIELDS = ("shape", "world", "lanes", "max_relays", "native", "kernels", "sircl")
KERNEL_SOURCES = ("oneshot/_oneshot_cute.py", "oneshot/_twoshot_cute.py", "oneshot/_allgather_cute.py",
                  "oneshot/_links_cute.py", "oneshot/_chain_cute.py", "oneshot/_scatter_cute.py",
                  "oneshot/_swing_cute.py", "oneshot/_cute_intrinsics.py", "oneshot/_timed_wait.py")
# Points between two measured sizes at which the cost models compete.
INTERIOR_POINTS = 7
PACKAGE = Path(__file__).resolve().parent


class TuningError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Choice:
    """One candidate of a collective at a size."""

    backend: str = "sircl"
    algorithm: Optional[str] = None
    schedule: Optional[str] = None
    grid: Optional[int] = None
    piece: Optional[int] = None
    stagger: Optional[int] = None
    gather_stagger: Optional[int] = None

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise TuningError(f"backend {self.backend!r} is not one of {BACKENDS}")
        if self.algorithm is not None and self.algorithm not in ALGORITHMS:
            raise TuningError(f"algorithm {self.algorithm!r} is not one of {ALGORITHMS}")
        if self.schedule is not None and self.schedule not in SCHEDULES:
            raise TuningError(f"schedule {self.schedule!r} is not one of {SCHEDULES}")
        if self.algorithm is not None and self.schedule is not None:
            raise TuningError("a candidate names an algorithm or a schedule, not both")
        if self.grid is not None and (self.grid < 1 or self.grid & (self.grid - 1) or self.grid > 1024):
            raise TuningError(f"grid {self.grid} is not a power of two up to 1024")
        if self.piece is not None and (self.piece < 16 or self.piece % 16):
            raise TuningError(f"piece {self.piece} is not a positive multiple of 16 bytes")
        if self.stagger is not None and not 0 <= self.stagger <= 4:
            raise TuningError(f"stagger {self.stagger} is not 0 to 4")
        if self.gather_stagger is not None and not 0 <= self.gather_stagger <= 4:
            raise TuningError(f"all-gather stagger {self.gather_stagger} is not 0 to 4")
        if self.backend == "nccl" and any(v is not None for v in (self.algorithm, self.schedule, self.grid,
                                                                  self.piece, self.stagger, self.gather_stagger)):
            raise TuningError("an NCCL candidate names no SIRCL setting")

    def to_json(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)
                if getattr(self, field.name) is not None and (field.name != "backend" or self.backend != "sircl")}

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> "Choice":
        unknown = set(document) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise TuningError(f"unknown candidate fields {sorted(unknown)}")
        return cls(**{key: (int(value) if key in ("grid", "piece", "stagger", "gather_stagger") else str(value))
                      for key, value in document.items()})

    def label(self) -> str:
        if self.backend == "nccl":
            return "nccl"
        parts = [self.algorithm or self.schedule or "default"]
        if self.piece is not None:
            parts.append(f"piece {self.piece}")
        if self.stagger is not None:
            parts.append(f"stagger {self.stagger}")
        if self.gather_stagger is not None:
            parts.append(f"gather stagger {self.gather_stagger}")
        if self.grid is not None:
            parts.append(f"grid {self.grid}")
        return " ".join(parts)

    def order(self, collective: str) -> str:
        """The order of the sums ``collective`` gives under this choice: ``rank`` (the rank-ordered sum:
        one-shot, two-shot, two-shot pieces, scatter ops), ``chain`` or ``ring`` (each element summed along
        the chain or around the ring, deterministic and the same on every rank), ``swing``, ``nccl``
        (NCCL's own), ``rules`` (the session's schedule decides) or ``copy`` (all-gathers and all-to-alls
        move bytes without summing)."""
        if collective in ("all_gather", "all_to_all"):
            return "copy"
        if self.backend == "nccl":
            return "nccl"
        if self.schedule in ("chain", "ring"):
            return self.schedule
        if self.schedule == "pieces" or self.algorithm in ("oneshot", "twoshot"):
            return "rank"
        if self.algorithm == "swing":
            return "swing"
        return "rules"

    def items(self, collective: str, nbytes: int, world: int) -> int:
        """Link items, chain chunks or two-shot pieces one call moves (the cost model's per-item term)."""
        if self.backend == "nccl" or self.schedule is None or not self.piece:
            return 0
        if self.schedule == "ring":
            block = nbytes if collective == "all_gather" else -(-nbytes // max(1, world))
            per_link = (world - 1) * -(-block // self.piece)
            return 2 * per_link if collective == "all_reduce" else per_link
        return -(-nbytes // self.piece)


# -- keys and facts ------------------------------------------------------------------------------------


def shape_of(identity: Mapping[str, Any]) -> str:
    """The group shape of a layout identity (``routes.Layout.identity()``).

    A group of every position of its fabric is a ``cycle:<n>``, a ``pair`` or a ``path:<n>``. A group of
    ``k`` consecutive positions of a cycle of ``n`` with ``2 (k - 1) < n`` is a ``pair`` or ``path:<k>``
    as well: every shortest route between its members stays on its own arc, as on a path of ``k``.
    Other groups are ``strided:<kind>:<n>:<offsets>`` (positions relative to the first)."""
    kind = str(identity["kind"])
    size = int(identity["size"])
    positions = sorted(int(p) for p in identity["positions"])
    count = len(positions)
    if count == size:
        if kind == "cycle":
            return f"cycle:{size}"
        return "pair" if size == 2 else f"path:{size}"
    if kind == "cycle" and 2 * (count - 1) < size:
        members = set(positions)
        if any(all((start + step) % size in members for step in range(count)) for start in positions):
            return "pair" if count == 2 else f"path:{count}"
    first = positions[0]
    offsets = ",".join(str(p - first) for p in positions)
    return f"strided:{kind}:{size}:{offsets}"


def _digest(paths: Iterable[Path]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return hasher.hexdigest()[:16]


def native_hash(package: Path = PACKAGE) -> str:
    """The first 16 hex digits of the sha256 of the native layer's source."""
    return _digest([package / "oneshot" / "_roce_proxy.c"])


def kernels_hash(package: Path = PACKAGE) -> str:
    """The first 16 hex digits of the sha256 of the kernel sources (``KERNEL_SOURCES``, in that order)."""
    return _digest([package / name for name in KERNEL_SOURCES if (package / name).is_file()])


def sircl_version() -> str:
    """The package version and the native ABI."""
    from . import __version__
    from .oneshot._proxy import ABI_VERSION

    return f"{__version__}/abi{ABI_VERSION}"


def facts(identity: Mapping[str, Any], world: int, lanes: int, max_relays: int, *,
          package: Path = PACKAGE) -> dict[str, Any]:
    """The key fields of a group: its shape and sizes and this build's hashes and version."""
    return {"shape": shape_of(identity), "world": int(world), "lanes": int(lanes), "max_relays": int(max_relays),
            "native": native_hash(package), "kernels": kernels_hash(package), "sircl": sircl_version()}


def facts_for_layout(layout_text: str, lanes: int, *, package: Path = PACKAGE) -> dict[str, Any]:
    """:func:`facts` of a session over ``layout_text`` with ``lanes`` lanes per peer."""
    from . import routes

    layout = routes.Layout.parse(layout_text)
    relays = routes.derive_routes(layout, lanes).max_relays()
    return facts(layout.identity(), layout.world, lanes, relays, package=package)


# -- the table ----------------------------------------------------------------------------------------


def canonical_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def document_hash(document: Mapping[str, Any]) -> str:
    """The first 16 hex digits of the sha256 of the document's canonical JSON."""
    return hashlib.sha256(canonical_bytes(document)).hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class Interval:
    start: int
    choice: Choice
    nccl: bool


class Table:
    """A loaded, validated tuning table."""

    def __init__(self, document: Mapping[str, Any], source: str = "") -> None:
        if document.get("schema") != SCHEMA:
            raise TuningError(f"tuning table schema must be {SCHEMA}, got {document.get('schema')!r}")
        key = document.get("key")
        if not isinstance(key, Mapping) or any(field not in key for field in KEY_FIELDS):
            raise TuningError(f"tuning table key needs {', '.join(KEY_FIELDS)}")
        self.document = json.loads(canonical_bytes(document))
        self.key = dict(self.document["key"])
        self.source = source
        self.hash = document_hash(self.document)
        self._intervals: dict[tuple[str, str], list[Interval]] = {}
        for entry in self.document.get("decisions", ()):
            collective, mode = str(entry["collective"]), str(entry["mode"])
            if collective not in COLLECTIVES or mode not in MODES:
                raise TuningError(f"decisions for {collective!r} in mode {mode!r}: unknown collective or mode")
            intervals = [Interval(int(item["from"]), Choice.from_json(item["choice"]), bool(item.get("nccl")))
                         for item in entry["intervals"]]
            starts = [interval.start for interval in intervals]
            if not intervals or starts != sorted(set(starts)) or starts[0] < 1:
                raise TuningError(f"decisions for {collective} in mode {mode}: intervals must start at "
                                  "increasing positive sizes")
            if any(interval.choice.backend != "sircl" for interval in intervals):
                raise TuningError(f"decisions for {collective} in mode {mode}: an interval's choice is SIRCL's")
            if (collective, mode) in self._intervals:
                raise TuningError(f"decisions for {collective} in mode {mode} appear twice")
            self._intervals[(collective, mode)] = intervals

    @classmethod
    def load(cls, path: str | Path) -> "Table":
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise TuningError(f"cannot read the tuning table {path}: {error}") from None
        return cls(document, str(path))

    def mismatches(self, own: Mapping[str, Any]) -> list[str]:
        """Every key field whose value differs from ``own`` (:func:`facts`)."""
        return [f"{field}: table {self.key.get(field)!r}, here {own.get(field)!r}" for field in KEY_FIELDS
                if self.key.get(field) != own.get(field)]

    def _interval(self, collective: str, nbytes: int, mode: str) -> Optional[Interval]:
        intervals = self._intervals.get((collective, mode))
        if not intervals or nbytes < intervals[0].start:
            return None
        low, high = 0, len(intervals) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if intervals[middle].start <= nbytes:
                low = middle
            else:
                high = middle - 1
        return intervals[low]

    def decide(self, collective: str, nbytes: int, mode: str) -> Optional[Choice]:
        """The fastest SIRCL candidate for ``collective`` of ``nbytes`` in ``mode``, or None (no decision)."""
        interval = self._interval(collective, int(nbytes), mode)
        return None if interval is None else interval.choice

    def chosen(self) -> list[Choice]:
        """Every SIRCL choice of the decisions (a session prepares their launchers)."""
        return [interval.choice for intervals in self._intervals.values() for interval in intervals]

    def backend(self, collective: str, nbytes: int, mode: str) -> str:
        """``nccl`` where NCCL measured faster than the fastest SIRCL candidate, else ``sircl``."""
        interval = self._interval(collective, int(nbytes), mode)
        return "nccl" if interval is not None and interval.nccl else "sircl"


def table_paths(value: str) -> list[str]:
    """The table paths of a ``SIRCL_TUNING_TABLE`` value: comma-separated, blanks dropped."""
    return [item.strip() for item in value.split(",") if item.strip()]


def select_table(paths: Sequence[str | Path], own: Mapping[str, Any]) -> tuple[Optional[Table], dict[str, list[str]]]:
    """The table among ``paths`` whose key matches ``own`` (:func:`facts`), or None when none does, and the
    mismatches of every other table by path. A process holds sessions of several group shapes, so it names
    one table per shape; two different tables that both match ``own`` are refused."""
    matching: list[Table] = []
    unmatched: dict[str, list[str]] = {}
    for path in paths:
        table = Table.load(path)
        problems = table.mismatches(own)
        if problems:
            unmatched[str(path)] = problems
        else:
            matching.append(table)
    if len({table.hash for table in matching}) > 1:
        raise TuningError("several different tuning tables match this group shape and build: "
                          + ", ".join(table.source for table in matching))
    return (matching[0] if matching else None), unmatched


# -- building tables -----------------------------------------------------------------------------------


@dataclasses.dataclass
class _Candidate:
    choice: Choice
    points: dict[int, float]
    coefficients: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _solve(rows: list[list[float]], values: list[float]) -> Optional[list[float]]:
    """Least squares of ``rows @ x = values`` by the normal equations (small systems); None when singular."""
    size = len(rows[0])
    matrix = [[sum(r[i] * r[j] for r in rows) for j in range(size)] for i in range(size)]
    vector = [sum(r[i] * v for r, v in zip(rows, values)) for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) < 1e-12:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        vector[column], vector[pivot] = vector[pivot], vector[column]
        for row in range(size):
            if row != column:
                factor = matrix[row][column] / matrix[column][column]
                matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[column])]
                vector[row] -= factor * vector[column]
    return [vector[i] / matrix[i][i] for i in range(size)]


def fit(candidate: _Candidate, collective: str, world: int) -> tuple[float, float, float]:
    """``(a, b, c)`` of ``a + b * bytes + c * items``, with ``b`` and ``c`` not negative."""
    sizes = sorted(candidate.points)
    times = [candidate.points[size] for size in sizes]
    items = [candidate.choice.items(collective, size, world) for size in sizes]
    if len(sizes) == 1:
        return (times[0], 0.0, 0.0)
    terms = [("b", "c"), ("b",), ("c",), ()]
    for kept in terms:
        if "c" in kept and not any(items):
            continue
        # Bytes in MiB keep the normal equations well conditioned.
        rows = [[1.0] + ([size / 1048576.0] if "b" in kept else []) + ([float(item)] if "c" in kept else [])
                for size, item in zip(sizes, items)]
        if len(rows) < len(rows[0]):
            continue
        solution = _solve(rows, times)
        if solution is None or any(value < 0 for value in solution[1:]):
            continue
        a = solution[0]
        rest = iter(solution[1:])
        b = next(rest) / 1048576.0 if "b" in kept else 0.0
        c = next(rest) if "c" in kept else 0.0
        return (a, b, c)
    return (sum(times) / len(times), 0.0, 0.0)


def _model(candidate: _Candidate, collective: str, size: int, world: int) -> float:
    a, b, c = candidate.coefficients
    return max(a + b * size + c * candidate.choice.items(collective, size, world), 1e-6)


def _estimate(candidate: _Candidate, collective: str, size: int, world: int) -> Optional[float]:
    """The candidate's time at ``size``: measured there, else the cost model scaled through the measured
    times at the neighboring measured sizes (geometric interpolation of the ratio); None outside them."""
    if size in candidate.points:
        return candidate.points[size]
    sizes = sorted(candidate.points)
    if not sizes or size < sizes[0] or size > sizes[-1]:
        return None
    upper = next(s for s in sizes if s > size)
    lower = max(s for s in sizes if s < size)
    ratio_low = candidate.points[lower] / _model(candidate, collective, lower, world)
    ratio_high = candidate.points[upper] / _model(candidate, collective, upper, world)
    weight = (math.log(size) - math.log(lower)) / (math.log(upper) - math.log(lower))
    ratio = math.exp((1 - weight) * math.log(ratio_low) + weight * math.log(ratio_high))
    return _model(candidate, collective, size, world) * ratio


def _points(sizes: Sequence[int]) -> list[int]:
    found = set(sizes)
    for low, high in zip(sizes, sizes[1:]):
        for step in range(1, INTERIOR_POINTS + 1):
            value = math.exp(math.log(low) + (math.log(high) - math.log(low)) * step / (INTERIOR_POINTS + 1))
            point = int(round(value / 16.0)) * 16
            if low < point < high:
                found.add(point)
    return sorted(found)


def decide_intervals(collective: str, rows: Sequence[Mapping[str, Any]], world: int) -> list[dict[str, Any]]:
    """The decision intervals of one collective and mode from its measurement rows
    (``{"bytes", "choice", "p50_us"}``)."""
    candidates: dict[Choice, _Candidate] = {}
    for row in rows:
        choice = Choice.from_json(row["choice"])
        candidates.setdefault(choice, _Candidate(choice, {}))
        candidates[choice].points[int(row["bytes"])] = float(row["p50_us"])
    for candidate in candidates.values():
        candidate.coefficients = fit(candidate, collective, world)
    sizes = sorted({size for candidate in candidates.values() for size in candidate.points})
    intervals: list[dict[str, Any]] = []
    for point in _points(sizes):
        estimates = []
        for candidate in candidates.values():
            value = _estimate(candidate, collective, point, world)
            if value is not None:
                estimates.append((value, candidate.choice.label(), candidate.choice))
        sircl = [entry for entry in estimates if entry[2].backend == "sircl"]
        if not sircl:
            continue
        best_time, _, best = min(sircl, key=lambda entry: (entry[0], entry[1]))
        nccl = any(entry[2].backend == "nccl" and entry[0] < best_time for entry in estimates)
        current = {"from": point, "choice": best.to_json(), "nccl": nccl}
        if not intervals or (intervals[-1]["choice"], intervals[-1]["nccl"]) != (current["choice"], nccl):
            intervals.append(current)
    return intervals


def build_document(key: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, run_id: str = "",
                   created: str = "") -> dict[str, Any]:
    """A tuning table from measurement rows ``{"collective", "mode", "bytes", "choice", "p50_us"}``."""
    world = int(key["world"])
    measurements = sorted(({"collective": str(row["collective"]), "mode": str(row["mode"]),
                            "bytes": int(row["bytes"]), "choice": Choice.from_json(row["choice"]).to_json(),
                            "p50_us": round(float(row["p50_us"]), 3)} for row in rows),
                          key=lambda row: (row["collective"], row["mode"], row["bytes"],
                                           json.dumps(row["choice"], sort_keys=True)))
    decisions = []
    for collective in COLLECTIVES:
        for mode in MODES:
            subset = [row for row in measurements if row["collective"] == collective and row["mode"] == mode]
            if subset:
                intervals = decide_intervals(collective, subset, world)
                if intervals:
                    decisions.append({"collective": collective, "mode": mode, "intervals": intervals})
    document = {"schema": SCHEMA, "key": dict(key), "run_id": run_id, "created": created,
                "measurements": measurements, "decisions": decisions}
    Table(document)
    return document


def render(document: Mapping[str, Any]) -> str:
    """The decisions of a table as text, one line per interval."""
    key = document["key"]
    lines = [f"tuning table {document_hash(document)}: {key['shape']}, {key['world']} ranks, {key['lanes']} lanes, "
             f"{key['max_relays']} relays, native {key['native']}, kernels {key['kernels']}, {key['sircl']}"]
    for entry in document["decisions"]:
        for index, interval in enumerate(entry["intervals"]):
            following = entry["intervals"][index + 1]["from"] if index + 1 < len(entry["intervals"]) else None
            span = f"{interval['from']}-{following - 1}" if following else f"{interval['from']}+"
            choice = Choice.from_json(interval["choice"])
            lines.append(f"  {entry['collective']:<14} {entry['mode']:<5} {span:>22} bytes: {choice.label()}"
                         f"  [{choice.order(entry['collective'])} order]"
                         + ("  (NCCL faster)" if interval["nccl"] else ""))
        orders = sorted({Choice.from_json(interval["choice"]).order(entry["collective"])
                         for interval in entry["intervals"]})
        lines.append(f"  {entry['collective']:<14} {entry['mode']:<5} sums: "
                     + ("rank-ordered at every size" if orders == ["rank"] else
                        "copies, no sums" if orders == ["copy"] else
                        f"{', '.join(orders)} orders by size (rank-ordered only where marked rank)"))
    return "\n".join(lines)


__all__ = ["ALGORITHMS", "BACKENDS", "COLLECTIVES", "Choice", "KEY_FIELDS", "MODES", "SCHEDULES", "SCHEMA",
           "Table", "TuningError", "build_document", "decide_intervals", "document_hash", "facts",
           "facts_for_layout", "fit", "kernels_hash", "native_hash", "render", "select_table", "shape_of",
           "sircl_version", "table_paths"]
