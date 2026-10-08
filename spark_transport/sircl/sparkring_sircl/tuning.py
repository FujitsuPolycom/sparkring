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
  session's rules apply). An all-reduce decision whose choice names an algorithm (one-shot, two-shot,
  Swing) holds only up to the largest message it was measured at (``until``): those algorithms run a
  message in one op within the session's capacity, so a larger message has no decision there;
- ``settings``: the session variables of :data:`SETTINGS` that the chosen candidates ran under and need,
  as :func:`table_settings` derives them from the tune session. A session that takes the table applies
  each one its environment leaves unset, so every choice runs as it was measured.

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
# Session variables a table records (``settings``) and a session that takes it applies where its environment
# leaves them unset: the link slots, whose count holds the chosen staggers and keeps as many link items in
# flight as the measurements had; the link slot, which holds the chosen link pieces; the chain slot, which
# holds the chosen chain chunks; and the large-message piece of chosen two-shot pieces.
SETTINGS = ("SIRCL_LINK_SLOTS", "SIRCL_LINK_SLOT_BYTES", "SIRCL_CHAIN_SLOT_BYTES", "SIRCL_LARGE_PIECE_BYTES")
# The settings below whose table value a chosen candidate cannot run (a piece above the slot, a stagger the
# slots cannot hold); a launcher refuses a smaller value it would set itself.
MINIMUM_SETTINGS = ("SIRCL_LINK_SLOTS", "SIRCL_LINK_SLOT_BYTES", "SIRCL_CHAIN_SLOT_BYTES")
# The fields of a session's stats() that hold those settings' values.
SETTING_STATS = {"SIRCL_LINK_SLOTS": "link_slots", "SIRCL_LINK_SLOT_BYTES": "link_slot_bytes",
                 "SIRCL_CHAIN_SLOT_BYTES": "chain_slot_bytes", "SIRCL_LARGE_PIECE_BYTES": "large_piece_bytes"}
# A recorded slot is a multiple of this and at least the session's default slot (oneshot.runtime's
# DEFAULT_LINK_SLOT_BYTES and DEFAULT_CHAIN_SLOT_BYTES), so a table never shrinks a slot below its default.
SLOT_ALIGNMENT = 4096
DEFAULT_LINK_SLOT_BYTES = 512 << 10
DEFAULT_CHAIN_SLOT_BYTES = 1 << 20
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
        settings = self.document.get("settings", {})
        if (not isinstance(settings, Mapping) or any(name not in SETTINGS for name in settings)
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 1
                       for value in settings.values())):
            raise TuningError(f"tuning table settings are positive integers of {', '.join(SETTINGS)}")
        self.settings: dict[str, int] = dict(settings)
        self._intervals: dict[tuple[str, str], list[Interval]] = {}
        self._until: dict[tuple[str, str], int] = {}
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
            if "until" in entry:
                until = entry["until"]
                if not isinstance(until, int) or isinstance(until, bool) or until < starts[-1]:
                    raise TuningError(f"decisions for {collective} in mode {mode}: until must be a size from the "
                                      "last interval's start on")
                self._until[(collective, mode)] = until
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
        if not intervals or nbytes < intervals[0].start or nbytes > self._until.get((collective, mode), nbytes):
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

    def decided(self) -> list[tuple[str, str, Choice]]:
        """``(collective, mode, choice)`` of every interval of the decisions."""
        return [(collective, mode, interval.choice) for (collective, mode), intervals in self._intervals.items()
                for interval in intervals]

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
    (``{"bytes", "choice", "p50_us"}``). Between measured sizes a size no SIRCL candidate's measurements
    span keeps the previous interval's choice."""
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


def single_op_limit(collective: str, rows: Sequence[Mapping[str, Any]], intervals: Sequence[Mapping[str, Any]]
                    ) -> Optional[int]:
    """``until`` of an all-reduce's decisions whose last interval names an algorithm (one-shot, two-shot,
    Swing): the largest message that algorithm was measured at. Those algorithms run a message in one op
    within the session's capacity; above it only a schedule (two-shot pieces, chain, ring) decides, and a
    table without one there leaves larger messages to the session's rules. None otherwise."""
    if collective != "all_reduce" or not intervals:
        return None
    last = Choice.from_json(intervals[-1]["choice"])
    if last.algorithm is None:
        return None
    measured = [int(row["bytes"]) for row in rows if Choice.from_json(row["choice"]).algorithm is not None
                and Choice.from_json(row["choice"]).backend == "sircl"]
    return max(measured) if measured else None


def settings_conflicts(settings: Mapping[str, int], given: Mapping[str, str]) -> list[str]:
    """``NAME=value`` of every :data:`MINIMUM_SETTINGS` entry that ``given`` (variable -> text) sets below
    the table's ``settings``: a session given it cannot run the table's choices that need more."""
    conflicts = []
    for name in MINIMUM_SETTINGS:
        text = str(given.get(name, "")).strip()
        if name in settings and text.isdigit() and int(text) < settings[name]:
            conflicts.append(f"{name}={text} (the table's {settings[name]})")
    return conflicts


def table_settings(decisions: Sequence[Mapping[str, Any]], session: Mapping[str, Any]) -> dict[str, int]:
    """The :data:`SETTINGS` a table records for ``decisions`` measured in a session whose ``stats()`` fields
    are ``session`` (``link_slots``, ``link_slot_bytes``, ``chain_slot_bytes``, ``large_piece_bytes``).

    With a link schedule among the choices (ring, or a chain all-gather or reduce-scatter): the tune
    session's link slot count, and a link slot holding the largest chosen link piece (rounded up to
    :data:`SLOT_ALIGNMENT`, at least the default slot). With a chain all-reduce: a
    chain slot holding the largest chosen chain chunk (at least the default). With two-shot pieces of an
    all-reduce: the tune session's large-message piece. A setting the session did not report is left out."""
    link_pieces: list[int] = []
    chain_pieces: list[int] = []
    links = chain = pieces = False
    for entry in decisions:
        collective = str(entry["collective"])
        for item in entry["intervals"]:
            choice = Choice.from_json(item["choice"])
            if choice.schedule == "ring" or (choice.schedule == "chain" and collective != "all_reduce"):
                links = True
                link_pieces += [choice.piece] if choice.piece else []
            elif choice.schedule == "chain":
                chain = True
                chain_pieces += [choice.piece] if choice.piece else []
            elif choice.schedule == "pieces" and collective == "all_reduce":
                pieces = True

    def slot(largest: int, default: int) -> int:
        return max(default, -(-largest // SLOT_ALIGNMENT) * SLOT_ALIGNMENT)

    found: dict[str, int] = {}
    if links and session.get("link_slots"):
        found["SIRCL_LINK_SLOTS"] = int(session["link_slots"])
    if links and session.get("link_slot_bytes"):
        found["SIRCL_LINK_SLOT_BYTES"] = min(int(session["link_slot_bytes"]),
                                             slot(max(link_pieces, default=0), DEFAULT_LINK_SLOT_BYTES))
    if chain and session.get("chain_slot_bytes"):
        found["SIRCL_CHAIN_SLOT_BYTES"] = min(int(session["chain_slot_bytes"]),
                                              slot(max(chain_pieces, default=0), DEFAULT_CHAIN_SLOT_BYTES))
    if pieces and session.get("large_piece_bytes"):
        found["SIRCL_LARGE_PIECE_BYTES"] = int(session["large_piece_bytes"])
    return found


def build_document(key: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, run_id: str = "",
                   created: str = "", session: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """A tuning table from measurement rows ``{"collective", "mode", "bytes", "choice", "p50_us"}`` taken in a
    session whose ``stats()`` fields are ``session`` (the table's ``settings``, :func:`table_settings`;
    none without it)."""
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
                    entry: dict[str, Any] = {"collective": collective, "mode": mode, "intervals": intervals}
                    until = single_op_limit(collective, subset, intervals)
                    if until is not None:
                        entry["until"] = until
                    decisions.append(entry)
    document = {"schema": SCHEMA, "key": dict(key), "run_id": run_id, "created": created,
                "measurements": measurements, "decisions": decisions}
    settings = table_settings(decisions, session or {})
    if settings:
        document["settings"] = settings
    Table(document)
    return document


def render(document: Mapping[str, Any]) -> str:
    """The decisions of a table as text, one line per interval."""
    key = document["key"]
    lines = [f"tuning table {document_hash(document)}: {key['shape']}, {key['world']} ranks, {key['lanes']} lanes, "
             f"{key['max_relays']} relays, native {key['native']}, kernels {key['kernels']}, {key['sircl']}"]
    settings = document.get("settings") or {}
    if settings:
        lines.append("  settings, applied by a session that takes the table where its environment leaves them "
                     "unset: " + ", ".join(f"{name}={value}" for name, value in settings.items()))
    for entry in document["decisions"]:
        for index, interval in enumerate(entry["intervals"]):
            following = entry["intervals"][index + 1]["from"] if index + 1 < len(entry["intervals"]) else None
            span = (f"{interval['from']}-{following - 1}" if following else
                    f"{interval['from']}-{entry['until']}" if "until" in entry else f"{interval['from']}+")
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
           "MINIMUM_SETTINGS", "SETTINGS", "SETTING_STATS", "Table", "TuningError", "build_document",
           "decide_intervals", "document_hash", "facts", "facts_for_layout", "fit", "kernels_hash", "native_hash",
           "render", "select_table", "settings_conflicts", "shape_of", "single_op_limit", "sircl_version",
           "table_paths", "table_settings"]
