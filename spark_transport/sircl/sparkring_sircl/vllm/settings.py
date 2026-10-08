"""Environment settings of SIRCL's vLLM adapter.

The adapter reads the variables below in every vLLM process. The ring session
itself reads its own ``SIRCL_*`` variables (listed in ``sparkring_sircl/env.py``:
``SIRCL_PEER_ROUTES``, ``SIRCL_ALLREDUCE_CAPACITY_BYTES``,
``SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES``, ``SIRCL_ALLGATHER_MAX_BYTES`` and the
rest); this module declares only what the adapter adds on top.

Parsing is strict. A malformed value raises :class:`SettingError` naming the
variable, because ranks that parse a value differently would route the same
collective to different transports.

The physical fabric has no default. A four-rank instance on four Sparks of an
eight-Spark ring and a four-rank instance on a four-Spark ring differ only in
cabling, and only the second may run NCCL between its first and last rank, so
the adapter refuses to guess (:func:`fabric_text`).

``SIRCL_FABRIC`` describes the physical cabling of the whole instance. It is
distinct from the ring session's ``SIRCL_LAYOUT`` (one session's fabric and rank
positions, read by the session package when no layout is passed): the adapter
derives every session's layout from ``SIRCL_FABRIC`` and passes it explicitly,
because one vLLM process holds sessions of different groups.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping

MODES = ("custom", "disabled")
# never (the default): SIRCL carries every collective of every multi-rank group. auto, the opt-in: NCCL may
# carry a collective where the group's cabling allows it, as the adapter's rules decide. topology: another
# name for auto. In every mode a tuning table chooses only among SIRCL's settings.
NCCL_MODES = ("never", "auto", "topology")
LARGE_ALLREDUCE_MODES = ("auto", "sircl", "nccl")
DEFAULT_GROUPS = ("tp", "dcp")
DEFAULT_SESSION_MODULES = ("sparkring_sircl.oneshot",)
DEFAULT_P2P_GROUPS = ("pp", "tp", "dcp")
DEFAULT_P2P_MODULE = "sparkring_sircl.p2p"


# The ring session's large-message schedules and the geometry of its chain and link collectives
# (oneshot/runtime.py, RoceOneshotAllReduce._configure, which reads them at construction). They apply to
# the tensor-parallel session only: SIRCL's adapter builds every decode-context-parallel session with
# these variables removed (dcp_collectives.py), so a value chosen for the whole ring never reaches a
# session over part of it. A DCP group's ranks cover part of the tensor-parallel group's fabric, and the
# runtime runs chain and ring ops only where every Spark of a session's fabric hosts one of its ranks, so
# a DCP session keeps the session defaults.
TP_SESSION_VARIABLES = (
    "SIRCL_LARGE_SCHEDULE", "SIRCL_GATHER_SCHEDULE", "SIRCL_SCATTER_SCHEDULE",
    "SIRCL_CHAIN_MIN_BYTES", "SIRCL_RING_MIN_BYTES",
    "SIRCL_CHAIN_SLOTS", "SIRCL_CHAIN_SLOT_BYTES", "SIRCL_CHAIN_BLOCKS", "SIRCL_CHAIN_UNROLL",
    "SIRCL_CHAIN_CHUNK_BYTES",
    "SIRCL_LINK_SLOTS", "SIRCL_LINK_SLOT_BYTES", "SIRCL_LINK_BLOCKS", "SIRCL_LINK_UNROLL",
    "SIRCL_LINK_CHUNK_BYTES", "SIRCL_GATHER_LINK_CHUNK_BYTES", "SIRCL_SCATTER_LINK_CHUNK_BYTES",
    "SIRCL_REDUCE_LINK_CHUNK_BYTES", "SIRCL_RING_STAGGER", "SIRCL_RING_GATHER_STAGGER",
)


class SettingError(ValueError):
    """An adapter environment variable holds a value the adapter cannot accept."""


@dataclasses.dataclass(frozen=True)
class Variable:
    name: str
    meaning: str
    default: str


VARIABLES: tuple[Variable, ...] = (
    Variable("SIRCL_MODE", "custom enables the adapter in every vLLM process; disabled leaves "
             "vLLM unchanged", "disabled"),
    Variable("SIRCL_FABRIC", "physical fabric of the Sparks this instance runs on: ring:N, "
             "path:N, pair or pair:2 (two cables)", "none (required with SIRCL_MODE=custom)"),
    Variable("SIRCL_RANK_POSITIONS", "fabric position of every global rank of this instance, "
             "comma-separated in rank order", "0,1,...,world-1"),
    Variable("SIRCL_GROUPS", "vLLM group kinds that get a SIRCL session (tp, dcp)",
             ",".join(DEFAULT_GROUPS)),
    Variable("SIRCL_NCCL", "never forbids NCCL collectives on every multi-rank group; auto lets NCCL "
             "carry collectives only where the group's cabling allows it, as the adapter's rules decide "
             "(topology is another name for auto); a tuning table never sends a call to NCCL", "never"),
    Variable("SIRCL_LARGE_ALLREDUCE", "all-reduces above the session's dispatch ceiling: auto "
             "(NCCL where the cabling allows it and no graph is captured, else chunked on "
             "SIRCL), sircl (always chunked on SIRCL) or nccl (NCCL; refused where the cabling "
             "forbids it)", "auto"),
    Variable("SIRCL_RELAY_PER_PEER_BYTES", "per-peer bytes of one gather or scatter op on a "
             "group with relayed lanes; overrides the relay-load rule (75 % of a 512 KiB hairpin "
             "queue over the busiest relay's load)", "measured value or the relay-load rule"),
    Variable("SIRCL_RECEIPT_DIR", "directory for one JSON receipt per rank and group", "unset"),
    Variable("SIRCL_SESSION_MODULE", "module that provides the ring session class AllReduce, "
             "API_VERSION and is_supported", ",".join(DEFAULT_SESSION_MODULES)),
    Variable("SIRCL_P2P_GROUPS", "vLLM group kinds that get SIRCL point-to-point channels: pp (pipeline-parallel "
             "groups), and tp and dcp groups that have a SIRCL session; none disables them",
             ",".join(DEFAULT_P2P_GROUPS)),
    Variable("SIRCL_P2P_MODULE", "module that provides the point-to-point channels class PointToPoint, "
             "API_VERSION and is_supported", DEFAULT_P2P_MODULE),
    Variable("SIRCL_VLLM_SHIMS", "version-pinned vLLM shims to install: dcp_all_to_all, "
             "fused_allreduce_rms_norm", "none"),
    Variable("SIRCL_FUSED_NORM", "1 runs vLLM's post-all-reduce RMSNorm helper as one fused SIRCL "
             "kernel on tensor-parallel groups with a session, where the result is bit-identical to "
             "SIRCL's all-reduce followed by vLLM's vllm_c RMSNorm (research-only; see norm_fusion.py); "
             "0 keeps vLLM's all-reduce and RMSNorm", "0"),
)


def _environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _raw(name: str, environ: Mapping[str, str] | None) -> str | None:
    value = _environ(environ).get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _choice(name: str, choices: tuple[str, ...], default: str,
            environ: Mapping[str, str] | None) -> str:
    raw = _raw(name, environ)
    if raw is None:
        return default
    value = raw.lower()
    if value not in choices:
        raise SettingError(f"{name} must be one of {', '.join(choices)}, got {raw!r}")
    return value


def mode(environ: Mapping[str, str] | None = None) -> str:
    return _choice("SIRCL_MODE", MODES, "disabled", environ)


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "custom"


def nccl_mode(environ: Mapping[str, str] | None = None) -> str:
    """``SIRCL_NCCL``: never (the default) or auto (``topology`` is read as auto)."""
    mode = _choice("SIRCL_NCCL", NCCL_MODES, "never", environ)
    return "auto" if mode == "topology" else mode


def large_allreduce(environ: Mapping[str, str] | None = None) -> str:
    return _choice("SIRCL_LARGE_ALLREDUCE", LARGE_ALLREDUCE_MODES, "auto", environ)


def fabric_text(environ: Mapping[str, str] | None = None) -> str:
    raw = _raw("SIRCL_FABRIC", environ)
    if raw is None:
        raise SettingError(
            "SIRCL_FABRIC is not set; name the physical fabric (for example ring:8 for four "
            "Sparks of an eight-Spark ring, ring:4 for a four-Spark ring). The adapter does not "
            "guess, because only a closed ring lets NCCL connect a group's first and last rank"
        )
    return raw


def rank_positions(world: int, environ: Mapping[str, str] | None = None) -> tuple[int, ...]:
    raw = _raw("SIRCL_RANK_POSITIONS", environ)
    if raw is None:
        return tuple(range(world))
    try:
        positions = tuple(int(item) for item in raw.split(",") if item.strip())
    except ValueError:
        raise SettingError(f"SIRCL_RANK_POSITIONS must list integers, got {raw!r}") from None
    if len(positions) != world:
        raise SettingError(
            f"SIRCL_RANK_POSITIONS must name {world} positions, one per global rank; got "
            f"{len(positions)}"
        )
    if len(set(positions)) != len(positions) or min(positions) < 0:
        raise SettingError(f"SIRCL_RANK_POSITIONS must name distinct non-negative positions, "
                           f"got {raw!r}")
    return positions


def groups(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    raw = _raw("SIRCL_GROUPS", environ)
    if raw is None:
        return DEFAULT_GROUPS
    items = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    if not items or len(set(items)) != len(items):
        raise SettingError(f"SIRCL_GROUPS must list distinct group kinds, got {raw!r}")
    return items


def relay_per_peer_bytes(environ: Mapping[str, str] | None = None) -> int | None:
    raw = _raw("SIRCL_RELAY_PER_PEER_BYTES", environ)
    if raw is None:
        return None
    try:
        value = int(raw, 0)
    except ValueError:
        raise SettingError(f"SIRCL_RELAY_PER_PEER_BYTES must be an integer, got {raw!r}") from None
    if value < 16 or value % 16:
        raise SettingError("SIRCL_RELAY_PER_PEER_BYTES must be a positive multiple of 16")
    return value


def p2p_groups(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Group kinds that get point-to-point channels (``SIRCL_P2P_GROUPS``; ``none`` for none)."""
    raw = _raw("SIRCL_P2P_GROUPS", environ)
    if raw is None:
        return DEFAULT_P2P_GROUPS
    items = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    if items == ("none",):
        return ()
    unknown = [item for item in items if item not in DEFAULT_P2P_GROUPS]
    if not items or unknown or len(set(items)) != len(items):
        raise SettingError(f"SIRCL_P2P_GROUPS must list distinct kinds of {', '.join(DEFAULT_P2P_GROUPS)}, or "
                           f"none; got {raw!r}")
    return items


def p2p_module(environ: Mapping[str, str] | None = None) -> str:
    return _raw("SIRCL_P2P_MODULE", environ) or DEFAULT_P2P_MODULE


def receipt_dir(environ: Mapping[str, str] | None = None) -> str | None:
    return _raw("SIRCL_RECEIPT_DIR", environ)


def session_modules(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    raw = _raw("SIRCL_SESSION_MODULE", environ)
    if raw is None:
        return DEFAULT_SESSION_MODULES
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def shims(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    raw = _raw("SIRCL_VLLM_SHIMS", environ)
    if raw is None:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def fused_norm(environ: Mapping[str, str] | None = None) -> bool:
    """Whether ``SIRCL_FUSED_NORM`` asks for the fused all-reduce + residual add + RMSNorm (0 or 1)."""
    raw = _raw("SIRCL_FUSED_NORM", environ)
    if raw is None:
        return False
    if raw not in ("0", "1"):
        raise SettingError(f"SIRCL_FUSED_NORM must be 0 or 1, got {raw!r}")
    return raw == "1"


def byte_count(name: str, default: int, environ: Mapping[str, str] | None = None) -> int:
    """A non-negative plain integer byte count (the vLLM plugins' convention)."""
    raw = _raw(name, environ)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SettingError(f"{name} must be an integer byte count, got {raw!r}") from None
    if value < 0:
        raise SettingError(f"{name} must not be negative, got {raw}")
    return value
