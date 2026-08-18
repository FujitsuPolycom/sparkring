"""Deterministic control-port namespace for process-local SIRCL sessions.

Every selected TP4 adapter derives this complete plan from the process
environment before it creates a native session.  Reserving every session that
the executable configuration can instantiate prevents a later lazy bind from
colliding with an already-running transport family.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

from spark_tp4_query_contract import MAX_QUERY_ROWS
from spark_tp4_query_row_provider import resolve_query_rows

_MIN_CONTROL_PORT = 1
_MAX_CONTROL_PORT = 65535

_EAGER_ALLREDUCE_DEFAULT_PORTS = (11000, 11001)
_EAGER_ALLREDUCE_PORT_STRIDE = 2
_EAGER_ALLREDUCE_PREFILL_MAX_QUERY_ROWS = 512
_EAGER_ALLREDUCE_DEFAULT_WIDTH_ELEMENTS = 6144

_EAGER_ALLGATHER_DEFAULT_BASE_PORT = 9490
_EAGER_ALLGATHER_PORT_STRIDE = 10
# The exact-signature adapter assigns one durable slot to each admitted
# signature.  Keep its source-level assertion synchronized with this tuple.
EAGER_ALLGATHER_PORT_SLOTS = tuple(range(8))
_EAGER_ALLGATHER_CKV_PORT_SLOTS = frozenset({2, 7})

_GRAPH_INDEXER_DEFAULT_PORTS = (9462, 9463)
_EAGER_DCP_DEFAULT_PORTS = (9890, 9891)
_GRAPH_DCP_DEFAULT_PORTS = (9892, 9893)
_GRAPH_ALLREDUCE_DEFAULT_PORTS = (9970, 9971)
_GRAPH_DUAL_PORT_Q40_DEFAULT_PORTS = (9972, 9973)
_EAGER_VOCAB_DEFAULT_PORTS = (9990, 9991)
_GRAPH_VOCAB_DEFAULT_PORTS = (10110, 10111)


@dataclass(frozen=True)
class PortReservation:
    """One native session's two process-local TCP control ports."""

    owner: str
    ports: tuple[int, int]


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _integer(
    environ: Mapping[str, str], name: str, default: int
) -> int:
    value = environ.get(name, str(default))
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def _flag(
    environ: Mapping[str, str], name: str, default: str = "0"
) -> bool:
    value = environ.get(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be '0', '1', or unset")
    return value == "1"


def _mode(
    environ: Mapping[str, str],
    name: str,
    allowed: frozenset[str],
) -> str:
    value = environ.get(name, "").lower()
    if value and value not in allowed:
        choices = ", ".join(repr(choice) for choice in sorted(allowed))
        raise ValueError(f"{name} must be one of {choices}, or unset")
    return value


def validate_control_port_pair(
    ports: tuple[int, int], *, owner: str
) -> tuple[int, int]:
    """Validate one pair independently of whether its family is selected."""

    if len(ports) != 2:
        raise ValueError(f"Spark TP4 {owner} requires exactly two control ports")
    port0, port1 = ports
    if not all(
        isinstance(port, int)
        and not isinstance(port, bool)
        and _MIN_CONTROL_PORT <= port <= _MAX_CONTROL_PORT
        for port in ports
    ):
        raise ValueError(
            f"Spark TP4 {owner} control ports must be in "
            f"[{_MIN_CONTROL_PORT}, {_MAX_CONTROL_PORT}]: {ports}"
        )
    if port0 == port1:
        raise ValueError(
            f"Spark TP4 {owner} control ports must be distinct: {ports}"
        )
    return ports


def _configured_pair(
    environ: Mapping[str, str],
    *,
    owner: str,
    name0: str,
    name1: str,
    defaults: tuple[int, int],
) -> tuple[int, int]:
    return validate_control_port_pair(
        (
            _integer(environ, name0, defaults[0]),
            _integer(environ, name1, defaults[1]),
        ),
        owner=owner,
    )


def _extension_maximum_query_rows(environ: Mapping[str, str]) -> int:
    """Row bound for non-default-width extension payloads.

    Extensions enumerate the contiguous row range regardless of any
    configured row-policy provider: a provider constrains default-width
    serving, not the extension widths.
    """
    return (
        _EAGER_ALLREDUCE_PREFILL_MAX_QUERY_ROWS
        if _flag(environ, "VLLM_SPARK_TP4_PREFILL_Q512")
        else MAX_QUERY_ROWS
    )


def _supported_allreduce_query_rows(
    environ: Mapping[str, str],
) -> tuple[int, ...]:
    return resolve_query_rows(environ)


_EAGER_ALLREDUCE_WIDTH_ENV = "VLLM_SPARK_TP4_EAGER_WIDTHS"
# Width cap: 1_048_576 elements x 512 max rows x 2 bytes = 1 GiB, which
# stays under the native single-RDMA-write bound of UINT32_MAX bytes by
# construction.
_EAGER_ALLREDUCE_MAX_WIDTH_ELEMENTS = 1_048_576
_EAGER_ALLREDUCE_BF16_BYTES = 2


def eager_allreduce_admitted_widths(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """Parse VLLM_SPARK_TP4_EAGER_WIDTHS into a sorted ascending tuple.

    Unset or empty -> (6144,). Otherwise comma-separated integer widths
    (elements per row). Fail-closed ValueError names the env var on any
    empty token, non-integer, out-of-range, or duplicated width.
    """
    environment = _environment(environ)
    raw = environment.get(_EAGER_ALLREDUCE_WIDTH_ENV, "")
    if not raw:
        return (_EAGER_ALLREDUCE_DEFAULT_WIDTH_ELEMENTS,)
    tokens = [token.strip() for token in raw.split(",")]
    widths: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if not token:
            raise ValueError(
                f"{_EAGER_ALLREDUCE_WIDTH_ENV} contains an empty token"
            )
        try:
            width = int(token)
        except ValueError as error:
            raise ValueError(
                f"{_EAGER_ALLREDUCE_WIDTH_ENV} token {token!r} is not "
                "an integer"
            ) from error
        if width < 1 or width > _EAGER_ALLREDUCE_MAX_WIDTH_ELEMENTS:
            raise ValueError(
                f"{_EAGER_ALLREDUCE_WIDTH_ENV} width {width} is out of "
                f"range [1, {_EAGER_ALLREDUCE_MAX_WIDTH_ELEMENTS}]"
            )
        if width in seen:
            raise ValueError(
                f"{_EAGER_ALLREDUCE_WIDTH_ENV} duplicates width {width}"
            )
        seen.add(width)
        widths.append(width)
    return tuple(sorted(widths))


def eager_allreduce_payload_sizes(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """Sorted unique payload byte counts for all admitted widths and rows.

    Each entry is rows * width * 2 bytes for every admitted width and every
    rows in [1, _extension_maximum_query_rows(environ)]. Cross-width
    duplicates collapse to one entry: the same byte count is the same
    native operation.
    """
    environment = _environment(environ)
    legacy, extensions = _eager_allreduce_size_regimes(environment)
    return tuple(sorted(legacy | set(extensions)))


def _eager_allreduce_size_regimes(
    environ: Mapping[str, str],
) -> tuple[set[int], tuple[int, ...]]:
    """Split admissible payload sizes into their two port regimes.

    Legacy sizes are the default width's row-denominated payloads over
    the resolved query-row set (spark_tp4_query_row_provider owns that
    resolution) and keep the row-slot port formula, holes included.
    Extension sizes come from non-default admitted widths over the
    contiguous row range; a row-policy provider is a default-width
    serving constraint and does not restrict them. A non-default-width
    payload equal to a supported legacy payload belongs to the legacy
    regime: an identical byte count is the same native operation.
    """
    default_bytes_per_row = (
        _EAGER_ALLREDUCE_DEFAULT_WIDTH_ELEMENTS
        * _EAGER_ALLREDUCE_BF16_BYTES
    )
    supported = _supported_allreduce_query_rows(environ)
    legacy = {rows * default_bytes_per_row for rows in supported}
    maximum = _extension_maximum_query_rows(environ)
    extensions: set[int] = set()
    for width in eager_allreduce_admitted_widths(environ):
        if width == _EAGER_ALLREDUCE_DEFAULT_WIDTH_ELEMENTS:
            continue
        bytes_per_row = width * _EAGER_ALLREDUCE_BF16_BYTES
        for rows in range(1, maximum + 1):
            size = rows * bytes_per_row
            if size not in legacy:
                extensions.add(size)
    return legacy, tuple(sorted(extensions))


def eager_allreduce_ports_for_payload(
    payload_bytes: int,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Control-port pair for one payload size under two slot regimes.

    A legacy payload (default width, resolved row set) occupies slot
    ``row - 1``; unsupported rows leave permanent holes. An extension
    payload (non-default width) occupies slot ``512 + i``, where ``i``
    is its position in the sorted extension-size tuple, past the
    largest legacy span. The pair is
    ``(base0 + 2*slot, base1 + 2*slot)``, validated as a pair.
    """
    environment = _environment(environ)
    legacy, extensions = _eager_allreduce_size_regimes(environment)
    default_bytes_per_row = (
        _EAGER_ALLREDUCE_DEFAULT_WIDTH_ELEMENTS
        * _EAGER_ALLREDUCE_BF16_BYTES
    )
    if payload_bytes in legacy:
        # Deployed row-slot formula: slot = row - 1, so unsupported rows
        # leave permanent holes in the port sequence. The exact-state
        # arena accounting depends on these exact pairs.
        slot = payload_bytes // default_bytes_per_row - 1
    elif payload_bytes in extensions:
        # Width extensions occupy slots past the largest legacy span
        # (Q512 ends at slot 511) so no row-policy or Q512 toggle
        # moves them.
        slot = (
            _EAGER_ALLREDUCE_PREFILL_MAX_QUERY_ROWS
            + extensions.index(payload_bytes)
        )
    else:
        raise ValueError(
            "unsupported Spark TP4 eager all-reduce payload size: "
            f"{payload_bytes} bytes"
        )
    base0 = _integer(
        environment,
        "SPARK_TP4_CONTROL_PORT0",
        _EAGER_ALLREDUCE_DEFAULT_PORTS[0],
    )
    base1 = _integer(
        environment,
        "SPARK_TP4_CONTROL_PORT1",
        _EAGER_ALLREDUCE_DEFAULT_PORTS[1],
    )
    offset = slot * _EAGER_ALLREDUCE_PORT_STRIDE
    return validate_control_port_pair(
        (base0 + offset, base1 + offset),
        owner=f"eager all-reduce payload {payload_bytes}",
    )


def _eager_allreduce_pair(
    query_rows: int, environ: Mapping[str, str]
) -> tuple[int, int]:
    supported = _supported_allreduce_query_rows(environ)
    if (
        not isinstance(query_rows, int)
        or isinstance(query_rows, bool)
        or query_rows not in supported
    ):
        raise ValueError(
            "Spark TP4 eager all-reduce query rows must be in the exact "
            f"supported set {supported}: {query_rows}"
        )
    payload_bytes = (
        query_rows
        * _EAGER_ALLREDUCE_DEFAULT_WIDTH_ELEMENTS
        * _EAGER_ALLREDUCE_BF16_BYTES
    )
    return eager_allreduce_ports_for_payload(payload_bytes, environ)


def _eager_allgather_pair(
    port_slot: int, environ: Mapping[str, str]
) -> tuple[int, int]:
    if port_slot not in EAGER_ALLGATHER_PORT_SLOTS:
        raise ValueError(
            "Spark TP4 eager all-gather port slot must be one of "
            f"{EAGER_ALLGATHER_PORT_SLOTS}: {port_slot}"
        )
    base = _integer(
        environ,
        "SPARK_TP4_ALLGATHER_BASE_PORT",
        _EAGER_ALLGATHER_DEFAULT_BASE_PORT,
    )
    port0 = base + port_slot * _EAGER_ALLGATHER_PORT_STRIDE
    return validate_control_port_pair(
        (port0, port0 + 1),
        owner=f"eager all-gather slot {port_slot}",
    )


def _active_eager_allgather_port_slots(
    environ: Mapping[str, str],
) -> tuple[int, ...]:
    ckv_enabled = _flag(
        environ, "SPARK_TP4_ALLGATHER_ENABLE_CKV"
    )
    return tuple(
        port_slot
        for port_slot in EAGER_ALLGATHER_PORT_SLOTS
        if ckv_enabled
        or port_slot not in _EAGER_ALLGATHER_CKV_PORT_SLOTS
    )


def _graph_allreduce_pair(environ: Mapping[str, str]) -> tuple[int, int]:
    return _configured_pair(
        environ,
        owner="graph all-reduce",
        name0="SPARK_TP4_GRAPH_CONTROL_PORT0",
        name1="SPARK_TP4_GRAPH_CONTROL_PORT1",
        defaults=_GRAPH_ALLREDUCE_DEFAULT_PORTS,
    )


def _graph_dual_port_q40_pair(
    environ: Mapping[str, str],
) -> tuple[int, int]:
    return _configured_pair(
        environ,
        owner="graph exact-Q40 dual-port all-reduce",
        name0="SPARK_TP4_GRAPH_DUAL_PORT_Q40_CONTROL_PORT0",
        name1="SPARK_TP4_GRAPH_DUAL_PORT_Q40_CONTROL_PORT1",
        defaults=_GRAPH_DUAL_PORT_Q40_DEFAULT_PORTS,
    )


def _graph_indexer_pair(environ: Mapping[str, str]) -> tuple[int, int]:
    return _configured_pair(
        environ,
        owner="graph indexer all-gather",
        name0="SPARK_TP4_GRAPH_INDEXER_CONTROL_PORT0",
        name1="SPARK_TP4_GRAPH_INDEXER_CONTROL_PORT1",
        defaults=_GRAPH_INDEXER_DEFAULT_PORTS,
    )


def _eager_dcp_pair(environ: Mapping[str, str]) -> tuple[int, int]:
    return _configured_pair(
        environ,
        owner="eager DCP",
        name0="SPARK_TP4_DCP_CONTROL_PORT0",
        name1="SPARK_TP4_DCP_CONTROL_PORT1",
        defaults=_EAGER_DCP_DEFAULT_PORTS,
    )


def _graph_dcp_pair(environ: Mapping[str, str]) -> tuple[int, int]:
    return _configured_pair(
        environ,
        owner="graph DCP",
        name0="SPARK_TP4_GRAPH_DCP_CONTROL_PORT0",
        name1="SPARK_TP4_GRAPH_DCP_CONTROL_PORT1",
        defaults=_GRAPH_DCP_DEFAULT_PORTS,
    )


def _eager_vocab_pair(environ: Mapping[str, str]) -> tuple[int, int]:
    return _configured_pair(
        environ,
        owner="eager vocabulary all-gather",
        name0="SPARK_TP4_VOCAB_CONTROL_PORT0",
        name1="SPARK_TP4_VOCAB_CONTROL_PORT1",
        defaults=_EAGER_VOCAB_DEFAULT_PORTS,
    )


def _graph_vocab_pair(environ: Mapping[str, str]) -> tuple[int, int]:
    return _configured_pair(
        environ,
        owner="graph vocabulary all-gather",
        name0="SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT0",
        name1="SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT1",
        defaults=_GRAPH_VOCAB_DEFAULT_PORTS,
    )


def active_port_reservations(
    environ: Mapping[str, str] | None = None,
) -> tuple[PortReservation, ...]:
    """Return every control-port pair the selected adapters may bind."""

    environment = _environment(environ)
    reservations: list[PortReservation] = []

    allreduce_mode = _mode(
        environment,
        "VLLM_SPARK_TP4_MODE",
        frozenset({"custom", "disabled", "shadow"}),
    )
    if allreduce_mode in {"custom", "shadow"}:
        for query_rows in _supported_allreduce_query_rows(environment):
            reservations.append(
                PortReservation(
                    f"eager_allreduce:q={query_rows}",
                    _eager_allreduce_pair(query_rows, environment),
                )
            )
        _, extension_sizes = _eager_allreduce_size_regimes(environment)
        for payload_bytes in extension_sizes:
            reservations.append(
                PortReservation(
                    f"eager_allreduce:payload={payload_bytes}",
                    eager_allreduce_ports_for_payload(
                        payload_bytes, environment
                    ),
                )
            )
        if allreduce_mode == "custom" and _flag(
            environment, "VLLM_SPARK_TP4_GRAPH_Q1"
        ):
            reservations.append(
                PortReservation(
                    "graph_allreduce", _graph_allreduce_pair(environment)
                )
            )
            if _flag(
                environment, "VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40"
            ):
                reservations.append(
                    PortReservation(
                        "graph_dual_port_q40_allreduce",
                        _graph_dual_port_q40_pair(environment),
                    )
                )

    allgather_mode = _mode(
        environment,
        "VLLM_SPARK_TP4_ALLGATHER_MODE",
        frozenset({"custom", "shadow"}),
    )
    if allgather_mode:
        for port_slot in _active_eager_allgather_port_slots(environment):
            reservations.append(
                PortReservation(
                    f"eager_allgather:slot={port_slot}",
                    _eager_allgather_pair(port_slot, environment),
                )
            )
        if allgather_mode == "custom" and _flag(
            environment, "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM"
        ):
            reservations.append(
                PortReservation(
                    "graph_indexer_allgather",
                    _graph_indexer_pair(environment),
                )
            )

    vocab_mode = _mode(
        environment,
        "VLLM_SPARK_TP4_VOCAB_MODE",
        frozenset({"custom", "shadow"}),
    )
    if vocab_mode:
        reservations.append(
            PortReservation("eager_vocab", _eager_vocab_pair(environment))
        )
        if vocab_mode == "custom" and _flag(
            environment, "VLLM_SPARK_TP4_GRAPH_Q1"
        ):
            reservations.append(
                PortReservation("graph_vocab", _graph_vocab_pair(environment))
            )

    dcp_mode = _mode(
        environment,
        "VLLM_SPARK_TP4_DCP_MODE",
        frozenset({"custom", "shadow"}),
    )
    if dcp_mode:
        query_enabled = _flag(
            environment, "VLLM_SPARK_TP4_DCP_QUERY_ENABLED", "1"
        )
        combine_enabled = _flag(
            environment, "VLLM_SPARK_TP4_DCP_COMBINE_ENABLED", "1"
        )
        if query_enabled or combine_enabled:
            reservations.append(
                PortReservation("eager_dcp", _eager_dcp_pair(environment))
            )
            graph_shadow = _flag(
                environment, "VLLM_SPARK_TP4_DCP_GRAPH_SHADOW"
            )
            graph_custom = _flag(
                environment, "VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM"
            )
            if graph_shadow and graph_custom:
                raise ValueError(
                    "Spark TP4 DCP graph shadow and custom modes are "
                    "mutually exclusive"
                )
            if graph_shadow and dcp_mode != "shadow":
                raise ValueError(
                    "VLLM_SPARK_TP4_DCP_GRAPH_SHADOW requires DCP shadow mode"
                )
            if graph_custom and dcp_mode != "custom":
                raise ValueError(
                    "VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM requires DCP custom mode"
                )
            if graph_shadow or graph_custom:
                reservations.append(
                    PortReservation("graph_dcp", _graph_dcp_pair(environment))
                )

    return tuple(reservations)


def validate_active_port_namespace(
    environ: Mapping[str, str] | None = None,
) -> tuple[PortReservation, ...]:
    """Reject out-of-range, internally duplicated, or shared active ports."""

    reservations = active_port_reservations(environ)
    owners_by_port: dict[int, list[str]] = defaultdict(list)
    for reservation in reservations:
        validate_control_port_pair(
            reservation.ports, owner=reservation.owner
        )
        for port in reservation.ports:
            owners_by_port[port].append(reservation.owner)

    collisions = {
        port: tuple(owners)
        for port, owners in sorted(owners_by_port.items())
        if len(owners) > 1
    }
    if collisions:
        details = "; ".join(
            f"port {port}: {', '.join(owners)}"
            for port, owners in collisions.items()
        )
        raise ValueError(
            "Spark TP4 active control-port namespaces collide: " + details
        )
    return reservations


def eager_allreduce_control_ports(
    query_rows: int, environ: Mapping[str, str] | None = None
) -> tuple[int, int]:
    environment = _environment(environ)
    ports = _eager_allreduce_pair(query_rows, environment)
    validate_active_port_namespace(environment)
    return ports


def graph_allreduce_control_ports(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    environment = _environment(environ)
    ports = _graph_allreduce_pair(environment)
    validate_active_port_namespace(environment)
    return ports


def graph_dual_port_q40_control_ports(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    environment = _environment(environ)
    ports = _graph_dual_port_q40_pair(environment)
    validate_active_port_namespace(environment)
    return ports


def eager_allgather_control_ports(
    port_slot: int, environ: Mapping[str, str] | None = None
) -> tuple[int, int]:
    environment = _environment(environ)
    ports = _eager_allgather_pair(port_slot, environment)
    validate_active_port_namespace(environment)
    return ports


def graph_indexer_control_ports(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    environment = _environment(environ)
    ports = _graph_indexer_pair(environment)
    validate_active_port_namespace(environment)
    return ports


def eager_dcp_control_ports(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    environment = _environment(environ)
    ports = _eager_dcp_pair(environment)
    validate_active_port_namespace(environment)
    return ports


def graph_dcp_control_ports(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    environment = _environment(environ)
    ports = _graph_dcp_pair(environment)
    validate_active_port_namespace(environment)
    return ports


def eager_vocab_control_ports(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    environment = _environment(environ)
    ports = _eager_vocab_pair(environment)
    validate_active_port_namespace(environment)
    return ports


def graph_vocab_control_ports(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    environment = _environment(environ)
    ports = _graph_vocab_pair(environment)
    validate_active_port_namespace(environment)
    return ports
