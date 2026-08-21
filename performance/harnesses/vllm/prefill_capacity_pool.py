"""Research-only capacity selector for a future tiled TP4 prefill engine.

The executable TP4 adapter still creates one eager native session per exact
payload size. This module describes one physical transport engine with four
logical capacity plans, but does not create an engine or call an unavailable
ABI. Selecting the feature in a serving process remains fail-closed until the
native transport accepts an operation-specific active byte count and streams
it through a fixed generation-tagged tile pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from spark_tp4_port_namespace import validate_control_port_pair

FEATURE_ENV = "VLLM_SPARK_TP4_PREFILL_CAPACITY_POOL"
PORT0_ENV = "SPARK_TP4_PREFILL_POOL_CONTROL_PORT0"
PORT1_ENV = "SPARK_TP4_PREFILL_POOL_CONTROL_PORT1"

TARGET_WIDTH = 6144
BF16_BYTES = 2
BYTES_PER_QUERY_ROW = TARGET_WIDTH * BF16_BYTES
CAPACITY_QUERY_ROWS = (40, 512, 1024, 4096)
DEFAULT_CONTROL_PORTS = (12500, 12501)
TILE_BYTES = 512 * 1024
TILES_PER_EDGE = 8
LANES_PER_EDGE = 2
LANE_PAYLOAD_BYTES = TILE_BYTES // LANES_PER_EDGE
SEND_RECEIVE_REGIONS_PER_LANE = 2
CONTROL_BYTES_PER_LANE = 64
REGISTERED_SLOT_BYTES_PER_EDGE = LANES_PER_EDGE * (
    SEND_RECEIVE_REGIONS_PER_LANE * LANE_PAYLOAD_BYTES
    + CONTROL_BYTES_PER_LANE
)
REGISTERED_BYTES_PER_EDGE = REGISTERED_SLOT_BYTES_PER_EDGE * TILES_PER_EDGE
LOGICAL_ONE_PLANE_PAYLOAD_BYTES_PER_EDGE = TILE_BYTES * TILES_PER_EDGE


@dataclass(frozen=True)
class SharedTransportKey:
    """Identity of the one physical tiled-prefill engine per rank."""

    engine_id: str
    control_ports: tuple[int, int]


@dataclass(frozen=True)
class CapacityPlan:
    """A logical maximum used to select kernels and descriptor bounds."""

    maximum_query_rows: int
    maximum_payload_bytes: int


@dataclass(frozen=True)
class CapacitySelection:
    """One operation routed through the shared engine under a logical plan."""

    query_rows: int
    active_bytes: int
    capacity_plan: CapacityPlan
    transport_key: SharedTransportKey

    @property
    def requires_active_bytes_submission(self) -> bool:
        """True when arena capacity alone cannot describe the operation."""

        return self.active_bytes != self.capacity_plan.maximum_payload_bytes


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    import os

    return os.environ if environ is None else environ


def capacity_pool_requested(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return the explicit research request; default is disabled."""

    value = _environment(environ).get(FEATURE_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{FEATURE_ENV} must be '0', '1', or unset")
    return value == "1"


def _integer(
    environ: Mapping[str, str], name: str, default: int
) -> int:
    value = environ.get(name, str(default))
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def shared_transport_key(
    environ: Mapping[str, str] | None = None,
) -> SharedTransportKey:
    """Return the proposed shared engine identity without binding its ports."""

    environment = _environment(environ)
    base0 = _integer(environment, PORT0_ENV, DEFAULT_CONTROL_PORTS[0])
    base1 = _integer(environment, PORT1_ENV, DEFAULT_CONTROL_PORTS[1])
    ports = validate_control_port_pair(
        (base0, base1),
        owner="research shared tiled-prefill engine",
    )
    return SharedTransportKey(
        engine_id="prefill-tile-engine",
        control_ports=ports,
    )


def select_capacity_plan(
    query_rows: int,
    environ: Mapping[str, str] | None = None,
) -> CapacitySelection:
    """Select the smallest logical plan while retaining one transport key."""

    if (
        not isinstance(query_rows, int)
        or isinstance(query_rows, bool)
        or not 1 <= query_rows <= CAPACITY_QUERY_ROWS[-1]
    ):
        raise ValueError(
            "query rows must be an integer in "
            f"[1, {CAPACITY_QUERY_ROWS[-1]}]: {query_rows!r}"
        )
    capacity_rows = next(
        value for value in CAPACITY_QUERY_ROWS if query_rows <= value
    )
    capacity_plan = CapacityPlan(
        maximum_query_rows=capacity_rows,
        maximum_payload_bytes=capacity_rows * BYTES_PER_QUERY_ROW,
    )
    return CapacitySelection(
        query_rows=query_rows,
        active_bytes=query_rows * BYTES_PER_QUERY_ROW,
        capacity_plan=capacity_plan,
        transport_key=shared_transport_key(environ),
    )


def _projected_exact_q_ports(
    query_rows: int, environ: Mapping[str, str]
) -> tuple[int, int]:
    base0 = _integer(environ, "SPARK_TP4_CONTROL_PORT0", 11000)
    base1 = _integer(environ, "SPARK_TP4_CONTROL_PORT1", 11001)
    offset = (query_rows - 1) * 2
    return validate_control_port_pair(
        (base0 + offset, base1 + offset),
        owner=f"projected exact-Q eager all-reduce Q{query_rows}",
    )


def transport_engine_audit(
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Describe exact-Q scaling and the bounded research replacement.

    Q1024 and Q4096 entries are projections of the present stride-two port
    formula, not claims that the executable adapter admits those shapes.
    """

    environment = _environment(environ)
    exact_projection = []
    for maximum in CAPACITY_QUERY_ROWS:
        exact_projection.append({
            "maximum_query_rows": maximum,
            "session_count_per_rank": maximum,
            "control_port_count_per_rank": maximum * 2,
            "first_control_ports": _projected_exact_q_ports(1, environment),
            "last_control_ports": _projected_exact_q_ports(
                maximum, environment
            ),
            "executable_today": maximum <= 512,
        })

    transport_key = shared_transport_key(environment)
    capacity_rows = [
        {
            "capacity_query_rows": value,
            "maximum_payload_bytes": value * BYTES_PER_QUERY_ROW,
        }
        for value in CAPACITY_QUERY_ROWS
    ]
    pool_ports = set(transport_key.control_ports)
    exact_decode_ports = {
        port
        for query_rows in range(1, CAPACITY_QUERY_ROWS[0] + 1)
        for port in _projected_exact_q_ports(query_rows, environment)
    }
    first_projected_collision = next(
        (
            query_rows
            for query_rows in range(1, CAPACITY_QUERY_ROWS[-1] + 1)
            if not pool_ports.isdisjoint(
                _projected_exact_q_ports(query_rows, environment)
            )
        ),
        None,
    )
    return {
        "status": "research-only",
        "native_dispatch_enabled": False,
        "exact_q_projection": exact_projection,
        "capacity_pool": {
            "transport_key": {
                "engine_id": transport_key.engine_id,
                "control_ports": transport_key.control_ports,
            },
            "logical_capacity_plans": capacity_rows,
            "physical_transport_engines_per_rank": 1,
            "logical_capacity_plan_count": len(CAPACITY_QUERY_ROWS),
            "control_port_count_per_rank": len(pool_ports),
            "ports_disjoint_from_exact_decode_q1_q40": pool_ports.isdisjoint(
                exact_decode_ports
            ),
            "coexistence_contract": (
                "capacity mode must replace exact-Q sessions above Q40 with one "
                "shared tiled-prefill engine; the namespaces do not coexist"
            ),
        },
        "q4096_physical_engine_reduction_factor": float(
            CAPACITY_QUERY_ROWS[-1]
        ),
        "first_projected_exact_q_collision_with_pool": (
            first_projected_collision
        ),
    }


def qualification_plan(
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return the blocked, machine-readable live qualification contract."""

    environment = _environment(environ)
    iterations = {40: 200, 512: 50, 1024: 25, 4096: 10}
    fixed_q_arms = []
    for query_rows in CAPACITY_QUERY_ROWS:
        selection = select_capacity_plan(query_rows, environment)
        fixed_q_arms.append({
            "query_rows": query_rows,
            "active_bytes": selection.active_bytes,
            "capacity_query_rows": (
                selection.capacity_plan.maximum_query_rows
            ),
            "maximum_payload_bytes": (
                selection.capacity_plan.maximum_payload_bytes
            ),
            "logical_payload_tiles": (
                selection.active_bytes + TILE_BYTES - 1
            )
            // TILE_BYTES,
            "transport_engine_id": selection.transport_key.engine_id,
            "control_ports": selection.transport_key.control_ports,
            "warmup_iterations": 10,
            "iterations": iterations[query_rows],
        })

    live_prerequisites = [
        {
            "id": "native-active-bytes-submission",
            "satisfied": False,
            "removal_gate": (
                "one native engine must submit aligned active_bytes not greater "
                "than its class maximum without recreation; zero, misaligned, "
                "and oversized submissions must fail before touching CUDA"
            ),
        },
        {
            "id": "generation-tagged-tile-pool",
            "satisfied": False,
            "removal_gate": (
                "each edge must expose fixed tiles addressed by generation and "
                "slot, carry active_bytes in the operation descriptor, and make "
                "unexpected generations or poisoned slots process-fatal"
            ),
        },
        {
            "id": "cumulative-credit-retirement",
            "satisfied": False,
            "removal_gate": (
                "each edge must publish a monotonic consumed-through watermark; "
                "output readiness must not wait for reciprocal slot retirement"
            ),
        },
        {
            "id": "capacity-port-namespace",
            "satisfied": False,
            "removal_gate": (
                "the shared namespace must reserve exactly one pair for the tiled "
                "engine, replace exact-Q prefill reservations above Q40, reject "
                "collisions, and confirm the selected pairs are free on all ranks"
            ),
        },
        {
            "id": "q4096-kernel-capacity",
            "satisfied": False,
            "removal_gate": (
                "staging, reduction, and output kernels must tile through Q4096 "
                "without a single-CTA whole-payload dependency or an arena-sized "
                "allocation per exact query width"
            ),
        },
        {
            "id": "four-rank-fixed-q-probe",
            "satisfied": False,
            "removal_gate": (
                "the standalone four-rank probe must accept Q40/Q512/Q1024/Q4096, "
                "report capacity-plan/engine/tile counters, and enforce a bounded "
                "watchdog with all-rank receipts"
            ),
        },
        {
            "id": "serving-engine-receipt",
            "satisfied": False,
            "removal_gate": (
                "serving evidence must attest one tiled engine per rank, zero "
                "exact-Q prefill sessions, selected plan and active "
                "bytes for every observed shape, and zero overflow or poison"
            ),
        },
    ]

    return {
        "schema": "sparkring-tp4-prefill-capacity-qualification/v1",
        "status": "research-only",
        "runnable": False,
        "adapter_dispatch_enabled": False,
        "feature_environment": FEATURE_ENV,
        "tile_pool": {
            "tile_bytes": TILE_BYTES,
            "tiles_per_edge": TILES_PER_EDGE,
            "lanes_per_edge": LANES_PER_EDGE,
            "lane_payload_bytes": LANE_PAYLOAD_BYTES,
            "send_receive_regions_per_lane": SEND_RECEIVE_REGIONS_PER_LANE,
            "control_bytes_per_lane": CONTROL_BYTES_PER_LANE,
            "registered_slot_bytes_per_edge": REGISTERED_SLOT_BYTES_PER_EDGE,
            "registered_bytes_per_edge": REGISTERED_BYTES_PER_EDGE,
            "logical_one_plane_payload_bytes_per_edge": (
                LOGICAL_ONE_PLANE_PAYLOAD_BYTES_PER_EDGE
            ),
            "descriptors_included": False,
            "allocation_rule": (
                "physical registered memory includes distinct send and receive "
                "storage plus one 64-byte control per lane; descriptors are "
                "excluded, and allocation is independent of the class maximum"
            ),
        },
        "fixed_q_arms": fixed_q_arms,
        "boundary_query_rows": [
            1,
            39,
            40,
            41,
            511,
            512,
            513,
            1023,
            1024,
            1025,
            4095,
            4096,
        ],
        "required_gates": {
            "correctness": (
                "all four ranks exit zero; exact integer-valued oracle has zero "
                "mismatches; random BF16 is within the declared association gate; "
                "fixed-seed serving output is token-identical"
            ),
            "lifecycle": (
                "one physical engine is created once per rank and reused across all "
                "plans and boundary shapes; engine, tile, generation, credit, overflow, and poison "
                "counters agree on all ranks"
            ),
            "performance": (
                "bracketed baseline/candidate p50 and p95 are reported separately "
                "for every fixed-Q arm; no result combines kernel, protocol, or "
                "dual-port effects under a narrower label"
            ),
            "serving": (
                "shape tracing proves Q40/Q512/Q1024/Q4096 dispatch when those "
                "shapes occur; prefill and C1/C4/C8 decode are reported under an "
                "otherwise matched launch profile"
            ),
        },
        "required_live_receipt": {
            "per_rank_topology": {
                "physical_transport_engine_count": 1,
                "edge_count": 2,
                "qp_count_per_edge": 1,
                "registered_tile_pool_count_per_edge": 1,
                "registered_tile_storage_bytes_per_edge": (
                    REGISTERED_BYTES_PER_EDGE
                ),
                "logical_one_plane_payload_bytes_per_edge": (
                    LOGICAL_ONE_PLANE_PAYLOAD_BYTES_PER_EDGE
                ),
                "descriptor_storage_accounting": "separate",
                "logical_capacity_plan_count": len(CAPACITY_QUERY_ROWS),
                "exact_q_prefill_session_count": 0,
            },
            "per_operation": [
                "query_rows",
                "active_bytes",
                "capacity_plan_maximum_query_rows",
                "first_tile_slot",
                "logical_tile_count",
                "generation",
            ],
            "per_edge_monotonic_counters": [
                "published_generation",
                "consumed_through_generation",
                "tiles_acquired",
                "tiles_recycled",
            ],
            "fatal_counters": {
                "unexpected_generation": 0,
                "poisoned_slot": 0,
                "descriptor_overflow": 0,
                "credit_regression": 0,
            },
        },
        "live_prerequisites": live_prerequisites,
        "transport_engine_audit": transport_engine_audit(environment),
    }
