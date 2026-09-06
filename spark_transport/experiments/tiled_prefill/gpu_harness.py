"""Research-only GPU launch contract for tiled TP4 prefill all-reduce.

The contract answers one bounded question: can BF16 ``[Q, 6144]`` payloads
through Q4096 be represented as exact 512-KiB transport tiles, with multiple
64-KiB worker CTAs per tile and explicit slot-reuse dependencies?  It does not
execute CUDA or RDMA and production transport code does not consume it.

Run the inspectable contract with::

    python -m spark_transport.experiments.tiled_prefill.gpu_harness \
        --query-rows 512 4096
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from typing import Iterable

from spark_transport.experiments.tiled_prefill.substrate import (
    OperationPlan,
    TileTicket,
    TiledCapacityPlanner,
)


PAYLOAD_TILE_BYTES = 512 * 1024
WORKER_CTA_BYTES = 64 * 1024
WORKER_THREADS = 256
CONTROL_THREADS = 32
SLOTS_PER_EDGE = 8
BF16_BYTES = 2
BF16_PAIR_BYTES = 4
CONTROL_ALIGNMENT = 64
CONTROL_BYTES = 64


TIMING_FIELD_SEMANTICS = {
    "isolated_warmup_operations": (
        "completed untimed operations before isolated sampling"
    ),
    "isolated_measured_operations": (
        "number of independently synchronized isolated timing samples"
    ),
    "steady_state_warmup_operations": (
        "completed untimed operations before the back-to-back burst"
    ),
    "steady_state_measured_operations": (
        "number of operations inside the back-to-back timing envelope"
    ),
    "host_submit_us_per_operation": (
        "host elapsed time to submit one complete tiled operation"
    ),
    "device_output_ready_us_min": (
        "minimum isolated GPU time from operation start until every active "
        "output byte is written; final slot retirement is excluded"
    ),
    "device_output_ready_us_p50": (
        "median isolated GPU time from operation start until every active "
        "output byte is written; final slot retirement is excluded"
    ),
    "device_output_ready_us_p95": (
        "p95 isolated GPU time from operation start until every active "
        "output byte is written; final slot retirement is excluded"
    ),
    "steady_state_device_us_per_operation": (
        "mean GPU time per back-to-back operation over a measured burst; "
        "includes credit carry between operations"
    ),
    "device_fully_retired_us_p50": (
        "median isolated GPU-visible time from operation start until all "
        "final-generation slots reach the peer-consumed watermark"
    ),
    "device_fully_retired_us_p95": (
        "p95 isolated GPU-visible time from operation start until all "
        "final-generation slots reach the peer-consumed watermark"
    ),
    "slot_credit_wait_us_p50": (
        "median diagnostic envelope waiting for prior-generation slot credit"
    ),
    "slot_credit_wait_us_p95": (
        "p95 diagnostic envelope waiting for prior-generation slot credit"
    ),
    "stage_phase1_gpu_us_p50": (
        "GPU envelope containing active input staging worker nodes"
    ),
    "phase1_remote_wait_us_p50": (
        "GPU protocol-node envelope from phase-1 publication through exact "
        "remote-sequence observation"
    ),
    "reduce_phase1_gpu_us_p50": (
        "GPU envelope containing first-association BF16 worker nodes"
    ),
    "phase2_remote_wait_us_p50": (
        "GPU protocol-node envelope from phase-2 handoff through exact "
        "remote-sequence observation"
    ),
    "reduce_phase2_gpu_us_p50": (
        "GPU envelope containing final BF16 output worker nodes"
    ),
    "active_payload_gib_per_s_p50": (
        "active payload bytes divided by device_output_ready_us_p50; this is "
        "not wire bandwidth"
    ),
    "steady_state_active_payload_gib_per_s": (
        "active payload bytes divided by steady_state_device_us_per_operation; "
        "this is not wire bandwidth"
    ),
}


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class EndpointTileStorageLayout:
    """Fixed registered storage for one payload tile on one endpoint.

    A counter-rotating tile uses both tensor halves on each endpoint over its
    two phases.  Each half therefore owns distinct send, receive, and control
    regions.  Aliasing send and receive storage could reduce this footprint,
    but requires a separate RDMA ownership proof and is not assumed here.
    """

    payload_tile_bytes: int
    stripe_capacity_bytes: int
    lane_receive_offset: int
    lane_control_offset: int
    lane_stride: int
    lanes_per_endpoint: int
    slot_stride: int
    slots_per_edge: int
    registered_tile_storage_bytes_per_edge: int
    logical_payload_capacity_bytes_per_edge: int

    @classmethod
    def counter_rotating_512k(cls) -> EndpointTileStorageLayout:
        stripe_capacity = PAYLOAD_TILE_BYTES // 2
        receive_offset = _align_up(stripe_capacity, CONTROL_ALIGNMENT)
        control_offset = _align_up(
            receive_offset + stripe_capacity, CONTROL_ALIGNMENT
        )
        lane_stride = control_offset + CONTROL_BYTES
        slot_stride = 2 * lane_stride
        return cls(
            payload_tile_bytes=PAYLOAD_TILE_BYTES,
            stripe_capacity_bytes=stripe_capacity,
            lane_receive_offset=receive_offset,
            lane_control_offset=control_offset,
            lane_stride=lane_stride,
            lanes_per_endpoint=2,
            slot_stride=slot_stride,
            slots_per_edge=SLOTS_PER_EDGE,
            registered_tile_storage_bytes_per_edge=(
                slot_stride * SLOTS_PER_EDGE
            ),
            logical_payload_capacity_bytes_per_edge=(
                PAYLOAD_TILE_BYTES * SLOTS_PER_EDGE
            ),
        )


@dataclass(frozen=True)
class GpuTileDescriptor:
    """Exact active range and worker geometry for one logical payload tile."""

    ordinal: int
    ticket: TileTicket
    input_offset_bytes: int
    output_offset_bytes: int
    active_bytes: int
    stripe_active_bytes: int
    worker_ctas_per_bulk_phase: int
    active_worker_ctas_per_bulk_phase: int
    slot_storage_offset_bytes: int


@dataclass(frozen=True)
class PipelineNode:
    """One executor-neutral dependency node in the correctness DAG."""

    node_id: str
    tile_ordinal: int | None
    phase: str
    executor: str
    dependencies: tuple[str, ...]
    blocks: int
    threads: int
    active_bytes: int
    timing_field: str | None


@dataclass(frozen=True)
class TiledGpuHarnessPlan:
    """Complete research launch contract for one BF16 query width."""

    status: str
    query_rows: int
    payload_bytes: int
    capacity_class: str
    storage: EndpointTileStorageLayout
    descriptors: tuple[GpuTileDescriptor, ...]
    nodes: tuple[PipelineNode, ...]
    output_ready_node: str
    fully_retired_node: str
    timing_fields: tuple[str, ...]

    @property
    def tile_count(self) -> int:
        return len(self.descriptors)

    @property
    def worker_ctas_per_bulk_phase(self) -> int:
        return sum(
            descriptor.worker_ctas_per_bulk_phase
            for descriptor in self.descriptors
        )

    @property
    def active_worker_ctas_per_bulk_phase(self) -> int:
        return sum(
            descriptor.active_worker_ctas_per_bulk_phase
            for descriptor in self.descriptors
        )

    @property
    def partial_tail_bytes(self) -> int:
        tail = self.descriptors[-1].active_bytes
        return 0 if tail == PAYLOAD_TILE_BYTES else tail

    def receipt(self) -> dict:
        """Return a compact JSON-safe description for offline inspection."""

        return {
            "status": self.status,
            "executor_available": False,
            "query_rows": self.query_rows,
            "payload_bytes": self.payload_bytes,
            "capacity_class": self.capacity_class,
            "tile_count": self.tile_count,
            "payload_tile_bytes": PAYLOAD_TILE_BYTES,
            "worker_cta_bytes": WORKER_CTA_BYTES,
            "worker_ctas_per_bulk_phase": self.worker_ctas_per_bulk_phase,
            "active_worker_ctas_per_bulk_phase": (
                self.active_worker_ctas_per_bulk_phase
            ),
            "partial_tail_bytes": self.partial_tail_bytes,
            "storage": asdict(self.storage),
            "pipeline_node_count": len(self.nodes),
            "output_ready_node": self.output_ready_node,
            "fully_retired_node": self.fully_retired_node,
            "timing_fields": {
                field: TIMING_FIELD_SEMANTICS[field]
                for field in self.timing_fields
            },
            "timing_modes_required": [
                "isolated_output_ready_and_retirement",
                "back_to_back_steady_state_throughput",
            ],
            "correctness_fields_required": [
                "mismatched_active_elements",
                "input_guard_corruptions",
                "output_guard_corruptions",
                "inactive_input_sentinel_corruptions",
                "inactive_output_sentinel_corruptions",
                "unexpected_generation_count",
                "credit_regression_count",
                "output_ready_before_final_retirement_count",
                "teardown_pending_tiles",
                "teardown_pending_operations",
            ],
            "limitations": [
                "no CUDA or RDMA execution",
                "no latency or bandwidth claim",
                "protocol nodes are logical and require a native executor",
                "the DAG does not assign streams; one-stream execution would "
                "serialize independent tiles",
                "diagnostic timing envelopes may overlap and must not be summed",
            ],
        }


def _worker_ctas_for_tile(active_bytes: int) -> int:
    if active_bytes <= 0 or active_bytes > PAYLOAD_TILE_BYTES:
        raise ValueError("tile active_bytes must be in [1, 512 KiB]")
    if active_bytes % BF16_PAIR_BYTES != 0:
        raise ValueError(
            "counter-rotating BF16 tile must split into whole BF16 pairs"
        )
    stripe_bytes = active_bytes // 2
    ctas_per_stripe = (
        stripe_bytes + WORKER_CTA_BYTES - 1
    ) // WORKER_CTA_BYTES
    return 2 * ctas_per_stripe


def _descriptor(
    operation: OperationPlan,
    *,
    index: int,
    storage: EndpointTileStorageLayout,
) -> GpuTileDescriptor:
    tile = operation.tiles[index]
    ordinal = tile.ticket.ordinal(operation.slots_per_edge)
    return GpuTileDescriptor(
        ordinal=ordinal,
        ticket=tile.ticket,
        input_offset_bytes=tile.offset_bytes,
        output_offset_bytes=tile.offset_bytes,
        active_bytes=tile.active_bytes,
        stripe_active_bytes=tile.active_bytes // 2,
        # A fixed eight-CTA shape keeps descriptor indexing and graph-capture
        # addresses stable. CTAs outside an active tail return before any
        # load or store; active_worker_ctas records the useful subset.
        worker_ctas_per_bulk_phase=PAYLOAD_TILE_BYTES // WORKER_CTA_BYTES,
        active_worker_ctas_per_bulk_phase=_worker_ctas_for_tile(
            tile.active_bytes
        ),
        slot_storage_offset_bytes=tile.ticket.slot * storage.slot_stride,
    )


def _tile_node_id(index: int, phase: str) -> str:
    return f"tile{index:03d}.{phase}"


def _build_nodes(
    descriptors: tuple[GpuTileDescriptor, ...],
    *,
    slots_per_edge: int,
) -> tuple[PipelineNode, ...]:
    nodes: list[PipelineNode] = []
    releases: list[str] = []
    retirements: list[str] = []

    for index, descriptor in enumerate(descriptors):
        acquire = _tile_node_id(index, "acquire_credit")
        stage = _tile_node_id(index, "stage_phase1")
        exchange1 = _tile_node_id(index, "publish_wait_phase1")
        reduce1 = _tile_node_id(index, "reduce_phase1")
        exchange2 = _tile_node_id(index, "handoff_wait_phase2")
        reduce2 = _tile_node_id(index, "reduce_phase2")
        release = _tile_node_id(index, "release_output")
        retire = _tile_node_id(index, "retire_slot")
        acquire_dependencies = (
            (_tile_node_id(index - slots_per_edge, "retire_slot"),)
            if index >= slots_per_edge
            else ()
        )

        nodes.extend(
            (
                PipelineNode(
                    acquire,
                    descriptor.ordinal,
                    "acquire_credit",
                    "protocol_cta",
                    acquire_dependencies,
                    1,
                    CONTROL_THREADS,
                    descriptor.active_bytes,
                    "slot_credit_wait_us_p50",
                ),
                PipelineNode(
                    stage,
                    descriptor.ordinal,
                    "stage_phase1",
                    "worker_grid",
                    (acquire,),
                    descriptor.worker_ctas_per_bulk_phase,
                    WORKER_THREADS,
                    descriptor.active_bytes,
                    "stage_phase1_gpu_us_p50",
                ),
                PipelineNode(
                    exchange1,
                    descriptor.ordinal,
                    "publish_wait_phase1",
                    "protocol_cta",
                    (stage,),
                    1,
                    CONTROL_THREADS,
                    descriptor.active_bytes,
                    "phase1_remote_wait_us_p50",
                ),
                PipelineNode(
                    reduce1,
                    descriptor.ordinal,
                    "reduce_phase1",
                    "worker_grid",
                    (exchange1,),
                    descriptor.worker_ctas_per_bulk_phase,
                    WORKER_THREADS,
                    descriptor.active_bytes,
                    "reduce_phase1_gpu_us_p50",
                ),
                PipelineNode(
                    exchange2,
                    descriptor.ordinal,
                    "handoff_wait_phase2",
                    "protocol_cta",
                    (reduce1,),
                    1,
                    CONTROL_THREADS,
                    descriptor.active_bytes,
                    "phase2_remote_wait_us_p50",
                ),
                PipelineNode(
                    reduce2,
                    descriptor.ordinal,
                    "reduce_phase2",
                    "worker_grid",
                    (exchange2,),
                    descriptor.worker_ctas_per_bulk_phase,
                    WORKER_THREADS,
                    descriptor.active_bytes,
                    "reduce_phase2_gpu_us_p50",
                ),
                PipelineNode(
                    release,
                    descriptor.ordinal,
                    "release_output",
                    "protocol_cta",
                    (reduce2,),
                    1,
                    CONTROL_THREADS,
                    descriptor.active_bytes,
                    None,
                ),
                PipelineNode(
                    retire,
                    descriptor.ordinal,
                    "retire_slot",
                    "peer_credit",
                    (release,),
                    0,
                    0,
                    descriptor.active_bytes,
                    None,
                ),
            )
        )
        releases.append(release)
        retirements.append(retire)

    nodes.extend(
        (
            PipelineNode(
                "operation.output_ready",
                None,
                "operation_output_ready",
                "cuda_event",
                tuple(releases),
                0,
                0,
                sum(item.active_bytes for item in descriptors),
                "device_output_ready_us_p50",
            ),
            PipelineNode(
                "operation.fully_retired",
                None,
                "operation_fully_retired",
                "peer_credit",
                tuple(retirements),
                0,
                0,
                sum(item.active_bytes for item in descriptors),
                "device_fully_retired_us_p50",
            ),
        )
    )
    return tuple(nodes)


def build_harness_plan(
    query_rows: int, *, elements_per_row: int = 6144
) -> TiledGpuHarnessPlan:
    """Build and validate the tiled GPU dependency contract."""

    operation = TiledCapacityPlanner.tp4_bf16_allreduce(
        elements_per_query_row=elements_per_row
    ).plan(query_rows)
    if operation.tile_bytes != PAYLOAD_TILE_BYTES:
        raise ValueError("GPU contract requires 512-KiB payload tiles")
    if operation.slots_per_edge != SLOTS_PER_EDGE:
        raise ValueError("GPU contract requires eight slots per edge")

    storage = EndpointTileStorageLayout.counter_rotating_512k()
    descriptors = tuple(
        _descriptor(operation, index=index, storage=storage)
        for index in range(operation.tile_count)
    )
    nodes = _build_nodes(descriptors, slots_per_edge=operation.slots_per_edge)
    plan = TiledGpuHarnessPlan(
        status="research-only, offline contract",
        query_rows=query_rows,
        payload_bytes=operation.active_bytes,
        capacity_class=operation.capacity_class,
        storage=storage,
        descriptors=descriptors,
        nodes=nodes,
        output_ready_node="operation.output_ready",
        fully_retired_node="operation.fully_retired",
        timing_fields=tuple(TIMING_FIELD_SEMANTICS),
    )
    validate_harness_plan(plan)
    return plan


def validate_harness_plan(plan: TiledGpuHarnessPlan) -> None:
    """Fail closed on active-range, worker, and dependency mistakes."""

    if plan.status != "research-only, offline contract":
        raise ValueError("tiled GPU harness must remain research-only")
    if sum(item.active_bytes for item in plan.descriptors) != plan.payload_bytes:
        raise ValueError("tile active ranges do not cover the payload exactly")
    expected_offset = 0
    for index, descriptor in enumerate(plan.descriptors):
        if descriptor.input_offset_bytes != expected_offset:
            raise ValueError("tile input ranges must be contiguous")
        if descriptor.output_offset_bytes != expected_offset:
            raise ValueError("tile output ranges must be contiguous")
        if descriptor.active_bytes % BF16_PAIR_BYTES != 0:
            raise ValueError("tile active range must contain whole BF16 pairs")
        if descriptor.stripe_active_bytes * 2 != descriptor.active_bytes:
            raise ValueError("tile stripes must exactly partition active bytes")
        if descriptor.stripe_active_bytes > plan.storage.stripe_capacity_bytes:
            raise ValueError("active stripe exceeds fixed slot storage")
        if (
            descriptor.input_offset_bytes % 16 != 0
            or descriptor.output_offset_bytes % 16 != 0
            or descriptor.stripe_active_bytes % 16 != 0
            or descriptor.slot_storage_offset_bytes % 16 != 0
        ):
            raise ValueError("vectorized tile ranges must be 16-byte aligned")
        if descriptor.worker_ctas_per_bulk_phase != (
            PAYLOAD_TILE_BYTES // WORKER_CTA_BYTES
        ):
            raise ValueError("worker launch must retain eight CTAs per tile")
        if descriptor.active_worker_ctas_per_bulk_phase != (
            _worker_ctas_for_tile(descriptor.active_bytes)
        ):
            raise ValueError("active worker CTA count does not cover the tail")
        if descriptor.ticket.slot != index % SLOTS_PER_EDGE:
            raise ValueError("descriptor slot does not follow the tile ring")
        if descriptor.slot_storage_offset_bytes != (
            descriptor.ticket.slot * plan.storage.slot_stride
        ):
            raise ValueError("descriptor does not name its exact slot storage")
        if (
            descriptor.slot_storage_offset_bytes + plan.storage.slot_stride
            > plan.storage.registered_tile_storage_bytes_per_edge
        ):
            raise ValueError("descriptor exceeds registered tile storage")
        expected_ticket = TileTicket.from_ordinal(index, SLOTS_PER_EDGE)
        if descriptor.ticket != expected_ticket or descriptor.ordinal != index:
            raise ValueError("descriptor generation does not match its ordinal")
        expected_offset += descriptor.active_bytes

    by_id = {node.node_id: node for node in plan.nodes}
    if len(by_id) != len(plan.nodes):
        raise ValueError("pipeline node IDs must be unique")
    positions = {node.node_id: index for index, node in enumerate(plan.nodes)}
    for node in plan.nodes:
        for dependency in node.dependencies:
            if dependency not in by_id:
                raise ValueError(f"pipeline dependency is missing: {dependency}")
            if positions[dependency] >= positions[node.node_id]:
                raise ValueError("pipeline dependencies must be acyclic")

    phase_order = (
        "acquire_credit",
        "stage_phase1",
        "publish_wait_phase1",
        "reduce_phase1",
        "handoff_wait_phase2",
        "reduce_phase2",
        "release_output",
        "retire_slot",
    )
    for index, descriptor in enumerate(plan.descriptors):
        phase_ids = [_tile_node_id(index, phase) for phase in phase_order]
        for previous, current in zip(phase_ids, phase_ids[1:]):
            if by_id[current].dependencies != (previous,):
                raise ValueError("tile phase dependency chain changed")
        acquire = by_id[phase_ids[0]]
        expected_credit = (
            (_tile_node_id(index - SLOTS_PER_EDGE, "retire_slot"),)
            if index >= SLOTS_PER_EDGE
            else ()
        )
        if acquire.dependencies != expected_credit:
            raise ValueError("slot reuse lacks the exact prior-generation credit")
        for phase in ("stage_phase1", "reduce_phase1", "reduce_phase2"):
            worker = by_id[_tile_node_id(index, phase)]
            if (
                worker.executor != "worker_grid"
                or worker.blocks != descriptor.worker_ctas_per_bulk_phase
                or worker.threads != WORKER_THREADS
            ):
                raise ValueError("bulk phase worker geometry changed")

    final_tile_start = max(0, plan.tile_count - SLOTS_PER_EDGE)
    final_retirements = {
        _tile_node_id(index, "retire_slot")
        for index in range(final_tile_start, plan.tile_count)
    }
    output_ready = by_id[plan.output_ready_node]
    if final_retirements & set(output_ready.dependencies):
        raise ValueError(
            "output readiness must not wait for final-generation retirement"
        )
    if set(plan.timing_fields) != set(TIMING_FIELD_SEMANTICS):
        raise ValueError("timing receipt fields changed")


def _parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the research-only tiled TP4 GPU contract"
    )
    parser.add_argument(
        "--query-rows",
        type=int,
        nargs="+",
        required=True,
        help="BF16 [Q, 6144] query widths in [1, 4096]",
    )
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    options = _parse_args(arguments)
    receipts = [
        build_harness_plan(query_rows).receipt()
        for query_rows in options.query_rows
    ]
    print(json.dumps(receipts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
