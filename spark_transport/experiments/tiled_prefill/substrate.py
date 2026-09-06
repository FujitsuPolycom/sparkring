"""CPU-only substrate for bounded, tiled TP4 collective planning.

The module describes transport work without importing CUDA, vLLM, or native
transport bindings.  It is research-only: production sessions do not consume
these descriptors until a separate native integration proves their wire and
retirement invariants.
"""

from __future__ import annotations

from dataclasses import dataclass


MAX_TILE_GENERATION = (1 << 64) - 1


class TileProtocolError(RuntimeError):
    """A generation, slot, or credit invariant was violated."""


class UnexpectedTileTicket(TileProtocolError):
    """A descriptor named a slot generation other than the expected one."""


class TileCreditUnavailable(TileProtocolError):
    """A slot's prior generation has not reached the consumed watermark."""


@dataclass(frozen=True)
class CapacityClass:
    """One stable session identity covering a contiguous query-width range."""

    name: str
    maximum_query_rows: int


@dataclass(frozen=True)
class TileTicket:
    """Generation-tagged identity for one physical tile-pool slot."""

    generation: int
    slot: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or not 1 <= self.generation <= MAX_TILE_GENERATION
        ):
            raise ValueError(
                f"generation must be in [1, {MAX_TILE_GENERATION}]"
            )
        if (
            isinstance(self.slot, bool)
            or not isinstance(self.slot, int)
            or self.slot < 0
        ):
            raise ValueError("slot must be a nonnegative integer")

    @classmethod
    def from_ordinal(cls, ordinal: int, slots_per_edge: int) -> TileTicket:
        """Map a zero-based logical tile ordinal onto a reusable slot."""

        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
        ):
            raise ValueError("tile ordinal must be a nonnegative integer")
        if (
            isinstance(slots_per_edge, bool)
            or not isinstance(slots_per_edge, int)
            or slots_per_edge <= 0
        ):
            raise ValueError("slots_per_edge must be a positive integer")
        generation_index, slot = divmod(ordinal, slots_per_edge)
        if generation_index >= MAX_TILE_GENERATION:
            raise OverflowError("tile ticket generation exhausted")
        return cls(generation=generation_index + 1, slot=slot)

    def ordinal(self, slots_per_edge: int) -> int:
        """Recover the zero-based logical ordinal for this pool geometry."""

        if (
            isinstance(slots_per_edge, bool)
            or not isinstance(slots_per_edge, int)
            or slots_per_edge <= 0
        ):
            raise ValueError("slots_per_edge must be a positive integer")
        if self.slot >= slots_per_edge:
            raise ValueError(
                f"ticket slot {self.slot} is outside a {slots_per_edge}-slot pool"
            )
        return (self.generation - 1) * slots_per_edge + self.slot


@dataclass(frozen=True)
class TileDescriptor:
    """One contiguous active byte range assigned to a tagged tile slot."""

    ticket: TileTicket
    offset_bytes: int
    active_bytes: int


def validate_tile_acquisition(
    ticket: TileTicket,
    *,
    expected_ordinal: int,
    consumed_through: int,
    slots_per_edge: int,
) -> None:
    """Validate identity and cumulative credit before reusing one tile.

    ``consumed_through`` is an inclusive logical-tile ordinal.  Generation one
    has no predecessor.  Every later generation of a slot requires the peer to
    have consumed that slot's immediately preceding ordinal.
    """

    expected = TileTicket.from_ordinal(expected_ordinal, slots_per_edge)
    if ticket != expected:
        raise UnexpectedTileTicket(
            "unexpected tile ticket: "
            f"expected generation={expected.generation} slot={expected.slot}, "
            f"got generation={ticket.generation} slot={ticket.slot}"
        )
    prior_ordinal = expected_ordinal - slots_per_edge
    if prior_ordinal >= 0 and consumed_through < prior_ordinal:
        raise TileCreditUnavailable(
            "tile slot reuse requires peer consumption through "
            f"ordinal {prior_ordinal}; watermark is {consumed_through}"
        )


@dataclass(frozen=True)
class OperationPlan:
    """Capacity decision for one logical collective operation."""

    family: str
    collective: str
    dtype: str
    reduction_algebra: str
    query_rows: int
    active_bytes: int
    capacity_class: str
    capacity_query_rows: int
    session_key: TiledSessionKey
    tile_bytes: int
    slots_per_edge: int
    tiles: tuple[TileDescriptor, ...]

    @property
    def tile_count(self) -> int:
        return len(self.tiles)

    @property
    def arena_bytes_per_edge(self) -> int:
        return self.tile_bytes * self.slots_per_edge


@dataclass(frozen=True)
class TiledSessionKey:
    """Complete compatibility identity for one reusable tiled session."""

    family: str
    collective: str
    dtype: str
    reduction_algebra: str
    bytes_per_query_row: int
    capacity_class: str
    capacity_query_rows: int
    tile_bytes: int
    slots_per_edge: int


@dataclass(frozen=True)
class PayloadPartition:
    """A disjoint contiguous byte range reduced as one symbolic value."""

    name: str
    offset_bytes: int
    active_bytes: int


@dataclass(frozen=True)
class MatchingExchange:
    """One synchronous TP4 matching exchange for one payload partition."""

    stage: int
    partition: str
    endpoint: int
    matching_mask: int


@dataclass(frozen=True)
class Tp4AllreduceSchedule:
    """Topology-specific all-reduce schedule independent of native transport."""

    name: str
    payload_bytes: int
    partitions: tuple[PayloadPartition, ...]
    exchanges: tuple[MatchingExchange, ...]


@dataclass(frozen=True)
class AllreduceScheduleVerification:
    """Symbolic completeness and byte accounting for a TP4 all-reduce plan."""

    complete: bool
    contributor_masks: dict[str, tuple[int, int, int, int]]
    association_fingerprints: dict[str, tuple[str, str, str, str]]
    edge_bytes_per_rank: tuple[int, int]
    stage_edge_bytes_per_rank: tuple[tuple[int, int], ...]
    ideal_critical_path_bytes_per_rank: int


@dataclass(frozen=True, order=True)
class BidirectionalRingChunk:
    """One quarter-shard of one directional tensor half."""

    half: int
    shard: int


@dataclass(frozen=True)
class BidirectionalRingTransfer:
    """One direct-neighbor reduce or gather transfer."""

    stage: int
    phase: str
    direction: int
    sender_rank: int
    receiver_rank: int
    endpoint: int
    chunk: BidirectionalRingChunk


@dataclass(frozen=True)
class BidirectionalRingAllreduceSchedule:
    """Two counter-rotating TP4 rings over disjoint tensor halves."""

    payload_bytes: int
    chunk_bytes: int
    transfers: tuple[BidirectionalRingTransfer, ...]


@dataclass(frozen=True)
class BidirectionalRingAllreduceVerification:
    """Contributor closure and per-rank physical-edge byte accounting."""

    complete: bool
    stage_count: int
    contributor_masks_by_rank: tuple[tuple[int, ...], ...]
    edge_bytes_by_rank: tuple[tuple[int, int], ...]
    total_bytes_per_rank: int
    ideal_critical_path_bytes_per_rank: int


@dataclass(frozen=True, order=True)
class GatherSegment:
    """One independently relayable half of a rank-owned all-gather shard."""

    owner_rank: int
    part: int


@dataclass(frozen=True)
class DirectedSegmentTransfer:
    """Segments copied from one TP4 rank to a directly cabled peer."""

    stage: int
    sender_rank: int
    receiver_rank: int
    endpoint: int
    segments: tuple[GatherSegment, ...]


@dataclass(frozen=True)
class Tp4AllgatherSchedule:
    """Directed TP4 all-gather plan over two halves of every local shard."""

    name: str
    shard_bytes: int
    transfers: tuple[DirectedSegmentTransfer, ...]


@dataclass(frozen=True)
class AllgatherScheduleVerification:
    """Symbolic shard coverage and per-rank endpoint byte accounting."""

    complete: bool
    segments_by_rank: tuple[tuple[GatherSegment, ...], ...]
    edge_bytes_by_rank: tuple[tuple[int, int], ...]
    stage_edge_bytes_by_rank: tuple[tuple[tuple[int, int], ...], ...]
    ideal_critical_path_bytes_per_rank: int


def _rank_shard_segments(rank: int) -> tuple[GatherSegment, GatherSegment]:
    return (GatherSegment(rank, 0), GatherSegment(rank, 1))


def sequential_allgather_schedule(shard_bytes: int) -> Tp4AllgatherSchedule:
    """Return recursive doubling over serialized XOR-1 and XOR-3 edges."""

    if shard_bytes <= 0 or shard_bytes % 2 != 0:
        raise ValueError("shard_bytes must split into two nonempty halves")
    transfers: list[DirectedSegmentTransfer] = []
    for rank in range(4):
        transfers.append(
            DirectedSegmentTransfer(
                0,
                rank,
                rank ^ 1,
                0,
                _rank_shard_segments(rank),
            )
        )
        transfers.append(
            DirectedSegmentTransfer(
                1,
                rank,
                rank ^ 3,
                1,
                _rank_shard_segments(rank)
                + _rank_shard_segments(rank ^ 1),
            )
        )
    return Tp4AllgatherSchedule(
        name="sequential_recursive_doubling",
        shard_bytes=shard_bytes,
        transfers=tuple(transfers),
    )


def split_relay_allgather_schedule(shard_bytes: int) -> Tp4AllgatherSchedule:
    """Return direct-to-both-neighbors then opposite-rank half relays."""

    if shard_bytes <= 0 or shard_bytes % 2 != 0:
        raise ValueError("shard_bytes must split into two nonempty halves")
    transfers: list[DirectedSegmentTransfer] = []
    for rank in range(4):
        own = _rank_shard_segments(rank)
        transfers.extend(
            (
                DirectedSegmentTransfer(0, rank, rank ^ 1, 0, own),
                DirectedSegmentTransfer(0, rank, rank ^ 3, 1, own),
                DirectedSegmentTransfer(
                    1,
                    rank,
                    rank ^ 1,
                    0,
                    (GatherSegment(rank ^ 3, 1),),
                ),
                DirectedSegmentTransfer(
                    1,
                    rank,
                    rank ^ 3,
                    1,
                    (GatherSegment(rank ^ 1, 0),),
                ),
            )
        )
    return Tp4AllgatherSchedule(
        name="two_hop_split_relay",
        shard_bytes=shard_bytes,
        transfers=tuple(transfers),
    )


def verify_allgather_schedule(
    schedule: Tp4AllgatherSchedule,
) -> AllgatherScheduleVerification:
    """Prove complete TP4 shard delivery and account directed wire bytes."""

    if schedule.shard_bytes <= 0 or schedule.shard_bytes % 2 != 0:
        raise ValueError("schedule shard_bytes must split into two halves")
    segment_bytes = schedule.shard_bytes // 2
    expected_segments = frozenset(
        GatherSegment(owner, part)
        for owner in range(4)
        for part in range(2)
    )
    holdings = [set(_rank_shard_segments(rank)) for rank in range(4)]
    edge_bytes = [[0, 0] for _ in range(4)]
    stages = sorted({transfer.stage for transfer in schedule.transfers})
    if stages != list(range(len(stages))):
        raise ValueError("schedule stages must be contiguous from zero")
    stage_edge_bytes: list[tuple[tuple[int, int], ...]] = []
    for stage in stages:
        before = [set(rank_segments) for rank_segments in holdings]
        stage_bytes = [[0, 0] for _ in range(4)]
        used_sender_endpoints: set[tuple[int, int]] = set()
        for transfer in (
            candidate
            for candidate in schedule.transfers
            if candidate.stage == stage
        ):
            if transfer.sender_rank not in range(4) or transfer.receiver_rank not in range(4):
                raise ValueError("TP4 transfer ranks must be in [0, 3]")
            if transfer.endpoint not in (0, 1):
                raise ValueError("TP4 endpoint must be zero or one")
            matching_mask = 1 if transfer.endpoint == 0 else 3
            if transfer.receiver_rank != transfer.sender_rank ^ matching_mask:
                raise ValueError("all-gather transfer does not follow its endpoint matching")
            key = (transfer.sender_rank, transfer.endpoint)
            if key in used_sender_endpoints:
                raise ValueError(
                    "one rank endpoint cannot carry two transfers in one stage"
                )
            used_sender_endpoints.add(key)
            selected = frozenset(transfer.segments)
            if not selected or not selected <= expected_segments:
                raise ValueError("all-gather transfer names invalid segments")
            if not selected <= before[transfer.sender_rank]:
                missing = sorted(selected - before[transfer.sender_rank])
                raise ValueError(
                    "all-gather transfer sends unavailable segments: "
                    f"rank={transfer.sender_rank} missing={missing}"
                )
            holdings[transfer.receiver_rank].update(selected)
            transferred_bytes = len(selected) * segment_bytes
            edge_bytes[transfer.sender_rank][transfer.endpoint] += transferred_bytes
            stage_bytes[transfer.sender_rank][transfer.endpoint] += transferred_bytes
        stage_edge_bytes.append(
            tuple((rank_bytes[0], rank_bytes[1]) for rank_bytes in stage_bytes)
        )

    complete = all(rank_segments == expected_segments for rank_segments in holdings)
    return AllgatherScheduleVerification(
        complete=complete,
        segments_by_rank=tuple(
            tuple(sorted(rank_segments)) for rank_segments in holdings
        ),
        edge_bytes_by_rank=tuple(
            (rank_bytes[0], rank_bytes[1]) for rank_bytes in edge_bytes
        ),
        stage_edge_bytes_by_rank=tuple(stage_edge_bytes),
        ideal_critical_path_bytes_per_rank=sum(
            max(max(per_rank) for per_rank in stage)
            for stage in stage_edge_bytes
        ),
    )


def sequential_allreduce_schedule(payload_bytes: int) -> Tp4AllreduceSchedule:
    """Return the serialized XOR-1 then XOR-3 TP4 reduction tree."""

    if payload_bytes <= 0:
        raise ValueError("payload_bytes must be positive")
    return Tp4AllreduceSchedule(
        name="sequential_tree",
        payload_bytes=payload_bytes,
        partitions=(PayloadPartition("whole", 0, payload_bytes),),
        exchanges=(
            MatchingExchange(0, "whole", 0, 1),
            MatchingExchange(1, "whole", 1, 3),
        ),
    )


def counter_rotating_allreduce_schedule(
    payload_bytes: int,
) -> Tp4AllreduceSchedule:
    """Return two tensor halves traversing opposite TP4 matching orders."""

    if payload_bytes <= 0 or payload_bytes % 2 != 0:
        raise ValueError("payload_bytes must split into two nonempty halves")
    half = payload_bytes // 2
    return Tp4AllreduceSchedule(
        name="counter_rotating_halves",
        payload_bytes=payload_bytes,
        partitions=(
            PayloadPartition("lower", 0, half),
            PayloadPartition("upper", half, half),
        ),
        exchanges=(
            MatchingExchange(0, "lower", 0, 1),
            MatchingExchange(0, "upper", 1, 3),
            MatchingExchange(1, "lower", 1, 3),
            MatchingExchange(1, "upper", 0, 1),
        ),
    )


def _ring_endpoint(rank: int, direction: int) -> int:
    if direction not in (-1, 1) or rank not in range(4):
        raise ValueError("TP4 ring direction/rank is invalid")
    neighbor = (rank + direction) % 4
    if neighbor == rank ^ 1:
        return 0
    if neighbor == rank ^ 3:
        return 1
    raise ValueError("TP4 ring neighbor is not directly connected")


def bidirectional_ring_allreduce_schedule(
    payload_bytes: int,
) -> BidirectionalRingAllreduceSchedule:
    """Build concurrent clockwise/counter-clockwise RS+AG rings.

    Each tensor half has four equal rank-owned shards. Three reduce-scatter
    stages and three all-gather stages move one shard per direction and rank.
    """

    if payload_bytes <= 0 or payload_bytes % 8 != 0:
        raise ValueError(
            "bidirectional ring payload must split into eight chunks"
        )
    chunk_bytes = payload_bytes // 8
    transfers: list[BidirectionalRingTransfer] = []
    for direction, half in ((1, 0), (-1, 1)):
        for hop in range(3):
            for rank in range(4):
                transfers.append(
                    BidirectionalRingTransfer(
                        stage=hop,
                        phase="reduce_scatter",
                        direction=direction,
                        sender_rank=rank,
                        receiver_rank=(rank + direction) % 4,
                        endpoint=_ring_endpoint(rank, direction),
                        chunk=BidirectionalRingChunk(
                            half=half,
                            shard=(rank - direction * hop) % 4,
                        ),
                    )
                )
        for hop in range(3):
            for rank in range(4):
                transfers.append(
                    BidirectionalRingTransfer(
                        stage=3 + hop,
                        phase="all_gather",
                        direction=direction,
                        sender_rank=rank,
                        receiver_rank=(rank + direction) % 4,
                        endpoint=_ring_endpoint(rank, direction),
                        chunk=BidirectionalRingChunk(
                            half=half,
                            shard=(rank + direction - direction * hop) % 4,
                        ),
                    )
                )
    return BidirectionalRingAllreduceSchedule(
        payload_bytes=payload_bytes,
        chunk_bytes=chunk_bytes,
        transfers=tuple(transfers),
    )


def verify_bidirectional_ring_allreduce_schedule(
    schedule: BidirectionalRingAllreduceSchedule,
) -> BidirectionalRingAllreduceVerification:
    """Symbolically execute a TP4 bidirectional ring all-reduce."""

    if (
        schedule.payload_bytes <= 0
        or schedule.payload_bytes % 8 != 0
        or schedule.chunk_bytes != schedule.payload_bytes // 8
    ):
        raise ValueError("invalid bidirectional ring payload geometry")
    chunks = tuple(
        BidirectionalRingChunk(half, shard)
        for half in range(2)
        for shard in range(4)
    )
    holdings = [
        {chunk: 1 << rank for chunk in chunks}
        for rank in range(4)
    ]
    edge_bytes = [[0, 0] for _ in range(4)]
    stages = sorted({transfer.stage for transfer in schedule.transfers})
    if stages != list(range(6)):
        raise ValueError("bidirectional ring requires exactly six stages")
    critical_bytes = 0
    for stage in stages:
        before = [dict(values) for values in holdings]
        used: set[tuple[int, int]] = set()
        stage_edge_bytes = [[0, 0] for _ in range(4)]
        for transfer in (
            item for item in schedule.transfers if item.stage == stage
        ):
            if transfer.sender_rank not in range(4):
                raise ValueError("TP4 sender rank must be in [0, 3]")
            if transfer.direction not in (-1, 1):
                raise ValueError("TP4 direction must be -1 or +1")
            expected_receiver = (
                transfer.sender_rank + transfer.direction
            ) % 4
            if transfer.receiver_rank != expected_receiver:
                raise ValueError("ring transfer does not reach its neighbor")
            expected_endpoint = _ring_endpoint(
                transfer.sender_rank, transfer.direction
            )
            if transfer.endpoint != expected_endpoint:
                raise ValueError("ring transfer uses the wrong TP4 endpoint")
            key = (transfer.sender_rank, transfer.endpoint)
            if key in used:
                raise ValueError("one endpoint has two transfers in one stage")
            used.add(key)
            if transfer.chunk not in before[transfer.sender_rank]:
                raise ValueError("ring transfer sends an unavailable chunk")
            source = before[transfer.sender_rank][transfer.chunk]
            if transfer.phase == "reduce_scatter":
                local = before[transfer.receiver_rank][transfer.chunk]
                holdings[transfer.receiver_rank][transfer.chunk] = (
                    local | source
                )
            elif transfer.phase == "all_gather":
                if source != 15:
                    raise ValueError("all-gather sent an incompletely reduced chunk")
                holdings[transfer.receiver_rank][transfer.chunk] = source
            else:
                raise ValueError("unknown ring all-reduce phase")
            edge_bytes[transfer.sender_rank][transfer.endpoint] += (
                schedule.chunk_bytes
            )
            stage_edge_bytes[transfer.sender_rank][transfer.endpoint] += (
                schedule.chunk_bytes
            )
        if len(used) != 8:
            raise ValueError("each stage must use both endpoints on every rank")
        critical_bytes += max(
            max(rank_bytes) for rank_bytes in stage_edge_bytes
        )

    masks = tuple(
        tuple(holdings[rank][chunk] for chunk in chunks)
        for rank in range(4)
    )
    edge_tuple = tuple(tuple(values) for values in edge_bytes)
    return BidirectionalRingAllreduceVerification(
        complete=all(mask == 15 for values in masks for mask in values),
        stage_count=len(stages),
        contributor_masks_by_rank=masks,
        edge_bytes_by_rank=edge_tuple,  # type: ignore[arg-type]
        total_bytes_per_rank=sum(edge_tuple[0]),
        ideal_critical_path_bytes_per_rank=critical_bytes,
    )


def verify_allreduce_schedule(
    schedule: Tp4AllreduceSchedule,
) -> AllreduceScheduleVerification:
    """Prove TP4 contributor closure and account bytes on both endpoints."""

    if schedule.payload_bytes <= 0:
        raise ValueError("schedule payload_bytes must be positive")
    partitions = sorted(schedule.partitions, key=lambda item: item.offset_bytes)
    expected_offset = 0
    by_name: dict[str, PayloadPartition] = {}
    for partition in partitions:
        if not partition.name or partition.name in by_name:
            raise ValueError("payload partition names must be unique and nonempty")
        if partition.offset_bytes != expected_offset or partition.active_bytes <= 0:
            raise ValueError("payload partitions must exactly and contiguously cover bytes")
        by_name[partition.name] = partition
        expected_offset += partition.active_bytes
    if expected_offset != schedule.payload_bytes:
        raise ValueError("payload partitions must exactly and contiguously cover bytes")

    stages = sorted({exchange.stage for exchange in schedule.exchanges})
    if stages != list(range(len(stages))):
        raise ValueError("schedule stages must be contiguous from zero")

    contributors = {
        partition.name: [1 << rank for rank in range(4)]
        for partition in partitions
    }
    expressions = {
        partition.name: [f"r{rank}" for rank in range(4)]
        for partition in partitions
    }
    edge_bytes = [0, 0]
    stage_edge_bytes: list[tuple[int, int]] = []
    for stage in stages:
        stage_exchanges = tuple(
            exchange
            for exchange in schedule.exchanges
            if exchange.stage == stage
        )
        used_endpoints: set[int] = set()
        used_partitions: set[str] = set()
        stage_bytes = [0, 0]
        before = {name: values.copy() for name, values in contributors.items()}
        expressions_before = {
            name: values.copy() for name, values in expressions.items()
        }
        for exchange in stage_exchanges:
            if exchange.partition not in by_name:
                raise ValueError(
                    f"exchange names unknown partition {exchange.partition!r}"
                )
            if exchange.endpoint not in (0, 1):
                raise ValueError("TP4 endpoint must be zero or one")
            expected_mask = 1 if exchange.endpoint == 0 else 3
            if exchange.matching_mask != expected_mask:
                raise ValueError(
                    "TP4 endpoint/matching mismatch: "
                    f"endpoint {exchange.endpoint} requires XOR-{expected_mask}"
                )
            if exchange.endpoint in used_endpoints:
                raise ValueError("one endpoint cannot carry two exchanges in one stage")
            if exchange.partition in used_partitions:
                raise ValueError("one partition cannot be exchanged twice in one stage")
            used_endpoints.add(exchange.endpoint)
            used_partitions.add(exchange.partition)
            partition_bytes = by_name[exchange.partition].active_bytes
            stage_bytes[exchange.endpoint] += partition_bytes
            edge_bytes[exchange.endpoint] += partition_bytes
            source = before[exchange.partition]
            contributors[exchange.partition] = [
                source[rank] | source[rank ^ exchange.matching_mask]
                for rank in range(4)
            ]
            source_expressions = expressions_before[exchange.partition]
            expressions[exchange.partition] = [
                "(" + "+".join(
                    sorted(
                        (
                            source_expressions[rank],
                            source_expressions[
                                rank ^ exchange.matching_mask
                            ],
                        )
                    )
                ) + ")"
                for rank in range(4)
            ]
        stage_edge_bytes.append((stage_bytes[0], stage_bytes[1]))

    expected_contributors = (1 << 4) - 1
    masks = {
        name: tuple(values)
        for name, values in contributors.items()
    }
    fingerprints = {
        name: tuple(values)
        for name, values in expressions.items()
    }
    complete = all(
        mask == expected_contributors
        for values in masks.values()
        for mask in values
    )
    return AllreduceScheduleVerification(
        complete=complete,
        contributor_masks=masks,  # type: ignore[arg-type]
        association_fingerprints=fingerprints,  # type: ignore[arg-type]
        edge_bytes_per_rank=(edge_bytes[0], edge_bytes[1]),
        stage_edge_bytes_per_rank=tuple(stage_edge_bytes),
        ideal_critical_path_bytes_per_rank=sum(
            max(per_endpoint) for per_endpoint in stage_edge_bytes
        ),
    )


class TiledCapacityPlanner:
    """Map arbitrary bounded widths onto a small set of stable sessions."""

    def __init__(
        self,
        *,
        family: str,
        collective: str,
        dtype: str,
        reduction_algebra: str,
        bytes_per_query_row: int,
        capacity_classes: tuple[CapacityClass, ...],
        tile_bytes: int,
        slots_per_edge: int,
    ) -> None:
        for name, value in (
            ("family", family),
            ("collective", collective),
            ("dtype", dtype),
            ("reduction_algebra", reduction_algebra),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        for name, value in (
            ("bytes_per_query_row", bytes_per_query_row),
            ("tile_bytes", tile_bytes),
            ("slots_per_edge", slots_per_edge),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not capacity_classes:
            raise ValueError("planner requires at least one capacity class")
        prior_maximum = 0
        names: set[str] = set()
        for capacity in capacity_classes:
            if not isinstance(capacity, CapacityClass):
                raise ValueError("capacity_classes must contain CapacityClass values")
            if not capacity.name or capacity.name in names:
                raise ValueError("capacity class names must be unique and nonempty")
            if (
                isinstance(capacity.maximum_query_rows, bool)
                or not isinstance(capacity.maximum_query_rows, int)
                or capacity.maximum_query_rows <= prior_maximum
            ):
                raise ValueError(
                    "capacity maxima must be positive and strictly increasing"
                )
            names.add(capacity.name)
            prior_maximum = capacity.maximum_query_rows
        self._family = family
        self._collective = collective
        self._dtype = dtype
        self._reduction_algebra = reduction_algebra
        self._bytes_per_query_row = bytes_per_query_row
        self._capacity_classes = capacity_classes
        self._tile_bytes = tile_bytes
        self._slots_per_edge = slots_per_edge

    @classmethod
    def tp4_bf16_allreduce(
        cls,
        *,
        elements_per_query_row: int = 6144,
    ) -> TiledCapacityPlanner:
        """Return a bounded Q1-Q4096 contiguous BF16 TP4 planner."""

        if (
            isinstance(elements_per_query_row, bool)
            or not isinstance(elements_per_query_row, int)
            or elements_per_query_row <= 0
        ):
            raise ValueError(
                "elements_per_query_row must be a positive integer"
            )

        return cls(
            family=f"tp4_allreduce_bf16_{elements_per_query_row}",
            collective="all_reduce",
            dtype="bf16",
            reduction_algebra="sum",
            bytes_per_query_row=elements_per_query_row * 2,
            capacity_classes=(
                CapacityClass("latency_q40", 40),
                CapacityClass("medium_q512", 512),
                CapacityClass("streaming_q4096", 4096),
                CapacityClass("extended_q8192", 8192),
            ),
            tile_bytes=512 * 1024,
            slots_per_edge=8,
        )

    def plan(
        self,
        query_rows: int,
        *,
        first_tile_ordinal: int = 0,
    ) -> OperationPlan:
        """Return the smallest stable capacity class containing ``query_rows``."""

        if isinstance(query_rows, bool) or not isinstance(query_rows, int):
            raise ValueError("query_rows must be an integer")
        if (
            isinstance(first_tile_ordinal, bool)
            or not isinstance(first_tile_ordinal, int)
            or first_tile_ordinal < 0
        ):
            raise ValueError("first_tile_ordinal must be a nonnegative integer")
        for capacity in self._capacity_classes:
            if 1 <= query_rows <= capacity.maximum_query_rows:
                active_bytes = query_rows * self._bytes_per_query_row
                tile_count = (
                    active_bytes + self._tile_bytes - 1
                ) // self._tile_bytes
                tiles = tuple(
                    TileDescriptor(
                        ticket=TileTicket.from_ordinal(
                            first_tile_ordinal + index,
                            self._slots_per_edge,
                        ),
                        offset_bytes=index * self._tile_bytes,
                        active_bytes=min(
                            self._tile_bytes,
                            active_bytes - index * self._tile_bytes,
                        ),
                    )
                    for index in range(tile_count)
                )
                return OperationPlan(
                    family=self._family,
                    collective=self._collective,
                    dtype=self._dtype,
                    reduction_algebra=self._reduction_algebra,
                    query_rows=query_rows,
                    active_bytes=active_bytes,
                    capacity_class=capacity.name,
                    capacity_query_rows=capacity.maximum_query_rows,
                    session_key=TiledSessionKey(
                        family=self._family,
                        collective=self._collective,
                        dtype=self._dtype,
                        reduction_algebra=self._reduction_algebra,
                        bytes_per_query_row=self._bytes_per_query_row,
                        capacity_class=capacity.name,
                        capacity_query_rows=capacity.maximum_query_rows,
                        tile_bytes=self._tile_bytes,
                        slots_per_edge=self._slots_per_edge,
                    ),
                    tile_bytes=self._tile_bytes,
                    slots_per_edge=self._slots_per_edge,
                    tiles=tiles,
                )
        maximum = self._capacity_classes[-1].maximum_query_rows
        raise ValueError(f"query_rows must be in [1, {maximum}]")
