"""Pure block-page codec for heterogeneous KV-cache groups.

The codec treats every layer page as opaque bytes. Group block tables define
logical ordering; layer names, dtypes, page shapes, bytes-per-page, and the
semantic reuse window define the layout. This preserves compressed attention
and recurrent compressor state without assigning misleading per-token
geometry to either one.
"""

from __future__ import annotations

import hashlib
import zlib
import json
import struct
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

_MAGIC = b"SPHP1\x00"
_DELTA_MAGIC = b"SPHD1\x00"
_HEADER_LENGTH = struct.Struct("<I")
_DELTA_HEADER_SCHEMA_V1 = "sparkcache-hybrid-page-delta/v1"
_DELTA_HEADER_SCHEMA_V2 = "sparkcache-hybrid-page-delta/v2"

# Payload integrity modes. ``sha256`` is the content-addressed format
# (publication schema ``page-tail-cow-v2``): every payload byte hashes to a
# digest bound by authenticated metadata. ``fast`` and ``none`` are the
# ``page-tail-cow-v3-*`` schemas: identities, manifests and stage headers stay
# cryptographic, bulk payloads carry a CRC32 per object and per layer tail
# (``fast``) or no check beyond length and geometry (``none``), and the
# digest fields of headers and manifests hold boundary tokens (digests of the
# boundary's identity, not of its bytes).
PAYLOAD_INTEGRITY_MODES = ("sha256", "fast", "none")
FLAT_PAGE_SCHEMAS = ("page-tail-cow-v2", "page-tail-cow-v3-fast", "page-tail-cow-v3-none")

# Page groups the codec never persists. vLLM rebuilds their state at every
# persistent boundary: transient groups hold in-flight compression rings,
# replay groups hold request-private prefill state valid only for the last
# replay-reserve rows. Snapshots must give them zero pages.
NON_PERSISTED_REUSE_POLICIES = frozenset({"transient", "replay"})


def check_group_block_policies(
    layout: PageLayout,
    counts: Sequence[int],
) -> None:
    """Enforce each group's persistence rule on parsed per-group page counts.

    Persisted groups must contribute at least one page. Transient and replay
    groups contribute exactly zero pages, so a store publishes none of their
    bytes and a restore writes none back.
    """

    for group, count in zip(layout.groups, counts, strict=True):
        if group.reuse_policy in NON_PERSISTED_REUSE_POLICIES:
            if count != 0:
                raise HybridCodecError(
                    f"{group.reuse_policy} page group must contribute zero blocks"
                )
        elif count <= 0:
            raise HybridCodecError(
                "every page group must contribute at least one block"
            )


def payload_integrity_for_schema(publication_schema: str) -> str:
    """Return the payload integrity mode a publication schema implies."""

    if publication_schema == "page-tail-cow-v3-fast":
        return "fast"
    if publication_schema == "page-tail-cow-v3-none":
        return "none"
    return "sha256"


def boundary_token(context_digest: str, committed_tokens: int) -> str:
    """Digest naming one published boundary without hashing its bytes."""

    return hashlib.sha256(
        b"sparkcache-boundary-token/v1\x00"
        + f"{context_digest}:{int(committed_tokens)}".encode("ascii")
    ).hexdigest()


def payload_checksum(data) -> int:
    """The ``fast`` payload checksum: CRC32 as zlib computes it."""

    return zlib.crc32(data) & 0xFFFFFFFF


def payload_matches(integrity: str, data, *, sha256: str | None, crc32: int | None) -> bool:
    """Check payload bytes under one integrity mode (length is checked by callers)."""

    if integrity == "sha256":
        return hashlib.sha256(data).hexdigest() == sha256
    if integrity == "fast":
        return crc32 is not None and payload_checksum(data) == crc32
    if integrity == "none":
        return True
    raise HybridCodecError("payload integrity mode is unsupported")
class HybridCodecError(ValueError):
    """A block-page snapshot does not match its declared layout."""


class PageDeltaPayloadChecksumError(HybridCodecError):
    """Stored delta tail bytes differ from their authenticated header digest."""


@dataclass(frozen=True)
class PageLayer:
    name: str
    dtype: str
    page_shape: tuple[int, ...]
    bytes_per_page: int

    def __post_init__(self) -> None:
        if not self.name or not self.dtype or not self.page_shape:
            raise HybridCodecError("page layer metadata is incomplete")
        if self.bytes_per_page <= 0 or any(size <= 0 for size in self.page_shape):
            raise HybridCodecError("page layer geometry must be positive")


@dataclass(frozen=True)
class PageGroup:
    block_size: int
    layers: tuple[PageLayer, ...]
    reuse_policy: str = "full"
    reuse_window_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.block_size <= 0 or not self.layers:
            raise HybridCodecError("page group geometry is incomplete")
        if self.reuse_policy not in {
            "full",
            "sliding",
            "recurrent_align",
            "transient",
            "replay",
        }:
            raise HybridCodecError("page group reuse policy is unsupported")
        if self.reuse_policy in {"full", "recurrent_align", "transient"} and (
            self.reuse_window_tokens is not None
        ):
            raise HybridCodecError(
                f"{self.reuse_policy} page group cannot declare a reuse window"
            )
        if self.reuse_policy == "sliding" and (
            self.reuse_window_tokens is None or self.reuse_window_tokens <= 1
        ):
            raise HybridCodecError("sliding page group reuse window is invalid")
        if self.reuse_policy == "replay" and (
            self.reuse_window_tokens is None or self.reuse_window_tokens < 1
        ):
            # The window is the request-private replay reserve: the prompt
            # suffix a persistent boundary must leave to recompute.
            raise HybridCodecError("replay page group reuse window is invalid")
        names = [layer.name for layer in self.layers]
        if names != sorted(names) or len(names) != len(set(names)):
            raise HybridCodecError("page group layers must be sorted and unique")


@dataclass(frozen=True)
class PageLayout:
    groups: tuple[PageGroup, ...]

    def __post_init__(self) -> None:
        if not self.groups:
            raise HybridCodecError("page layout has no groups")
        names = [layer.name for group in self.groups for layer in group.layers]
        if len(names) != len(set(names)):
            raise HybridCodecError("a layer belongs to multiple page groups")

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PagePayloadSpan:
    layer_name: str
    group_index: int
    source_start: int
    source_end: int
    page_count: int
    bytes_per_page: int


@dataclass(frozen=True)
class PageSnapshotPlan:
    header_bytes: int
    total_bytes: int
    block_counts: tuple[int, ...]
    spans: tuple[PagePayloadSpan, ...]


@dataclass(frozen=True)
class PageDeltaTail:
    """One authenticated delta payload extent and its final layer offset."""

    layer_name: str
    group_index: int
    source_start: int
    source_end: int
    destination_byte_offset: int
    sha256: str | None
    crc32: int | None = None


@dataclass(frozen=True)
class PageDeltaPlan:
    """Validated delta header geometry without materializing its base.

    Under ``integrity`` ``sha256`` the two snapshot digests are SHA-256 of the
    base and result bytes; under ``fast`` and ``none`` they are boundary
    tokens and the tails carry CRC32 values instead of digests.
    """

    header_bytes: int
    total_bytes: int
    base_block_counts: tuple[int, ...]
    result_block_counts: tuple[int, ...]
    base_boundary_tokens: int
    result_boundary_tokens: int
    base_snapshot_sha256: str
    result_snapshot_sha256: str
    reused_pages_by_group: tuple[int, ...]
    tails: tuple[PageDeltaTail, ...]
    integrity: str = "sha256"


@dataclass(frozen=True)
class _VerifiedPageSnapshot:
    """Process-local immutable bytes with their verified full-snapshot digest.

    Only hashing, authenticated object reads, and delta helpers construct it.
    It is never accepted from persistent metadata or a public restore caller.
    """

    payload: bytes
    sha256: str


def _verify_page_snapshot_bytes(
    payload: bytes | bytearray | memoryview | _VerifiedPageSnapshot,
) -> _VerifiedPageSnapshot:
    if isinstance(payload, _VerifiedPageSnapshot):
        return payload
    immutable = bytes(payload)
    return _VerifiedPageSnapshot(immutable, hashlib.sha256(immutable).hexdigest())


def page_snapshot_encoded_size(
    layout: PageLayout,
    block_counts: Sequence[int],
) -> int:
    """Return the exact encoded size without allocating page payload bytes."""

    header = encode_page_snapshot_header(layout, block_counts)
    counts = tuple(int(count) for count in block_counts)
    payload_bytes = sum(
        count * layer.bytes_per_page
        for group, count in zip(layout.groups, counts, strict=True)
        for layer in group.layers
    )
    return len(header) + payload_bytes


def encode_page_snapshot_header(
    layout: PageLayout,
    block_counts: Sequence[int],
) -> bytes:
    """Encode only the deterministic SPHP1 header for one page snapshot."""

    if len(block_counts) != len(layout.groups):
        raise HybridCodecError("block-count vector disagrees with page groups")
    counts = tuple(int(count) for count in block_counts)
    check_group_block_policies(layout, counts)
    header = json.dumps(
        {
            "schema": "sparkcache-hybrid-pages/v1",
            "layout_sha256": layout.digest,
            "block_counts": counts,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _MAGIC + _HEADER_LENGTH.pack(len(header)) + header


def plan_page_snapshot(
    layout: PageLayout,
    encoded_prefix: bytes | bytearray | memoryview,
    expected_block_counts: Sequence[int],
    *,
    total_bytes: int | None = None,
) -> PageSnapshotPlan:
    """Validate the small SPHP1 header and describe payload byte extents.

    ``encoded_prefix`` may be the complete snapshot or only the header bytes.
    SparkCache CUDA restore uses the latter after authenticating the containing .spcc
    object, avoiding a Python slice for every layer payload.
    """

    prefix = len(_MAGIC) + _HEADER_LENGTH.size
    view = memoryview(encoded_prefix).cast("B")
    if len(view) < prefix or bytes(view[: len(_MAGIC)]) != _MAGIC:
        raise HybridCodecError("hybrid page snapshot has an invalid prefix")
    (header_length,) = _HEADER_LENGTH.unpack_from(view, len(_MAGIC))
    header_end = prefix + header_length
    if header_length <= 0 or header_end > len(view):
        raise HybridCodecError("hybrid page snapshot header is truncated")
    try:
        header = json.loads(bytes(view[prefix:header_end]))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HybridCodecError("hybrid page snapshot header is invalid") from error
    try:
        counts = tuple(int(count) for count in expected_block_counts)
    except (TypeError, ValueError) as error:
        raise HybridCodecError("hybrid page block counts are invalid") from error
    if (
        not isinstance(header, dict)
        or header.get("schema") != "sparkcache-hybrid-pages/v1"
        or header.get("layout_sha256") != layout.digest
        or tuple(header.get("block_counts", ())) != counts
        or len(counts) != len(layout.groups)
    ):
        raise HybridCodecError("hybrid page snapshot identity or block counts differ")
    check_group_block_policies(layout, counts)
    if total_bytes is None:
        total = len(view)
    elif isinstance(total_bytes, bool) or not isinstance(total_bytes, int):
        raise HybridCodecError("hybrid page total bytes must be an integer")
    else:
        total = total_bytes
    if total < header_end or len(view) > total:
        raise HybridCodecError("hybrid page snapshot total length is invalid")

    spans = []
    offset = header_end
    for group_index, (group, count) in enumerate(zip(layout.groups, counts)):
        for layer in group.layers:
            size = count * layer.bytes_per_page
            end = offset + size
            if end > total:
                raise HybridCodecError(
                    f"hybrid page snapshot is truncated at {layer.name}"
                )
            spans.append(
                PagePayloadSpan(
                    layer_name=layer.name,
                    group_index=group_index,
                    source_start=offset,
                    source_end=end,
                    page_count=count,
                    bytes_per_page=layer.bytes_per_page,
                )
            )
            offset = end
    if offset != total:
        raise HybridCodecError("hybrid page snapshot has trailing bytes")
    return PageSnapshotPlan(
        header_bytes=header_end,
        total_bytes=total,
        block_counts=counts,
        spans=tuple(spans),
    )


def encode_page_snapshot(
    layout: PageLayout,
    block_counts: Sequence[int],
    layer_payloads: Mapping[str, bytes],
) -> bytes:
    if len(block_counts) != len(layout.groups):
        raise HybridCodecError("block-count vector disagrees with page groups")
    counts = tuple(int(count) for count in block_counts)
    check_group_block_policies(layout, counts)
    expected_names = {layer.name for group in layout.groups for layer in group.layers}
    if set(layer_payloads) != expected_names:
        raise HybridCodecError("page payload names disagree with layout")

    parts: list[bytes] = []
    for group, count in zip(layout.groups, counts):
        for layer in group.layers:
            payload = layer_payloads[layer.name]
            expected = count * layer.bytes_per_page
            if len(payload) != expected:
                raise HybridCodecError(
                    f"layer {layer.name} carries {len(payload)} bytes, expected {expected}"
                )
            parts.append(payload)
    return b"".join((encode_page_snapshot_header(layout, counts), *parts))


def decode_page_snapshot(
    layout: PageLayout,
    encoded: bytes,
    expected_block_counts: Sequence[int],
) -> dict[str, bytes]:
    plan = plan_page_snapshot(layout, encoded, expected_block_counts)
    return {
        span.layer_name: encoded[span.source_start : span.source_end]
        for span in plan.spans
    }


def _validate_delta_boundaries(
    base_boundary_tokens: int,
    result_boundary_tokens: int,
) -> None:
    if (
        isinstance(base_boundary_tokens, bool)
        or isinstance(result_boundary_tokens, bool)
        or not isinstance(base_boundary_tokens, int)
        or not isinstance(result_boundary_tokens, int)
        or base_boundary_tokens <= 0
        or result_boundary_tokens <= base_boundary_tokens
    ):
        raise HybridCodecError("page delta boundaries are invalid")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def plan_page_delta(
    layout: PageLayout,
    encoded_prefix: bytes | bytearray | memoryview,
    *,
    base_block_counts: Sequence[int],
    result_block_counts: Sequence[int],
    base_boundary_tokens: int,
    result_boundary_tokens: int,
    total_bytes: int | None = None,
) -> PageDeltaPlan:
    """Validate a delta header and describe its ordered layer-tail extents.

    ``encoded_prefix`` may contain only the complete header. Callers that
    authenticate macro objects can therefore plan source ranges without
    assembling the complete delta or reading its base snapshot.
    """

    _validate_delta_boundaries(base_boundary_tokens, result_boundary_tokens)
    prefix_bytes = len(_DELTA_MAGIC) + _HEADER_LENGTH.size
    view = memoryview(encoded_prefix).cast("B")
    if len(view) < prefix_bytes or bytes(view[: len(_DELTA_MAGIC)]) != _DELTA_MAGIC:
        raise HybridCodecError("hybrid page delta has an invalid prefix")
    (header_length,) = _HEADER_LENGTH.unpack_from(view, len(_DELTA_MAGIC))
    header_end = prefix_bytes + header_length
    if header_length <= 0 or header_end > len(view):
        raise HybridCodecError("hybrid page delta header is truncated")
    try:
        header = json.loads(bytes(view[prefix_bytes:header_end]))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HybridCodecError("hybrid page delta header is invalid") from error
    expected_keys = {
        "schema",
        "layout_sha256",
        "base_snapshot_sha256",
        "result_snapshot_sha256",
        "base_block_counts",
        "result_block_counts",
        "base_boundary_tokens",
        "result_boundary_tokens",
        "reused_pages_by_group",
        "layer_tails",
    }
    if not isinstance(header, dict):
        raise HybridCodecError("hybrid page delta header fields differ")
    if header.get("schema") == _DELTA_HEADER_SCHEMA_V2:
        expected_keys = expected_keys | {"integrity"}
        integrity = header.get("integrity")
        if integrity not in ("fast", "none"):
            raise HybridCodecError("hybrid page delta integrity mode is invalid")
    else:
        integrity = "sha256"
    if set(header) != expected_keys:
        raise HybridCodecError("hybrid page delta header fields differ")
    raw_base_counts = header["base_block_counts"]
    raw_result_counts = header["result_block_counts"]
    if (
        not isinstance(raw_base_counts, list)
        or not isinstance(raw_result_counts, list)
        or any(type(value) is not int for value in raw_base_counts)
        or any(type(value) is not int for value in raw_result_counts)
        or type(header["base_boundary_tokens"]) is not int
        or type(header["result_boundary_tokens"]) is not int
    ):
        raise HybridCodecError("hybrid page delta header geometry is invalid")
    try:
        base_counts = tuple(int(value) for value in base_block_counts)
        result_counts = tuple(int(value) for value in result_block_counts)
    except (TypeError, ValueError) as error:
        raise HybridCodecError("hybrid page delta block counts are invalid") from error
    if (
        len(base_counts) != len(layout.groups)
        or len(result_counts) != len(layout.groups)
        or any(
            result < base
            for base, result in zip(base_counts, result_counts, strict=True)
        )
    ):
        raise HybridCodecError("page delta block counts do not extend the base")
    check_group_block_policies(layout, base_counts)
    check_group_block_policies(layout, result_counts)
    if (
        header["schema"] not in (_DELTA_HEADER_SCHEMA_V1, _DELTA_HEADER_SCHEMA_V2)
        or header["layout_sha256"] != layout.digest
        or tuple(raw_base_counts) != base_counts
        or tuple(raw_result_counts) != result_counts
        or header["base_boundary_tokens"] != base_boundary_tokens
        or header["result_boundary_tokens"] != result_boundary_tokens
        or not _is_sha256(header["base_snapshot_sha256"])
        or not _is_sha256(header["result_snapshot_sha256"])
    ):
        raise HybridCodecError("hybrid page delta identity or base differs")
    reused = header["reused_pages_by_group"]
    raw_tails = header["layer_tails"]
    expected_layers = [
        (group_index, layer)
        for group_index, group in enumerate(layout.groups)
        for layer in group.layers
    ]
    if (
        not isinstance(reused, list)
        or len(reused) != len(layout.groups)
        or not isinstance(raw_tails, list)
        or len(raw_tails) != len(expected_layers)
    ):
        raise HybridCodecError("hybrid page delta descriptor count differs")
    if total_bytes is None:
        total = len(view)
    elif isinstance(total_bytes, bool) or not isinstance(total_bytes, int):
        raise HybridCodecError("hybrid page delta total bytes must be an integer")
    else:
        total = total_bytes
    if total < header_end or len(view) > total:
        raise HybridCodecError("hybrid page delta total length is invalid")

    tails: list[PageDeltaTail] = []
    source_offset = header_end
    layer_index = 0
    for group_index, group in enumerate(layout.groups):
        reused_pages = reused[group_index]
        if (
            type(reused_pages) is not int
            or not 0 <= reused_pages <= base_counts[group_index]
        ):
            raise HybridCodecError("hybrid page delta reuse count is invalid")
        for layer in group.layers:
            descriptor = raw_tails[layer_index]
            layer_index += 1
            expected_bytes = (
                result_counts[group_index] - reused_pages
            ) * layer.bytes_per_page
            if integrity == "sha256":
                valid = (
                    isinstance(descriptor, dict)
                    and set(descriptor) == {"name", "bytes", "sha256"}
                    and _is_sha256(descriptor["sha256"])
                )
            else:
                valid = (
                    isinstance(descriptor, dict)
                    and set(descriptor) == {"name", "bytes", "crc32"}
                    and (
                        descriptor["crc32"] is None
                        if integrity == "none"
                        else type(descriptor["crc32"]) is int
                        and 0 <= descriptor["crc32"] <= 0xFFFFFFFF
                    )
                )
            if (
                not valid
                or descriptor["name"] != layer.name
                or type(descriptor["bytes"]) is not int
                or descriptor["bytes"] != expected_bytes
            ):
                raise HybridCodecError("hybrid page delta layer descriptor differs")
            source_end = source_offset + expected_bytes
            if source_end > total:
                raise HybridCodecError("hybrid page delta payload is truncated")
            tails.append(
                PageDeltaTail(
                    layer_name=layer.name,
                    group_index=group_index,
                    source_start=source_offset,
                    source_end=source_end,
                    destination_byte_offset=reused_pages * layer.bytes_per_page,
                    sha256=descriptor.get("sha256"),
                    crc32=descriptor.get("crc32"),
                )
            )
            source_offset = source_end
    if source_offset != total:
        raise HybridCodecError("hybrid page delta has trailing bytes")
    return PageDeltaPlan(
        header_bytes=header_end,
        total_bytes=total,
        base_block_counts=base_counts,
        result_block_counts=result_counts,
        base_boundary_tokens=base_boundary_tokens,
        result_boundary_tokens=result_boundary_tokens,
        base_snapshot_sha256=header["base_snapshot_sha256"],
        result_snapshot_sha256=header["result_snapshot_sha256"],
        reused_pages_by_group=tuple(reused),
        tails=tuple(tails),
        integrity=integrity,
    )


def _tail_descriptor(integrity: str, name: str, tail: bytes) -> dict[str, object]:
    if integrity == "sha256":
        return {"name": name, "bytes": len(tail), "sha256": hashlib.sha256(tail).hexdigest()}
    return {
        "name": name,
        "bytes": len(tail),
        "crc32": payload_checksum(tail) if integrity == "fast" else None,
    }


def _delta_header_fields(integrity: str) -> dict[str, object]:
    if integrity == "sha256":
        return {"schema": _DELTA_HEADER_SCHEMA_V1}
    return {"schema": _DELTA_HEADER_SCHEMA_V2, "integrity": integrity}


def encode_page_delta(
    layout: PageLayout,
    base_snapshot: bytes,
    result_snapshot: bytes,
    *,
    base_block_counts: Sequence[int],
    result_block_counts: Sequence[int],
    base_boundary_tokens: int,
    result_boundary_tokens: int,
    base_snapshot_sha256: str | None = None,
    integrity: str = "sha256",
    result_snapshot_sha256: str | None = None,
) -> bytes:
    """Encode changed page suffixes relative to one authenticated snapshot.

    Reuse is established page-by-page across every layer in a page group. A
    group reuses a page only when the result carries byte-identical opaque
    state at the same logical page index. When the base boundary lies inside
    a page, changed bytes make that complete terminal page part of the delta;
    preceding byte-identical pages remain reusable. The base snapshot digest
    and both semantic boundaries are bound into the delta header.

    ``base_snapshot_sha256`` is the digest the caller already verified the
    base bytes against; when given, the base is not hashed again.
    """

    _validate_delta_boundaries(base_boundary_tokens, result_boundary_tokens)
    if integrity not in PAYLOAD_INTEGRITY_MODES:
        raise HybridCodecError("payload integrity mode is unsupported")
    if integrity != "sha256" and not (
        _is_sha256(base_snapshot_sha256) and _is_sha256(result_snapshot_sha256)
    ):
        raise HybridCodecError("boundary tokens are required without payload digests")
    if base_snapshot_sha256 is None:
        base_snapshot_sha256 = hashlib.sha256(base_snapshot).hexdigest()
    elif not _is_sha256(base_snapshot_sha256):
        raise HybridCodecError("base snapshot digest is invalid")
    base_counts = tuple(int(value) for value in base_block_counts)
    result_counts = tuple(int(value) for value in result_block_counts)
    if len(base_counts) != len(layout.groups) or len(result_counts) != len(
        layout.groups
    ):
        raise HybridCodecError("page delta block counts disagree with layout")
    if any(
        result < base for base, result in zip(base_counts, result_counts)
    ):
        raise HybridCodecError("page delta block counts do not extend the base")
    check_group_block_policies(layout, base_counts)
    check_group_block_policies(layout, result_counts)
    base_payloads = decode_page_snapshot(layout, base_snapshot, base_counts)
    result_payloads = decode_page_snapshot(layout, result_snapshot, result_counts)

    reused_by_group: list[int] = []
    payload_parts: list[bytes] = []
    layer_tails: list[dict[str, object]] = []
    for group_index, group in enumerate(layout.groups):
        reusable = 0
        for page_index in range(base_counts[group_index]):
            if all(
                base_payloads[layer.name][
                    page_index * layer.bytes_per_page : (page_index + 1)
                    * layer.bytes_per_page
                ]
                == result_payloads[layer.name][
                    page_index * layer.bytes_per_page : (page_index + 1)
                    * layer.bytes_per_page
                ]
                for layer in group.layers
            ):
                reusable += 1
                continue
            break
        reused_by_group.append(reusable)
        for layer in group.layers:
            tail = result_payloads[layer.name][reusable * layer.bytes_per_page :]
            payload_parts.append(tail)
            layer_tails.append(_tail_descriptor(integrity, layer.name, tail))
    if integrity == "sha256":
        result_snapshot_sha256 = hashlib.sha256(result_snapshot).hexdigest()
    header = json.dumps(
        {
            **_delta_header_fields(integrity),
            "layout_sha256": layout.digest,
            "base_snapshot_sha256": base_snapshot_sha256,
            "result_snapshot_sha256": result_snapshot_sha256,
            "base_block_counts": base_counts,
            "result_block_counts": result_counts,
            "base_boundary_tokens": base_boundary_tokens,
            "result_boundary_tokens": result_boundary_tokens,
            "reused_pages_by_group": reused_by_group,
            "layer_tails": layer_tails,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return b"".join(
        (_DELTA_MAGIC, _HEADER_LENGTH.pack(len(header)), header, *payload_parts)
    )


def _encode_page_delta_from_capture(
    layout: PageLayout,
    base_snapshot: bytes | bytearray | memoryview | _VerifiedPageSnapshot,
    captured_tail: object,
    *,
    base_block_counts: Sequence[int],
    result_block_counts: Sequence[int],
    reused_pages_by_group: Sequence[int],
    base_boundary_tokens: int,
    result_boundary_tokens: int,
    base_snapshot_sha256: str | None,
    materialize_result: bool,
    integrity: str = "sha256",
    result_snapshot_sha256: str | None = None,
) -> tuple[bytes, bytearray | None]:
    """Shared encoder behind the two public capture-delta entry points.

    With ``materialize_result`` the complete result snapshot is assembled in
    one buffer and hashed as a whole; otherwise its digest is accumulated
    from the verified base prefixes and captured tails without assembling it.
    """

    _validate_delta_boundaries(base_boundary_tokens, result_boundary_tokens)
    if integrity not in PAYLOAD_INTEGRITY_MODES:
        raise HybridCodecError("payload integrity mode is unsupported")
    if integrity != "sha256" and not (
        _is_sha256(base_snapshot_sha256) and _is_sha256(result_snapshot_sha256)
    ):
        raise HybridCodecError("boundary tokens are required without payload digests")
    try:
        base_counts = tuple(int(value) for value in base_block_counts)
        result_counts = tuple(int(value) for value in result_block_counts)
        reused = tuple(int(value) for value in reused_pages_by_group)
    except (TypeError, ValueError) as error:
        raise HybridCodecError("captured page extension counts are invalid") from error
    if not (
        len(base_counts)
        == len(result_counts)
        == len(reused)
        == len(layout.groups)
    ):
        raise HybridCodecError("captured page extension geometry differs")
    if any(
        result < base
        or not 0 <= shared <= base
        for base, result, shared in zip(
            base_counts, result_counts, reused, strict=True
        )
    ):
        raise HybridCodecError("captured page extension counts are invalid")
    check_group_block_policies(layout, base_counts)
    check_group_block_policies(layout, result_counts)

    if isinstance(base_snapshot, _VerifiedPageSnapshot):
        # The carrier already proves its digest; never hash it again.
        if base_snapshot_sha256 is None:
            base_snapshot_sha256 = base_snapshot.sha256
        base_view = memoryview(base_snapshot.payload).cast("B")
    else:
        base_view = memoryview(base_snapshot).cast("B")
    base_plan = plan_page_snapshot(
        layout,
        base_view,
        base_counts,
        total_bytes=len(base_view),
    )
    total_bytes = getattr(captured_tail, "total_bytes", None)
    read_range = getattr(captured_tail, "read_range", None)
    if isinstance(captured_tail, (bytes, bytearray, memoryview)):
        captured_view = memoryview(captured_tail).cast("B")
        total_bytes = len(captured_view)

        def read_range(start: int, end: int) -> bytes:
            return captured_view[start:end].tobytes()

    if (
        type(total_bytes) is not int
        or total_bytes <= 0
        or not callable(read_range)
    ):
        raise HybridCodecError("captured page extension payload is invalid")

    result_header = encode_page_snapshot_header(layout, result_counts)
    result_buffer: bytearray | None = None
    result_view: memoryview | None = None
    result_cursor = 0
    if materialize_result:
        result_buffer = bytearray(page_snapshot_encoded_size(layout, result_counts))
        result_view = memoryview(result_buffer)
        result_view[: len(result_header)] = result_header
        result_cursor = len(result_header)
        result_digest = None
    elif integrity == "sha256":
        result_digest = hashlib.sha256(result_header)
    else:
        result_digest = None
    payload_parts: list[bytes] = []
    layer_tails: list[dict[str, object]] = []
    cursor = 0
    span_index = 0
    for group_index, group in enumerate(layout.groups):
        shared = reused[group_index]
        tail_pages = result_counts[group_index] - shared
        for layer in group.layers:
            span = base_plan.spans[span_index]
            span_index += 1
            prefix_end = span.source_start + shared * layer.bytes_per_page
            prefix = base_view[span.source_start : prefix_end]
            tail_bytes = tail_pages * layer.bytes_per_page
            end = cursor + tail_bytes
            if end > total_bytes:
                raise HybridCodecError("captured page extension payload is truncated")
            tail = read_range(cursor, end)
            if not isinstance(tail, bytes) or len(tail) != tail_bytes:
                raise HybridCodecError("captured page extension range differs")
            if result_view is not None:
                prefix_bytes = prefix_end - span.source_start
                result_view[result_cursor : result_cursor + prefix_bytes] = prefix
                result_cursor += prefix_bytes
                result_view[result_cursor : result_cursor + tail_bytes] = tail
                result_cursor += tail_bytes
            elif result_digest is not None:
                result_digest.update(prefix)
                result_digest.update(tail)
            payload_parts.append(tail)
            layer_tails.append(_tail_descriptor(integrity, layer.name, tail))
            cursor = end
    if cursor != total_bytes:
        raise HybridCodecError("captured page extension has trailing bytes")
    if result_view is not None:
        if result_cursor != len(result_buffer):
            raise HybridCodecError("captured page extension result is incomplete")
        result_view.release()
    if integrity != "sha256":
        result_sha256 = result_snapshot_sha256
    elif result_view is not None:
        # The retained result is hashed as one buffer, so the digest bound
        # into the header is the digest of exactly the bytes returned.
        result_sha256 = hashlib.sha256(result_buffer).hexdigest()
    else:
        result_sha256 = result_digest.hexdigest()
    if base_snapshot_sha256 is None:
        base_snapshot_sha256 = hashlib.sha256(base_view).hexdigest()
    elif not _is_sha256(base_snapshot_sha256):
        raise HybridCodecError("base snapshot digest is invalid")

    header = json.dumps(
        {
            **_delta_header_fields(integrity),
            "layout_sha256": layout.digest,
            "base_snapshot_sha256": base_snapshot_sha256,
            "result_snapshot_sha256": result_sha256,
            "base_block_counts": base_counts,
            "result_block_counts": result_counts,
            "base_boundary_tokens": base_boundary_tokens,
            "result_boundary_tokens": result_boundary_tokens,
            "reused_pages_by_group": reused,
            "layer_tails": layer_tails,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    delta = b"".join(
        (_DELTA_MAGIC, _HEADER_LENGTH.pack(len(header)), header, *payload_parts)
    )
    return delta, result_buffer


def encode_page_delta_from_capture(
    layout: PageLayout,
    base_snapshot: bytes | bytearray | memoryview,
    captured_tail: object,
    *,
    base_block_counts: Sequence[int],
    result_block_counts: Sequence[int],
    reused_pages_by_group: Sequence[int],
    base_boundary_tokens: int,
    result_boundary_tokens: int,
    base_snapshot_sha256: str | None = None,
    integrity: str = "sha256",
    result_snapshot_sha256: str | None = None,
) -> bytes:
    """Encode an authenticated delta directly from selected captured pages.

    The caller supplies reuse counts established before native capture. The
    captured payload must contain each layer's non-reused result pages in
    layout order. The result snapshot checksum is computed incrementally from
    the verified base prefixes and captured page bytes, so no complete result
    snapshot or page-by-page comparison is required.

    ``base_snapshot_sha256`` is the digest the caller already verified the
    base bytes against. When it is given, the base is not hashed again; the
    caller is responsible for passing the bytes that hashed to it.
    """

    delta, _result = _encode_page_delta_from_capture(
        layout,
        base_snapshot,
        captured_tail,
        base_block_counts=base_block_counts,
        result_block_counts=result_block_counts,
        reused_pages_by_group=reused_pages_by_group,
        base_boundary_tokens=base_boundary_tokens,
        result_boundary_tokens=result_boundary_tokens,
        base_snapshot_sha256=base_snapshot_sha256,
        materialize_result=False,
        integrity=integrity,
        result_snapshot_sha256=result_snapshot_sha256,
    )
    return delta


def encode_page_delta_from_capture_with_result(
    layout: PageLayout,
    base_snapshot: bytes | bytearray | memoryview,
    captured_tail: object,
    *,
    base_block_counts: Sequence[int],
    result_block_counts: Sequence[int],
    reused_pages_by_group: Sequence[int],
    base_boundary_tokens: int,
    result_boundary_tokens: int,
    base_snapshot_sha256: str | None = None,
    integrity: str = "sha256",
    result_snapshot_sha256: str | None = None,
) -> tuple[bytes, bytearray]:
    """Encode a delta as ``encode_page_delta_from_capture`` and keep the result.

    Returns ``(delta, result_snapshot)``. The result snapshot is the complete
    encoded result whose SHA-256 the delta header binds as
    ``result_snapshot_sha256``; the digest was computed over that buffer. The
    caller owns the buffer and must not modify it if it retains it under
    that digest.
    """

    delta, result = _encode_page_delta_from_capture(
        layout,
        base_snapshot,
        captured_tail,
        base_block_counts=base_block_counts,
        result_block_counts=result_block_counts,
        reused_pages_by_group=reused_pages_by_group,
        base_boundary_tokens=base_boundary_tokens,
        result_boundary_tokens=result_boundary_tokens,
        base_snapshot_sha256=base_snapshot_sha256,
        materialize_result=True,
        integrity=integrity,
        result_snapshot_sha256=result_snapshot_sha256,
    )
    assert result is not None
    return delta, result


def page_delta_header_prefix(encoded: bytes | bytearray | memoryview) -> bytes:
    """Return the SPHD1 magic, length, and JSON header of one encoded delta.

    The prefix is sufficient for ``plan_page_delta`` with ``total_bytes``.
    Callers must take it only from bytes they have authenticated.
    """

    prefix_bytes = len(_DELTA_MAGIC) + _HEADER_LENGTH.size
    view = memoryview(encoded).cast("B")
    if len(view) < prefix_bytes or bytes(view[: len(_DELTA_MAGIC)]) != _DELTA_MAGIC:
        raise HybridCodecError("hybrid page delta has an invalid prefix")
    (header_length,) = _HEADER_LENGTH.unpack_from(view, len(_DELTA_MAGIC))
    header_end = prefix_bytes + header_length
    if header_length <= 0 or header_end > len(view):
        raise HybridCodecError("hybrid page delta header is truncated")
    return bytes(view[:header_end])


def apply_page_delta(
    layout: PageLayout,
    base_snapshot: bytes,
    encoded_delta: bytes,
    *,
    base_block_counts: Sequence[int],
    result_block_counts: Sequence[int],
    base_boundary_tokens: int,
    result_boundary_tokens: int,
) -> bytes:
    """Verify and apply one page-semantic delta to its exact base snapshot."""

    return _apply_verified_page_delta(
        layout,
        _verify_page_snapshot_bytes(base_snapshot),
        encoded_delta,
        base_block_counts=base_block_counts,
        result_block_counts=result_block_counts,
        base_boundary_tokens=base_boundary_tokens,
        result_boundary_tokens=result_boundary_tokens,
    ).payload


def _apply_verified_page_delta(
    layout: PageLayout,
    base: _VerifiedPageSnapshot,
    encoded_delta: bytes | bytearray,
    *,
    base_block_counts: Sequence[int],
    result_block_counts: Sequence[int],
    base_boundary_tokens: int,
    result_boundary_tokens: int,
) -> _VerifiedPageSnapshot:
    """Reuse the preceding immutable result proof and verify every delta stage."""

    plan = plan_page_delta(
        layout,
        encoded_delta,
        base_block_counts=base_block_counts,
        result_block_counts=result_block_counts,
        base_boundary_tokens=base_boundary_tokens,
        result_boundary_tokens=result_boundary_tokens,
        total_bytes=len(encoded_delta),
    )
    if plan.base_snapshot_sha256 != base.sha256:
        raise HybridCodecError("hybrid page delta identity or base differs")
    base_plan = plan_page_snapshot(
        layout,
        base.payload,
        plan.base_block_counts,
    )
    # Join authenticated source views directly into the sole result allocation.
    # Decoding layers and concatenating their prefixes would copy the complete
    # base and result repeatedly at every stage of a growing flat history.
    parts: list[bytes | memoryview] = [
        encode_page_snapshot_header(layout, plan.result_block_counts)
    ]
    with (
        memoryview(base.payload) as base_view,
        memoryview(encoded_delta) as encoded_view,
    ):
        for tail, span in zip(plan.tails, base_plan.spans, strict=True):
            payload_tail = encoded_view[tail.source_start : tail.source_end]
            if hashlib.sha256(payload_tail).hexdigest() != tail.sha256:
                raise HybridCodecError("hybrid page delta payload checksum mismatch")
            prefix_end = span.source_start + tail.destination_byte_offset
            parts.extend((base_view[span.source_start:prefix_end], payload_tail))
        result = b"".join(parts)
    if len(result) != page_snapshot_encoded_size(layout, plan.result_block_counts):
        raise HybridCodecError("hybrid page delta result size differs")
    result_sha256 = hashlib.sha256(result).hexdigest()
    if result_sha256 != plan.result_snapshot_sha256:
        raise HybridCodecError("hybrid page delta result checksum mismatch")
    return _VerifiedPageSnapshot(result, result_sha256)


class _PageHistoryReconstruction:
    """Private layer buffers with a complete checksum proof at each boundary.

    A reconstruction owns its mutable buffers for one restore only. Every
    delta authenticates its exact base, tail bytes, and canonical result before
    the following stage can use it. Only the final immutable snapshot escapes.
    """

    def __init__(
        self,
        layout: PageLayout,
        snapshot: bytes | bytearray,
        block_counts: Sequence[int],
        boundary_tokens: int,
    ) -> None:
        verified = _verify_page_snapshot_bytes(snapshot)
        plan = plan_page_snapshot(layout, verified.payload, block_counts)
        self._layout = layout
        self._counts = plan.block_counts
        self._boundary = boundary_tokens
        self._sha256 = verified.sha256
        self._valid = True
        self._header = verified.payload[:plan.header_bytes]
        with memoryview(verified.payload) as view:
            self._layers = [
                bytearray(view[span.source_start:span.source_end])
                for span in plan.spans
            ]

    def apply(
        self,
        encoded_delta: bytes | bytearray,
        *,
        base_block_counts: Sequence[int],
        result_block_counts: Sequence[int],
        base_boundary_tokens: int,
        result_boundary_tokens: int,
    ) -> None:
        if not self._valid:
            raise HybridCodecError("hybrid page history has an unverified stage")
        self._valid = False
        if (tuple(base_block_counts) != self._counts
                or base_boundary_tokens != self._boundary):
            raise HybridCodecError("hybrid page history boundary differs")
        plan = plan_page_delta(
            self._layout,
            encoded_delta,
            base_block_counts=base_block_counts,
            result_block_counts=result_block_counts,
            base_boundary_tokens=base_boundary_tokens,
            result_boundary_tokens=result_boundary_tokens,
        )
        if plan.base_snapshot_sha256 != self._sha256:
            raise HybridCodecError("hybrid page delta identity or base differs")
        with memoryview(encoded_delta) as encoded_view:
            # Check every tail before changing a buffer. A bad intermediate
            # result is fatal even when a subsequent stage overwrites its data.
            for tail in plan.tails:
                if hashlib.sha256(encoded_view[tail.source_start:tail.source_end]).hexdigest() != tail.sha256:
                    raise HybridCodecError("hybrid page delta payload checksum mismatch")
            for buffer, tail in zip(self._layers, plan.tails, strict=True):
                existing = len(buffer) - tail.destination_byte_offset
                # A bytearray slice assignment can first copy its buffer
                # source into a temporary bytearray. Assign through a view for
                # fixed-size overwrites, then release it before growing.
                with memoryview(buffer) as target:
                    target[tail.destination_byte_offset:] = encoded_view[
                        tail.source_start:tail.source_start + existing
                    ]
                buffer.extend(encoded_view[tail.source_start + existing:tail.source_end])
        header = encode_page_snapshot_header(self._layout, plan.result_block_counts)
        if len(header) + sum(map(len, self._layers)) != page_snapshot_encoded_size(
            self._layout, plan.result_block_counts):
            raise HybridCodecError("hybrid page delta result size differs")
        result_digest = hashlib.sha256(header)
        for buffer in self._layers:
            result_digest.update(buffer)
        result_sha256 = result_digest.hexdigest()
        if result_sha256 != plan.result_snapshot_sha256:
            raise HybridCodecError("hybrid page delta result checksum mismatch")
        self._header = header
        self._counts = plan.result_block_counts
        self._boundary = result_boundary_tokens
        self._sha256 = result_sha256
        self._valid = True

    def finish(self) -> bytes:
        if not self._valid:
            raise HybridCodecError("hybrid page history has an unverified stage")
        return b"".join((self._header, *self._layers))


def compose_page_history(
    layout: PageLayout,
    base_snapshot: bytes | bytearray | _VerifiedPageSnapshot,
    base_block_counts: Sequence[int],
    stage_plans: Sequence[PageDeltaPlan],
    read_tail: Callable[[int, PageDeltaTail], bytes | bytearray],
    *,
    _return_verified: bool = False,
) -> bytes | _VerifiedPageSnapshot:
    """Compose an ordered delta history by reading each surviving tail once.

    ``stage_plans`` must come from authenticated delta headers, oldest first.
    Each final layer byte has exactly one writer: the newest stage whose tail
    covers it, or the base snapshot. Tails that no later stage supersedes are
    read through ``read_tail`` and checked against their header-bound SHA-256
    before any byte is copied; superseded tails are never requested.

    Every stage header binds the digest of its base, so consecutive headers
    form a chain from the base snapshot's computed digest to the newest
    result digest. The composed result is hashed once and must equal that
    newest digest. Intermediate results are not materialized or rehashed:
    no byte of an intermediate result reaches the output unless it is also a
    byte of the final result, which the final digest authenticates.
    """

    if not stage_plans:
        raise HybridCodecError("page history contains no delta")
    base = _verify_page_snapshot_bytes(base_snapshot)
    base_plan = plan_page_snapshot(layout, base.payload, base_block_counts)
    # Per destination layer: [destination_start, destination_end, stage, start]
    # where stage -1 is the base snapshot and start is the source byte offset
    # of destination_start (base: snapshot offset; delta: offset in the tail).
    ranges: list[list[tuple[int, int, int, int]]] = [
        [(0, span.source_end - span.source_start, -1, span.source_start)]
        for span in base_plan.spans
    ]
    current_counts = base_plan.block_counts
    current_sha256 = base.sha256
    for stage_index, plan in enumerate(stage_plans):
        if (
            plan.base_block_counts != current_counts
            or plan.base_snapshot_sha256 != current_sha256
            or len(plan.tails) != len(ranges)
        ):
            raise HybridCodecError("hybrid page delta identity or base differs")
        for destination_index, tail in enumerate(plan.tails):
            cutoff = tail.destination_byte_offset
            kept = []
            for start, end, owner, source in ranges[destination_index]:
                if start >= cutoff:
                    break
                kept.append((start, min(end, cutoff), owner, source))
            if (kept[-1][1] if kept else 0) != cutoff:
                raise HybridCodecError("hybrid page delta prefix coverage differs")
            length = tail.source_end - tail.source_start
            if length:
                kept.append((cutoff, cutoff + length, stage_index, 0))
            ranges[destination_index] = kept
        current_counts = plan.result_block_counts
        current_sha256 = plan.result_snapshot_sha256

    header = encode_page_snapshot_header(layout, current_counts)
    total = page_snapshot_encoded_size(layout, current_counts)
    result_plan = plan_page_snapshot(layout, header, current_counts, total_bytes=total)
    result = bytearray(total)
    result[: len(header)] = header
    with memoryview(result) as output, memoryview(base.payload) as base_view:
        for destination_index, (span, layer_ranges) in enumerate(
            zip(result_plan.spans, ranges, strict=True)
        ):
            covered = 0
            for start, end, owner, source in layer_ranges:
                if start != covered:
                    raise HybridCodecError("hybrid page history coverage has a gap")
                target = output[span.source_start + start : span.source_start + end]
                if owner < 0:
                    target[:] = base_view[source : source + end - start]
                else:
                    tail = stage_plans[owner].tails[destination_index]
                    payload = read_tail(owner, tail)
                    if (
                        len(payload) != tail.source_end - tail.source_start
                        or not payload_matches(
                            stage_plans[owner].integrity, payload,
                            sha256=tail.sha256, crc32=tail.crc32,
                        )
                    ):
                        raise PageDeltaPayloadChecksumError(
                            "hybrid page delta payload checksum mismatch"
                        )
                    with memoryview(payload) as tail_view:
                        target[:] = tail_view[: end - start]
                    del payload
                target.release()
                covered = end
            if covered != span.source_end - span.source_start:
                raise HybridCodecError("hybrid page history coverage differs")
    # Under fast and none the result digest is a boundary token, so the
    # composed bytes are bound by their per-tail checks alone.
    if (
        stage_plans[-1].integrity == "sha256"
        and hashlib.sha256(result).hexdigest() != current_sha256
    ):
        raise HybridCodecError("hybrid page delta result checksum mismatch")
    immutable = bytes(result)
    if _return_verified:
        return _VerifiedPageSnapshot(immutable, current_sha256)
    return immutable


def materialize_page_extension_capture(
    layout: PageLayout,
    base_snapshot: bytes,
    captured_tail: object,
    *,
    base_block_counts: Sequence[int],
    result_block_counts: Sequence[int],
    reused_pages_by_group: Sequence[int],
) -> bytes:
    """Build a complete snapshot from verified base bytes and captured pages."""

    base_counts = tuple(int(value) for value in base_block_counts)
    result_counts = tuple(int(value) for value in result_block_counts)
    reused = tuple(int(value) for value in reused_pages_by_group)
    if not (
        len(base_counts)
        == len(result_counts)
        == len(reused)
        == len(layout.groups)
    ):
        raise HybridCodecError("captured page extension geometry differs")
    if any(
        result < base
        or not 0 <= shared <= base
        for base, result, shared in zip(
            base_counts, result_counts, reused, strict=True
        )
    ):
        raise HybridCodecError("captured page extension counts are invalid")
    check_group_block_policies(layout, base_counts)
    check_group_block_policies(layout, result_counts)
    total_bytes = getattr(captured_tail, "total_bytes", None)
    read_range = getattr(captured_tail, "read_range", None)
    if isinstance(captured_tail, (bytes, bytearray, memoryview)):
        view = memoryview(captured_tail)
        total_bytes = len(view)

        def read_range(start: int, end: int) -> bytes:
            return view[start:end].tobytes()

    if (
        type(total_bytes) is not int
        or total_bytes <= 0
        or not callable(read_range)
    ):
        raise HybridCodecError("captured page extension payload is invalid")
    base_payloads = decode_page_snapshot(layout, base_snapshot, base_counts)
    result_payloads: dict[str, bytes] = {}
    cursor = 0
    for group_index, group in enumerate(layout.groups):
        shared = reused[group_index]
        tail_pages = result_counts[group_index] - shared
        for layer in group.layers:
            tail_bytes = tail_pages * layer.bytes_per_page
            end = cursor + tail_bytes
            if end > total_bytes:
                raise HybridCodecError("captured page extension payload is truncated")
            tail = read_range(cursor, end)
            if not isinstance(tail, bytes) or len(tail) != tail_bytes:
                raise HybridCodecError("captured page extension range differs")
            prefix = base_payloads[layer.name][
                : shared * layer.bytes_per_page
            ]
            result_payloads[layer.name] = prefix + tail
            cursor = end
    if cursor != total_bytes:
        raise HybridCodecError("captured page extension has trailing bytes")
    return encode_page_snapshot(layout, result_counts, result_payloads)


def split_snapshot(encoded: bytes, part_count: int) -> tuple[bytes, ...]:
    if part_count <= 0 or len(encoded) < part_count:
        raise HybridCodecError("hybrid snapshot cannot cover the declared span")
    width, remainder = divmod(len(encoded), part_count)
    parts = []
    offset = 0
    for index in range(part_count):
        size = width + (1 if index < remainder else 0)
        parts.append(encoded[offset : offset + size])
        offset += size
    return tuple(parts)
