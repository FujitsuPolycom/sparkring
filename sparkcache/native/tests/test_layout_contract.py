from __future__ import annotations

import ctypes
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "spark_cache_placement.h"
CUDA = ROOT / "src" / "spark_cache_placement.cu"


class Destination(ctypes.Structure):
    _fields_ = [
        ("destination_base", ctypes.c_uint64),
        ("destination_rows", ctypes.c_uint64),
        ("destination_row_stride_bytes", ctypes.c_uint32),
        ("bytes_per_token", ctypes.c_uint32),
        ("record_kind", ctypes.c_uint32),
        ("source_layer_ordinal", ctypes.c_uint32),
    ]


class Chunk(ctypes.Structure):
    _fields_ = [
        ("arena_offset_bytes", ctypes.c_uint64),
        ("encoded_bytes", ctypes.c_uint32),
        ("payload_offset_bytes", ctypes.c_uint32),
        ("first_slot_index", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("record_mask", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("record_offset_bytes", ctypes.c_uint32 * 4),
        ("record_length_bytes", ctypes.c_uint32 * 4),
    ]


class TransposedSource(ctypes.Structure):
    _fields_ = [
        ("source_offset_bytes", ctypes.c_uint64),
        ("destination_index", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class Config(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("arena_mode", ctypes.c_uint32),
        ("arena_bytes", ctypes.c_uint64),
        ("max_destinations", ctypes.c_uint32),
        ("max_slots", ctypes.c_uint32),
        ("max_chunks_per_slab", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class Stats(ctypes.Structure):
    _fields_ = [
        ("source_bytes", ctypes.c_uint64),
        ("staged_h2d_bytes", ctypes.c_uint64),
        ("restored_rows", ctypes.c_uint64),
        ("slot_uploads", ctypes.c_uint32),
        ("destination_table_uploads", ctypes.c_uint32),
        ("slabs_submitted", ctypes.c_uint32),
        ("scatter_kernel_launches", ctypes.c_uint32),
        ("device_error", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class AbiInfo(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("cudart_version", ctypes.c_uint32),
        ("arena_count", ctypes.c_uint32),
        ("max_record_kinds", ctypes.c_uint32),
        ("sizeof_config", ctypes.c_uint32),
        ("sizeof_destination", ctypes.c_uint32),
        ("sizeof_chunk", ctypes.c_uint32),
        ("sizeof_transposed_source", ctypes.c_uint32),
        ("sizeof_stats", ctypes.c_uint32),
        ("sizeof_arena_view", ctypes.c_uint32),
        ("capability_flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 5),
    ]


class ArenaView(ctypes.Structure):
    _fields_ = [
        ("host_address", ctypes.c_uint64),
        ("device_address", ctypes.c_uint64),
        ("capacity_bytes", ctypes.c_uint64),
        ("arena_index", ctypes.c_uint32),
        ("arena_mode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


def test_c_abi_layout_is_fixed() -> None:
    assert ctypes.sizeof(Destination) == 32
    assert ctypes.sizeof(Chunk) == 64
    assert ctypes.sizeof(TransposedSource) == 16
    assert ctypes.sizeof(Config) == 48
    assert ctypes.sizeof(Stats) == 56
    assert ctypes.sizeof(AbiInfo) == 64
    assert ctypes.sizeof(ArenaView) == 40
    text = HEADER.read_text(encoding="utf-8")
    assert "#define SPARK_CACHE_PLACEMENT_ABI_VERSION 1u" in text
    assert "#define SPARK_CACHE_PLACEMENT_MAX_RECORD_KINDS 4u" in text
    for symbol in (
        "spark_cache_placement_query_abi",
        "spark_cache_placement_acquire_arena_view",
        "spark_cache_placement_get_stats",
        "spark_cache_placement_runtime_last_error",
        "spark_cache_placement_copy_last_error",
        "spark_cache_placement_status_string",
    ):
        assert symbol in text


def test_exact_glm_inventory_and_fallback_slab_counts() -> None:
    local_rows = 393_216 // 4
    target_bytes = local_rows * 368
    indexer_bytes = local_rows * 132
    total = 79 * target_bytes + 22 * indexer_bytes
    assert total == 3_143_368_704

    def slabs(cap_mib: int) -> int:
        cap = cap_mib * 1024 * 1024
        return math.ceil(79 / (cap // target_bytes)) + math.ceil(
            22 / (cap // indexer_bytes)
        )

    assert slabs(128) == 30
    assert slabs(256) == 14


def test_direct_path_has_one_slot_upload_and_one_kernel_per_slab() -> None:
    text = CUDA.read_text(encoding="utf-8")
    begin = text.index("spark_cache_placement_begin_restore")
    acquire = text.index("spark_cache_placement_acquire_arena")
    begin_body = text[begin:acquire]
    assert begin_body.count("cudaMemcpy(") == 1
    assert "placement->stats.slot_uploads = 1;" in begin_body

    direct = text.index("spark_cache_placement_submit_direct_slab")
    transposed = text.index("spark_cache_placement_submit_transposed_slab")
    direct_body = text[direct:transposed]
    assert direct_body.count("scatter_direct_kernel<<<") == 1
    assert "cudaHostAllocMapped" not in direct_body
    assert 'b"".join' not in direct_body


def test_mapped_path_does_not_stage_the_arena() -> None:
    text = CUDA.read_text(encoding="utf-8")
    prepare = re.search(
        r"SparkCachePlacementStatus prepare_source_arena\(.+?\n\}",
        text,
        re.DOTALL,
    )
    assert prepare is not None
    body = prepare.group(0)
    assert (
        "placement->config.arena_mode ==\n      SPARK_CACHE_ARENA_STAGED_DEVICE"
    ) in body
    assert "cudaMemcpyAsync" in body
    assert "SPARK_CACHE_ARENA_MAPPED_HOST" not in body


def test_parser_names_the_outer_sha_trust_boundary() -> None:
    text = HEADER.read_text(encoding="utf-8")
    assert "after the caller has verified the manifest's outer SHA-256" in text
    assert "spark_cache_parse_verified_v1_chunk" in text
