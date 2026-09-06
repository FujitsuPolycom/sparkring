"""Offline tests for tiled-prefill input and numerical correctness."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from spark_transport.experiments.tiled_prefill.correctness_oracle import (
    GUARD_BYTES,
    INPUT_GENERATION_PERIOD,
    INACTIVE_INPUT_SENTINEL,
    INACTIVE_OUTPUT_SENTINEL,
    INPUT_GUARD_SENTINEL,
    OUTPUT_GUARD_SENTINEL,
    OracleTile,
    TiledHalfAssociation,
    fill_correctness_input,
    fill_expected_output,
    initialize_guarded_buffers,
    payload_geometry,
    validate_active_output,
    validate_sentinels,
    expected_output_bf16_bits,
    input_bf16_bits,
)


def test_bf16_oracle_preserves_the_half_specific_association_trees() -> None:
    assert INPUT_GENERATION_PERIOD == 7
    assert INPUT_GENERATION_PERIOD != 8

    lower = expected_output_bf16_bits(
        element=0,
        generation=7,
        association=TiledHalfAssociation.XOR1_THEN_XOR3,
    )
    upper = expected_output_bf16_bits(
        element=0,
        generation=7,
        association=TiledHalfAssociation.XOR3_THEN_XOR1,
    )

    assert lower == 0x4384  # BF16 264: ((r0+r1)+(r2+r3)).
    assert upper == 0x4382  # BF16 260: ((r0+r3)+(r1+r2)).
    assert lower != upper
    assert input_bf16_bits(rank=0, element=0, generation=1) != (
        input_bf16_bits(rank=0, element=0, generation=9)
    )


@pytest.mark.parametrize(
    ("query_rows", "expected_active", "expected_capacity", "expected_inactive"),
    (
        (40, 491_520, 524_288, 32_768),
        (513, 6_303_744, 6_815_744, 512_000),
        (1025, 12_595_200, 13_107_200, 512_000),
    ),
)
def test_partial_tail_guards_cover_inactive_input_and_output_capacity(
    query_rows: int,
    expected_active: int,
    expected_capacity: int,
    expected_inactive: int,
) -> None:
    geometry = payload_geometry(query_rows)
    assert geometry.active_bytes == expected_active
    assert geometry.capacity_bytes == expected_capacity
    assert geometry.inactive_bytes == expected_inactive

    buffers = initialize_guarded_buffers(geometry, guard_bytes=GUARD_BYTES)
    assert validate_sentinels(buffers, geometry).as_receipt_fields() == {
        "input_guard_corruptions": 0,
        "output_guard_corruptions": 0,
        "inactive_input_sentinel_corruptions": 0,
        "inactive_output_sentinel_corruptions": 0,
    }

    payload_end = GUARD_BYTES + geometry.active_bytes
    buffers.input[0] ^= INPUT_GUARD_SENTINEL
    buffers.output[-1] ^= OUTPUT_GUARD_SENTINEL
    buffers.input[payload_end] ^= INACTIVE_INPUT_SENTINEL
    buffers.output[payload_end] ^= INACTIVE_OUTPUT_SENTINEL

    assert validate_sentinels(buffers, geometry).as_receipt_fields() == {
        "input_guard_corruptions": 1,
        "output_guard_corruptions": 1,
        "inactive_input_sentinel_corruptions": 1,
        "inactive_output_sentinel_corruptions": 1,
    }


def test_active_output_comparison_checks_both_halves_of_a_partial_tile() -> None:
    geometry = payload_geometry(40)
    tile = OracleTile(
        input_offset_bytes=0,
        output_offset_bytes=0,
        active_bytes=geometry.active_bytes,
        generation=1,
    )
    buffers = initialize_guarded_buffers(geometry, guard_bytes=GUARD_BYTES)
    fill_correctness_input(buffers, tile, rank=2)
    fill_expected_output(buffers, tile)

    assert validate_active_output(buffers, (tile,)) == 0
    assert validate_sentinels(buffers, geometry).as_receipt_fields() == {
        "input_guard_corruptions": 0,
        "output_guard_corruptions": 0,
        "inactive_input_sentinel_corruptions": 0,
        "inactive_output_sentinel_corruptions": 0,
    }
    lower_byte = GUARD_BYTES
    upper_byte = GUARD_BYTES + tile.active_bytes // 2
    buffers.output[lower_byte] ^= 1
    buffers.output[upper_byte] ^= 1

    assert validate_active_output(buffers, (tile,)) == 2


def test_portable_oracle_header_spells_out_both_bf16_trees() -> None:
    header = (Path(__file__).parent / "tiled_correctness_oracle.hpp").read_text(
        encoding="utf-8"
    )

    assert "kInputGenerationPeriod = 7" in header
    assert "static_assert(kInputGenerationPeriod != 8U)" in header
    assert "bf16_add_bits(inputs[0], inputs[1])" in header
    assert "bf16_add_bits(inputs[2], inputs[3])" in header
    assert "bf16_add_bits(inputs[0], inputs[3])" in header
    assert "bf16_add_bits(inputs[1], inputs[2])" in header
    assert "kInputGuardSentinel" in header
    assert "kInactiveInputSentinel" in header
    assert "cuda_runtime" not in header
    assert "verbs" not in header


def test_cuda_correctness_kernels_use_the_qualification_receipt_abi() -> None:
    directory = Path(__file__).parent
    interface = (directory / "tiled_correctness_kernels.cuh").read_text(
        encoding="utf-8"
    )
    source = (directory / "tiled_correctness_kernels.cu").read_text(
        encoding="utf-8"
    )

    assert '#include "tiled_bulk_kernels.cuh"' in interface
    for launch in (
        "launch_initialize_correctness_receipt",
        "launch_fill_correctness_sentinels",
        "launch_fill_correctness_input",
        "launch_validate_correctness",
        "launch_validate_correctness_sentinels",
    ):
        assert launch in interface
        assert launch in source
    for counter in (
        "mismatched_active_elements",
        "input_guard_corruptions",
        "output_guard_corruptions",
        "inactive_input_sentinel_corruptions",
        "inactive_output_sentinel_corruptions",
    ):
        assert counter in source
    assert "TiledHalfAssociation::kXor1ThenXor3" in source
    assert "TiledHalfAssociation::kXor3ThenXor1" in source
    assert "TiledOracleHalf::kLowerXor1ThenXor3" in source
    assert "TiledOracleHalf::kUpperXor3ThenXor1" in source
    assert "descriptor_pointer[0]" in source
    assert "active_payload_bytes" in source
    assert "payload_capacity_bytes" in source
    for forbidden in (
        "ibverbs",
        "verbs_endpoint",
        "DoorbellControl",
        '#include "tiled_executor',
        "TiledExecutor::",
    ):
        assert forbidden not in interface
        assert forbidden not in source
    cmake = (directory.parents[1] / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    library_start = cmake.index("add_library(spark_transport")
    library_end = cmake.index("\n)", library_start)
    production_library = cmake[library_start:library_end]
    assert "tiled_correctness_kernels" not in production_library
    assert "tiled_correctness_oracle" not in production_library


def test_portable_cpp_oracle(tmp_path: Path) -> None:
    directory = Path(__file__).parent
    test_source = directory / "tiled_correctness_oracle_test.cpp"
    assert test_source.is_file()
    configured = os.environ.get("CXX")
    compiler = shlex.split(configured) if configured else []
    if not compiler:
        discovered = shutil.which("g++") or shutil.which("clang++")
        if discovered:
            compiler = [discovered]
    if not compiler:
        pytest.skip("no C++ compiler is available for the portable oracle")
    executable = tmp_path / "tiled_correctness_oracle_test"
    subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            str(test_source),
            "-o",
            str(executable),
        ],
        check=True,
        cwd=directory,
    )
    subprocess.run([str(executable)], check=True, cwd=directory)


def test_cuda_correctness_translation_unit_compiles_when_nvcc_is_available(
    tmp_path: Path,
) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is unavailable; CUDA compilation is not claimed")
    directory = Path(__file__).parent
    subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-c",
            str(directory / "tiled_correctness_kernels.cu"),
            "-o",
            str(tmp_path / "tiled_correctness_kernels.o"),
        ],
        check=True,
        cwd=directory,
    )
