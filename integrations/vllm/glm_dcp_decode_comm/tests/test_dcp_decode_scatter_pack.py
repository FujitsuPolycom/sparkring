"""The packed all-to-all's host side over SIRCL's CUDA stand-in (``scatter_driver.py``, a process of its own)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from glm_dcp_decode_comm import layout as L

DRIVER = Path(__file__).resolve().parent / "scatter_driver.py"


def _drive(mode: str, sircl_root: Path) -> dict:
    if not (sircl_root.parent / "tests" / "teardown_driver.py").exists():
        pytest.skip("the SIRCL tree's tests/teardown_driver.py (the CUDA stand-in) is not available")
    env = {key: value for key, value in os.environ.items() if not key.startswith("SIRCL_")}
    done = subprocess.run([sys.executable, str(DRIVER), mode, str(sircl_root)], capture_output=True, text=True,
                          env=env, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def kernel(sircl_root):
    return _drive("kernel", sircl_root)


@pytest.fixture(scope="module")
def session(sircl_root):
    return _drive("session", sircl_root)


def test_the_kernel_object_refuses_geometries_it_cannot_run(kernel):
    errors = kernel["errors"]
    assert "at least two ranks" in errors["one rank"]
    assert "threads >= world_size * lane_count" in errors["few threads"]
    assert "slot count must be a power of two" in errors["slots"]
    assert "power-of-two count of at least 4 heads" in errors["heads"]


def test_the_kernel_bakes_the_wire_geometry_of_its_head_count(kernel):
    geometry = L.WireGeometry(1, 8)
    assert kernel["geometry"] == [geometry.row_packs, geometry.row_shift, geometry.head_shift,
                                  geometry.lse_row_packs, geometry.lse_shift] == [512, 9, 6, 2, 1]


def test_a_specialization_compiles_once_and_checks_its_launch_arguments(kernel):
    assert kernel["prepared_before"] is False and kernel["same_launcher"] is True
    assert kernel["prepared_after"] == [True, False]
    assert kernel["compiles"] == [["ScatterPackLaunch", 23, "glm_dcp_decode_comm packed all-to-all",
                                   ["dcp-scatter-pack", 4, 1, 512, 2, 128, 2, 8, 0]]]
    [launch] = kernel["launches"]
    chunk_packs = 3 * 8 * L.RECORD_BYTES // 16
    assert launch[:3] == [["pointer", 16], ["pointer", 32], ["pointer", 48]]
    assert launch[3:7] == [4 * chunk_packs, 4 * chunk_packs * 16, chunk_packs, 3] and launch[-1] == "stream"
    assert "chunk packs" in kernel["bad_chunk"]


@pytest.mark.parametrize("case,fits", [
    ("31,8,1048576,None", True),      # 4 x 31 x 8 x 1028 B = 1,019,776 B within a 1 MiB op
    ("32,8,1048576,None", False),     # 1,052,672 B: two ops
    ("31,8,1048576,131072", False),   # a 254,936-byte chunk above the 128 KiB relay-safe piece
    ("15,8,1048576,131072", True),    # 123,360 B
    ("16,8,1048576,131072", False),   # 131,584 B
    ("1,6,1048576,None", False),      # six heads per rank
    ("63,4,1048576,None", True),      # 4 x 63 x 4 x 1028 B = 1,036,224 B
    ("64,4,1048576,None", False),
    ("1,8,16384,None", False),        # the DCP group's op limit below one op of 32,896 B
])
def test_one_op_is_decided_by_the_groups_op_limit_and_the_sessions_piece_rules(session, case, fits):
    assert session["fits"][case] is fits


def test_a_session_without_scatter_collectives_never_fits(session):
    assert session["fits_unavailable"] is False


def test_an_eager_packed_all_to_all_follows_the_sessions_scatter_launch(session):
    rows, heads = 3, 8
    chunk = L.wire_chunk_bytes(rows, heads)
    nbytes = 4 * chunk
    grid = session["grid"]
    out_ptr, lse_ptr, recv_ptr = session["pointers"]
    assert session["eager"] is True
    assert session["eager_events"] == [
        ["launcher", [4, 1, 512, 2, 128, 2, heads, 0]],
        ["tuned", "all_to_all", nbytes], "lock", "health", ["counters", grid], ["order", False],
        ["launch", [out_ptr, lse_ptr, recv_ptr, nbytes // 16, nbytes, chunk // 16, rows,
                    4 * heads * L.V_DIM * 2, L.V_DIM * 2, 4 * heads * 4, chunk, 0x10000, 0x20000, 0x30000, 0x40000,
                    2 << 20, 0x50000, 0x60000 + grid, 0x70000 + grid, 0x50040, 4321, grid]],
        ["mark", False], "health", "unlock", "untuned"]


def test_the_grid_is_the_sessions_large_message_grid(session):
    packs = 4 * L.wire_chunk_bytes(3, 8) // 16
    per_block = 2 * 512
    required = -(-packs // per_block)
    assert session["grid"] == min(1 << (required - 1).bit_length(), 32)


def test_inside_a_capture_the_launch_skips_the_after_check_and_an_unprepared_kernel_declines(session):
    assert session["captured"] is True
    assert ["order", True] in session["captured_events"] and ["mark", True] in session["captured_events"]
    assert session["captured_events"].count("health") == 1
    assert session["unprepared"] is False and session["unprepared_events"] == []
