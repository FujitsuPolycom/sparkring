"""A captured op finds its launcher at the blocks per role it runs with (``RoceOneshotAllReduce.prepare``,
``op_blocks``).

A launcher is compiled for one number of blocks per role, and inside a CUDA graph capture a missing one raises.
The ring harness's tune sets each candidate's blocks with ``set_op_blocks`` and captures its graph cases, so
every count it sets must be compiled first. ``blocks_driver.py`` configures one rank of the cycle of eight from
the session's real configuration steps (CUDA modules mocked, in a process of its own), prepares it as the
harness's setup does and then with ``op_blocks``, and looks every chain and link kernel up at blocks per role
1, 2 and 4 as a capture does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DRIVER = Path(__file__).with_name("blocks_driver.py")
PROJECT = Path(__file__).resolve().parents[1]
KERNELS = ("ring_reduce", "ring_gather", "ring_scatter", "chain_gather", "chain_scatter", "chain_reduce")


@pytest.fixture(scope="module")
def prepared():
    pytest.importorskip("torch")
    env = {key: value for key, value in os.environ.items() if not key.startswith("SIRCL_")}
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    process = subprocess.run([sys.executable, str(DRIVER), "ring:8"], capture_output=True, text=True, env=env,
                             timeout=300)
    assert process.returncode == 0, process.stderr[-3000:]
    return json.loads(process.stdout.strip().splitlines()[-1])


def test_the_cycle_of_eight_runs_every_chain_and_link_kernel(prepared):
    available = prepared["available"]
    assert available["chain"] and available["links"] and available["ring"]
    assert set(available["blocks"]) == set(KERNELS)


def test_without_op_blocks_a_capture_at_other_blocks_finds_no_launcher(prepared):
    """On the cycle of eight the tune's first chain candidate runs at 1 block per role, and after the harness's setup
    preparation only the session's own count, 4, has a launcher."""
    own = prepared["available"]["blocks"]
    before = prepared["before"]
    assert before["chain_reduce 1"] == ("SIRCL chain all-reduce for torch.bfloat16 was not prepared before CUDA "
                                        "graph capture; call prepare()")
    for kernel in KERNELS:
        for blocks in (1, 2, 4):
            assert (f"{kernel} {blocks}" in before) == (blocks != own[kernel]), (kernel, blocks)


def test_prepare_compiles_every_named_kernel_at_every_count_once(prepared):
    assert prepared["after"] == {}
    assert prepared["compiles"] == prepared["launchers"] == len(KERNELS) * 3
    assert prepared["op_blocks"] == {}


def test_prepare_refuses_malformed_op_blocks_before_compiling(prepared):
    refused = prepared["refused"]
    assert all(refused), refused
    assert "not 'chain'" in refused[0]
    assert all("counts of blocks per role from 1 to 64" in message for message in refused[1:])
