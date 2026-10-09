"""A session's constructor reads ``SIRCL_POST_ORDER`` and ``SIRCL_PROGRESS_CPU`` under the lock that constructors
hold while they set them for their native contexts (``oneshot/runtime.py``, ``_process_environment``).

Sessions of one process may be constructed at once (the GPU emulation's rank threads). Each sets its own
resolved peer list and progress-thread CPU in the process environment while it creates and starts its native
context; a rank that read the variables then would take another rank's list and fail setup, or place its
progress thread on another rank's CPU. ``construction_driver.py`` runs the real ``_configure`` from several
threads in a process of its own (CUDA modules mocked); these tests check what each rank took.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DRIVER = Path(__file__).with_name("construction_driver.py")
PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def taken():
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    env = {key: value for key, value in os.environ.items() if key not in ("SIRCL_POST_ORDER", "SIRCL_PROGRESS_CPU")}
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    process = subprocess.run([sys.executable, str(DRIVER)], capture_output=True, text=True, env=env, timeout=300)
    assert process.returncode == 0, process.stderr[-3000:]
    return json.loads(process.stdout.strip().splitlines()[-1])


def test_a_rank_configuring_while_another_holds_its_values_waits_and_takes_the_process_values(taken):
    held = taken["held"]
    assert "error" not in held, held
    assert held["finished_while_held"] is False
    assert held["peers"] == taken["expected"]["1"] and held["cpu"] is None


def test_ranks_configured_at_once_each_take_their_own_order(taken):
    rounds = taken["rounds"]
    assert rounds["errors"] == []
    assert len(rounds["taken"]) == 10 * 4
    for _round, rank, peers, cpu in rounds["taken"]:
        assert peers == taken["expected"][str(rank)] and cpu is None
    assert len({tuple(peers) for peers in taken["expected"].values()}) == 4      # every rank's order differs
