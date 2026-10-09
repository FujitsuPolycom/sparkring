"""A collective's ops do not depend on a rank's pointer alignment (``oneshot/_aligned.py``,
``RoceOneshotAllReduce.large_reduce_staging``).

The ring and chain kernels need 16-byte aligned pointers, and a rank's alignment is a fact of its own memory. If
it chose the ops, one rank with an offset view would run transport pieces while its peers ran one ring or chain
op, each waiting for flags the other never writes. ``plan_driver.py`` configures every rank of a group from the
session's real configuration steps (CUDA modules mocked, in a process of its own) and asks each what it runs for
every combination of input and output alignment: the ops must be the same on every rank, and a rank stages
exactly the buffers that are not aligned when an op is a ring or chain op. The working-buffer helpers are checked
on host tensors.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DRIVER = Path(__file__).with_name("plan_driver.py")
PROJECT = Path(__file__).resolve().parents[1]
CASES = {"ring:8": [3 << 20, (2 << 20) + 16, 64 << 20], "path:0-3": [3 << 20, 8 << 20, (8 << 20) + 48]}


@pytest.fixture(scope="module")
def plans():
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    env = {key: value for key, value in os.environ.items() if not key.startswith("SIRCL_")}
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    found = {}
    for layout, sizes in CASES.items():
        process = subprocess.run([sys.executable, str(DRIVER), layout, ",".join(map(str, sizes))],
                                 capture_output=True, text=True, env=env, timeout=300)
        assert process.returncode == 0, process.stderr[-3000:]
        found[layout] = json.loads(process.stdout.strip().splitlines()[-1])
    return found


def test_every_rank_runs_the_same_ops_whatever_its_alignment(plans):
    for layout, ranks in plans.items():
        for nbytes in map(str, CASES[layout]):
            for mode in ("eager", "graph"):
                expected = ranks["0"][nbytes][f"{mode},1,1"][0]
                links = any(kind in ("ring", "chain") for kind, _, _ in expected)
                for rank, sizes in ranks.items():
                    for combo, (ops, stage_in, stage_out) in sizes[nbytes].items():
                        if not combo.startswith(mode):
                            continue
                        _, input_aligned, output_aligned = combo.split(",")
                        where = f"{layout} rank {rank} {nbytes} B {combo}"
                        assert ops == expected, where
                        assert stage_in == (links and input_aligned == "0"), where
                        assert stage_out == (links and output_aligned == "0"), where


def test_the_cases_cover_ring_and_chain_ops(plans):
    def kinds(layout, nbytes):
        return [kind for kind, _, _ in plans[layout]["0"][str(nbytes)]["eager,1,1"][0]]

    assert kinds("ring:8", 3 << 20) == ["ring"]                    # the cycle of eight's built-in plan
    assert kinds("ring:8", (2 << 20) + 16) == ["ring", "pieces"]   # a ring prefix and a remainder
    assert kinds("path:0-3", 8 << 20) == ["chain"]                 # the chain from its 8 MiB minimum
    assert kinds("path:0-3", 3 << 20) == ["pieces"]                # below it: no staging needed


LARGE = {"ring:8": [(3 << 20) + 6, 786438, 278], "path:0-3": [(8 << 20) + 6, 786438, 2097158]}


@pytest.mark.parametrize("layout", sorted(LARGE))
def test_all_reduce_large_returns_the_whole_message_with_its_padded_tail(layout):
    # The real all_reduce_large on host tensors, every launch a copy of its input (one rank's sum): messages that
    # are not whole 16-byte packs end in a padded one-pack op, which uses scratch of its own; the output must
    # still be the whole message for aligned and unaligned inputs and outputs, through ring, chain and pieces.
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    env = {key: value for key, value in os.environ.items() if not key.startswith("SIRCL_")}
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    process = subprocess.run([sys.executable, str(DRIVER), "large", layout, ",".join(map(str, LARGE[layout]))],
                             capture_output=True, text=True, env=env, timeout=300)
    assert process.returncode == 0, process.stderr[-3000:]
    found = json.loads(process.stdout.strip().splitlines()[-1])
    assert len(found) == 2 * len(LARGE[layout]) * 4
    for case, (ops, elements, expected, equal) in found.items():
        assert elements == expected and equal, (layout, case, ops, elements)
        assert ops and ops[-1] == "pieces", (layout, case, ops)            # the padded tail is the last op
    kinds = {kind for ops, _, _, _ in found.values() for kind in ops}
    assert kinds >= ({"ring", "pieces"} if layout == "ring:8" else {"chain", "pieces"}), kinds


def test_working_buffers_copy_only_what_is_not_aligned():
    torch = pytest.importorskip("torch")
    from sparkring_sircl.oneshot import _aligned

    base = torch.arange(40, dtype=torch.bfloat16)
    offset = base[1:33]                                   # two bytes past the allocation's alignment
    assert not _aligned.aligned(offset) and _aligned.aligned(base)
    assert _aligned.aligned_input(base) is base
    work = _aligned.aligned_input(offset)
    assert _aligned.aligned(work) and torch.equal(work, offset) and work.data_ptr() != offset.data_ptr()
    assert _aligned.aligned_output(base) is base
    fresh = _aligned.aligned_output(offset)
    assert _aligned.aligned(fresh) and fresh.shape == offset.shape and fresh.dtype == offset.dtype
    kept = _aligned.aligned_output(offset, keep=True)
    assert _aligned.aligned(kept) and torch.equal(kept, offset)
    fresh.fill_(7)
    assert _aligned.copy_back(fresh, offset) is offset and bool((offset == 7).all())
    assert bool((base[0] == 0) and (base[33:] == torch.arange(33, 40, dtype=torch.bfloat16)).all())
    assert _aligned.copy_back(base, base) is base
