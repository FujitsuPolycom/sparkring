"""Row partition, padding, all-gather layout and launch cutting (``layout.py``), without torch."""

from __future__ import annotations

import itertools

import pytest
from glm_dsa_indexer_split import layout

TP = 8
ROW_COUNTS = list(range(1, 70)) + [127, 128, 129, 255, 256, 257, 511, 1000, 1023, 4095, 4096, 4097, 8191, 8192]


@pytest.mark.parametrize("dcp", [1, 2, 4])
@pytest.mark.parametrize("rows", ROW_COUNTS)
def test_partition_covers_every_row_once(dcp, rows):
    plans = [layout.row_layout(rows, TP, dcp, rank) for rank in range(TP)]
    padded = -(-rows // TP) * TP
    for plan in plans:
        assert plan.padded == padded and plan.padded % TP == 0 and plan.padded - rows < TP
        assert plan.per_rank == padded // TP
        assert plan.copies == TP // dcp and plan.group == plan.tp_rank // dcp
        # The rank's all-gather slot lies inside its group's block (rows below `rows`).
        assert plan.block_start <= min(plan.send_start, rows) and min(plan.send_stop, rows) <= plan.block_stop
        assert 0 <= plan.block_start <= plan.block_stop <= rows
    # The members of a group share one block; the groups' blocks tile [0, rows) in group order.
    blocks = {}
    for plan in plans:
        blocks.setdefault(plan.group, set()).add((plan.block_start, plan.block_stop))
    assert all(len(spans) == 1 for spans in blocks.values())
    ordered = [next(iter(blocks[g])) for g in range(TP // dcp)]
    position = 0
    for start, stop in ordered:
        assert start == min(position, rows)
        position = max(position, stop)
    assert position == rows
    # The send slots tile [0, padded) in rank order: the all-gather's rank-major output is the row order.
    assert [(p.send_start, p.send_stop) for p in plans] == [
        (r * padded // TP, (r + 1) * padded // TP) for r in range(TP)]


@pytest.mark.parametrize("dcp", [1, 2, 4])
@pytest.mark.parametrize("rows", [1, 7, 8, 9, 37, 253, 300, 8192])
def test_all_gather_layout_reassembles_rows(dcp, rows):
    """Each group fills its block; rank t sends rows [t S, (t+1) S); the concatenation in rank order is the step."""
    truth = [f"row{i}" for i in range(rows)]
    gathered = []
    filled = {}
    for rank in range(TP):
        plan = layout.row_layout(rows, TP, dcp, rank)
        buffer = ["pad"] * (plan.padded + TP)
        for row in range(plan.block_start, plan.block_stop):
            buffer[row] = truth[row]  # what the group's launches and merge write
        filled[rank] = buffer
        gathered.extend(buffer[plan.send_start:plan.send_stop])
    assert len(gathered) == -(-rows // TP) * TP
    assert gathered[:rows] == truth
    for rank in range(TP):
        plan = layout.row_layout(rows, TP, dcp, rank)
        after = list(filled[rank])
        after[:plan.block_start] = gathered[:plan.block_start]
        after[plan.block_stop:rows] = gathered[plan.block_stop:rows]
        assert after[:rows] == truth


@pytest.mark.parametrize("rows", [1, 3, 8, 9, 300, 8192])
def test_work_per_rank_is_one_eighth(rows):
    for dcp in (1, 2, 4):
        plans = [layout.row_layout(rows, TP, dcp, rank) for rank in range(TP)]
        per_rank = -(-rows // TP)
        assert max(p.block_rows for p in plans) <= dcp * per_rank
        assert sum(p.block_rows for p in plans) == rows * dcp  # each row scored by the d members of one group


def test_owner_and_group_of_a_row():
    plan = layout.row_layout(8192, TP, 4, 5)
    assert plan.per_rank == 1024
    assert (plan.block_start, plan.block_stop) == (4096, 8192)
    assert (plan.send_start, plan.send_stop) == (5120, 6144)
    assert plan.owner(5120) == 5 and plan.scoring_group(5120) == 1 and plan.scoring_group(4095) == 0


def test_short_steps_leave_late_groups_empty():
    plan = layout.row_layout(1, TP, 4, 4)
    assert (plan.block_start, plan.block_stop) == (1, 1) and plan.block_rows == 0
    assert (plan.send_start, plan.send_stop) == (4, 5)
    first = layout.row_layout(1, TP, 4, 0)
    assert (first.block_start, first.block_stop) == (0, 1)


@pytest.mark.parametrize("tp,dcp", [(8, 8), (8, 3), (4, 4), (8, 16), (0, 1)])
def test_unsupported_geometry_refuses(tp, dcp):
    with pytest.raises(ValueError):
        layout.check_geometry(tp, dcp)


def test_bad_arguments_refuse():
    with pytest.raises(ValueError):
        layout.row_layout(0, TP, 4, 0)
    with pytest.raises(ValueError):
        layout.row_layout(10, TP, 4, 8)
    with pytest.raises(ValueError):
        layout.block_launches([(0, 4, "a")], 0, 4, full=True, cap=0)


def _spans(sizes_by_request):
    spans, position = [], 0
    for key, sizes in sizes_by_request:
        for size in sizes:
            spans.append((position, position + size, key))
            position += size
    return spans, position


def _check_launches(launches, spans, start, stop, cap, full):
    rows = [row for launch in launches for row in range(launch.start, launch.stop)]
    assert rows == list(range(start, stop))
    for launch in launches:
        assert launch.rows <= cap
        assert launch.fragments[0].start == launch.start and launch.fragments[-1].stop == launch.stop
        keys = {spans[f.chunk][2] for f in launch.fragments}
        assert len(keys) == 1  # one request per launch
        for a, b in zip(launch.fragments, launch.fragments[1:]):
            assert a.stop == b.start and b.chunk == a.chunk + 1
        for fragment in launch.fragments:
            lo, hi, _ = spans[fragment.chunk]
            assert lo <= fragment.start < fragment.stop <= hi
        if not full:
            assert len(launch.fragments) == 1


@pytest.mark.parametrize("full", [False, True])
def test_block_launches(full):
    cases = [
        [("a", [128] * 64)],  # one request at 1M context: the image's 128-row launches
        [("a", [1024] * 8)],  # 128K context
        [("a", [4096, 2614, 4096, 1482])],  # irregular logits-budget split of one request
        [("a", [37]), ("b", [96, 96, 19]), ("c", [5])],  # several requests in one step
    ]
    for sizes in cases:
        spans, total = _spans(sizes)
        for start, stop in itertools.chain([(0, total), (0, 0)],
                                           [(total * i // 5, total * (i + 2) // 5) for i in range(4)]):
            for cap in (96, 4096):
                if any(hi - lo > cap for lo, hi, _ in spans) and not full:
                    continue
                launches = layout.block_launches(spans, start, stop, full=full, cap=cap)
                _check_launches(launches, spans, start, stop, cap, full)
                if full and stop > start:
                    # Full launches leave at most one short launch per request run.
                    runs = {}
                    for launch in launches:
                        runs.setdefault(spans[launch.fragments[0].chunk][2], []).append(launch.rows)
                    for counts in runs.values():
                        assert all(c == cap for c in counts[:-1])


def test_full_launches_merge_the_image_launches_of_a_block():
    spans, total = _spans([("a", [128] * 64)])
    block = layout.block_launches(spans, 4096, 8192, full=True, cap=4096)
    assert [(launch.start, launch.stop, len(launch.fragments)) for launch in block] == [(4096, 8192, 32)]
    image_sized = layout.block_launches(spans, 4096, 8192, full=False, cap=4096)
    assert len(image_sized) == 32 and all(launch.rows == 128 for launch in image_sized)


def test_full_launches_never_join_two_requests():
    spans, total = _spans([("a", [50]), ("b", [50])])
    launches = layout.block_launches(spans, 0, total, full=True, cap=4096)
    assert [(launch.start, launch.stop) for launch in launches] == [(0, 50), (50, 100)]
