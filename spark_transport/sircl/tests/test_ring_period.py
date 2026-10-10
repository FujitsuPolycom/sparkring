"""The ring harness summary's period of back-to-back calls: alternating ranks, NCCL by period, table, tuning."""

from __future__ import annotations

from sparkring_sircl.ring import plan, summary, worker
from sparkring_sircl.ring.site import Site


def _site() -> Site:
    return Site.from_json({
        "schema": "sircl-ring-site/v1", "image": "sha256:0123456789ab", "lan_interface": "lan0",
        "control_port": 29650, "remote_dir": "/tmp/sircl-ring",
        "ring": [{"name": f"spark{i}", "ssh": f"op@192.0.2.{10 + i}", "lan_address": f"192.0.2.{10 + i}"}
                 for i in range(8)],
    })


def _rank(rank, runs):
    return {"global_rank": rank, "group": rank // 2, "host": f"spark{rank}", "runs": runs, "error": None,
            "exit_code": 0}


BASE = {"dim": 0, "correct": True, "mismatched_calls": 0, "checked": 1, "counters": {}}


def test_period_of_alternating_ranks_is_the_time_between_completions():
    # Rank 0 starts every second call late: its peer's data was there (85 us), then it waits (245 us).
    early, late = [85.0, 245.0] * 10, [245.0, 85.0] * 10
    group_period, ranks = summary.period([early, late])
    assert ranks == [165.0, 165.0] and group_period == 165.0
    assert summary.period([[10.0, 30.0, 11.0, 500.0]])[0] == 20.5     # one slow call does not set it
    assert summary.period([[7.0]]) == (7.0, [7.0])
    assert summary.period([[7.0], []]) == (None, [7.0, None])


def test_merge_records_the_period_and_compares_with_nccl_by_period():
    built = plan.build_plan(_site(), "pairs", "run1", digest="d" * 16).to_json()
    sircl = {**BASE, "collective": "all_reduce_large", "mode": "graph", "shape": [4194304], "bytes": 8388608,
             "algorithm": "ring, pieces of 524288"}
    nccl = {**BASE, "collective": "nccl_all_reduce", "mode": "graph", "shape": [4194304], "bytes": 8388608,
            "algorithm": "NCCL 2.32.3", "baseline": "nccl"}
    small = {**BASE, "collective": "all_reduce", "mode": "graph", "shape": [8], "bytes": 16}
    results = []
    for rank in range(8):
        times = [355.0, 533.0] * 10 if rank % 2 == 0 else [533.0, 355.0] * 10
        results.append(_rank(rank, [{**sircl, "times_us": times}, {**nccl, "times_us": [431.0] * 20},
                                    {**small, "times_us": [20.0, 22.0] * 10}]))
    merged = summary.merge(built, results)
    first = merged["cases"][0]
    assert first["slowest_p50_us"] == 533.0 and first["period_us"] == 444.0
    assert first["rank_period_us"] == [444.0, 444.0]
    assert first["period_busbw_gbps"] == round(8388608 / 444.0e-6 / 1e9, 3)
    assert first["vs_nccl"] == round(431.0 / 533.0, 3)
    assert first["nccl_period_us"] == 431.0 and first["vs_nccl_period"] == round(431.0 / 444.0, 3)
    assert merged["cases"][1]["period_busbw_gbps"] == round(8388608 / 431.0e-6 / 1e9, 3)
    lines = summary.table(merged).splitlines()
    assert "[period 444.0 us, busbw 18.89 GB/s, NCCL 0.97x by period]" in lines[2]
    assert "[period 431.0 us" in lines[3]
    assert "period" not in lines[4]          # rows below 1 MiB keep the percentiles alone


def test_tuning_rows_rank_candidates_by_period():
    case = {"group": 0, "mode": "graph", "bytes": 4194304, "correct": True, "slowest_p50_us": 275.7,
            "period_us": 249.2, "tune": {"collective": "all_reduce", "choice": {"backend": "sircl"}}}
    older = {**case, "bytes": 8388608, "slowest_p50_us": 460.0}
    del older["period_us"]
    rows = summary.tuning_rows({"cases": [case, older]})[0]
    assert [row["p50_us"] for row in rows] == [249.2, 460.0]


def test_sircl_and_nccl_rows_of_one_size_and_time_have_one_bus_bandwidth():
    assert worker.bandwidths("nccl_all_reduce", 8388608, 8, 727.30) == \
        worker.bandwidths("all_reduce_large", 8388608, 8, 727.30)
    assert worker.bandwidths("nccl_all_reduce", 8388608, 8, 727.30)["busbw_gbps"] == 20.184
    assert worker.bandwidths("nccl_all_gather", 2097152, 8, 727.30) == \
        worker.bandwidths("all_gather_large", 2097152, 8, 727.30)
    built = plan.build_plan(_site(), "ring8", "run1", digest="d" * 16).to_json()
    runs = [{**BASE, "collective": "all_reduce_large", "mode": "eager", "shape": [4194304], "bytes": 8388608,
             "algorithm": "ring, pieces of 262144", "times_us": [727.30] * 4},
            {**BASE, "collective": "nccl_all_reduce", "mode": "eager", "shape": [4194304], "bytes": 8388608,
             "algorithm": "NCCL 2.32.3", "baseline": "nccl", "times_us": [727.30] * 4}]
    merged = summary.merge(built, [{**_rank(rank, runs), "group": 0} for rank in range(8)])
    sircl, nccl = merged["cases"]
    assert sircl["busbw_gbps"] == nccl["busbw_gbps"] == 20.184
    assert sircl["period_busbw_gbps"] == nccl["period_busbw_gbps"] == 20.184
    lines = summary.table(merged).splitlines()
    assert "    20.18" in lines[2] and "    20.18" in lines[3]


class _Dist:
    def __init__(self, log):
        self.log = log

    def barrier(self, group=None):
        self.log.append("barrier")


class _Session:
    def __init__(self, log):
        self.log = log

    def event_trace_records(self):
        self.log.append("take")
        return {"lost": {"native": 0, "kernel": 0}, "records": [(1, "native", "OP", 4, 1)], "offset_ns": 0,
                "offset_error_ns": 5}

    def check_health(self):
        self.log.append("health")


class _Torch:
    class cuda:
        @staticmethod
        def synchronize():
            pass


def test_traced_calls_run_back_to_back_after_a_barrier():
    log = []
    harness = type("H", (), {"torch": _Torch, "dist": _Dist(log), "process_group": None})()
    trace = worker.Harness._traced_call(harness, _Session(log), lambda: log.append("call"))
    assert log == ["take", "barrier"] + ["call"] * worker.TRACE_CALLS + ["health", "take"]
    assert trace["calls"] == worker.TRACE_CALLS and trace["records"] == [[1, "native", "OP", 4, 1]]
