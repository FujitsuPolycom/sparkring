"""SIRCL receipts and NCCL log lines, collected on each Spark and judged on Node A; offline.

Receipts follow ``sircl-vllm-receipt/v1`` as SIRCL's adapter writes them
(``sparkring_sircl.vllm.receipt``); container logs come from a fake ``docker
logs``.
"""
import io
import json
import os
from pathlib import Path

import pytest

from runtime.common.test_transport import TP2, TP4, sircl_deployment
from runtime.host import transport_receipts as receipts


# The session settings a pair's sessions report under the default tuning table's pair row, which sets a
# 1 MiB link slot.
PAIR_STATS = {"link_slot_bytes": 1048576}


def receipt(rank, group="tp", *, nccl="none", pynccl="skipped", rows=None, state="ready", tuning=None,
            large_blocks=None, stats=None):
    rows = rows if rows is not None else [
        {"collective": "all_reduce", "backend": "sircl", "method": "oneshot", "calls": 120, "reason": ""},
        {"collective": "all_gather", "backend": "sircl", "method": "session", "calls": 40, "reason": ""}]
    record = {"schema": receipts.RECEIPT_SCHEMA, "group": group, "global_rank": rank, "rank": rank, "world": 2,
              "nccl": nccl, "pynccl": pynccl, "session": "s-1", "state": state, "decisions": rows, "tuning": tuning}
    session_stats = dict(stats or {})
    if large_blocks is not None:
        session_stats["large_blocks"] = large_blocks
    if session_stats:
        record["session_stats"] = session_stats
    return record


def report(rank, records, *, init=(), library=(), complete=True):
    return {"schema": receipts.REPORT_SCHEMA, "rank": rank, "backend": "sircl",
            "container": {"running": True, "started_at": "2026-10-08T00:00:00.123456789Z"},
            "receipts": [{"file": f"rank{record['global_rank']}-{record['group']}.json", **record} for record in records],
            "problems": [], "log": {"init": list(init), "library": list(library), "tail": ["ready"],
                                    "complete": complete}}


def pair_reports(**options):
    options.setdefault("stats", PAIR_STATS)
    return [report(rank, [receipt(rank, **options)]) for rank in (0, 1)]


def test_a_deployment_without_nccl_whose_receipts_show_only_sircl_is_as_expected():
    lock, _ = sircl_deployment(TP2, "pair", 2, [0, 1])
    verdict = receipts.evaluate(lock, pair_reports(), now=lambda: 0)
    assert verdict["verdict"] == "as-expected" and verdict["nccl_observed"] == "absent", verdict["problems"]
    assert verdict["expected"] == "SIRCL carries every collective; NCCL creates no communicator"
    assert verdict["groups"]["tp"]["rows"][0] == {"collective": "all_gather", "backend": "sircl", "method": "session",
                                                  "calls": 80}
    assert receipts.text(verdict) == "Transport: sircl, NCCL: absent"
    assert any("NCCL-free receipts" in line for line in verdict["lines"])


@pytest.mark.parametrize("change, problem", [
    ({"rows": [{"collective": "all_reduce", "backend": "nccl", "method": "eager", "calls": 14}]}, "reached NCCL"),
    ({"pynccl": "built"}, "PyNccl"),
    ({"state": "setup"}, "state setup"),
    ({"rows": [{"collective": "all_reduce", "backend": "refuse", "method": "-", "calls": 1}]}, "refused"),
    ({"tuning": "0123456789abcdef"}, "tuning table 0123456789abcdef"),
    ({"large_blocks": 16}, "grid cap is 16 blocks"),
])
def test_a_receipt_that_differs_from_the_section_fails_the_check_and_names_the_rank(change, problem):
    lock, _ = sircl_deployment(TP2, "pair", 2, [0, 1])
    # A tuning row with a launch grid cap, which the sessions then report.
    lock["transport"]["tuning"]["settings"]["large_blocks"] = 32
    verdict = receipts.evaluate(lock, pair_reports(**{"large_blocks": 32, **change}), now=lambda: 0)
    assert verdict["verdict"] == "differs"
    assert any(problem in line and line.startswith("rank ") for line in verdict["problems"]), verdict["problems"]
    assert receipts.text(verdict).startswith("Transport check failed: rank ")


def test_an_nccl_communicator_in_a_log_fails_a_deployment_without_nccl():
    lock, _ = sircl_deployment(TP2, "pair", 2, [0, 1])
    reports = pair_reports()
    reports[1]["log"]["init"] = ["spark-b:1:1 [0] NCCL INFO ncclCommInitRank comm 0x1 rank 1 nranks 2"]
    verdict = receipts.evaluate(lock, reports, now=lambda: 0)
    assert verdict["verdict"] == "differs" and verdict["nccl_observed"] == "present"
    assert any("rank 1: its log shows 1 NCCL communicator" in line for line in verdict["problems"])


def test_with_nccl_auto_an_nccl_row_on_a_cabled_group_is_expected():
    lock, _ = sircl_deployment(TP2, "pair", 2, [0, 1], nccl="auto")
    rows = [{"collective": "all_reduce", "backend": "sircl", "method": "oneshot", "calls": 10},
            {"collective": "all_reduce", "backend": "nccl", "method": "eager", "calls": 3}]
    verdict = receipts.evaluate(lock, pair_reports(nccl="all", pynccl="built", rows=rows), now=lambda: 0)
    assert verdict["verdict"] == "as-expected" and verdict["nccl_observed"] == "present", verdict["problems"]
    assert receipts.text(verdict) == "Transport: sircl; NCCL carried collectives where the cabling allows, as expected"


def test_a_rank_without_a_receipt_or_a_spark_that_cannot_be_read():
    lock, _ = sircl_deployment(TP4, "cycle", 4, [0, 1, 2, 3])
    reports = [report(rank, [receipt(rank)]) for rank in range(3)] + [report(3, [])]
    verdict = receipts.evaluate(lock, reports, now=lambda: 0)
    assert verdict["verdict"] == "differs" and "rank 3: no tensor-parallel receipt" in verdict["problems"]
    reports[3] = {"schema": receipts.REPORT_SCHEMA, "rank": 3, "error": "ssh: connect timed out"}
    verdict = receipts.evaluate(lock, reports, now=lambda: 0)
    assert verdict["verdict"] == "unknown" and verdict["problems"][0] == "rank 3: ssh: connect timed out"
    assert "could not be read from every Spark" in receipts.text(verdict)


def test_the_verdict_and_the_receipts_are_recorded_beside_the_deployment(tmp_path):
    lock, _ = sircl_deployment(TP2, "pair", 2, [0, 1])
    reports = pair_reports()
    recorded = receipts.record(tmp_path, reports, receipts.evaluate(lock, reports, now=lambda: 0), now=lambda: 0)
    target = Path(recorded["receipts"])
    assert target == tmp_path / "receipts" / "19700101T000000Z"
    assert sorted(path.name for path in target.iterdir()) == ["log-rank0.json", "log-rank1.json", "rank0-tp.json",
                                                              "rank1-tp.json", "verdict.json"]
    assert "file" not in json.loads((target / "rank0-tp.json").read_text())
    assert receipts.latest(tmp_path) == recorded
    again = receipts.record(tmp_path, reports, receipts.evaluate(lock, reports, now=lambda: 0), now=lambda: 0)
    assert again["receipts"].endswith("19700101T000000Z-2") and receipts.latest(tmp_path) == again


class Logs:
    """``docker logs``: the given lines, once."""

    def __init__(self, lines):
        self.lines, self.argv = lines, None

    def __call__(self, argv, **options):
        self.argv = argv
        lines = self.lines

        class Process:
            stdout = io.StringIO("".join(line + "\n" for line in lines))

            def kill(self):
                pass

            def wait(self):
                return 0
        return Process()


def test_the_log_scan_keeps_nccl_lines_and_the_last_lines():
    lines = ["INFO starting", "host:1:1 [0] NCCL INFO Bootstrap: Using eth0", "vLLM is using nccl==2.32.3",
             *[f"line {n}" for n in range(300)]]
    logs = Logs(lines)
    scanned = receipts.scan_log("abc", popen=logs)
    assert logs.argv == ["docker", "--context", "default", "logs", "abc"]
    assert scanned["init"] == ["vLLM is using nccl==2.32.3"]
    assert scanned["library"] == ["host:1:1 [0] NCCL INFO Bootstrap: Using eth0"]
    assert len(scanned["tail"]) == receipts.TAIL_LINES and scanned["tail"][-1] == "line 299" and scanned["complete"]


def test_a_spark_reports_the_receipts_its_running_container_wrote(tmp_path):
    lock, _ = sircl_deployment(TP2, "pair", 2, [0, 1])
    directory = tmp_path / "srv/sparkring/sircltest/sircl/receipts"
    directory.mkdir(parents=True)
    (directory / "rank0-tp.json").write_text(json.dumps(receipt(0)))
    (directory / "rank0-world.json").write_text(json.dumps({"schema": "other"}))
    stale = directory / "rank0-dcp.json"
    stale.write_text(json.dumps(receipt(0, "dcp")))
    os.utime(stale, (1_000_000, 1_000_000))
    info = {"Id": "abc", "State": {"Running": True, "StartedAt": "2026-10-08T00:00:00.5Z"}}
    value = receipts.host_report(lock, 0, info, root=tmp_path, popen=Logs(["ready"]))
    assert [item["file"] for item in value["receipts"]] == ["rank0-tp.json"]
    assert value["problems"] == ["rank0-world.json is not a sircl-vllm-receipt/v1 receipt"]
    assert value["log"]["tail"] == ["ready"] and value["container"]["running"]
    plain = {key: item for key, item in lock.items() if key != "transport"}
    assert receipts.host_report(plain, 0, info, root=tmp_path) == {"schema": receipts.REPORT_SCHEMA, "rank": 0,
                                                                     "backend": "prepared"}


def test_dockers_start_time_reads_with_nanoseconds():
    assert receipts.started_epoch("1970-01-01T00:00:10.250000000Z") == pytest.approx(10.25)
    assert receipts.started_epoch("0001-01-01T00:00:00Z") is None and receipts.started_epoch(None) is None


class Runner:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def remote(self, rank, operation, **options):
        self.calls.append((rank, operation))
        answer = self.answers[rank]
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_node_a_reads_every_rank_and_records_one_verdict(tmp_path):
    lock, _ = sircl_deployment(TP2, "pair", 2, [0, 1])
    runner = Runner({0: report(0, [receipt(0)]), 1: RuntimeError("ssh failed\nspark1: Connection refused")})
    verdict = receipts.check(tmp_path, lock, runner, now=lambda: 0)
    assert sorted(runner.calls) == [(0, "transport-receipts"), (1, "transport-receipts")]
    assert verdict["verdict"] == "unknown" and verdict["problems"][0] == "rank 1: spark1: Connection refused"
    assert receipts.latest(tmp_path)["verdict"] == "unknown"


def test_the_rows_and_its_tables_session_settings_are_checked_against_every_session():
    lock, _ = sircl_deployment(TP2, "pair", 2, [0, 1])
    # The row sets the link slot; the table, which the session takes, records the link slots and chain slot.
    lock["transport"]["tuning"]["settings"] = {"link_slot": 1048576}
    lock["transport"]["tuning"]["tables"] = [{"path": "runtime/tables/pair.json", "sha256": "0" * 64, "hash": "1" * 16,
                                              "settings": {"SIRCL_LINK_SLOTS": 16, "SIRCL_LINK_SLOT_BYTES": 524288,
                                                           "SIRCL_CHAIN_SLOT_BYTES": 2097152}, "sessions": ["tp"]}]
    reports = pair_reports(tuning="1" * 16)
    for item in reports:
        item["receipts"][0]["session_stats"] = {"link_slots": 16, "link_slot_bytes": 1048576,
                                                "chain_slot_bytes": 2097152}
    verdict = receipts.evaluate(lock, reports, now=lambda: 0)
    assert verdict["verdict"] == "as-expected", verdict["problems"]
    assert ("tuning settings: the sessions report the row's and its table's chain_slot_bytes 2097152, "
            "link_slot_bytes 1048576, link_slots 16") in verdict["lines"]
    reports[1]["receipts"][0]["session_stats"]["link_slots"] = 8
    verdict = receipts.evaluate(lock, reports, now=lambda: 0)
    assert "rank 1: the session's link_slots is 8, the tuning row and its table set 16" in verdict["problems"]
    reports[1]["receipts"][0].pop("session_stats")
    verdict = receipts.evaluate(lock, reports, now=lambda: 0)
    assert verdict["verdict"] == "as-expected"
    assert "tuning settings of rank 1: its receipt states no session statistics (not judged)" in verdict["lines"]


def test_each_receipt_names_the_deployments_nccl_mode_and_the_verdict_states_the_rule():
    lock, _ = sircl_deployment(TP2, "pair", 2, [0, 1])
    reports = pair_reports()
    verdict = receipts.evaluate(lock, reports, now=lambda: 0)
    assert verdict["nccl_rule"] == "NCCL: opt-in only (auto); tables choose among SIRCL options"
    assert any(line.startswith("NCCL mode: not stated by the receipts of rank 0 group tp") for line in verdict["lines"])
    for item in reports:
        item["receipts"][0].update(nccl_mode="never", nccl_rule=verdict["nccl_rule"])
    verdict = receipts.evaluate(lock, reports, now=lambda: 0)
    assert verdict["verdict"] == "as-expected", verdict["problems"]
    assert ("NCCL mode: every receipt names never (NCCL: opt-in only (auto); tables choose among SIRCL options)"
            in verdict["lines"])
    reports[1]["receipts"][0]["nccl_mode"] = "auto"
    verdict = receipts.evaluate(lock, reports, now=lambda: 0)
    assert "rank 1 group tp: the receipt names NCCL mode auto, the deployment sets never" in verdict["problems"]


def builtin_stats(shape):
    """Session statistics of a session that took SIRCL's built-in plan for ``shape``."""
    return {"tuning": {"table": "f" * 16, "path": f"builtin:{shape}",
                       "key": {"shape": shape, "world": 8, "lanes": 2, "max_relays": 3},
                       "decisions": {"all_reduce/eager/ring piece 524288": 161}}}


def test_a_session_on_its_shapes_built_in_plan_matches_a_plan_without_a_measured_table():
    from runtime.common.test_transport import eight_spark_image
    lock, section = sircl_deployment("glm53-nvfp4-tp8", "cycle", 8, list(range(8)), image=eight_spark_image())
    assert receipts.transport.session_table(section) is None

    def reports(stats):
        expected = receipts.transport.expected_session_settings(section)
        return [report(rank, [dict(receipt(rank, tuning="f" * 16, stats={**expected, **stats}), world=8),
                              dict(receipt(rank, group=f"dcp:{rank // 4}"), world=4)]) for rank in range(8)]

    verdict = receipts.evaluate(lock, reports(builtin_stats("cycle:8")), now=lambda: 0)
    assert verdict["verdict"] == "as-expected", verdict["problems"]
    assert any(line.startswith("tuning: SIRCL's built-in plan for cycle:8 (table ffffffffffffffff) decides for "
                               "rank 0 group tp, rank 1 group tp") for line in verdict["lines"]), verdict["lines"]
    # A measured table the plan did not match, or a built-in source for a shape without a built-in plan,
    # is still another table.
    for stats in ({"tuning": {"table": "f" * 16, "path": "/tables/cycle8.json"}}, builtin_stats("cycle:6")):
        verdict = receipts.evaluate(lock, reports(stats), now=lambda: 0)
        assert any(problem.startswith("rank 0 group tp: tuning table ffffffffffffffff, the plan matched none")
                   for problem in verdict["problems"]), verdict["problems"]


def test_with_decode_context_parallelism_every_rank_needs_its_decode_context_parallel_receipt():
    from runtime.common.test_transport import eight_spark_image
    lock, section = sircl_deployment("glm53-nvfp4-tp8", "cycle", 8, list(range(8)), image=eight_spark_image())
    assert section["group"]["dcp"] == 4
    tensor = [report(rank, [dict(receipt(rank), world=8)]) for rank in range(8)]
    verdict = receipts.evaluate(lock, tensor, now=lambda: 0)
    assert verdict["verdict"] == "differs"
    assert "rank 3: no decode-context-parallel receipt with a SIRCL session (--dcp-size 4)" in verdict["problems"]
    both = [report(rank, [dict(receipt(rank), world=8), dict(receipt(rank, group=f"dcp:{rank // 4}"), world=4)])
            for rank in range(8)]
    verdict = receipts.evaluate(lock, both, now=lambda: 0)
    assert verdict["verdict"] == "as-expected", verdict["problems"]
    # The section matched no table for the decode-context-parallel sessions, so their rules choose.
    both[2]["receipts"][1]["tuning"] = "2" * 16
    verdict = receipts.evaluate(lock, both, now=lambda: 0)
    assert any(problem.startswith("rank 2 group dcp:0: tuning table 2222222222222222, the plan matched none")
               for problem in verdict["problems"]), verdict["problems"]
    # With a table for them, each decode-context-parallel session must decide from it.
    lock["transport"]["tuning"]["tables"] = [{"path": "runtime/tables/dcp4.json", "sha256": "0" * 64,
                                              "hash": "2" * 16, "settings": {}, "sessions": ["dcp"]}]
    verdict = receipts.evaluate(lock, both, now=lambda: 0)
    assert any(problem.startswith("rank 0 group dcp:0: tuning table none (the rules choose), the plan matched "
                                  "2222222222222222") for problem in verdict["problems"]), verdict["problems"]
    for rank in range(8):
        both[rank]["receipts"][1]["tuning"] = "2" * 16
    verdict = receipts.evaluate(lock, both, now=lambda: 0)
    assert verdict["verdict"] == "as-expected", verdict["problems"]
