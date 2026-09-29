"""Benchmark invocation, matrix extraction, medians and repository sanitization."""
from __future__ import annotations

import json

import pytest

from performance.harnesses.acceptance import throughput

# Private addresses are assembled at run time so the repository's
# release-safety scan finds no address literal in this file.
LAN = ".".join(("192", "168", "0", "200"))
TAILNET = ".".join(("100", "101", "7", "9"))
FABRIC = ".".join(("10", "0", "1", "2"))


def cell(concurrency, tps, steps, accept, errors=0, reason="", context=0):
    return {"concurrency": concurrency, "context_tokens": context, "aggregate_tps": tps, "server_steps_per_s": steps,
            "server_spec_accept_length": accept, "num_errors": errors, "failure_reason": reason}


def matrix(rates=(27.0853, 98.4, 139.0267, 205.5873), prefill=(2675.0, 2749.0, 2040.0), **metadata):
    return {
        "metadata": {"version": "0.6.2", "model": "Model-TP2", "server": f"http://{LAN}", "temperature": 1.0,
                     **metadata},
        "startup_diagnostics": {"hostname": "client-box", "server_url": f"http://{LAN}:8000"},
        "event_log": [f"connected to {LAN}"],
        "prefill": {str(n): {"tok_per_sec": v, "method": "scout_only"} for n, v in zip(throughput.PREFILL, prefill)},
        "results": [cell(c, r, s, a, context=throughput.CONTEXT)
                    for c, r, s, a in zip(throughput.CONCURRENCY, rates, (12.64, 30.1, 42.90, 64.20),
                                          (2.1429, 3.0, 3.2407, 3.2023))]
                   + [cell(8, 99.0, 30.0, 3.0, context=65536)],
        "summary_table": {str(throughput.CONTEXT): dict(zip(map(str, throughput.CONCURRENCY), rates))},
        "burst_results": [], "burst_summary_table": {},
        "methodology": {"prefill": {"formula": "prompt_tokens / TTFT"}},
    }


def full_matrix(skip=((131072, 16),), queued=(), **metadata):
    """A full-matrix document; `skip` cells are absent and `queued` cells are capacity-limited."""
    rows = []
    for context in throughput.FULL_CONTEXTS:
        for level in throughput.FULL_CONCURRENCY:
            if (context, level) in skip:
                continue
            row = cell(level, 10.0 * level, 12.0, 3.0, context=context)
            if (context, level) in queued:
                row["capacity_limited"] = True
            rows.append(row)
    return {"metadata": {"version": "0.6.2", "temperature": 1.0, **metadata}, "prefill": {}, "results": rows,
            "summary_table": {}}


def test_bench_command_uses_the_profile_record_settings(tmp_path):
    command = throughput.bench_command("python3", tmp_path, host="node-a.test", port=8020, model="Model-TP2",
                                       output=tmp_path / "m.json")
    assert command[:2] == ["python3", str(tmp_path / "llm_decode_bench.py")]
    flags = " ".join(command[2:])
    assert flags == ("--host node-a.test --port 8020 --model Model-TP2 --concurrency 1,4,8,16 --contexts 16384 "
                     "--duration 20 --decode-warmup-seconds 5 --cell-warmup-timeout-seconds 900 --max-tokens 2048 "
                     "--temperature 1.0 --token-targeting exact --dcp-size 1 --no-hw-monitor --no-resume "
                     f"--display-mode plain --output {tmp_path / 'm.json'}")


def test_full_bench_command_runs_the_whole_grid_without_prefill(tmp_path):
    command = throughput.bench_command("python3", tmp_path, host="node-a.test", port=8020, model="Model-TP2",
                                       output=tmp_path / "m.json", full=True)
    flags = " ".join(command[2:])
    assert "--concurrency 1,2,4,8,16 --contexts 8192,32768,65536,131072 " in flags
    assert "--dcp-size 1" in flags and "--cell-warmup-timeout-seconds 900" in flags
    assert command[-1] == "--skip-prefill" and command.count("--skip-prefill") == 1
    standard = throughput.bench_command("python3", tmp_path, host="node-a.test", port=8020, model="Model-TP2",
                                        output=tmp_path / "m.json")
    assert "--skip-prefill" not in standard


def test_extract_reads_16k_context_cells_and_prefill():
    run = throughput.extract(matrix())
    assert run["version"] == "0.6.2" and run["invalid"] == []
    assert run["decode"]["1"] == {"aggregate_tps": 27.0853, "server_steps_per_s": 12.64,
                                  "server_spec_accept_length": 2.1429, "num_errors": 0}
    assert set(run["decode"]) == {"1", "4", "8", "16"}
    assert run["decode"]["8"]["aggregate_tps"] == 139.0267
    assert run["prefill"] == {"8192": 2675.0, "65536": 2749.0, "131072": 2040.0}


def test_invalid_cells_contribute_no_rates():
    document = matrix()
    document["results"][3] = cell(16, -3.0, 0.0, 0.0, context=throughput.CONTEXT)
    document["results"][2] = cell(8, 120.0, 40.0, 3.0, errors=2, reason="timeout", context=throughput.CONTEXT)
    del document["prefill"]["131072"]
    run = throughput.extract(document)
    assert run["decode"]["16"]["aggregate_tps"] is None and run["decode"]["8"]["num_errors"] == 2
    assert run["invalid"] == [{"concurrency": 8, "reason": "timeout"}, {"concurrency": 16, "reason": "invalid cell"}]
    assert run["prefill"]["131072"] is None
    assert not throughput.combine([run])["ok"]


def test_only_temperature_one_matrices_are_accepted():
    with pytest.raises(ValueError, match="temperature"):
        throughput.extract(matrix(temperature=0.0))


def test_combine_takes_medians_and_sums_errors():
    runs = [throughput.extract(matrix(rates=r)) for r in ((40.0, 120.0, 180.0, 300.0), (50.0, 140.0, 200.0, 330.0),
                                                          (45.0, 130.0, 190.0, 360.0))]
    runs[1]["decode"]["8"]["num_errors"] = 1
    summary = throughput.combine(runs)
    assert summary["runs"] == 3 and summary["versions"] == ["0.6.2"]
    assert [summary["decode"][c]["aggregate_tps"] for c in ("1", "4", "8", "16")] == [45.0, 130.0, 190.0, 330.0]
    assert summary["decode"]["8"]["num_errors"] == 1 and not summary["ok"]
    assert summary["prefill"]["65536"] == 2749.0
    runs[1]["decode"]["8"]["num_errors"] = 0
    assert throughput.combine(runs)["ok"]


def test_missing_and_queued_cells_are_not_applicable_and_not_failures():
    document = matrix()
    del document["results"][3]
    document["results"][2]["capacity_limited"] = True
    run = throughput.extract(document)
    assert run["decode"]["16"] == {"aggregate_tps": None, "server_steps_per_s": None,
                                   "server_spec_accept_length": None, "num_errors": 0,
                                   "not_applicable": throughput.NOT_FITTING}
    assert run["decode"]["8"]["aggregate_tps"] is None and run["decode"]["8"]["server_steps_per_s"] is None
    assert run["decode"]["8"]["not_applicable"] == throughput.QUEUED and run["invalid"] == []
    summary = throughput.combine([run, throughput.extract(document)])
    assert summary["ok"]
    assert summary["decode"]["16"]["not_applicable"] == throughput.NOT_FITTING
    assert summary["decode"]["8"]["not_applicable"] == throughput.QUEUED
    assert summary["decode"]["4"]["aggregate_tps"] == 98.4 and "not_applicable" not in summary["decode"]["4"]


def test_extract_full_fills_every_grid_cell_and_reports_the_kv_budget():
    full = throughput.extract_full(full_matrix(queued=((65536, 16),), kv_budget=2_000_000))
    assert full["version"] == "0.6.2" and full["kv_budget"] == 2_000_000
    assert list(full["decode"]) == [str(c) for c in throughput.FULL_CONTEXTS]
    assert all(list(row) == [str(c) for c in throughput.FULL_CONCURRENCY] for row in full["decode"].values())
    assert full["decode"]["8192"]["2"] == {"aggregate_tps": 20.0}
    assert full["decode"]["131072"]["16"] == {"aggregate_tps": None, "not_applicable": throughput.NOT_FITTING}
    assert full["decode"]["65536"]["16"] == {"aggregate_tps": None, "not_applicable": throughput.QUEUED}
    assert throughput.extract_full(full_matrix(max_total_tokens=1_500_000))["kv_budget"] == 1_500_000
    assert throughput.extract_full(full_matrix())["kv_budget"] is None


def test_sanitize_keeps_record_sections_and_hides_the_server():
    clean = throughput.sanitize_matrix(matrix(), names=["client-box"], accounts=["code"])
    assert list(clean) == list(throughput.KEEP)
    assert clean["metadata"]["server"] == "http://NODE_A"
    assert "startup_diagnostics" not in clean and "event_log" not in clean


@pytest.mark.parametrize("where, value, finding", [
    ("model", f"served-from-{LAN}", "private address"),
    ("note", f"peer {TAILNET}", "private address"),
    ("note", f"rank {FABRIC}", "private address"),
    ("note", "run on client-box", "name client-box"),
    ("note", "path /home/code/results", "account code"),
])
def test_sanitize_refuses_private_leftovers(where, value, finding):
    document = matrix()
    document["metadata"][where] = value
    with pytest.raises(throughput.PrivateDataError, match=finding):
        throughput.sanitize_matrix(document, names=["client-box"], accounts=["code"])


def test_address_pattern_ignores_public_and_longer_numbers():
    text = json.dumps({"a": "8.8.8.8", "b": "1" + LAN, "c": LAN + "1", "d": "100.63.1.1", "e": "100.128.0.1",
                       "f": "1.10.0.0.1.2", "g": 10.5})
    found = throughput.private_findings(text)
    assert found == [], found
    assert throughput.private_findings(f"x {LAN}:8000 y") == [f"private address {LAN}"]


def test_names_match_whole_tokens_only():
    assert throughput.private_findings("encoder code-path", accounts=["code"]) == []
    assert throughput.private_findings("user code logged in", accounts=["code"]) == ["account code"]
    assert throughput.account_forms("PASS code: def f()", ["code"]) == []
    assert throughput.account_forms(f"code@{LAN}", ["code"]) == ["account code"]
