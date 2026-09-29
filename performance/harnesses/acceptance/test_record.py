"""Record naming, README rounding, text sanitization and record rendering."""
from __future__ import annotations

from pathlib import Path

import pytest

from performance.harnesses.acceptance import install, record, throughput
from performance.harnesses.acceptance.profile_info import ProfileInfo

LAN = ".".join(("192", "168", "0", "200"))
FABRIC = ".".join(("10", "0", "1", "2"))
ROOT = Path(__file__).resolve().parents[3]
IMAGE = "dev-20260927-b12xcache-cuda1342-nccl2323-status032"
PROFILE = ProfileInfo(
    id="mimo-v26-flash-mopd-tp2", title="MiMo-V2.6-Flash-MOPD on two Sparks", status="implemented",
    served_model_name="MiMo-V2.6-Flash-MOPD-TP2", port=8020, nodes=2, topology="direct-pair-2",
    repository="XiaomiMiMo/MiMo-V2.6-Flash-MOPD", revision="0123456789abcdef" * 2 + "01234567", image=IMAGE,
    release=f"runtime/releases/{IMAGE}/release.json", features=frozenset({"tools", "image", "reasoning"}),
    thinking_off={"chat_template_kwargs": {"enable_thinking": False}})


def summary(rates=(27.0853, 98.4, 139.0267, 205.5873), prefill=(2675.0, 2749.0, 2040.0), runs=1):
    decode = {str(c): {"aggregate_tps": r, "server_steps_per_s": s, "server_spec_accept_length": a, "num_errors": 0,
                       "missing_runs": 0}
              for c, r, s, a in zip(throughput.CONCURRENCY, rates, (12.64, 30.1, 42.9008, 64.199),
                                    (2.1429, 3.0, 3.2407, 3.2023))}
    return {"runs": runs, "versions": ["0.6.2"], "decode": decode,
            "prefill": dict(zip(map(str, throughput.PREFILL), prefill)), "invalid": [], "ok": True,
            "files": ["tp2-matrix.json"] if runs == 1 else [f"tp2-matrix-run{i}.json" for i in range(1, runs + 1)]}


def test_record_name_uses_the_image_prefix():
    assert record.image_short(IMAGE) == "dev-20260927-b12xcache"
    assert record.image_short("custom-image") == "custom-image"
    assert record.record_name(IMAGE, "mimo-v26-tp2", "20260927") == "dev-20260927-b12xcache-mimo-v26-tp2-20260927"
    for topic, date in (("Bad Topic", "20260927"), ("ok", "2026-09-27"), ("../x", "20260927")):
        with pytest.raises(ValueError):
            record.record_name(IMAGE, topic, date)


@pytest.mark.parametrize("rates, prefill, expected", [
    ((56.2, 150.5, 196.6, 283.3), 3666.0, ("56.2 / 151 / 197 / 283", "3,666")),
    ((63.75, 149.49, 196.5, 279.5), 4284.4, ("63.8 / 149 / 197 / 280", "4,284")),
    ((27.0853, 98.4, 139.0267, 205.5873), 2749.0, ("27.1 / 98 / 139 / 206", "2,749")),
    ((40.2, 80.0, 104.0, None), None, ("40.2 / 80 / 104 / —", "—")),
])
def test_readme_values_round_half_up_like_the_readme(rates, prefill, expected):
    data = summary(rates=rates, prefill=(1.0, prefill, 1.0))
    assert record.readme_values(data) == expected


def test_readme_line_names_profile_and_port():
    assert record.readme_line(PROFILE, summary()) == \
        "README values for `mimo-v26-flash-mopd-tp2` (port 8020): decode 27.1 / 98 / 139 / 206; prefill 64K 2,749"


def test_throughput_row_matches_the_record_table_format():
    assert record.throughput_row(summary()) == \
        "| 27.1 / 98.4 / 139.0 / 205.6 | 12.6 / 30.1 / 42.9 / 64.2 | 2.14 / 3.00 / 3.24 / 3.20 | 2,675 / 2,749 / 2,040 |"


def test_text_sanitization_replaces_known_hosts_and_numbers_other_addresses():
    text = f"Model ready: http://{LAN}:8020/v1\nNode 1 via {FABRIC} and {FABRIC}; spark-r0 done\n"
    clean = record.sanitize_text(text, replace={LAN: "NODE_A", "spark-r0": "NODE_A"}, names=["client-box"],
                                 users=["code"])
    assert clean == "Model ready: http://NODE_A:8020/v1\nNode 1 via ADDRESS_1 and ADDRESS_1; NODE_A done\n"


@pytest.mark.parametrize("text", ["ran on client-box", "see /home/code/out", f"ssh code@{LAN.replace('192', 'x')}"])
def test_text_sanitization_refuses_names_and_accounts(text):
    with pytest.raises(throughput.PrivateDataError):
        record.sanitize_text(text, replace={}, names=["client-box"], users=["code"])


def functional(failed=()):
    names = ["count", "arithmetic", "code", "tool call", "forced tool call", "image", "thinking on"]
    rows = [{"name": n, "status": "FAIL" if n in failed else "PASS", "detail": "x"} for n in names]
    passed = sum(r["status"] == "PASS" for r in rows)
    return {"checks": rows, "passed": passed, "failed": len(rows) - passed, "skipped": 0, "ok": not failed}


STRESS = {"rounds": 8, "n": 256, "degenerate": 0, "wrong": 2, "errors": 0, "seconds": 95.4, "wrong_ids": ["a7"],
          "ok": True}
INSTALL = {"state": "complete", "ok": True, "only_document": True, "api_ready_seconds": 545.3,
           "source_revision": "c17cf23cec72" + "0" * 28}


def render(**changes):
    arguments = dict(profile=PROFILE, name="dev-20260927-b12xcache-mimo-v26-flash-mopd-tp2-20260927",
                     record_dir=ROOT / "performance/records/images", repo_root=ROOT,
                     files={"functional": "functional.txt", "stress": "stress.json", "matrices": ["tp2-matrix.json"],
                            "install_phases": "install-phases.txt"},
                     install=INSTALL, source=install.Source("published", "main"), functional=functional(),
                     stress=STRESS, summary=summary(), client="a separate machine on Node A's network",
                     harness_revision="abcdef123456")
    arguments.update(changes)
    return record.render(**arguments)


def test_record_has_the_evidence_sections_and_values():
    text = render()
    assert text.splitlines()[1] == "# MiMo-V2.6-Flash-MOPD on two Sparks with the installer image"
    assert ("Status: **implemented; all 7 functional checks passed; a 256-request correctness screen returned no "
            "degenerate or failed response; measured on one pair; single-run timing; not serving-qualified**.") in text
    for heading in ("## Conditions", "## Measurement", "## Result", "## Conclusion", "## Limitations"):
        assert f"\n{heading}\n" in text
    assert "installed source commit `c17cf23cec72`" in text
    assert f"[`release.json`](../../../runtime/releases/{IMAGE}/release.json)" in text
    assert "[installer phases](dev-20260927-b12xcache-mimo-v26-flash-mopd-tp2-20260927/install-phases.txt)" in text
    assert "Node 0's API readiness step took 545.3 s" in text
    assert "| 27.1 / 98.4 / 139.0 / 205.6 | 12.6 / 30.1 / 42.9 / 64.2 | 2.14 / 3.00 / 3.24 / 3.20 | 2,675 / 2,749 / 2,040 |" in text
    assert "| Decode 1 / 4 / 8 / 16 streams at 16K (tok/s) | Steps/s |" in text
    assert "decode 1 / 4 / 8 / 16 users at 16K context 27.1 / 98 / 139 / 206 tok/s, prefill 64K 2,749 tok/s" in text
    assert "**Full matrix**" not in text
    assert "0 degenerate, 0 failed and 2 wrong, to questions `a7`" in text
    assert "`{\"chat_template_kwargs\": {\"enable_thinking\": false}}`" in text
    assert "Each cell ran once." in text


def test_record_links_resolve_from_the_record_directory():
    text = render()
    base = ROOT / "performance/records/images"
    for target in ("../../../runtime/releases/", "../../../performance/harnesses/acceptance/accept_profile.py"):
        assert target in text
    assert (base / f"../../../runtime/releases/{IMAGE}/release.json").resolve().is_file()
    assert (base / "../../../performance/harnesses/acceptance/accept_profile.py").resolve().is_file()


def test_record_without_installation_or_throughput_says_so():
    text = render(install=None, source=None, summary=None, functional=functional(failed=("image",)), stress=None,
                  files={"functional": "functional.txt"}, status="research-only")
    assert "Status: **research-only; 6 of 7 functional checks passed; measured on one pair; not serving-qualified**." in text
    assert "This run installed nothing" in text and "- **Installation:** none in this run." in text
    assert "failed: image" in text and "- Throughput was not measured." in text
    assert "## Result\n\n**Correctness.** 6 of 7" in text


def test_repeated_runs_report_medians():
    text = render(summary=summary(runs=3), files={"functional": "functional.txt", "stress": "stress.json",
                                                   "matrices": [f"tp2-matrix-run{i}.json" for i in (1, 2, 3)]})
    assert "median of 3 benchmark runs" in text and "each value's median and the sum of request errors" in text
    assert "matrices: [run 1](" in text


def test_invalid_status_is_refused():
    with pytest.raises(ValueError):
        render(status="validated")


def full(kv_budget=2_000_000):
    decode = {str(context): {str(level): {"aggregate_tps": 1000.0 * level / (context // 1024)}
                             for level in throughput.FULL_CONCURRENCY}
              for context in throughput.FULL_CONTEXTS}
    decode["131072"]["16"] = {"aggregate_tps": None, "not_applicable": throughput.NOT_FITTING}
    return {"version": "0.6.2", "kv_budget": kv_budget, "decode": decode, "file": "tp2-full-matrix.json"}


def test_full_matrix_lines_print_a_table_and_explain_each_dash():
    lines = record.full_matrix_lines(full(), "[matrix](x/tp2-full-matrix.json)")
    assert lines[0] == "**Full matrix** ([matrix](x/tp2-full-matrix.json)), decode tok/s by added context and streams:"
    assert lines[2:4] == ["| Context | 1 | 2 | 4 | 8 | 16 |", "|---|---|---|---|---|---|"]
    assert lines[4] == "| 8K | 125.0 | 250.0 | 500.0 | 1000.0 | 2000.0 |"
    assert lines[7] == "| 128K | 7.8 | 15.6 | 31.3 | 62.5 | — |"
    assert lines[-1] == "A dash: exceeds the KV cache (2,000,000 tokens)."
    data = full(kv_budget=None)
    data["decode"]["65536"]["16"] = {"aggregate_tps": None, "not_applicable": throughput.QUEUED}
    assert record.full_matrix_lines(data, "m")[-1] ==         f"A dash: exceeds the KV cache; {throughput.QUEUED}."
    complete = full()
    complete["decode"]["131072"]["16"] = {"aggregate_tps": 100.0}
    assert not any(line.startswith("A dash") for line in record.full_matrix_lines(complete, "m"))


def test_record_with_the_full_matrix_adds_its_table():
    text = render(full=full(), files={"functional": "functional.txt", "stress": "stress.json",
                                      "matrices": ["tp2-matrix.json"], "full_matrix": "tp2-full-matrix.json",
                                      "install_phases": "install-phases.txt"})
    assert "- **Full matrix:** the same benchmark settings at 1, 2, 4, 8 and 16 streams" in text
    assert ("**Full matrix** ([matrix](dev-20260927-b12xcache-mimo-v26-flash-mopd-tp2-20260927/"
            "tp2-full-matrix.json)), decode tok/s") in text
    assert "A dash: exceeds the KV cache (2,000,000 tokens)." in text
