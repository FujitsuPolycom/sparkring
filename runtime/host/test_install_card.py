"""Completion cards built by install, with only host I/O and the clock simulated."""
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from runtime.host import colour, dashboard_admin, install_workflow as flow, progress, start_checks
from runtime.host.test_colour import Terminal
from runtime.host.test_install_sircl import install, sircl  # noqa: F401
from runtime.host.test_install_workflow import PROFILE, machine, sparks  # noqa: F401


@pytest.fixture
def completed_card(sircl, monkeypatch, capsys):  # noqa: F811
    _, lock, _ = sircl
    original = flow.retained_source.apply

    def apply(path, operation, **kwargs):
        result = original(path, operation, **kwargs)
        if operation == "verify":
            # The rank smoke action saves the checks in its receipt; exercise
            # that receipt reader and execute's result construction unchanged.
            status = {"workers": {"ranks": [{"effective": {
                "max_model_len": {"state": "known", "value": 1048576},
                "max_num_seqs": {"state": "known", "value": 8}}}]}}
            checks = start_checks.judge(None, status,
                                       'vllm:cache_config_info{kv_cache_size_tokens="12700000"} 1\n',
                                       None, {}, checks=())
            receipt = {"actions": {"smoke:0": {"result": {"stdout": json.dumps({"start_checks": checks})}}}}
            (Path(path) / "card-smoke.json").write_text(json.dumps(receipt))
            (Path(path) / "state.json").write_text(json.dumps({"receipt": "card-smoke.json"}))
        return result

    monkeypatch.setattr(flow.retained_source, "apply", apply)
    ticks = iter((100.0, 460.0))
    monkeypatch.setattr(flow, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    monkeypatch.setattr(dashboard_admin, "page_url", lambda: "http://192.0.2.50:8090/")
    assert install(lock, "--no-auto-recover") == 0
    result = json.loads(capsys.readouterr().out)
    return result, ["Model ready: " + flow.model_ready_line(result), *flow.summary_lines(result)]


def test_card_uses_install_facts_and_smoke_observations(completed_card):
    result, marked = completed_card
    lines = [colour.plain(line) for line in marked]
    assert lines[0] == f"Model ready: {result['model']} on Sparks 0-1 (spark0, spark1) · 6 min"
    assert lines[1] == f"  Profile:         {PROFILE} · TP2 · SIRCL"
    assert lines[2] == "  Capacity:        1,048,576 context · 13M KV tokens · up to 8 requests"
    assert lines[3] == f'  API:             {result["api_url"]}   (model "{result["model"]}")'
    assert lines[4] == f"  Model status:    {result['dashboard_url']}"
    assert lines[5] == "  Ring dashboard:  http://192.0.2.50:8090/"
    assert f"  Start again:     {result['checkpoint']['command']}" in lines
    assert "  Recovery:        off → sudo sparkring recover on" in lines
    assert result["commands"]["switch_back"] is None
    saved = json.loads((Path(result["deployment"]) / flow.RESULT_FILE).read_text())
    assert saved == result


def test_live_and_replayed_commands_match_and_logs_are_plain(completed_card, monkeypatch):
    _, marked = completed_card
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("SPARKRING_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    colour.set_mode(None)
    replay = colour.CardReplay()
    expected = []
    for line in marked:
        live = colour.render(line + "\n", Terminal())
        assert colour.render(replay.line(colour.plain(line) + "\n"), Terminal()) == live
        assert colour.render(line, io.StringIO()) == colour.plain(line)
        expected.extend(re.findall(re.escape(colour.STYLE) + "(.*?)" + re.escape(colour.RESET), live))
    assert len(expected) == 7  # curl, logs, stop, start, recovery, uninstall's two commands
    log = (progress.directory() / "install.log").read_bytes()
    assert not any(byte < 0x20 and byte != 0x0a for byte in log)

    # Read the actual log produced by install, including a tail whose heading
    # is outside the displayed window and appended output under --follow.
    for options, env, tty in (([], {}, True), (["--plain"], {}, True),
                              (["--details"], {}, True), ([], {"NO_COLOR": ""}, True),
                              (["--color=never"], {}, True), ([], {}, False)):
        with monkeypatch.context() as patch:
            for key, value in env.items():
                patch.setenv(key, value)
            stream = Terminal() if tty else io.StringIO()
            patch.setattr(progress.sys, "stdout", stream)
            assert progress.main(["--lines", "100", *options]) == 0
            text = stream.getvalue()
            if options or env or not tty:
                assert "\x1b" not in text and "\x00" not in text
            else:
                assert re.findall(re.escape(colour.STYLE) + "(.*?)" + re.escape(colour.RESET), text) == expected

    stream = Terminal()
    monkeypatch.setattr(progress.sys, "stdout", stream)
    path = progress.directory() / "install.log"
    # Seed the heading, append the rest while the follower waits, then stop.
    path.write_text(colour.plain(marked[0]) + "\n")
    appended = False

    def append_or_stop(_):
        nonlocal appended
        if appended:
            raise KeyboardInterrupt
        with path.open("a") as target:
            target.write("\n".join(colour.plain(line) for line in marked[1:]) + "\n")
        appended = True

    monkeypatch.setattr(progress.time, "sleep", append_or_stop)
    assert progress.main(["--follow", "--lines", "1"]) == 0
    assert re.findall(re.escape(colour.STYLE) + "(.*?)" + re.escape(colour.RESET), stream.getvalue()) == expected
    stream.seek(0)
    stream.truncate()
    assert progress.main(["--lines", "5"]) == 0
    assert colour.STYLE + "sudo apt remove sparkring" + colour.RESET in stream.getvalue()
    colour.set_mode(None)


@pytest.mark.parametrize("tokens, expected", [(9500000, "9.5M"), (10000000, "10M"), (12700000, "13M")])
def test_observed_kv_count_rounding(tokens, expected):
    checks = start_checks.judge(None, None, f'vllm:cache_config_info{{kv_cache_size_tokens="{tokens}"}} 1\n',
                               None, {}, checks=())
    assert flow._capacity_text(checks) == expected + " KV tokens"


def test_unknown_capacity_and_non_card_commands_stay_plain():
    assert start_checks.judge(None, None, None, None, {}, checks=())["capacity"] == {}
    replay = colour.CardReplay()
    for line in ("    curl http://example.invalid/\n", "  Uninstall: sudo apt remove sparkring\n"):
        assert replay.line(line) == line


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="POSIX pseudo-terminal required")
def test_logs_on_a_real_tty_and_pipe(completed_card, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("SPARKRING_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    command = [sys.executable, "-m", "scripts.sparkring", "logs", "--lines", "100"]
    pipe = subprocess.run(command, check=True, capture_output=True)
    assert b"\x1b" not in pipe.stdout and b"\x00" not in pipe.stdout
    master, slave = os.openpty()
    try:
        subprocess.run(command, check=True, stdout=slave, stderr=subprocess.PIPE, timeout=20)
        os.close(slave)
        slave = None
        chunks = []
        while True:
            try:
                part = os.read(master, 65536)
            except OSError:
                break
            if not part:
                break
            chunks.append(part)
        tty = b"".join(chunks)
        assert colour.STYLE.encode() + b"curl " in tty
        assert colour.STYLE.encode() + b"sudo apt remove sparkring" + colour.RESET.encode() in tty
    finally:
        os.close(master)
        if slave is not None:
            os.close(slave)
