"""Offline authenticated-channel lifecycle and data-boundary tests."""
import io
import json
import os
from pathlib import Path

from runtime.host import dashboard, telemetry_live as live
from runtime.host.test_dashboard import PAIRS, collector
from runtime.host.test_dashboard_rates import document, member


def snapshot(at=1000):
    return member(document(at, 10, 30, plugin=True))["plugin_counters"]


def test_only_configured_sparks_once_per_group(tmp_path):
    controller = tmp_path / "controller"
    controller.mkdir()
    (tmp_path / "dashboard").mkdir()
    hosts = [{"host": f"root@192.0.2.{n+10}"} for n in range(8)]
    (controller / "cluster.json").write_text(json.dumps({"plan": {"spec": {"hosts": hosts}}}))
    rows = [{"positions": [0, 1], "deployment_id": "tp2"},
            {"positions": [2, 3], "deployment_id": "dcp2"},
            {"positions": [4, 5], "deployment_id": "stopped", "stopped": True}]
    (tmp_path / "dashboard/site.json").write_text(json.dumps({"deployments": rows}))
    assert live.heads(controller) == ({hosts[n]["host"]: ["tp2" if n < 2 else "dcp2"] for n in range(4)}, hosts[0]["host"])


def test_lease_quiet_disable_expiry_and_future(tmp_path):
    lease, quiet, disabled = [tmp_path / s for s in ("lease", "quiet", "disabled")]
    def active(at):
        return live.active(at, lease, quiet, disabled)
    assert not active(1000)
    lease.touch()
    os.utime(lease, (1000, 1000))
    assert active(1000) and active(1006)
    assert not active(1007) and not active(999)
    quiet.write_text("quiet")
    assert not active(1001)
    quiet.unlink()
    disabled.touch()
    assert not active(1001)


def test_forwarding_reuses_channel_local_reads_and_stops(tmp_path):
    opened, closed, pulses = [], [], []
    class Channel:
        def __init__(self, host):
            opened.append(host)
            self.host = host
        def sample(self):
            pulses.append(self.host)
            return {"dcp2": snapshot(), "other": snapshot(), "bad": {}}
        def close(self):
            closed.append(self.host)
    reads = []
    output = tmp_path / "live.json"
    f = live.Forwarder(channel=Channel, read=lambda: reads.append(True) or {"tp2": snapshot()}, output=output)
    for at in (1000, 1002, 1004):
        f.cycle({"local": ["tp2"], "remote": ["dcp2"]}, "local", at)
    result = json.loads(output.read_text())
    assert set(result["telemetry"]) == {"tp2", "dcp2"}
    assert result["telemetry"]["dcp2"]["gen"] == 30
    assert opened == ["remote"] and pulses == ["remote"] * 3 and len(reads) == 3
    f.cycle({}, "local", 1006)
    assert closed == ["remote"] and not output.exists()
    f.cycle({}, "local", 1008)
    assert len(reads) == 3 and len(pulses) == 3


def test_failed_channel_backs_off_without_affecting_local(tmp_path):
    attempts = []
    def fail(host):
        attempts.append(host)
        raise OSError("unreachable")
    f = live.Forwarder(channel=fail, read=lambda: {"tp2": snapshot()}, output=tmp_path / "live")
    for at in (1000, 1002, 1004):
        f.cycle({"local": ["tp2"], "remote": ["dcp2"]}, "local", at)
    assert attempts == ["remote"]
    assert "tp2" in json.loads(f.output.read_text())["telemetry"]


def test_stream_is_bounded_no_inventory_and_ends_on_eof_or_silence():
    reads, sink, clock = [], io.StringIO(), iter([0, 0, 2, 2])
    live.stream(input_stream=io.StringIO("\n\n"), output_stream=sink,
                read=lambda: reads.append(True) or {}, wait=lambda *a: ([1], [], []), clock=lambda: next(clock))
    assert len(reads) == 2 and sink.getvalue() == "{}\n{}\n"
    live.stream(input_stream=io.StringIO("\n"), output_stream=sink,
                read=lambda: reads.append(True) or {}, wait=lambda *a: ([], [], []))
    assert len(reads) == 2


def test_live_snapshot_wins_over_minute_cache_then_falls_back(tmp_path):
    found, api, clock = collector(tmp_path, rows=PAIRS[:1])
    found.live_path, found.viewer_path = tmp_path / "live", tmp_path / "viewer"
    found.mark_viewed()
    found.cycle()
    model = next(iter(found.models.values()))
    model.row["deployment_id"] = "one"
    # A still-valid minute cache must not displace a newer head snapshot.
    found.node_status = {"telemetry": {"one": snapshot(clock.now - 40)}}
    value = snapshot(clock.now)
    found.live_path.write_text(json.dumps({"at": clock.now, "telemetry": {"one": value}}))
    found.cycle()
    assert found.plugin_snapshot(model, clock.now) == value
    clock.now += 14
    found.mark_viewed()
    found.cycle()
    assert found.plugin_snapshot(model, clock.now) is None
    assert model.metrics_at == clock.now
    model.row["stopped"] = True
    found.cycle()
    assert model.state == "stopped" and model.metrics is None
    assert found.viewer_path.exists()


def test_agent_transport_preserves_existing_authentication():
    from runtime.host import discovery
    argv = discovery.ssh_argv("root@192.0.2.10", ["/usr/bin/sparkring", "node", "telemetry"])
    assert argv == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    "root@192.0.2.10", "/usr/bin/sparkring node telemetry"]
    assert "StrictHostKeyChecking=no" not in argv


def test_service_and_agent_wiring():
    unit = Path(dashboard.__file__).parents[2] / "packaging/debian/sparkring-dashboard.service"
    assert "RuntimeDirectory=sparkring-dashboard" in unit.read_text()



def test_newer_idle_metrics_displace_busy_plugin(tmp_path):
    found, _, clock = collector(tmp_path, rows=PAIRS[:1])
    found.mark_viewed()
    found.cycle()
    model = next(iter(found.models.values()))
    model.row["deployment_id"] = "one"
    found.live_telemetry = {"one": snapshot(clock.now - 2)}
    model.observe_metrics({"gen": 300, "prompt": 100, "running": 0, "waiting": 0}, clock.now)
    assert found.plugin_snapshot(model, clock.now) is None


def test_real_agent_stream_exits_when_renewals_stop():
    import subprocess
    import sys
    import pytest
    if sys.platform == "win32":
        pytest.skip("agent stdin select is a Linux service contract")
    program = ("from runtime.host import telemetry_live as t; "
               "t.TTL = .15; t.stream(read=lambda: {})")
    child = subprocess.Popen([sys.executable, "-u", "-c", program], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        child.stdin.write(b"\n")
        child.stdin.flush()
        assert child.stdout.readline() == b"{}\n"
        assert child.wait(timeout=3) == 0
        assert child.stdout.read() == b""
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate()



def test_channel_receives_a_real_local_pipe_and_closes_its_worker():
    import subprocess
    import sys
    import time
    import pytest
    if sys.platform == "win32":
        pytest.skip("agent stdin select is a Linux service contract")
    commands = []
    def spawn(argv, **kwargs):
        commands.append(argv)
        return subprocess.Popen([sys.executable, "-u", "-c",
            "from runtime.host import telemetry_live as t; t.stream(read=lambda: {})"], **kwargs)
    channel = live.Channel("root@192.0.2.10", popen=spawn)
    try:
        before = channel.updated
        assert channel.sample() == {}
        deadline = time.monotonic() + 3
        while channel.updated == before and time.monotonic() < deadline:
            time.sleep(.01)
        assert channel.updated > before
        assert len(commands) == 1 and commands[0][-1] == "/usr/bin/sparkring node telemetry"
    finally:
        channel.close()
    assert channel.process.poll() is not None and not channel.reader.is_alive()
