"""Offline contracts for the DeepSeek four-Spark cycle controller.

Covers command ordering (workers before the head), container-probe and
API-probe loops, idempotent stop, dry-run (no SSH execution), and argument
validation. Everything runs against a fake cluster inventory and a mocked
SSH layer; no cluster is contacted.
"""

from __future__ import annotations

import pathlib
import subprocess
from types import SimpleNamespace

import pytest
import shlex

import deepseek_v4_cycle_ctl as ctl


@pytest.fixture
def fake_ranks():
    """Four ranks: ids 0-3, SSH targets carry no real site identity."""
    return tuple(
        SimpleNamespace(id=i, ssh_target=f"user@rank{i}.example.invalid")
        for i in range(4)
    )


def _rank_of(target: str) -> int:
    return int(target.split("@")[1].lstrip("rank")[0])


@pytest.fixture
def sample_cluster(tmp_path: pathlib.Path) -> pathlib.Path:
    """Minimal but schema-valid cluster inventory for a four-Spark cycle."""
    text = """\
schema_version: 1
cluster:
  name: deepseek-v4-flash-four-rank-cycle
  description: offline test inventory for the cycle controller
topology:
  mtu: 9000
  link_speed_mbps: 200000
  edges:
  - id: r0-r1
    subnet: 192.0.2.0/24
    endpoints:
    - 0
    - 1
  - id: r1-r2
    subnet: 198.51.100.0/24
    endpoints:
    - 1
    - 2
  - id: r2-r3
    subnet: 203.0.113.0/24
    endpoints:
    - 2
    - 3
  - id: r3-r0
    subnet: 198.18.0.0/24
    endpoints:
    - 3
    - 0
ranks:
- id: 0
  ssh_target: operator@198.18.1.10
  management:
    interface: eth0
    address: 198.18.1.10
  ring_ports:
  - edge: r0-r1
    interface: eth1
    address: 192.0.2.10
    rdma_device: mlx5_0
    rdma_port: 1
    roce_gid_index: 3
  - edge: r3-r0
    interface: eth2
    address: 198.18.0.10
    rdma_device: mlx5_1
    rdma_port: 1
    roce_gid_index: 3
  transport_peers:
  - rank: 1
    address: 198.18.1.11
  - rank: 3
    address: 198.18.1.13
- id: 1
  ssh_target: operator@198.18.1.11
  management:
    interface: eth0
    address: 198.18.1.11
  ring_ports:
  - edge: r0-r1
    interface: eth1
    address: 192.0.2.11
    rdma_device: mlx5_0
    rdma_port: 1
    roce_gid_index: 3
  - edge: r1-r2
    interface: eth2
    address: 198.51.100.11
    rdma_device: mlx5_1
    rdma_port: 1
    roce_gid_index: 3
  transport_peers:
  - rank: 0
    address: 198.18.1.10
  - rank: 2
    address: 198.18.1.12
- id: 2
  ssh_target: operator@198.18.1.12
  management:
    interface: eth0
    address: 198.18.1.12
  ring_ports:
  - edge: r1-r2
    interface: eth1
    address: 198.51.100.12
    rdma_device: mlx5_0
    rdma_port: 1
    roce_gid_index: 3
  - edge: r2-r3
    interface: eth2
    address: 203.0.113.12
    rdma_device: mlx5_1
    rdma_port: 1
    roce_gid_index: 3
  transport_peers:
  - rank: 1
    address: 198.18.1.11
  - rank: 3
    address: 198.18.1.13
- id: 3
  ssh_target: operator@198.18.1.13
  management:
    interface: eth0
    address: 198.18.1.13
  ring_ports:
  - edge: r2-r3
    interface: eth1
    address: 203.0.113.13
    rdma_device: mlx5_0
    rdma_port: 1
    roce_gid_index: 3
  - edge: r3-r0
    interface: eth2
    address: 198.18.0.13
    rdma_device: mlx5_1
    rdma_port: 1
    roce_gid_index: 3
  transport_peers:
  - rank: 2
    address: 198.18.1.12
  - rank: 0
    address: 198.18.1.10
preflight:
  ssh_timeout_seconds: 45
  required_free_ports:
  - 8015
  - 29755
"""
    p = tmp_path / "cluster.yaml"
    p.write_text(text)
    return p


class FakeSSH:
    """Records every SSH command; lets tests stage container/API states."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.calls: list[str] = []
        self.containers_up: dict[int, bool] = {}
        self.api_ready = False

        def fake_run(ssh_target, command, timeout=None, capture=False):
            self.calls.append(command)
            rank = _rank_of(ssh_target)
            if "docker rm -f" in command:
                self.containers_up[rank] = False
                return subprocess.CompletedProcess(["ssh"], 0, "", "")
            if "docker ps" in command:
                up = self.containers_up.get(rank, False)
                return subprocess.CompletedProcess(
                    ["ssh"], 0, f"deepseek-v4-flash-r{rank}\n" if up else "", ""
                )
            if "curl -fsS" in command and "/v1/models" in command:
                return subprocess.CompletedProcess(
                    ["ssh"], 0 if self.api_ready else 7, "", ""
                )
            if "nohup" in command and "--run" in command:
                self.containers_up[rank] = True
                return subprocess.CompletedProcess(["ssh"], 0, "", "")
            return subprocess.CompletedProcess(["ssh"], 0, "", "")

        monkeypatch.setattr(ctl, "_run_ssh", fake_run)
        # Fake launches update state synchronously; polling needs no wall-clock delay.
        monkeypatch.setattr(ctl.time, "sleep", lambda _seconds: None)


def _start_order(calls):
    """Ranks launched, in call order."""
    launches = [c for c in calls if "nohup" in c and "--run" in c]
    return [int(c.split("rank-")[1].split(".")[0]) for c in launches]


def test_start_orders_workers_before_head(monkeypatch, fake_ranks):
    ssh = FakeSSH(monkeypatch)
    rc = ctl.start_ranks(fake_ranks, "/srv/sparkring", "deepseek-v4-flash-r",
                         "/tmp", dry_run=False, wait_api_minutes=0)
    assert rc == 0
    assert _start_order(ssh.calls) == [1, 2, 3, 0]


def test_start_launch_command_shape(fake_ranks):
    cmd = ctl._launch_command("/srv/sparkring", 2, "/tmp/ctl.log")
    assert "/srv/sparkring/scripts/deepseek_v4_cycle_serve.sh --run" in cmd
    assert "/srv/sparkring/rank-2.env" in cmd
    assert "nohup" in cmd and "</dev/null &" in cmd


def test_start_skips_running_rank(monkeypatch, fake_ranks):
    ssh = FakeSSH(monkeypatch)
    ssh.containers_up = {1: True}  # rank 1 already up
    rc = ctl.start_ranks(fake_ranks, "/srv/sparkring", "deepseek-v4-flash-r",
                         "/tmp", dry_run=False, wait_api_minutes=0)
    assert rc == 0
    assert _start_order(ssh.calls) == [2, 3, 0]  # rank 1 skipped


def test_start_waits_for_api(monkeypatch, fake_ranks):
    ssh = FakeSSH(monkeypatch)
    ssh.api_ready = True
    rc = ctl.start_ranks(fake_ranks, "/srv/sparkring", "deepseek-v4-flash-r",
                         "/tmp", dry_run=False, wait_api_minutes=40)
    assert rc == 0
    assert any("curl -fsS" in c for c in ssh.calls), "expected an API probe"


def test_start_aborts_when_container_never_appears(monkeypatch, fake_ranks):
    calls = []

    def never_up(ssh_target, command, timeout=None, capture=False):
        calls.append((ssh_target, command))
        if "docker ps" in command:
            rank = _rank_of(ssh_target)
            appeared = rank == 1 and any(
                target == ssh_target and "nohup" in prior
                for target, prior in calls
            )
            return subprocess.CompletedProcess(
                ["ssh"], 0, f"deepseek-v4-flash-r{rank}\n" if appeared else "", ""
            )
        if "docker rm -f" in command:
            return subprocess.CompletedProcess(["ssh"], 0, "", "")
        if "nohup" in command:
            return subprocess.CompletedProcess(["ssh"], 0, "", "")
        return subprocess.CompletedProcess(["ssh"], 0, "", "")

    monkeypatch.setattr(ctl, "_run_ssh", never_up)
    monkeypatch.setattr(ctl.time, "sleep", lambda s: None)
    rc = ctl.start_ranks(fake_ranks, "/srv/sparkring", "deepseek-v4-flash-r",
                         "/tmp", dry_run=False, wait_container=2,
                         wait_api_minutes=0)
    assert rc == 1
    assert any(
        _rank_of(target) == 1 and "docker rm -f" in command
        for target, command in calls
    )


def test_authenticated_api_probe_reads_key_on_head(monkeypatch):
    commands = []

    def probe(_ssh_target, command, timeout=None, capture=False):
        commands.append(command)
        return subprocess.CompletedProcess(["ssh"], 0, "", "")

    monkeypatch.setattr(ctl, "_run_ssh", probe)

    assert ctl._head_api_ready("operator@head", 8888, "/secure/api-keys")
    assert "/secure/api-keys" in commands[0]
    assert "Authorization: Bearer" in commands[0]


def test_status_reports_ssh_transport_failure(monkeypatch, fake_ranks, capsys):
    monkeypatch.setattr(
        ctl,
        "_run_ssh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["ssh"], 255, "", "connection refused"
        ),
    )

    assert ctl.status_ranks(fake_ranks, "deepseek-v4-flash-r", 8888) == 1
    assert "SSH ERROR" in capsys.readouterr().out


def test_stop_returns_failure_when_ssh_transport_fails(monkeypatch, fake_ranks):
    monkeypatch.setattr(
        ctl,
        "_run_ssh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["ssh"], 255, "", "connection refused"
        ),
    )

    assert ctl.stop_ranks(fake_ranks, "deepseek-v4-flash-r") == 1


def test_stop_removes_workers_first(monkeypatch, fake_ranks):
    ssh = FakeSSH(monkeypatch)
    ssh.containers_up = {0: True, 1: True, 2: True, 3: True}
    rc = ctl.stop_ranks(fake_ranks, "deepseek-v4-flash-r")
    assert rc == 0
    removes = [c for c in ssh.calls if "docker rm -f" in c]
    names = [c.split("docker rm -f ")[1].split()[0] for c in removes]
    assert names == ["deepseek-v4-flash-r1", "deepseek-v4-flash-r2",
                     "deepseek-v4-flash-r3", "deepseek-v4-flash-r0"]


def test_status_reports_up_and_api(monkeypatch, fake_ranks, capsys):
    ssh = FakeSSH(monkeypatch)
    ssh.containers_up = {0: True, 1: False, 2: True, 3: True}
    ssh.api_ready = True
    rc = ctl.status_ranks(fake_ranks, "deepseek-v4-flash-r", 8888)
    out = capsys.readouterr().out
    assert rc == 1
    assert "rank0" in out and "UP" in out
    assert "rank1" in out and "down" in out
    assert "200 OK" in out


def test_dry_run_executes_no_ssh(monkeypatch, fake_ranks):
    def fake_run(argv, **kwargs):
        raise AssertionError("dry-run must not invoke SSH")

    monkeypatch.setattr(ctl, "_run_ssh", fake_run)
    rc = ctl.start_ranks(fake_ranks, "/srv/sparkring", "deepseek-v4-flash-r",
                         "/tmp", dry_run=True)
    assert rc == 0


def test_main_missing_cluster(tmp_path):
    missing = tmp_path / "nope.yaml"
    rc = ctl.main(["--cluster", str(missing), "--repo", "/srv/sparkring",
                   "status"])
    assert rc == 2


def test_status_cli_uses_api_key_file(monkeypatch, sample_cluster):
    """Authenticated status uses the same remote credential path as start."""
    monkeypatch.setattr(ctl, "_container_running", lambda *_args: True)
    probes = []

    def ready(target, port, api_key_file=None):
        probes.append((port, api_key_file))
        return api_key_file == "/run/secrets/serving-api-key"

    monkeypatch.setattr(ctl, "_head_api_ready", ready)
    assert ctl.main([
        "status", "--cluster", str(sample_cluster), "--repo", "/srv/sparkring",
        "--api-key-file", "/run/secrets/serving-api-key",
    ]) == 0
    assert probes == [(8000, "/run/secrets/serving-api-key")]


def test_failed_ssh_submission_checks_current_rank_ownership(monkeypatch, fake_ranks):
    """A lost SSH acknowledgement does not prove the current launch did nothing."""
    monkeypatch.setattr(ctl, "_container_running", lambda *_args: False)
    monkeypatch.setattr(ctl, "_run_ssh", lambda *_args, **_kwargs:
                        subprocess.CompletedProcess(["ssh"], 255, "", "connection lost"))
    rollback = []
    monkeypatch.setattr(ctl, "_rollback_started", lambda ranks, prefix, token:
                        rollback.append([rank.id for rank in ranks]))
    assert ctl.start_ranks(fake_ranks, "/srv/sparkring", "deepseek-v4-flash-r", "/tmp") == 1
    assert rollback == [[1]]


def test_main_loads_cluster_and_runs_status(monkeypatch, sample_cluster,
                                            capsys):
    targets = {rank.ssh_target: rank.id for rank in ctl.load_cluster(sample_cluster).ranks}
    monkeypatch.setitem(globals(), "_rank_of", lambda target: targets[target])
    ssh = FakeSSH(monkeypatch)
    ssh.containers_up = {0: True, 1: True, 2: True, 3: True}
    ssh.api_ready = True
    rc = ctl.main(["--cluster", str(sample_cluster),
                   "--repo", "/srv/sparkring", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "rank0" in out and "UP" in out
    assert "200 OK" in out


def test_main_dry_run_status(monkeypatch, sample_cluster, capsys):
    def fake_run(argv, **kwargs):
        raise AssertionError("dry-run must not invoke SSH")

    monkeypatch.setattr(ctl, "_run_ssh", fake_run)
    rc = ctl.main(["--cluster", str(sample_cluster),
                   "--repo", "/srv/sparkring", "--dry-run", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
def test_launch_paths_are_single_shell_arguments():
    repo = "/srv/model files;not-a-command"
    log = "/tmp/model log;not-a-command"
    command = ctl._launch_command(repo, 2, log)
    words = shlex.split(command)
    assert words[1] == repo
    assert f"{repo}/{ctl.LAUNCH_REL}" in words
    assert f"{repo}/rank-2.env" in words
    assert ">" + log in words


def test_docker_failure_is_not_an_absent_container(monkeypatch):
    monkeypatch.setattr(ctl, "_run_ssh", lambda *a, **kw: subprocess.CompletedProcess([], 1, "", "daemon unavailable"))
    with pytest.raises(ctl.SSHTransportError):
        ctl._container_running("fixture-host", "model-r0")


def test_stop_is_idempotent_when_all_containers_are_absent(monkeypatch, fake_ranks):
    ssh = FakeSSH(monkeypatch)
    for _ in range(2):
        assert ctl.stop_ranks(fake_ranks, "deepseek-v4-flash-r") == 0
        assert not any(ssh.containers_up.values())
    assert len([c for c in ssh.calls if "docker ps" in c]) == 8
    assert not _start_order(ssh.calls)


@pytest.mark.parametrize("owner,removed", [("owned", True), ("foreign", False), ("<no value>", False)])
def test_rollback_checks_ownership_then_removes_immutable_id(owner, removed):
    import os
    shell = "bash" if os.name != "nt" else "C:/Program Files/Git/bin/bash.exe"
    identifier = "a" * 64
    stub = f"""docker() {{
    if [ "$1" = inspect ]; then printf '%s\\n' '{identifier} {owner}'; return 0; fi
    [ "$1" = rm ] && [ "$2" = -f ] && [ "$3" = '{identifier}' ] || return 99
    printf 'REMOVED\\n'
}}
"""
    result = subprocess.run([shell, "-c", stub + ctl._rollback_command("stopped-model", "owned")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert ("REMOVED" in result.stdout) is removed


def test_custom_prefix_is_forwarded_with_unique_launch_owner(monkeypatch, fake_ranks):
    commands = []
    monkeypatch.setattr(ctl, "_container_running", lambda *args: bool(commands))
    monkeypatch.setattr(ctl.time, "sleep", lambda *_: None)
    monkeypatch.setattr(ctl, "_run_ssh", lambda target, command, **kwargs:
                        commands.append(command) or subprocess.CompletedProcess([], 0))
    assert ctl.start_ranks(fake_ranks, "/repo", "custom-r", "/tmp", wait_api_minutes=0) == 0
    assert "SPARKRING_CONTAINER_PREFIX=custom-r" in commands[0]
    assert "SPARKRING_LAUNCH_ID=" in commands[0]


def test_timeout_is_an_unknown_remote_state(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("fixture", 1)
    monkeypatch.setattr(ctl.subprocess, "run", timeout)
    with pytest.raises(ctl.SSHTransportError):
        ctl._run_ssh("fixture", "unused")


def test_non_four_rank_inventory_rejected_before_remote_work(monkeypatch, sample_cluster):
    monkeypatch.setattr(ctl, "load_cluster", lambda *_: SimpleNamespace(ranks=[SimpleNamespace(id=i) for i in range(6)]))
    monkeypatch.setattr(ctl, "_run_ssh", lambda *a, **kw: pytest.fail("remote work forbidden"))
    assert ctl.main(["start", "--cluster", str(sample_cluster), "--repo", "/repo"]) == 2


@pytest.mark.parametrize("missing_rank,expected", [(None, 0), (2, 1)])
def test_status_requires_every_worker_even_when_head_api_is_healthy(monkeypatch, fake_ranks, missing_rank, expected):
    missing_target = next((rank.ssh_target for rank in fake_ranks if rank.id == missing_rank), None)
    monkeypatch.setattr(ctl, "_container_running", lambda target, name: target != missing_target)
    monkeypatch.setattr(ctl, "_head_api_ready", lambda *args: True)
    assert ctl.status_ranks(fake_ranks, "deepseek-v4-flash-r", 8000) == expected
