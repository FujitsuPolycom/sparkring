"""The site file's Docker command: parsing, validation, every Docker invocation, plan print.

A Spark whose SSH user is not in the docker group runs Docker through a
prefix such as ``sudo -n docker``. The site file sets it at the top level
(default ``docker``) and per ring entry; every Docker invocation the harness
sends to a Spark must start with that Spark's command.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess

import pytest

from sparkring_sircl.ring import cli, plan, remote
from sparkring_sircl.ring.site import Site, SiteError

SUDO = "sudo -n docker"
SPARK3 = "op@192.0.2.13"
_CALL = re.compile(r"(?<![\w/.-])docker (?=info|image|ps|rm|run|inspect)")


def _document(size: int = 8) -> dict:
    return {
        "schema": "sircl-ring-site/v1", "image": "sha256:0123456789ab", "lan_interface": "lan0",
        "control_port": 29650, "remote_dir": "/tmp/sircl-ring",
        "ring": [{"name": f"spark{i}", "ssh": f"op@192.0.2.{10 + i}", "lan_address": f"192.0.2.{10 + i}"}
                 for i in range(size)],
    }


def _override(document: dict, position: int = 3, docker: object = SUDO) -> dict:
    document["ring"][position]["docker"] = docker
    return document


def _calls(command: str) -> list[tuple[str, str]]:
    """``(text before, subcommand)`` of every Docker invocation in ``command``."""
    return [(command[:match.start()], command[match.end():].split()[0]) for match in _CALL.finditer(command)]


def test_docker_command_defaults_and_overrides():
    site = Site.from_json(_document())
    assert site.docker == "docker" and {host.docker for host in site.ring} == {"docker"}
    document = _override(_document(), docker="sudo   -n  docker")
    document["docker"] = "/usr/bin/docker"
    site = Site.from_json(document)
    assert site.docker == "/usr/bin/docker"
    assert [host.docker for host in site.ring] == ["/usr/bin/docker"] * 3 + [SUDO] + ["/usr/bin/docker"] * 4


@pytest.mark.parametrize("value", ["", "   ", "sudo -n", "docker; reboot", "$(docker)", "sudo -n 'docker'",
                                   "docker > /dev/null", "REPLACE_DOCKER", "a b c d e f g h docker", 5,
                                   ["sudo", "docker"], None])
def test_unusable_docker_commands_are_refused(value):
    with pytest.raises(SiteError, match=r"ring entry 3 \(spark3\) docker"):
        Site.from_json(_override(_document(), docker=value))
    document = _document()
    document["docker"] = value
    with pytest.raises(SiteError, match="docker"):
        Site.from_json(document)


def test_every_docker_invocation_starts_with_the_sparks_command():
    site = Site.from_json(_override(_document()))
    built = plan.build_plan(site, "two-tp4", "run1", digest="d" * 16)
    for rank in built.ranks:
        expected = SUDO if rank.position == 3 else "docker"
        assert rank.docker == expected
        for command in (remote.docker_run(built, rank), remote.container_state(rank.container, docker=rank.docker)):
            calls = _calls(command)
            assert calls, command
            assert all(before.endswith("sudo -n ") == (expected == SUDO) for before, _ in calls), command
    commands = {
        "prepare": remote.prepare_command("img", "/tmp/x", "run1", "d" * 16, docker=SUDO),
        "remove-run": remote.remove_harness_containers("run1", docker=SUDO),
        "remove-all": remote.remove_harness_containers(docker=SUDO),
        "running": remote.running_containers(docker=SUDO),
        "preflight": remote.preflight_script("img", "lan0", docker=SUDO),
    }
    subcommands = set()
    for name, command in commands.items():
        calls = _calls(command)
        assert calls and all(before.endswith("sudo -n ") for before, _ in calls), name
        subcommands |= {sub for _, sub in calls}
    assert subcommands == {"run", "rm", "ps", "info", "image"}
    assert not any(before.endswith("sudo -n ") for before, _ in _calls(remote.running_containers()))
    with pytest.raises(ValueError, match="Docker command"):
        remote.running_containers(docker="docker; reboot")


def test_plan_print_shows_each_sparks_docker_command():
    site = Site.from_json(_override(_document()))
    built = plan.build_plan(site, "path4", "run1", digest="d" * 16)
    lines = [line for line in plan.render_text(built).splitlines() if ", container " in line]
    assert len(lines) == 4
    for line in lines:
        expected = SUDO if line.strip().startswith("rank 3: spark3") else "docker"
        assert line.endswith(f'docker command "{expected}"'), line
    assert [rank["docker"] for rank in built.to_json()["ranks"]] == ["docker", "docker", "docker", SUDO]


def test_cli_sends_each_spark_its_docker_command(monkeypatch, tmp_path):
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_override(_document())))
    sent: list[tuple[str, str]] = []

    def fake_ssh(target, command, *, timeout=60, input_bytes=None, binary="ssh"):
        sent.append((target, command))
        if " inspect -f " in command:      # container states: every rank exited cleanly
            ranks = re.findall(r'echo "(\d+) \$\(', command)
            return remote.Result(0, "".join(f"{rank} exited 0\n" for rank in ranks), "")
        if "ps -aq" in command:            # removal: nothing left
            return remote.Result(0, "0\n", "")
        return remote.Result(0, "", "")

    monkeypatch.setattr(remote, "ssh", fake_ssh)
    common = ["--site", str(site_file), "--config", "two-tp4", "--run-id", "r1"]
    cli.main(["preflight", *common])
    cli.main(["stage", *common])
    assert cli.main(["run", *common, "--output", str(tmp_path / "results")]) == 0
    assert cli.main(["cleanup", "--site", str(site_file)]) == 0
    seen: dict[str, list[tuple[bool, str]]] = {}
    for target, command in sent:
        for before, sub in _calls(command):
            seen.setdefault(target, []).append((before.endswith("sudo -n "), sub))
    assert set(seen) == {f"op@192.0.2.{10 + i}" for i in range(8)}
    assert {sub for _, sub in seen[SPARK3]} == {"info", "image", "ps", "run", "inspect", "rm"}
    assert all(prefixed for prefixed, _ in seen[SPARK3])
    for target, calls in seen.items():
        if target != SPARK3:
            assert not any(prefixed for prefixed, _ in calls), target


def _preflight_ssh(target, command, *, timeout=60, input_bytes=None, binary="ssh"):
    """Spark 3 refuses plain ``docker`` (its SSH user is not in the docker group); sudo works."""
    if "info --format" not in command:
        return remote.Result(0, "", "")
    plain = any(not before.endswith("sudo -n ") for before, _ in _calls(command))
    if target == SPARK3 and plain:
        return remote.Result(0, "docker\tunusable: permission denied while trying to connect to the Docker "
                                "daemon socket\nimage\tmissing\n", "")
    return remote.Result(0, "docker\t27.3.1\nimage\tsha256:abc\n", "")


def test_preflight_names_a_spark_whose_docker_command_is_refused(monkeypatch):
    monkeypatch.setattr(remote, "ssh", _preflight_ssh)
    site = Site.from_json(_document())
    ok, lines = cli.preflight(site, [plan.build_plan(site, "pairs", "r1", digest="d" * 16)], force=False)
    refused = [line for line in lines if "cannot reach the Docker daemon" in line]
    assert len(refused) == 1 and refused[0].startswith('BLOCKER: spark3: the docker command "docker"')
    assert "permission denied" in refused[0] and '"sudo -n docker"' in refused[0]
    assert not any("spark3: image" in line for line in lines)
    site = Site.from_json(_override(_document()))
    ok, lines = cli.preflight(site, [plan.build_plan(site, "pairs", "r1", digest="d" * 16)], force=False)
    assert not any("cannot reach the Docker daemon" in line for line in lines)
    assert any(line.startswith('spark3: docker command "sudo -n docker" (server 27.3.1)') for line in lines)


_DOCKER_STUB = """#!/bin/sh
if [ -n "$DOCKER_DENIED" ]; then
  echo "permission denied while trying to connect to the Docker daemon socket" >&2
  exit 1
fi
case "$1" in
  info) echo 27.3.1 ;;
  image) echo sha256:abc ;;
  ps) [ -n "$FAKE_IDS" ] && printf '%s\\n' $FAKE_IDS ;;
  inspect) echo "exited 0" ;;
  run) echo started ;;
esac
exit 0
"""

_SUDO_STUB = """#!/bin/sh
[ "$1" = "-n" ] && shift
unset DOCKER_DENIED
exec "$@"
"""


@pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None, reason="needs a POSIX shell")
@pytest.mark.parametrize("prefix, denied, usable", [
    ("docker", False, True), ("docker", True, False), (SUDO, True, True),
], ids=["docker", "docker-refused", "sudo-docker"])
def test_docker_commands_in_a_real_shell(tmp_path, prefix, denied, usable):
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, text in (("docker", _DOCKER_STUB), ("sudo", _SUDO_STUB)):
        path = stubs / name
        path.write_text(text)
        path.chmod(0o755)
    env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}"}
    env.pop("FAKE_IDS", None)
    if denied:
        env["DOCKER_DENIED"] = "1"

    def run(command: str, **extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", command], env={**env, **extra}, capture_output=True, text=True,
                              timeout=30)

    values = cli._parse_lines(run(remote.preflight_script("img", "lo", docker=prefix)).stdout)
    if usable:
        assert values["docker"] == "27.3.1" and values["image"] == "sha256:abc"
    else:
        assert values["docker"] == "unusable: permission denied while trying to connect to the Docker daemon socket"
        assert values["image"] == "missing"
    removal = run(remote.remove_harness_containers("r1", docker=prefix))
    assert (removal.returncode, removal.stdout.strip()) == ((0, "0") if usable else (1, ""))
    if usable:
        assert run(remote.remove_harness_containers(docker=prefix), FAKE_IDS="c1 c2").stdout.strip() == "2"
    built = plan.build_plan(Site.from_json(_document()), "pairs", "r1", digest="d" * 16)
    started = run(remote.docker_run(built, dataclasses.replace(built.ranks[0], docker=prefix)))
    assert (started.returncode == 0, started.stdout.strip() == "started") == (usable, usable)
    prepared = run(remote.prepare_command("img", "/tmp/x", "r1", "d" * 16, docker=prefix))
    assert (prepared.returncode == 0) == usable
    assert run(remote.container_state("c", docker=prefix)).stdout.strip() == ("exited 0" if usable else "missing")
    assert (run(remote.running_containers(docker=prefix)).returncode == 0) == usable
