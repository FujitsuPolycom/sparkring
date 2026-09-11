"""Execute container cleanup guards against a fake engine, without Docker or SSH."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml
import sparkring_runtime as runtime
import sparkring_generic_launcher as generic
from sparkring_site import load_site, validate_site
from test_sparkring_site import six_ring_document

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    return (
        load_site(ROOT / "scripts/config/glm53-flash-tp4-site.example.yaml"),
        runtime.load_runtime_profile(
            ROOT / "scripts/config/glm53-flash-dflash2-bf16-tp4-dcp1.example.json"
        ),
    )


def test_six_rank_generic_plan_rejects_before_execution(inputs, monkeypatch):
    _, profile = inputs
    site = validate_site(six_ring_document(yaml.safe_load((ROOT / "scripts/config/glm53-flash-tp4-site.example.yaml").read_text())))
    monkeypatch.setattr(
        runtime, "execute", lambda *args: pytest.fail("remote execution")
    )
    with pytest.raises(runtime.ProfileError, match="exactly four ranks"):
        generic.build_actions(site, profile, "plan")


def run_fake_engine(tmp_path, script, profile, mode):
    shell = (
        shutil.which("bash") if os.name != "nt" else "C:/Program Files/Git/bin/bash.exe"
    )
    if not shell or not Path(shell).is_file():
        pytest.skip("Bash required")
    engine = tmp_path / "docker"
    engine.write_text(
        r"""#!/usr/bin/env bash
case "$1" in
 info) [[ "$MODE" != daemon ]];;
 ps)
   [[ "$MODE" != inventory ]] || exit 1
   [[ "$MODE" == absent ]] || printf '%s\n' "$NAME"
   ;;
 inspect)
   if [[ "$3" == '{{.Id}}' ]]; then printf '%s\n' "$ID"; exit; fi
   # Every ownership query must use the inspected ID, never the reusable name.
   [[ "${!#}" == "$ID" ]] || exit 1
   case "$3" in
    *org.sparkring.managed*) [[ "$MODE" == foreign ]] && echo false || echo true;;
    *org.sparkring.profile*) echo "$PROFILE";;
    *org.sparkring.bundle*) [[ "$MODE" == badbundle ]] && echo wrong || echo bundle;;
    *org.sparkring.service*) echo service;;
    *org.sparkring.source-profile*) echo source;;
    *) exit 1;;
   esac;;
 rm) printf '%s' "$3" > "$LOG";;
 exec) printf '%s' "$2" > "$LOG";;
 *) exit 99;;
esac
""",
        newline="\n",
    )
    engine.chmod(0o755)
    log = tmp_path / "removed"
    env = {
        **os.environ,
        "MODE": mode,
        "NAME": runtime.container_name(profile, 0),
        "ID": "a" * 64,
        "PROFILE": profile.profile_id,
        "LOG": log.as_posix(),
    }
    path = tmp_path.as_posix()
    if os.name == "nt":
        path = "/" + path[0].lower() + path[2:]
    command = 'export PATH="' + path + ':$PATH"; ' + script
    result = subprocess.run(
        [shell, "--noprofile", "--norc", "-c", command],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result, log


@pytest.mark.parametrize(
    "mode,expected", [("daemon", 74), ("inventory", 74), ("absent", 0), ("present", 1)]
)
def test_rollback_requires_verified_absence(inputs, tmp_path, mode, expected):
    site, profile = inputs
    action = runtime.verify_rollback_actions(site, profile)[0]
    result, log = run_fake_engine(tmp_path, action.argv[-1], profile, mode)
    assert result.returncode == expected, result.stderr
    assert not log.exists()


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize(
    "mode,expected",
    [("daemon", 74), ("inventory", 74), ("absent", 0), ("foreign", 73), ("present", 0)],
)
def test_stop_checks_labels_and_removes_only_captured_id(
    inputs, tmp_path, mode, expected, owned
):
    site, profile = inputs
    owner = runtime.Ownership("bundle", "service", "source") if owned else None
    action = runtime.stop_actions(site, profile, ownership=owner)[0]
    result, log = run_fake_engine(tmp_path, action.argv[-1], profile, mode)
    assert result.returncode == expected, result.stderr
    if mode == "present":
        assert log.read_text() == "a" * 64
    else:
        assert not log.exists()


def test_stop_checks_bundle_label(inputs, tmp_path):
    site, profile = inputs
    action = runtime.stop_actions(
        site, profile, ownership=runtime.Ownership("bundle", "service", "source")
    )[0]
    result, log = run_fake_engine(tmp_path, action.argv[-1], profile, "badbundle")
    assert result.returncode == 73
    assert not log.exists()


@pytest.mark.parametrize("mode,expected", [("foreign", 73), ("present", 0)])
def test_health_probe_uses_owned_immutable_id(inputs, tmp_path, mode, expected):
    import dataclasses
    site, profile = inputs
    profile = dataclasses.replace(profile, health_check=("python", "probe.py"))
    action = runtime.health_check_actions(site, profile)[0]
    result, log = run_fake_engine(tmp_path, action.argv[-1], profile, mode)
    assert result.returncode == expected
    if mode == "present":
        assert log.read_text() == "a" * 64
    else:
        assert not log.exists()
