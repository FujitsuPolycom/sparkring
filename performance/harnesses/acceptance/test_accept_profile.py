"""End-to-end runs of accept_profile.main with fake SSH, HTTP and benchmark; no host is contacted."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import pytest

from performance.harnesses.acceptance import accept_profile, install, profile_info, record
from performance.harnesses.acceptance.fakes import FakeBench, FakeModel, FakeNodeA

LAN = ".".join(("192", "168", "0", "200"))
PROFILE = "mimo-v26-flash-mopd-tp2"
SERVED = "MiMo-V2.6-Flash-MOPD-TP2"
REVISION = "c17cf23cec72" + "0" * 28
STDERR = (f"Source revision: {REVISION}\nSparkRing install. Progress: /var/log/sparkring/install.log\n"
          "Done: Node 0: Wait for API readiness (550.2s)\nModel ready: http://" + LAN + ":8020/v1\n")
COMPLETE = json.dumps({"schema": install.RESULT_SCHEMA, "state": "complete", "profile": PROFILE, "nodes": 2,
                       "api_url": f"http://{LAN}:8020/v1"}, indent=2) + "\n"
# The record is named after the image the profile runs.
NAME = record.image_short(profile_info.load(PROFILE).image) + "-" + PROFILE + "-20260927"


def require_repository_drive(path):
    """Records link into the repository with relative paths, which cannot cross Windows drives."""
    if os.path.splitdrive(str(path))[0].lower() != os.path.splitdrive(str(accept_profile.ROOT))[0].lower():
        pytest.skip("the temporary directory and the repository are on different drives")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Harness:
    def __init__(self, tmp_path, node=None, model=None, bench=None):
        self.tmp = tmp_path
        self.node = node or FakeNodeA(COMPLETE, STDERR)
        self.model = model or FakeModel(SERVED)
        self.bench = bench or FakeBench(f"http://{LAN}")
        self.bench_dir = tmp_path / "bench"
        self.bench_dir.mkdir(exist_ok=True)
        (self.bench_dir / "llm_decode_bench.py").write_text("", encoding="utf-8")
        self.out, self.records = tmp_path / "out", tmp_path / "records"
        self.lines, self.targets, self.clock = [], [], Clock()

    def ssh(self, target):
        self.targets.append(target)
        return self.node

    def env(self):
        return accept_profile.Environment(
            ssh=self.ssh, http=self.model, run_process=self.bench, clock=self.clock, sleep=self.clock.sleep,
            now=lambda: datetime.datetime(2026, 9, 27, 12, 0, tzinfo=datetime.timezone.utc), log=self.lines.append,
            hostname=lambda: "client-box", revision=lambda: "abcdef123456")

    def args(self, *extra, install_source="published"):
        mode = ["--install", install_source, "--node-a", f"code@{LAN}"] if install_source else ["--skip-install"]
        return ["--profile", PROFILE, "--api-host", LAN, *mode, "--stress-rounds", "1", "--bench-dir",
                str(self.bench_dir), "--out", str(self.out), "--record-root", str(self.records), "--date", "20260927",
                *extra]

    def main(self, *extra, **kwargs):
        return accept_profile.main(self.args(*extra, **kwargs), env=self.env())


def test_full_run_installs_checks_measures_and_writes_a_private_data_free_record(tmp_path, capsys):
    require_repository_drive(tmp_path)
    h = Harness(tmp_path)
    assert h.main() == 0
    assert h.targets == [f"code@{LAN}"] and len(h.node.launched) == 1
    assert "--profile mimo-v26-flash-mopd-tp2 --yes --json" in h.node.launched[0]
    install_result = json.loads((h.out / "install.json").read_text())
    assert install_result["ok"] and install_result["source_revision"] == REVISION
    assert install_result["api_ready_seconds"] == 550.2
    assert (h.out / "install/stderr.log").read_text() == STDERR
    assert json.loads((h.out / "readiness.json").read_text())["served_models"] == [SERVED]
    assert json.loads((h.out / "functional.json").read_text())["passed"] == 7
    assert json.loads((h.out / "stress.json").read_text())["n"] == 32
    command = h.bench.calls[0]
    assert command[command.index("--port") + 1] == "8020" and command[command.index("--model") + 1] == SERVED
    readme = "README values for `mimo-v26-flash-mopd-tp2` (port 8020): decode 27.1 / 139 / 206; prefill 64K 2,749"
    assert capsys.readouterr().out.splitlines() == [readme, readme]

    markdown = (h.records / f"{NAME}.md").read_text()
    directory = h.records / NAME
    assert sorted(p.name for p in directory.iterdir()) == ["functional.txt", "install-phases.txt", "stress.json",
                                                           "tp2-matrix.json"]
    matrix = json.loads((directory / "tp2-matrix.json").read_text())
    assert matrix["metadata"]["server"] == "http://NODE_A" and "startup_diagnostics" not in matrix
    assert "Model ready: http://NODE_A:8020/v1" in (directory / "install-phases.txt").read_text()
    for path in [h.records / f"{NAME}.md", *directory.iterdir()]:
        text = path.read_text()
        assert LAN not in text and "client-box" not in text and "code@" not in text, path
    assert "installed source commit `c17cf23cec72`" in markdown
    assert "at commit `abcdef123456`" in markdown
    assert json.loads((h.out / "record.json").read_text())["readme"] == readme


def test_second_invocation_reuses_every_saved_result(tmp_path):
    require_repository_drive(tmp_path)
    h = Harness(tmp_path)
    assert h.main() == 0
    scripts, requests, runs = len(h.node.scripts), len(h.model.requests), len(h.bench.calls)
    assert h.main() == 0
    assert (len(h.node.scripts), len(h.model.requests), len(h.bench.calls)) == (scripts, requests, runs)
    assert any("using the saved result" in line for line in h.lines)


def test_redo_keeps_earlier_outputs_and_runs_the_step_again(tmp_path):
    h = Harness(tmp_path)
    assert h.main("--steps", "readiness,functional") == 0
    requests = len(h.model.requests)
    assert h.main("--steps", "functional", "--redo", "functional") == 0
    assert len(h.model.requests) == requests + 7
    assert (h.out / "functional.json.20260927T120000Z").is_file()
    assert (h.out / "functional.txt.20260927T120000Z").is_file() and (h.out / "functional.json").is_file()
    assert len(h.node.launched) == 1


def test_interrupted_install_is_followed_not_restarted(tmp_path):
    require_repository_drive(tmp_path)
    h = Harness(tmp_path, node=FakeNodeA(COMPLETE, STDERR, polls_before_exit=10))
    assert h.main("--install-timeout", "60") == 1
    assert (h.out / "install-launch.json").is_file() and not (h.out / "install.json").exists()
    assert any("may still be running" in line for line in h.lines)
    assert h.main() == 0
    assert len(h.node.launched) == 1
    assert any("following run mimo-v26-flash-mopd-tp2-20260927T120000Z" in line for line in h.lines)


@pytest.mark.parametrize("stdout, code, reason", [
    (json.dumps({"schema": install.RESULT_SCHEMA, "state": "failed", "stage": "build", "message": "no build"}), 2,
     "installer state failed: no build"),
    (json.dumps({"schema": install.RESULT_SCHEMA, "state": "needs_input", "field": "approval", "message": "approve"}),
     3, "installer state needs_input: approve"),
    ("", 2, "holds no sparkring-install-result/v1 document"),
])
def test_unsuccessful_install_stops_before_any_request(tmp_path, stdout, code, reason):
    h = Harness(tmp_path, node=FakeNodeA(stdout, "Source revision: " + REVISION + "\n", exit_code=code))
    assert h.main() == 1
    assert h.model.requests == [] and h.bench.calls == []
    assert any(reason in line for line in h.lines)
    saved = json.loads((h.out / "install.json").read_text())
    assert saved["complete"] and not saved["ok"]
    scripts = len(h.node.scripts)
    assert h.main() == 1 and len(h.node.scripts) == scripts


def test_skip_install_never_opens_ssh(tmp_path):
    require_repository_drive(tmp_path)
    h = Harness(tmp_path)
    assert h.main("--steps", "readiness,functional,record", install_source=None) == 0
    assert h.targets == []
    markdown = (h.records / f"{NAME}.md").read_text()
    assert "This run installed nothing" in markdown and "- Throughput was not measured." in markdown


def test_record_names_the_image_the_installer_verified(tmp_path):
    """An install with --image-lock serves another image than the profile's release; the record names that one."""
    require_repository_drive(tmp_path)
    release = "dev-20260928-example-cuda1342-nccl2323-status033"
    result = {**json.loads(COMPLETE), "transaction": {"verification": {"image_release": release}}}
    h = Harness(tmp_path, node=FakeNodeA(json.dumps(result) + "\n", STDERR))
    assert h.main("--install-arg=--image-lock", "--install-arg=/var/tmp/lock.json") == 0
    assert json.loads((h.out / "install.json").read_text())["image_release"] == release
    name = record.image_short(release) + "-" + PROFILE + "-20260927"
    markdown = (h.records / f"{name}.md").read_text()
    assert f"on installer image `{release}`" in markdown
    assert f"runtime/releases/{release}/release.json" in markdown
    assert f"`install.sh --profile {PROFILE} --yes --json --image-lock LOCK`" in markdown
    assert "/var/tmp/lock.json" not in markdown


def test_a_checkpoint_choice_reaches_the_installer_and_names_the_served_model(tmp_path):
    ring = "glm53-flash-nvfp4-spark-tp4"
    chosen = profile_info.load(ring, checkpoint="nvidia-nvfp4")
    assert (chosen.served_model_name, chosen.repository, chosen.checkpoint, chosen.port) == (
        "GLM-5.3-Flash-NVFP4-NVIDIA-TP4", "nvidia/GLM-5.3-Flash-NVFP4", "nvidia-nvfp4", 8015)
    assert chosen.revision == "da920bb0b9f4a06727223a349e55468e38352348"
    assert profile_info.load(ring, checkpoint="nvfp4-spark") == profile_info.load(ring)
    with pytest.raises(ValueError, match="lists"):
        profile_info.load(ring, checkpoint="main")
    h = Harness(tmp_path, model=FakeModel(chosen.served_model_name))
    argv = h.args("--checkpoint", "nvidia-nvfp4")
    argv[argv.index(PROFILE)] = ring
    accept_profile.main(argv, env=h.env())
    command = json.loads((h.out / "install-launch.json").read_text())["command"]
    assert f"--profile {ring} --yes --json --checkpoint nvidia-nvfp4" in command


def test_private_leftover_refuses_the_whole_record(tmp_path):
    h = Harness(tmp_path, bench=FakeBench(f"http://{LAN}", note="measured from client-box"))
    assert h.main() == 2
    assert any("refusing to write the matrix" in line and "client-box" in line for line in h.lines)
    assert not h.records.exists() or list(h.records.iterdir()) == []


def test_existing_record_is_never_overwritten(tmp_path):
    h = Harness(tmp_path)
    h.records.mkdir()
    (h.records / f"{NAME}.md").write_text("kept\n")
    assert h.main("--steps", "readiness,record", install_source=None) == 2
    assert (h.records / f"{NAME}.md").read_text() == "kept\n"


def test_failed_benchmark_is_not_saved_and_retried_next_time(tmp_path):
    h = Harness(tmp_path, bench=FakeBench(f"http://{LAN}", exit_code=1))
    assert h.main("--steps", "throughput", install_source=None) == 1
    assert not (h.out / "throughput.json").exists()
    h.bench.exit_code = 0
    assert h.main("--steps", "throughput", "--repeats", "1", install_source=None) == 0
    assert json.loads((h.out / "throughput.json").read_text())["files"] == ["tp2-matrix.json"]


def test_repeats_write_one_matrix_per_run(tmp_path):
    h = Harness(tmp_path)
    assert h.main("--steps", "throughput", "--repeats", "3", install_source=None) == 0
    saved = json.loads((h.out / "throughput.json").read_text())
    assert saved["runs"] == 3 and saved["files"] == [f"tp2-matrix-run{i}.json" for i in (1, 2, 3)]
    assert len(h.bench.calls) == 3


def test_argument_errors(tmp_path):
    h = Harness(tmp_path)
    assert accept_profile.main(["--profile", PROFILE, "--api-host", LAN, "--install", "published", "--out",
                                str(h.out)], env=h.env()) == 2
    with pytest.raises(SystemExit):
        accept_profile.main(["--profile", PROFILE, "--api-host", LAN, "--install", "published", "--skip-install",
                             "--out", str(h.out)], env=h.env())
    assert h.main("--steps", "install", install_source=None) == 2
    assert accept_profile.main(["--profile", "glm53-flash-spark-tp2-dcp1", "--api-host", LAN, "--skip-install",
                                "--out", str(h.out)], env=h.env()) == 2
    assert h.targets == [] and h.model.requests == []


def test_skip_check_accepts_hyphenated_names(tmp_path):
    h = Harness(tmp_path)
    assert h.main("--steps", "functional", "--skip-check", "tool-call", "--skip-check", "image",
                  install_source=None) == 0
    saved = json.loads((h.out / "functional.json").read_text())
    assert {r["name"] for r in saved["checks"] if r["status"] == "SKIP"} == {"tool call", "image"}


def test_thinking_switch_override_reaches_every_direct_request(tmp_path):
    h = Harness(tmp_path)
    assert h.main("--steps", "functional", "--thinking-off", '{"chat_template_kwargs": {"thinking": false}}',
                  install_source=None) == 0
    bodies = [body for method, _, body in h.model.requests if method == "POST"]
    assert [b.get("chat_template_kwargs") for b in bodies] == [{"thinking": False}] * 6 + [None]


def test_entry_point_is_a_script_path():
    assert Path(accept_profile.__file__).name == "accept_profile.py"
    assert accept_profile.ROOT == Path(__file__).resolve().parents[3]


class Quiet:
    """Stands in for dashboard_quiet.held: records the hold, its release and the steps between them."""

    def __init__(self, harness, state="on"):
        self.harness, self.state, self.events = harness, state, []

    def __call__(self, label, **kwargs):
        import contextlib
        quiet = self

        @contextlib.contextmanager
        def held():
            quiet.events.append(("hold", label, kwargs["ssh"][-1] if kwargs.get("ssh") else None,
                                 len(quiet.harness.model.requests)))
            try:
                yield {"state": quiet.state, "where": kwargs.get("where"), "holder": "accept_profile-0-ab12cd34",
                       "was_on_before": False, "service": "active", "detail": None}
            finally:
                quiet.events.append(("release", len(quiet.harness.model.requests), len(quiet.harness.bench.calls)))
        return held()


def test_the_dashboard_is_quiet_from_the_first_request_to_the_end_and_every_result_says_so(tmp_path):
    require_repository_drive(tmp_path)
    h = Harness(tmp_path)
    quiet = Quiet(h)
    env = h.env()
    env.quiet = quiet
    assert accept_profile.main(h.args(), env=env) == 0
    # Held on Node A before the first request (after the installation) and released after the benchmark.
    assert quiet.events[0] == ("hold", "accept_profile", f"code@{LAN}", 0)
    assert quiet.events[1][0] == "release" and quiet.events[1][1] > 0 and quiet.events[1][2] == 1
    for step in ("readiness", "functional", "stress", "throughput"):
        assert json.loads((h.out / f"{step}.json").read_text())["dashboard_quiet"]["state"] == "on"
    assert "dashboard_quiet" not in json.loads((h.out / "install.json").read_text())
    markdown = (h.records / f"{NAME}.md").read_text()
    assert "- **Dashboard:** quiet mode was on while the harness measured" in markdown


def test_a_failed_step_still_releases_the_hold(tmp_path):
    h = Harness(tmp_path, model=FakeModel("Another-Model"))
    quiet = Quiet(h)
    env = h.env()
    env.quiet = quiet
    assert accept_profile.main(h.args("--ready-timeout", "1", install_source=None), env=env) == 1
    assert [event[0] for event in quiet.events] == ["hold", "release"]
    # Without --node-a or --dashboard-ssh the hold is asked of the client itself.
    assert quiet.events[0][2] is None


def test_an_absent_dashboard_is_recorded_and_the_run_goes_on(tmp_path):
    require_repository_drive(tmp_path)
    h = Harness(tmp_path)
    quiet = Quiet(h, state="dashboard absent")
    env = h.env()
    env.quiet = quiet
    assert accept_profile.main(h.args(), env=env) == 0
    assert json.loads((h.out / "throughput.json").read_text())["dashboard_quiet"]["state"] == "dashboard absent"
    assert "- **Dashboard:** dashboard absent" in (h.records / f"{NAME}.md").read_text()


@pytest.mark.parametrize("complete", [False, True])
def test_install_command_mismatch_refuses_resume_without_remote_effects(tmp_path, complete):
    h = Harness(tmp_path)
    if complete:
        assert h.main("--steps", "readiness") == 0
    else:
        h.node.polls_before_exit = 10
        assert h.main("--install-timeout", "60", "--steps", "readiness") == 1
    launch = (h.out / "install-launch.json").read_bytes()
    scripts, requests = len(h.node.scripts), len(h.model.requests)
    assert h.main("--install-arg=--image-lock=/var/tmp/other-lock.json", "--steps", "readiness") == 1
    assert len(h.node.scripts) == scripts and len(h.model.requests) == requests
    assert (h.out / "install-launch.json").read_bytes() == launch
    assert any("--redo install" in line and "command" in line.lower() for line in h.lines)
    assert any("--image-lock=/var/tmp/other-lock.json" in line for line in h.lines)


def test_changed_install_source_refuses_saved_result(tmp_path):
    h = Harness(tmp_path)
    assert h.main("--steps", "readiness") == 0
    scripts = len(h.node.scripts)
    assert h.main("--steps", "readiness", install_source="published:release-test") == 1
    assert len(h.node.scripts) == scripts
    assert any("release-test" in line and "--redo install" in line for line in h.lines)


def test_completed_install_without_launch_identity_requires_redo(tmp_path):
    h = Harness(tmp_path)
    assert h.main("--steps", "readiness") == 0
    (h.out / "install-launch.json").unlink()
    scripts = len(h.node.scripts)
    assert h.main("--steps", "readiness") == 1
    assert len(h.node.scripts) == scripts
    assert any("--redo install" in line for line in h.lines)


@pytest.mark.parametrize("path", ["C:/Program Files/Git/var/tmp/lock.json", r"C:\tools\lock.json"])
def test_windows_install_argument_is_rejected_before_ssh(tmp_path, path):
    h = Harness(tmp_path)
    assert h.main("--steps", "readiness", "--install-arg=--image-lock=" + path) == 1
    assert h.targets == [] and h.node.launched == []
    assert not (h.out / "install-launch.json").exists()
    assert any("MSYS_NO_PATHCONV=1" in line for line in h.lines)


def test_missing_install_argument_path_prevents_launch(tmp_path):
    h = Harness(tmp_path)
    h.node.paths = set()
    assert h.main("--steps", "readiness", "--install-arg=--image-lock=/var/tmp/not-present.json") == 1
    assert h.node.launched == [] and h.model.requests == []
    assert not (h.out / "install-launch.json").exists()
    assert any("/var/tmp/not-present.json" in line for line in h.lines)


def test_explicit_redo_allows_changed_command_and_preserves_outputs(tmp_path):
    h = Harness(tmp_path)
    assert h.main("--steps", "readiness") == 0
    assert h.main("--steps", "readiness", "--redo", "install", "--install-arg=--image-lock=/var/tmp/other-lock.json") == 0
    assert len(h.node.launched) == 2
    assert len(h.model.requests) == 2
    assert (h.out / "readiness.json.20260927T120000Z").is_file()
    assert (h.out / "install-launch.json.20260927T120000Z").is_file()
    assert "--image-lock=/var/tmp/other-lock.json" in json.loads((h.out / "install-launch.json").read_text())["command"]


def test_redo_install_sets_aside_all_acceptance_results(tmp_path):
    require_repository_drive(tmp_path)
    h = Harness(tmp_path)
    assert h.main() == 0
    requests, runs = len(h.model.requests), len(h.bench.calls)
    assert h.main("--redo", "install", "--topic", "second-installation") == 0
    assert len(h.model.requests) > requests and len(h.bench.calls) == runs + 1
    for step in accept_profile.STEPS:
        assert (h.out / (accept_profile.RESULTS[step] + ".20260927T120000Z")).is_file()
