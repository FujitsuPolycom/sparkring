"""Install, status and automatic recovery keep a deployment's SparkCache budget."""
import json

import pytest

from runtime.common import image_lock, installer, sparkcache
from runtime.host import controller, install_workflow, operator_view, recovery
from runtime.host import test_install_workflow, test_persistence
from runtime.host.test_recovery import Clock, DEAD, Sparks, run
from scripts import installer_runner, sparkring

PROFILE = "qwen38-flash-next-csf-sparkcache-tp2"
machine = test_install_workflow.machine
sparks = test_install_workflow.sparks
plain_up = test_persistence.plain_up


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "image.json"
    image = json.loads(json.dumps(image_lock.default()))
    image["profiles"] = sorted(set(image["profiles"]) | {PROFILE})
    path.write_text(json.dumps(image), encoding="utf-8")
    return path


@pytest.mark.parametrize("size", [None, "64G", "64GiB", "68719476736"])
def test_install_plan_records_budget_and_prints_one_line(machine, image_path, capsys, size):
    flags = ["--sparkcache-size", size] if size else []
    assert sparkring.main(["install", "--profile", PROFILE, "--image-lock", str(image_path), "--plan", *flags]) == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert [line for line in output.splitlines() if line.startswith("SparkCache:")] == [
        f"SparkCache: {64 if size else 32} GiB per Spark"]
    locks = list((controller.STATE / "deployments").glob(PROFILE + "*/deployment.lock.json"))
    assert len(locks) == 1
    lock = installer.load(locks[0].parent)
    assert sparkcache.deployment_budget(lock) == (64 if size else 32) * 1024**3
    if size:
        assert lock["serving"]["sparkcache_size"] == 64 * 1024**3
        plan = install_workflow.saved_plan(locks[0].parent)
        assert f"--sparkcache-size {size}" in plan["command"]
    for spec in installer.specifications(lock):
        assert sparkcache.budget(spec.command) == (64 if size else 32) * 1024**3


def test_ready_message_repeats_size_spelling(machine, image_path, capsys):
    args = ["install", "--profile", PROFILE, "--image-lock", str(image_path), "--sparkcache-size", "64G", "--yes"]
    assert sparkring.main([*args, "--plan"]) == 0
    capsys.readouterr()
    assert sparkring.main(args) == 0
    captured = capsys.readouterr()
    lines = (captured.out + captured.err).splitlines()
    assert any("Start again:" in line and "--sparkcache-size 64G" in line for line in lines)


def test_install_refuses_size_on_plain_profile_before_survey(machine, sparks, capsys):
    sparks.survey = lambda *a, **k: pytest.fail("invalid option reached the survey")
    assert sparkring.main(["install", "--profile", "qwen38-flash-next-tp2", "--sparkcache-size", "64G", "--plan"]) == 2
    assert "--sparkcache-size requires a SparkCache profile" in capsys.readouterr().err
    assert not list((controller.STATE / "deployments").glob("qwen*/deployment.lock.json"))


@pytest.mark.parametrize("mode", [None, "sha256", "none"])
def test_up_restart_recovery_and_status_keep_size(plain_up, image_path, monkeypatch, capsys, mode):
    from runtime.host import discovery, node
    cluster = installer.read(controller.STATE / "cluster.json")
    simulated = Sparks([row["host"] for row in cluster["plan"]["spec"]["hosts"]])
    monkeypatch.setattr(discovery, "ssh", simulated.ssh)
    monkeypatch.setattr(node, "status", lambda: {"state": "network-configured"})
    commands = []
    environments = []

    def call(runner, target, argv, timeout):
        if argv[1] == "start":
            # Inspect the actual saved deployment that the lifecycle runner loads.
            commands.extend(spec.command for spec in installer.specifications(runner.lock))
            environments.extend(spec.environment for spec in installer.specifications(runner.lock))
        return simulated.rank_operation(runner, target, argv, timeout)

    monkeypatch.setattr(installer_runner.Runner, "_call", call)
    flags = ["--sparkcache-integrity", mode, "--sparkcache-debug"] if mode else []
    assert controller.lifecycle(["up", PROFILE, "--image-lock", str(image_path), "--sparkcache-size", "64G", "--execute", *flags]) == 0
    directory = controller.STATE / "deployments" / PROFILE
    simulated.directory = directory.resolve()
    recorded = (directory / "deployment.lock.json").read_bytes()
    recovery.started(directory)
    for operation in ("down", "up", "restart"):
        assert controller.lifecycle([operation, "--execute", "--yes"]) == 0
        assert (directory / "deployment.lock.json").read_bytes() == recorded
    clock = Clock()
    assert run(clock, api=DEAD)["state"] == "confirming"
    clock.advance(60)
    assert run(clock, api=DEAD)["state"] == "recovered"
    assert (directory / "deployment.lock.json").read_bytes() == recorded
    assert len(commands) >= 8
    assert all(sparkcache.budget(command) == 64 * 1024**3 for command in commands)
    if mode:
        assert all(sparkcache.connector(command)["kv_connector_extra_config"]["spark_cache_payload_integrity"] == mode
                   for command in commands)
        assert all(env["SPARK_CONTEXT_CACHE_TRACE_REUSE"] == "1" and
                   env["VLLM_LOGGING_CONFIG_PATH"] == sparkcache.DEBUG_TARGET for env in environments)
    capsys.readouterr()
    assert controller.lifecycle(["status"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines.count("SparkCache: 64 GiB per Spark") == 1
    assert lines.count("SparkCache integrity: " + (mode or "fast")) == 1
    assert lines.count("SparkCache logs: debug") == bool(mode)
    assert controller.lifecycle(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["deployment"]["sparkcache_size"] == 64 * 1024**3
    public = operator_view.view(controller.STATE)
    assert operator_view.lines(public).count("SparkCache: 64 GiB per Spark") == 1
    assert operator_view.lines(public).count("SparkCache integrity: " + (mode or "fast")) == 1
    assert operator_view.lines(public).count("SparkCache logs: debug") == bool(mode)


def test_install_help_has_terse_size_description(capsys):
    with pytest.raises(SystemExit) as stopped:
        install_workflow.main(["--help"])
    assert stopped.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "--sparkcache-size SIZE" in text
    assert "per-Spark SparkCache disk budget (default 32G)" in text
    assert "SparkCache payload check: fast, sha256 or none (default fast)" in text
    assert "detailed SparkCache logs" in text


@pytest.mark.parametrize("mode", ["fast", "sha256", "none"])
@pytest.mark.parametrize("debug", [False, True])
def test_install_plan_persists_integrity_and_debug(machine, image_path, capsys, mode, debug):
    flags = ["--sparkcache-integrity", mode, *(["--sparkcache-debug"] if debug else [])]
    assert sparkring.main(["install", "--profile", PROFILE, "--image-lock", str(image_path), "--plan", *flags]) == 0
    captured = capsys.readouterr()
    lines = (captured.out + captured.err).splitlines()
    assert lines.count("SparkCache integrity: " + mode) == 1
    assert lines.count("SparkCache logs: debug") == debug
    path, = (controller.STATE / "deployments").glob(PROFILE + "*/deployment.lock.json")
    lock = installer.load(path.parent)
    assert lock["serving"]["sparkcache_integrity"] == mode
    assert bool(lock["serving"].get("sparkcache_debug")) == debug
    command = install_workflow.saved_plan(path.parent)["command"]
    assert "--sparkcache-integrity " + mode in command
    assert ("--sparkcache-debug" in command) == debug


@pytest.mark.parametrize("flags", [["--sparkcache-integrity", "fast"], ["--sparkcache-debug"]])
def test_install_refuses_cache_options_before_survey(machine, sparks, capsys, flags):
    sparks.survey = lambda *a, **k: pytest.fail("invalid option reached the survey")
    assert sparkring.main(["install", "--profile", "qwen38-flash-next-tp2", "--plan", *flags]) == 2
    assert flags[0] + " requires a SparkCache profile" in capsys.readouterr().err
    assert not list((controller.STATE / "deployments").glob("qwen*/deployment.lock.json"))
