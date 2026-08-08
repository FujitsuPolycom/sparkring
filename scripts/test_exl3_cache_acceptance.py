from __future__ import annotations

import json

import acceptance_gate
import exl3_cache_acceptance as cache


def argv(command: str = "run") -> list[str]:
    return [
        "--site",
        "generated/site.yaml",
        "--profile",
        "generated/launch.json",
        "--base-url",
        "http://rank0.test:8000",
        "--model",
        "glm-5.2-exl3-tr3-3.25bpw",
        "--probe-id",
        "acceptance-test-1",
        command,
    ]


def lmcache_status(objects: int) -> str:
    ranks = {}
    for rank in range(4):
        status = {
            "is_healthy": True,
            "registered_gpu_ids": [1000 + rank],
            "storage_manager": {
                "l1_manager": {
                    "is_healthy": True,
                    "total_object_count": objects,
                    "memory_used_bytes": objects * 4096,
                    "write_locked_count": 0,
                    "read_locked_count": 0,
                    "temporary_count": 0,
                }
            },
        }
        ranks[str(rank)] = {
            "exit_code": 0,
            "stdout": '{"status":"healthy"}' + json.dumps(status),
            "stderr": "",
        }
    return json.dumps({"server_health": ranks, "ready": {}})


class FakeExecutor:
    def __init__(self, counts=(0, 10, 10, 0, 10), fail_command=None):
        self.counts = iter(counts)
        self.fail_command = fail_command
        self.commands = []

    def run(self, command, timeout=None):
        command = list(command)
        self.commands.append(command)
        name = command[-1]
        failed = name == self.fail_command
        stdout = lmcache_status(next(self.counts)) if name == "status" else "{}"
        return acceptance_gate.CommandResult(
            argv=command,
            exit_code=1 if failed else 0,
            stdout=stdout,
            stderr="simulated failure" if failed else "",
            duration_seconds=0.1,
        )


class FakeHttp:
    def __init__(self, ttfts=(1.0, 0.2, 0.3, 1.1, 0.2), texts=None):
        self.ttfts = iter(ttfts)
        self.texts = iter(texts or ["stable"] * 5)
        self.streams = 0

    def get_json(self, url, timeout=30.0):
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/v1/models"):
            return 200, {"data": [{"id": "glm-5.2-exl3-tr3-3.25bpw"}]}
        return 404, {}

    def stream_completion(self, url, payload, timeout=1800.0):
        self.streams += 1
        return acceptance_gate.StreamSample(
            ttft_seconds=next(self.ttfts),
            total_seconds=1.5,
            tokens=8,
            text=next(self.texts),
        )


def test_plan_is_connection_free_and_discloses_mutation(capsys):
    refusing = acceptance_gate.RefusingExecutor()
    assert cache.main(argv("plan"), executor=refusing, http=refusing) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["mutates_remote"] is True
    assert document["execute_requested"] is False
    assert "restart-engines" in document["commands"]
    assert "restart-stack" in document["commands"]


def test_run_without_execute_remains_a_dry_run(capsys):
    refusing = acceptance_gate.RefusingExecutor()
    assert cache.main(argv(), executor=refusing, http=refusing) == 0
    assert json.loads(capsys.readouterr().out)["execute_requested"] is False


def test_execute_requires_exact_confirmation(capsys):
    result = cache.main(argv()[:-1] + ["--execute", "run"])
    assert result == cache.EXIT_CONFIG_ERROR
    assert cache.CONFIRMATION in capsys.readouterr().err


def test_full_cache_boundary_passes_with_engine_reuse_and_server_reset(capsys):
    executor = FakeExecutor()
    http = FakeHttp()
    arguments = argv()[:-1] + [
        "--execute",
        "--confirmation",
        cache.CONFIRMATION,
        "run",
    ]
    assert cache.main(arguments, executor=executor, http=http) == cache.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "pass"
    assert report["cache_snapshots"]["after_engine_restart"]["0"]["objects"] == 10
    assert report["cache_snapshots"]["after_server_restart"]["0"]["objects"] == 0
    assert report["cache_snapshots"]["final"]["0"]["objects"] == 10
    assert http.streams == 5
    assert [command[-1] for command in executor.commands] == [
        "status",
        "status",
        "restart-engines",
        "status",
        "restart-stack",
        "status",
        "status",
    ]


def test_output_divergence_is_a_functional_failure(capsys):
    executor = FakeExecutor()
    http = FakeHttp(texts=["stable", "stable", "changed", "stable", "stable"])
    arguments = argv()[:-1] + [
        "--execute",
        "--confirmation",
        cache.CONFIRMATION,
        "run",
    ]
    assert (
        cache.main(arguments, executor=executor, http=http)
        == cache.EXIT_FUNCTIONAL_FAIL
    )
    report = json.loads(capsys.readouterr().out)
    assert any("output changed" in failure for failure in report["failures"])


def test_server_restart_must_empty_volatile_l1(capsys):
    executor = FakeExecutor(counts=(0, 10, 10, 4, 10))
    arguments = argv()[:-1] + [
        "--execute",
        "--confirmation",
        cache.CONFIRMATION,
        "run",
    ]
    assert (
        cache.main(arguments, executor=executor, http=FakeHttp())
        == cache.EXIT_FUNCTIONAL_FAIL
    )
    report = json.loads(capsys.readouterr().out)
    assert any("not empty" in failure for failure in report["failures"])


def test_launcher_restart_failure_is_reported(capsys):
    executor = FakeExecutor(fail_command="restart-engines")
    arguments = argv()[:-1] + [
        "--execute",
        "--confirmation",
        cache.CONFIRMATION,
        "run",
    ]
    assert (
        cache.main(arguments, executor=executor, http=FakeHttp())
        == cache.EXIT_FUNCTIONAL_FAIL
    )
    report = json.loads(capsys.readouterr().out)
    assert "restart-engines exited 1" in report["failures"][0]
