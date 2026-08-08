from __future__ import annotations

import hashlib
import json

import acceptance_gate
import exl3_correctness_gate as correctness


MODEL = "glm-5.2-exl3-tr3-3.25bpw"


def token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(
        acceptance_gate.canonical_json(token_ids).encode("utf-8")
    ).hexdigest()


def write_config(tmp_path, *, expected="observed", duplicate=False):
    cases = [
        {
            "id": "focused-case",
            "prompt": "Return exactly the word stable.",
            "seed": 20260807,
            "max_tokens": 16,
            "expected_token_ids_sha256": (
                token_hash([10, 20, 30]) if expected == "observed" else expected
            ),
        }
    ]
    if duplicate:
        cases.append(dict(cases[0]))
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps({"schema": correctness.SCHEMA, "cases": cases}),
        encoding="utf-8",
    )
    return path


def argv(path, command="run"):
    return [
        "--config",
        str(path),
        "--base-url",
        "http://rank0.test:8000",
        "--model",
        MODEL,
        command,
    ]


class FakeHttp:
    def __init__(self, token_runs=None, completion_status=200):
        self.token_runs = iter(token_runs or [[10, 20, 30]] * 3)
        self.completion_status = completion_status
        self.requests = []

    def get_json(self, url, timeout=30.0):
        self.requests.append(("GET", url))
        return 200, {"status": "ok"}

    def post_json(self, url, payload, timeout=30.0):
        self.requests.append(("POST", url))
        if url.endswith("/v1/completions"):
            return self.completion_status, {"choices": [{"text": "stable"}]}
        if url.endswith("/tokenize"):
            return 200, {"tokens": next(self.token_runs)}
        return 404, {}


def test_plan_is_connection_free_and_allows_focused_case(tmp_path, capsys):
    path = write_config(tmp_path, expected=None)
    refusing = acceptance_gate.RefusingExecutor()
    assert correctness.main(argv(path, "plan"), http=refusing) == correctness.EXIT_OK
    plan = json.loads(capsys.readouterr().out)
    assert plan["mutates_remote"] is False
    assert plan["execute_requested"] is False
    assert plan["case_ids"] == ["focused-case"]


def test_run_without_execute_remains_dry(tmp_path, capsys):
    path = write_config(tmp_path)
    assert correctness.main(argv(path), http=acceptance_gate.RefusingExecutor()) == 0
    assert json.loads(capsys.readouterr().out)["execute_requested"] is False


def test_expected_token_hash_passes(tmp_path, capsys):
    path = write_config(tmp_path)
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(args, http=FakeHttp()) == correctness.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "pass"
    assert report["cases"][0]["status"] == "pass"
    assert len(report["cases"][0]["observations"]) == 3


def test_null_expected_hash_records_baseline_but_does_not_pass(tmp_path, capsys):
    path = write_config(tmp_path, expected=None)
    args = argv(path)[:-1] + ["--execute", "run"]
    assert (
        correctness.main(args, http=FakeHttp())
        == correctness.EXIT_BASELINE_RECORDED
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "baseline-recorded"
    assert report["baselines"] == ["focused-case"]


def test_repeat_divergence_is_functional_failure(tmp_path, capsys):
    path = write_config(tmp_path)
    http = FakeHttp(token_runs=[[10, 20, 30], [10, 99], [10, 20, 30]])
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_FUNCTIONAL_FAIL
    report = json.loads(capsys.readouterr().out)
    assert report["failures"] == ["focused-case"]
    assert report["cases"][0]["failure"] == "repetitions diverged"


def test_duplicate_case_is_configuration_error(tmp_path, capsys):
    path = write_config(tmp_path, duplicate=True)
    assert correctness.main(argv(path, "plan")) == correctness.EXIT_CONFIG_ERROR
    assert "duplicate correctness case id" in capsys.readouterr().err


def test_bad_expected_hash_is_configuration_error(tmp_path, capsys):
    path = write_config(tmp_path, expected="not-a-hash")
    assert correctness.main(argv(path, "plan")) == correctness.EXIT_CONFIG_ERROR
    assert "must be null or SHA-256" in capsys.readouterr().err


def test_http_failure_is_functional_failure(tmp_path, capsys):
    path = write_config(tmp_path)
    args = argv(path)[:-1] + ["--execute", "run"]
    assert (
        correctness.main(args, http=FakeHttp(completion_status=500))
        == correctness.EXIT_FUNCTIONAL_FAIL
    )
    assert "completion failed with HTTP 500" in capsys.readouterr().err
