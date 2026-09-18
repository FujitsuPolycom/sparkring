"""Protected source gates retain a common oracle and reject false pass evidence."""

import json
from pathlib import Path

import pytest

from .standalone_gate import digest, run, selected_paths, source_environment


def fixture_suite(tmp_path, body="def test_common(): assert True\n"):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "common.py").write_text(body)
    (tests / "baseline.py").write_text("def test_baseline_api(): assert True\n")
    (tests / "candidate.py").write_text("def test_candidate_api(): assert True\n")
    suite = {
        "schema": "sparkring-protected-source-suite/v1",
        "component": "vllm",
        "common": ["common.py"],
        "baseline_only": ["baseline.py"],
        "candidate_only": ["candidate.py"],
        "files": {path.name: digest(path) for path in tests.glob("*.py")},
    }
    manifest = tests / "suite.json"
    manifest.write_text(json.dumps(suite))
    return manifest


def test_variant_scopes_never_replace_common_checks(tmp_path):
    manifest = fixture_suite(tmp_path)
    _, baseline = selected_paths(manifest, "baseline")
    _, candidate = selected_paths(manifest, "candidate")
    _, upstream = selected_paths(manifest, "upstream")
    assert baseline["common"] == candidate["common"] == upstream["common"]
    assert set(baseline) == {"common", "baseline_only"}
    assert set(candidate) == set(upstream) == {"common", "candidate_only"}


def test_modified_protected_test_is_rejected(tmp_path):
    manifest = fixture_suite(tmp_path)
    (manifest.parent / "common.py").write_text("def test_weaker(): pass\n")
    with pytest.raises(ValueError, match="differs"):
        selected_paths(manifest, "candidate")


@pytest.mark.parametrize(
    "selection", ["../escape.py", "common.py::-k bypass", "common.py::"]
)
def test_selection_cannot_escape_or_inject_pytest_flags(tmp_path, selection):
    manifest = fixture_suite(tmp_path)
    suite = json.loads(manifest.read_text())
    suite["common"] = [selection]
    manifest.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="selection"):
        selected_paths(manifest, "candidate")


def test_source_environment_is_selected_and_not_inherited(monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "--ignore=common.py")
    monkeypatch.setenv("PYTEST_PLUGINS", "untrusted")
    monkeypatch.setenv("SPARKRING_TEST_SOURCE_ROOT", "wrong")
    env = source_environment("selected", "protected-baseline", "protected-tests")
    assert (
        env["SPARKRING_TEST_SOURCE_ROOT"]
        == env["SPARKRING_B12X_SOURCE_ROOT"]
        == "selected"
    )
    assert env["SPARKRING_ORACLE_BASELINE"] == "protected-baseline"
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert "PYTEST_ADDOPTS" not in env and "PYTEST_PLUGINS" not in env


@pytest.mark.parametrize("skipped", [False, True])
def test_runner_ignores_ambient_conftest_and_refuses_skipped_pass(
    tmp_path, monkeypatch, skipped
):
    body = (
        "import pytest\n@pytest.mark.skip(reason='missing evidence')\ndef test_common(): assert True\n"
        if skipped
        else "def test_common(): assert True\n"
    )
    manifest = fixture_suite(tmp_path, body)
    (tmp_path / "conftest.py").write_text(
        "raise RuntimeError('ambient conftest executed')\n"
    )
    source, baseline = tmp_path / "source", tmp_path / "baseline-source"
    (source / "vllm").mkdir(parents=True)
    (baseline / "vllm").mkdir(parents=True)
    for name, value in {
        "GATE": "cpu",
        "SUBJECT": "candidate-sha",
        "INPUT": "policy-input",
        "VARIANT": "candidate",
    }.items():
        monkeypatch.setenv("SPARKRING_UPGRADE_" + name, value)
    result = run(source, baseline, manifest, tmp_path / "result.json")
    assert result["outcome"] == ("failed" if skipped else "passed")
    assert result["skipped"] == int(skipped)
    assert result["scopes"]["candidate_only"]["outcome"] == "passed"
    assert result["subject_sha256"] == "candidate-sha"
