"""Configuration proof and intentional failures; never start containers or inference."""
import copy
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from runtime.common import compose_validation as validation, standalone_compose
from scripts import sparkring


@pytest.fixture
def recipe(tmp_path):
    path = tmp_path / "compose.yaml"
    path.write_text(standalone_compose.render(standalone_compose.SUPPORTED[0]), encoding="utf-8")
    return path


@pytest.mark.parametrize("profile", standalone_compose.SUPPORTED)
def test_real_compose_and_mock_rank_registration_pass(profile, tmp_path):
    if not shutil.which("docker"):
        pytest.skip("Docker Compose CLI required, but no daemon needed")
    path = tmp_path / "compose.yaml"
    path.write_text(standalone_compose.render(profile), encoding="utf-8")
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[:4] == ["docker", "--context", "default", "compose"]
        assert "version" in argv or "config" in argv
        assert not set(argv) & {"up", "pull", "create", "start", "run", "down"}
        return subprocess.run(argv, **kwargs)
    report = validation.validate(path, run=run)
    assert report["passed"], report
    assert len(report["checks"]) == 11
    assert len(calls) == 9  # version, two ranks, no profile, five missing inputs
    assert len(report["mock_events"]) == 3
    assert "remain untested" in report["scope"]


@pytest.mark.parametrize("mutation", ["image", "weights-writable", "headless", "rank", "external-file", "external-include", "checkpoint", "duplicate"])
def test_broken_recipes_are_rejected_before_invoking_compose(recipe, mutation):
    text = recipe.read_text(encoding="utf-8")
    if mutation == "duplicate":
        recipe.write_text(text + "\nservices: {}\n", encoding="utf-8")
    else:
        data = yaml.safe_load(text)
        service = data["services"]["rank0"]
        if mutation == "image":
            service["image"] = "unverified:latest"
        elif mutation == "weights-writable":
            service["volumes"][0]["read_only"] = False
        elif mutation == "headless":
            service["command"].append("--headless")
        elif mutation == "rank":
            command = service["command"]
            command[command.index("--node-rank") + 1] = "1"
        elif mutation == "external-file":
            service["env_file"] = "/private/credentials.env"
        elif mutation == "external-include":
            data["include"] = ["https://example.invalid/compose.yaml"]
        else:
            data["x-sparkring"]["model_revision"] = "0" * 40
        recipe.write_text(yaml.safe_dump(data), encoding="utf-8")
    report = validation.validate(recipe, run=lambda *a, **k: pytest.fail("Must reject before Docker can read external inputs"))
    assert not report["passed"]
    assert report["checks"][-1]["passed"] is False


def test_comments_and_yaml_formatting_are_not_runtime_changes(recipe):
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    # Inline service mappings can replace the generated anchor layout.
    del data["x-runtime"]
    del data["x-environment"]
    profile, _, _ = validation.inspect_recipe("# reformatted\n" + yaml.safe_dump(data))
    assert profile == standalone_compose.SUPPORTED[0]


def test_compose_unavailable_is_not_a_passing_validation(recipe):
    report = validation.validate(recipe, run=lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="plugin unavailable"))
    assert not report["passed"]
    assert report["checks"][-1]["name"] == "Docker Compose CLI"


def test_compose_selecting_both_ranks_is_rejected(recipe):
    def run(argv, **kwargs):
        if "version" in argv:
            return SimpleNamespace(returncode=0, stdout="test", stderr="")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"services": {"rank0": {}, "rank1": {}}}), stderr="")
    report = validation.validate(recipe, run=run)
    assert not report["passed"]
    assert "only the requested rank" in report["checks"][-1]["error"]


def test_environment_does_not_inherit_tokens_or_compose_selectors(monkeypatch):
    for key in ("HF_TOKEN", "COMPOSE_PROFILES", "COMPOSE_FILE", "DOCKER_HOST"):
        monkeypatch.setenv(key, "must-not-propagate")
    monkeypatch.setenv("ProgramFiles", "C:/Program Files")
    env = validation.environment(0)
    assert not any(env.get(key) for key in ("HF_TOKEN", "COMPOSE_PROFILES", "COMPOSE_FILE", "DOCKER_HOST"))
    assert {key.upper(): value for key, value in env.items()}["PROGRAMFILES"] == "C:/Program Files"


@pytest.mark.parametrize("mutation", ["image", "model", "master", "port", "headless", "rank", "same-host"])
def test_mock_pair_detects_incompatible_rank_registrations(mutation):
    profile = standalone_compose.SUPPORTED[0]
    services = {rank: validation.expected_service(profile, "nvfp4-spark", rank, validation.environment(rank)) for rank in (0, 1)}
    assert validation.mock_pair(services)[-1].endswith("simulated pair ready")
    bad = copy.deepcopy(services)
    worker = bad[1]
    if mutation == "image":
        worker["image"] = "different"
    elif mutation == "same-host":
        worker["environment"]["VLLM_HOST_IP"] = bad[0]["environment"]["VLLM_HOST_IP"]
    elif mutation == "headless":
        worker["command"].remove("--headless")
    else:
        flag = {"model": "--served-model-name", "master": "--master-addr", "port": "--master-port", "rank": "--node-rank"}[mutation]
        worker["command"][worker["command"].index(flag) + 1] = "different"
    with pytest.raises(ValueError):
        validation.mock_pair(bad)


def test_cli_emits_json_and_preserves_existing_reports(recipe, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(validation, "validate", lambda path: {"passed": True, "file": str(path), "checks": []})
    output = tmp_path / "report.json"
    assert sparkring.main(["validate-compose", str(recipe), "--json", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert json.loads(output.read_text())["passed"] is True
    assert sparkring.main(["validate-compose", str(recipe), "--output", str(output)]) == 2
    assert "already exists" in capsys.readouterr().err
