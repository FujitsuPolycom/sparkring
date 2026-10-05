"""`sparkring install --model`: a Hugging Face model on the simulated pair and ring of test_install_workflow.

The Hub is the fake of scripts/test_pin_checkpoint.py, so no test contacts the network.
"""
import json

import pytest

from runtime.common import generic_model, installer
from runtime.host import controller, node
from runtime.host.test_install_workflow import cluster, machine, sparks  # noqa: F401 (fixtures)
from runtime.host.test_ring_halves import MESH, install, ops, recorded, result, ring  # noqa: F401 (fixtures)
from scripts import sparkring
from scripts.test_pin_checkpoint import REVISION, FakeHub, revision_files

MODEL = "example-owner/Example-Model"
TOOLS = ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]


@pytest.fixture
def hub(monkeypatch):
    small, lfs = revision_files()
    fake = FakeHub(small, lfs)
    resolve = generic_model.resolve
    monkeypatch.setattr(generic_model, "resolve", lambda text, **options: resolve(text, hub=fake, **options))
    return fake


def test_a_pair_serves_a_hugging_face_model_named_on_the_command_line(machine, hub, capsys):  # noqa: F811
    assert sparkring.main(["install", "--model", MODEL, "--yes", "--json", "--", *TOOLS]) == 0
    out = capsys.readouterr()
    done = json.loads(out.out)
    assert done["state"] == "complete" and done["profile"] == "generic-vllm-tp2" and done["model"] == "Example-Model"
    assert f"Generic model: {MODEL} at {REVISION[:12]}, served as Example-Model" in out.err
    lock = installer.load(done["deployment"])
    card = lock["selection"]
    assert (card["model_repository"], card["model_revision"]) == (MODEL, REVISION)
    assert card["generic"]["arguments"] == TOOLS
    # The plan's command repeats the request at its commit, so a later run installs the same deployment.
    command = done["checkpoint"]["command"]
    assert command == f"sudo sparkring install --model {MODEL}@{REVISION} -- " + " ".join(TOOLS)
    assert [row["model"] for row in lock["site"]["ranks"]] == [installer.checkpoint_directory("test", card)] * 2
    assert sparkring.main(["install", "--model", f"{MODEL}@{REVISION}", "--yes", "--json", "--", *TOOLS]) == 0
    assert json.loads(capsys.readouterr().out)["deployment"] == done["deployment"]
    # Another name or other arguments install another deployment.
    assert sparkring.main(["install", "--model", MODEL, "--name", "mine", "--yes", "--json"]) == 0
    other = json.loads(capsys.readouterr().out)
    assert other["deployment"] != done["deployment"] and other["model"] == "mine"
    assert other["checkpoint"]["command"] == f"sudo sparkring install --model {MODEL}@{REVISION} --name mine"


def test_a_ring_serves_it_on_all_four_or_on_one_half(ring, hub, capsys):  # noqa: F811
    assert install("--model", MODEL) == 0
    whole = result(capsys)
    assert whole["profile"] == "generic-vllm-tp4" and whole["nodes"] == 4 and "placement" not in whole
    ops(ring)
    assert install("--model", MODEL, "--on", "2,3", "--", "--max-num-seqs", "4") == 0
    half = result(capsys)
    assert half["profile"] == "generic-vllm-tp2" and half["placement"] == [2, 3]
    assert [row["profile"] for row in half["stops"]] == ["generic-vllm-tp4"]
    assert ops(ring) == ["generic-vllm-tp4:down", "park-ring", "generic-vllm-tp2@23:up", "generic-vllm-tp2@23:verify"]
    assert recorded() == {None: None, (0, 1): None, (2, 3): "generic-vllm-tp2@23"}
    assert half["checkpoint"]["command"] == f"sudo sparkring install --model {MODEL}@{REVISION} --on 2,3 -- --max-num-seqs 4"


@pytest.mark.parametrize("argv, message", [
    (["--model", MODEL, "--profile", "qwen38-flash-next-tp2"], "give --model or --profile"),
    (["--name", "mine", "--profile", "qwen38-flash-next-tp2"], "go with --model"),
    (["--profile", "qwen38-flash-next-tp2", "--", "--enforce-eager"], "go with --model"),
])
def test_model_options_go_together(machine, capsys, argv, message):  # noqa: F811
    with pytest.raises(SystemExit) as stop:
        sparkring.main(["install", "--yes", *argv])
    assert stop.value.code == 2 and message in capsys.readouterr().err


@pytest.mark.parametrize("extra, message", [
    (["--port", "9000"], "SparkRing sets vLLM's --port itself; use --api-port"),
    (["--tensor-parallel-size", "1"], "SparkRing sets vLLM's --tensor-parallel-size itself"),
    (["/srv/other-model"], "follows no option"),
])
def test_sparkring_s_own_vllm_options_are_refused_before_any_change(machine, hub, capsys, extra, message):  # noqa: F811
    deployments = controller.STATE / "deployments"
    before = sorted(deployments.iterdir()) if deployments.exists() else []
    assert sparkring.main(["install", "--model", MODEL, "--yes", "--json", "--", *extra]) == 3
    refused = json.loads(capsys.readouterr().out)
    assert refused["field"] == "model" and message in refused["message"]
    assert (sorted(deployments.iterdir()) if deployments.exists() else []) == before and hub.downloads == []


def test_a_template_profile_is_not_installed_by_its_name(machine, capsys):  # noqa: F811
    assert sparkring.main(["install", "--profile", "generic-vllm-tp2", "--yes", "--json"]) != 0
    refused = json.loads(capsys.readouterr().out)
    assert refused["state"] in ("needs_input", "failed")
    node.save(controller.STATE, "unused.json", {})
