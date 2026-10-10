"""``sparkring install --image REF --transport libsircl --plan``; SSH to the Sparks simulated.

Each simulated Spark answers ``docker image inspect``, ``sha256sum`` of the
host library and the two runs of the stock-image probe.
"""
import copy
import json
from pathlib import Path

import pytest
import yaml

from runtime.common import fabric_document, installer
from runtime.common.test_stock_image import FUNCTIONS, LIBRARY, REFERENCE, TARGET, binding, facts, image
from runtime.host import controller, fabric, install_workflow as flow, relays
from runtime.host.test_install_workflow import machine, sparks  # noqa: F401  (pytest fixtures)
from scripts import sparkring

MARKER = {"binary": relays.MARKER_BINARY, "sha256": "ab" * 32}


class Sparks:
    """``discovery.ssh`` of Sparks that hold the stock image and the host library."""

    def __init__(self):
        self.images, self.facts, self.binding, self.digests, self.calls = {}, {}, {}, {}, []

    def __call__(self, host, argv, *, data=None, **options):
        self.calls.append((host, list(argv), data is not None))
        if argv[:5] == ["sudo", "-n", "docker", "image", "inspect"]:
            return json.dumps([self.images.get(host, image())])
        if argv[0] == "sha256sum":
            return f"{self.digests.get(host, 'b' * 64)}  {argv[1]}\n"
        assert argv[:4] == ["sudo", "-n", "docker", "run"] and "--network" in argv and "--read-only" in argv
        assert data is not None and "--pull" in argv and argv[argv.index("--pull") + 1] == "never"
        if "--binding" in argv:
            environment = dict(item.split("=", 1) for item in argv[argv.index("--entrypoint") - 1::-1]
                               if "=" in item and item.startswith(("VLLM_NCCL_SO_PATH=", "LD_PRELOAD=")))
            assert environment == {"VLLM_NCCL_SO_PATH": TARGET, "LD_PRELOAD": TARGET}
            return "STOCK-IMAGE-PROBE " + json.dumps(self.binding.get(host, binding())) + "\n"
        return "STOCK-IMAGE-PROBE " + json.dumps(self.facts.get(host, facts())) + "\n"


@pytest.fixture
def stock(machine, monkeypatch):  # noqa: F811
    """The machine's cluster with its fabric document recorded and simulated Sparks."""
    cluster = installer.read(controller.STATE / "cluster.json")
    document, _ = fabric.prepare(cluster["plan"], cluster="test", marker=MARKER)
    (controller.STATE / "fabric.json").write_text(fabric_document.encoded(document))
    value = Sparks()
    monkeypatch.setattr(flow.discovery, "ssh", value)
    return value, document


def plan(*extra, arguments=("--served-model-name", "m")):
    return sparkring.main(["install", "--image", REFERENCE, "--transport", "libsircl", "--libsircl-library", LIBRARY,
                           "--model-path", "/srv/models/m", "--json", *extra, "--", *arguments])


def test_a_plan_checks_every_spark_and_writes_each_ranks_compose_file(stock, capsys):
    value, document = stock
    assert plan("--plan") == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["state"] == "planned" and result["status"] == "research-only"
    assert result["group"]["name"] == "pair" and result["positions"] == [0, 1]
    assert result["image"] == {"reference": REFERENCE, "id": image()["Id"]}
    assert result["library"] == {"path": LIBRARY, "sha256": "b" * 64}
    directory = Path(result["directory"]["node_a"])
    assert result["files"] == ["compose.rank0.yaml", "compose.rank1.yaml"]
    for rank in (0, 1):
        service = yaml.safe_load((directory / f"compose.rank{rank}.yaml").read_text())["services"]["model"]
        assert service["environment"]["LIBSIRCL_POSITION"] == str(rank)
        assert service["environment"]["VLLM_NCCL_SO_PATH"] == service["environment"]["LD_PRELOAD"] == TARGET
        assert "--served-model-name" in service["command"]
    assert json.loads((directory / "plan.json").read_text())["name"] == result["name"]
    # Two probes per Spark: the facts, then the binding; never a pull.
    probes = [call for call in value.calls if call[1][:4] == ["sudo", "-n", "docker", "run"]]
    assert len(probes) == 4 and all(call[2] for call in probes)
    assert not any("pull" == word for call in value.calls for word in call[1])
    assert "Stock image " + REFERENCE in out.err and "Start every rank:" in out.err
    assert result["commands"]["start"][0].endswith("compose.rank0.yaml up -d'")


@pytest.mark.parametrize("extra, field, message", [
    ((), "plan", "plans only: add --plan"),
    (("--plan", "--profile", "qwen38-flash-next-tp2"), "profile", "takes no --profile"),
])
def test_requests_the_option_does_not_plan_are_refused_before_any_spark_is_asked(stock, capsys, extra, field, message):
    value, _ = stock
    assert plan(*extra) == 3
    refused = json.loads(capsys.readouterr().out)
    assert refused["field"] == field and message in refused["message"] and not value.calls


def test_a_stock_image_without_the_libsircl_transport_is_refused(stock, capsys):
    assert sparkring.main(["install", "--image", REFERENCE, "--plan", "--json", "--libsircl-library", LIBRARY,
                           "--model-path", "/srv/models/m"]) == 3
    assert "runs only with --transport libsircl" in json.loads(capsys.readouterr().out)["message"]


def test_a_spark_whose_image_libsircl_cannot_run_on_is_refused_with_each_reason(stock, capsys):
    value, _ = stock
    hosts = [host["host"] for host in installer.read(controller.STATE / "cluster.json")["plan"]["spec"]["hosts"]]
    value.facts[hosts[1]] = facts(**{"torch.nccl": "static", "library.fail_stop": False})
    assert plan("--plan") == 3
    refused = json.loads(capsys.readouterr().out)
    assert refused["field"] == "image"
    problems = refused["details"]["problems"][f"rank 1 ({hosts[1]})"]
    assert any("torch NCCL is static" in problem for problem in problems)
    assert any("no fail-stop mode" in problem for problem in problems)


def test_a_torch_that_bound_another_nccl_and_differing_bytes_across_sparks_are_refused(stock, capsys):
    value, _ = stock
    hosts = [host["host"] for host in installer.read(controller.STATE / "cluster.json")["plan"]["spec"]["hosts"]]
    value.binding[hosts[0]] = binding(torch_nccl_version=[2, 29, 7])
    value.digests[hosts[1]] = "c" * 64
    assert plan("--plan") == 3
    problems = json.loads(capsys.readouterr().out)["details"]["problems"]
    assert "torch reports NCCL [2, 29, 7]" in problems[f"rank 0 ({hosts[0]})"][0]
    assert problems["library"] == [f"the Sparks hold different bytes at {LIBRARY}"]


def test_arguments_after_double_dash_need_a_stock_image(machine, capsys):  # noqa: F811
    with pytest.raises(SystemExit):
        sparkring.main(["install", "--profile", "qwen38-flash-next-tp2", "--plan", "--", "--enforce-eager"])
    assert "use them with --image REF" in capsys.readouterr().err


def test_the_fixture_functions_are_twenty_distinct_names():
    # The 20 names of PyNccl's NCCLLibrary in Local Inference Lab's integration/karmic-kraken-beta at 57a80980bb.
    assert len(FUNCTIONS) == 20 and len(set(FUNCTIONS)) == 20
    assert copy.deepcopy(facts())["vllm"]["pynccl_functions"] == FUNCTIONS
