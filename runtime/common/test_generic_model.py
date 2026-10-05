"""A generic deployment: a Hugging Face model named at install time, served on a template profile.

No test contacts the network: the Hub is the fake of scripts/test_pin_checkpoint.py.
"""
import copy
import io
import json
import urllib.error

import pytest

from runtime.common import generic_model, installer, installer_image
from runtime.common.test_compose_installer import install_site
from scripts.test_pin_checkpoint import REVISION, WEIGHTS, FakeHub, revision_files

REPOSITORY = "example-owner/Example-Model"


def hub(config=b'{"model_type": "example"}\n'):
    small, lfs = revision_files()
    small["config.json"] = config
    return FakeHub(small, lfs)


def resolved(config=b'{"model_type": "example"}\n', **options):
    return generic_model.resolve(REPOSITORY, hub=hub(config), **options)


@pytest.mark.parametrize("text, expected", [
    ("Qwen/Qwen3-32B", ("Qwen/Qwen3-32B", None)),
    ("Qwen/Qwen3-32B@main", ("Qwen/Qwen3-32B", "main")),
    ("org/model.v2@" + "a" * 40, ("org/model.v2", "a" * 40)),
])
def test_a_model_is_owner_name_with_an_optional_revision(text, expected):
    assert generic_model.parse_model(text) == expected


@pytest.mark.parametrize("text", ["Qwen3-32B", "a/b/c", "org/--model", "org/model@", "org/model@bad rev", "../x/y"])
def test_other_model_names_are_refused(text):
    with pytest.raises(ValueError):
        generic_model.parse_model(text)


def test_vllm_arguments_are_options_with_values_and_never_sparkring_s_own():
    given = ["--max-model-len", "65536", "--enable-auto-tool-choice", "--tool-call-parser=hermes", "--rope-scaling",
             '{"type":"yarn","factor":2}', "--seed", "-1"]
    assert generic_model.arguments(given) == given
    for owned, instead in (("--port", "--api-port"), ("--host", "--api-bind"), ("--served-model-name", "--name"),
                           ("--tensor-parallel-size", None), ("-tp", None), ("--data-parallel-rank", None),
                           ("--master-port=1", None)):
        with pytest.raises(ValueError) as refused:
            generic_model.arguments([owned, "2"] if "=" not in owned else [owned])
        assert "SparkRing sets vLLM's" in str(refused.value)
        assert (f"use {instead}" in str(refused.value)) if instead else "use" not in str(refused.value)
    for bad, message in ((["/models/other"], "follows no option"), (["--a", "b", "c"], "follows no option"),
                         (["--x", "a\nb"], "control characters")):
        with pytest.raises(ValueError, match=message):
            generic_model.arguments(bad)


def test_arguments_replace_the_template_s_values_keep_its_switches_and_add_others():
    template = ["--port", "8000", "--max-model-len", "32768", "--enable-prefix-caching"]
    merged = generic_model.merge(template, ["--max-model-len=8192", "--enable-prefix-caching", "--quantization", "fp8",
                                            "--enforce-eager"])
    assert merged == ["--port", "8000", "--max-model-len", "8192", "--enable-prefix-caching", "--quantization", "fp8",
                      "--enforce-eager"]


def test_resolving_pins_the_commit_and_reads_the_configuration_without_downloading_weights():
    hub_ = hub(b'{"model_type": "example", "max_position_embeddings": 8192, "auto_map": {"AutoConfig": "x.Y"}}\n')
    value = generic_model.resolve(REPOSITORY + "@main", hub=hub_, extra=["--enforce-eager"])
    assert value["model"]["repository"] == REPOSITORY and value["model"]["revision"] == REVISION
    assert value["pins"]["weights"] == sorted(WEIGHTS) and not set(WEIGHTS) & set(hub_.downloads)
    assert value["model"]["config_sha256"] == value["pins"]["files"]["config.json"]["sha256"]
    assert value["served_model_name"] == "Example-Model" and value["arguments"] == ["--enforce-eager"]
    # The context follows the model's own maximum when it is below the template's.
    assert value["context_length"] == 8192 and value["remote_code"] is True
    assert any("--trust-remote-code" in line for line in generic_model.notes(value))
    plain = resolved(served_model_name="mine")
    assert plain["context_length"] == generic_model.DEFAULT_CONTEXT and plain["remote_code"] is False
    assert plain["served_model_name"] == "mine"


@pytest.mark.parametrize("code, message", [(401, "gated or private"), (403, "gated or private"), (404, "was not found")])
def test_a_repository_the_hub_does_not_serve_is_named(code, message):
    class Refusing(FakeHub):
        def commit(self, repository, revision):
            raise urllib.error.HTTPError("https://huggingface.co", code, "refused", {}, io.BytesIO())
    small, lfs = revision_files()
    with pytest.raises(ValueError, match=message):
        generic_model.resolve(REPOSITORY, hub=Refusing(small, lfs))


def test_a_model_without_a_weight_index_is_refused():
    small, lfs = revision_files()
    del lfs["model.safetensors.index.json"]
    with pytest.raises(ValueError, match="cannot be served as a generic model"):
        generic_model.resolve(REPOSITORY, hub=FakeHub(small, lfs))


@pytest.mark.parametrize("nodes", [2, 4])
def test_a_generic_lock_records_the_request_and_serves_it_on_the_template(nodes):
    profile = generic_model.profile_for(nodes)
    value = resolved(extra=["--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--max-num-seqs", "8"])
    lock = installer.make_lock(profile, install_site(nodes), "1" * 40, "2" * 64,
                               image_runtime=installer_image.for_profile(profile), generic=value)
    card = lock["selection"]
    assert (card["profile"], card["model_repository"], card["model_revision"]) == (profile, REPOSITORY, REVISION)
    assert installer.checkpoint_pins(card) == value["pins"]
    assert installer.checkpoint_contract(card) == value["model"]
    assert installer.validate(copy.deepcopy(lock)) == lock
    specs = installer.specifications(lock)
    assert len(specs) == nodes
    command = specs[0].command
    at = command.index
    assert command[at("--served-model-name") + 1] == "Example-Model"
    assert command[at("--tensor-parallel-size") + 1] == str(nodes)
    assert command[at("--max-model-len") + 1] == str(generic_model.DEFAULT_CONTEXT)
    assert command[at("--max-num-seqs") + 1] == "8" and command.count("--max-num-seqs") == 1
    assert command[at("--tool-call-parser") + 1] == "hermes" and "--enable-auto-tool-choice" in command
    assert installer.connection(lock)["model"] == "Example-Model"
    assert installer.connection(lock)["port"] == (8000 if nodes == 2 else 8015)
    # The environment carries the fabric settings, not another model's.
    environment = specs[0].environment
    assert environment["VLLM_ENABLE_ROCE_ALLREDUCE"] == "1" and not any(key.startswith("VLLM_QWEN") for key in environment)


def test_a_changed_request_or_pins_fails_the_lock():
    profile = generic_model.profile_for(2)
    value = resolved()
    lock = installer.make_lock(profile, install_site(2), "1" * 40, "2" * 64,
                               image_runtime=installer_image.for_profile(profile), generic=value)
    changed = copy.deepcopy(lock)
    changed["selection"]["generic"]["served_model_name"] = "other"
    with pytest.raises(ValueError):
        installer.validate(changed)
    tampered = copy.deepcopy(lock["selection"])
    tampered["generic"]["pins"]["files"]["config.json"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="config.json differs"):
        installer.checkpoint_pins(tampered)


def test_a_template_installs_only_with_a_request_and_a_profile_never_takes_one():
    value = resolved()
    with pytest.raises(ValueError, match="template of `sparkring install --model`"):
        installer.make_lock("generic-vllm-tp2", install_site(2), "1" * 40, "2" * 64,
                            image_runtime=installer_image.for_profile("generic-vllm-tp2"))
    with pytest.raises(ValueError, match="serves its own model"):
        installer.make_lock("qwen38-flash-next-tp2", install_site(2), "1" * 40, "2" * 64,
                            image_runtime=installer_image.for_profile("qwen38-flash-next-tp2"), generic=value)


def test_the_templates_run_on_every_installer_image():
    for row in installer_image.catalog():
        if row["lock"]["schema"] == installer_image.SCHEMA:
            for profile in generic_model.PROFILES.values():
                assert installer_image.validate(row["lock"], profile) is row["lock"]


def test_serving_settings_apply_to_a_generic_deployment():
    profile = generic_model.profile_for(2)
    lock = installer.make_lock(profile, install_site(2), "1" * 40, "2" * 64,
                               image_runtime=installer_image.for_profile(profile), generic=resolved(),
                               settings={"context_length": 16384, "max_concurrency": 4, "api_port": 9100})
    command = installer.specifications(lock)[0].command
    assert command[command.index("--max-model-len") + 1] == "16384"
    assert command[command.index("--max-num-seqs") + 1] == "4"
    assert installer.connection(lock)["port"] == 9100
    assert json.loads(json.dumps(lock))["selection"]["generic"]["schema"] == generic_model.SCHEMA


def test_a_repository_with_more_files_than_a_lock_carries_is_refused():
    small, lfs = revision_files()
    for number in range(400):
        small[f"extra/file-{number:04d}-with-a-long-name-so-the-list-grows.json"] = b"{}" + bytes([10])
    with pytest.raises(ValueError, match="holds about 200 files"):
        generic_model.resolve(REPOSITORY, hub=FakeHub(small, lfs))
