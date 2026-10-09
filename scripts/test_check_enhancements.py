"""CPU tests of the enhancement catalog, its checker and its generated view."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import check_enhancements as ce  # noqa: E402
from scripts import generate_enhancements  # noqa: E402

LIBSIRCL_IMAGE = "dev-20261008-kraken-csf-sircl-libsircl-cu1342-nccl2323-status034"
LIBSIRCL_IMAGE_ID = "sha256:816c6d6a7e96f226475c2873cd180ec0d95d1ecb79e8322451816d2cc63d6747"


@pytest.fixture(scope="module")
def catalog():
    return ce.load_catalog()


def row(report, key):
    return next(item for item in report["entries"] if item["id"] == key)


def view(**changes):
    values = dict(profile="test", checkpoint=None, model_repository="example/model", model_revision="a" * 40,
                  model_name="GLM-5.3", tp=8, dcp=4, topology="direct-cycle-8", image="child-image", transport="sircl",
                  environment={}, arguments={}, source="profile")
    values.update(changes)
    return ce.Deployment(**values)


def entry(key, **changes):
    value = {"id": key, "title": key, "description": "d", "category": "kernel-backend", "sides": ["decode"],
             "models": [{"name": "GLM-5.3"}], "parallelism": {"supported": [{"tp": "any", "dcp": "any"}]},
             "enable": {"environment": {key.upper(): "1"}}, "detect": [{"env": key.upper(), "equals": "1"}],
             "provided_by": {"images": ["parent-image"]}, "status": "research-only", "gain": None, "measurements": [],
             "evidence": ["private record 2026-10-08: label"], "profiles": []}
    value.update(changes)
    return value


def synthetic(*entries, checks=()):
    images = {"parent-image": {"label": "parent-image", "parent": None, "transports": ["prepared"], "short_id": None},
              "child-image": {"label": "child-image", "parent": "parent-image", "transports": ["sircl"], "short_id": None},
              "unrelated-image": {"label": "unrelated-image", "parent": None, "transports": ["prepared"], "short_id": None}}
    return {"images": images, "checks": list(checks), "enhancements": list(entries), "not_adopted": []}


# -- the repository's catalog -----------------------------------------------------------------------


def test_repository_catalog_is_valid_and_its_profile_lists_match_the_configurations(catalog):
    assert ce.validate(catalog) == []


def test_generated_view_is_fresh():
    assert generate_enhancements.generate(check=True) == len(ce.load_catalog()["enhancements"])


def test_view_has_one_section_per_model_and_one_reference_per_entry(catalog):
    text = generate_enhancements.render(catalog)
    for name in ("GLM-5.3", "GLM-5.3-Flash", "Qwen3.8-Flash-Next", "DeepSeek-V4.1-Flash"):
        assert f"\n## {name}\n" in text
    for item in catalog["enhancements"]:
        assert text.count(f"\n### `{item['id']}`\n") == 1


def test_catalog_holds_no_private_path_or_address(catalog):
    assert not [text for text in ce._strings(catalog) if ce.PRIVATE_SHAPES.search(text)]


# -- conditions -------------------------------------------------------------------------------------


def test_arguments_parse_values_bare_flags_and_equals():
    assert ce.parse_arguments(["serve", "--tensor-parallel-size", "8", "--async-scheduling",
                               "--kv-cache-dtype=fp8", "--enable-prefix-caching"]) == {
        "--tensor-parallel-size": "8", "--async-scheduling": True, "--kv-cache-dtype": "fp8",
        "--enable-prefix-caching": True}


def test_unset_defaults_apply_by_source():
    condition = {"env": "SIRCL_LINK_SLOTS", "equals": "16", "installer_unset_means": "16"}
    assert ce.holds(condition, view(source="profile"))
    assert not ce.holds(condition, view(source="plan"))
    package_default = {"env": "SIRCL_ONESHOT_MAX_BYTES", "equals": "28672", "unset_means": "28672"}
    assert ce.holds(package_default, view(source="dump"))
    assert not ce.holds(package_default, view(environment={"SIRCL_ONESHOT_MAX_BYTES": "65536"}))


def test_json_argument_paths_and_list_operators():
    arguments = {"--quantization-config": json.dumps({"linear": "mxfp8", "ignore": ["*kv_b_proj", "*.x"]}),
                 "--speculative-config": '{"method": "mtp", "enable_adaptive_verification": true}'}
    deployment = view(arguments=arguments, environment={"SPARKRING_FEATURES": "qwen-collectives,qwen4-prefill"})
    assert ce.holds({"arg": "--quantization-config", "json": "linear", "equals": "mxfp8"}, deployment)
    assert ce.holds({"arg": "--quantization-config", "json": "ignore", "lacks_item": "*.indexer.*"}, deployment)
    assert not ce.holds({"arg": "--quantization-config", "json": "ignore", "lacks_item": "*.x"}, deployment)
    assert ce.holds({"arg": "--speculative-config", "json": "enable_adaptive_verification", "equals": True},
                    deployment)
    assert ce.holds({"env": "SPARKRING_FEATURES", "has_item": "qwen4-prefill"}, deployment)
    assert not ce.holds({"arg": "--hf-overrides", "json": "a.b", "has_item": "x"}, deployment)
    assert ce.holds({"env": "N", "at_least": 589824}, view(environment={"N": "589824"}))
    assert not ce.holds({"env": "N", "at_least": 589824}, view(environment={"N": "524288"}))


# -- evaluation -------------------------------------------------------------------------------------


def test_statuses_order_and_notes():
    catalog = synthetic(
        entry("on", provided_by={"images": ["parent-image"]}),
        entry("small", gain={"percent": 2, "summary": "s"}),
        entry("large", gain={"percent": 9, "summary": "l"}, requires=["on"]),
        entry("unmeasured"),
        entry("replaced", gain={"percent": 50, "summary": "r"}, superseded_by="on"),
        entry("refused", parallelism={"supported": [{"tp": [2], "dcp": [1]}],
                                      "refused": [{"tp": [8], "dcp": "any", "by": "vLLM", "text": "no TP8"}]}),
        entry("elsewhere", provided_by={"images": ["unrelated-image"]}, gain={"percent": 30, "summary": "e"}),
        entry("unported", provided_by={"images": [], "port": "write it"}, gain={"percent": 4, "summary": "u"}),
        entry("other-size", parallelism={"supported": [{"tp": [4], "dcp": "any"}]}),
        entry("other-model", models=[{"name": "Qwen3.8-Flash-Next"}]),
    )
    report = ce.evaluate(catalog, view(environment={"ON": "1", "REFUSED": "1"}))
    assert [(item["id"], item["status"]) for item in report["entries"]] == [
        ("large", "missing"), ("small", "missing"), ("unmeasured", "missing"), ("replaced", "missing"),
        ("elsewhere", "needs-port"), ("unported", "needs-port"), ("refused", "refused-here"), ("on", "enabled")]
    assert report["other_sizes"] == ["other-size"]
    assert "vLLM: no TP8" in row(report, "refused")["notes"]
    assert any("startup fails" in note for note in row(report, "refused")["notes"])
    assert "carried by unrelated-image" in row(report, "elsewhere")["notes"]
    assert "write it" in row(report, "unported")["notes"]
    assert any("takes its place" in note for note in row(report, "replaced")["notes"])
    assert report["warnings"] == ["refused is enabled but refused at TP8/DCP4"]


def test_image_ancestry_entry_without_switch_and_host_setting():
    catalog = synthetic(entry("native", detect=[]), entry("host", detect=None, provided_by={"host": True}))
    assert row(ce.evaluate(catalog, view(image="child-image")), "native")["status"] == "enabled"
    assert row(ce.evaluate(catalog, view(image="unrelated-image")), "native")["status"] == "needs-port"
    assert row(ce.evaluate(catalog, view()), "host")["status"] == "missing"
    assert row(ce.evaluate(catalog, view(enabled=frozenset({"host"}))), "host")["status"] == "enabled"


def test_checkpoint_specific_model_record_and_gain_by_tp():
    item = entry("ckpt", models=[
        {"name": "GLM-5.3", "checkpoints": ["example/model@aaaa"], "gain": {"percent": 7, "summary": "c",
                                                                           "by_tp": {"2": 11}}},
        {"name": "GLM-5.3", "gain": {"percent": 1, "summary": "g"}}], provided_by={"images": ["parent-image"]})
    catalog = synthetic(item)
    assert row(ce.evaluate(catalog, view()), "ckpt")["gain_percent"] == 7
    assert row(ce.evaluate(catalog, view(tp=2)), "ckpt")["gain_percent"] == 11
    assert row(ce.evaluate(catalog, view(model_revision="b" * 40)), "ckpt")["gain_percent"] == 1


def test_check_warns_on_quantized_linears_without_a_linear_backend(catalog):
    quantized = view(image=LIBSIRCL_IMAGE, arguments={"--quantization-config": '{"linear": "mxfp8"}'})
    assert any(w.startswith("linear-backend-explicit") for w in ce.evaluate(catalog, quantized)["warnings"])
    explicit = view(image=LIBSIRCL_IMAGE, arguments={"--quantization-config": '{"linear": "mxfp8"}',
                                                     "--linear-backend": "b12x"})
    assert not any(w.startswith("linear-backend-explicit") for w in ce.evaluate(catalog, explicit)["warnings"])
    bf16_dense = view(image=LIBSIRCL_IMAGE, arguments={"--quantization": "modelopt_fp4"})
    assert not ce.evaluate(catalog, bf16_dense)["warnings"]


# -- deployments from the repository's profiles ----------------------------------------------------


def test_glm53_tp8_profile_on_its_release_image_and_on_the_libsircl_image(catalog):
    released = ce.evaluate(catalog, ce.deployment("glm53-nvfp4-tp8", catalog=catalog))
    assert (released["tp"], released["dcp"], released["transport"]) == (8, 4, "sircl")
    assert released["image"] == "dev-20261004-kraken-cuda1342-nccl2323-status034"
    assert row(released, "glm53-ckv-gather")["status"] == "needs-port"
    assert row(released, "glm53-dsa-b12x-attention-fix")["status"] == "needs-port"
    built = ce.evaluate(catalog, ce.deployment("glm53-nvfp4-tp8", catalog=catalog, image=LIBSIRCL_IMAGE))
    # The profile turns the CKV gather on; the image's key gather for DSA prefill stays off.
    assert row(built, "glm53-ckv-gather")["status"] == "enabled"
    assert row(built, "glm53-dcp-indexer-key-gather")["status"] == "missing"
    assert row(built, "glm53-dsa-b12x-attention-fix")["status"] == "enabled"
    assert row(built, "sircl-cycle8-tuning-row")["status"] == "enabled"
    assert row(built, "glm53-indexer-prefill-split")["status"] == "needs-port"
    missing = [item["gain_percent"] for item in built["entries"] if item["status"] == "missing"]
    measured = [percent for percent in missing if percent is not None]
    assert measured == sorted(measured, reverse=True)
    assert "glm53-flash-mhc-prefill-shard" not in {item["id"] for item in built["entries"]}


def test_tp8_refusals_of_two_and_four_rank_prefill_features(catalog):
    csf = ce.evaluate(catalog, ce.deployment("glm53-flash-csf-tp8", catalog=catalog, image=LIBSIRCL_IMAGE))
    assert row(csf, "glm53-flash-kda-prefill-coalescing")["status"] == "refused-here"
    assert "glm53-flash-kda-prefill-coalescing is enabled but refused at TP8/DCP1" in csf["warnings"]
    assert row(csf, "glm53-flash-csf-checkpoint")["status"] == "enabled"
    qwen = ce.evaluate(catalog, ce.deployment("qwen38-flash-next-qad-tp8", catalog=catalog))
    assert row(qwen, "qwen38-hc-prefill-row-ownership")["status"] == "refused-here"
    assert "qwen38-collectives-feature" not in {item["id"] for item in qwen["entries"]}


def test_checkpoint_selection_changes_the_model(catalog):
    stock = ce.evaluate(catalog, ce.deployment("qwen38-flash-next-tp2", catalog=catalog))
    derived = ce.evaluate(catalog, ce.deployment("qwen38-flash-next-tp2", catalog=catalog,
                                                 checkpoint="qad-step5500-mxfp8-attention"))
    assert row(stock, "qwen38-mxfp8-attention-checkpoint")["status"] == "missing"
    assert row(derived, "qwen38-mxfp8-attention-checkpoint")["status"] == "enabled"
    assert derived["model"]["name"] == "Qwen3.8-Flash-Next"


def test_sircl_session_settings_in_a_profile_do_not_select_the_transport():
    settings = {"SIRCL_LARGE_SCHEDULE": "ring", "SIRCL_LINK_SLOT_BYTES": "1048576"}
    sircl_image = {"transports": ["libsircl", "prepared", "sircl"]}
    assert ce.infer_transport(settings, "direct-cycle-4", None)[0] == "prepared"
    assert ce.infer_transport(settings, "direct-cycle-4", sircl_image)[0] == "sircl"
    # What SIRCL's adapter sets on every container it runs does.
    assert ce.infer_transport(dict(settings, SIRCL_MODE="custom"), "direct-cycle-4", None)[0] == "sircl"
    assert ce.infer_transport({"VLLM_PLUGINS": "b12x_loader,sircl"}, "direct-pair-2", None)[0] == "sircl"


def test_plan_input_reads_the_arm_command_and_the_image(tmp_path, catalog):
    _, config = ce.profile_configuration("glm53-nvfp4-tp8")
    environment = dict(config["environment"], SIRCL_FUSED_NORM="1", VLLM_B12X_MLA_CKV_GATHER="1",
                       VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS="589824")
    tokens = ["docker", "create"] + [part for key, value in environment.items() for part in ("--env", f"{key}={value}")]
    # The arm's command without the profile's --linear-backend, so the catalog's check warns.
    arguments = list(config["vllm_args"])
    del arguments[arguments.index("--linear-backend"):arguments.index("--linear-backend") + 2]
    tokens += [LIBSIRCL_IMAGE_ID, "serve", "/models/target"] + arguments + [
        "--quantization-config", '{"linear":"mxfp8","ignore":["*kv_b_proj"]}']
    plan = {"schema": "serving-ab-plan/v1", "profile": "glm53-nvfp4-tp8", "model": config["model"],
            "image": LIBSIRCL_IMAGE_ID, "arms": ["S+"], "commands": {"S+": [tokens]}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    report = ce.evaluate(catalog, ce.deployment("glm53-nvfp4-tp8", catalog=catalog, plan=path))
    assert (report["image"], report["transport"], report["source"]) == (LIBSIRCL_IMAGE, "sircl", "plan")
    assert row(report, "sircl-fused-allreduce-rmsnorm")["status"] == "enabled"
    assert row(report, "glm53-ckv-gather")["status"] == "enabled"
    assert row(report, "sircl-cycle8-tuning-row")["status"] == "missing"
    assert any(w.startswith("linear-backend-explicit") for w in report["warnings"])


def test_container_dumps_replace_the_profile_settings(tmp_path, catalog):
    environment = tmp_path / "env.txt"
    environment.write_text("PATH=/usr/bin\nVLLM_GLM53_MHC_PREFILL_SHARD=1\nVLLM_PLUGINS=b12x_loader\n",
                           encoding="utf-8")
    arguments = tmp_path / "args.json"
    arguments.write_text(json.dumps(["--tensor-parallel-size", "4", "--decode-context-parallel-size", "2",
                                     "--async-scheduling"]), encoding="utf-8")
    deployment = ce.deployment("glm53-flash-nvfp4-spark-tp4", catalog=catalog, environment_dump=environment,
                               arguments_dump=arguments)
    assert (deployment.tp, deployment.dcp, deployment.source) == (4, 2, "dump")
    report = ce.evaluate(catalog, deployment)
    assert row(report, "glm53-flash-mhc-prefill-shard")["status"] == "enabled"
    assert row(report, "glm53-flash-kda-prefill-coalescing")["status"] == "missing"
    assert ce.read_environment_dump(_write(tmp_path / "e.json", '["A=1", "B=x=y"]')) == {"A": "1", "B": "x=y"}
    assert ce.read_arguments_dump(_write(tmp_path / "a.txt", "vllm serve --tp 8 \\\n  --dcp 4")) == [
        "vllm", "serve", "--tp", "8", "--dcp", "4"]


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


# -- validation and command line ------------------------------------------------------------------


def test_validation_rejects_private_data_unknown_names_and_drift(catalog):
    broken = copy.deepcopy(catalog)
    first = broken["enhancements"][0]
    first["description"] += " see C:" + "\\Users\\someone\\notes"
    first["models"].append({"name": "Unknown-Model"})
    first["evidence"].append("notes/missing.md")
    problems = ce.validate(broken, profiles=False)
    assert any("private path" in problem for problem in problems)
    assert any("Unknown-Model" in problem for problem in problems)
    assert any("notes/missing.md" in problem for problem in problems)
    drifted = copy.deepcopy(catalog)
    target = next(item for item in drifted["enhancements"] if item["id"] == "async-scheduling")
    target["profiles"] = target["profiles"][1:]
    assert any(problem.startswith("enhancement async-scheduling: profiles") for problem in ce.validate(drifted))


def test_command_line(capsys):
    assert ce.main(["glm53-nvfp4-tp8", "--image", LIBSIRCL_IMAGE, "--enabled", "host-sm-clock-lock",
                    "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert row(report, "host-sm-clock-lock")["status"] == "enabled"
    assert ce.main(["no-such-profile"]) == 2
    assert ce.main(["glm53-nvfp4-tp8", "--enabled", "no-such-entry"]) == 2
    assert ce.main(["--validate"]) == 0


def test_glm53_tp8_profile_on_the_plugin_layer_image_enables_the_plugin_set(catalog):
    report = ce.evaluate(catalog, ce.deployment("glm53-nvfp4-tp8", catalog=catalog, image="glm53-plugins-af06e272"))
    for entry in ("glm53-indexer-prefill-split", "glm53-latent-shard", "glm53-mtp-eh-proj-tp", "b12x-linear-backend"):
        assert row(report, entry)["status"] == "enabled", entry
    assert row(report, "b12x-linear-backend")["gain_percent"] == 61
    # The plugins' parent image does not carry them.
    parent = ce.evaluate(catalog, ce.deployment("glm53-nvfp4-tp8", catalog=catalog, image=LIBSIRCL_IMAGE))
    assert row(parent, "glm53-latent-shard")["status"] == "needs-port"
