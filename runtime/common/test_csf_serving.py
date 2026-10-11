"""GLM-5.3 NVFP4-CSF on eight Sparks: the profile, its serving directory, its recipe and its image gating.

The profile keeps every setting of glm53-nvfp4-tp8 and changes only what the
NVFP4-CSF container needs. Its default checkpoint is the serving directory, a
derived checkpoint without a donor whose base is the profile's own model
(runtime/common/csf_serving.py, runtime/common/derived_checkpoint.py).
"""
import copy
import hashlib
import json
import subprocess
import sys

import pytest

from runtime.common import csf_serving, derived_checkpoint, image_lock, installer, setup, thinking, toolchain_profiles
from runtime.common.test_image_lock import sircl_lock
from runtime.common.test_transport import eight_spark_image, sircl_deployment
from scripts import pin_csf_serving

PROFILE = "glm53-nvfp4-csf-tp8"
NATIVE = "glm53-nvfp4-tp8"
NAME = "csf-serving"
REPOSITORY = "local-inference-lab/GLM-5.3-NVFP4-CSF"
REVISION = "a4d406932f2db96db8e852867c7939f448e0cd6f"
SERVING = "sparkring-derived/GLM-5.3-NVFP4-CSF-Serving"
BUILD = "sparkring-kraken-beta-20261007-bc9ea774"
LINKS = {"amax_checkpoint.json", "chat_template.jinja", "generation_config.json", "hf_quant_config.json",
         "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"}


def configuration(profile=PROFILE):
    return toolchain_profiles.read(installer.ROOT / f"profiles/{profile}/config.json")


def option(arguments, flag):
    return arguments[arguments.index(flag) + 1] if flag in arguments else None


def test_the_profile_is_the_native_profile_with_only_the_csf_changes():
    csf, native = configuration(), configuration(NATIVE)
    assert csf["status"] == native["status"] == "research-only"
    assert set(csf) - set(native) == {"checkpoint", "checkpoints", "start_checks"} and not set(native) - set(csf)
    changed = {key for key in native if csf[key] != native[key]}
    assert changed == {"model", "served_model_name", "cache_namespace", "vllm_args"}
    assert csf["environment"] == native["environment"] and csf["smoke"] == native["smoke"]
    assert (csf["served_model_name"], csf["cache_namespace"]) == ("GLM-5.3-NVFP4-CSF-TP8", "glm53-nvfp4-csf")
    # The checkpoint is the same model: the container's source configuration and index are the native profile's.
    assert csf["model"]["repository"] == REPOSITORY and csf["model"]["revision"] == REVISION
    assert (csf["model"]["config_sha256"], csf["model"]["index_sha256"]) == (
        native["model"]["config_sha256"], native["model"]["index_sha256"])
    metadata, _ = installer.profiles.load(PROFILE)
    assert (metadata["status"], metadata["recommendation"]) == ("research-only", "alternative")
    assert PROFILE in installer.profiles.research_catalog()
    assert PROFILE not in {row["id"] for row in installer.read(installer.ROOT / "profiles/catalog.json")["profiles"]}
    assert PROFILE in installer.installer_image.SIRCL_ONLY and PROFILE not in installer.installer_image.SUPPORTED


def test_the_rendered_arguments_differ_from_the_native_profiles_only_by_the_csf_flags():
    """Every speed setting of glm53-nvfp4-tp8 stays: the only changed vLLM options are the format's."""
    csf, native = configuration(), configuration(NATIVE)
    served = toolchain_profiles.checkpoint_settings(csf, None)
    assert served["vllm_args"] == csf["vllm_args"] and served["model"]["repository"] == SERVING
    expected = list(native["vllm_args"])
    expected[expected.index("--quantization") + 1] = "nvfp4_csf"
    expected[expected.index("--load-format") + 1] = "nvfp4_csf"
    at = expected.index("--hf-overrides")
    del expected[at:at + 2]
    assert csf["vllm_args"] == expected
    for flag, value in (("--tensor-parallel-size", "8"), ("--decode-context-parallel-size", "4"),
                        ("--attention-backend", "B12X"), ("--kv-cache-dtype", "fp8_ds_mla"),
                        ("--moe-backend", "b12x"), ("--linear-backend", "b12x")):
        assert option(csf["vllm_args"], flag) == option(native["vllm_args"], flag) == value
    assert json.loads(option(csf["vllm_args"], "--speculative-config"))["num_speculative_tokens"] == 2
    assert option(csf["vllm_args"], "--quantization-config") == option(native["vllm_args"], "--quantization-config")
    # On SIRCL ring sessions of an eight-Spark ring the two profiles' rank commands differ by exactly these options,
    # the served name and the checkpoint-keyed cache paths.
    csf_lock, _ = sircl_deployment(PROFILE, "cycle", 8, list(range(8)), image=eight_spark_image())
    native_lock, _ = sircl_deployment(NATIVE, "cycle", 8, list(range(8)), image=eight_spark_image())
    for mine, theirs in zip(installer.specifications(csf_lock), installer.specifications(native_lock), strict=True):
        ours, other = list(mine.command), list(theirs.command)
        assert option(ours, "--served-model-name") == "GLM-5.3-NVFP4-CSF-TP8"
        for tokens, name in ((ours, "GLM-5.3-NVFP4-CSF-TP8"), (other, "GLM-5.3-NVFP4-TP8")):
            tokens.remove(name)
        at = other.index("--hf-overrides")
        del other[at:at + 2]
        assert [(a, b) for a, b in zip(ours, other, strict=True) if a != b] == [
            ("nvfp4_csf", "modelopt_fp4"), ("nvfp4_csf", "b12x")]
        assert ours[ours.index("nvfp4_csf") - 1] == "--quantization"
        changed = {key for key in set(mine.environment) | set(theirs.environment)
                   if mine.environment.get(key) != theirs.environment.get(key)}
        assert changed and all(key.endswith("_DIR") or key.endswith("_HOME") or key.endswith("_ROOT")
                               or key == "FLASHINFER_WORKSPACE_BASE" for key in changed), changed


def test_the_selection_acquires_the_container_and_serves_its_serving_directory():
    card = setup.selection(PROFILE)
    assert (card["target_variant"], card["model_repository"], card["model_revision"]) == (NAME, REPOSITORY, REVISION)
    pins = installer.checkpoint_pins(card)
    assert pins["index"] == "metadata/model.safetensors.index.json" and len(pins["weights"]) == 85
    assert all(name.startswith("tensors/") for name in pins["weights"])
    manifest = derived_checkpoint.load(card)
    model = configuration()["checkpoints"][NAME]["model"]
    assert (manifest["repository"], manifest["revision"]) == (model["repository"], model["revision"]) == (
        SERVING, derived_checkpoint.identity(manifest["base"], None, manifest["recipe"]))
    assert manifest["donor"] is None and manifest["base"] == {"repository": REPOSITORY, "revision": REVISION}
    assert manifest["recipe"]["path"] == pin_csf_serving.RECIPE
    assert hashlib.sha256((installer.ROOT / pin_csf_serving.RECIPE).read_bytes()).hexdigest() == manifest["recipe"]["sha256"]
    required = set(pins["files"]) - set(pins["optional"])
    files = manifest["files"]
    moved = {"metadata/" + name for name in LINKS}
    assert set(files) == (required - moved) | LINKS | {"derivation.json"} and len(files) == 189
    assert sorted(derived_checkpoint.files_of(manifest, "recipe")) == ["config.json", "derivation.json"]
    for name in LINKS:
        assert files[name]["source"] == "metadata/" + name
        assert files[name]["sha256"] == pins["files"]["metadata/" + name]["sha256"]
    assert files["config.json"]["sha256"] == model["config_sha256"]
    assert files["model.safetensors.index.json"]["sha256"] == model["index_sha256"] == pins["files"][pins["index"]]["sha256"]
    assert manifest["transform"] == {"checkpoint_root": "/models/target", "family": "glm53_744b_nvfp4",
                                     "layout": "lil-nvfp4-csf-checkpoint/1",
                                     "source_config": {"name": "metadata/config.json",
                                                       "sha256": pins["files"]["metadata/config.json"]["sha256"]}}
    # The committed manifest is the generator's output for its own config.json pins.
    config = {key: files["config.json"][key] for key in ("size", "sha256")}
    generated = pin_csf_serving.manifest(pins, SERVING, manifest["transform"], config)
    assert generated == manifest
    path = derived_checkpoint.manifest_path(model)
    assert path.read_bytes().decode("utf-8").replace("\r\n", "\n") == pin_csf_serving.encode(generated)
    assert pin_csf_serving.model(generated, pins) == model
    # The views of the selection: the base for acquisition and the serving directory for the containers.
    view = derived_checkpoint.view(card, manifest)
    assert installer.checkpoint_contract(view) == model and installer.checkpoint_contract(card) == configuration()["model"]
    assert derived_checkpoint.listed()[(SERVING, manifest["revision"])] == {
        "name": NAME, "profiles": [PROFILE], "base": {"repository": REPOSITORY, "revision": REVISION}, "donor": None}
    # The thinking record is the served directory's, whose chat template is the container's.
    assert thinking.of(PROFILE)["checkpoint"] == f"{SERVING}@{manifest['revision']}"
    assert files["chat_template.jinja"]["sha256"] in thinking.of(PROFILE)["source"]["chat_templates"]


def test_storage_planning_counts_the_container_and_the_recipe_files():
    card = setup.selection(PROFILE)
    pins = installer.checkpoint_pins(card)
    manifest = derived_checkpoint.load(card)
    from runtime.host import checkpoint_plan
    sizes = [entry["size"] for name, entry in pins["files"].items() if name not in pins["optional"]]
    recipe = list(derived_checkpoint.files_of(manifest, "recipe").values())
    assert setup.pinned_checkpoint_bytes(card) == checkpoint_plan.required_space(sizes + recipe)
    assert round(sum(sizes) / 2 ** 30, 1) == 412.5 and sum(recipe) < 16 << 10
    from runtime.host import storage
    references = storage.profile_references()
    assert PROFILE in references["checkpoints"][f"{REPOSITORY}@{REVISION}"]
    assert PROFILE in references["checkpoints"][f"{SERVING}@{manifest['revision']}"]
    assert references["derived"][f"{SERVING}@{manifest['revision']}"]["donor"] is None
    # Profiles that only SIRCL ring sessions run reference their checkpoints too.
    assert NATIVE in references["checkpoints"]["local-inference-lab/GLM-5.3-NVFP4@b472e4ee53f6a9862da5486c56c6ca21be3dab70"]


def test_rendered_containers_mount_the_serving_directory_where_its_config_names_the_checkpoint_root():
    lock, _ = sircl_deployment(PROFILE, "cycle", 8, list(range(8)), image=eight_spark_image())
    manifest = derived_checkpoint.load(lock["selection"])
    base = installer.checkpoint_directory(lock["site"]["name"], lock["selection"])
    serving = derived_checkpoint.directory(base, manifest)
    assert serving == base.rsplit("/", 2)[0] + f"/sparkring-derived--GLM-5.3-NVFP4-CSF-Serving/{manifest['revision']}"
    for spec in installer.specifications(lock):
        mounts = {mount.target: mount for mount in spec.mounts}
        assert (mounts[csf_serving.CHECKPOINT_ROOT].source, mounts[csf_serving.CHECKPOINT_ROOT].read_only) == (serving, True)
        assert spec.command[spec.command.index("serve") + 1] == csf_serving.CHECKPOINT_ROOT
    phases = [phase["id"] for phase in installer.operation_plan(lock, "up")["phases"]]
    assert phases[:5] == ["prerequisites", "source", "model", "image", "derive"]


def test_the_checkpoint_needs_the_vllm_build_that_reads_csf():
    card = setup.selection(PROFILE)
    assert image_lock.CHECKPOINT_BUILDS[f"{REPOSITORY}@{REVISION}"] == (BUILD,)
    reads = sircl_lock(profiles=sorted({*installer.installer_image.default_lock()["profiles"], PROFILE}))
    reads["sircl"]["vllm_pins"] = [BUILD]
    assert image_lock.checkpoint_problem(reads, card) is None
    other = copy.deepcopy(reads)
    other["sircl"]["vllm_pins"] = ["lil-image-aba309e4610c"]
    problem = image_lock.checkpoint_problem(other, card)
    assert problem.startswith(f"Checkpoint {REPOSITORY} at {REVISION[:12]} (--checkpoint {NAME}) needs an image whose "
                              f"vLLM is the pinned build {BUILD}")
    # The plan states the checkpoint's research status; no preferred checkpoint is involved.
    notice = image_lock.checkpoint_notice(reads, PROFILE, card, preferred=False)
    assert notice.startswith(f"Checkpoint {NAME} ({REPOSITORY} at {REVISION[:12]}) is research-only")
    assert toolchain_profiles.preferred_checkpoint(configuration()) is None
    # Release 2026.10.2's lock admits the profile, and its image reads the checkpoint.
    release = installer.read(installer.ROOT / "runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-"
                                              "plugins-status036/installer-image.json")
    assert PROFILE in release["profiles"] and image_lock.checkpoint_problem(release, card) is None


def test_a_default_derived_entry_must_hold_only_its_model_and_derive_from_the_profiles_model():
    profile = configuration()
    assert toolchain_profiles.checkpoint_names(profile) == (NAME, (NAME,))
    for change in ({"derived": {"base": NAME}}, {"arguments": {"--max-num-seqs": "8"}},
                   {"served_model_name": "Other-TP8"}):
        changed = copy.deepcopy(profile)
        changed["checkpoints"][NAME].update(change)
        with pytest.raises(ValueError):
            toolchain_profiles.checkpoint_names(changed)
    changed = copy.deepcopy(profile)
    changed["model"] = dict(changed["checkpoints"][NAME]["model"])
    with pytest.raises(ValueError, match="published checkpoint"):
        toolchain_profiles.checkpoint_names(changed)


# The recipe --------------------------------------------------------------------------

SOURCE = (b'{\n  "architectures": ["GlmMoeDsaForCausalLM"],\n  "model_type": "glm_moe_dsa",\n'
          b'  "quantization_config": {"quant_algo": "NVFP4", "ignore": ["lm_head"]},\n  "vocab_size": 7\n}\n')


def container(path, source=SOURCE, *, family="glm53_744b_nvfp4", recorded=None):
    """A container root holding manifest.json and metadata/config.json; returns the record for it."""
    (path / "metadata").mkdir(parents=True)
    (path / "metadata/config.json").write_bytes(source)
    digest = hashlib.sha256(SOURCE).hexdigest()
    (path / "manifest.json").write_text(json.dumps({"schema": csf_serving.LAYOUT, "family": family,
                                                    "metadata_sha256": {"config.json": recorded or digest}}))
    return {"schema": derived_checkpoint.RECORD_SCHEMA, "checkpoint_root": csf_serving.CHECKPOINT_ROOT,
            "family": "glm53_744b_nvfp4", "layout": csf_serving.LAYOUT,
            "source_config": {"name": "metadata/config.json", "sha256": digest}}


def test_the_recipe_wraps_the_source_quantization_config_and_keeps_every_other_key(tmp_path):
    record = container(tmp_path / "base")
    out = tmp_path / "out"
    out.mkdir()
    sizes = csf_serving.derive(str(tmp_path / "base"), str(out), record, log=lambda text: None)
    written = json.loads((out / "config.json").read_text())
    source = json.loads(SOURCE)
    assert list(written) == list(source)
    assert {key: value for key, value in written.items() if key != "quantization_config"} == {
        key: value for key, value in source.items() if key != "quantization_config"}
    assert written["quantization_config"] == {"quant_method": "nvfp4_csf", "format_version": 1,
                                              "checkpoint_root": "/models/target",
                                              "source_quantization_config": source["quantization_config"]}
    assert (out / "derivation.json").read_bytes() == derived_checkpoint.record_bytes(record)
    assert sizes == {"config.json": (out / "config.json").stat().st_size,
                     "derivation.json": (out / "derivation.json").stat().st_size}
    assert (out / "config.json").read_bytes() == csf_serving.serving_config(SOURCE)
    # The same bytes from the installer's invocation: the file's text run with python3 -c and the record as JSON.
    again = tmp_path / "again"
    again.mkdir()
    code = (installer.ROOT / pin_csf_serving.RECIPE).read_text(encoding="utf-8")
    subprocess.run([sys.executable, "-I", "-c", code, "--base", str(tmp_path / "base"), "--out", str(again),
                    "--record", json.dumps(record, sort_keys=True)], check=True, capture_output=True)
    assert {path.name: path.read_bytes() for path in again.iterdir()} == {
        path.name: path.read_bytes() for path in out.iterdir()}


@pytest.mark.parametrize("change, message", [
    ({"source": SOURCE.replace(b'"vocab_size": 7', b'"vocab_size": 8')}, "has SHA-256"),
    ({"recorded": "0" * 64}, "has SHA-256"),
    ({"family": "glm53_nvfp4"}, "names layout"),
])
def test_the_recipe_refuses_a_container_that_differs_and_writes_nothing(tmp_path, change, message):
    record = container(tmp_path / "base", **change)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(csf_serving.RecipeError, match=message):
        csf_serving.derive(str(tmp_path / "base"), str(out), record, log=lambda text: None)
    assert list(out.iterdir()) == []
    code = (installer.ROOT / pin_csf_serving.RECIPE).read_text(encoding="utf-8")
    done = subprocess.run([sys.executable, "-I", "-c", code, "--base", str(tmp_path / "base"), "--out", str(out),
                           "--record", json.dumps(record)], capture_output=True, text=True)
    assert done.returncode == 1 and done.stderr.startswith("csf_serving: ") and list(out.iterdir()) == []


def test_the_recipe_refuses_a_serving_configuration_and_a_layout_that_would_shadow_a_file():
    with pytest.raises(csf_serving.RecipeError, match="already a serving configuration"):
        csf_serving.serving_config(csf_serving.serving_config(SOURCE))
    with pytest.raises(csf_serving.RecipeError, match="no quantization_config"):
        csf_serving.serving_config(b'{"model_type": "glm_moe_dsa"}')
    required = ["config.json", "metadata/config.json", "metadata/model.safetensors.index.json", "metadata/tokenizer.json"]
    assert csf_serving.layout(required) == {"model.safetensors.index.json": "metadata/model.safetensors.index.json",
                                            "tokenizer.json": "metadata/tokenizer.json"}
    with pytest.raises(csf_serving.RecipeError, match="cannot link"):
        csf_serving.layout([*required, "tokenizer.json"])
    with pytest.raises(csf_serving.RecipeError, match="lacks config.json"):
        csf_serving.layout(["metadata/tokenizer.json"])


def test_a_serving_manifest_whose_config_or_links_differ_from_the_pins_is_refused():
    card = setup.selection(PROFILE)
    manifest = derived_checkpoint.load(card)
    model = configuration()["checkpoints"][NAME]["model"]
    base = installer.checkpoint_pins(card)
    for path, value, message in (
            (("files", "tokenizer.json", "sha256"), "0" * 64, "tokenizer.json differs from the base's pins"),
            (("files", "tokenizer.json", "source"), "metadata/tokenizer_config.json", "tokenizer.json differs"),
            (("files", "config.json", "sha256"), "0" * 64, "config.json or the index differs"),
            (("donor",), {"repository": REPOSITORY, "revision": REVISION, "files": []}, "donor must be null"),
            (("transform", "checkpoint_root"), "/srv/other", "derivation.json differs from the record")):
        changed = copy.deepcopy(manifest)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            derived_checkpoint.check(changed, model, base, None)
