"""Image switches bind receipts and preserve model, mesh and lifecycle ownership."""
import copy
import dataclasses
import hashlib
import json
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from runtime.common import installer, installer_image, profiles, setup
from runtime.common.test_installer import site
from runtime.host import controller, models

PROFILE = "qwen38-flash-next-qad-tp4"


def image_lock(profile=PROFILE):
    return {"schema": "sparkring-installer-image/v1", "name": "qwen-cuda1342-status03",
            "profile": profile, "image_id": "sha256:" + "a" * 64, "image_reference": "sha256:" + "a" * 64,
            "parent_receipt_sha256": "b" * 64, "toolchain_receipt_sha256": "c" * 64,
            "composition_sha256": "d" * 64, "transport_profile": "tp2-rocenante-adaptive-prepared",
            "transport_manifest_sha256": "e" * 64, "status_version": "0.3.0"}


def ring_site():
    value = site(4)
    for row in value["hosts"]:
        row["fabric"] = {"site_path": "/etc/sparkring/site.json", "site_sha256": "1" * 64, "plan_sha256": "2" * 64}
    return value


def test_image_selection_preserves_profile_weights_network_and_model_arguments(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Offline render contacted a host"))
    original = setup.selection(PROFILE)
    baseline = installer.make_lock(PROFILE, ring_site(), "1" * 40, "2" * 64)
    selected = installer.make_lock(PROFILE, ring_site(), "1" * 40, "2" * 64, image_runtime=image_lock())
    assert installer.validate(selected) == selected
    assert setup.selection(PROFILE) == original
    assert selected["selection"]["profile_release"] == original["release"]
    assert selected["site"] == baseline["site"] and selected["id"] != baseline["id"]
    for before, after in zip(installer.specifications(baseline), installer.specifications(selected), strict=True):
        assert after.command == before.command[1:]
        assert after.entrypoint == installer_image.ENTRYPOINT
        assert after.mounts[:-1] == before.mounts and after.devices == before.devices
        assert after.mounts[-1].target == installer_image.BINDING_TARGET and after.mounts[-1].read_only
        assert after.environment["SPARKRING_RUNTIME_BINDING"] == installer_image.BINDING_TARGET
        assert after.environment["B12X_ROCE_HCA"] == before.environment["B12X_ROCE_HCA"]
        assert after.environment["B12X_ROCE_PEER_HCA_MAP"] == before.environment["B12X_ROCE_PEER_HCA_MAP"]
        assert after.environment["VLLM_PLUGINS"] == "b12x_loader,sparkring_status"
        assert after.environment["VLLM_QWEN3_8_FLASH_NEXT_HC_TP"] == "0"
        assert after.environment["VLLM_QWEN3_8_HC_PREFILL_MODE"] == "shard"
        assert after.environment[installer_image.TOOL_CHOICE_CONTRACT] == "1"
        assert "PYTHONPATH" not in after.environment and "LD_PRELOAD" not in after.environment
        # Image-scoped caches follow the selected image. Compiled B12X kernels
        # carry their own content key and are shared across installer images of
        # one CUDA toolkit, so no B12X-keyed variable names an image.
        assert "a" * 12 in after.environment["XDG_CACHE_HOME"]
        assert before.image_id[7:19] not in after.environment["XDG_CACHE_HOME"]
        assert after.environment["B12X_COMPILE_CACHE_DIR"] == before.environment["B12X_COMPILE_CACHE_DIR"]
        assert "cuda" + installer_image.CUDA_VERSION in after.environment["B12X_COMPILE_CACHE_DIR"]
        keyed = {key: setting for key, setting in after.environment.items()
                 if key.startswith(("B12X_", "CUTE_", "CUTLASS_")) and key != "CUTE_DSL_CACHE_DIR"}
        assert "B12X_ROCE_CACHE_DIR" not in keyed
        assert not any(image[7:19] in setting for setting in keyed.values()
                       for image in (before.image_id, after.image_id))
    rendered = installer.rendered(selected)
    assert len(rendered) == 8
    assert "toolchain/toolchain.py" in rendered["rank0/compose.yaml"]
    assert "python3" in rendered["rank0/compose.yaml"]
    assert "Development" in selected["selection"]["evidence_scope"]


@pytest.mark.parametrize("setting", ["0", "1"])
def test_profile_environment_selects_the_tool_choice_policy(setting):
    baseline = installer.make_lock(PROFILE, ring_site(), "1" * 40, "2" * 64)
    spec = installer.specifications(baseline)[0]
    assert installer_image.TOOL_CHOICE_CONTRACT not in spec.environment
    chosen = dataclasses.replace(spec, environment={**spec.environment, installer_image.TOOL_CHOICE_CONTRACT: setting})
    adapted = installer_image.adapt(chosen, image_lock(), binding="/run/binding.json", source_root="/opt/source")
    assert adapted.environment[installer_image.TOOL_CHOICE_CONTRACT] == setting


@pytest.mark.parametrize("change", ["tag", "digest", "profile", "unknown", "version"])
def test_ambiguous_or_unsupported_image_inputs_are_rejected(change):
    value = image_lock()
    if change == "tag":
        value["image_reference"] = "example/image:latest"
    elif change == "digest":
        value["toolchain_receipt_sha256"] = "abc"
    elif change == "profile":
        value["profile"] = "qwen38-flash-next-qad-tp4-sparkcache"
    elif change == "unknown":
        value["environment"] = {"PYTHONPATH": "/unverified"}
    else:
        value["status_version"] = "future"
    with pytest.raises(ValueError):
        installer.make_lock(PROFILE, ring_site(), "1" * 40, "2" * 64, image_runtime=value)


def test_replaced_image_or_receipt_invalidates_saved_deployment():
    lock = installer.make_lock(PROFILE, ring_site(), "1" * 40, "2" * 64, image_runtime=image_lock())
    for field in ("image_id", "parent_receipt_sha256", "toolchain_receipt_sha256"):
        changed = copy.deepcopy(lock)
        changed["image_runtime"][field] = ("sha256:" if field == "image_id" else "") + "f" * 64
        with pytest.raises(ValueError):
            installer.validate(changed)


def admission_fixture():
    value = image_lock()
    parent = {"schema": "sparkring-external-installed/v1", "composition_sha256": value["composition_sha256"],
              "capabilities": {"transport_profile": value["transport_profile"],
                               "transport_manifest_sha256": value["transport_manifest_sha256"],
                               "runtime_status": {"version": "0.3.0"}, "features": ["qwen-collectives", "qwen4-prefill"],
                               "hc_supported_modes": {"4": [{"projection_tp": "0", "prefill_row_ownership": "shard"}]}}}
    raw_parent = json.dumps(parent).encode()
    value["parent_receipt_sha256"] = hashlib.sha256(raw_parent).hexdigest()
    toolchain = {"schema": "sparkring-toolchain-installed/v1", "variant": "combined",
                 "parent_receipt_sha256": value["parent_receipt_sha256"], "nccl_version": 23203, "nvcc": "V13.4.92"}
    raw_toolchain = json.dumps(toolchain).encode()
    value["toolchain_receipt_sha256"] = hashlib.sha256(raw_toolchain).hexdigest()
    image = {"Id": value["image_id"], "Os": "linux", "Architecture": "arm64", "Config": {"Entrypoint": list(installer_image.ENTRYPOINT)}}
    return value, image, {installer_image.PARENT_RECEIPT: raw_parent, installer_image.TOOLCHAIN_RECEIPT: raw_toolchain}


@pytest.mark.parametrize("change", [None, "architecture", "entrypoint", "parent", "toolchain", "installed-tree"])
def test_admission_verifies_receipts_and_installed_tree_without_gpu_or_network(change):
    value, image, receipts = admission_fixture()
    if change == "architecture":
        image["Architecture"] = "amd64"
    elif change == "entrypoint":
        image["Config"]["Entrypoint"] = ["python3", "/unverified.py"]
    elif change in ("parent", "toolchain"):
        path = installer_image.PARENT_RECEIPT if change == "parent" else installer_image.TOOLCHAIN_RECEIPT
        receipts[path] += b" "
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "image":
            return SimpleNamespace(stdout=json.dumps([image]))
        assert "--gpus" not in argv
        assert argv[argv.index("--runtime") + 1] == "runc"
        assert argv[argv.index("--network") + 1] == "none" and "--read-only" in argv
        if argv[-1] in receipts:
            assert kwargs["text"] is False
            return SimpleNamespace(stdout=receipts[argv[-1]])
        assert argv[-2:] == [installer_image.ENTRYPOINT[1], "verify"]
        if change == "installed-tree":
            raise subprocess.CalledProcessError(1, argv)
        return SimpleNamespace(stdout="verified")
    if change:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            installer_image.admit(value, run=run)
    else:
        result = installer_image.admit(value, run=run)
        assert result["image_id"] == value["image_id"] and result["serving_qualified"] is False
        assert len(calls) == 4


def test_qwen_recipe_follows_the_profile_environment():
    assert installer_image.qwen_recipe({}) == ({"projection_tp": "1", "prefill_row_ownership": "off"}, set())
    shard = {"VLLM_QWEN3_8_HC_PREFILL_MODE": "shard", "SPARKRING_FEATURES": "qwen-collectives, qwen4-prefill"}
    assert installer_image.qwen_recipe(shard) == ({"projection_tp": "0", "prefill_row_ownership": "shard"},
                                                  {"qwen-collectives", "qwen4-prefill"})


@pytest.mark.parametrize("supported", [False, True])
def test_tp2_row_sharding_requires_an_image_that_declares_it(supported):
    value, image, receipts = admission_fixture()
    value["profile"] = "qwen38-flash-next-tp2"
    parent = json.loads(receipts[installer_image.PARENT_RECEIPT])
    modes = [{"projection_tp": "1", "prefill_row_ownership": "off"}]
    if supported:
        modes.append({"projection_tp": "0", "prefill_row_ownership": "shard"})
    parent["capabilities"]["hc_supported_modes"]["2"] = modes
    receipts[installer_image.PARENT_RECEIPT] = json.dumps(parent).encode()
    value["parent_receipt_sha256"] = hashlib.sha256(receipts[installer_image.PARENT_RECEIPT]).hexdigest()
    toolchain = json.loads(receipts[installer_image.TOOLCHAIN_RECEIPT])
    toolchain["parent_receipt_sha256"] = value["parent_receipt_sha256"]
    receipts[installer_image.TOOLCHAIN_RECEIPT] = json.dumps(toolchain).encode()
    value["toolchain_receipt_sha256"] = hashlib.sha256(receipts[installer_image.TOOLCHAIN_RECEIPT]).hexdigest()
    def run(argv, **kwargs):
        if argv[1] == "image":
            return SimpleNamespace(stdout=json.dumps([image]))
        if argv[-1] in receipts:
            return SimpleNamespace(stdout=receipts[argv[-1]])
        return SimpleNamespace(stdout="verified")
    environment = {"VLLM_QWEN3_8_HC_PREFILL_MODE": "shard", "SPARKRING_FEATURES": "qwen-collectives,qwen4-prefill"}
    def admit():
        return installer_image.admit(value, run=run, profile="qwen38-flash-next-tp2", nodes=2, environment=environment)
    if supported:
        assert admit()["image_id"] == value["image_id"]
    else:
        with pytest.raises(ValueError, match="Qwen topology"):
            admit()


def test_share_export_preserves_image_lock_without_private_registry(tmp_path):
    value = image_lock()
    value["image_reference"] = "registry.private.example:5000/rehearsal@sha256:" + "f" * 64
    bundle = b"offline fixture"
    lock = installer.make_lock(PROFILE, ring_site(), "1" * 40, hashlib.sha256(bundle).hexdigest(), image_runtime=value)
    directory = tmp_path / "deployment"
    installer.write(directory / "deployment.lock.json", lock)
    (directory / "source.bundle").write_bytes(bundle)
    output = tmp_path / "share.zip"
    installer.export(directory, output, share=True)
    with zipfile.ZipFile(output) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    assert b"registry.private.example" not in b"".join(files.values())
    exported = json.loads(files["image-lock.json"])
    assert exported["image_id"] == value["image_id"]
    assert exported["parent_receipt_sha256"] == value["parent_receipt_sha256"]
    assert b"--image-lock image-lock.json" in files["README.txt"]
    assert b"toolchain/toolchain.py" in files["rank0/compose.yaml"]


def test_controller_allows_preview_while_another_deployment_is_running(tmp_path, monkeypatch):
    from runtime.host import retained_source
    monkeypatch.setattr(controller, "STATE", tmp_path)
    installer.write(tmp_path / "cluster.json", {"plan": {"nodes": [0, 1, 2, 3]}})
    installer.write(tmp_path / "active.json", {"path": str(tmp_path / "baseline")})
    installer.write(tmp_path / "baseline" / "deployment.lock.json", {"id": "d" * 64})
    installer.write(tmp_path / "deployments" / (PROFILE + "-candidate") / "deployment.lock.json",
                    {"site": {"ranks": []}, "site_input": {}, "image_runtime": installer_image.default_lock()})
    monkeypatch.setattr(retained_source, "review", lambda *a, **k: {"profile": PROFILE, "hosts": [], "phases": []})
    monkeypatch.setattr(retained_source, "apply", lambda directory, operation, **k: {"state": {"operation": "up", "complete": True}}
                        if operation == "saved-status" else pytest.fail("a model action ran"))
    assert controller.lifecycle(["up", PROFILE, "--instance", "candidate", "--plan"]) == 0
    with pytest.raises(ValueError, match="sparkring down"):
        controller.lifecycle(["up", PROFILE, "--instance", "candidate", "--execute"])
    assert installer.read(tmp_path / "active.json")["path"] == str(tmp_path / "baseline")



def test_up_takes_a_named_image_only_with_an_exact_profile():
    with pytest.raises(ValueError, match="require up with an exact profile"):
        controller.lifecycle(["status", "--image", "statusrows"])


def test_release_tags_name_published_installer_images():
    tags = installer_image.release_tags()
    assert tags["2026.10.0"] == installer_image.DEFAULT_LOCK.parent.name
    assert set(tags.values()) <= {row["name"] for row in installer_image.catalog()}


def test_catalog_lists_the_default_first_and_only_registry_images():
    rows = installer_image.catalog()
    assert rows[0]["path"] == installer_image.DEFAULT_LOCK and rows[0]["default"] and rows[0]["tags"] == ["2026.10.0"]
    assert sum(row["default"] for row in rows) == 1
    assert all("@sha256:" in row["lock"]["image_reference"] for row in rows)


def test_image_names_resolve_by_release_name_tag_or_unique_part():
    statusrows = installer_image.RELEASES / "dev-20261001-statusrows-cuda1342-nccl2323-status034" / "installer-image.json"
    assert installer_image.lock_path("statusrows") == statusrows
    assert installer_image.lock_path(statusrows.parent.name) == statusrows
    assert installer_image.lock_path("2026.09.5").parent.name == "dev-20260927-mimovision-cuda1342-nccl2323-status032"
    # The default image, however it is named, is no selection.
    for name in ("2026.10.0", "kraken", installer_image.DEFAULT_LOCK.parent.name):
        assert installer_image.lock_path(name) is None
    with pytest.raises(ValueError, match="matches several images"):
        installer_image.lock_path("20261001")
    with pytest.raises(ValueError, match="No installer image is named status"):
        installer_image.lock_path("status")


def test_an_image_without_the_profile_names_the_images_that_run_it():
    lock = installer.read(installer_image.lock_path("2026.09.5"))
    with pytest.raises(ValueError, match="images that run it: dev-20261001-kraken") as refused:
        installer_image.for_profile("mimo-v26-flash-mopd-tp2", lock)
    assert "mimovision" not in str(refused.value).split("images that run it:")[1]

SHARED = ("deepseek-v41-flash-tp4", "glm53-flash-nvfp4-spark-tp2", "glm53-flash-nvfp4-spark-tp4",
          "mimo-v26-flash-mopd-tp2", "mimo-v26-flash-mopd-tp4", "qwen38-flash-next-qad-tp4", "qwen38-flash-next-tp2",
          "swift15-qwen38-flash-next-tp2", "swift15-qwen38-flash-next-tp4")
SWIFT = ("swift15-qwen38-flash-next-tp2", "swift15-qwen38-flash-next-tp4")


def test_release_lock_lists_every_installer_profile_on_one_image():
    lock = installer_image.default_lock()
    assert lock["schema"] == installer_image.SCHEMA and tuple(lock["profiles"]) == SHARED
    assert installer.INSTALLABLE == frozenset(SHARED)
    assert lock["image_reference"].startswith("ghcr.io/fujitsupolycom/sparkring@sha256:")
    for profile in SHARED:
        assert installer_image.for_profile(profile) == lock
        card = setup.selection(profile)
        if not profile.startswith("qwen"):
            # Shared-image profiles select the same image through their release.
            assert card["image_id"] == lock["image_id"] and card["image_reference"] == lock["image_reference"]
    with pytest.raises(ValueError, match="not admitted"):
        installer_image.for_profile("glm53-flash-spark-tp4-dcp1-sparkcache")


def test_replaced_profile_ids_keep_release_locks_valid_and_name_their_replacement():
    locks = sorted((installer_image.ROOT / "runtime/releases").glob("*/installer-image.json"))
    listed = set()
    for path in locks:
        value = json.loads(path.read_text(encoding="utf-8"))
        for profile in installer_image.profiles_of(value):
            installer_image.validate(value, profile)
            listed.add(profile)
    assert set(profiles.REPLACED) <= listed
    assert not set(profiles.REPLACED) & set(installer_image.default_lock()["profiles"])
    for old, replacement in profiles.REPLACED.items():
        assert replacement in installer_image.SUPPORTED and old not in profiles.catalog()
        for call in (installer_image.for_profile, profiles.load, lambda value: models.select(value, 2)):
            with pytest.raises(ValueError, match="use " + replacement):
                call(old)


@pytest.mark.parametrize("change", ["unsorted", "unknown", "empty"])
def test_shared_lock_profile_list_is_exact(change):
    value = copy.deepcopy(installer_image.default_lock())
    if change == "unsorted":
        value["profiles"] = list(reversed(value["profiles"]))
    elif change == "unknown":
        value["profiles"] = sorted([*value["profiles"], "qwen38-flash-next-qad-tp4-sparkcache"])
    else:
        value["profiles"] = []
    with pytest.raises(ValueError):
        installer_image.validate(value, PROFILE)


@pytest.mark.parametrize("profile", [profile for profile in installer_image.SUPPORTED
                                     if profile not in installer_image.QWEN4_EXP])
def test_other_models_render_on_the_shared_image(monkeypatch, profile):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Offline render contacted a host"))
    lock_value = installer_image.default_lock()
    nodes = 4 if profile.endswith("tp4") else 2
    raw = ring_site() if nodes == 4 else site(2)
    lock = installer.make_lock(profile, raw, "1" * 40, "2" * 64, image_runtime=lock_value)
    assert lock["backend"] == "compose" and lock["selection"]["release"] == lock_value["name"]
    specs = installer.specifications(lock)
    assert len(specs) == nodes
    for rank, spec in enumerate(specs):
        assert spec.image_id == lock_value["image_id"] and spec.entrypoint == installer_image.ENTRYPOINT
        assert spec.command[0] == "serve" and spec.command[spec.command.index("--node-rank") + 1] == str(rank)
        assert ("--headless" in spec.command) == bool(rank)
        assert spec.environment["VLLM_PLUGINS"].split(",")[:2] == ["b12x_loader", "sparkring_status"]
        assert "VLLM_QWEN3_8_FLASH_NEXT_HC_TP" not in spec.environment
        assert not any(key.startswith(("VLLM_QWEN3_8_", "QWEN_")) for key in spec.environment)
        assert spec.environment["SPARKCACHE_ENABLED"] == "0" and "--kv-transfer-config" not in spec.command
        assert "--enable-prefix-caching" in spec.command
        # Named and required tool_choice fail closed on an image that carries the policy.
        assert spec.environment[installer_image.TOOL_CHOICE_CONTRACT] == "1"
        assert spec.environment["VLLM_NCCL_SO_PATH"] == "/opt/sparkring/toolchain/nccl/lib/libnccl.so.2"
        family = json.loads((installer.ROOT / "profiles" / profile / "config.json").read_text())["cache_namespace"]
        assert spec.environment["XDG_CACHE_HOME"] == f"/cache/{family}-{lock_value['image_id'][7:19]}-{lock['selection']['model_revision'][:12]}"
        assert spec.mounts[-1].target == installer_image.BINDING_TARGET
        if rank == 0:
            assert spec.health_command[0] == "python3"
            # Health failures count only after the installer's 30-minute readiness window.
            assert spec.health_start_period == installer_image.HEALTH_START_SECONDS == 1800
    connection = installer.connection(lock)
    assert connection["model"].endswith(f"-TP{nodes}")


def test_qwen_admission_features_do_not_apply_to_other_models():
    value, image, receipts = admission_fixture()
    value = {**{k: v for k, v in value.items() if k != "profile"}, "schema": installer_image.SCHEMA,
             "profiles": ["glm53-flash-nvfp4-spark-tp4", PROFILE], "image_bytes": 1, "download_bytes": 1}
    parent = json.loads(receipts[installer_image.PARENT_RECEIPT])
    parent["capabilities"]["features"] = []
    raw = json.dumps(parent).encode()
    receipts[installer_image.PARENT_RECEIPT] = raw
    value["parent_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    toolchain = json.loads(receipts[installer_image.TOOLCHAIN_RECEIPT])
    toolchain["parent_receipt_sha256"] = value["parent_receipt_sha256"]
    raw = json.dumps(toolchain).encode()
    receipts[installer_image.TOOLCHAIN_RECEIPT] = raw
    value["toolchain_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    def run(argv, **kwargs):
        if argv[1] == "image":
            return SimpleNamespace(stdout=json.dumps([image]))
        if argv[-1] in receipts:
            return SimpleNamespace(stdout=receipts[argv[-1]])
        return SimpleNamespace(stdout="verified")
    assert installer_image.admit(value, run=run, profile="glm53-flash-nvfp4-spark-tp4", nodes=4)["serving_qualified"] is False
    with pytest.raises(ValueError, match="Qwen topology"):
        installer_image.admit(value, run=run, profile=PROFILE, nodes=4)
    # A derivative checkpoint of the Qwen3.8-Flash-Next architecture needs the same image features.
    value["profiles"] = sorted([*value["profiles"], "swift15-qwen38-flash-next-tp4"])
    with pytest.raises(ValueError, match="Qwen topology"):
        installer_image.admit(value, run=run, profile="swift15-qwen38-flash-next-tp4", nodes=4)


@pytest.mark.parametrize("profile", SWIFT)
def test_swift_renders_the_qwen_architecture_settings_for_its_checkpoint_format(monkeypatch, profile):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Offline render contacted a host"))
    lock_value = installer_image.default_lock()
    nodes = 4 if profile.endswith("tp4") else 2
    lock = installer.make_lock(profile, ring_site() if nodes == 4 else site(2), "1" * 40, "2" * 64,
                               image_runtime=lock_value)
    assert lock["backend"] == "compose" and lock["selection"]["release"] == lock_value["name"]
    specs = installer.specifications(lock)
    assert len(specs) == nodes
    for rank, spec in enumerate(specs):
        assert spec.image_id == lock_value["image_id"] and spec.entrypoint == installer_image.ENTRYPOINT
        env, command = spec.environment, spec.command
        # The Qwen features, HC token-row ownership and decode dispatch apply as for Qwen3.8-Flash-Next.
        assert env["SPARKRING_FEATURES"] == "qwen-collectives,qwen4-prefill"
        assert env["VLLM_QWEN3_8_HC_PREFILL_MODE"] == "shard" and env["VLLM_QWEN3_8_FLASH_NEXT_HC_TP"] == "0"
        assert env["QWEN_DISPATCH_AR_BYTES"] == "327680" and env["VLLM_QWEN4_EXP_MXFP8_HC"] == "1"
        # The checkpoint declares ModelOpt NVFP4, which modelopt_mixed refuses.
        assert command[command.index("--quantization") + 1] == "modelopt_fp4"
        # Its draft experts are BF16, so no quantized draft MoE backend is named.
        draft = json.loads(command[command.index("--speculative-config") + 1])
        assert draft["method"] == "mtp" and draft["num_speculative_tokens"] == 3 and "moe_backend" not in draft
        # Two Sparks read the 95.4 GiB BF16 PLE table from disk; four hold it in GPU memory.
        if nodes == 2:
            assert env["VLLM_PLE_TABLE_MEMORY"] == "disk" and "VLLM_PLE_CPU_OFFLOAD" not in env
        else:
            assert env["VLLM_PLE_CPU_OFFLOAD"] == "0" and "VLLM_PLE_TABLE_MEMORY" not in env
        assert env["SPARKCACHE_ENABLED"] == "0" and "--enable-prefix-caching" in command
        assert env["XDG_CACHE_HOME"].startswith("/cache/swift15-qwen38-flash-next-")
        if rank == 0:
            assert spec.health_start_period == installer_image.HEALTH_START_SECONDS
    assert installer.connection(lock)["model"] == f"Swift-1.5-Qwen3.8-Flash-Next-NVFP4-TP{nodes}"
