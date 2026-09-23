"""External-image admission and Qwen plans retain source and operational bounds."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from runtime.common import external_candidate as external, native_candidate, qwen_flash_next as qwen


RELEASE = "shared-2026.09.4-rc.4"
IMAGE = "sha256:" + "1" * 64
TRANSPORT = "tp2-rocenante-adaptive-prepared"


def fixture():
    sources = {
        name: {"commit": "a" * 40, "upstream": "b" * 40, "baseline_commit": "c" * 40,
               "archive": name + ".tar", "archive_sha256": "d" * 64,
               "baseline_archive": name + "-base.tar", "baseline_archive_sha256": "e" * 64}
        for name in ("vllm", "b12x")
    }
    base = {"reference": "example/runtime@sha256:" + "2" * 64, "config_id": "sha256:" + "3" * 64}
    contract = "/opt/sparkring/contracts/cache.json"
    installed = {
        "schema": "sparkring-external-installed/v1", "composition_sha256": "4" * 64,
        "base": base, "sources": sources,
        "files": {contract: "5" * 64, "/opt/sparkring/transports/" + TRANSPORT + "/manifest.json": "6" * 64},
        "capabilities": {
            "features": ["qwen-collectives", "qwen4-prefill"], "sparkcache_contract": contract,
            "transport_profile": TRANSPORT, "transport_manifest_sha256": "6" * 64,
            "hc_supported_modes": external.HC_SUPPORTED_MODES, "serving_qualified": False,
        },
    }
    raw = json.dumps(installed).encode()
    publication = {
        "schema": "sparkring-shared-image-publication/v1", "release": RELEASE,
        "runtime_layout": external.LAYOUT, "platform": "linux/arm64", "image_id": IMAGE,
        "image_reference": "ghcr.io/fujitsupolycom/sparkring@sha256:" + "7" * 64,
        "registry_manifest_digest": "sha256:" + "7" * 64,
        "anonymous_config_verified": True, "anonymous_pull_completed": True,
        "installed_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "composition_sha256": "4" * 64, "base": base, "sources": sources,
        "features": ["qwen-collectives", "qwen4-prefill"],
        "hc_supported_modes": external.HC_SUPPORTED_MODES,
        "sparkcache_contract": {"path": contract, "sha256": "5" * 64},
        "transport": {"profile": TRANSPORT, "manifest_sha256": "6" * 64},
        "profiles": {"profiles/external/config.json": "8" * 64},
    }
    inspection = {"Id": IMAGE, "Os": "linux", "Architecture": "arm64",
                  "Config": {"Entrypoint": [external.PYTHON, external.ENTRYPOINT]}}
    verified = {"schema": "sparkring-external-verification/v1", "composition_sha256": "4" * 64,
                "sources": sources, "files_verified": 2, "framework_native_rebuilt": False,
                "owned_python_roots": [],
                "serving_qualified": False}
    return publication, inspection, raw, verified


@pytest.fixture
def registered(tmp_path, monkeypatch):
    record, inspection, raw, verified = fixture()
    directory = tmp_path / "runtime/releases" / RELEASE
    directory.mkdir(parents=True)
    publication = directory / "publication.json"
    publication.write_text(json.dumps(record))
    monkeypatch.setattr(native_candidate, "ROOT", tmp_path)
    monkeypatch.setattr(external, "ROOT", tmp_path)
    return record, publication, inspection, raw, verified


def test_exact_external_publication_and_receipt_are_accepted(registered):
    record, _, info, raw, verified = registered
    assert external.publication(RELEASE, image_id=IMAGE) == record
    result = external.validate(record, IMAGE, info, raw, verified)
    assert result["schema"] == "sparkring-external-host-verification/v1"


@pytest.mark.parametrize("field,value", [
    ("runtime_layout", "native"), ("composition_sha256", ""), ("base", {}),
    ("sources", {}), ("features", []), ("sparkcache_contract", {}),
    ("profiles", {"../escape.json": "8" * 64}), ("anonymous_pull_completed", False),
])
def test_publication_admits_only_complete_external_identity(registered, field, value):
    record, path, _, _, _ = registered
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        external.publication(RELEASE)


@pytest.mark.parametrize("field,value", [
    ("composition_sha256", "0" * 64), ("sources", {}), ("files_verified", 0),
    ("framework_native_rebuilt", True), ("schema", "sparkring-native-verification/v1"),
])
def test_source_or_verification_mismatch_is_rejected(registered, field, value):
    record, _, info, raw, verified = registered
    verified[field] = value
    with pytest.raises(ValueError, match="composition/source"):
        external.validate(record, IMAGE, info, raw, verified)


@pytest.mark.parametrize("mutation", ["transport", "contract", "hc", "base"])
def test_installed_contract_matches_publication_and_actual_inventory(registered, mutation):
    record, _, info, raw, verified = registered
    installed = json.loads(raw)
    if mutation == "transport":
        installed["files"]["/opt/sparkring/transports/" + TRANSPORT + "/manifest.json"] = "0" * 64
    elif mutation == "contract":
        installed["capabilities"]["sparkcache_contract"] = "/wrong.json"
    elif mutation == "hc":
        installed["capabilities"]["hc_supported_modes"] = {}
    else:
        installed["base"]["config_id"] = "sha256:" + "0" * 64
    raw = json.dumps(installed).encode()
    record["installed_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="composition/source"):
        external.validate(record, IMAGE, info, raw, verified)


def test_bad_platform_is_rejected_before_running_container(registered):
    _, _, info, _, _ = registered
    info["Architecture"] = "amd64"
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[1:3] == ["image", "inspect"]
        return SimpleNamespace(stdout=json.dumps([info]))

    with pytest.raises(ValueError, match="platform"):
        external.verify_image(IMAGE, RELEASE, run=run)
    assert len(calls) == 1


def test_host_verifier_uses_external_receipt_and_gpu_free_network_free_checks(registered):
    _, _, info, raw, verified = registered
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1:3] == ["image", "inspect"]:
            return SimpleNamespace(stdout=json.dumps([info]))
        assert "runc" in argv and "none" in argv and "NVIDIA_VISIBLE_DEVICES=void" in argv
        return SimpleNamespace(stdout=raw if external.RECEIPT in argv else json.dumps(verified))

    assert external.verify_image(IMAGE, RELEASE, run=run)["image_id"] == IMAGE
    assert len(calls) == 3


def opt_in_profile(nodes, cached):
    directory = qwen.CONFIG_ROOT if nodes == 2 else qwen.TP4_CONFIG.parent
    profile = qwen.read(directory / ("sparkcache.json" if cached else "config.json"))
    profile["image_extension"] = "external-base"
    profile["image_release"] = RELEASE
    profile["container_envelope"] = copy.deepcopy(external.ENVELOPE)
    profile["environment"].update({
        "VLLM_QWEN3_8_FLASH_NEXT_HC_TP": "1", "VLLM_QWEN3_8_HC_PREFILL_MODE": "off",
    })
    profile["vllm_args"] += ["--model-loader-extra-config", '{"read_mode":"bounce","io_threads":8}']
    return profile


def register_profile(registered, profile):
    record, publication, _, _, _ = registered
    name = next(iter(record["profiles"]))
    path = external.ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile))
    record["profiles"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    publication.write_text(json.dumps(record))
    return path


@pytest.mark.parametrize("nodes,cached", [(2, False), (2, True), (4, False), (4, True)])
def test_external_qwen_plan_retains_operating_parameters_and_pins_hooks(registered, nodes, cached):
    record, _, _, _, _ = registered
    profile = opt_in_profile(nodes, cached)
    before = copy.deepcopy(profile)
    register_profile(registered, profile)
    spec = qwen.container_spec(
        profile, rank=0, master="192.0.2.1", host_ip="192.0.2.1", interface="test0",
        image=IMAGE, model="/models/qwen", cache="/runtime/cache", remote=True,
    )
    assert profile == before
    assert spec.entrypoint == ("python3",)
    assert spec.command[:3] == (external.ENTRYPOINT, "serve", "/models/target")
    assert spec.health_command[0] == "python3" and spec.working_dir == "/"
    assert spec.cap_add == ("IPC_LOCK",) and spec.security_opt == ("seccomp=unconfined",)
    assert spec.environment["VLLM_QWEN3_8_FLASH_NEXT_HC_TP"] == "1"
    assert spec.environment["VLLM_QWEN3_8_HC_PREFILL_MODE"] == "off"
    assert spec.environment["LD_PRELOAD"] == external.NCCL
    assert spec.environment["SPARKRING_TRANSPORT_MANIFEST_SHA256"] == record["transport"]["manifest_sha256"]
    for name in ("NCCL_IB_HCA", "B12X_ROCE_HCA", "OMP_NUM_THREADS", "SPARKRING_FEATURES",
                 "VLLM_QWEN3_8_PREFILL_COALESCE", "VLLM_QWEN3_8_FLASH_NEXT_OVERLAP"):
        assert spec.environment[name] == before["environment"][name]
    for flag in ("--tensor-parallel-size", "--decode-context-parallel-size", "--max-model-len",
                 "--max-num-seqs", "--max-num-batched-tokens", "--kv-cache-memory-bytes",
                 "--speculative-config", "--compilation-config", "--model-loader-extra-config"):
        assert spec.command[spec.command.index(flag) + 1] == before["vllm_args"][before["vllm_args"].index(flag) + 1]
    assert "/opt/venv" not in json.dumps([spec.environment, spec.command, spec.entrypoint])
    assert qwen.image_verification_options(profile) == {"external_release": RELEASE}
    if cached:
        transfer = json.loads(spec.command[spec.command.index("--kv-transfer-config") + 1])
        extra = transfer["kv_connector_extra_config"]
        assert extra["spark_cache_async_page_capture_lease_contract"] == record["sparkcache_contract"]["path"]
        assert extra["spark_cache_async_page_capture_vllm_root"] == external.SITE
        assert extra["spark_cache_root"].endswith(record["composition_sha256"][:16] + "-" + external.profile_identity(profile)[:16])


def test_changed_pinned_profile_is_not_canonical(registered):
    profile = opt_in_profile(2, False)
    path = register_profile(registered, profile)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="serving profile changed"):
        qwen.canonical(profile)


@pytest.mark.parametrize("mutation", ["envelope", "sequence_limit", "capture_limit", "loader", "tp2_prefill"])
def test_external_operational_scope_is_explicit(registered, mutation):
    record, _, _, _, _ = registered
    profile = opt_in_profile(2, False)
    if mutation == "envelope":
        profile["container_envelope"] = {}
    elif mutation == "tp2_prefill":
        profile["environment"]["SPARKRING_FEATURES"] = "qwen4-prefill"
    else:
        flag, value = {
            "sequence_limit": ("--max-num-seqs", "64"),
            "capture_limit": ("--compilation-config", '{"max_cudagraph_capture_size":128}'),
            "loader": ("--model-loader-extra-config", '{"read_mode":"bounce","io_threads":16}'),
        }[mutation]
        profile["vllm_args"][profile["vllm_args"].index(flag) + 1] = value
    with pytest.raises(ValueError):
        external.profile_settings(profile, record)


def test_external_selection_cannot_mix_legacy_source_or_admission(registered):
    profile = opt_in_profile(2, False)
    with pytest.raises(ValueError, match="legacy source"):
        qwen.image_policy(profile, local_source_extension="lil-r37-qwen-prefill")
    with pytest.raises(ValueError, match="cannot be combined"):
        qwen.verify_image(IMAGE, external_release=RELEASE, native_release="shared-2026.09.3")


@pytest.mark.parametrize("nodes,projection,rows", [(2, "1", "off"), (4, "1", "off"), (4, "0", "shard")])
def test_external_profile_preserves_its_explicit_hc_ownership(registered, nodes, projection, rows):
    record, _, _, _, _ = registered
    profile = opt_in_profile(nodes, True)
    profile["environment"].update({
        "VLLM_QWEN3_8_FLASH_NEXT_HC_TP": projection, "VLLM_QWEN3_8_HC_PREFILL_MODE": rows,
    })
    selected = external.profile_settings(profile, record)
    assert selected["environment"]["VLLM_QWEN3_8_FLASH_NEXT_HC_TP"] == projection
    assert selected["environment"]["VLLM_QWEN3_8_HC_PREFILL_MODE"] == rows
    assert selected["environment"]["VLLM_MXFP8_LM_HEAD"] == "0"
    assert "B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE" not in selected["environment"]


@pytest.mark.parametrize("nodes,projection,rows", [(2, "0", "shard"), (2, "0", "off"), (4, "1", "shard"), (4, "0", "off"), (4, None, "shard")])
def test_external_profile_rejects_unsupported_or_conflicting_hc_ownership(nodes, projection, rows):
    profile = opt_in_profile(nodes, True)
    profile["environment"].update({
        "VLLM_QWEN3_8_FLASH_NEXT_HC_TP": projection, "VLLM_QWEN3_8_HC_PREFILL_MODE": rows,
    })
    with pytest.raises(ValueError, match="mutually exclusive"):
        external.validate_profile_contract(profile)


def test_persistent_cache_identity_changes_with_hc_mode_and_is_stable_for_one_profile(registered):
    record, _, _, _, _ = registered
    projection = opt_in_profile(4, True)
    rows = copy.deepcopy(projection)
    rows["environment"].update({
        "VLLM_QWEN3_8_FLASH_NEXT_HC_TP": "0", "VLLM_QWEN3_8_HC_PREFILL_MODE": "shard",
    })

    def cache_root(profile):
        args = external.profile_settings(profile, record)["vllm_args"]
        return json.loads(args[args.index("--kv-transfer-config") + 1])["kv_connector_extra_config"]["spark_cache_root"]

    assert cache_root(projection) != cache_root(rows)
    assert cache_root(rows) == cache_root(copy.deepcopy(rows))
    changed = copy.deepcopy(rows)
    changed["environment"]["VLLM_MXFP8_LM_HEAD"] = "1"
    assert cache_root(changed) != cache_root(rows)


def status_record():
    return {
        "distribution": "sparkring-runtime-status", "version": "0.1.0",
        "plugin_name": "sparkring_status", "python_root": external.STATUS_PYTHON_ROOT,
        "wheel_sha256": "a" * 64, "source_archive_sha256": "b" * 64,
        "entry_points": external.STATUS_ENTRY_POINTS,
    }


def test_status_activation_requires_the_published_artifact(registered):
    record, _, _, _, _ = registered
    profile = opt_in_profile(2, True)
    profile["environment"]["VLLM_PLUGINS"] += ",sparkring_status"
    with pytest.raises(ValueError, match="must match"):
        external.profile_settings(profile, record)
    record["runtime_status"] = status_record()
    assert external.profile_settings(profile, record)["environment"]["VLLM_PLUGINS"] == "b12x_loader,sparkring_status"
    profile["environment"]["VLLM_PLUGINS"] = "b12x_loader"
    with pytest.raises(ValueError, match="must match"):
        external.profile_settings(profile, record)


def test_status_publication_requires_both_official_entry_points(registered):
    record, path, _, _, _ = registered
    record["runtime_status"] = status_record()
    path.write_text(json.dumps(record))
    assert external.publication(RELEASE)["runtime_status"] == status_record()
    record["runtime_status"]["entry_points"] = {"vllm.endpoint_plugins": {}}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="status artifact"):
        external.publication(RELEASE)


def test_status_host_admission_binds_the_import_root_and_complete_verification(registered):
    record, _, info, raw, verification = registered
    record["runtime_status"] = status_record()
    installed = json.loads(raw)
    installed["capabilities"]["runtime_status"] = status_record()
    installed["python_roots"] = {external.STATUS_PYTHON_ROOT: {"sparkring_runtime_status/plugin.py": "c" * 64}}
    installed["files"][external.STATUS_PYTHON_ROOT + "/sparkring_runtime_status/plugin.py"] = "c" * 64
    installed["files"][external.SITE + "/sparkring_runtime_status.pth"] = external.sha((external.STATUS_PYTHON_ROOT + "\n").encode())
    raw = json.dumps(installed).encode()
    record["installed_receipt_sha256"] = external.sha(raw)
    verification["files_verified"] = len(installed["files"])
    verification["owned_python_roots"] = [external.STATUS_PYTHON_ROOT]
    assert external.validate(record, IMAGE, info, raw, verification)["image_id"] == IMAGE
    verification["owned_python_roots"] = []
    with pytest.raises(ValueError, match="composition/source"):
        external.validate(record, IMAGE, info, raw, verification)
