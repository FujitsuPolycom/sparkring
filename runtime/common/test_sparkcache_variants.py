"""Opt-in SparkCache variants preserve their base profiles' serving settings.

The GLM, Qwen and DeepSeek variants pin checkpoints and isolated roots. They
run only on a lock that lists them: no published lock of this package does;
locks recorded by the derived compositions supply explicit admission.
"""

import json
from pathlib import Path

import pytest

from runtime.common import installer, installer_image, toolchain_profiles
from runtime.common.test_installer import site
from runtime.host import check
from runtime.host.storage import CACHE_NAME

VARIANTS = installer_image.SPARKCACHE_VARIANTS
QWEN = installer_image.QWEN_SPARKCACHE_VARIANTS
DEEPSEEK = installer_image.DEEPSEEK_SPARKCACHE_VARIANTS[0]
BASES = {DEEPSEEK: "deepseek-v41-flash-tp4", QWEN[0]: "qwen38-flash-next-tp2", QWEN[1]: "qwen38-flash-next-qad-tp4", "glm53-flash-nvfp4-sparkcache-tp2": "glm53-flash-nvfp4-spark-tp2",
         "glm53-flash-nvfp4-sparkcache-tp4": "glm53-flash-nvfp4-spark-tp4",
         "glm53-flash-nvfp4-sparkcache-crc32-tp2": "glm53-flash-nvfp4-spark-tp2",
         "glm53-flash-nvfp4-sparkcache-crc32-tp4": "glm53-flash-nvfp4-spark-tp4"}
CACHE_ROOT = "/cache/glm53-flash-sparkcache-ab53c097d7ca-dec48abd33ef"
CHECKPOINT_IDENTITY = "6af0343976eefbfa0fb59d87befcf6d10f02fef90ba0e7881dd1201fe0ecd76e"
SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"
CONTRACT = SITE_PACKAGES + "/sparkcache/runtime_patches/vllm-connector-jobs-gab86b70.json"


def cache_root(profile):
    if profile == DEEPSEEK:
        return "/cache/deepseek-v41-flash-sparkcache-ab53c097d7ca-dba1be0a40aa"
    if profile in QWEN:
        revision = "dc5b8ab2fa14" if profile == QWEN[0] else "60215d26cf5e"
        return f"/cache/{profile}-ab53c097d7ca-{revision}"
    if profile in installer_image.SPARKCACHE_CRC32_VARIANTS:
        return f"/cache/glm53-flash-sparkcache-crc32-{profile[-3:]}-8abbe9da67b8-dec48abd33ef"
    return CACHE_ROOT


def model_profile(profile):
    if profile == DEEPSEEK:
        return "deepseek-v41-b12x-hma"
    return "qwen38-flash-next-hybrid" if profile in QWEN else "glm53-flash-hybrid"


def checkpoint_identity(profile):
    if profile == DEEPSEEK:
        return "0b447c086ca74be5e7ed9e98c424637f1f8015ee38fd1e767370415377ed401e"
    if profile in QWEN:
        return {QWEN[0]: "7906c93c59bf153168170b601cf36ae7cdd31927797dc4608320fe1ba8ec2c78",
                QWEN[1]: "759a6ce3138e95cddaa4dc8e53480274d2a3e75154cf1893ff275237b1ded2b6"}[profile]
    return CHECKPOINT_IDENTITY


def development_lock():
    """A development image lock that lists the variants beside the shared profiles.

    The derived layer's record step writes the real one; no lock of this
    package lists the variants.
    """
    value = json.loads(json.dumps(installer_image.default_lock()))
    value["profiles"] = sorted([*value["profiles"], *VARIANTS])
    return value


def ring_site(nodes):
    raw = site(nodes)
    if nodes == 4:
        for row in raw["hosts"]:
            row["fabric"] = {"site_path": "/etc/sparkring/site.json", "site_sha256": "1" * 64,
                             "plan_sha256": "2" * 64}
    return raw


def connector(profile_id):
    lock = development_lock()
    raw = ring_site(4 if profile_id.endswith("tp4") else 2)
    lock_value = installer.make_lock(profile_id, raw, "1" * 40, "2" * 64, image_runtime=lock)
    spec = installer.specifications(lock_value)[0]
    command = list(spec.command)
    connector_json = json.loads(command[command.index("--kv-transfer-config") + 1])
    return lock_value, spec, connector_json


@pytest.mark.parametrize("profile", VARIANTS)
def test_variant_keeps_every_speed_setting_of_its_base_profile(profile):
    base = toolchain_profiles.read(installer_image.ROOT / "profiles" / BASES[profile] / "config.json")
    variant = toolchain_profiles.read(installer_image.ROOT / "profiles" / profile / "config.json")
    selected = base if profile == DEEPSEEK else toolchain_profiles.checkpoint_settings(
        base, "qad-step5500-ple1000" if profile == QWEN[1] else "csf")
    # Compare the complete effective configuration, not selected speed flags.
    selected = json.loads(json.dumps(selected))
    actual = json.loads(json.dumps(variant))
    for value in (selected, actual):
        for field in ("checkpoint", "checkpoints", "preferred_checkpoint", "checkpoint_aliases",
                      "status", "qualification"):
            value.pop(field, None)
    index = actual["vllm_args"].index("--kv-transfer-config")
    del actual["vllm_args"][index:index + 2]
    actual["environment"].pop("SPARK_CONTEXT_CACHE_ROOT")
    actual["environment"]["SPARKCACHE_ENABLED"] = "0"
    # JSON-valued arguments are semantically ordered objects.
    for value in (selected, actual):
        args = value["vllm_args"]
        at = args.index("--speculative-config") + 1
        args[at] = json.dumps(json.loads(args[at]), sort_keys=True)
    if profile not in (*QWEN, DEEPSEEK):
        # The recorded GLM variants omit this optional base-profile threshold.
        assert selected["environment"].pop("B12X_W4A16_A4_PREFILL_MIN_TOKENS") == "1536"
    assert actual == selected



@pytest.mark.parametrize("profile", VARIANTS)
def test_variant_renders_the_connector_settings_and_storage_paths(profile):
    lock, spec, connector_json = connector(profile)
    assert spec.environment["SPARKCACHE_ENABLED"] == "1"
    assert spec.environment["SPARK_CONTEXT_CACHE_ROOT"] == cache_root(profile)
    # The root matches the name `sparkring storage` classifies as a cache
    # directory: <family>-<image 12>-<revision 12>.
    assert CACHE_NAME.fullmatch(Path(cache_root(profile)).name)
    assert connector_json["kv_connector"] == "SparkContextCacheConnector"
    assert connector_json["kv_connector_module_path"] == "sparkcache.spark_context_cache_connector"
    assert connector_json["kv_role"] == "kv_both"
    assert connector_json["kv_load_failure_policy"] == "recompute"
    extra = connector_json["kv_connector_extra_config"]
    assert extra["spark_cache_model_profile"] == model_profile(profile)
    assert extra["spark_cache_publication_schema"] == "tail-cow-v2"
    if profile == DEEPSEEK:
        assert extra["spark_cache_draft_policy"] == "colocated_target"
        assert "spark_cache_draft_checkpoint_sha256" not in extra
        assert extra["spark_cache_cuda_restore"] is False
        assert extra["spark_cache_async_page_capture"] is False
        assert extra["spark_cache_async_page_capture_slot_bytes"] == 1024**3
        assert extra["spark_cache_async_page_capture_slot_count"] == 2
        assert extra["spark_cache_min_span_tokens"] == 4096
        assert extra["spark_cache_max_span_tokens"] == 524288
        assert extra["spark_cache_max_bytes"] == 32 * 1024**3
        assert extra["spark_cache_low_watermark_bytes"] == 28 * 1024**3
        assert extra["spark_cache_target_checkpoint_sha256"] == checkpoint_identity(profile)
        assert extra["spark_cache_payload_integrity"] == "fast"
        assert spec.memory == 115964116992
        return
    assert extra["spark_cache_draft_policy"] == "separate"
    assert extra["spark_cache_target_checkpoint_sha256"] == checkpoint_identity(profile)
    assert extra["spark_cache_draft_checkpoint_sha256"] == checkpoint_identity(profile)
    assert extra["spark_cache_max_span_tokens"] == {QWEN[0]: 49152, QWEN[1]: 61920}.get(
        profile, 262144 if profile in installer_image.SPARKCACHE_CRC32_VARIANTS else 65536)
    if profile in QWEN:
        assert extra["spark_cache_async_page_capture_slot_bytes"] == 512 * 1024**2
        assert extra["spark_cache_async_page_capture_slot_count"] == 2
        assert extra["spark_cache_min_span_tokens"] == 4096
        assert extra["spark_cache_max_bytes"] == 32 * 1024**3
        assert extra["spark_cache_low_watermark_bytes"] == 28 * 1024**3
    assert extra["spark_cache_async_page_capture"] is True
    assert extra["spark_cache_async_page_capture_lease_mode"] == "connector-jobs"
    assert extra["spark_cache_async_page_capture_lease_contract"] == CONTRACT
    assert extra["spark_cache_async_page_capture_vllm_root"] == SITE_PACKAGES
    assert extra["spark_cache_cuda_restore"] is True
    assert extra["spark_cache_cuda_placement_library"] == \
        "/opt/sparkring/sparkcache/lib/libspark_cache_placement.so"
    assert extra["spark_cache_async_page_capture_library"] == \
        "/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so"
    assert extra["spark_cache_root"] == cache_root(profile)
    if profile in (*installer_image.SPARKCACHE_CRC32_VARIANTS, *QWEN):
        assert extra["spark_cache_payload_integrity"] == "fast"


@pytest.mark.parametrize("profile", VARIANTS)
def test_no_image_of_this_package_runs_the_variant(profile):
    from runtime.common import image_lock
    for row in installer_image.catalog():
        with pytest.raises(ValueError, match="not admitted"):
            image_lock.validate(row["lock"], profile)
    with pytest.raises(ValueError, match="not admitted"):
        installer_image.for_profile(profile)


@pytest.mark.parametrize("profile", VARIANTS)
def test_a_lock_that_lists_the_variant_admits_it(profile):
    lock = development_lock()
    assert installer_image.validate(lock, profile)["name"] == lock["name"]
    nodes = installer_image.profile_nodes(profile)
    assert nodes == (4 if profile.endswith("tp4") else 2)


@pytest.mark.parametrize("profile", VARIANTS)
def test_check_reports_the_connector_state(profile):
    lock_value, spec, _ = connector(profile)
    state = check.connector_state(lock_value)
    assert state["enabled"] is True
    assert state["model_profile"] == model_profile(profile)
    assert state["lease_mode"] == "connector-jobs"
    assert state["lease_contract"] == CONTRACT
    assert state["cache_root"] == cache_root(profile)
    assert state["checkpoint_identity"].startswith(checkpoint_identity(profile)[:12])


def test_check_reports_no_connector_state_for_a_cache_off_profile():
    lock_value = installer.make_lock("glm53-flash-nvfp4-spark-tp2", ring_site(2), "1" * 40, "2" * 64,
                                     image_runtime=installer_image.default_lock())
    assert check.connector_state(lock_value) is None


@pytest.mark.parametrize("profile", QWEN)
def test_qwen_checkpoint_receipt_matches_complete_committed_manifest_and_effective_settings(profile):
    import hashlib
    root = installer_image.ROOT / "profiles"
    value = json.loads((root / profile / "config.json").read_text())
    receipt = json.loads((root / profile / "checkpoint-identity.json").read_text())
    inputs = receipt["inputs"]
    model = value["model"]
    pins = root / "checkpoints" / model["repository"].replace("/", "--") / (model["revision"] + ".json")
    assert inputs["artifacts"]["manifest_sha256"] == hashlib.sha256(pins.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert inputs["model_config"]["sha256"] == model["config_sha256"]
    assert (inputs["repository"], inputs["revision"]) == (model["repository"], model["revision"])
    args = value["vllm_args"]
    assert inputs["draft"] == {"mode": "embedded", "config": json.loads(args[args.index("--speculative-config") + 1])}
    assert inputs["quantization"] == {"method": args[args.index("--quantization") + 1],
                                       "load_format": args[args.index("--load-format") + 1], "encoding": "safetensors"}
    envelope = {"schema": receipt["schema"], "inputs": inputs}
    digest = hashlib.sha256(json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
    assert digest == receipt["checkpoint_sha256"] == checkpoint_identity(profile)
    assert profile in installer_image.QWEN4_EXP
    assert not (root / profile / "compose").exists()


def test_every_other_installer_profile_keeps_sparkcache_off():
    """The SparkCache-off invariant holds for every installer profile but the variants."""
    for profile in installer_image.SUPPORTED:
        config = json.loads((installer_image.ROOT / "profiles" / profile / "config.json").read_text())
        enabled = config["environment"].get("SPARKCACHE_ENABLED")
        connector_on = "--kv-transfer-config" in config["vllm_args"]
        if profile in VARIANTS:
            assert enabled == "1" and connector_on
        else:
            assert enabled == "0" and not connector_on, profile


CRC32_TUNED_SETTINGS = {
    "spark_cache_cuda_restore_io_workers": 8,
    "spark_cache_max_span_tokens": 262144,
    "spark_cache_max_bytes": 32 * 1024**3,
    "spark_cache_low_watermark_bytes": 28 * 1024**3,
}


@pytest.mark.parametrize("profile", installer_image.SPARKCACHE_CRC32_VARIANTS)
def test_crc32_variant_preserves_prior_settings_except_integrity_root_and_restore_capacity(profile):
    root = installer_image.ROOT / "profiles"
    prior = json.loads((root / profile.replace("-crc32", "") / "config.json").read_text())
    variant = json.loads((root / profile / "config.json").read_text())
    variant["environment"]["SPARK_CONTEXT_CACHE_ROOT"] = prior["environment"]["SPARK_CONTEXT_CACHE_ROOT"]
    index = variant["vllm_args"].index("--kv-transfer-config") + 1
    settings = json.loads(variant["vllm_args"][index])
    previous = json.loads(prior["vllm_args"][index])
    assert settings["kv_connector_extra_config"].pop("spark_cache_payload_integrity") == "fast"
    settings["kv_connector_extra_config"]["spark_cache_root"] = previous["kv_connector_extra_config"]["spark_cache_root"]
    # Restore and capacity settings measured on the CRC32 layer: eight CUDA
    # restore I/O workers and a 256K-token span bounded by 32 GiB per rank.
    for key, value in CRC32_TUNED_SETTINGS.items():
        assert settings["kv_connector_extra_config"].pop(key) == value, key
        previous["kv_connector_extra_config"].pop(key)
    assert settings == previous
    variant["vllm_args"][index] = prior["vllm_args"][index]
    assert variant == prior


def test_deepseek_identity_matches_checkpoint_manifest_and_embedded_draft():
    import hashlib
    value = json.loads((installer_image.ROOT / "profiles" / DEEPSEEK / "config.json").read_text())
    vendor = installer_image.ROOT / "integrations/sparkcache/4a17f1f5198e/deploy/deepseek_v41"
    receipt = json.loads((vendor / "checkpoint-identity-receipt.json").read_text())
    inputs = receipt["inputs"]
    assert inputs == json.loads((vendor / "checkpoint-identity-inputs.json").read_text())
    sums = (installer_image.ROOT / "profiles" / DEEPSEEK / "SHA256SUMS").read_bytes()
    assert sums == (vendor / "checkpoint-sha256sums.txt").read_bytes()
    assert hashlib.sha256(sums).hexdigest() == inputs["artifacts"]["manifest_sha256"]
    assert (inputs["repository"], inputs["revision"]) == (value["model"]["repository"], value["model"]["revision"])
    assert inputs["model_config"]["sha256"] == value["model"]["config_sha256"]
    args = value["vllm_args"]
    draft = json.loads(args[args.index("--speculative-config") + 1])
    for backend in ("attention_backend", "moe_backend"):
        draft[backend] = args[args.index("--" + backend.replace("_", "-")) + 1]
    assert inputs["draft"] == {"mode": "embedded", "config": draft}
    envelope = {"schema": receipt["schema"], "inputs": inputs}
    digest = hashlib.sha256(json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
    assert digest == receipt["checkpoint_sha256"] == checkpoint_identity(DEEPSEEK)
    assert DEEPSEEK not in installer_image.QWEN4_EXP
    assert not (installer_image.ROOT / "profiles" / DEEPSEEK / "compose").exists()
