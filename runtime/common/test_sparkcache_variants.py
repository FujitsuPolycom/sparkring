"""The opt-in SparkCache installer-profile variants of the GLM-5.3-Flash profiles.

The variants enable the SparkContextCacheConnector for the CSF checkpoint, keep
every speed setting of their cache-off profile, and run only on an image whose
lock lists them: no lock of this package does, only the lock the derived layer
runtime/images/compositions/installer-sparkcache-gab86b70 records does.
"""

import json
from pathlib import Path

import pytest

from runtime.common import installer, installer_image, toolchain_profiles
from runtime.common.test_installer import site
from runtime.host import check
from runtime.host.storage import CACHE_NAME

VARIANTS = installer_image.SPARKCACHE_VARIANTS
BASES = {"glm53-flash-nvfp4-sparkcache-tp2": "glm53-flash-nvfp4-spark-tp2",
         "glm53-flash-nvfp4-sparkcache-tp4": "glm53-flash-nvfp4-spark-tp4"}
CACHE_ROOT = "/cache/glm53-flash-sparkcache-ab53c097d7ca-dec48abd33ef"
CHECKPOINT_IDENTITY = "6af0343976eefbfa0fb59d87befcf6d10f02fef90ba0e7881dd1201fe0ecd76e"
SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"
CONTRACT = SITE_PACKAGES + "/sparkcache/runtime_patches/vllm-connector-jobs-gab86b70.json"


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
    base_options = {argument for argument in base["vllm_args"] if argument.startswith("--")}
    variant_options = {argument for argument in variant["vllm_args"] if argument.startswith("--")}
    # The connector is the only option the variant adds; it removes none.
    assert variant_options - base_options == {"--kv-transfer-config"}
    assert "--async-scheduling" in variant["vllm_args"]
    assert "--enable-prefix-caching" in variant["vllm_args"]
    for option in ("--kda-prefill-backend", "--attention-backend", "--linear-backend", "--moe-backend"):
        assert variant["vllm_args"][variant["vllm_args"].index(option) + 1] == \
            base["vllm_args"][base["vllm_args"].index(option) + 1]
    # The CSF checkpoint's loader and draft settings replace the base entry's.
    assert variant["vllm_args"][variant["vllm_args"].index("--quantization") + 1] == "nvfp4_csf"
    assert variant["vllm_args"][variant["vllm_args"].index("--load-format") + 1] == "nvfp4_csf"
    draft = json.loads(variant["vllm_args"][variant["vllm_args"].index("--speculative-config") + 1])
    assert draft["num_speculative_tokens"] == 3 and draft["moe_backend"] == "marlin"
    assert draft["draft_sample_method"] == "probabilistic" and draft["rejection_sample_method"] == "standard"
    environment = dict(base["environment"])
    environment.update(variant["environment"])
    assert environment == variant["environment"] or set(variant["environment"]) <= set(environment)


@pytest.mark.parametrize("profile", VARIANTS)
def test_variant_renders_the_connector_settings_and_storage_paths(profile):
    lock, spec, connector_json = connector(profile)
    assert spec.environment["SPARKCACHE_ENABLED"] == "1"
    assert spec.environment["SPARK_CONTEXT_CACHE_ROOT"] == CACHE_ROOT
    # The root matches the name `sparkring storage` classifies as a cache
    # directory: <family>-<image 12>-<revision 12>.
    assert CACHE_NAME.fullmatch(Path(CACHE_ROOT).name)
    assert connector_json["kv_connector"] == "SparkContextCacheConnector"
    assert connector_json["kv_connector_module_path"] == "sparkcache.spark_context_cache_connector"
    assert connector_json["kv_role"] == "kv_both"
    assert connector_json["kv_load_failure_policy"] == "recompute"
    extra = connector_json["kv_connector_extra_config"]
    assert extra["spark_cache_model_profile"] == "glm53-flash-hybrid"
    assert extra["spark_cache_publication_schema"] == "tail-cow-v2"
    assert extra["spark_cache_draft_policy"] == "separate"
    assert extra["spark_cache_target_checkpoint_sha256"] == CHECKPOINT_IDENTITY
    assert extra["spark_cache_draft_checkpoint_sha256"] == CHECKPOINT_IDENTITY
    assert extra["spark_cache_max_span_tokens"] == 65536
    assert extra["spark_cache_async_page_capture"] is True
    assert extra["spark_cache_async_page_capture_lease_mode"] == "connector-jobs"
    assert extra["spark_cache_async_page_capture_lease_contract"] == CONTRACT
    assert extra["spark_cache_async_page_capture_vllm_root"] == SITE_PACKAGES
    assert extra["spark_cache_cuda_restore"] is True
    assert extra["spark_cache_cuda_placement_library"] == \
        "/opt/sparkring/sparkcache/lib/libspark_cache_placement.so"
    assert extra["spark_cache_async_page_capture_library"] == \
        "/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so"
    assert extra["spark_cache_root"] == CACHE_ROOT


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
    assert state["model_profile"] == "glm53-flash-hybrid"
    assert state["lease_mode"] == "connector-jobs"
    assert state["lease_contract"] == CONTRACT
    assert state["cache_root"] == CACHE_ROOT
    assert state["checkpoint_identity"].startswith("6af0343976ee")


def test_check_reports_no_connector_state_for_a_cache_off_profile():
    lock_value = installer.make_lock("glm53-flash-nvfp4-spark-tp2", ring_site(2), "1" * 40, "2" * 64,
                                     image_runtime=installer_image.default_lock())
    assert check.connector_state(lock_value) is None


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
