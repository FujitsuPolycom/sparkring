from __future__ import annotations

import json
from pathlib import Path

import yaml

from prepare_glm53_pr42_page_base_flight_profile import resolve


ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT
    / "scripts/config/glm53-flash-dflash7-pr42-page-base-flight-"
    "fastsafetensors-sparkcache-tp4-dcp1.example.json"
)
SITE = ROOT / "scripts/config/glm53-flash-tp4-site.example.yaml"
DIGESTS = {
    "cuda_placement_library_sha256": "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c",
    "native_elf_manifest_sha256": "2b" * 32,
    "native_dispatch_manifest_sha256": "3c" * 32,
    "source_receipt_sha256": "4d" * 32,
}


def _argument(profile: dict, option: str) -> str:
    args = profile["extra_vllm_args"]
    return args[args.index(option) + 1]


def test_pr42_profile_is_operationally_isolated_without_identity_geometry_change() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert profile["image"] == (
        "sparkring-glm53-sparkcache:"
        "dflash7-pr42-page-base-flight-singletonfix-arm64"
    )
    assert profile["image_id"] == (
        "sha256:35b58a7bf414059c65b8f74e4e4b17ee6a81b7008e1bffbc9bd298b5e08c739e"
    )
    assert profile["profile_id"].startswith("glm53-flash-dflash7-pr42-page-base-flight")
    assert "pr42-page-base-flight" in profile["container_name"]
    assert "pr42-page-base-flight" in _argument(profile, "--served-model-name")
    transfer = json.loads(_argument(profile, "--kv-transfer-config"))
    extra = transfer["kv_connector_extra_config"]
    assert extra["spark_cache_root"].endswith("dflash7-pr42-page-base-flight")
    assert extra["spark_cache_clear_once"] == (
        "sparkring-dflash7-pr42-page-base-flight-a1511d26-singleton"
    )
    assert extra["spark_cache_publication_schema"] == "tail-cow-v1"
    assert extra["spark_cache_model_profile"] == "glm53-flash-hybrid"
    assert profile["identity"]["target_cache_identity_sha256"] == (
        "a35e6bf2875c1875609b8deaec404c07c6cc80259e4222fc0b51e649498bd6b9"
    )
    assert profile["identity"]["draft_weights_sha256"] == (
        "b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b"
    )


def test_pr42_profile_resolves_exact_source_and_feature_labels() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    site = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    profile["model_host_path"] = "/srv/models/glm53"
    profile["extra_volumes"][0]["host"] = "/srv/models/dflash"
    profile["extra_volumes"][1]["host"] = "/srv/cache/pr42-page-base-flight"
    resolved, resolved_site = resolve(
        profile,
        site,
        image="local/pr42-page-base-flight@sha256:" + "a" * 64,
        image_id="sha256:" + "b" * 64,
        **DIGESTS,
    )
    assert resolved_site["runtime"]["container_image"] == resolved["image"]
    assert resolved["identity"]["sparkcache_source_revision"] == (
        "a1511d26a1fe2b17b24561bc52e376bf7f54b06a"
    )
    labels = resolved["required_image_labels"]
    assert labels["org.sparkcache.deployment-profile"] == (
        "glm53-flash-dflash7-python-overlay"
    )
    assert labels["org.sparkcache.feature.page-base-read-flight"] == (
        "implemented-gpu-free-tested"
    )
    assert labels["org.sparkcache.feature.page-base-read-flight-pr"] == "42"
    assert labels["org.sparkcache.diagnostic-fix"].startswith(
        "page-header-source-bytes-fix=229d7d6"
    )
    assert labels["org.sparkcache.page-header-source-bytes-fix"] == "229d7d6"
    assert labels[
        "org.sparkcache.page-base-read-flight-singleton-later-cohorts"
    ] == "a1511d26a1fe2b17b24561bc52e376bf7f54b06a"
    assert labels["org.sparkcache.cuda-placement-library-sha256"] == (
        DIGESTS["cuda_placement_library_sha256"]
    )
    assert "REPLACE_WITH" not in json.dumps(resolved)
