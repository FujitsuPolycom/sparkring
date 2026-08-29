from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

import sparkring_generic_launcher as launcher
from prepare_glm53_e105_profile import ResolveError, resolve
from sparkring_site import load_site


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/config"
SITE = CONFIG / "glm53-flash-e10536a-tp4-site.example.yaml"
PROFILES = {
    "dflash": CONFIG / "glm53-flash-e10536a-dflash2-bf16-sparkcache-tp4-dcp1.example.json",
    "mtp_static": CONFIG / "glm53-flash-e10536a-mtp5-sparkcache-tp4-dcp1.example.json",
    "mtp_adaptive": CONFIG / "glm53-flash-e10536a-mtp5-adaptive-sparkcache-tp4-dcp1.example.json",
}
TARGET = "a35e6bf2875c1875609b8deaec404c07c6cc80259e4222fc0b51e649498bd6b9"


def _argument(profile: dict, option: str) -> str:
    args = profile["extra_vllm_args"]
    return args[args.index(option) + 1]


def _mtp_identity(policy: str) -> str:
    value = f"glm53-embedded-mtp-v1\0{TARGET}\0{5}\0{policy}".encode()
    return hashlib.sha256(value).hexdigest()


def test_profiles_separate_runtime_upgrade_from_speculator_changes() -> None:
    profiles = {name: json.loads(path.read_text()) for name, path in PROFILES.items()}
    configs = {
        name: json.loads(_argument(profile, "--speculative-config"))
        for name, profile in profiles.items()
    }
    assert configs["dflash"]["method"] == "dflash"
    assert configs["dflash"]["num_speculative_tokens"] == 5
    assert configs["mtp_static"] == {
        "method": "mtp",
        "num_speculative_tokens": 5,
        "moe_backend": "humming",
        "attention_backend": "B12X",
    }
    assert configs["mtp_adaptive"]["adaptive_speculative_tokens_initial"] == 3
    assert configs["mtp_adaptive"]["adaptive_speculative_tokens_window"] == 32


def test_embedded_mtp_identities_are_derived_and_distinct() -> None:
    static = json.loads(PROFILES["mtp_static"].read_text())
    adaptive = json.loads(PROFILES["mtp_adaptive"].read_text())
    assert static["identity"]["mtp_cache_identity_sha256"] == _mtp_identity("static")
    assert adaptive["identity"]["mtp_cache_identity_sha256"] == _mtp_identity(
        "adaptive:3:32"
    )
    identities = set()
    clear_tokens = set()
    for path in PROFILES.values():
        profile = json.loads(path.read_text())
        transfer = json.loads(_argument(profile, "--kv-transfer-config"))
        extra = transfer["kv_connector_extra_config"]
        identities.add(extra["spark_cache_draft_checkpoint_sha256"])
        clear_tokens.add(extra["spark_cache_clear_once"])
        assert extra["spark_cache_native_restore"] is True
        assert extra["spark_cache_native_arena_bytes"] == 256 * 1024**2
        assert extra["spark_cache_native_io_workers"] == 8
        assert extra["spark_cache_load_threads"] == 2
    assert len(identities) == len(clear_tokens) == 3


def test_resolver_produces_aligned_twenty_gib_profile() -> None:
    profile = json.loads(PROFILES["dflash"].read_text())
    site = yaml.safe_load(SITE.read_text())
    resolved_profile, resolved_site = resolve(
        copy.deepcopy(profile),
        copy.deepcopy(site),
        image="local/e105-sparkcache@sha256:" + "a" * 64,
        image_id="sha256:" + "b" * 64,
        parent_image="local/e105-runtime@sha256:" + "c" * 64,
        parent_image_id="sha256:" + "d" * 64,
        native_library_sha256="e" * 64,
    )
    assert resolved_site["runtime"]["container_image"] == resolved_profile["image"]
    assert resolved_site["runtime"]["container_image_digest"] == resolved_profile["image_id"]
    assert resolved_site["serving"]["kv_cache_bytes_per_rank"] == 20 * 1024**3
    assert "REPLACE_WITH" not in json.dumps(resolved_profile)


@pytest.mark.parametrize("profile_path", PROFILES.values(), ids=PROFILES.keys())
def test_each_resolved_profile_passes_site_alignment(
    profile_path: Path, tmp_path: Path
) -> None:
    profile = json.loads(profile_path.read_text())
    site = yaml.safe_load(SITE.read_text())
    profile, site = resolve(
        profile,
        site,
        image="local/e105-sparkcache@sha256:" + "a" * 64,
        image_id="sha256:" + "b" * 64,
        parent_image="local/e105-runtime@sha256:" + "c" * 64,
        parent_image_id="sha256:" + "d" * 64,
        native_library_sha256="e" * 64,
    )
    profile_path_out = tmp_path / "profile.json"
    site_path_out = tmp_path / "site.yaml"
    profile_path_out.write_text(json.dumps(profile), encoding="utf-8")
    site_path_out.write_text(yaml.safe_dump(site, sort_keys=False), encoding="utf-8")
    loaded_profile = launcher.load_profile(profile_path_out)
    loaded_site = load_site(site_path_out)
    launcher._validate_site_profile_alignment(loaded_site, loaded_profile)
    actions = launcher.build_actions(loaded_site, loaded_profile, "plan")
    assert len(actions) == 4


def test_resolver_rejects_unverified_native_library() -> None:
    profile = json.loads(PROFILES["dflash"].read_text())
    site = yaml.safe_load(SITE.read_text())
    with pytest.raises(ResolveError, match="native library"):
        resolve(
            profile, site,
            image="image",
            image_id="sha256:" + "a" * 64,
            parent_image="parent",
            parent_image_id="sha256:" + "b" * 64,
            native_library_sha256="short",
        )
