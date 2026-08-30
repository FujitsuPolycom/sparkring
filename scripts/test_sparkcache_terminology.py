"""GPU-free contracts for SparkCache CUDA configuration terminology."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparkcache_terminology import (
    CANONICAL_CONNECTOR_KEYS,
    SparkCacheTerminologyError,
    canonicalize_connector_arguments,
    canonicalize_connector_extra_config,
    resolve_string_alias,
)


ROOT = Path(__file__).resolve().parents[1]


def test_every_connector_alias_normalizes_to_its_canonical_key() -> None:
    legacy = {
        old: index
        for index, old in enumerate(CANONICAL_CONNECTOR_KEYS.values(), start=1)
    }

    normalized = canonicalize_connector_extra_config(legacy)

    assert normalized == {
        canonical: index
        for index, canonical in enumerate(CANONICAL_CONNECTOR_KEYS, start=1)
    }


def test_connector_alias_conflict_is_rejected_even_across_json_types() -> None:
    with pytest.raises(SparkCacheTerminologyError, match="conflicting values"):
        canonicalize_connector_extra_config(
            {
                "spark_cache_cuda_restore": True,
                "spark_cache_native_restore": 1,
            }
        )


def test_runtime_argument_normalizes_legacy_connector_json() -> None:
    transfer = {
        "kv_connector": "SparkContextCacheConnector",
        "kv_connector_extra_config": {
            "spark_cache_native_restore": True,
            "spark_cache_native_library": "/opt/lib/libspark_cache_placement.so",
        },
    }

    normalized = canonicalize_connector_arguments(
        ("--kv-transfer-config", json.dumps(transfer))
    )
    extra = json.loads(normalized[1])["kv_connector_extra_config"]

    assert extra == {
        "spark_cache_cuda_restore": True,
        "spark_cache_cuda_placement_library": (
            "/opt/lib/libspark_cache_placement.so"
        ),
    }


def test_runtime_argument_rejects_conflicting_connector_json() -> None:
    transfer = {
        "kv_connector_extra_config": {
            "spark_cache_cuda_restore": True,
            "spark_cache_native_restore": False,
        }
    }
    with pytest.raises(SparkCacheTerminologyError, match="conflicting values"):
        canonicalize_connector_arguments(
            ("--kv-transfer-config=" + json.dumps(transfer),)
        )


def test_cli_alias_accepts_one_name_or_equal_values_and_rejects_conflicts() -> None:
    assert (
        resolve_string_alias(
            "a" * 64,
            None,
            canonical_name="--cuda-placement-library-sha256",
            legacy_name="--native-library-sha256",
        )
        == "a" * 64
    )
    assert (
        resolve_string_alias(
            "b" * 64,
            "b" * 64,
            canonical_name="--cuda-placement-library-sha256",
            legacy_name="--native-library-sha256",
        )
        == "b" * 64
    )
    with pytest.raises(SparkCacheTerminologyError, match="conflicting values"):
        resolve_string_alias(
            "a" * 64,
            "b" * 64,
            canonical_name="--cuda-placement-library-sha256",
            legacy_name="--native-library-sha256",
        )


def test_generated_profiles_emit_only_canonical_connector_and_label_names() -> None:
    forbidden = (
        "spark_cache_native_restore",
        "spark_cache_native_library",
        "spark_cache_native_library_sha256",
        "spark_cache_native_arena_bytes",
        "spark_cache_native_io_workers",
        "REPLACE_WITH_NATIVE_LIBRARY_SHA256",
        "org.sparkcache.native-library-sha256",
    )
    profiles = tuple((ROOT / "scripts/config").glob("*sparkcache*.json"))
    assert profiles
    for path in profiles:
        text = path.read_text(encoding="utf-8")
        assert not any(name in text for name in forbidden), path.name


def test_operator_prose_uses_sparkcache_cuda_restore_and_placement_names() -> None:
    paths = (
        ROOT / "docs/GLM53_B12X_KDA_ADAPTIVE_MTP_SPARKCACHE_TP4_QUICKSTART.md",
        ROOT / "docs/GLM53_E10536A_SPARKCACHE_TP4_QUICKSTART.md",
        ROOT / "recipes/sparkcache/README.md",
        ROOT / "runtime/glm53-flash-adaptive-mtp-python-overlay/README.md",
        ROOT / "runtime/glm53-flash-b12x-kda-adaptive-mtp/README.md",
        ROOT / "scripts/config/README.md",
    )
    forbidden = (
        "sparkcache native",
        "native restore",
        "native placement",
        "native direct restore",
        "native page placement",
        "--native-library-sha256",
    )
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    assert "sparkcache cuda restore" in text
    assert "sparkcache cuda placement" in text
    assert not any(phrase in text for phrase in forbidden)
