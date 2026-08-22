"""GPU-free contract tests for the qualified SparkCache compositions."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = ROOT / "recipes" / "sparkcache"
RECIPE_PATHS = sorted(RECIPE_DIR.glob("*.json"))
WHEEL_SHA256 = (
    "87c17d8dab5052f5a7833349dc9b99b76a3b6531ca6f0d3deff812f724fecdcc"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_composition_registry_has_all_qualified_profiles() -> None:
    assert [path.name for path in RECIPE_PATHS] == [
        "deepseek-v4-flash-0731-tp2-dcp1.json",
        "deepseek-v4-flash-0731-tp4-dcp1.json",
        "glm52-exl3-r7-3.5bpw-tp4-dcp4.json",
    ]


def test_compositions_pin_artifact_and_fail_closed_policy() -> None:
    for path in RECIPE_PATHS:
        recipe = _load(path)
        assert recipe["schema"] == "sparkring-sparkcache-composition/v1"
        assert recipe["status"] == "qualified"
        assert (path.parent / recipe["base_recipe"]).resolve().is_file()
        assert recipe["runtime"]["sparkcache"]["version"] == "0.1.0a1"
        assert recipe["runtime"]["sparkcache"]["wheel_sha256"] == WHEEL_SHA256
        assert recipe["serving"]["max_num_batched_tokens"] == 4096
        assert recipe["serving"]["scheduler_budget_status"] == "qualified"
        assert recipe["sparkcache"]["kv_load_failure_policy"] == "recompute"
        assert recipe["sparkcache"]["streaming_snapshots"] is False
        assert recipe["sparkcache"]["native_restore"] is False


def test_unsupported_scheduler_budget_is_not_an_operator_setting() -> None:
    for path in RECIPE_PATHS:
        recipe = _load(path)
        assert 8192 not in recipe["serving"].values()
        assert any("8192 remains unsupported" in item for item in recipe["limitations"])


def test_parallelism_matches_physical_rank_count() -> None:
    for path in RECIPE_PATHS:
        recipe = _load(path)
        serving = recipe["serving"]
        assert serving["tensor_parallel_size"] == recipe["hardware"]["ranks"]
        assert serving["node_count"] == recipe["hardware"]["ranks"]
        assert serving["decode_context_parallel_size"] in (1, 4)
