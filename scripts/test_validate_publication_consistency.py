from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_publication_consistency as publication  # noqa: E402


@pytest.fixture()
def publication_tree(tmp_path: Path) -> Path:
    for directory in ("docs", "recipes"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    for filename in ("README.md", "CONTRIBUTING.md"):
        shutil.copy2(ROOT / filename, tmp_path / filename)
    return tmp_path


def _load(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _write(root: Path, relative: str, document: dict) -> None:
    (root / relative).write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )


def _messages(root: Path) -> list[str]:
    return [problem.render() for problem in publication.validate(root)]


def test_repository_publication_is_consistent() -> None:
    assert publication.validate(ROOT) == []


def test_missing_evidence_reports_source_pointer_and_target(
    publication_tree: Path,
) -> None:
    status = _load(publication_tree, "docs/STATUS.json")
    status["components"]["exl3_r7_mtp4_nvfp4_ckv_operator_default"][
        "machine_readable_evidence"
    ] = "docs/configurations/missing.json"
    _write(publication_tree, "docs/STATUS.json", status)

    messages = _messages(publication_tree)
    assert any("docs/STATUS.json#/components/" in message for message in messages)
    assert any(
        "referenced file does not exist: docs/configurations/missing.json" in message
        for message in messages
    )


def test_missing_required_recipe_is_rejected(publication_tree: Path) -> None:
    recipe_id = "glm52-exl3-r7-3.5bpw"
    (publication_tree / f"recipes/{recipe_id}.json").unlink()

    assert any(
        message
        == f"ERROR: recipes#/{recipe_id}: required public configuration recipe is missing or invalid"
        for message in _messages(publication_tree)
    )


def test_two_defaults_and_status_pointer_drift_are_aggregated(
    publication_tree: Path,
) -> None:
    nf3_path = "recipes/glm52-nf3-hybrid.json"
    nf3 = _load(publication_tree, nf3_path)
    nf3["default"] = True
    _write(publication_tree, nf3_path, nf3)
    status = _load(publication_tree, "docs/STATUS.json")
    status["lanes"]["public-functional"]["default_configuration"] = nf3_path
    _write(publication_tree, "docs/STATUS.json", status)

    messages = _messages(publication_tree)
    assert any(
        "expected exactly one default recipe; found 2" in message
        for message in messages
    )
    assert any("expected a non-default recipe" in message for message in messages)


def test_public_default_and_accepted_alternative_identities_are_fixed(
    publication_tree: Path,
) -> None:
    default_path = "recipes/glm52-exl3-tr3-3.25bpw.json"
    r7_path = "recipes/glm52-exl3-r7-3.5bpw.json"
    default = _load(publication_tree, default_path)
    default["default"] = False
    _write(publication_tree, default_path, default)
    r7 = _load(publication_tree, r7_path)
    r7["default"] = True
    _write(publication_tree, r7_path, r7)
    status = _load(publication_tree, "docs/STATUS.json")
    lane = status["lanes"]["public-functional"]
    lane["definition"] = r7_path
    lane["default_configuration"] = r7_path
    lane["accepted_alternative"] = r7_path
    status["components"]["exl3_recipe"]["default"] = False
    _write(publication_tree, "docs/STATUS.json", status)

    messages = _messages(publication_tree)
    assert any("to be the public default" in message for message in messages)
    assert any("expected accepted NF3 alternative" in message for message in messages)


def test_recipe_maturity_and_operator_role_drift_fail(
    publication_tree: Path,
) -> None:
    exl3_path = "recipes/glm52-exl3-tr3-3.25bpw.json"
    exl3 = _load(publication_tree, exl3_path)
    exl3["maturity"] = "accepted"
    _write(publication_tree, exl3_path, exl3)
    status = _load(publication_tree, "docs/STATUS.json")
    r7 = status["components"]["exl3_r7_mtp4_nvfp4_ckv_operator_default"]
    r7["public_functional_default"] = True
    _write(publication_tree, "docs/STATUS.json", status)

    messages = _messages(publication_tree)
    assert any(
        "recipes/glm52-exl3-tr3-3.25bpw.json#/maturity" in message
        for message in messages
    )
    assert any("operator acceptance must not promote" in message for message in messages)


def test_required_status_roles_and_lane_evidence_are_checked(
    publication_tree: Path,
) -> None:
    status = _load(publication_tree, "docs/STATUS.json")
    status["components"]["exl3_recipe"]["main_advertised_configuration"] = False
    status["components"]["exl3_r7_mtp4_nvfp4_ckv_operator_default"]["accepted"] = False
    del status["lanes"]["reference"]["evidence"]
    status["lanes"]["public-functional"]["evidence"] = "docs/RESULTS.md"
    _write(publication_tree, "docs/STATUS.json", status)

    messages = _messages(publication_tree)
    assert any("main_advertised_configuration" in message for message in messages)
    assert any("operator_default/accepted" in message for message in messages)
    assert any("#/lanes/reference/evidence" in message for message in messages)
    assert any("#/lanes/public-functional/evidence" in message for message in messages)


def test_coordinated_role_drift_cannot_redefine_public_contract(
    publication_tree: Path,
) -> None:
    default_path = "recipes/glm52-exl3-tr3-3.25bpw.json"
    default = _load(publication_tree, default_path)
    default["maturity"] = "accepted"
    _write(publication_tree, default_path, default)
    status = _load(publication_tree, "docs/STATUS.json")
    status["components"]["exl3_recipe"]["maturity"] = "accepted"
    r7 = status["components"]["exl3_r7_mtp4_nvfp4_ckv_operator_default"]
    r7["maturity"] = "candidate"
    r7["operator_default"] = False
    _write(publication_tree, "docs/STATUS.json", status)

    messages = _messages(publication_tree)
    assert any("#/components/exl3_recipe/maturity" in message for message in messages)
    assert any("operator_default/maturity" in message for message in messages)
    assert any("operator_default/operator_default" in message for message in messages)


def test_accepted_snapshot_evidence_references_are_required(
    publication_tree: Path,
) -> None:
    status = _load(publication_tree, "docs/STATUS.json")
    component = status["components"]["exl3_r7_mtp4_nvfp4_ckv_operator_default"]
    del component["accepted_prefill_snapshot"]["evidence"]
    del component["accepted_decode_snapshot"]["evidence"]
    del component["accepted_coding_snapshot"]["evidence"]
    _write(publication_tree, "docs/STATUS.json", status)

    messages = _messages(publication_tree)
    assert sum("required evidence reference is missing" in message for message in messages) == 3


def test_r7_status_policy_is_projected_from_recipe(publication_tree: Path) -> None:
    status = _load(publication_tree, "docs/STATUS.json")
    policy = status["components"][
        "exl3_r7_mtp4_nvfp4_ckv_operator_default"
    ]["exact_q40_policy"]
    policy["capacity_rows"] = 41
    policy["route_block_rows"] = 4
    _write(publication_tree, "docs/STATUS.json", status)

    messages = _messages(publication_tree)
    assert any("exact_q40_policy/capacity_rows" in message for message in messages)
    assert any("exact_q40_policy/route_block_rows" in message for message in messages)


def test_volatile_test_totals_are_rejected(publication_tree: Path) -> None:
    status = _load(publication_tree, "docs/STATUS.json")
    status["components"]["offline_python_contracts"]["last_observed"] = {
        "passed": 9999,
        "skipped": 1,
    }
    _write(publication_tree, "docs/STATUS.json", status)
    contributing = publication_tree / "CONTRIBUTING.md"
    prose = contributing.read_text(encoding="utf-8")
    prose = prose.replace(
        "| **All five (what CI runs)**",
        "| 9999 passed, 1 skipped\n| **All five (what CI runs)**",
    )
    contributing.write_text(prose, encoding="utf-8")

    messages = _messages(publication_tree)
    assert any("volatile test totals" in message for message in messages)
    assert any("volatile pytest summary" in message for message in messages)


@pytest.mark.parametrize(
    "relative",
    ["docs/STATUS.json", "recipes/glm52-exl3-tr3-3.25bpw.json"],
)
def test_malformed_json_reports_location_without_traceback(
    publication_tree: Path, relative: str
) -> None:
    (publication_tree / relative).write_text('{"broken":\n', encoding="utf-8")

    messages = _messages(publication_tree)
    matching = [
        message for message in messages if message.startswith(f"ERROR: {relative}:")
    ]
    assert matching
    assert "line 2, column 1" in matching[0]
    assert "Traceback" not in "\n".join(messages)


@pytest.mark.parametrize(
    "relative", ["docs/STATUS.json", "docs/profiles/README.md"]
)
def test_invalid_utf8_reports_error_without_traceback(
    publication_tree: Path, relative: str
) -> None:
    (publication_tree / relative).write_bytes(b"\xff\xfe")

    messages = _messages(publication_tree)
    assert any(relative in message and "UTF-8" in message for message in messages)
    assert "Traceback" not in "\n".join(messages)


def test_profile_registry_must_project_recipe_model_identities(
    publication_tree: Path,
) -> None:
    registry = publication_tree / "docs/profiles/README.md"
    prose = registry.read_text(encoding="utf-8").replace(
        "d7d79c2d14599dfce7a5d12b85f7ad73f40e623d",
        "missing-default-revision",
    )
    registry.write_text(prose, encoding="utf-8")

    assert any(
        "docs/profiles/README.md: missing glm52-exl3-tr3-3.25bpw model revision"
        in message
        for message in _messages(publication_tree)
    )


def test_default_recipe_doc_must_project_validated_image_identity(
    publication_tree: Path,
) -> None:
    recipe_doc = publication_tree / "docs/EXL3_RECIPE.md"
    prose = recipe_doc.read_text(encoding="utf-8").replace(
        "sha256:20c4099f2e7e3dd3c8ab64f7d7930bde4f372df1895aa3ffa593252ca04ae96f",
        "missing-image-identity",
    )
    recipe_doc.write_text(prose, encoding="utf-8")

    assert any(
        "docs/EXL3_RECIPE.md: missing glm52-exl3-tr3-3.25bpw "
        "validated image identity"
        in message
        for message in _messages(publication_tree)
    )


def test_problems_are_deterministic_and_deduplicated(publication_tree: Path) -> None:
    status = _load(publication_tree, "docs/STATUS.json")
    status["lanes"]["public-functional"]["definition"] = "recipes/missing.json"
    status["lanes"]["public-functional"]["default_configuration"] = (
        "recipes/missing.json"
    )
    _write(publication_tree, "docs/STATUS.json", status)

    first = publication.validate(publication_tree)
    second = publication.validate(publication_tree)
    assert first == second
    assert first == sorted(set(first))


def test_unsafe_reference_path_is_rejected(publication_tree: Path) -> None:
    r7_path = "recipes/glm52-exl3-r7-3.5bpw.json"
    recipe = _load(publication_tree, r7_path)
    recipe["publication"]["promotion_checklist"] = "../private.md"
    _write(publication_tree, r7_path, recipe)

    assert any(
        "unsafe repository-relative path '../private.md'" in message
        for message in _messages(publication_tree)
    )


def test_r7_receipt_is_not_loaded_through_unsafe_path(publication_tree: Path) -> None:
    r7_path = "recipes/glm52-exl3-r7-3.5bpw.json"
    recipe = _load(publication_tree, r7_path)
    recipe["publication"]["machine_readable_evidence"] = "../outside.json"
    _write(publication_tree, r7_path, recipe)
    (publication_tree.parent / "outside.json").write_text("{}\n", encoding="utf-8")

    messages = _messages(publication_tree)
    assert any(
        "unsafe repository-relative path '../outside.json'" in message
        for message in messages
    )
    assert "Traceback" not in "\n".join(messages)


def test_incomplete_recipe_reports_shape_error_without_traceback(
    publication_tree: Path,
) -> None:
    recipe_path = "recipes/glm52-exl3-tr3-3.25bpw.json"
    recipe = _load(publication_tree, recipe_path)
    del recipe["model"]
    _write(publication_tree, recipe_path, recipe)

    messages = _messages(publication_tree)
    assert any(
        message == f"ERROR: {recipe_path}#/model: expected an object"
        for message in messages
    )
    assert "Traceback" not in "\n".join(messages)


def test_incomplete_runtime_reports_error_without_traceback(
    publication_tree: Path,
) -> None:
    recipe_path = "recipes/glm52-exl3-tr3-3.25bpw.json"
    recipe = _load(publication_tree, recipe_path)
    recipe["runtime"] = {}
    _write(publication_tree, recipe_path, recipe)

    messages = _messages(publication_tree)
    assert any("cannot project non-string" in message for message in messages)
    assert "Traceback" not in "\n".join(messages)


def test_results_remains_canonical_for_clean_checkout_receipt(
    publication_tree: Path,
) -> None:
    recipe_path = "recipes/glm52-exl3-tr3-3.25bpw.json"
    recipe = _load(publication_tree, recipe_path)
    recipe["publication"]["image_source_commit"] = "0" * 40
    recipe["publication"]["launcher_fix_commit"] = "deadbeef"
    _write(publication_tree, recipe_path, recipe)
    status = _load(publication_tree, "docs/STATUS.json")
    receipt = status["components"]["exl3_recipe"]["clean_checkout_receipt"]
    receipt["image_source_commit"] = "0" * 40
    receipt["launcher_fix_commit"] = "deadbeef"
    _write(publication_tree, "docs/STATUS.json", status)

    messages = _messages(publication_tree)
    assert any("canonical image source commit" in message for message in messages)
    assert any("canonical launcher correction commit" in message for message in messages)


def test_canonical_lane_statement_cannot_disappear(publication_tree: Path) -> None:
    lane_path = publication_tree / "docs/PUBLIC_FUNCTIONAL_TARGET.md"
    prose = lane_path.read_text(encoding="utf-8").replace(
        "EXL3 plus\n> LMCache is the default and main advertised configuration",
        "EXL3 plus\n> LMCache is an available configuration",
    )
    lane_path.write_text(prose, encoding="utf-8")

    assert any(
        "missing canonical lane statement" in message
        for message in _messages(publication_tree)
    )


def test_cli_contract_reports_success_and_failure(
    publication_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert publication.main(["--root", str(publication_tree)]) == 0
    assert capsys.readouterr().out == "publication consistency: PASS\n"

    (publication_tree / "docs/STATUS.json").write_text("{", encoding="utf-8")
    assert publication.main(["--root", str(publication_tree)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "publication consistency: FAIL" in captured.err
