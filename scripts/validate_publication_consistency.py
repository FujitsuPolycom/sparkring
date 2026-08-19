#!/usr/bin/env python3
"""Fail closed when SparkRing's public claim projections disagree.

The validator is offline and read-only. Recipes own executable configuration,
``docs/STATUS.json`` owns maturity and role assignment, and
``docs/RESULTS.md`` owns measured claims. These checks compare projections;
they do not select a replacement canonical source when sources disagree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_RECIPE_ID = "glm52-exl3-tr3-3.25bpw"
NF3_RECIPE_ID = "glm52-nf3-hybrid"
R7_RECIPE_ID = "glm52-exl3-r7-3.5bpw"
REQUIRED_RECIPE_IDS = (DEFAULT_RECIPE_ID, NF3_RECIPE_ID, R7_RECIPE_ID)
REQUIRED_COMPONENT_ROLES = {
    "/components/exl3_recipe/maturity": "live-validated",
    "/components/exl3_recipe/default": True,
    "/components/nf3_recipe/maturity": "accepted",
    "/components/nf3_recipe/default_public_functional_target": False,
    "/components/exl3_r7_mtp4_nvfp4_ckv_operator_default/maturity": "accepted",
    "/components/exl3_r7_mtp4_nvfp4_ckv_operator_default/accepted": True,
    "/components/exl3_r7_mtp4_nvfp4_ckv_operator_default/default": False,
    "/components/exl3_r7_mtp4_nvfp4_ckv_operator_default/operator_default": True,
    "/components/exl3_r7_mtp4_nvfp4_ckv_operator_default/public_functional_default": False,
}
RECIPE_SCHEMA = "sparkring-recipe/v1"
STATUS_SCHEMA = "sparkring-repository-status/v1"
PYTEST_SUMMARY_RE = re.compile(
    r"\b\d[\d,]*\s+passed\b(?:\s*,\s*\d[\d,]*\s+skipped\b)?",
    re.IGNORECASE,
)
_MISSING = object()


@dataclass(frozen=True, order=True)
class Problem:
    """One deterministic publication-consistency diagnostic."""

    source: str
    pointer: str
    message: str

    def render(self) -> str:
        location = self.source + (f"#{self.pointer}" if self.pointer else "")
        return f"ERROR: {location}: {self.message}"


class PublicationValidator:
    """Compare public claim projections through one offline interface."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.problems: list[Problem] = []
        self.status: dict[str, Any] | None = None
        self.recipes: dict[str, tuple[str, dict[str, Any]]] = {}

    def problem(self, source: str, pointer: str, message: str) -> None:
        self.problems.append(Problem(source, pointer, message))

    def load_object(
        self, relative: str, *, expected_schema: str | None = None
    ) -> dict[str, Any] | None:
        path = self.root / relative
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.problem(relative, "", "file does not exist")
            return None
        except json.JSONDecodeError as exc:
            self.problem(
                relative,
                "",
                f"malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            )
            return None
        except UnicodeDecodeError as exc:
            self.problem(relative, "", f"file is not valid UTF-8: {exc}")
            return None
        except OSError as exc:
            self.problem(relative, "", f"cannot read file: {exc}")
            return None
        if not isinstance(document, dict):
            self.problem(relative, "", "JSON root must be an object")
            return None
        if expected_schema is not None and document.get("schema") != expected_schema:
            self.problem(
                relative,
                "/schema",
                f"expected {expected_schema!r}; got {document.get('schema')!r}",
            )
        return document

    @staticmethod
    def value(document: dict[str, Any], pointer: str) -> Any:
        current: Any = document
        tokens = pointer.strip("/").split("/") if pointer != "/" else []
        for token in tokens:
            if not isinstance(current, dict) or token not in current:
                return _MISSING
            current = current[token]
        return current

    def equal(
        self,
        source: str,
        document: dict[str, Any],
        pointer: str,
        expected: Any,
    ) -> None:
        actual = self.value(document, pointer)
        if actual != expected:
            shown = "<missing>" if actual is _MISSING else repr(actual)
            self.problem(source, pointer, f"expected {expected!r}; got {shown}")

    def validate_reference(
        self,
        source: str,
        pointer: str,
        value: Any,
        *,
        parse_json: bool = False,
    ) -> bool:
        if not isinstance(value, str) or not value:
            self.problem(source, pointer, "expected a non-empty repository-relative path")
            return False
        posix = PurePosixPath(value)
        if posix.is_absolute() or ".." in posix.parts or "\\" in value:
            self.problem(source, pointer, f"unsafe repository-relative path {value!r}")
            return False
        target = (self.root / Path(*posix.parts)).resolve()
        try:
            target.relative_to(self.root)
        except ValueError:
            self.problem(source, pointer, f"path escapes repository root: {value!r}")
            return False
        if not target.is_file():
            self.problem(source, pointer, f"referenced file does not exist: {value}")
            return False
        if parse_json or target.suffix.lower() == ".json":
            try:
                parsed = json.loads(target.read_text(encoding="utf-8"))
            except UnicodeDecodeError as exc:
                self.problem(source, pointer, f"referenced JSON is not valid UTF-8: {value}: {exc}")
                return False
            except json.JSONDecodeError as exc:
                self.problem(
                    source,
                    pointer,
                    f"referenced JSON is malformed at line {exc.lineno}, "
                    f"column {exc.colno}: {value}",
                )
                return False
            except OSError as exc:
                self.problem(source, pointer, f"cannot read referenced JSON {value}: {exc}")
                return False
            if not isinstance(parsed, dict):
                self.problem(source, pointer, f"referenced JSON root is not an object: {value}")
                return False
        return True

    def load_inputs(self) -> None:
        self.status = self.load_object("docs/STATUS.json", expected_schema=STATUS_SCHEMA)
        recipes_dir = self.root / "recipes"
        if not recipes_dir.is_dir():
            self.problem("recipes", "", "recipe directory does not exist")
            return
        for path in sorted(recipes_dir.glob("*.json")):
            relative = path.relative_to(self.root).as_posix()
            recipe = self.load_object(relative, expected_schema=RECIPE_SCHEMA)
            if recipe is None:
                continue
            recipe_id = recipe.get("recipe_id")
            if not isinstance(recipe_id, str):
                self.problem(relative, "/recipe_id", "expected a string recipe identity")
                continue
            if recipe_id != path.stem:
                self.problem(
                    relative,
                    "/recipe_id",
                    f"expected filename identity {path.stem!r}; got {recipe_id!r}",
                )
            if recipe_id in self.recipes:
                self.problem(relative, "/recipe_id", f"duplicate recipe identity {recipe_id!r}")
            if not self.validate_recipe_shape(relative, recipe):
                continue
            self.recipes[recipe_id] = (relative, recipe)

    def validate_recipe_shape(self, source: str, recipe: dict[str, Any]) -> bool:
        """Reject incomplete recipe objects before projection checks run."""

        valid = True
        required_objects = ("hardware", "model", "runtime", "serving", "publication")
        for field in required_objects:
            if not isinstance(recipe.get(field), dict):
                self.problem(source, f"/{field}", "expected an object")
                valid = False
        if not isinstance(recipe.get("maturity"), str):
            self.problem(source, "/maturity", "expected a string")
            valid = False
        if not isinstance(recipe.get("default"), bool):
            self.problem(source, "/default", "expected a boolean")
            valid = False
        model = recipe.get("model")
        if isinstance(model, dict):
            for field in ("repository", "revision"):
                if not isinstance(model.get(field), str) or not model[field]:
                    self.problem(source, f"/model/{field}", "expected a non-empty string")
                    valid = False
        return valid

    def recipe_by_path(self, relative: Any) -> dict[str, Any] | None:
        if not isinstance(relative, str):
            return None
        for source, recipe in self.recipes.values():
            if source == relative:
                return recipe
        self.problem(
            "docs/STATUS.json",
            "/lanes/public-functional",
            f"references unknown recipe {relative!r}",
        )
        return None

    def validate_recipe_roles(self) -> None:
        for recipe_id in REQUIRED_RECIPE_IDS:
            if recipe_id not in self.recipes:
                self.problem(
                    "recipes",
                    f"/{recipe_id}",
                    "required public configuration recipe is missing or invalid",
                )
        if self.status is None:
            return
        for pointer, expected in REQUIRED_COMPONENT_ROLES.items():
            self.equal("docs/STATUS.json", self.status, pointer, expected)
        lane = self.value(self.status, "/lanes/public-functional")
        if not isinstance(lane, dict):
            self.problem(
                "docs/STATUS.json", "/lanes/public-functional", "expected an object"
            )
            return
        defaults = [
            (recipe_id, source)
            for recipe_id, (source, recipe) in self.recipes.items()
            if recipe.get("default") is True
        ]
        if len(defaults) != 1:
            self.problem(
                "recipes",
                "/default",
                f"expected exactly one default recipe; found {len(defaults)}",
            )
        else:
            default_id, default_path = defaults[0]
            if default_id != DEFAULT_RECIPE_ID:
                self.problem(
                    "recipes",
                    "/default",
                    f"expected {DEFAULT_RECIPE_ID!r} to be the public default; "
                    f"got {default_id!r}",
                )
            for field in ("definition", "default_configuration"):
                self.equal(
                    "docs/STATUS.json",
                    self.status,
                    f"/lanes/public-functional/{field}",
                    default_path,
                )
        alternative_path = lane.get("accepted_alternative")
        alternative = self.recipe_by_path(alternative_path)
        expected_alternative = self.recipes.get(NF3_RECIPE_ID)
        if expected_alternative is not None and alternative_path != expected_alternative[0]:
            self.problem(
                "docs/STATUS.json",
                "/lanes/public-functional/accepted_alternative",
                f"expected accepted NF3 alternative {expected_alternative[0]!r}; "
                f"got {alternative_path!r}",
            )
        if alternative is not None and alternative.get("default") is not False:
            self.problem(
                "docs/STATUS.json",
                "/lanes/public-functional/accepted_alternative",
                f"expected a non-default recipe; {alternative_path!r} is default",
            )
        if DEFAULT_RECIPE_ID in self.recipes:
            _, default_recipe = self.recipes[DEFAULT_RECIPE_ID]
            self.equal(
                "docs/STATUS.json",
                self.status,
                "/lanes/public-functional/evidence",
                default_recipe["publication"].get("evidence"),
            )
        self.validate_recipe_role(
            DEFAULT_RECIPE_ID,
            "/components/exl3_recipe",
            default_field="default",
        )
        self.validate_recipe_role(
            NF3_RECIPE_ID,
            "/components/nf3_recipe",
            default_field="default_public_functional_target",
        )
        self.validate_r7_role()
        self.equal(
            "docs/STATUS.json",
            self.status,
            "/components/exl3_recipe/main_advertised_configuration",
            True,
        )

    def validate_recipe_role(
        self, recipe_id: str, status_pointer: str, *, default_field: str
    ) -> None:
        if self.status is None or recipe_id not in self.recipes:
            return
        source, recipe = self.recipes[recipe_id]
        component = self.value(self.status, status_pointer)
        if not isinstance(component, dict):
            self.problem("docs/STATUS.json", status_pointer, "expected an object")
            return
        # STATUS owns maturity and role. Recipes project those decisions so
        # executable configuration can fail closed before deployment.
        self.equal(source, recipe, "/maturity", component.get("maturity"))
        self.equal(source, recipe, "/default", component.get(default_field))
        projections = {
            "entrypoint": source,
            "model": recipe["model"]["repository"],
            "zero_build_ready": recipe.get("publication", {}).get("zero_build_ready"),
            "local_build_ready": recipe.get("publication", {}).get("local_build_ready"),
        }
        for field, expected in projections.items():
            if field in component or field in {"entrypoint", "maturity", "model"}:
                self.equal(
                    "docs/STATUS.json",
                    self.status,
                    f"{status_pointer}/{field}",
                    expected,
                )
        evidence = recipe.get("publication", {}).get("evidence")
        if evidence is not None and "evidence" in component:
            self.equal(
                "docs/STATUS.json",
                self.status,
                f"{status_pointer}/evidence",
                evidence,
            )

    def validate_default_receipt(self) -> None:
        if self.status is None or DEFAULT_RECIPE_ID not in self.recipes:
            return
        _, recipe = self.recipes[DEFAULT_RECIPE_ID]
        receipt_pointer = "/components/exl3_recipe/clean_checkout_receipt"
        receipt = self.value(self.status, receipt_pointer)
        if not isinstance(receipt, dict):
            self.problem("docs/STATUS.json", receipt_pointer, "expected an object")
            return
        projections = {
            "image_source_commit": recipe["publication"].get("image_source_commit"),
            "launcher_fix_commit": recipe["publication"].get("launcher_fix_commit"),
            "image_id": recipe["runtime"].get("validated_local_image_id"),
            "identical_image_ranks": recipe["hardware"].get("ranks"),
        }
        for field, expected in projections.items():
            self.equal(
                "docs/STATUS.json",
                self.status,
                f"{receipt_pointer}/{field}",
                expected,
            )
        ranks_match = receipt.get("identical_image_ranks") == recipe["hardware"].get(
            "ranks"
        )
        if recipe["publication"].get("identical_four_rank_image") != ranks_match:
            self.problem(
                "docs/STATUS.json",
                f"{receipt_pointer}/identical_image_ranks",
                "does not agree with recipe publication.identical_four_rank_image",
            )

    def validate_r7_role(self) -> None:
        if self.status is None or R7_RECIPE_ID not in self.recipes:
            return
        recipe_source, recipe = self.recipes[R7_RECIPE_ID]
        pointer = "/components/exl3_r7_mtp4_nvfp4_ckv_operator_default"
        component = self.value(self.status, pointer)
        if not isinstance(component, dict):
            self.problem("docs/STATUS.json", pointer, "expected an object")
            return
        model = f"{recipe['model']['repository']}@{recipe['model']['revision']}"
        self.equal(recipe_source, recipe, "/maturity", component.get("maturity"))
        self.equal(recipe_source, recipe, "/default", component.get("default"))
        self.equal(
            recipe_source,
            recipe,
            "/publication/operator_default",
            component.get("operator_default"),
        )
        self.equal(
            recipe_source,
            recipe,
            "/publication/acceptance_scope",
            component.get("acceptance_scope"),
        )
        projections = {
            "accepted": True,
            "public_functional_default": False,
            "model": model,
            "entrypoint": recipe["publication"].get("evidence"),
            "machine_readable_evidence": recipe["publication"].get(
                "machine_readable_evidence"
            ),
        }
        for field, expected in projections.items():
            self.equal("docs/STATUS.json", self.status, f"{pointer}/{field}", expected)
        self.equal(
            "docs/STATUS.json",
            self.status,
            "/lanes/public-functional/operator_accepted_profile",
            recipe["publication"].get("evidence"),
        )
        if component.get("public_functional_default") is not False:
            self.problem(
                "docs/STATUS.json",
                f"{pointer}/public_functional_default",
                "operator acceptance must not promote the public-functional default",
            )
        self.equal(
            "docs/STATUS.json",
            self.status,
            "/components/exl3_r7_arm64_builder/promotion_checklist",
            recipe["publication"].get("promotion_checklist"),
        )
        self.validate_r7_receipt(recipe_source, recipe, component)

    def validate_r7_receipt(
        self,
        recipe_source: str,
        recipe: dict[str, Any],
        component: dict[str, Any],
    ) -> None:
        relative = recipe["publication"].get("machine_readable_evidence")
        if not isinstance(relative, str):
            self.problem(
                recipe_source,
                "/publication/machine_readable_evidence",
                "expected a repository-relative JSON evidence path",
            )
            return
        if not self.validate_reference(
            recipe_source,
            "/publication/machine_readable_evidence",
            relative,
            parse_json=True,
        ):
            return
        receipt = self.load_object(relative)
        if receipt is None:
            return
        model = f"{recipe['model']['repository']}@{recipe['model']['revision']}"
        route_block_rows = self.value(
            recipe, "/serving/exact_q40_policy/route_block_rows"
        )
        capacity_rows = self.value(recipe, "/serving/exact_q40_policy/capacity_rows")
        for pointer, projected in (
            ("/serving/exact_q40_policy/capacity_rows", capacity_rows),
            ("/serving/exact_q40_policy/route_block_rows", route_block_rows),
        ):
            if projected is _MISSING:
                self.problem(recipe_source, pointer, "required R7 serving projection is missing")
        receipt_projections = {
            "/maturity": recipe.get("maturity"),
            "/operator_default": recipe["publication"].get("operator_default"),
            "/public_functional_default": False,
            "/acceptance_scope": recipe["publication"].get("acceptance_scope"),
            "/policy/model": model,
            "/policy/image_id": recipe["runtime"].get("validated_local_image_id"),
            "/policy/capacity_rows": recipe["serving"].get("max_query_rows"),
            "/policy/kv_cache_bytes_per_rank": recipe["serving"].get(
                "kv_cache_bytes_per_rank"
            ),
            "/policy/reported_kv_capacity_tokens": recipe["serving"].get(
                "reported_kv_tokens"
            ),
        }
        if route_block_rows is not _MISSING:
            receipt_projections["/policy/route_block_rows"] = route_block_rows
        for pointer, expected in receipt_projections.items():
            self.equal(relative, receipt, pointer, expected)
        status_projections = {
            "/maximum_query_rows": recipe["serving"].get("max_query_rows"),
            "/maximum_model_length": recipe["serving"].get("max_model_len"),
            "/maximum_batched_tokens": recipe["serving"].get(
                "max_num_batched_tokens"
            ),
            "/kv_cache_dtype": recipe["serving"].get("kv_cache_dtype"),
            "/kv_cache_bytes_per_rank": recipe["serving"].get(
                "kv_cache_bytes_per_rank"
            ),
            "/reported_kv_capacity_tokens": recipe["serving"].get(
                "reported_kv_tokens"
            ),
        }
        if capacity_rows is not _MISSING:
            status_projections["/exact_q40_policy/capacity_rows"] = capacity_rows
        if route_block_rows is not _MISSING:
            status_projections["/exact_q40_policy/route_block_rows"] = route_block_rows
        for suffix, expected in status_projections.items():
            actual = self.value(component, suffix)
            if actual != expected:
                shown = "<missing>" if actual is _MISSING else repr(actual)
                self.problem(
                    "docs/STATUS.json",
                    "/components/exl3_r7_mtp4_nvfp4_ckv_operator_default" + suffix,
                    f"expected {expected!r}; got {shown}",
                )

    def validate_selected_references(self) -> None:
        if self.status is None:
            return
        status_references = [
            ("/lanes/reference/evidence", False),
            ("/lanes/public-functional/evidence", False),
            ("/lanes/public-functional/operator_accepted_profile", False),
            ("/components/exl3_r7_arm64_builder/promotion_checklist", False),
            (
                "/components/exl3_r7_mtp4_nvfp4_ckv_operator_default/"
                "machine_readable_evidence",
                True,
            ),
            (
                "/components/exl3_r7_mtp4_nvfp4_ckv_operator_default/"
                "predecessor_machine_readable_evidence",
                True,
            ),
            (
                "/components/exl3_r7_mtp4_nvfp4_ckv_operator_default/"
                "accepted_prefill_snapshot/evidence",
                True,
            ),
            (
                "/components/exl3_r7_mtp4_nvfp4_ckv_operator_default/"
                "accepted_decode_snapshot/evidence",
                True,
            ),
            (
                "/components/exl3_r7_mtp4_nvfp4_ckv_operator_default/"
                "accepted_coding_snapshot/evidence",
                True,
            ),
        ]
        for pointer, parse_json in status_references:
            value = self.value(self.status, pointer)
            if value is _MISSING:
                self.problem(
                    "docs/STATUS.json",
                    pointer,
                    "required evidence reference is missing",
                )
                continue
            self.validate_reference(
                "docs/STATUS.json", pointer, value, parse_json=parse_json
            )
        for source, recipe in self.recipes.values():
            publication = recipe.get("publication", {})
            for field in (
                "evidence",
                "machine_readable_evidence",
                "promotion_checklist",
                "historical_lane",
            ):
                if field in publication:
                    self.validate_reference(
                        source,
                        f"/publication/{field}",
                        publication[field],
                        parse_json=field == "machine_readable_evidence",
                    )

    def validate_profile_identity_owners(self) -> None:
        """Project recipe identities into their canonical human-readable owners."""

        registry_required: list[tuple[str, Any]] = []
        for recipe_id in (DEFAULT_RECIPE_ID, R7_RECIPE_ID):
            if recipe_id not in self.recipes:
                continue
            _, recipe = self.recipes[recipe_id]
            registry_required.extend(
                [
                    (f"{recipe_id} model repository", recipe["model"]["repository"]),
                    (f"{recipe_id} model revision", recipe["model"]["revision"]),
                ]
            )

        projections: list[tuple[str, list[tuple[str, Any]]]] = [
            ("docs/profiles/README.md", registry_required),
        ]
        if DEFAULT_RECIPE_ID in self.recipes:
            _, default_recipe = self.recipes[DEFAULT_RECIPE_ID]
            projections.append(
                (
                    "docs/EXL3_RECIPE.md",
                    [
                        (
                            f"{DEFAULT_RECIPE_ID} validated image identity",
                            default_recipe["runtime"].get("validated_local_image_id"),
                        )
                    ],
                )
            )

        for relative, required in projections:
            try:
                prose = (self.root / relative).read_text(encoding="utf-8")
            except UnicodeError as exc:
                self.problem(relative, "", f"canonical profile prose is not valid UTF-8: {exc}")
                continue
            except OSError as exc:
                self.problem(relative, "", f"cannot read canonical profile prose: {exc}")
                continue
            for label, token in required:
                if not isinstance(token, str):
                    self.problem(relative, "", f"cannot project non-string {label}")
                elif token not in prose:
                    self.problem(relative, "", f"missing {label}: {token}")

    def validate_canonical_claim_owners(self) -> None:
        """Anchor projections to the lane and measured-claim specifications."""

        try:
            lane_text = (self.root / "docs/PUBLIC_FUNCTIONAL_TARGET.md").read_text(
                encoding="utf-8"
            )
        except UnicodeError as exc:
            self.problem("docs/PUBLIC_FUNCTIONAL_TARGET.md", "", f"lane definition is not valid UTF-8: {exc}")
        except OSError as exc:
            self.problem("docs/PUBLIC_FUNCTIONAL_TARGET.md", "", f"cannot read lane definition: {exc}")
        else:
            normalized = " ".join(
                line.lstrip("> ").strip() for line in lane_text.splitlines()
            )
            required_phrases = (
                "EXL3 plus LMCache CS512 is the default and main advertised configuration",
                "accepted NF3 matrix",
                "does not change the public default or accepted NF3 matrix",
            )
            for phrase in required_phrases:
                if phrase not in normalized:
                    self.problem(
                        "docs/PUBLIC_FUNCTIONAL_TARGET.md",
                        "",
                        f"missing canonical lane statement: {phrase!r}",
                    )

        try:
            results = (self.root / "docs/RESULTS.md").read_text(encoding="utf-8")
        except UnicodeError as exc:
            self.problem("docs/RESULTS.md", "", f"measured claims are not valid UTF-8: {exc}")
            return
        except OSError as exc:
            self.problem("docs/RESULTS.md", "", f"cannot read measured claims: {exc}")
            return
        if DEFAULT_RECIPE_ID not in self.recipes:
            return
        _, recipe = self.recipes[DEFAULT_RECIPE_ID]
        projections = (
            (
                "image source commit",
                r"source commit\s*`([^`]+)`",
                recipe["publication"].get("image_source_commit"),
            ),
            (
                "launcher correction commit",
                r"launcher correction\s*`([^`]+)`",
                recipe["publication"].get("launcher_fix_commit"),
            ),
            (
                "validated image identity",
                r"Exact image\s*`([^`]+)`",
                recipe["runtime"].get("validated_local_image_id"),
            ),
        )
        for label, pattern, projected in projections:
            match = re.search(pattern, results, re.IGNORECASE | re.DOTALL)
            if match is None:
                self.problem("docs/RESULTS.md", "", f"cannot locate canonical {label}")
            elif match.group(1) != projected:
                self.problem(
                    "docs/RESULTS.md",
                    "",
                    f"canonical {label} is {match.group(1)!r}; recipe projects {projected!r}",
                )

    def validate_stable_status(self) -> None:
        if self.status is not None:
            volatile = self.value(
                self.status, "/components/offline_python_contracts/last_observed"
            )
            if volatile is not _MISSING:
                self.problem(
                    "docs/STATUS.json",
                    "/components/offline_python_contracts/last_observed",
                    "volatile test totals do not belong in present-state status; "
                    "reference the blocking CI job instead",
                )
        try:
            prose = (self.root / "CONTRIBUTING.md").read_text(encoding="utf-8")
        except UnicodeError as exc:
            self.problem("CONTRIBUTING.md", "", f"contributor guide is not valid UTF-8: {exc}")
            return
        except OSError as exc:
            self.problem("CONTRIBUTING.md", "", f"cannot read contributor guide: {exc}")
            return
        start = prose.find("## Test suites and where they run")
        end = prose.find("\n## ", start + 3) if start >= 0 else -1
        section = prose[start : end if end >= 0 else None] if start >= 0 else ""
        match = PYTEST_SUMMARY_RE.search(section)
        if match:
            self.problem(
                "CONTRIBUTING.md",
                "",
                f"volatile pytest summary in present-state test matrix: {match.group(0)!r}",
            )

    def run(self) -> list[Problem]:
        self.load_inputs()
        self.validate_recipe_roles()
        self.validate_default_receipt()
        self.validate_selected_references()
        self.validate_profile_identity_owners()
        self.validate_canonical_claim_owners()
        self.validate_stable_status()
        return sorted(set(self.problems))


def validate(root: Path) -> list[Problem]:
    """Return every publication inconsistency under ``root``."""

    return PublicationValidator(root).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate SparkRing publication roles, identities, and evidence paths."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the checkout containing this script)",
    )
    args = parser.parse_args(argv)
    problems = validate(args.root)
    if problems:
        for problem in problems:
            print(problem.render(), file=sys.stderr)
        noun = "error" if len(problems) == 1 else "errors"
        print(
            f"publication consistency: FAIL ({len(problems)} {noun})",
            file=sys.stderr,
        )
        return 1
    print("publication consistency: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
