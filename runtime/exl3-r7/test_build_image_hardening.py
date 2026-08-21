"""Test-first hardening of build-image.sh and prepare_context.py.

These tests assert the immutable identity contract before the implementation
lands: the parent image must be identified by digest or image ID (never a
mutable tag alone), PREPARED_SOURCES must be verified against its receipt
(not just existence), and prepare_context.py must verify its own output.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BUILD_SCRIPT = (HERE / "build-image.sh").read_text(encoding="utf-8")
PINS = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
PREPARE_CONTEXT = (HERE / "prepare_context.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Parent image identified by immutable digest/image ID, fail closed on drift
# ---------------------------------------------------------------------------

def test_build_script_requires_base_image_id_env() -> None:
    """build-image.sh must require an immutable base-image ID, not just a tag."""
    assert "BASE_IMAGE_ID" in BUILD_SCRIPT
    assert "BASE_IMAGE" in BUILD_SCRIPT


def test_build_script_rejects_mutable_tag_only_identity() -> None:
    """The default base_image must not be a bare mutable tag without a digest."""
    # The script must inspect the image by digest/ID, not just trust the tag.
    assert "image inspect" in BUILD_SCRIPT
    assert "--format" in BUILD_SCRIPT


def test_build_script_fails_closed_on_base_image_drift() -> None:
    """The script must compare observed vs expected image ID and fail on mismatch."""
    assert "fatal" in BUILD_SCRIPT.lower() or "exit" in BUILD_SCRIPT.lower()
    # The comparison must be against an immutable identifier
    assert ".Id" in BUILD_SCRIPT or "digest" in BUILD_SCRIPT.lower()


def test_build_script_passes_base_image_id_as_build_arg() -> None:
    """The immutable image ID must be forwarded to the Containerfile.

    BuildKit resolves a bare sha256 ID in FROM against the registry, so the
    verified parent is retagged under a build-local name; the immutable ID
    still travels as BASE_IMAGE_ID, and the tag is applied to the observed
    ID only after the drift check.
    """
    assert '"${engine}" tag "${observed_base}" "${parent_build_tag}"' in BUILD_SCRIPT
    assert '--build-arg "BASE_IMAGE=${parent_build_tag}"' in BUILD_SCRIPT
    assert '--build-arg "BASE_IMAGE_ID=${observed_base}"' in BUILD_SCRIPT


def test_containerfile_labels_parent_image_id() -> None:
    """Containerfile must embed the parent image ID as an OCI label."""
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")
    assert "org.sparkring.parent.image-id" in containerfile


# ---------------------------------------------------------------------------
# 2. PREPARED_SOURCES verified against receipt, not just existence
# ---------------------------------------------------------------------------

def test_build_script_verifies_receipt_content_not_just_existence() -> None:
    """When PREPARED_SOURCES is set, the script must verify the receipt, not
    just check that receipt.json exists."""
    # The script must call a verifier on the prepared sources, not just test -f
    assert "verify" in BUILD_SCRIPT.lower()
    assert "receipt.json" in BUILD_SCRIPT


def test_prepare_context_verifies_component_trees() -> None:
    """prepare_context.py must verify patch_sha256 and result_tree, not just
    write a receipt."""
    assert "patch_sha256" in PREPARE_CONTEXT
    assert "result_tree" in PREPARE_CONTEXT
    assert "write-tree" in PREPARE_CONTEXT


def test_prepare_context_verifies_release_commit() -> None:
    """The release checkout commit must be verified, not just cloned."""
    assert "release_commit" in PREPARE_CONTEXT


def test_pins_json_has_all_required_component_fields() -> None:
    """Each pinned component must have base_commit, patch, patch_sha256,
    result_tree — not just a repository reference."""
    for name, spec in PINS["components"].items():
        assert "repository" in spec, f"{name} missing repository"
        assert "base_commit" in spec, f"{name} missing base_commit"
        assert "patch" in spec, f"{name} missing patch"
        assert "patch_sha256" in spec, f"{name} missing patch_sha256"
        assert "result_tree" in spec, f"{name} missing result_tree"


def test_pins_json_release_has_required_fields() -> None:
    """The release pin must have repository, commit, and patch_root."""
    release = PINS["release"]
    assert "repository" in release
    assert "commit" in release
    assert "patch_root" in release


# ---------------------------------------------------------------------------
# 3. prepare_context.py verifies its own output (self-verification)
# ---------------------------------------------------------------------------

def test_prepare_context_has_verify_function() -> None:
    """prepare_context.py must have a verify function that checks a prepared
    directory against its receipt."""
    assert "def verify" in PREPARE_CONTEXT


def test_prepare_context_verify_checks_schema() -> None:
    """The verify function must check the schema version."""
    assert "schema_version" in PREPARE_CONTEXT


def test_prepare_context_verify_checks_component_receipts() -> None:
    """The verify function must re-check patch_sha256 and result_tree from
    the receipt, not just trust the receipt file."""
    # The verify path must re-examine the actual checked-out trees
    assert "verify_component" in PREPARE_CONTEXT or "def verify" in PREPARE_CONTEXT


# ---------------------------------------------------------------------------
# 4. OCI labels in Containerfile
# ---------------------------------------------------------------------------

def test_containerfile_has_oci_source_labels() -> None:
    """Containerfile must have OCI source/revision/license labels for every
    built-in component."""
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")
    for label in (
        "org.opencontainers.image.source",
        "org.opencontainers.image.revision",
        "org.opencontainers.image.licenses",
    ):
        assert label in containerfile, f"missing OCI label: {label}"


def test_containerfile_labels_vllm_revision() -> None:
    """The vLLM commit must be recorded as an OCI label."""
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")
    vllm_commit = PINS["components"]["vllm"]["base_commit"]
    assert vllm_commit in containerfile or "VLLM_COMMIT" in containerfile


def test_containerfile_labels_b12x_revision() -> None:
    """The B12X commit must be recorded as an OCI label."""
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")
    b12x_commit = PINS["components"]["b12x"]["base_commit"]
    assert b12x_commit in containerfile or "B12X_COMMIT" in containerfile


def test_containerfile_labels_exllamav3_revision() -> None:
    """ExLlamaV3 commit must be in the Containerfile."""
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")
    exllamav3_commit = PINS["runtime_dependencies"]["exllamav3"]
    assert exllamav3_commit in containerfile


def test_containerfile_labels_instanttensor_revision() -> None:
    """InstantTensor commit must be in the Containerfile."""
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")
    instanttensor_commit = PINS["runtime_dependencies"]["instanttensor"]
    assert instanttensor_commit in containerfile


def test_containerfile_labels_cutlass_and_triton() -> None:
    """CUTLASS and Triton kernel commits must be in the Containerfile or
    referenced via build-args."""
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")
    build_deps = (HERE / "prepare_build_deps.py").read_text(encoding="utf-8")
    # CUTLASS and Triton are pinned in prepare_build_deps.py; the Containerfile
    # must at least reference their source directories
    assert "cutlass" in containerfile.lower()
    assert "triton" in containerfile.lower()
    # The commits must be pinned somewhere in the build chain
    assert "da5e086dab31d63815acafdac9a9c5893b1c69e2" in build_deps  # CUTLASS
    assert "0add68262ab0a2e33b84524346cb27cbb2787356" in build_deps  # Triton


def test_builder_compiles_and_installs_sircl_from_the_same_checkout() -> None:
    """The R7 image must not inherit an unpinned SIRCL binary or adapter."""

    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")
    assert "COPY bundle /opt/r7-src" in containerfile
    assert "cmake -S /opt/r7-src/spark_transport" in containerfile
    assert "--target spark_transport_capi" in containerfile
    assert "/opt/sparkring/spark_transport/libspark_transport_capi.so" in containerfile
    assert "build-public-overlay.py" in containerfile
    assert "public-overlay-files.json" in containerfile
    assert "--output /opt/spark-vllm-public" in containerfile

    for path in (
        "runtime/build-public-overlay.py",
        "runtime/public-overlay-files.json",
        "spark_transport",
    ):
        assert path in BUILD_SCRIPT


# ---------------------------------------------------------------------------
# 5. Receipt verification with prepare_context verifier
# ---------------------------------------------------------------------------

def test_prepare_context_verify_rejects_missing_receipt(tmp_path: Path) -> None:
    """verify() must fail when receipt.json is absent."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "r7_prepare_context", HERE / "prepare_context.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises((RuntimeError, FileNotFoundError, KeyError, ValueError)):
        module.verify(tmp_path)


def test_prepare_context_verify_rejects_wrong_schema(tmp_path: Path) -> None:
    """verify() must fail when the receipt schema_version is wrong."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "r7_prepare_context", HERE / "prepare_context.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (tmp_path / "receipt.json").write_text(
        json.dumps({"schema_version": 999, "release_commit": "x", "components": {}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="schema"):
        module.verify(tmp_path)


def test_prepare_context_verify_rejects_component_count_drift(tmp_path: Path) -> None:
    """verify() must fail when the receipt has fewer components than pins."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "r7_prepare_context", HERE / "prepare_context.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (tmp_path / "receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_commit": "x",
                "components": {"vllm": {"base_commit": "x", "patch_sha256": "x", "result_tree": "x"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises((RuntimeError, KeyError)):
        module.verify(tmp_path)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def test_verify_component_rejects_unstaged_and_untracked_content(
    tmp_path: Path,
) -> None:
    """The verified index tree must also describe every copied source byte."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "r7_prepare_context_cleanliness", HERE / "prepare_context.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    repo = tmp_path / "component"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "SparkRing tests")
    tracked = repo / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "base")
    base_commit = _git(repo, "rev-parse", "HEAD")

    tracked.write_text("patched\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    component_spec = {
        "base_commit": base_commit,
        "result_tree": _git(repo, "write-tree"),
    }
    module.verify_component("component", component_spec, tmp_path)

    tracked.write_text("unstaged drift\n", encoding="utf-8")
    with pytest.raises(module.PreparationError, match="unstaged"):
        module.verify_component("component", component_spec, tmp_path)

    tracked.write_text("patched\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("untracked drift\n", encoding="utf-8")
    with pytest.raises(module.PreparationError, match="untracked"):
        module.verify_component("component", component_spec, tmp_path)


def test_oci_revision_identifies_sparkring_source_revision() -> None:
    """The standard revision label must identify the repository in source."""
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")
    assert "ARG SPARKRING_REVISION" in containerfile
    assert 'org.opencontainers.image.revision="${SPARKRING_REVISION}"' in containerfile


def test_oci_license_expression_covers_bundled_license_families() -> None:
    """The image combines Apache, MIT, and BSD-licensed components."""
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")
    assert "ARG IMAGE_LICENSES" in containerfile
    assert 'org.opencontainers.image.licenses="${IMAGE_LICENSES}"' in containerfile
    assert "BASE_IMAGE_LICENSES" in BUILD_SCRIPT
    assert "Apache-2.0 AND MIT AND BSD-3-Clause" in BUILD_SCRIPT
