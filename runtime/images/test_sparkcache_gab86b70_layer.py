"""The SparkCache lease layer over image sha256:d52737a109e0 stays pinned.

The layer ships the connector-job lease contract that matches the release
image's kraken-line vLLM and the three SparkCache files that verify it. These
tests pin the descriptor's file list and hashes from the repository, so a
source or descriptor edit that would change the built layer fails offline.
"""

import json
from pathlib import Path

from runtime.images import derived_layer

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = ROOT / "runtime/images/compositions/installer-sparkcache-gab86b70/descriptor.json"
PARENT_LOCK = ROOT / (
    "runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036"
    "/installer-image.json"
)
RELEASE_README = ROOT / "runtime/releases/dev-20261010-sparkcache-gab86b70-plugins-status036/README.md"
SITE = derived_layer.SITE

CONTRACT_SOURCE = "integrations/sparkcache/gab86b70/runtime_patches/vllm-connector-jobs-gab86b70.json"
CONTRACT_SHA256 = "ab53c097d7ca408e995bac7c2bc915b64119881ed64b9cec0b3cc3b30ed2f990"
EXPECTED_FILES = {
    SITE + "sparkcache/runtime_patches/vllm-connector-jobs-gab86b70.json": (
        CONTRACT_SOURCE, CONTRACT_SHA256, None),
    SITE + "sparkcache/runtime_patches/verify_lease_contract.py": (
        "integrations/sparkcache/gab86b70/runtime_patches/verify_lease_contract.py",
        "acfd9a24482887efab21b9b952413406c933d683c478766285c74aa4d6cbfc3e",
        "acfd9a24482887efab21b9b952413406c933d683c478766285c74aa4d6cbfc3e"),
    SITE + "sparkcache/runtime_patches/generic_connector_contract.py": (
        "integrations/sparkcache/gab86b70/runtime_patches/generic_connector_contract.py",
        "543a448294e1b0b118fdd8fa16acb03496d14e0db122f44aea13c5f438405a67",
        "543a448294e1b0b118fdd8fa16acb03496d14e0db122f44aea13c5f438405a67"),
    SITE + "sparkcache/streaming/manager_page_factory.py": (
        "integrations/sparkcache/gab86b70/streaming/manager_page_factory.py",
        "a3a0c24f01339baaeabff29fe3dc4f951301496ede2fd855748ca1b7b63eeef1",
        "a3a0c24f01339baaeabff29fe3dc4f951301496ede2fd855748ca1b7b63eeef1"),
}


def test_descriptor_validates_and_matches_the_expected_files():
    record = derived_layer.descriptor(DESCRIPTOR)
    assert record["id"] == "sparkcache-gab86b70"
    assert record["parent_lock"] == PARENT_LOCK.relative_to(ROOT).as_posix()
    assert record["provenance"].startswith(derived_layer.RECEIPTS)
    assert set(record["files"]) == set(EXPECTED_FILES)
    for target, (source, sha256, inherited) in EXPECTED_FILES.items():
        row = record["files"][target]
        assert row["source"] == source
        assert row["sha256"] == sha256
        assert row["inherited_sha256"] == inherited


def test_layer_sources_carry_their_pinned_bytes():
    record = derived_layer.descriptor(DESCRIPTOR)
    for target, row in record["files"].items():
        data = (ROOT / row["source"]).read_bytes()
        import hashlib
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        # A restated file must be byte-identical to the parent image's installed
        # file: the repository source is the pinned parent bytes.
        if row["inherited_sha256"] is not None:
            assert row["inherited_sha256"] == row["sha256"]


def test_parent_lock_is_the_release_image_and_the_contract_binds_it():
    lock = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    assert lock["image_id"].startswith("sha256:d52737a109e0")
    contract = json.loads((ROOT / CONTRACT_SOURCE).read_text(encoding="utf-8"))
    assert contract["schema"] == "sparkring-vllm-kv-block-lease-contract/v1"
    assert contract["connector_api"] == "jj-block-state-read-leases/v1"
    assert contract["image_id"] == "sha256:d52737a109e0"
    assert contract["vllm_commit"] == "gab86b7073"


def _gate_literals():
    """REQUIRED_SEMANTICS and REQUIRED_SYMBOLS of the shipped generic gate, read statically.

    The gate module imports its sibling verifier relatively, so the repository
    cannot import it as a package; the literals are exactly what
    ``verify_connector_job_contract`` enforces.
    """

    import ast
    gate = ROOT / "integrations/sparkcache/gab86b70/runtime_patches/generic_connector_contract.py"
    tree = ast.parse(gate.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("CONNECTOR_API", "REQUIRED_SEMANTICS", "REQUIRED_SYMBOLS"):
                values[name] = ast.literal_eval(node.value)
    assert set(values) == {"CONNECTOR_API", "REQUIRED_SEMANTICS", "REQUIRED_SYMBOLS"}
    return values


def test_shipped_contract_covers_the_generic_gate():
    record = derived_layer.descriptor(DESCRIPTOR)
    added = [row for row in record["files"].values() if row["inherited_sha256"] is None]
    assert len(added) == 1 and added[0]["sha256"] == CONTRACT_SHA256
    contract = json.loads((ROOT / CONTRACT_SOURCE).read_text(encoding="utf-8"))
    gate = _gate_literals()
    assert contract["connector_api"] == gate["CONNECTOR_API"]
    assert set(gate["REQUIRED_SEMANTICS"]) <= set(contract["required_semantics"])
    records = {row["path"]: row for row in contract["files"]}
    for path, symbols in gate["REQUIRED_SYMBOLS"].items():
        assert path in records
        assert set(symbols) <= set(records[path].get("required_symbols", ()))


def test_release_readme_and_builder_name_the_layer():
    text = RELEASE_README.read_text(encoding="utf-8")
    assert "installer-sparkcache-gab86b70" in text
    assert "glm53-flash-nvfp4-sparkcache-tp2" in text and "glm53-flash-nvfp4-sparkcache-tp4" in text
    builders = json.loads((ROOT / "runtime/images/builders.json").read_text(encoding="utf-8"))
    rows = [row for row in builders["builders"] if row["id"] == "installer-sparkcache-gab86b70"]
    assert len(rows) == 1
    assert rows[0]["path"] == "runtime/images/derived_layer.py"
    # The release has no lock until a Spark records the built image, so the
    # builder names no release selection.
    assert "releases" not in rows[0]
    assert not (RELEASE_README.parent / "installer-image.json").exists()
