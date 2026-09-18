"""Contract hash refresh is not permission to skip compatibility evidence."""

import pytest
import json

from .contract_rebind import rebind
from .contracts import Refused, sha
from .sources import tree_digest
from .build_native import prepare_source_binding


def fixture(tmp_path, before, after):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "module.py").write_text(before)
    (candidate / "module.py").write_text(after)
    contract = dict(
        schema="sparkring-vllm-kv-block-lease-contract/v1",
        files=[
            dict(
                path="module.py",
                sha256=sha((baseline / "module.py").read_bytes()),
                required_symbols=["Lease.read"],
            )
        ],
    )
    oracle = dict(
        schema="sparkring-upgrade-gate/v1",
        outcome="passed",
        variant="candidate",
        input_sha256="f" * 64,
        subject_sha256=tree_digest(candidate),
        assertions=4,
        skipped=0,
    )
    return contract, baseline, candidate, oracle


def test_comments_can_change_without_changing_contract_behavior(tmp_path):
    before = "LIMIT=1\nclass Lease:\n def read(self):\n  return LIMIT\n"
    values = fixture(tmp_path, before, before + "# A source-level comment.\n")
    contract, baseline, candidate, oracle = values
    result, proof = rebind(
        contract, baseline, candidate, "a" * 40, oracle, input_sha256="f" * 64
    )
    assert result["files"][0]["sha256"] != contract["files"][0]["sha256"]
    assert proof["serving_qualified"] is False


@pytest.mark.parametrize(
    "after",
    [
        "LIMIT=2\nclass Lease:\n def read(self):\n  return LIMIT\n",
        "LIMIT=1\nclass Lease:\n def read(self):\n  return 0\n",
    ],
)
def test_changed_globals_or_interface_cannot_be_rehashed(tmp_path, after):
    contract, baseline, candidate, oracle = fixture(
        tmp_path, "LIMIT=1\nclass Lease:\n def read(self):\n  return LIMIT\n", after
    )
    with pytest.raises(Refused, match="migration required"):
        rebind(contract, baseline, candidate, "a" * 40, oracle, input_sha256="f" * 64)


def test_oracle_for_other_source_cannot_authorize_rebinding(tmp_path):
    contract, baseline, candidate, oracle = fixture(
        tmp_path,
        "class Lease:\n def read(self): pass\n",
        "class Lease:\n def read(self): pass\n",
    )
    oracle["subject_sha256"] = "b" * 64
    with pytest.raises(Refused, match="exact source"):
        rebind(contract, baseline, candidate, "a" * 40, oracle, input_sha256="f" * 64)


@pytest.mark.parametrize("changed", [False, True])
def test_byte_only_contract_files_must_remain_identical(tmp_path, changed):
    before = "VALUE = 1\n"
    contract, baseline, candidate, oracle = fixture(
        tmp_path, before, before + "# changed\n" if changed else before
    )
    contract["files"][0]["required_symbols"] = []
    if changed:
        with pytest.raises(Refused, match="Byte-only"):
            rebind(
                contract, baseline, candidate, "a" * 40, oracle, input_sha256="f" * 64
            )
    else:
        _, proof = rebind(
            contract, baseline, candidate, "a" * 40, oracle, input_sha256="f" * 64
        )
        assert proof["files"][0]["byte_identical"] is True


def test_rebound_contract_does_not_retain_stale_source_metadata(tmp_path):
    source = "class Lease:\n def read(self): pass\n"
    contract, baseline, candidate, oracle = fixture(tmp_path, source, source)
    contract.update(
        vllm_commit="b" * 40,
        vllm_tree="c" * 40,
        semantic_review={"comparison_tree": "d" * 40},
    )
    result, proof = rebind(
        contract, baseline, candidate, "a" * 40, oracle, input_sha256="f" * 64
    )
    assert result["vllm_commit"] == "a" * 40
    assert "vllm_tree" not in result
    assert (
        result["semantic_review"]["candidate_tree_sha256"]
        == proof["candidate_tree_sha256"]
    )


def test_native_builder_installs_versioned_binding_from_accepted_oracle(tmp_path):
    source = "class Lease:\n def read(self): pass\n"
    contract, baseline, candidate, oracle = fixture(
        tmp_path, source, source + "# text\n"
    )
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(contract))
    oracle["gate"] = "source-oracle"
    old_path = "/opt/sparkring/contracts/reference.json"
    parent = {
        "integration_contracts": {old_path: {}},
        "files": {old_path: sha(path.read_bytes())},
    }
    policy = {
        "_root": str(tmp_path),
        "foundation": {
            "source_binding": {
                "contract": path.name,
                "reference_source": str(baseline),
                "oracle": "source-oracle",
            }
        },
    }
    bundle = {
        "input_sha256": "f" * 64,
        "sources": {"vllm": {"oracles": [oracle], "target_commit": "a" * 40}},
    }
    context = tmp_path / "context"
    context.mkdir()
    binding, active = prepare_source_binding(
        policy, bundle, {"vllm": candidate}, parent, context
    )
    assert active == [binding["destination"]]
    assert old_path not in active
    assert json.loads((context / binding["file"]).read_text())["files"][0][
        "sha256"
    ] == sha((candidate / "module.py").read_bytes())
    assert (
        json.loads((context / binding["proof_file"]).read_text())["serving_qualified"]
        is False
    )
