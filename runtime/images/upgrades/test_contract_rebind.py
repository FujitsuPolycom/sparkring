"""Contract hash refresh is not permission to skip compatibility evidence."""

import pytest

from .contract_rebind import rebind
from .contracts import Refused, sha
from .sources import tree_digest


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
