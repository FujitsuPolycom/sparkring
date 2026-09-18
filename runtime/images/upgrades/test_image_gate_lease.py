"""Image admission executes the installed SparkCache contract implementation."""

import json
from pathlib import Path

from .contracts import sha
from .image_gate import verify_bindings


def fixture(root, *, required="Lease.retain", verifier=True):
    site = root / "opt/venv/lib/python3.12/site-packages"
    source = site / "b12x/kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class Lease:\n def retain(self): pass\ndef compile_auxiliary(): pass\n"
    )
    contract = root / "opt/sparkring/contracts/vllm-connector-jobs-test.json"
    contract.parent.mkdir(parents=True)
    contract.write_text(
        json.dumps(
            dict(
                schema="sparkring-vllm-kv-block-lease-contract/v1",
                files=[
                    dict(
                        path="b12x/kernel.py",
                        sha256=sha(source.read_bytes()),
                        required_symbols=[required],
                    )
                ],
            )
        )
    )
    installed = root / "opt/sparkring/receipts/native-installed.json"
    installed.parent.mkdir(parents=True)
    installed.write_text(
        json.dumps({"active_contracts": ["/" + contract.relative_to(root).as_posix()]})
    )
    if verifier:
        path = site / "sparkcache/runtime_patches/verify_lease_contract.py"
        path.parent.mkdir(parents=True)
        path.write_text("""import json
class ContractError(RuntimeError): pass
def verify_contract(root, contract):
    assert root.name == "site-packages"
    data = json.loads(contract.read_text())
    names = data["files"][0]["required_symbols"]
    if names != ["Lease.retain"]:
        raise ContractError("invalid required symbol 'compile_auxiliary'")
    (root / "verifier-called").write_text(str(contract))
    return [root / "b12x/kernel.py"]
""")
    return site, contract


def test_image_gate_invokes_installed_verifier_on_active_contract(tmp_path):
    site, contract = fixture(tmp_path)
    assertions, failed = verify_bindings(tmp_path)
    assert not failed and assertions >= 2
    assert Path((site / "verifier-called").read_text()) == contract


def test_image_gate_catches_consumer_schema_error_despite_matching_file_hash(tmp_path):
    fixture(tmp_path, required="compile_auxiliary")
    _, failed = verify_bindings(tmp_path)
    assert any("invalid required symbol" in item for item in failed)


def test_image_gate_fails_closed_when_installed_verifier_is_missing(tmp_path):
    fixture(tmp_path, verifier=False)
    _, failed = verify_bindings(tmp_path)
    assert any("SparkCache lease verifier" in item for item in failed)
