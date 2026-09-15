"""Source identity and planner boundary checks for the checkpoint payload."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("checkpoint_installer", ROOT / "install.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
MANIFEST_SHA = "0970d29ec33e9f8525a2cc55989ab0deb937bff88b5f16d4d5280035359e4c55"


def test_packaged_payload_identity_and_syntax():
    result = subprocess.run([sys.executable, str(ROOT / "install.py"), "verify-context",
                             "--manifest-sha256", MANIFEST_SHA], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["payloads_verified"] == 18


def test_preimage_mismatch_and_unexpected_file_are_rejected(tmp_path):
    path = tmp_path / "module.py"
    path.write_bytes(b"source")
    installer.verify_files({"/module.py": installer.sha(b"source")}, root=tmp_path)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        installer.verify_files({"/module.py": installer.sha(b"different")}, root=tmp_path)
    with pytest.raises(RuntimeError, match="Expected absent"):
        installer.verify_files({"/module.py": None}, root=tmp_path)


def planner():
    manifest = json.loads((ROOT / "patch-manifest.json").read_bytes())
    row = next(v for k, v in manifest["replacements"].items()
               if k.endswith("/recurrent_prefill_checkpoint.py"))
    namespace = {}
    exec(compile((ROOT / row["payload"]).read_bytes(), "checkpoint_planner", "exec"), namespace)
    return namespace


def test_fresh_prompt_exports_publication_and_predecessor():
    functions = planner()
    plan = functions["fresh_prompt_plan"](start=0, end=8192, prompt=8192,
        num_tokens=8192, block_size=512, publications=(6144,), shared_prefix_boundary=0)
    # The planner adds the predecessor checkpoint two blocks before prompt end;
    # metadata indexes the zero-based block ending at each checkpoint boundary.
    assert plan == (0, 8192, (6144, 7168))
    assert functions["checkpoint_metadata"](plan, 0, 8192, 512, 2) == ([6144, 7168], [11, 13])
    with pytest.raises(ValueError, match="actual query span"):
        functions["checkpoint_metadata"](plan, 512, 8192, 512, 2)
    with pytest.raises(ValueError, match="capacity"):
        functions["checkpoint_metadata"](plan, 0, 8192, 512, 1)


@pytest.mark.parametrize("start,end,prompt,tokens", [(512, 8192, 8192, 7680),
    (0, 16384, 16384, 16384), (0, 8191, 8191, 8191), (0, 4096, 8192, 4096)])
def test_ineligible_prefill_planner_returns_none(start, end, prompt, tokens):
    assert planner()["fresh_prompt_plan"](start=start, end=end, prompt=prompt,
        num_tokens=tokens, block_size=512, publications=()) is None
