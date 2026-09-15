"""Execute the installer inventory tail on temporary files, without an image."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize("entrypoint", ["verify-performance.py", "start-performance.py"])
def test_entrypoints_are_installed_attested_and_checked(tmp_path, entrypoint):
    source = tmp_path / "source"
    source.mkdir()
    (source / "verify.py").write_bytes(b"verify fixture")
    (source / "start.py").write_bytes(b"start fixture")
    def mapped_path(value):
        path = Path(value)
        return tmp_path / str(value).lstrip("/") if str(value).startswith("/opt/") else path
    for name in ("warmup_dflash.py", "serve-with-warmup.py", "startup_admission.py", "scheduler_liveness.py"):
        path = mapped_path("/opt/sparkring/bin/" + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    library = mapped_path("/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so")
    library.parent.mkdir(parents=True, exist_ok=True)
    library.write_bytes(b"library")
    mapped_path("/opt/sparkring/receipts").mkdir(parents=True)
    tree = ast.parse((HERE / "install.py").read_text())
    begin = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "files" for t in node.targets))
    scope = {"Path": mapped_path, "hashlib": hashlib, "json": json, "shutil": shutil,
             "SOURCE": source, "SITE": tmp_path / "site", "context": {"sparkcache_commit": "fixture"},
             "continuation_transform": {}, "attribution_transform": {}, "mhc_transform": {}}
    exec(compile(ast.Module(body=tree.body[begin:], type_ignores=[]), "installer-tail", "exec"), scope)
    receipt = json.loads(mapped_path("/opt/sparkring/receipts/mtp3-performance.json").read_text())
    name = "/opt/sparkring/bin/" + entrypoint
    # The sandbox paths are used consistently by the real installer and verifier.
    assert str(mapped_path(name)) in receipt["files"]
    verifier = ast.parse((HERE / "verify.py").read_text())
    body = [n for n in verifier.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    verification = compile(ast.Module(body=body, type_ignores=[]), "verifier", "exec")
    exec(verification, scope)
    mapped_path(name).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Runtime source differs"):
        exec(verification, scope)
