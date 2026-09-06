"""Authenticate staged files before any deployment helper can execute."""

import hashlib
import json
import subprocess
import sys

import pytest

from scripts.deploy_engine import plan_digest
from scripts.deploy_trust import trusted_check, trusted_script


def fixture(tmp_path):
    root = tmp_path / "workspace"
    source = root / "source"
    (source / "scripts").mkdir(parents=True)
    script = source / "scripts/entry.py"
    script.write_text("print('executed verified helper')\n")
    launch = root / "launch"
    launch.mkdir()
    names = ["site.json", "fabric.json", "launch-rank.sh", "fabric-plan.json"]
    names += [f"rank{i}.env" for i in range(4)]
    for name in names:
        (launch / name).write_text("{}" if name.endswith(".json") else "original")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    document = {
        "spec": {"workspace": str(root)},
        "source": {"files": {"scripts/entry.py": sha(script)}},
        "launch_files": {name: sha(launch / name) for name in names},
    }
    (root / "preparation.json").write_text(json.dumps(document))
    return root, document


def run(argv):
    return subprocess.run(argv, capture_output=True, text=True, timeout=15)


def test_verified_script_runs_after_checks(tmp_path):
    root, document = fixture(tmp_path)
    argv = trusted_script(
        str(root), plan_digest(document), "scripts/entry.py", [], python=sys.executable
    )
    assert argv[1:3] == ["-I", "-B"]
    result = run(argv)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "executed verified helper"


def test_substituted_verifier_cannot_execute(tmp_path):
    root, document = fixture(tmp_path)
    marker = tmp_path / "executed"
    (root / "source/scripts/entry.py").write_text(
        f"from pathlib import Path; Path({str(marker)!r}).touch(); print('{{\"verified\":true}}')"
    )
    result = run(
        trusted_script(
            str(root),
            plan_digest(document),
            "scripts/entry.py",
            [],
            python=sys.executable,
        )
    )
    assert result.returncode != 0
    assert not marker.exists()
    assert "source file changed" in result.stderr


@pytest.mark.parametrize("change", ["changed_env", "removed_entry", "extra_file"])
def test_launch_map_cannot_reauthorize_changed_files(tmp_path, change):
    root, document = fixture(tmp_path)
    if change == "extra_file":
        (root / "launch/unreviewed.env").write_text("changed")
    else:
        (root / "launch/rank0.env").write_text("changed")
        (root / "launch/fabric-plan.json").write_text(json.dumps({"files": {}}))
        if change == "removed_entry":
            del document["launch_files"]["rank0.env"]
            (root / "preparation.json").write_text(json.dumps(document))
    result = run(trusted_check(str(root), plan_digest(document), python=sys.executable))
    assert result.returncode != 0


def test_missing_launch_allowed_only_for_source_preparation(tmp_path):
    root, document = fixture(tmp_path)
    del document["launch_files"]
    (root / "preparation.json").write_text(json.dumps(document))
    digest = plan_digest(document)
    assert run(trusted_check(str(root), digest, python=sys.executable)).returncode != 0
    assert (
        run(
            trusted_check(
                str(root), digest, require_launch=False, python=sys.executable
            )
        ).returncode
        == 0
    )


def test_symlinked_source_parent_is_rejected(tmp_path):
    root, document = fixture(tmp_path)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    result = run(
        trusted_check(
            str(root),
            plan_digest(document),
            source_root=str(alias / "source"),
            python=sys.executable,
        )
    )
    assert result.returncode != 0
    assert "symlink" in result.stderr


def test_unlisted_python_initializer_is_rejected_before_import(tmp_path):
    root, document = fixture(tmp_path)
    (root / "source/scripts/__init__.py").write_text("raise RuntimeError('injected')")
    result = run(
        trusted_script(
            str(root),
            plan_digest(document),
            "scripts/entry.py",
            [],
            python=sys.executable,
        )
    )
    assert result.returncode != 0
    assert "Unlisted executable source" in result.stderr


def test_large_source_map_is_not_embedded_in_process_arguments(tmp_path):
    root, document = fixture(tmp_path)
    document["source"]["files"].update(
        {f"scripts/file{i}.py": "a" * 64 for i in range(5000)}
    )
    argv = trusted_check(str(root), plan_digest(document), python=sys.executable)
    assert sum(map(len, argv)) < 20000


def test_untrusted_cached_bytecode_cannot_replace_verified_source(tmp_path):
    import os
    import py_compile

    root, document = fixture(tmp_path)
    module = root / "source/scripts/dep.py"
    clean, injected = "value = 'clean'\n", "value = 'other'\n"
    module.write_text(injected)
    timestamp = module.stat().st_mtime_ns
    py_compile.compile(str(module), doraise=True)
    module.write_text(clean)
    os.utime(module, ns=(timestamp, timestamp))
    document["source"]["files"]["scripts/dep.py"] = hashlib.sha256(
        module.read_bytes()
    ).hexdigest()
    (root / "preparation.json").write_text(json.dumps(document))
    result = run(
        trusted_check(
            str(root),
            plan_digest(document),
            python=sys.executable,
            after="from scripts.dep import value\nprint(value)",
        )
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "clean"


def test_ready_plan_checks_controller_source_before_import(tmp_path, monkeypatch):
    from scripts import deploy_runtime
    from scripts.test_deploy_runtime import prepared

    root, files = fixture(tmp_path)
    document = prepared()
    document["source"] = files["source"]
    document["launch_files"] = files["launch_files"]
    document["controller_launch"] = str(root / "launch")
    document["controller_source"] = str(root / "source")
    helper = root / "source/scripts/deploy_runtime.py"
    marker = tmp_path / "executed-unverified-helper"
    helper.write_text("def probe_readiness(*args): return {'ready': True}\n")
    document["source"]["files"]["scripts/deploy_runtime.py"] = hashlib.sha256(
        helper.read_bytes()
    ).hexdigest()
    (root / "prepared.json").write_text(json.dumps(document))
    monkeypatch.setattr(deploy_runtime, "ROOT", root / "source")
    plan = deploy_runtime.build_runtime_plan(document, "ready")
    helper.write_text(f"from pathlib import Path; Path({str(marker)!r}).touch()\n")
    result = run(plan["phases"][0]["actions"][0]["argv"])
    assert result.returncode != 0
    assert not marker.exists()
    assert "source file changed" in result.stderr


@pytest.mark.parametrize("filename", ["subprocess.py", "unlisted/__init__.py"])
def test_root_level_and_unlisted_package_shadowing_is_rejected(tmp_path, filename):
    root, document = fixture(tmp_path)
    path = root / "source" / filename
    path.parent.mkdir(exist_ok=True)
    path.write_text("raise RuntimeError('untrusted module executed')")
    result = run(
        trusted_script(
            str(root),
            plan_digest(document),
            "scripts/entry.py",
            [],
            python=sys.executable,
        )
    )
    assert result.returncode != 0
    assert "Unlisted executable source" in result.stderr
    assert "untrusted module executed" not in result.stderr


def test_trusted_script_preserves_authenticated_sibling_imports(tmp_path):
    root, document = fixture(tmp_path)
    source = root / "source/scripts"
    (source / "entry.py").write_text("from sibling import value\nprint(value)\n")
    (source / "sibling.py").write_text("value='verified sibling'\n")
    document["source"]["files"] = {
        "scripts/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.iterdir()
    }
    (root / "preparation.json").write_text(json.dumps(document))
    result = run(
        trusted_script(
            str(root),
            plan_digest(document),
            "scripts/entry.py",
            [],
            python=sys.executable,
        )
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "verified sibling"


def test_ready_executes_from_extracted_controller_snapshot(tmp_path):
    import io
    import tarfile
    from pathlib import Path
    from scripts import deploy_runtime
    from scripts.deploy_stage import extract_source
    from scripts.test_deploy_runtime import prepared

    root, launch = fixture(tmp_path)
    payloads = {
        name: (deploy_runtime.ROOT / name).read_bytes()
        for name in (
            "scripts/deploy_runtime.py",
            "scripts/deploy_engine.py",
            "scripts/deploy_trust.py",
        )
    }
    # Only the host probe is simulated; the staged readiness coordinator executes.
    payloads["runtime/glm53-spark-mtp3-mesh/wait_managed_ready.py"] = (
        "def load_launch(path): return {'launch':str(path)}\n"
        "def wait(plan,timeout): return {'schema':'sparkring-managed-model-readiness/v1','ready':True}\n"
    ).encode()
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        for name, data in payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            target.addfile(member, io.BytesIO(data))
    file_map = {
        name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()
    }
    snapshot = tmp_path / "controller-source"
    extract_source(archive, snapshot, file_map)
    document = prepared()
    document.update(
        source={"files": file_map},
        controller_source=str(snapshot),
        controller_launch=str(root / "launch"),
        launch_files=launch["launch_files"],
    )
    (root / "prepared.json").write_text(json.dumps(document))
    plan = deploy_runtime.build_runtime_plan(document, "ready")
    result = run(plan["phases"][0]["actions"][0]["argv"])
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["ready"] is True
    assert Path(output["receipt"]).is_file()


def test_install_uses_same_process_isolated_verification_and_execution():
    from scripts.deploy_runtime import build_runtime_plan
    from scripts.test_deploy_runtime import prepared

    plan = build_runtime_plan(prepared(), "install")
    for action in plan["phases"][0]["actions"]:
        argv = action["argv"]
        assert argv[:5] == ["sudo", "-n", "python3", "-I", "-B"]
        program = argv[argv.index("-c") + 1]
        assert "_verify_preparation(*config)" in program
        assert "runpy.run_path" in program
        assert "sys.pycache_prefix=cache" in program
