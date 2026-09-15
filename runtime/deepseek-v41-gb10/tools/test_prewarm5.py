"""CPU-only checks for the FlashInfer prewarm exit status."""

import runpy
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


BUILDER = Path(__file__).parents[1] / "build-image.sh"


def _shell_path(path):
    value = str(path.resolve()).replace("\\", "/")
    return "/mnt/" + value[0].lower() + value[2:] if os.name == "nt" else value


def _fragment(tmp_path, program):
    prefix = "\n".join(("set -uo pipefail", "TAG=fixture", "HERE=/unused", "VERIFY=1",
        "LOG=" + shlex.quote(_shell_path(tmp_path / "log")),
        "SRC=/unused", "CTX=/unused", "log() { :; }",
        'fail() { echo "FAILED: $*" >&2; exit 78; }'))
    script = tmp_path / "fragment.sh"
    script.write_text(prefix + "\n" + program + "\necho BUILD_OK\n", encoding="utf-8", newline="\n")
    return subprocess.run(["bash", _shell_path(script)],
                          text=True, capture_output=True, timeout=10)


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash unavailable")
def test_failed_container_inventory_cannot_select_idle_build_defaults(tmp_path):
    text = BUILDER.read_text(encoding="utf-8")
    fragment = text.split("# Cgroup sizing.", 1)[1].split('SRC="$WORK/vllm-src"', 1)[0]
    fragment = "# Cgroup sizing." + fragment
    result = _fragment(tmp_path, 'docker() { return 23; }\n' + fragment)
    assert result.returncode != 0 and "BUILD_OK" not in result.stdout
    assert "cannot inspect running containers" in result.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash unavailable")
def test_failed_final_source_copy_cannot_reach_build(tmp_path):
    text = BUILDER.read_text(encoding="utf-8")
    command = next(line for line in text.splitlines() if line.lstrip().startswith("rsync -a"))
    result = _fragment(tmp_path, 'rsync() { return 23; }\n' + command)
    assert result.returncode == 78 and "BUILD_OK" not in result.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash unavailable")
@pytest.mark.parametrize("exit_status", [0, 23])
def test_optional_verification_propagates_inner_pipeline_failure(tmp_path, exit_status):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    timeout = fake_bin / "timeout"
    timeout.write_text(f'#!/bin/sh\necho "VERIFY fixture"\nexit {exit_status}\n', encoding="utf-8", newline="\n")
    timeout.chmod(0o755)
    text = BUILDER.read_text(encoding="utf-8")
    fragment = 'if [ "${VERIFY:-0}" = 1 ]; then' + text.split('if [ "${VERIFY:-0}" = 1 ]; then', 1)[1].split("\nfi", 1)[0] + "\nfi"
    shim = 'export PATH=' + shlex.quote(_shell_path(fake_bin)) + ':"$PATH"\n'
    # Execute only the requested Bash snippet; never invoke a container engine.
    shim += 'docker() { while [[ "$1" != "$TAG:overlay5" ]]; do shift; done; shift; bash "$@"; }\n'
    result = _fragment(tmp_path, shim + fragment)
    assert result.returncode == (0 if exit_status == 0 else 78), result.stderr
    assert ("BUILD_OK" in result.stdout) == (exit_status == 0)


@pytest.mark.parametrize("failure", ["sparse", "gemm-missing", "success"])
def test_prewarm_requires_sparse_success_and_compiled_gemm(monkeypatch, failure):
    calls = []

    def sparse():
        calls.append("sparse")
        if failure == "sparse":
            raise RuntimeError("compiler failed")
        return object()

    def gemm():
        calls.append("gemm")
        return SimpleNamespace(is_compiled=lambda: failure != "gemm-missing")

    for name in ("flashinfer", "flashinfer.mla", "flashinfer.jit"):
        stub = ModuleType(name)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, name, stub)
    sparse_module = ModuleType("flashinfer.mla._sparse_mla_sm120")
    sparse_module.get_sparse_mla_sm120_module = sparse
    gemm_module = ModuleType("flashinfer.jit.gemm")
    gemm_module.gen_gemm_sm120_module_cutlass_mxfp8 = gemm
    monkeypatch.setitem(sys.modules, sparse_module.__name__, sparse_module)
    monkeypatch.setitem(sys.modules, gemm_module.__name__, gemm_module)
    with pytest.raises(SystemExit) as result:
        runpy.run_path(
            str(Path(__file__).with_name("prewarm5.py")), run_name="__main__"
        )
    assert result.value.code == (0 if failure == "success" else 1)
    assert calls == (["sparse"] if failure == "sparse" else ["sparse", "gemm"])
