"""Offline source, build-context and runtime-environment admission for SGLang."""

import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from runtime.images import sglang_extension as extension


OLD_REVISION = "fb2764a5cf321eaa5070ca8f9e892818f477c16d"
ADAPTER = b"""from pathlib import Path
import ctypes
import os

def _load_cudart():
    paths = ['/unselected/cuda/lib64/libcudart.so.13']
    seen, errors = set(), []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            return ctypes.CDLL(path)
        except OSError as error:
            errors.append(str(error))
    raise RuntimeError('; '.join(errors))
"""


def put(root, name, raw):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def git(root, *args):
    return subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "-c", "gc.auto=0", "-C", str(root), *args],
        stderr=subprocess.STDOUT, text=True).strip()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("Git archives the pinned source fixture")
    source = tmp_path / "adapter"
    source.mkdir()
    # A committed CRLF source is legitimate and must remain independent of Git's
    # checkout conversion settings on the machine preparing the context.
    boot = f"REVISION = '{OLD_REVISION}'\r\nVALUE = 'committed'\r\n".encode()
    put(source, "boot.py", boot)
    put(source, "adapter/engram_backend.py", ADAPTER.replace(b"\n", b"\r\n"))
    git(source, "init")
    git(source, "add", ".")
    git(source, "-c", "user.name=Source Fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "Add committed adapter fixture")
    revision = git(source, "rev-parse", "HEAD")
    runtime = tmp_path / "runtime"
    package = runtime / "combined-image"
    put(runtime, "entrypoint.py", b"# fixture entrypoint\n")
    put(runtime, "patch-multikey.py", b"# fixture authentication patch\n")
    put(runtime, "patches/fixture.patch", b"fixture patch\n")
    put(package, "Dockerfile", b"FROM fixture@sha256:abc\r\nCOPY . /fixture\r\n")
    put(package, "sglang-python", b"#!/bin/sh\nexec /opt/sglang/bin/python3 \"$@\"\n")
    nccl = put(tmp_path, "libnccl.so.2", b"pinned native-library fixture\n")
    manifest = {"id": "sglang-extension-fixture", "schema": "sparkring-sglang-extension/v1",
                "adapter": {"commit": revision}, "model": {"revision": "a" * 40},
                "nccl": {"sha256": extension.sha(nccl.read_bytes())}}
    put(package, "manifest.json", (json.dumps(manifest) + "\n").encode())
    monkeypatch.setattr(extension, "RUNTIME", runtime)
    monkeypatch.setattr(extension, "PACKAGE", package)
    return SimpleNamespace(source=source, runtime=runtime, package=package, nccl=nccl,
                           manifest=manifest, output=tmp_path / "context", boot=boot)


def test_archive_reads_committed_sources_and_ignores_dirty_untracked_files(fixture):
    (fixture.source / "boot.py").write_bytes(b"dirty tracked bytes\n")
    put(fixture.source, "untracked-secret.txt", b"must not enter context\n")
    archive = extension.adapter_sources(fixture.source, fixture.manifest)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        assert stream.extractfile("boot.py").read() == fixture.boot
        assert "untracked-secret.txt" not in stream.getnames()
    extension.prepare(fixture.source, fixture.nccl, fixture.output)
    assert not (fixture.output / "mia/untracked-secret.txt").exists()
    assert (fixture.source / "boot.py").read_bytes() == b"dirty tracked bytes\n"
    assert b"VALUE = 'committed'" in (fixture.output / "mia/boot.py").read_bytes()


def test_wrong_adapter_head_rejected_before_context_creation(fixture, monkeypatch):
    manifest = copy.deepcopy(fixture.manifest)
    manifest["adapter"]["commit"] = "f" * 40
    monkeypatch.setattr(extension, "pin", lambda: manifest)
    with pytest.raises(ValueError, match="HEAD differs"):
        extension.prepare(fixture.source, fixture.nccl, fixture.output)
    assert not fixture.output.exists()


def test_wrong_nccl_rejected_before_archiving_or_creating_context(fixture, monkeypatch):
    fixture.nccl.write_bytes(b"wrong native library")
    monkeypatch.setattr(extension, "adapter_sources", lambda *_: pytest.fail("Source archive was reached before NCCL admission"))
    with pytest.raises(ValueError, match="NCCL library differs"):
        extension.prepare(fixture.source, fixture.nccl, fixture.output)
    assert not fixture.output.exists()


def test_existing_context_is_preserved(fixture):
    put(fixture.output, "keep.txt", b"existing bytes")
    with pytest.raises(ValueError, match="must not already exist"):
        extension.prepare(fixture.source, fixture.nccl, fixture.output)
    assert (fixture.output / "keep.txt").read_bytes() == b"existing bytes"
    assert len(list(fixture.output.iterdir())) == 1


def test_adapter_revision_and_cuda_library_selection_with_crlf(fixture, monkeypatch):
    receipt = extension.prepare(fixture.source, fixture.nccl, fixture.output)
    boot = (fixture.output / "mia/boot.py").read_bytes()
    assert boot == fixture.boot.replace(OLD_REVISION.encode(), b"a" * 40)
    code = (fixture.output / "mia/adapter/engram_backend.py").read_bytes()
    assert b"\r\n" not in code
    namespace = {}
    exec(compile(code, "engram_backend.py", "exec"), namespace)
    calls = []
    namespace["ctypes"] = SimpleNamespace(CDLL=lambda path: calls.append(path) or "loaded")
    torch_site = fixture.runtime / "python/site-packages"
    namespace["torch"] = SimpleNamespace(__file__=str(torch_site / "torch/__init__.py"))
    selected = str(torch_site / "nvidia/cu13/lib/libcudart.so.13")
    monkeypatch.setenv("CUDA_HOME", "/foreign/vllm/cuda")
    assert namespace["_load_cudart"]() == "loaded"
    assert calls == [selected]
    monkeypatch.delenv("CUDA_HOME")
    assert namespace["_load_cudart"]() == "loaded"
    assert calls == [selected, selected]
    changes = receipt["adapter_modifications"]
    assert changes["boot.py"] == {"before": extension.sha(fixture.boot), "after": extension.sha(boot)}
    assert changes["adapter/engram_backend.py"] == {
        "before": extension.sha(ADAPTER.replace(b"\n", b"\r\n")), "after": extension.sha(code)}


@pytest.mark.parametrize("replacement", [b"REVISION = 'different'\n", b"REVISION = '" + OLD_REVISION.encode() + b"'\n" + b"REVISION = '" + OLD_REVISION.encode() + b"'\n"])
def test_adapter_revision_guard_rejects_missing_or_duplicate_assignment(fixture, replacement):
    (fixture.source / "boot.py").write_bytes(replacement)
    with pytest.raises(ValueError, match="revision assignment differs"):
        extension.adapt_mia(fixture.source, fixture.manifest)
    assert (fixture.source / "boot.py").read_bytes() == replacement


def test_context_receipt_binds_every_materialized_build_input(fixture):
    receipt = extension.prepare(fixture.source, fixture.nccl, fixture.output)
    assert receipt["composition"] == fixture.manifest
    assert json.loads((fixture.output / "runtime/context.json").read_bytes()) == receipt
    actual = {path.relative_to(fixture.output).as_posix(): extension.sha(path.read_bytes())
              for path in fixture.output.rglob("*") if path.is_file()
              and path != fixture.output / "runtime/context.json"}
    assert receipt["inputs"] == actual
    assert actual["libnccl.so.2"] == fixture.manifest["nccl"]["sha256"]
    assert "Dockerfile" in actual and ".dockerignore" in actual
    assert (fixture.output / "Dockerfile").read_bytes() == b"FROM fixture@sha256:abc\nCOPY . /fixture\n"


@pytest.mark.parametrize("name,kind", [("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                                       ("nested/../../escape", tarfile.REGTYPE),
                                       ("C:/escape", tarfile.REGTYPE), ("C:escape", tarfile.REGTYPE),
                                       ("nested\\escape", tarfile.REGTYPE),
                                       ("symlink", tarfile.SYMTYPE), ("hardlink", tarfile.LNKTYPE),
                                       ("device", tarfile.CHRTYPE)])
def test_archive_paths_and_links_rejected_before_context_creation(fixture, monkeypatch, name, kind):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        member = tarfile.TarInfo(name)
        member.type = kind
        if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            member.linkname = "../escape"
        stream.addfile(member)
    monkeypatch.setattr(extension, "adapter_sources", lambda *_: archive.getvalue())
    with pytest.raises(ValueError, match="unsupported path or link"):
        extension.prepare(fixture.source, fixture.nccl, fixture.output)
    assert not fixture.output.exists()


def test_runtime_wrapper_clears_vllm_environment_and_preserves_arguments():
    wrapper = Path(__file__).parents[1] / "deepseek-v41-sglang/combined-image/sglang-python"
    if os.name == "nt":
        bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        if not bash.exists():
            pytest.skip("Git Bash is required to execute the POSIX runtime wrapper on Windows")
    else:
        bash = shutil.which("bash")
        if not bash:
            pytest.skip("Bash is required to inspect the wrapper without its installed Python")
    # Preserve only process-startup necessities, never unrelated local secrets.
    env = {key: value for key, value in os.environ.items()
           if key.upper() in {"SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "PATH"}}
    cleared = ("PYTHONHOME", "LD_PRELOAD", "NCCL_ROOT", "NCCL_INCLUDE_DIR", "NCCL_LIB_DIR",
               "VLLM_NCCL_SO_PATH", "NCCL_LOCAL_INFERENCE_PATH", "PYTORCH_HOME",
               "TORCHINDUCTOR_CUTLASS_DIR", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "TRITON_CUDACRT_PATH",
               "TRITON_CUDART_PATH", "TRITON_CUPTI_LIB_PATH", "TRITON_CUPTI_INCLUDE_PATH")
    replaced = ("PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "VIRTUAL_ENV",
                "TORCH_EXTENSIONS_DIR", "SGLANG_CACHE_DIR", "TRITON_CACHE_DIR", "TRITON_PTXAS_PATH",
                "TRITON_CUOBJDUMP_PATH", "TRITON_NVDISASM_PATH", "TORCH_CUDA_ARCH_LIST")
    # Export the foreign runtime values after Bash starts so an invalid preload
    # cannot prevent the shell from reaching the wrapper being tested.
    script = "; ".join(f"export {key}=/foreign/vllm/value" for key in cleared + replaced)
    script += '; exec() { command env; printf "\\nSPARKRING_TEST_ARGS\\n"; printf "%s\\n" "$@"; }; source "$1" -c "print(1)" "space argument"'
    result = subprocess.run([str(bash), "--noprofile", "--norc", "-c", script, "wrapper-test", wrapper.as_posix()],
                            env=env, capture_output=True, text=True, check=True)
    environment, args = result.stdout.split("\nSPARKRING_TEST_ARGS\n")
    values = dict(line.split("=", 1) for line in environment.splitlines() if "=" in line)
    assert all(key not in values for key in cleared)
    assert values["VIRTUAL_ENV"] == "/opt/sglang"
    assert values["CUDA_HOME"] == values["CUDA_PATH"] == "/usr/local/cuda-13.0"
    assert values["PYTHONPATH"] == "/opt/dsv41/adapter:/opt/sglang/lib/python3.12/site-packages"
    assert values["PYTHONNOUSERSITE"] == "1"
    assert values["LD_LIBRARY_PATH"] == "/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
    assert values["SGLANG_RUST_BUILD_MODE"] == "never"
    for name, tool in (("TRITON_PTXAS_PATH", "ptxas"), ("TRITON_CUOBJDUMP_PATH", "cuobjdump"),
                       ("TRITON_NVDISASM_PATH", "nvdisasm")):
        assert values[name] == f"/usr/local/cuda-13.0/bin/{tool}"
    assert values["TORCH_CUDA_ARCH_LIST"] == "12.1"
    assert values["PATH"].startswith("/opt/sglang/bin:/usr/local/cuda-13.0/bin:")
    for name in ("SGLANG_CACHE_DIR", "SGLANG_DG_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "TORCH_EXTENSIONS_DIR",
                 "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        assert values[name].startswith("/root/.cache/sparkring/sglang")
    assert args.splitlines() == ["/opt/sglang/bin/python3", "-c", "print(1)", "space argument"]
