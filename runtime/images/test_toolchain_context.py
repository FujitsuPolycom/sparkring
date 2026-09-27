import json
import io
import tarfile

import pytest

from runtime.images import toolchain_context as context
from runtime.images.toolchain_runtime import configure_environment, verify_loaded_libraries


def test_candidate_library_replaces_profile_nccl_override():
    original = {"LD_PRELOAD": "/opt/local-inference/nccl/lib/libnccl.so.2:/lib/other.so",
                "VLLM_NCCL_SO_PATH": "/old/libnccl.so.2", "PATH": "/usr/bin",
                "LD_LIBRARY_PATH": "/old/lib", "SIRCL_ENABLED": "1"}
    result = configure_environment({"variant": "combined", "cuda": {"version": "13.4.2"}}, original)
    expected = "/opt/sparkring/toolchain/nccl/lib/libnccl.so.2"
    assert result["LD_PRELOAD"].split()[0] == expected
    assert result["LD_PRELOAD"].split()[-1] == "/lib/other.so"
    assert "/opt/local-inference/nccl/lib/libnccl.so.2" not in result["LD_PRELOAD"]
    assert result["VLLM_NCCL_SO_PATH"] == expected
    assert result["NCCL_LOCAL_INFERENCE_PATH"] == expected
    assert result["SIRCL_ENABLED"] == "1"
    assert original["VLLM_NCCL_SO_PATH"] == "/old/libnccl.so.2"
    assert result["TRITON_PTXAS_PATH"] == "/usr/local/cuda-13.4/bin/ptxas"


def test_cuda_control_keeps_nccl_selection():
    original = {"LD_PRELOAD": "/old/libnccl.so.2", "VLLM_NCCL_SO_PATH": "/old/libnccl.so.2"}
    result = configure_environment({"variant": "cuda", "cuda": {"version": "13.4.2"}}, original)
    assert original["LD_PRELOAD"] in result["LD_PRELOAD"].split()
    assert result["VLLM_NCCL_SO_PATH"] == original["VLLM_NCCL_SO_PATH"]


def test_nccl_control_keeps_compiler():
    original = {"CUDA_HOME": "/old/cuda", "TRITON_PTXAS_PATH": "/old/ptxas"}
    result = configure_environment({"variant": "nccl"}, original)
    assert result["CUDA_HOME"] == "/old/cuda"
    assert result["TRITON_PTXAS_PATH"] == "/old/ptxas"
    assert result["PYTHONPATH"].split(":")[0] == "/opt/sparkring/toolchain/python"


def test_library_environment_is_idempotent_and_overrides_bundled_cuda():
    lock = {"variant": "combined", "cuda": {"version": "13.4.2"}}
    original = {"PYTHONPATH": "", "CUDA_VERSION": "13.0.2",
                "LD_PRELOAD": "/vendor/libcublas.so.13:/vendor/libcudart.so.13"}
    result = configure_environment(lock, original)
    assert configure_environment(lock, result) == result
    assert "/vendor/" not in result["LD_PRELOAD"]
    assert result["CUDA_VERSION"] == "13.4.2"
    assert result["PYTHONPATH"] == "/opt/sparkring/toolchain/python"


def test_loaded_library_gate_rejects_mixed_cuda_and_nccl():
    for path in ("/vendor/libcublasLt.so.13", "/vendor/libcudart.so.13", "/old/libnccl.so.2"):
        with pytest.raises(ValueError, match="Unselected"):
            verify_loaded_libraries({"variant": "combined"}, "0-1 r-xp 0 0 0 " + path)


def test_loaded_library_gate_accepts_selected_paths():
    paths = ["/usr/local/cuda-13.4/targets/sbsa-linux/lib/libcudart.so.13.4.92",
             "/opt/sparkring/toolchain/nccl/lib/libnccl.so.2.32.3"]
    assert verify_loaded_libraries({"variant": "combined"}, "\n".join(paths)) == sorted(paths)


def test_prepare_rejects_unbound_archive_without_output(tmp_path):
    archive = tmp_path / "wrong.tar"
    archive.write_bytes(b"wrong source")
    output = tmp_path / "context"
    with pytest.raises(ValueError, match="source archive hash differs"):
        context.prepare(context.HERE / "cuda134-nccl232.json", archive, output)
    assert not output.exists()


@pytest.mark.parametrize("variant", ["combined", "nccl", "cuda"])
def test_vendor_search_aliases_bind_selected_libraries(variant):
    from runtime.images.toolchain_runtime import vendor_search_aliases

    aliases = vendor_search_aliases({"variant": variant})
    assert ("nvidia/nccl/lib" in aliases) == (variant != "cuda")
    assert ("nvidia/cu13/lib" in aliases) == (variant != "nccl")
    if variant != "cuda":
        assert aliases["nvidia/nccl/lib"] == "/opt/sparkring/toolchain/nccl/lib"


def test_prepare_rejects_crlf_source_even_when_its_hash_matches(tmp_path):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as output_tar:
        data = b"// windows-exported source\r\n"
        entry = tarfile.TarInfo("src/transport/generic.cc")
        entry.size = len(data)
        output_tar.addfile(entry, io.BytesIO(data))
    lock = json.loads((context.HERE / "cuda134-nccl232.json").read_text())
    lock["nccl"]["archive_sha256"] = context.digest(archive)
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    output = tmp_path / "context"
    with pytest.raises(ValueError, match="requires LF"):
        context.prepare(lock_path, archive, output)
    assert not output.exists()


@pytest.mark.parametrize("variant", ["combined", "cuda", "nccl"])
def test_variants_have_distinct_layers(tmp_path, variant):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as output_tar:
        for name in ("src/transport/generic.cc", "src/transport/net_ib/connect.cc"):
            data = b"// fixture\n"
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            output_tar.addfile(entry, io.BytesIO(data))
    lock = json.loads((context.HERE / "cuda134-nccl232.json").read_text())
    lock["nccl"]["archive_sha256"] = context.digest(archive)
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    output = tmp_path / "context"
    context.prepare(lock_path, archive, output, variant)
    dockerfile = (output / "Dockerfile").read_text()
    assert b"\r" not in (output / "build-nccl.sh").read_bytes()
    assert ("AS cuda134" in dockerfile) == (variant != "nccl")
    assert ("AS nccl_build" in dockerfile) == (variant != "cuda")
    assert lock["parent"]["reference"] in dockerfile
    assert ("nvidia/nccl/lib" in dockerfile) == (variant != "cuda")
    assert ("nvidia/cu13/lib" in dockerfile) == (variant != "nccl")
    assert "ENV PYTHONPATH=/opt/sparkring/toolchain/python:" in dockerfile
    with pytest.raises(ValueError, match="already exists"):
        context.prepare(lock_path, archive, output, variant)
