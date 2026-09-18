"""CPU-only wheel and source identity checks for native compilation."""

import json
import base64
import csv
import hashlib
import io
from pathlib import Path
import struct
import zipfile

import pytest

from . import native_worker, sources


def wheel(path, name="vllm", native=True, extra=None):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            name + "-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: " + name + "\nVersion: 1.0\n",
        )
        archive.writestr(name + "/__init__.py", "")
        if native:
            header = bytearray(64)
            header[:6] = b"\x7fELF\x02\x01"
            struct.pack_into("<H", header, 18, 183)
            archive.writestr(name + "/_C.so", header)
        if extra:
            archive.writestr(extra, b"invalid")


def test_native_wheel_records_source_package_and_binary(tmp_path):
    path = tmp_path / "vllm.whl"
    wheel(path)
    value = native_worker.wheel_record(path, "vllm")
    assert value["native_members"] == ["vllm/_C.so"]
    assert value["version"] == "1.0"
    assert len(value["sha256"]) == 64


@pytest.mark.parametrize(
    "extra",
    [
        "../escape",
        "/absolute",
        "C:/drive",
        "torch/tamper.py",
        "vllm-1.0.data/scripts/evil",
    ],
)
def test_wheel_cannot_modify_unowned_paths(tmp_path, extra):
    path = tmp_path / "vllm.whl"
    wheel(path, extra=extra)
    with pytest.raises(ValueError):
        native_worker.wheel_record(path, "vllm")


def test_source_only_wheel_is_not_a_native_build(tmp_path):
    path = tmp_path / "vllm.whl"
    wheel(path, native=False)
    with pytest.raises(ValueError, match="no compiled"):
        native_worker.wheel_record(path, "vllm")


def test_wrong_distribution_refused(tmp_path):
    path = tmp_path / "vllm.whl"
    wheel(path, name="another")
    with pytest.raises(ValueError, match="distribution identity"):
        native_worker.wheel_record(path, "vllm")


def test_standalone_worker_uses_controller_source_identity(tmp_path):
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine/a.py").write_bytes(b"x=1\n")
    assert native_worker.source_digest(tmp_path) == sources.tree_digest(tmp_path)


def test_native_recipe_rejects_unapproved_arch_before_compilation(tmp_path):
    path = tmp_path / "descriptor.json"
    path.write_text(
        json.dumps(
            dict(
                schema="sparkring-native-wheel-build/v1",
                sources={"vllm": {}, "b12x": {}},
                architecture="8.0",
            )
        )
    )
    with pytest.raises(ValueError, match="targets GB10"):
        native_worker.build(path, tmp_path / "source", tmp_path / "work")


def test_worker_pins_build_mode_in_real_wheel_commands_and_receipt(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    trees = {}
    for name in ("vllm", "b12x"):
        folder = source / name
        folder.mkdir(parents=True)
        (folder / "setup.py").write_text("# CPU-only build fixture\n")
        trees[name] = {"tree_sha256": native_worker.source_digest(folder)}
    descriptor = {
        "schema": "sparkring-native-wheel-build/v1",
        "sources": trees,
        "architecture": "12.1a",
        "jobs": 1,
        "torch_version": "2.13.0",
        "distribution_version": "1.0",
        "input_sha256": "a" * 64,
        "build_type": "Release",
    }
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps(descriptor))
    observed = []

    def compile_fixture(argv, **kwargs):
        assert argv[2:4] == ["pip", "wheel"]
        observed.append(kwargs["env"])
        name = Path(argv[-1]).name
        directory = Path(argv[argv.index("--wheel-dir") + 1])
        wheel(directory / (name + "-1.0.whl"), name=name, native=name == "vllm")
        if name == "vllm":
            (Path(argv[-1]) / "CMakeCache.txt").write_text(
                "CMAKE_PROJECT_NAME:STATIC=vllm_extensions\nCMAKE_BUILD_TYPE:STRING="
                + kwargs["env"]["CMAKE_BUILD_TYPE"]
                + "\n"
            )

    monkeypatch.setattr(native_worker, "run_checked", compile_fixture)
    monkeypatch.setattr(native_worker.metadata, "version", lambda name: "2.13.0")
    monkeypatch.setenv("CMAKE_BUILD_TYPE", "Debug")
    result = native_worker.build(path, source, tmp_path / "work")
    assert len(observed) == 2
    assert all(env["CMAKE_BUILD_TYPE"] == "Release" for env in observed)
    assert result["build_type"] == "Release"
    assert result["build_mode_evidence"]["source"] == "generated-cmake"


@pytest.mark.parametrize("mode", [None, "RelWithDebInfo"])
def test_missing_or_mismatched_generated_build_mode_is_rejected(tmp_path, mode):
    if mode is not None:
        (tmp_path / "CMakeCache.txt").write_text(
            "CMAKE_PROJECT_NAME:STATIC=vllm_extensions\nCMAKE_BUILD_TYPE:STRING="
            + mode
            + "\n"
        )
    with pytest.raises(ValueError, match="CMake"):
        native_worker.cmake_build_evidence(tmp_path, "Release")


def test_dependency_cmake_cache_cannot_prove_vllm_build_mode(tmp_path):
    (tmp_path / "CMakeCache.txt").write_text(
        "CMAKE_PROJECT_NAME:STATIC=dependency\nCMAKE_BUILD_TYPE:STRING=Release\n"
    )
    with pytest.raises(ValueError, match="Missing generated"):
        native_worker.cmake_build_evidence(tmp_path, "Release")


def metadata_wheel(path, *, dsl="4.7.1"):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "vllm-1.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: vllm\nVersion: 1\n"
            f"Requires-Dist: nvidia-cutlass-dsl[cu13]=={dsl}\n"
            "Requires-Dist: quack-kernels==0.6.5\n"
            "Requires-Dist: flashinfer-python==0.6.18.post1\n"
            "Requires-Dist: torchcodec>=0.14\n"
            'Requires-Dist: pynvvideocodec==2.0.4; extra == "video"\n',
        )
        archive.writestr("vllm-1.dist-info/RECORD", "")
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", header, 18, 183)
        archive.writestr("vllm/_C.so", header)
        archive.writestr("vllm/engine.py", "CODE = 'retained'\n")


def test_gb10_metadata_preserves_all_backend_dependencies_and_native_bytes(
    tmp_path, monkeypatch
):
    path = tmp_path / "vllm.whl"
    metadata_wheel(path)
    monkeypatch.setattr(
        native_worker.metadata,
        "version",
        lambda name: {"nvidia-cutlass-dsl": "4.6.2", "quack-kernels": "0.6.4"}[name],
    )
    before = native_worker.wheel_record(path, "vllm")
    result = native_worker.normalize_gb10_metadata(path, "gb10-dsl462-quack064")
    after = native_worker.wheel_record(path, "vllm")
    assert before["native_hashes"] == after["native_hashes"]
    assert after["requires_dist"] == [
        "nvidia-cutlass-dsl[cu13]==4.6.2",
        "quack-kernels==0.6.4",
        "flashinfer-python==0.6.18.post1",
        "torchcodec>=0.14",
        'pynvvideocodec==2.0.4; extra == "video"',
    ]
    assert result["original_sha256"] == before["sha256"]
    assert result["normalized_sha256"] == after["sha256"]
    with zipfile.ZipFile(path) as archive:
        for name, digest, size in csv.reader(
            io.StringIO(archive.read("vllm-1.dist-info/RECORD").decode())
        ):
            if digest:
                content = archive.read(name)
                assert int(size) == len(content)
                assert (
                    digest
                    == "sha256="
                    + base64.urlsafe_b64encode(hashlib.sha256(content).digest())
                    .rstrip(b"=")
                    .decode()
                )


def test_gb10_metadata_rejects_unreviewed_pin_without_rewriting_wheel(
    tmp_path, monkeypatch
):
    path = tmp_path / "vllm.whl"
    metadata_wheel(path, dsl="4.8.0")
    monkeypatch.setattr(
        native_worker.metadata,
        "version",
        lambda name: {"nvidia-cutlass-dsl": "4.6.2", "quack-kernels": "0.6.4"}[name],
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Unreviewed"):
        native_worker.normalize_gb10_metadata(path, "gb10-dsl462-quack064")
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "relocation",
    [
        "purelib/flashinfer_jit_cache/kernel.so",
        "scripts/override",
        "purelib/torch/evil.py",
    ],
)
def test_jit_cache_relocation_is_limited_to_its_own_package(tmp_path, relocation):
    path = tmp_path / "jit.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "flashinfer_jit_cache-1.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: flashinfer-jit-cache\nVersion: 1\n",
        )
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", header, 18, 183)
        archive.writestr("flashinfer_jit_cache-1.data/" + relocation, header)
    if relocation.startswith("purelib/flashinfer_jit_cache/"):
        assert (
            len(
                native_worker.wheel_record(path, "flashinfer-jit-cache")[
                    "native_members"
                ]
            )
            == 1
        )
    else:
        with pytest.raises(ValueError, match="relocate only"):
            native_worker.wheel_record(path, "flashinfer-jit-cache")
