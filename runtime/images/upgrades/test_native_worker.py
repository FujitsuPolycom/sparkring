"""CPU-only wheel and source identity checks for native compilation."""

import json
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
