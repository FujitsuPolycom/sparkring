"""CPU contracts for immutable mesh image inputs and device-free verification."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(f"mesh_{name}", HERE / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


builder = module("build_image")
verifier = module("verify_mesh_image")


def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "transport.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = {"files": [{"path": "transport.py", "sha256": builder.sha256(root / "transport.py")} ]}
    path = root / builder.MANIFEST
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, builder.sha256(path)


def test_bundle_exact_files(tmp_path):
    root, digest = bundle(tmp_path)
    assert len(builder.verify_bundle(root, digest)) == 1


def test_bundle_requires_manifest_pin(tmp_path):
    root, _ = bundle(tmp_path)
    with pytest.raises(ValueError, match="manifest differs"):
        builder.verify_bundle(root, "0" * 64)


def test_bundle_rejects_modified_file(tmp_path):
    root, digest = bundle(tmp_path)
    (root / "transport.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its manifest"):
        builder.verify_bundle(root, digest)


def test_bundle_rejects_extra_file(tmp_path):
    root, digest = bundle(tmp_path)
    (root / "unexpected.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unmanifested"):
        builder.verify_bundle(root, digest)


@pytest.mark.parametrize("relative", ["../secret", "/etc/passwd", "a/../../b", "C:/file", "a\\b", ""])
def test_manifest_rejects_escaping_paths(relative):
    with pytest.raises(ValueError):
        builder.safe_relative(relative)


def test_manifest_accepts_relative_module():
    assert builder.safe_relative("b12x/comm/roce/api.py").as_posix() == "b12x/comm/roce/api.py"


def test_parent_requires_exact_arm64_identity():
    expected = {"image_id": "sha256:abc"}
    builder.validate_parent({"Id": "sha256:abc", "Architecture": "arm64", "Os": "linux"}, expected)
    with pytest.raises(ValueError, match="ID differs"):
        builder.validate_parent({"Id": "sha256:other", "Architecture": "arm64", "Os": "linux"}, expected)
    with pytest.raises(ValueError, match="linux/arm64"):
        builder.validate_parent({"Id": "sha256:abc", "Architecture": "amd64", "Os": "linux"}, expected)


def test_prepare_refuses_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="already exists"):
        builder.prepare(tmp_path, tmp_path)


def test_file_map_rejects_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        verifier.verify_file_map(tmp_path, {"../test": "0" * 64})


def test_container_verification_has_no_device_or_network_access():
    source = (HERE / "verify_mesh_image.py").read_text(encoding="utf-8")
    assert '"--network", "none"' in source
    assert '"--cap-drop", "ALL"' in source
    assert '"--read-only"' in source
    assert '"--gpus"' not in source
    assert '"--device"' not in source.split("def verify_external", 1)[1]
    assert '"--privileged"' not in source
    assert "torch.cuda.is_initialized()" in source


def test_dockerfile_does_not_change_parent_kernel_installation():
    source = (HERE / "Dockerfile").read_text(encoding="utf-8")
    assert "pip install" not in source
    assert "apt-get" not in source
    assert "--attach" not in source
    assert "-libverbs -lmlx5" in source
    assert "COPY bundle/ /opt/spark-sircl/" in source


def test_pins_use_native_mtp_only():
    pins = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
    assert pins["speculation"]["method"] == "mtp"
    assert pins["speculation"]["num_speculative_tokens"] == 3
    assert pins["target"]["repository"].endswith("-Spark")


def test_schema_accepts_research_only_status():
    schema = json.loads((HERE / "image-receipt.schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["status"]["const"] == "research-only"
    assert "image_id" in schema["required"]


def test_image_warmup_override_is_explicit_and_source_hashed():
    recipe = (HERE / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY warmup_dflash.py /opt/sparkring/bin/warmup_dflash.py" in recipe
    assert "ENV SPARKRING_WARMUP_TEMPERATURE=1" in recipe
    source = (HERE / "build_image.py").read_text(encoding="utf-8")
    assert '"warmup_dflash.py": HERE.parent' in source
    assert '"helper_sha256": sha256(context / "warmup_dflash.py")' in source


def test_warmup_verifier_requires_matching_source_and_temperature(tmp_path):
    helper = tmp_path / "warmup.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    digest = builder.sha256(helper)
    assert verifier.verify_warmup(helper, digest, {"SPARKRING_WARMUP_TEMPERATURE": "1"})["temperature"] == 1.0
    with pytest.raises(ValueError, match="temperature"):
        verifier.verify_warmup(helper, digest, {"SPARKRING_WARMUP_TEMPERATURE": "0"})
    with pytest.raises(ValueError, match="content pin"):
        verifier.verify_warmup(helper, "0" * 64, {"SPARKRING_WARMUP_TEMPERATURE": "1"})
