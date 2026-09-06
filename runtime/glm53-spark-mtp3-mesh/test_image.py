"""CPU contracts for immutable mesh image inputs and device-free verification."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_layered_file_map_checks_unchanged_parent_and_exact_overrides(tmp_path):
    (tmp_path / "vllm").mkdir()
    unchanged = tmp_path / "vllm" / "unchanged.py"
    replaced = tmp_path / "vllm" / "replaced.py"
    unchanged.write_text("parent\n", encoding="utf-8")
    replaced.write_text("result\n", encoding="utf-8")
    parent = {
        "vllm/unchanged.py": verifier.sha256(unchanged),
        "vllm/replaced.py": "0" * 64,
    }
    overrides = {
        "vllm/replaced.py": {
            "base_sha256": "0" * 64,
            "result_sha256": verifier.sha256(replaced),
        }
    }
    assert verifier.verify_layered_file_map(tmp_path, parent, overrides) == {
        "parent_files": 2,
        "overrides": 1,
    }
    unchanged.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content pin"):
        verifier.verify_layered_file_map(tmp_path, parent, overrides)


def test_layered_file_map_rejects_unbound_or_wrong_base_override(tmp_path):
    (tmp_path / "vllm").mkdir()
    source = tmp_path / "vllm" / "source.py"
    source.write_text("result\n", encoding="utf-8")
    parent = {"vllm/source.py": "1" * 64}
    with pytest.raises(ValueError, match="base identity"):
        verifier.verify_layered_file_map(
            tmp_path,
            parent,
            {"vllm/source.py": {
                "base_sha256": "2" * 64,
                "result_sha256": verifier.sha256(source),
            }},
        )
    with pytest.raises(ValueError, match="absent from the parent"):
        verifier.verify_layered_file_map(
            tmp_path,
            parent,
            {"vllm/other.py": {
                "base_sha256": "1" * 64,
                "result_sha256": verifier.sha256(source),
            }},
        )


def test_complete_package_file_map_rejects_unmanifested_module(tmp_path):
    package = tmp_path / "b12x"
    package.mkdir()
    declared = package / "declared.py"
    declared.write_text("VALUE = 1\n", encoding="utf-8")
    records = {"b12x/declared.py": verifier.sha256(declared)}
    assert verifier.verify_complete_package_file_map(tmp_path, "b12x", records) == 1
    (package / "unmanifested.py").write_text("VALUE = 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="complete manifest"):
        verifier.verify_complete_package_file_map(tmp_path, "b12x", records)


def test_compute_environment_requires_cuda_and_quantization_contract():
    expected = {
        "CUDA_HOME": "/opt/cuda-13.3",
        "TRITON_PTXAS_PATH": "/opt/cuda-13.3/bin/ptxas",
        "VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH": "1",
        "VLLM_B12X_DENSE_ACTIVATION_MODE": "auto",
        "VLLM_MTP_NVFP4_LM_HEAD": "1",
        "VLLM_LM_HEAD_A16": "1",
        "VLLM_MXFP8_LM_HEAD": "0",
    }
    assert verifier.verify_required_environment(expected, expected) == expected
    for name in expected:
        changed = dict(expected)
        changed[name] = "wrong"
        with pytest.raises(ValueError, match=name):
            verifier.verify_required_environment(changed, expected)


def test_compute_verifier_composes_parent_vllm_and_complete_b12x(tmp_path):
    compute = tmp_path / "compute"
    receipts = tmp_path / "receipts"
    site = tmp_path / "site"
    cuda = tmp_path / "cuda"
    for path in (compute, receipts, site / "vllm", site / "b12x", cuda / "bin"):
        path.mkdir(parents=True, exist_ok=True)
    unchanged = site / "vllm" / "unchanged.py"
    override = site / "vllm" / "override.py"
    b12x_python = site / "b12x" / "__init__.py"
    b12x_notice = site / "b12x" / "README.md"
    unchanged.write_text("unchanged\n", encoding="utf-8")
    override.write_text("result\n", encoding="utf-8")
    b12x_python.write_text("", encoding="utf-8")
    b12x_notice.write_text("source\n", encoding="utf-8")
    (cuda / "bin" / "ptxas").write_text("tool\n", encoding="utf-8")
    components = {"cuda_nvcc/archive.tar.xz": "a" * 64}
    (cuda / "sparkring-component-manifest.json").write_text(
        json.dumps(components), encoding="utf-8"
    )
    base_hash = "b" * 64
    lock = {
        "schema": "sparkring-glm53-compute-source/v1",
        "vllm": {
            "base_revision": "e02",
            "files": [["vllm/override.py", base_hash, verifier.sha256(override)]],
        },
        "b12x": {
            "revision": "b58",
            "tree": "tree",
            "package_files_sha256": verifier.file_map_sha256({
                "b12x/__init__.py": verifier.sha256(b12x_python),
                "b12x/README.md": verifier.sha256(b12x_notice),
            }),
        },
        "cuda": {"version": "13.3", "components": components},
        "environment": {
            "VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH": "1",
            "VLLM_B12X_DENSE_ACTIVATION_MODE": "auto",
            "VLLM_MTP_NVFP4_LM_HEAD": "1",
            "VLLM_LM_HEAD_A16": "1",
            "VLLM_MXFP8_LM_HEAD": "0",
        },
    }
    lock_path = compute / "source-lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    lock_hash = verifier.sha256(lock_path)
    (receipts / "vllm-source-manifest.json").write_text(
        json.dumps({
            "commit": "e02",
            "files": {
                "vllm/unchanged.py": verifier.sha256(unchanged),
                "vllm/override.py": base_hash,
            },
        }),
        encoding="utf-8",
    )
    b12x_files = {
        "b12x/__init__.py": verifier.sha256(b12x_python),
        "b12x/README.md": verifier.sha256(b12x_notice),
    }
    installed_path = tmp_path / "installed.json"
    installed_path.write_text(
        json.dumps({
            "schema": "sparkring-glm53-compute-installed/v1",
            "source_lock_sha256": lock_hash,
            "vllm_revision": "e02",
            "vllm_overrides": {"vllm/override.py": verifier.sha256(override)},
            "b12x_revision": "b58",
            "b12x_tree": "tree",
            "b12x_files": b12x_files,
            "cuda_components": components,
            "environment": lock["environment"],
            "target_head_quantization": False,
        }),
        encoding="utf-8",
    )
    profile = {"compute": {
        "source_lock": "compute/source-lock.json",
        "source_lock_sha256": lock_hash,
        "vllm_base_revision": "e02",
        "b12x_revision": "b58",
        "b12x_tree": "tree",
        "cuda_version": "13.3",
    }}
    source = {"files": {
        "compute/source-lock.json": lock_hash,
        "compute/b12x-source/b12x/__init__.py": verifier.sha256(b12x_python),
        "compute/b12x-source/b12x/README.md": verifier.sha256(b12x_notice),
        "compute/b12x-source/pyproject.toml": "a" * 64,
        "compute/b12x-source/tests/test_example.py": "b" * 64,
    }}
    environment = {
        **lock["environment"],
        "CUDA_HOME": "/opt/cuda-13.3",
        "TRITON_PTXAS_PATH": "/opt/cuda-13.3/bin/ptxas",
    }
    result = verifier.verify_compute(
        profile,
        {"vllm": {"commit": "e02"}},
        source,
        environment,
        compute_root=compute,
        receipt_path=installed_path,
        site=site,
        base_receipts=receipts,
        cuda_root_override=cuda,
        ptxas_runner=lambda *args, **kwargs: SimpleNamespace(
            stdout="ptxas release 13.3", stderr=""
        ),
    )
    assert result["vllm_parent_files"] == 2
    assert result["vllm_overrides"] == 1
    assert result["b12x_files"] == 2
    assert result["proposal_head_nvfp4"] is True
    assert result["target_head_quantization"] is False

    changed_source = json.loads(json.dumps(source))
    changed_source["files"]["compute/b12x-source/b12x/README.md"] = "f" * 64
    with pytest.raises(ValueError, match="source-receipt-bound"):
        verifier.verify_compute(
            profile,
            {"vllm": {"commit": "e02"}},
            changed_source,
            environment,
            compute_root=compute,
            receipt_path=installed_path,
            site=site,
            base_receipts=receipts,
            cuda_root_override=cuda,
            ptxas_runner=lambda *args, **kwargs: SimpleNamespace(
                stdout="ptxas release 13.3", stderr=""
            ),
        )

    unchanged.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content pin"):
        verifier.verify_compute(
            profile,
            {"vllm": {"commit": "e02"}},
            source,
            environment,
            compute_root=compute,
            receipt_path=installed_path,
            site=site,
            base_receipts=receipts,
            cuda_root_override=cuda,
            ptxas_runner=lambda *args, **kwargs: SimpleNamespace(
                stdout="ptxas release 13.3", stderr=""
            ),
        )


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


def test_profile_pins_exact_compute_source_and_quantization_environment():
    pins = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
    compute = pins["compute"]
    lock_path = HERE / compute["source_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert verifier.sha256(lock_path) == compute["source_lock_sha256"]
    assert lock["schema"] == "sparkring-glm53-compute-source/v1"
    assert lock["vllm"]["base_revision"] == compute["vllm_base_revision"]
    assert lock["b12x"]["revision"] == compute["b12x_revision"]
    assert lock["b12x"]["tree"] == compute["b12x_tree"]
    assert lock["cuda"]["version"] == compute["cuda_version"] == "13.3"
    assert lock["environment"] == {
        "VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH": "1",
        "VLLM_B12X_DENSE_ACTIVATION_MODE": "auto",
        "VLLM_MTP_NVFP4_LM_HEAD": "1",
        "VLLM_LM_HEAD_A16": "1",
        "VLLM_MXFP8_LM_HEAD": "0",
    }
    assert len(lock["vllm"]["files"]) == 24


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
