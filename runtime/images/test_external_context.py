"""Exercise source ownership and pinned context inputs without Docker or a GPU."""

import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    "external_context", Path(__file__).with_name("external_context.py")
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def archive(path, files):
    with tarfile.open(path, "w") as target:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644
            target.addfile(member, io.BytesIO(data))


@pytest.fixture
def inputs(tmp_path):
    def pin(path):
        return {"path": path.relative_to(tmp_path).as_posix(), "sha256": builder.file_sha(path)}

    baseline = {
        "vllm": {
            "models/qwen4_exp/nvidia/hyperconnection.py": b"# parent HC\n",
            "deleted.py": b"# parent source\n",
            "retained.py": b"# unchanged\n",
        },
        "b12x": {
            "sequence/mtp_feedback/_kernels.py": b"# parent MTP\n",
            "adapter.py": b"# policy target\n",
        },
    }
    candidate = {
        "vllm": {
            "models/qwen4_exp/nvidia/hyperconnection.py": b"# projection TP HC\n",
            "retained.py": b"# unchanged\n",
            "added.py": b"# candidate source\n",
        },
        "b12x": {
            "sequence/mtp_feedback/_kernels.py": b"# candidate MTP\n",
            "adapter.py": b"# policy target\r\n",
        },
    }
    source_pins = {}
    inventory = {"architecture": "aarch64", "versions": {}, "packages": {}}
    for name in baseline:
        for kind, rows in (("baseline", baseline[name]), ("candidate", candidate[name])):
            archive(tmp_path / f"{name}-{kind}.tar", {name + "/" + key: raw for key, raw in rows.items()})
        source_pins[name] = {
            "archive": pin(tmp_path / f"{name}-candidate.tar"),
            "baseline_archive": pin(tmp_path / f"{name}-baseline.tar"),
            "commit": "a" * 40, "upstream": "b" * 40, "baseline_commit": "c" * 40,
        }
        base_files = {**baseline[name], "native.so": b"compiled", "vendor/generated.py": b"# generated\n"}
        inventory["packages"][name] = {
            "root": builder.SITE + name,
            "files": {key: {"sha256": digest(raw)} for key, raw in base_files.items()},
        }
    dump(tmp_path / "inventory.json", inventory)
    assets = tmp_path / "assets"
    asset_files = {
        "sparkcache/__init__.py": b"# inherited cache\n",
        "sparkcache-overrides/__init__.py": b"# integrated cache\n",
        "sparkcache-native/lib/cache.so": b"cache binary",
        "licenses/components.md": b"# Native helper licenses\n",
        "transports/sparkring_transport_selector.py": b"HOST_SOURCE_PREFIX = '/old/b12x/'\n",
        "transports/" + builder.TRANSPORT + "/manifest.json": json.dumps({
            "image_source_preimages": {"/opt/venv/lib/python3.12/site-packages/b12x/adapter.py": "0" * 64},
        }).encode(),
        "features/qwen-collectives/qwen38_collective_policy.py": (
            "SOURCE_HASHES = " + repr({"b12x.adapter": digest(b"# policy target\n")}) + "\n"
        ).encode(),
        "features/sparkring_features.py": b"# feature dispatcher\n",
        "features/capabilities.json": json.dumps({
            "features": {
                "qwen-collectives": {"files": {"qwen-collectives/qwen38_collective_policy.py": "0" * 64}},
                "qwen4-prefill": {},
            },
            "transport_profiles": {builder.TRANSPORT: {}},
        }).encode(),
        "nccl-lib/libnccl.so.2.31.2": b"patched NCCL",
        "nccl-lib/libnccl.so.2": b"patched NCCL",
        "nccl-lib/libnccl.so": b"patched NCCL",
    }
    parent = {"files": {}}
    export_files = {}
    for name, raw in asset_files.items():
        path = assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        origin = "/parent/" + name
        if not name.startswith("nccl-lib/"):
            parent["files"][origin] = digest(raw)
        export_files[name] = {"origin": origin, "sha256": digest(raw)}
    dump(assets / "parent-receipt.json", parent)
    dump(assets / "export.json", {
        "parent_image": "sha256:" + "d" * 64,
        "parent_receipt_sha256": builder.file_sha(assets / "parent-receipt.json"),
        "files": export_files,
        "scope": "Integration assets only",
    })
    contract = {"base_commit": "c" * 40, "files": [
        {"path": "vllm/models/qwen4_exp/nvidia/hyperconnection.py", "sha256": "0" * 64},
        {"path": "sparkcache/__init__.py", "sha256": "0" * 64},
    ]}
    dump(tmp_path / "cache-contract.json", contract)
    controller = tmp_path / "prefill"
    controller.mkdir()
    maintained = Path(__file__).resolve().parents[2] / "integrations/vllm/qwen4_prefill"
    for name in builder.PREFILL_FILES:
        shutil.copyfile(maintained / name, controller / name)
    shutil.copyfile(Path(__file__).with_name("external_base.py"), tmp_path / "installer.py")
    manifest = {
        "schema": "sparkring-external-inputs/v1", "platform": "linux/arm64",
        "base": {"reference": "example/runtime@sha256:" + "1" * 64, "config_id": "sha256:" + "2" * 64},
        "sources": source_pins,
        "base_inventory": pin(tmp_path / "inventory.json"),
        "installer": pin(tmp_path / "installer.py"),
        "parent_cache_contract": pin(tmp_path / "cache-contract.json"),
        "assets": {
            "root": "assets", "export_manifest": pin(assets / "export.json"),
            "parent_receipt": pin(assets / "parent-receipt.json"),
        },
        "prefill_controller": {
            "root": "prefill", "files": {name: builder.file_sha(controller / name) for name in builder.PREFILL_FILES},
        },
    }
    path = tmp_path / "inputs.json"
    dump(path, manifest)
    return path, manifest, pin


def test_context_retains_framework_ownership_and_binds_composed_sources(inputs, tmp_path):
    path, manifest, _ = inputs
    result = builder.prepare(path, tmp_path / "context")
    context = tmp_path / "context"
    descriptor = json.loads((context / "composition.json").read_bytes())
    files = descriptor["files"]

    assert files[builder.SITE + "vllm/deleted.py"] == {
        "before": digest(b"# parent source\n"), "after": None,
    }
    assert files[builder.SITE + "vllm/added.py"]["before"] is None
    for package in ("vllm", "b12x"):
        assert builder.SITE + package + "/native.so" not in files
        assert builder.SITE + package + "/vendor/generated.py" not in files
    assert builder.SITE + "vllm/retained.py" not in files
    assert (context / "payload" / (builder.SITE + "sparkcache/__init__.py").lstrip("/")).read_bytes() == b"# integrated cache\n"

    prefill = json.loads((context / "prefill-bundle/manifest.json").read_bytes())
    assert prefill["image_source_preimages"] == {
        builder.SITE + "vllm/models/qwen4_exp/nvidia/hyperconnection.py": digest(b"# projection TP HC\n"),
        builder.SITE + "b12x/sequence/mtp_feedback/_kernels.py": digest(b"# candidate MTP\n"),
    }
    transport_path = "/opt/sparkring/transports/" + builder.TRANSPORT + "/manifest.json"
    transport = json.loads((context / "payload" / transport_path.lstrip("/")).read_bytes())
    assert transport["image_source_preimages"] == {builder.SITE + "b12x/adapter.py": digest(b"# policy target\r\n")}
    policy = (context / "payload/opt/sparkring/features/qwen-collectives/qwen38_collective_policy.py").read_bytes()
    assert digest(b"# policy target\r\n").encode() in policy
    assert descriptor["sources"]["vllm"]["baseline_archive_sha256"] == manifest["sources"]["vllm"]["baseline_archive"]["sha256"]
    assert descriptor["provenance"]["asset_export"]["parent_image"] == "sha256:" + "d" * 64
    assert descriptor["provenance"]["prefill_controller_files"] == manifest["prefill_controller"]["files"]
    assert descriptor["capabilities"]["hc_supported_modes"] == builder.HC_SUPPORTED_MODES
    assert "hc_projection_tp" not in descriptor["capabilities"]
    assert descriptor["capabilities"]["serving_qualified"] is False
    assert result["composition_sha256"] == builder.file_sha(context / "composition.json")
    assert (context / "Dockerfile").read_text().startswith("FROM " + manifest["base"]["reference"] + "\n")
    assert str(tmp_path) not in (context / "composition.json").read_text()


def test_context_is_byte_reproducible_and_does_not_replace_an_output(inputs, tmp_path):
    path, _, _ = inputs
    first, second = tmp_path / "first", tmp_path / "second"
    builder.prepare(path, first)
    builder.prepare(path, second)
    snapshots = [
        {file.relative_to(root).as_posix(): file.read_bytes() for file in root.rglob("*") if file.is_file()}
        for root in (first, second)
    ]
    assert snapshots[0] == snapshots[1]
    with pytest.raises(ValueError, match="Output already exists"):
        builder.prepare(path, first)


def test_candidate_can_take_source_ownership_of_an_inherited_vendor_file(inputs, tmp_path):
    path, manifest, pin = inputs
    target = tmp_path / "vllm-candidate.tar"
    source = builder.source_entries(target, "vllm")
    contents = {name: raw for name, (raw, _) in source.items()}
    contents["vllm/vendor/generated.py"] = b"# reviewed vendor kernel fix\n"
    archive(target, contents)
    manifest["sources"]["vllm"]["archive"] = pin(target)
    dump(path, manifest)

    builder.prepare(path, tmp_path / "context")
    descriptor = json.loads((tmp_path / "context/composition.json").read_bytes())
    row = descriptor["files"][builder.SITE + "vllm/vendor/generated.py"]
    assert row["before"] == digest(b"# generated\n")
    assert row["after"] == digest(b"# reviewed vendor kernel fix\n")
    assert builder.SITE + "vllm/native.so" not in descriptor["files"]


@pytest.mark.parametrize("artifact", [
    "vllm-candidate.tar", "vllm-baseline.tar", "inventory.json", "installer.py",
    "cache-contract.json", "assets/export.json", "assets/parent-receipt.json",
    "prefill/qwen4_hc_fusion.py",
])
def test_pinned_input_mutations_are_rejected_before_output(inputs, tmp_path, artifact):
    path, _, _ = inputs
    target = tmp_path / artifact
    target.write_bytes(target.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError, match="Artifact digest differs"):
        builder.prepare(path, tmp_path / "context")
    assert not (tmp_path / "context").exists()


@pytest.mark.parametrize("mutation", ["change", "add", "receipt"])
def test_exported_assets_remain_bound_to_the_parent(inputs, tmp_path, mutation):
    path, manifest, pin = inputs
    if mutation == "change":
        (tmp_path / "assets/sparkcache/__init__.py").write_bytes(b"modified")
    elif mutation == "add":
        (tmp_path / "assets/sparkcache/unrecorded.py").write_bytes(b"unrecorded")
    else:
        receipt = tmp_path / "assets/parent-receipt.json"
        dump(receipt, {"files": {}})
        exported_path = tmp_path / "assets/export.json"
        exported = json.loads(exported_path.read_bytes())
        exported["parent_receipt_sha256"] = builder.file_sha(receipt)
        dump(exported_path, exported)
        manifest["assets"]["parent_receipt"] = pin(receipt)
        manifest["assets"]["export_manifest"] = pin(exported_path)
        dump(path, manifest)
    with pytest.raises(ValueError, match="Exported asset|Asset does not match parent"):
        builder.prepare(path, tmp_path / "context")


@pytest.mark.parametrize("name", ["vllm/extra.so", "vllm/../outside.py", "vllm/cache.pyc"])
def test_source_archive_rejects_non_source_payload(inputs, tmp_path, name):
    path, manifest, pin = inputs
    target = tmp_path / "vllm-candidate.tar"
    archive(target, {name: b"payload"})
    manifest["sources"]["vllm"]["archive"] = pin(target)
    dump(path, manifest)
    with pytest.raises(ValueError, match="framework binary|Invalid artifact path|bytecode"):
        builder.prepare(path, tmp_path / "context")


def test_collective_rebinding_requires_same_semantics(inputs, tmp_path):
    path, manifest, pin = inputs
    target = tmp_path / "b12x-candidate.tar"
    archive(target, {
        "b12x/adapter.py": b"# semantically different target\n",
        "b12x/sequence/mtp_feedback/_kernels.py": b"# candidate MTP\n",
    })
    manifest["sources"]["b12x"]["archive"] = pin(target)
    dump(path, manifest)
    with pytest.raises(ValueError, match="Collective adapter source identity changed"):
        builder.prepare(path, tmp_path / "context")
    assert not (tmp_path / "context").exists()
    assert not list(tmp_path.glob(".external-context-*"))


def status_artifacts(inputs, tmp_path, *, wheel_mutation=None, record_mutation=None):
    manifest_path, manifest, pin = inputs
    project = b'''[project]
name = "sparkring-runtime-status"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["fastapi>=0.115"]
[project.entry-points."vllm.endpoint_plugins"]
sparkring_status = "sparkring_runtime_status.plugin:StatusPlugin"
[project.entry-points."vllm.general_plugins"]
sparkring_status = "sparkring_runtime_status.plugin:register_worker_method"
'''
    source = {
        "pyproject.toml": project,
        "sparkring_runtime_status/__init__.py": b"# status package\n",
        "sparkring_runtime_status/plugin.py": b"class StatusPlugin: pass\ndef register_worker_method(): pass\n",
    }
    source_path = tmp_path / "runtime-status-source.tar.gz"
    archive(source_path, {"runtime_status/" + name: raw for name, raw in source.items()})
    distribution = "sparkring_runtime_status-0.1.0.dist-info"
    files = {name: raw for name, raw in source.items() if name.startswith("sparkring_runtime_status/")}
    files.update({
        distribution + "/METADATA": b"Metadata-Version: 2.4\nName: sparkring-runtime-status\nVersion: 0.1.0\nRequires-Python: >=3.10\nRequires-Dist: fastapi>=0.115\n\n",
        distribution + "/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        distribution + "/entry_points.txt": b"[vllm.endpoint_plugins]\nsparkring_status = sparkring_runtime_status.plugin:StatusPlugin\n[vllm.general_plugins]\nsparkring_status = sparkring_runtime_status.plugin:register_worker_method\n",
    })
    if wheel_mutation:
        wheel_mutation(files, distribution)
    rows = [[name, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode(), str(len(raw))]
            for name, raw in files.items()]
    rows.append([distribution + "/RECORD", "", ""])
    if record_mutation:
        record_mutation(rows)
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n").writerows(rows)
    files[distribution + "/RECORD"] = stream.getvalue().encode()
    wheel = tmp_path / "sparkring_runtime_status-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as package:
        for name, raw in files.items():
            package.writestr(name, raw)
    inventory = json.loads((tmp_path / "inventory.json").read_bytes())
    inventory["versions"]["fastapi"] = "0.115.0"
    dump(tmp_path / "inventory.json", inventory)
    manifest["base_inventory"] = pin(tmp_path / "inventory.json")
    manifest["runtime_status"] = {"wheel": pin(wheel), "source_archive": pin(source_path)}
    dump(manifest_path, manifest)
    return files


def test_status_wheel_is_source_bound_and_installed_under_its_own_python_root(inputs, tmp_path):
    files = status_artifacts(inputs, tmp_path)
    path, manifest, _ = inputs
    builder.prepare(path, tmp_path / "context")
    descriptor = json.loads((tmp_path / "context/composition.json").read_bytes())
    status = descriptor["capabilities"]["runtime_status"]
    assert status["entry_points"] == builder.STATUS_ENTRY_POINTS
    assert status["wheel_sha256"] == manifest["runtime_status"]["wheel"]["sha256"]
    assert status["source_archive_sha256"] == manifest["runtime_status"]["source_archive"]["sha256"]
    assert descriptor["python_roots"] == {builder.PYTHON_ROOT: {name: digest(raw) for name, raw in files.items()}}
    hook = tmp_path / "context/payload" / builder.SITE.lstrip("/") / "sparkring_runtime_status.pth"
    assert hook.read_bytes() == b"/opt/sparkring/python\n"
    assert not any(name.startswith(builder.SITE + "sparkring_runtime_status/") for name in descriptor["files"])
    assert descriptor["provenance"]["runtime_status"]["source_files"]["pyproject.toml"]


@pytest.mark.parametrize("mutation,message", [
    (lambda files, dist: files.__setitem__("sparkring_runtime_status/plugin.py", b"# substituted\n"), "differs from its pinned source"),
    (lambda files, dist: files.__setitem__("vllm/__init__.py", b"# framework override\n"), "outside its owners"),
    (lambda files, dist: files.__setitem__("sparkring_runtime_status/extra.so", b"native"), "not pure Python"),
    (lambda files, dist: files.__setitem__("sparkring_runtime_status/extra.pth", b"import unbound"), "not pure Python"),
    (lambda files, dist: files.__setitem__(dist + "/WHEEL", b"Root-Is-Purelib: true\nTag: cp312-cp312-linux_aarch64\n"), "platform-specific"),
    (lambda files, dist: files.__setitem__(dist + "/entry_points.txt", b"[vllm.general_plugins]\nsparkring_status = elsewhere:register\n"), "official API and worker"),
])
def test_status_artifact_rejects_foreign_code_native_payload_and_wrong_abi(inputs, tmp_path, mutation, message):
    status_artifacts(inputs, tmp_path, wheel_mutation=mutation)
    with pytest.raises(ValueError, match=message):
        builder.prepare(inputs[0], tmp_path / "context")
    assert not (tmp_path / "context").exists()


@pytest.mark.parametrize("mutation,message", [
    (lambda rows: rows.pop(0), "RECORD inventory"),
    (lambda rows: rows[0].__setitem__(1, "sha256=wrong"), "RECORD digest"),
    (lambda rows: rows.append(rows[0]), "duplicate entries"),
])
def test_status_wheel_record_must_cover_exact_payload(inputs, tmp_path, mutation, message):
    status_artifacts(inputs, tmp_path, record_mutation=mutation)
    with pytest.raises(ValueError, match=message):
        builder.prepare(inputs[0], tmp_path / "context")


@pytest.mark.parametrize("versions", [
    {"fastapi": "0.100.0"}, {"fastapi": "0.115.0", "sparkring_runtime_status": "0.1.0"},
])
def test_status_base_dependencies_and_distribution_ownership_are_explicit(inputs, tmp_path, versions):
    status_artifacts(inputs, tmp_path)
    path, manifest, pin = inputs
    inventory = json.loads((tmp_path / "inventory.json").read_bytes())
    inventory["versions"] = versions
    dump(tmp_path / "inventory.json", inventory)
    manifest["base_inventory"] = pin(tmp_path / "inventory.json")
    dump(path, manifest)
    with pytest.raises(ValueError, match="FastAPI dependency|already owns"):
        builder.prepare(path, tmp_path / "context")
