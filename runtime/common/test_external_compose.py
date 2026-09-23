"""Render opt-in external profiles with isolated publication and source fixtures."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys

import pytest
import yaml

from runtime.common import compose, external_candidate as external, native_candidate, profiles, qwen_flash_next as qwen
from runtime.common.test_external_candidate import RELEASE, fixture as external_fixture, opt_in_profile


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


@pytest.fixture
def registered_pair_and_ring(tmp_path, monkeypatch):
    original_root = compose.ROOT
    sources = set(compose.source_inventory("qwen38-flash-next-tp2-sparkcache"))
    sources.update(compose.source_inventory("qwen38-flash-next-qad-tp4-sparkcache"))
    sources.add("runtime/common/external_candidate.py")
    repository = tmp_path / "repository"
    for name in sources:
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / name, path)
    publication, _, _, _ = external_fixture()
    publication["profiles"] = {}
    catalog, sites, configs = [], {}, {}
    for nodes, old_id in ((2, "qwen38-flash-next-tp2"), (4, "qwen38-flash-next-qad-tp4")):
        identity = f"qwen38-flash-next-qad-tp{nodes}-eugr"
        profile = opt_in_profile(nodes, True)
        profile["status"] = "research-only"
        profile["qualification"] = {"scope": "Synthetic publication fixture only."}
        profile["environment"].update({
            "VLLM_QWEN3_8_FLASH_NEXT_HC_TP": "1", "VLLM_QWEN3_8_HC_PREFILL_MODE": "off",
            "NCCL_IB_PRESERVE_PCI_DOMAIN": "1", "NCCL_IB_ROUTE_DIAGNOSTICS": "1",
        })
        name = f"profiles/{identity}/config.json"
        write_json(repository / name, profile)
        publication["profiles"][name] = hashlib.sha256((repository / name).read_bytes()).hexdigest()
        metadata = qwen.read(original_root / f"profiles/{old_id}-sparkcache/profile.json")
        metadata.update({
            "id": identity, "status": "research-only", "recommendation": "alternative",
            "configuration": {"format": "serving-profile", "path": name},
            "release": f"runtime/releases/{RELEASE}/release.json",
            "guide": f"profiles/{identity}/README.md",
        })
        metadata["launcher"]["fixed_args"] = ["--profile", name]
        write_json(repository / f"profiles/{identity}/profile.json", metadata)
        (repository / metadata["guide"]).write_text("# Fixture profile\n")
        catalog.append({"id": identity, "path": f"profiles/{identity}/profile.json"})
        sites[identity] = compose.read_site(original_root / f"profiles/{old_id}/compose/site.example.yaml")
        configs[identity] = profile
    publication_path = f"runtime/releases/{RELEASE}/publication.json"
    write_json(repository / publication_path, publication)
    release = {
        "schema": "sparkring-release-selection/v1", "id": RELEASE,
        "selection": "published-immutable-reference", "image": publication["image_reference"],
        "inputs": [{"path": publication_path,
                    "sha256": hashlib.sha256((repository / publication_path).read_bytes()).hexdigest()}],
    }
    write_json(repository / f"runtime/releases/{RELEASE}/release.json", release)
    write_json(repository / "profiles/catalog.json", {"schema": "sparkring-catalog/v1", "profiles": catalog})
    load = profiles.load
    monkeypatch.setattr(profiles, "load", lambda identity: load(identity, root=repository))
    for module in (compose, external, native_candidate, qwen):
        monkeypatch.setattr(module, "ROOT", repository)
    return publication, sites, configs, repository


@pytest.mark.parametrize("identity", compose.EXTERNAL_PROFILES)
def test_external_compose_render_retains_every_rank_profile_and_envelope(registered_pair_and_ring, identity):
    publication, sites, configs, _ = registered_pair_and_ring
    specs, image = compose.specifications(identity, sites[identity])
    profile = configs[identity]
    assert len(specs) == qwen.node_count(profile)
    assert image == publication["image_reference"]
    for rank, spec in enumerate(specs):
        service = yaml.safe_load(compose.compose_text(spec, image))["services"]["model"]
        assert service["entrypoint"] == ["python3"]
        assert service["command"][0] == external.ENTRYPOINT
        assert service["cap_add"] == ["IPC_LOCK"]
        assert service["security_opt"] == ["seccomp=unconfined"]
        assert service["working_dir"] == "/"
        assert service["image"] == image
        assert ("--headless" in spec.command) == bool(rank)
        assert spec.environment["SPARKRING_TRANSPORT_MANIFEST_SHA256"] == publication["transport"]["manifest_sha256"]
        assert spec.environment["NCCL_IB_PRESERVE_PCI_DOMAIN"] == "1"
        assert spec.environment["NCCL_IB_ROUTE_DIAGNOSTICS"] == "1"
        assert spec.environment["VLLM_QWEN3_8_HC_PREFILL_MODE"] == "off"
        assert spec.environment["VLLM_QWEN3_8_FLASH_NEXT_HC_TP"] == "1"
        assert spec.environment["SPARKRING_FEATURES"] == profile["environment"]["SPARKRING_FEATURES"]
        assert spec.environment["B12X_ROCE_HCA"] == ",".join(sites[identity]["ranks"][rank]["hcas"])
        for flag in ("--max-model-len", "--max-num-seqs", "--max-num-batched-tokens",
                     "--kv-cache-memory-bytes", "--speculative-config", "--compilation-config"):
            assert spec.command[spec.command.index(flag) + 1] == profile["vllm_args"][profile["vllm_args"].index(flag) + 1]
        extra = json.loads(spec.command[spec.command.index("--kv-transfer-config") + 1])["kv_connector_extra_config"]
        assert extra["spark_cache_async_page_capture_vllm_root"] == external.SITE
        assert extra["spark_cache_async_page_capture_lease_contract"] == publication["sparkcache_contract"]["path"]


@pytest.mark.parametrize("identity", compose.EXTERNAL_PROFILES)
def test_staged_external_inventory_contains_the_full_admission_dependencies(registered_pair_and_ring, tmp_path, identity):
    publication, _, _, repository = registered_pair_and_ring
    inventory = compose.source_inventory(identity)
    assert {"runtime/common/external_candidate.py", "runtime/common/native_candidate.py",
            f"runtime/releases/{RELEASE}/publication.json", *publication["profiles"]} <= inventory.keys()
    if "tp4" in identity:
        from runtime.common import qwen_mesh
        assert set(qwen_mesh.SOURCE_FILES) <= inventory.keys()
    snapshot = tmp_path / "isolated-source"
    for name in inventory:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, path)
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from runtime.common import external_candidate, qwen_flash_next
assert Path(external_candidate.__file__).is_relative_to(Path(sys.argv[1]))
profile = qwen_flash_next.read(Path(sys.argv[1]) / 'profiles' / sys.argv[2] / 'config.json')
record = external_candidate.publication(profile['image_release'])
assert external_candidate.canonical_profile(profile) == profile
spec = qwen_flash_next.container_spec(profile, rank=0, master='192.0.2.1',
    host_ip='192.0.2.1', interface='test0', image=record['image_id'],
    model='/models/qwen', cache='/cache/qwen', remote=True)
assert spec.entrypoint == ('python3',)
assert spec.security_opt == ('seccomp=unconfined',)
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(snapshot), identity],
                            capture_output=True, text=True, timeout=30, cwd=tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("identity", compose.EXTERNAL_PROFILES)
def test_external_export_detects_controller_source_drift(registered_pair_and_ring, tmp_path, identity):
    _, sites, _, repository = registered_pair_and_ring
    output = tmp_path / "deployment"
    before = compose.render(identity, sites[identity], output)
    after, _ = compose.load_deployment(output)
    assert after == before
    path = repository / "runtime/common/external_candidate.py"
    path.write_bytes(path.read_bytes() + b"\n# staged controller change\n")
    with pytest.raises(ValueError, match="Deployment inputs changed"):
        compose.load_deployment(output)


@pytest.mark.parametrize("mutation", ["image", "first_input", "hash", "identity"])
def test_external_release_selection_cannot_alias_another_publication(registered_pair_and_ring, mutation):
    _, _, _, repository = registered_pair_and_ring
    release = qwen.read(repository / f"runtime/releases/{RELEASE}/release.json")
    changed = copy.deepcopy(release)
    if mutation == "image":
        changed["image"] = "example/other@sha256:" + "0" * 64
    elif mutation == "first_input":
        changed["inputs"][0]["path"] = "runtime/releases/another/publication.json"
    elif mutation == "hash":
        changed["inputs"][0]["sha256"] = "0" * 64
    else:
        changed["id"] = "shared-another"
    with pytest.raises(ValueError, match="must pin its publication"):
        external.release_publication(changed, RELEASE)


@pytest.mark.parametrize("nodes", [2, 4])
def test_profile_contract_validation_does_not_require_a_publication(monkeypatch, nodes):
    profile = opt_in_profile(nodes, True)

    def unavailable(*args, **kwargs):
        raise AssertionError("Publication is intentionally unavailable for a config draft")

    monkeypatch.setattr(native_candidate, "publication", unavailable)
    assert external.validate_profile_contract(profile) == nodes
