"""Local source-image trials preserve TP eligibility and bind every exported rank."""

import copy
import json
import shutil
import subprocess
import sys

import pytest

from runtime.common import compose, qwen_flash_next as adapter, source_candidate as source
from runtime.common.container_spec import docker_create
from runtime.common.test_compose import compose_cli as compose_cli
from scripts import sparkring_compose as coordinator

PROFILE = "qwen38-flash-next-qad-tp4"
IMAGE = "sha256:" + "a" * 64
PAIR = "qwen38-flash-next-tp2"


@pytest.fixture
def site():
    return compose.read_site(adapter.ROOT / f"profiles/{PROFILE}/compose/site.example.yaml")


@pytest.fixture
def pair():
    return compose.read_site(adapter.ROOT / f"profiles/{PAIR}/compose/site.example.yaml")


def options(**extra):
    return {"local_source_extension": source.IDENTITY, "local_image_id": IMAGE, **extra}


@pytest.mark.parametrize("profile_id", [PAIR, PAIR + "-sparkcache", PROFILE, PROFILE + "-sparkcache"])
def test_source_bootstrap_port_cannot_collide_with_api(profile_id):
    site = compose.read_site(adapter.ROOT / f"profiles/{profile_id.removesuffix('-sparkcache')}/compose/site.example.yaml")
    ordinary, _ = compose.specifications(profile_id, site)
    args = ordinary[0].command
    api_port = int(args[args.index("--port") + 1])
    with pytest.raises(ValueError, match="must differ from the inference API"):
        compose.specifications(profile_id, site, **options(local_master_port=api_port))


@pytest.mark.parametrize("profile_id", [PROFILE, PROFILE + "-sparkcache"])
def test_candidate_preserves_model_and_transport_while_enabling_reviewed_tp4_sources(site, profile_id):
    before, public_image = compose.specifications(profile_id, site)
    selected, local_image = compose.specifications(profile_id, site, **options())
    assert "@sha256:" in public_image and local_image.startswith("sparkring-local:")
    for original, spec in zip(before, selected):
        assert spec.image_id == IMAGE
        assert spec.command[:2] == (source.ENTRYPOINT, "serve")
        assert spec.environment["VLLM_QWEN3_8_HC_PREFILL_MODE"] == "shard"
        assert spec.environment["VLLM_QWEN3_8_PREFILL_COALESCE"] == "1"
        for name in ("SPARKRING_FEATURES", "QWEN_DISPATCH_AR_BYTES", "NCCL_IB_PRESERVE_PCI_DOMAIN",
                     "B12X_ROCE_PEER_HCA_MAP", "SPARKCACHE_ENABLED", "SIRCL_ENABLED"):
            assert spec.environment[name] == original.environment[name]
        for flag in ("--max-model-len", "--max-num-seqs", "--max-num-batched-tokens",
                     "--kv-cache-memory-bytes", "--tensor-parallel-size", "--decode-context-parallel-size",
                     "--master-port", "--speculative-config", "--block-size"):
            assert spec.command[spec.command.index(flag) + 1] == original.command[original.command.index(flag) + 1]
        assert spec.mounts == original.mounts
        argv = docker_create(spec)
        assert argv[argv.index(IMAGE) + 1:] == list(spec.command)
        if "--kv-transfer-config" in spec.command:
            extra = json.loads(spec.command[spec.command.index("--kv-transfer-config") + 1])["kv_connector_extra_config"]
            assert extra["spark_cache_async_page_capture_lease_contract"] == source.LEASE_CONTRACT
            assert extra["spark_cache_root"].endswith(source.IDENTITY)


def test_local_memory_and_port_options_are_bound_to_every_rank_and_deployment(site, tmp_path):
    ordinary, _ = compose.build(PROFILE + "-sparkcache", site, **options())
    settings = options(local_kv_cache_gib=40, local_master_port=29779)
    selected, files = compose.build(PROFILE + "-sparkcache", site, **settings)
    assert ordinary["id"] != selected["id"]
    assert compose.selection_options(selected) == settings
    assert "runtime/common/source_candidate.py" in selected["inputs"]
    assert f"runtime/images/compositions/{source.IDENTITY}/descriptor.json" in selected["inputs"]
    for rank in range(4):
        spec = json.loads(files[f"rank{rank}/container.json"])
        args = spec["command"]
        assert args[args.index("--kv-cache-memory-bytes") + 1] == str(40 * 1024 ** 3)
        assert args[args.index("--master-port") + 1] == "29779"
    target = tmp_path / "candidate"
    compose.render(PROFILE + "-sparkcache", site, target, **settings)
    manifest, _ = compose.load_deployment(target)
    assert manifest == selected
    edited = copy.deepcopy(selected)
    edited["local_kv_cache_gib"] = None
    (target / "deployment.json").write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError, match="inputs changed"):
        compose.load_deployment(target)


@pytest.mark.parametrize("mutation", [
    {"local_source_extension": "unregistered"}, {"local_image_id": None},
    {"local_image_id": "sparkring:mutable"}, {"local_kv_cache_gib": 41},
    {"local_kv_cache_gib": True}, {"local_master_port": 0}, {"local_master_port": 65536},
])
def test_candidate_inputs_fail_closed(site, mutation):
    with pytest.raises(ValueError):
        compose.specifications(PROFILE, site, **options(**mutation))


@pytest.mark.parametrize("profile_id", [PAIR, PAIR + "-sparkcache"])
def test_local_tp2_retains_pair_settings_without_tp4_compute_activation(pair, profile_id):
    before, public_image = compose.specifications(profile_id, pair)
    selected, local_image = compose.specifications(profile_id, pair, **options())
    assert len(selected) == 2 and "@sha256:" in public_image
    assert local_image == source.image_reference(source.IDENTITY, IMAGE)
    for original, spec in zip(before, selected):
        assert spec.image_id == IMAGE and spec.command[0] == source.ENTRYPOINT
        assert spec.environment["VLLM_QWEN3_8_HC_PREFILL_MODE"] == "off"
        assert spec.environment["VLLM_QWEN3_8_PREFILL_COALESCE"] == "1"
        assert spec.environment["SPARKRING_FEATURES"] == original.environment.get("SPARKRING_FEATURES", "")
        for name in ("B12X_ROCE_PEER_HCA_MAP", "B12X_ROCE_HCA", "NCCL_IB_HCA", "B12X_ROCE_PAIR_PATHS",
                     "SPARKRING_TRANSPORT_PROFILE", "SPARKCACHE_ENABLED", "SIRCL_ENABLED"):
            assert spec.environment[name] == original.environment[name]
        for flag in ("--max-model-len", "--max-num-seqs", "--max-num-batched-tokens",
                     "--kv-cache-memory-bytes", "--tensor-parallel-size", "--nnodes",
                     "--decode-context-parallel-size", "--master-port", "--speculative-config", "--block-size"):
            assert spec.command[spec.command.index(flag) + 1] == original.command[original.command.index(flag) + 1]
        assert spec.mounts == original.mounts
        if "--kv-transfer-config" in spec.command:
            extra = json.loads(spec.command[spec.command.index("--kv-transfer-config") + 1])["kv_connector_extra_config"]
            assert extra["spark_cache_async_page_capture_lease_contract"] == source.LEASE_CONTRACT
            assert extra["spark_cache_root"] == "/cache/persistent/qwen38-flash-next-qad-tp2-" + source.IDENTITY


def test_tp2_memory_and_port_trial_are_explicit_and_bound_to_both_ranks(pair, tmp_path):
    settings = options(local_kv_cache_gib=33, local_master_port=29639)
    ordinary, _ = compose.build(PAIR + "-sparkcache", pair, **options())
    selected, files = compose.build(PAIR + "-sparkcache", pair, **settings)
    assert selected["id"] != ordinary["id"]
    assert compose.selection_options(selected) == settings
    for rank in range(2):
        spec = json.loads(files[f"rank{rank}/container.json"])
        args = spec["command"]
        assert args[args.index("--kv-cache-memory-bytes") + 1] == str(33 * 1024 ** 3)
        assert args[args.index("--master-port") + 1] == "29639"
    target = tmp_path / "pair"
    compose.render(PAIR + "-sparkcache", pair, target, **settings)
    manifest, _ = compose.load_deployment(target)
    assert manifest == selected


@pytest.mark.parametrize("kv_gib", [24, 40, 41, True])
def test_tp2_does_not_inherit_the_tp4_memory_alternative(pair, kv_gib):
    with pytest.raises(ValueError, match="TP2 KV alternative is 33"):
        compose.specifications(PAIR, pair, **options(local_kv_cache_gib=kv_gib))


@pytest.mark.parametrize("filename", ["config.json", "sparkcache.json"])
def test_tp2_source_selection_is_local_only_and_public_digest_is_unchanged(pair, filename):
    profile = adapter.read(adapter.CONFIG_ROOT / filename)
    profile["image_extension"] = source.IDENTITY
    with pytest.raises(ValueError, match="requires an explicit local"):
        adapter.image_policy(profile)
    with pytest.raises(ValueError, match="cannot be overridden"):
        compose.specifications(PAIR, pair, local_image_id=IMAGE)


@pytest.mark.parametrize("environment", [
    {"VLLM_QWEN3_8_HC_PREFILL_MODE": "shard"},
    {"VLLM_QWEN3_8_HC_PREFILL_MODE": "control"},
    {"SPARKRING_FEATURES": "qwen-prefill"},
    {"SPARKRING_FEATURES": "qwen-collectives,qwen-prefill"},
])
def test_tp2_contract_rejects_tp4_hc_and_feature_activation(environment):
    profile = source.profile_settings(adapter.read(adapter.CONFIG_ROOT / "config.json"), source.IDENTITY)
    profile["environment"].update(environment)
    with pytest.raises(ValueError, match="does not support HC"):
        source.validate_profile_contract(profile)


@pytest.mark.parametrize("flag", ["--tensor-parallel-size", "--nnodes"])
def test_source_selection_rejects_mismatched_pair_geometry(flag):
    profile = adapter.read(adapter.CONFIG_ROOT / "config.json")
    profile["vllm_args"][profile["vllm_args"].index(flag) + 1] = "4"
    with pytest.raises(ValueError, match="matching TP2 pair"):
        adapter.image_policy(profile, local_source_extension=source.IDENTITY)


def test_pair_source_inventory_packages_complete_admission_ancestry(tmp_path):
    snapshot = tmp_path / "source"
    inventory = compose.source_inventory(PAIR + "-sparkcache", local_source_extension=source.IDENTITY)
    for name in inventory:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(compose.ROOT / name, target)
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from runtime.common import source_candidate, feature_candidate, cache_candidate, qwen_flash_next
assert Path(source_candidate.__file__).is_relative_to(Path(sys.argv[1]))
assert source_candidate.descriptor()['parent']['receipt_sha256']
assert feature_candidate.descriptor()['parent']['receipt_sha256']
assert cache_candidate.descriptor()['parent']['receipt_sha256']
profile = qwen_flash_next.read(qwen_flash_next.CONFIG_ROOT / 'sparkcache.json')
assert qwen_flash_next.image_verification_options(profile, local_source_extension=source_candidate.IDENTITY)['local_source_extension'] == source_candidate.IDENTITY
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(snapshot)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("settings", [{"local_kv_cache_gib": 40}, {"local_master_port": 29779}])
def test_public_profiles_reject_candidate_only_setting_overrides(site, settings):
    with pytest.raises(ValueError, match="require a local source extension"):
        compose.specifications(PROFILE, site, **settings)


def test_unchanged_parent_is_not_admitted_as_source_extension(site):
    with pytest.raises(ValueError, match="unchanged parent"):
        compose.specifications(PROFILE, site, **options(local_image_id=source.descriptor()["parent"]["image_id"]))


@pytest.mark.parametrize("profile_id,kv_gib", [(PROFILE + "-sparkcache", 40), (PAIR + "-sparkcache", 33)])
def test_candidate_compose_resolves_to_the_same_docker_spec(site, pair, compose_cli, profile_id, kv_gib):
    selected_site = pair if profile_id.startswith(PAIR) else site
    specs, image = compose.specifications(profile_id, selected_site,
                                         **options(local_kv_cache_gib=kv_gib, local_master_port=29779))
    for spec in specs:
        compose.check_equivalence(spec, image, compose.compose_text(spec, image))


@pytest.mark.parametrize("profile_id,kv_gib", [(PROFILE + "-sparkcache", 40), (PAIR + "-sparkcache", 33)])
def test_render_cli_propagates_explicit_local_selection(site, pair, tmp_path, profile_id, kv_gib):
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(pair if profile_id.startswith(PAIR) else site), encoding="utf-8")
    target = tmp_path / "deployment"
    assert coordinator.main([
        "render", profile_id, "--site", str(site_file), "--output", str(target),
        "--local-source-extension", source.IDENTITY, "--local-image-id", IMAGE,
        "--local-kv-cache-gib", str(kv_gib), "--local-master-port", "29779",
    ]) == 0
    manifest, _ = compose.load_deployment(target)
    assert compose.selection_options(manifest) == options(local_kv_cache_gib=kv_gib, local_master_port=29779)


@pytest.mark.parametrize("profile_id,kv_gib", [(PROFILE + "-sparkcache", 40), (PAIR + "-sparkcache", 33)])
def test_coordinator_admission_carries_the_explicit_source_selection(site, pair, tmp_path, monkeypatch, profile_id, kv_gib):
    selected_site = pair if profile_id.startswith(PAIR) else site
    manifest, files = compose.build(profile_id, selected_site,
                                   **options(local_kv_cache_gib=kv_gib, local_master_port=29779))
    stage = tmp_path / "staged"
    stage.mkdir()
    for name in ("compose.yaml", "container.json"):
        (stage / name).write_text(files["rank0/" + name], encoding="utf-8", newline="\n")
    (stage / "deployment.json").write_text(compose.encoded(manifest), encoding="utf-8", newline="\n")
    monkeypatch.setattr(coordinator, "stage_path", lambda *args: stage)
    observed = []
    monkeypatch.setattr(adapter, "verify_image", lambda image, **kwargs: observed.append((image, kwargs)) or {"verified": True})
    coordinator.host_operation("admit", {"manifest": manifest, "files": files, "rank": 0})
    assert observed[0][0] == IMAGE
    assert observed[0][1]["local_source_extension"] == source.IDENTITY
    assert json.loads((stage / "admission.json").read_text())["image_id"] == IMAGE
