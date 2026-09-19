"""Four-rank QAD selection, image admission and shared deployment contracts."""
import copy
import hashlib
import json
import subprocess

import pytest

from runtime.common import compose, feature_candidate, qwen_flash_next as adapter
from runtime.common.container_spec import docker_create
from runtime.common.test_compose import compose_cli as compose_cli
from scripts import sparkring_compose as coordinator

PROFILE = "qwen38-flash-next-qad-tp4"
LEGACY_BUILD = adapter.ROOT / "runtime/images/compositions/lil-r37-shared/local-build.json"
PUBLICATION = adapter.ROOT / "runtime/releases/shared-2026.09.3/publication.json"
BUILD = PUBLICATION
MAPS = ["1=0/2,2=0/3,3=1/3", "0=1/3,2=0/2,3=0/3",
        "0=1/2,1=1/3,3=0/2", "0=0/2,1=1/2,2=1/3"]


@pytest.fixture
def site():
    return compose.read_site(adapter.ROOT / f"profiles/{PROFILE}/compose/site.example.yaml")


@pytest.mark.parametrize("rank", range(4))
def test_tp4_settings_and_peer_maps_reach_both_backends(site, rank):
    specs, image = compose.specifications(PROFILE, site)
    spec = specs[rank]
    argv = docker_create(spec)
    assert image == adapter.read(PUBLICATION)["image_reference"]
    assert spec.image_id == adapter.read(BUILD)["image_id"]
    assert argv[argv.index(spec.image_id)+1:] == list(spec.command)
    for flag, value in [("--tensor-parallel-size", "4"), ("--nnodes", "4"),
                        ("--decode-context-parallel-size", "1"), ("--max-model-len", "262144"),
                        ("--max-num-seqs", "16"), ("--max-num-batched-tokens", "8192"),
                        ("--kv-cache-memory-bytes", "25769803776"), ("--node-rank", str(rank))]:
        assert spec.command[spec.command.index(flag)+1] == value
    assert spec.environment["B12X_ROCE_PEER_HCA_MAP"] == MAPS[rank]
    assert spec.environment["SPARKRING_FEATURES"] == "qwen-collectives,qwen4-prefill"
    assert spec.environment["VLLM_QWEN3_8_HC_PREFILL_MODE"] == "shard"
    assert spec.environment["VLLM_QWEN3_8_PREFILL_COALESCE"] == "1"
    assert spec.environment["QWEN_DISPATCH_AR_BYTES"] == "20480"
    assert spec.environment["NCCL_IB_PRESERVE_PCI_DOMAIN"] == "1"
    assert spec.environment["SPARKCACHE_ENABLED"] == "0"
    assert spec.environment["SIRCL_ENABLED"] == "0"
    assert "--kv-transfer-config" not in spec.command
    assert ("--headless" in spec.command) == (rank != 0)
    assert len(spec.mounts) == 2 and spec.mounts[0].read_only


def test_tp4_real_compose_resolution(site, compose_cli):
    specs, image = compose.specifications(PROFILE, site)
    for spec in specs:
        compose.check_equivalence(spec, image, compose.compose_text(spec, image))


@pytest.mark.parametrize("mutation", [
    lambda site: site["ranks"].pop(),
    lambda site: site["ranks"][0].pop("fabric"),
    lambda site: site["ranks"][0]["fabric"].update(site_sha256="unverified"),
    lambda site: site["ranks"][0].update(hcas=["mlx5_0", "mlx5_1"]),
])
def test_tp4_incomplete_rank_or_fabric_selection_is_rejected(site, mutation):
    mutation(site)
    with pytest.raises(ValueError):
        compose.specifications(PROFILE, site)


def test_native_profile_refuses_legacy_local_rebuild_selection(site, tmp_path, monkeypatch):
    metadata, _ = compose.profiles.load(PROFILE)
    metadata = dict(metadata, release="runtime/releases/qwen38-flash-next-qad-r37-shared/release.json")
    local_release = adapter.read(adapter.ROOT / metadata["release"])
    monkeypatch.setattr(compose.profiles, "load", lambda *args, **kwargs: (metadata, local_release))
    with pytest.raises(ValueError, match="Local image selection"):
        compose.build(PROFILE, site, local_image_id="sha256:" + "a" * 64)


@pytest.mark.parametrize("profile_id", ["qwen38-flash-next-tp2", PROFILE, PROFILE + "-sparkcache"])
def test_published_image_cannot_use_local_override(profile_id):
    owner = profile_id.removesuffix("-sparkcache")
    site = compose.read_site(adapter.ROOT / f"profiles/{owner}/compose/site.example.yaml")
    with pytest.raises(ValueError, match="cannot be overridden"):
        compose.specifications(profile_id, site, local_image_id="sha256:" + "a" * 64)


def test_registry_selection_preserves_the_tested_image_and_serving_spec(site, monkeypatch):
    published, image = compose.specifications(PROFILE, site)
    metadata, _ = compose.profiles.load(PROFILE)
    metadata = dict(metadata, release="runtime/releases/qwen38-flash-next-qad-r37-shared/release.json")
    local_release = adapter.read(adapter.ROOT / metadata["release"])
    monkeypatch.setattr(compose.profiles, "load", lambda *args, **kwargs: (metadata, local_release))
    with pytest.raises(ValueError, match="Local image selection"):
        compose.specifications(PROFILE, site)
    assert image == adapter.read(PUBLICATION)["image_reference"]
    assert {spec.image_id for spec in published} == {adapter.read(BUILD)["image_id"]}


def test_feature_verification_reads_chain_from_same_image(monkeypatch):
    image = adapter.read(LEGACY_BUILD)["image_id"]
    parent_image = adapter.publication()["image_id"]
    calls = []
    observed = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1:3] == ["image", "inspect"]:
            value = [{"Id": image, "Os": "linux", "Architecture": "arm64",
                      "Config": {"Entrypoint": ["/opt/venv/bin/python", adapter.candidate.ENTRYPOINT]}}]
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(value))
        assert "--network" in argv and argv[argv.index("--network")+1] == "none"
        assert "--gpus" not in argv and "--mount" not in argv
        if argv[-1] == feature_candidate.PARENT_RECEIPT:
            result = b"cache64 receipt"
        elif argv[-1] == "/opt/sparkring/receipts/candidate-installed.json":
            result = b"base receipt" if argv[-2] == parent_image else b"child receipt"
        elif feature_candidate.INSTALLER in argv:
            result = '{"feature": true}'
        else:
            result = '{"candidate": true}'
        return subprocess.CompletedProcess(argv, 0, stdout=result)

    monkeypatch.setattr(feature_candidate, "validate", lambda *args: observed.append(args) or {"accepted": True})
    assert adapter.verify_image(image, feature_enabled=True, run=run) == {"accepted": True}
    assert observed == [(image, b"child receipt", b"cache64 receipt", b"base receipt",
                         {"candidate": True}, {"feature": True})]
    assert len(calls) == 6


def test_base_tp4_config_cannot_enable_cache_by_toggling_an_environment_flag():
    profile = copy.deepcopy(adapter.read(adapter.TP4_CONFIG))
    profile["environment"]["SPARKCACHE_ENABLED"] = "1"
    with pytest.raises(ValueError, match="unchanged canonical"):
        adapter.canonical(profile)


@pytest.mark.parametrize("rank", range(4))
def test_tp4_cache_selection_preserves_compute_transport_and_memory(site, rank):
    native, native_image = compose.specifications(PROFILE, site)
    cached, cache_image = compose.specifications(PROFILE + "-sparkcache", site)
    base, spec = native[rank], cached[rank]
    assert cache_image == native_image and spec.image_id == base.image_id
    expected_env = dict(base.environment, SPARKCACHE_ENABLED="1")
    assert spec.environment == expected_env
    args = list(spec.command)
    assert args[args.index("--block-size") + 1] == "32"
    assert base.command[base.command.index("--block-size") + 1] == "32"
    transfer = json.loads(args[args.index("--kv-transfer-config") + 1])
    assert transfer["kv_connector"] == "SparkContextCacheConnector"
    assert transfer["kv_load_failure_policy"] == "recompute"
    extra = transfer["kv_connector_extra_config"]
    config = adapter.read(adapter.TP4_CONFIG)
    expected_identity = hashlib.sha256(
        (config["model"]["repository"] + "@" + config["model"]["revision"]).encode()
    ).hexdigest()
    assert extra["spark_cache_target_checkpoint_sha256"] == expected_identity
    assert extra["spark_cache_draft_checkpoint_sha256"] == expected_identity
    assert extra["spark_cache_model_profile"] == "qwen38-flash-next-hybrid"
    tp2 = adapter.read(adapter.CONFIG_ROOT / "sparkcache.json")["vllm_args"]
    tp2_extra = json.loads(tp2[tp2.index("--kv-transfer-config") + 1])["kv_connector_extra_config"]
    assert expected_identity == tp2_extra["spark_cache_target_checkpoint_sha256"]
    assert extra["spark_cache_root"] != tp2_extra["spark_cache_root"]
    assert args[args.index("--tensor-parallel-size") + 1] == "4"
    assert tp2[tp2.index("--tensor-parallel-size") + 1] == "2"
    assert extra["spark_cache_root"] != tp2_extra["spark_cache_root"]
    for flag in ("--kv-transfer-config",):
        index = args.index(flag)
        del args[index:index + 2]
    assert args == list(base.command)
    assert spec.mounts == base.mounts
    assert spec.memory == base.memory and spec.memory_swap == base.memory_swap


def test_tp4_cache_compose_has_four_hosts_and_source_bound_feature_admission(site, compose_cli):
    profile = PROFILE + "-sparkcache"
    specs, image = compose.specifications(profile, site)
    assert len(specs) == 4
    for spec in specs:
        compose.check_equivalence(spec, image, compose.compose_text(spec, image))
    inputs = compose.source_inventory(profile)
    assert "runtime/common/qwen_mesh.py" in inputs
    assert "runtime/common/feature_candidate.py" in inputs
    assert "runtime/images/compositions/lil-r37-shared/descriptor.json" in inputs
    assert "profiles/qwen38-flash-next-qad-tp4/config.json" in inputs
    assert "profiles/qwen38-flash-next-qad-tp4/sparkcache.json" in inputs
    manifest, files = compose.build(profile, site)
    phases = {phase["id"]: phase for phase in coordinator.plan(manifest, files, "start")["phases"]}
    assert len(phases["preflight"]["actions"]) == 4
    assert all(action["argv"][:2] == ["sudo", "-n"] for action in phases["preflight"]["actions"])


def test_tp4_direct_cache_container_name_is_distinct():
    site = dict(rank=0, master="192.0.2.1", host_ip="192.0.2.1", interface="test0",
                image=adapter.read(BUILD)["image_id"], model="/models/qad", cache="/cache/qad", remote=True)
    base = adapter.container_spec(adapter.read(adapter.TP4_CONFIG), **site)
    cached = adapter.container_spec(adapter.read(adapter.TP4_CACHE_CONFIG), **site)
    assert cached.name != base.name
    assert cached.image_id == base.image_id
