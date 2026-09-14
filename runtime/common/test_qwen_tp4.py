"""Four-rank QAD selection, image admission and shared deployment contracts."""
import copy
import json
import subprocess

import pytest

from runtime.common import compose, feature_candidate, qwen_flash_next as adapter
from runtime.common.container_spec import docker_create
from runtime.common.test_compose import compose_cli as compose_cli
from scripts import sparkring_compose as coordinator

PROFILE = "qwen38-flash-next-qad-tp4"
BUILD = adapter.ROOT / "runtime/images/compositions/lil-r37-shared/local-build.json"
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
    assert image == adapter.read(BUILD)["image_tag"]
    assert argv[argv.index(spec.image_id)+1:] == list(spec.command)
    for flag, value in [("--tensor-parallel-size", "4"), ("--nnodes", "4"),
                        ("--decode-context-parallel-size", "1"), ("--max-model-len", "262144"),
                        ("--max-num-seqs", "16"), ("--max-num-batched-tokens", "8192"),
                        ("--kv-cache-memory-bytes", "25769803776"), ("--node-rank", str(rank))]:
        assert spec.command[spec.command.index(flag)+1] == value
    assert spec.environment["B12X_ROCE_PEER_HCA_MAP"] == MAPS[rank]
    assert spec.environment["SPARKRING_FEATURES"] == "qwen-collectives,qwen-prefill"
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


def test_local_rebuild_identity_is_bound_to_export_and_plan(site, tmp_path):
    image_id = "sha256:" + "a" * 64
    reference, _ = compose.build(PROFILE, site)
    selected, _ = compose.build(PROFILE, site, local_image_id=image_id)
    assert reference["id"] != selected["id"]
    assert selected["image_id"] == image_id
    target = tmp_path / "deployment"
    compose.render(PROFILE, site, target, local_image_id=image_id)
    loaded, files = compose.load_deployment(target)
    assert loaded == selected
    plan = coordinator.plan(loaded, files, "start")
    phases = {phase["id"]: phase for phase in plan["phases"]}
    assert len(phases["preflight"]["actions"]) == 4
    assert all(action["argv"][:2] == ["sudo", "-n"] for action in phases["preflight"]["actions"])
    assert [a["host"] for a in phases["start-worker"]["actions"]] == ["spark1", "spark2", "spark3"]
    assert [a["host"] for a in phases["start-api"]["actions"]] == ["spark0"]


def test_published_tp2_image_cannot_use_local_override():
    site = compose.read_site(adapter.CONFIG_ROOT / "compose/site.example.yaml")
    with pytest.raises(ValueError, match="cannot be overridden"):
        compose.specifications("qwen38-flash-next-tp2", site, local_image_id="sha256:" + "a" * 64)


def test_feature_verification_reads_chain_from_same_image(monkeypatch):
    image = adapter.read(BUILD)["image_id"]
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


def test_canonical_tp4_config_cannot_enable_unqualified_cache():
    profile = copy.deepcopy(adapter.read(adapter.TP4_CONFIG))
    profile["environment"]["SPARKCACHE_ENABLED"] = "1"
    with pytest.raises(ValueError, match="unchanged canonical"):
        adapter.canonical(profile)
