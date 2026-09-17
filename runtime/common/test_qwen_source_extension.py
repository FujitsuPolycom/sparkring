"""Local source-image selection is explicit, TP4-only, and fully bound to exports."""

import copy
import json

import pytest

from runtime.common import compose, qwen_flash_next as adapter, source_candidate as source
from runtime.common.container_spec import docker_create
from runtime.common.test_compose import compose_cli as compose_cli
from scripts import sparkring_compose as coordinator

PROFILE = "qwen38-flash-next-qad-tp4"
IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def site():
    return compose.read_site(adapter.ROOT / f"profiles/{PROFILE}/compose/site.example.yaml")


def options(**extra):
    return {"local_source_extension": source.IDENTITY, "local_image_id": IMAGE, **extra}


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


@pytest.mark.parametrize("profile_id", ["qwen38-flash-next-tp2", "qwen38-flash-next-tp2-sparkcache"])
def test_tp2_cannot_accidentally_select_tp4_capabilities(profile_id):
    pair = compose.read_site(adapter.ROOT / "profiles/qwen38-flash-next-tp2/compose/site.example.yaml")
    with pytest.raises(ValueError, match="TP4"):
        compose.specifications(profile_id, pair, **options())


@pytest.mark.parametrize("settings", [{"local_kv_cache_gib": 40}, {"local_master_port": 29779}])
def test_public_profiles_reject_candidate_only_setting_overrides(site, settings):
    with pytest.raises(ValueError, match="require a local source extension"):
        compose.specifications(PROFILE, site, **settings)


def test_unchanged_parent_is_not_admitted_as_source_extension(site):
    with pytest.raises(ValueError, match="unchanged parent"):
        compose.specifications(PROFILE, site, **options(local_image_id=source.descriptor()["parent"]["image_id"]))


def test_candidate_compose_resolves_to_the_same_docker_spec(site, compose_cli):
    specs, image = compose.specifications(PROFILE + "-sparkcache", site,
                                         **options(local_kv_cache_gib=40, local_master_port=29779))
    for spec in specs:
        compose.check_equivalence(spec, image, compose.compose_text(spec, image))


def test_render_cli_propagates_explicit_local_selection(site, tmp_path):
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(site), encoding="utf-8")
    target = tmp_path / "deployment"
    assert coordinator.main([
        "render", PROFILE + "-sparkcache", "--site", str(site_file), "--output", str(target),
        "--local-source-extension", source.IDENTITY, "--local-image-id", IMAGE,
        "--local-kv-cache-gib", "40", "--local-master-port", "29779",
    ]) == 0
    manifest, _ = compose.load_deployment(target)
    assert compose.selection_options(manifest) == options(local_kv_cache_gib=40, local_master_port=29779)


def test_coordinator_admission_carries_the_explicit_source_selection(site, tmp_path, monkeypatch):
    manifest, files = compose.build(PROFILE + "-sparkcache", site,
                                   **options(local_kv_cache_gib=40, local_master_port=29779))
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
