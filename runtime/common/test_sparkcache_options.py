"""Deployment integrity and diagnostic settings survive rendering and reload."""
import argparse
import hashlib
import subprocess
import sys

import pytest

from runtime.common import compose, installer, installer_image, serving, sparkcache
from runtime.common.test_sparkcache_variants import development_lock, ring_site


@pytest.mark.parametrize("mode", ["fast", "sha256", "none"])
def test_arguments(mode):
    parser = argparse.ArgumentParser()
    serving.add_arguments(parser)
    assert serving.from_arguments(parser.parse_args([])) == {}
    assert serving.from_arguments(parser.parse_args(["--sparkcache-integrity", mode, "--sparkcache-debug"])) == {
        "sparkcache_integrity": mode, "sparkcache_debug": True}
    with pytest.raises(SystemExit):
        parser.parse_args(["--sparkcache-integrity", "crc32"])


@pytest.mark.parametrize("settings", [{"sparkcache_integrity": "crc32"}, {"sparkcache_integrity": True},
                                     {"sparkcache_debug": False}, {"sparkcache_debug": "yes"}])
def test_invalid_record_values(settings):
    with pytest.raises(ValueError):
        serving.normalized(settings)


@pytest.mark.parametrize("settings", [{"sparkcache_integrity": "none"}, {"sparkcache_debug": True}])
@pytest.mark.parametrize("command", [[], ["--kv-transfer-config", '{"kv_connector":"OtherConnector"}']])
def test_refuses_non_sparkcache(settings, command):
    with pytest.raises(ValueError, match="requires a SparkCache profile"):
        serving.apply(command, settings)


@pytest.mark.parametrize("profile", installer_image.SPARKCACHE_VARIANTS)
@pytest.mark.parametrize("mode", ["fast", "sha256", "none"])
def test_saved_settings_render_on_every_rank(profile, mode, tmp_path):
    raw = ring_site(4 if profile.endswith("tp4") else 2)
    bundle = b"retained deployment source"
    (tmp_path / "source.bundle").write_bytes(bundle)
    plain = installer.make_lock(profile, raw, "1" * 40, hashlib.sha256(bundle).hexdigest(),
                                image_runtime=development_lock())
    tuned = installer.make_lock(profile, raw, "1" * 40, hashlib.sha256(bundle).hexdigest(),
                                image_runtime=development_lock(),
                                settings={"sparkcache_integrity": mode, "sparkcache_debug": True})
    installer.write(tmp_path / "deployment.lock.json", tuned)
    restored = installer.load(tmp_path)
    assert restored["serving"]["sparkcache_integrity"] == mode
    assert restored["serving"]["sparkcache_debug"] is True
    assert restored["id"] != plain["id"]
    for rank, (original, changed) in enumerate(zip(installer.specifications(plain),
                                                  installer.specifications(restored), strict=True)):
        before, after = sparkcache.connector(original.command), sparkcache.connector(changed.command)
        assert after["kv_connector_extra_config"]["spark_cache_payload_integrity"] == mode
        # Upstream derives publication identity from integrity. No other key changes.
        after["kv_connector_extra_config"].pop("spark_cache_payload_integrity")
        before["kv_connector_extra_config"].pop("spark_cache_payload_integrity", None)
        assert after == before
        assert changed.environment == {**original.environment, "SPARK_CONTEXT_CACHE_TRACE_REUSE": "1",
                                       "VLLM_LOGGING_CONFIG_PATH": sparkcache.DEBUG_TARGET}
        mount = next(m for m in changed.mounts if m.target == sparkcache.DEBUG_TARGET)
        assert mount.read_only
        assert mount.source == restored["site"]["ranks"][rank]["repository"] + "/" + sparkcache.DEBUG_SOURCE
        assert all(m.target != sparkcache.DEBUG_TARGET for m in original.mounts)
    rendered = "\n".join(installer.rendered(restored).values())
    assert sparkcache.DEBUG_TARGET in rendered
    assert "SPARK_CONTEXT_CACHE_TRACE_REUSE" in rendered
    assert sparkcache.option_lines(sparkcache.deployment_options(restored)) == [
        "SparkCache integrity: " + mode, "SparkCache logs: debug"]
    assert sparkcache.option_lines(sparkcache.deployment_options(plain)) == ["SparkCache integrity: fast"]
    assert serving.describe({name: value for name, value in restored["serving"].items()
                             if name in sparkcache.SETTINGS}, ()) == []
    assert serving.label("sparkcache_debug", True) == "--sparkcache-debug"


def test_debug_configuration_emits_only_sparkcache_debug():
    # Isolate dictConfig from pytest's own logging handlers, and include loggers
    # created both before and after configuration, as vLLM workers do.
    code = '''
import json, logging, logging.config, sys
logging.getLogger("vllm.engine")
logging.getLogger("vllm.spark_context_cache.streaming")
logging.config.dictConfig(json.load(open(sys.argv[1])))
for name in ("vllm.spark_context_cache", "vllm.spark_context_cache.streaming", "sparkcache.streaming.timing"):
    logging.getLogger(name).debug("cache-debug %s", name)
for name in ("vllm", "vllm.engine", "vllm.worker"):
    logging.getLogger(name).debug("unwanted-engine-debug")
    logging.getLogger(name).info("engine-info %s", name)
'''
    result = subprocess.run([sys.executable, "-c", code, str(installer.ROOT / sparkcache.DEBUG_SOURCE)],
                            check=True, capture_output=True, text=True)
    assert result.stdout.count("cache-debug") == 3
    assert result.stdout.count("engine-info") == 3
    assert "unwanted-engine-debug" not in result.stdout + result.stderr


def test_non_sparkcache_status_has_no_cache_options():
    assert sparkcache.deployment_options({"selection": {"sparkcache": False}}) == {}


def test_compose_binds_debug_configuration_to_source_inventory():
    profile = "qwen38-flash-next-csf-sparkcache-tp2"
    lock = installer.make_lock(profile, ring_site(2), "1" * 40, "2" * 64, image_runtime=development_lock())
    specs, _ = compose.specifications(profile, installer.compose_site(lock),
                                      image_runtime=development_lock(),
                                      serving={"sparkcache_debug": True, "sparkcache_integrity": "none"})
    for spec in specs:
        assert spec.environment["VLLM_LOGGING_CONFIG_PATH"] == sparkcache.DEBUG_TARGET
        assert any(m.target == sparkcache.DEBUG_TARGET and m.read_only for m in spec.mounts)
    assert sparkcache.DEBUG_SOURCE in compose.source_inventory(profile)


def test_vendored_integrity_schemas_miss_each_other(tmp_path):
    code = '''
import os, sys
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, sys.argv[1])
from sparkcache.spark_context_cache_config import parse_connector_config
from sparkcache.spark_context_cache_store import ManifestStore, ContextChunk, StateRecord
class FullAttentionSpec:
    block_size = 512
    storage_block_size = 512
    page_size_bytes = 528
groups = NS(kv_cache_groups=(NS(kv_cache_spec=FullAttentionSpec(), is_eagle_group=False, layer_names=("full",)),))
def parse(mode):
    values = dict(spark_cache_root=sys.argv[2], spark_cache_model_profile="deepseek-v4-fp8-hma",
                  spark_cache_target_checkpoint_sha256="1" * 64, spark_cache_draft_checkpoint_sha256="2" * 64,
                  spark_cache_draft_policy="separate", spark_cache_publication_schema="tail-cow-v2")
    if mode is not None:
        values["spark_cache_payload_integrity"] = mode
    transfer = NS(get_from_extra_config=lambda key, default=None: values.get(key, default),
                  kv_load_failure_policy="recompute")
    config = NS(kv_transfer_config=transfer, cache_config=NS(block_size=512), model_config=NS(max_model_len=0),
                parallel_config=NS(tensor_parallel_size=1, decode_context_parallel_size=1,
                                   cp_kv_cache_interleave_size=1, pipeline_parallel_size=1))
    return parse_connector_config(config, transfer, groups)
os.environ.pop("SPARK_CONTEXT_CACHE_PAYLOAD_INTEGRITY", None)
schemas = {"fast": "page-tail-cow-v3-fast", "none": "page-tail-cow-v3-none", "sha256": "page-tail-cow-v2"}
identities = {}
for mode, schema in schemas.items():
    config = parse(mode)
    assert config.publication_schema == schema
    identities[mode] = config.build_identity(0, 0)
assert parse(None).build_identity(0, 0) == identities["fast"]
assert len({value.storage_key for value in identities.values()}) == 3
os.environ["SPARK_CONTEXT_CACHE_TRACE_REUSE"] = "1"
os.environ["VLLM_LOGGING_CONFIG_PATH"] = "diagnostic-config.json"
assert parse("fast").build_identity(0, 0) == identities["fast"]
for mode, identity in identities.items():
    store = ManifestStore(Path(sys.argv[2]) / mode)
    chunk = ContextChunk(0, identity.chunk_tokens,
                         {record: b"opaque-state" for record in identity.required_records})
    store.commit(identity=identity, context_digest="a" * 64, chunks=[chunk])
    assert store.lookup(identity, "a" * 64).is_hit
    for other_mode, other in identities.items():
        if other_mode != mode:
            result = store.lookup(other, "a" * 64)
            assert not result.is_hit and result.reason == "absent", result
'''
    subprocess.run([sys.executable, "-c", code, str(installer.ROOT / "integrations/sparkcache/4a17f1f5198e"),
                    str(tmp_path)], check=True, capture_output=True, text=True)
