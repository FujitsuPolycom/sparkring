"""SparkCache disk budgets change capacity without changing stored-entry identities."""
import argparse
import hashlib
import json

import pytest

from runtime.common import installer, installer_image, serving, sparkcache
from runtime.common.test_sparkcache_variants import development_lock, ring_site


@pytest.mark.parametrize("text, expected", [("32G", 32 * 1024**3), ("64GiB", 64 * 1024**3),
                                           ("500M", 500 * 1024**2), ("1tib", 1024**4),
                                           ("1024", 1024), ("123B", 123)])
def test_parse_size_and_serving_arguments(text, expected):
    parser = argparse.ArgumentParser()
    serving.add_arguments(parser)
    assert serving.from_arguments(parser.parse_args(["--sparkcache-size", text])) == {"sparkcache_size": expected}
    assert serving.normalized({"sparkcache_size": expected}) == {"sparkcache_size": expected}


@pytest.mark.parametrize("value", ["nonsense", "-1G", "0G", "1.5G", "32GG", "1GB/s", "1e9", "", True, 0, -1, 1.5])
def test_invalid_sizes(value):
    with pytest.raises(ValueError, match="positive size"):
        serving.normalized({"sparkcache_size": value})


@pytest.mark.parametrize("command", [[], ["--kv-transfer-config", '{"kv_connector":"OtherConnector"}']])
def test_refuses_non_sparkcache_connector(command):
    with pytest.raises(ValueError, match="requires a SparkCache profile"):
        serving.apply(command, {"sparkcache_size": 64 * 1024**3})


@pytest.mark.parametrize("profile", installer_image.SPARKCACHE_VARIANTS)
def test_default_override_persistence_and_identity(profile, tmp_path):
    raw = ring_site(4 if profile.endswith("tp4") else 2)
    plain = installer.make_lock(profile, raw, "1" * 40, "2" * 64, image_runtime=development_lock())
    size = 64 * 1024**3 + 1
    bundle = b"test retained source"
    (tmp_path / "source.bundle").write_bytes(bundle)
    tuned = installer.make_lock(profile, raw, "1" * 40, hashlib.sha256(bundle).hexdigest(), image_runtime=development_lock(),
                                settings={"sparkcache_size": size})
    installer.write(tmp_path / "deployment.lock.json", tuned)
    restored = installer.load(tmp_path)
    assert restored["serving"]["sparkcache_size"] == size
    assert sparkcache.deployment_budget(plain) == 32 * 1024**3
    assert sparkcache.deployment_budget(restored) == size
    assert restored["id"] != plain["id"]
    for original, changed in zip(installer.specifications(plain), installer.specifications(restored), strict=True):
        before = sparkcache.connector(original.command)
        after = sparkcache.connector(changed.command)
        extra = after["kv_connector_extra_config"]
        assert before["kv_connector_extra_config"]["spark_cache_max_bytes"] == 32 * 1024**3
        assert before["kv_connector_extra_config"]["spark_cache_low_watermark_bytes"] == 28 * 1024**3
        assert extra["spark_cache_max_bytes"] == size
        assert extra["spark_cache_low_watermark_bytes"] == size * 7 // 8
        # Compare the entire connector: roots, checkpoint identities, publication
        # schema and every setting outside the two capacity keys must match.
        extra.update(spark_cache_max_bytes=32 * 1024**3, spark_cache_low_watermark_bytes=28 * 1024**3)
        assert after == before
        assert changed.environment == original.environment
        floor = extra["spark_cache_async_page_capture_slot_bytes"]
        with pytest.raises(ValueError, match="must be at least"):
            serving.apply(original.command, {"sparkcache_size": floor - 1})
        assert sparkcache.budget(serving.apply(original.command, {"sparkcache_size": floor})) == floor
    rendered = "\n".join(installer.rendered(restored).values())
    assert str(size) in rendered and str(size * 7 // 8) in rendered
    assert installer.status(tmp_path)["profile"] == profile


def test_size_line_and_round_trip_label():
    size = 64 * 1024**3
    assert sparkcache.status_line(size) == "SparkCache: 64 GiB per Spark"
    assert serving.describe({"sparkcache_size": size}, ()) == []
    assert serving.label("sparkcache_size", size) == f"--sparkcache-size {size}"
    assert json.loads(json.dumps({"sparkcache_size": size}))["sparkcache_size"] == size


def test_status_keeps_an_override_when_the_profile_is_no_longer_packaged(tmp_path):
    lock = {"selection": {"sparkcache": True, "configuration": "absent/config.json"}}
    assert sparkcache.deployment_budget(lock, root=tmp_path) is None
    lock["serving"] = {"sparkcache_size": 64 * 1024**3}
    assert sparkcache.deployment_budget(lock, root=tmp_path) == 64 * 1024**3
