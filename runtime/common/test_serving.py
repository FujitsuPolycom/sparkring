"""Per-deployment serving settings replace only the named profile values and belong to the deployment's identity."""
import json

import pytest

from runtime.common import installer, installer_image, serving
from runtime.common.test_installer import site

QWEN = "qwen38-flash-next-tp2"
DEEPSEEK = "deepseek-v41-flash-tp4"
FLAGS = ("--limit-mm-per-prompt", "--max-model-len", "--max-num-seqs", "--kv-cache-memory-bytes")


def profile_args(profile):
    card = installer.setup.selection(profile)
    return json.loads((installer.ROOT / card["configuration"]).read_text())["vllm_args"]


def test_settings_replace_only_the_named_values():
    args = profile_args(QWEN)
    changed = serving.apply(args, {"max_images": 8, "context_length": 131072, "kv_cache_gib": 32, "max_concurrency": 4})

    def value(flag):
        return changed[changed.index(flag) + 1]
    assert json.loads(value("--limit-mm-per-prompt")) == {"image": 8, "video": 1}
    assert (value("--max-model-len"), value("--max-num-seqs")) == ("131072", "4")
    assert value("--kv-cache-memory-bytes") == str(32 * 2**30)
    replaced = {changed.index(flag) + 1 for flag in FLAGS}
    assert len(changed) == len(args) and all(a == b for i, (a, b) in enumerate(zip(args, changed)) if i not in replaced)
    assert serving.describe({"max_images": 8, "kv_cache_gib": 32}, tuple(args)) == [
        "--kv-cache-gib 32 (profile: 24)", "--max-images 8 (profile: 3)"]


def test_a_setting_the_profile_does_not_set_is_refused():
    args = profile_args(DEEPSEEK)
    with pytest.raises(ValueError, match="--max-videos does not apply to this profile: its --limit-mm-per-prompt sets no video limit"):
        serving.apply(args, {"max_videos": 1})
    with pytest.raises(ValueError, match="--kv-cache-gib does not apply to this profile: it sets no --kv-cache-memory-bytes"):
        serving.apply(args, {"kv_cache_gib": 20})


def test_only_named_whole_numbers_in_range_are_settings():
    assert serving.normalized({"max_images": None, "max_videos": 0}) == {"max_videos": 0}
    for bad, message in (({"context_length": 512}, "--context-length takes a whole number of at least 1024"),
                         ({"max_concurrency": 0}, "at least 1"), ({"max_images": True}, "whole number"),
                         ({"gpu_memory": 1}, "Unknown serving setting")):
        with pytest.raises(ValueError, match=message):
            serving.normalized(bad)


def test_settings_are_part_of_the_deployment_identity_and_every_rank_command():
    plain = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    tuned = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64, settings={"max_images": 8})
    assert "serving" not in plain and tuned["serving"] == {"max_images": 8} and tuned["id"] != plain["id"]
    assert installer.validate(tuned) == tuned
    with pytest.raises(ValueError, match="changed"):
        installer.validate(dict(tuned, serving={"max_images": 9}))
    for spec in installer.specifications(tuned):
        assert json.loads(spec.command[spec.command.index("--limit-mm-per-prompt") + 1]) == {"image": 8, "video": 1}
    composed = [text for name, text in installer.rendered(tuned).items() if name.endswith("compose.yaml")]
    assert len(composed) == 2 and all('"image":8' in text for text in composed)


def test_managed_deployments_take_no_settings():
    with pytest.raises(ValueError, match="Serving settings apply to Compose deployments"):
        installer.make_lock("glm53-flash-spark-tp4-dcp1-sparkcache", site(4), "1" * 40, "2" * 64, settings={"max_images": 2})


def test_init_refuses_a_setting_that_does_not_apply_before_creating_the_deployment(tmp_path, monkeypatch):
    from runtime.host import controller, native_mesh
    from runtime.host.test_native_mesh import cluster
    monkeypatch.setattr(installer.distribution, "identity", lambda root: "1" * 40)
    value = cluster()
    planned = native_mesh.select(controller.model_site(value, DEEPSEEK), value, DEEPSEEK, invoke=lambda *a, **k: '{"mesh":null}')
    with pytest.raises(ValueError, match="does not apply to this profile"):
        installer.init(tmp_path / "deployment", DEEPSEEK, planned, image_runtime=installer_image.for_profile(DEEPSEEK),
                       settings={"kv_cache_gib": 20})
    assert not (tmp_path / "deployment").exists()
