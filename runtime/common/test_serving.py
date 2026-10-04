"""Per-deployment serving settings replace only the named profile values and belong to the deployment's identity."""
import dataclasses
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
    changed = serving.apply(args, {"max_images": 8, "context_length": 131072, "kv_cache_gib": 16, "max_concurrency": 4})

    def value(flag):
        return changed[changed.index(flag) + 1]
    assert json.loads(value("--limit-mm-per-prompt")) == {"image": 8, "video": 1}
    assert (value("--max-model-len"), value("--max-num-seqs")) == ("131072", "4")
    assert value("--kv-cache-memory-bytes") == str(16 * 2**30)
    replaced = {changed.index(flag) + 1 for flag in FLAGS}
    assert len(changed) == len(args) and all(a == b for i, (a, b) in enumerate(zip(args, changed)) if i not in replaced)
    assert serving.describe({"max_images": 8, "kv_cache_gib": 16}, tuple(args)) == [
        "--kv-cache-gib 16 (profile: 24)", "--max-images 8 (profile: 3)"]


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


def test_a_kv_cache_up_to_a_tenth_above_the_profiles_is_accepted_with_a_warning():
    args = profile_args(QWEN)
    assert serving.apply(args, {"kv_cache_gib": 24}) == tuple(args) and serving.warnings({"kv_cache_gib": 24}, tuple(args)) == []
    raised = serving.apply(args, {"kv_cache_gib": 26})
    assert raised[raised.index("--kv-cache-memory-bytes") + 1] == str(26 * 2**30)
    assert serving.warnings({"kv_cache_gib": 26, "max_images": 9}, tuple(args)) == [
        "--kv-cache-gib 26 is above the profile's 24 GiB, leaving each Spark 2 GiB less for images and long "
        "requests. This value has not been validated as stable."]
    with pytest.raises(ValueError, match="--kv-cache-gib 27 is more than 26, a tenth above the profile's 24"):
        serving.apply(args, {"kv_cache_gib": 27})
    # A small profile value still admits one more GiB.
    assert [serving.ceiling(value) for value in (4, 10, 24, 40)] == [5, 11, 26, 44]


def test_save_cpu_is_a_switch_that_sets_the_reader_window_on_every_rank():
    assert serving.normalized({"save_cpu": True, "max_images": None}) == {"save_cpu": True}
    assert serving.normalized({"save_cpu": None}) == {}
    with pytest.raises(ValueError, match="--save-cpu is a switch without a value"):
        serving.normalized({"save_cpu": 1})
    assert serving.environment({"save_cpu": True, "max_images": 2}) == {"SPARKRING_SHM_BUSY_LOOP_S": "0.002"}
    args = profile_args(QWEN)
    assert serving.apply(args, {"save_cpu": True}) == tuple(args)
    assert serving.describe({"save_cpu": True, "max_images": 2}, tuple(args)) == [
        "--max-images 2 (profile: 3)", "--save-cpu (profile: off)"]
    plain = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    lock = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64, settings={"save_cpu": True})
    assert lock["id"] != plain["id"]
    for spec, default in zip(installer.specifications(lock), installer.specifications(plain), strict=True):
        assert spec.environment == {**default.environment, "SPARKRING_SHM_BUSY_LOOP_S": "0.002"}
        assert spec.command == default.command
    assert all("SPARKRING_SHM_BUSY_LOOP_S" not in spec.environment for spec in installer.specifications(plain))


def test_save_cpu_needs_an_image_that_reads_the_reader_window(tmp_path, monkeypatch):
    serving.check_image({"save_cpu": True, "max_images": 2}, "an-image", ("shm_reader_window",))
    serving.check_image({"max_images": 2}, "an-image", ())
    with pytest.raises(ValueError, match="--save-cpu needs an image whose vLLM reads SPARKRING_SHM_BUSY_LOOP_S, and an-image "
                                         "does not. Leave out --save-cpu or choose another image."):
        serving.check_image({"save_cpu": True}, "an-image", ())
    # init refuses it on an image without the reader window before it creates the deployment.
    monkeypatch.setattr(installer.distribution, "identity", lambda root: "1" * 40)
    image = installer_image.for_profile(QWEN, json.loads(installer_image.lock_path("plainstatus").read_text(encoding="utf-8")))
    with pytest.raises(ValueError, match="dev-20260928-plainstatus-cuda1342-nccl2323-status033 does not"):
        installer.init(tmp_path / "deployment", QWEN, site(), image_runtime=image, settings={"save_cpu": True})
    assert not (tmp_path / "deployment").exists()
    # A recorded deployment's lock is not checked again, so it still loads and runs.
    lock = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64, image_runtime=image, settings={"save_cpu": True})
    assert installer.specifications(lock)


def value_of(command, flag):
    return command[command.index(flag) + 1]


def test_api_port_replaces_the_port_on_every_rank_and_in_the_health_check():
    plain = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    moved = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64, settings={"api_port": 9100})
    assert moved["serving"] == {"api_port": 9100} and moved["id"] != plain["id"]
    specs = installer.specifications(moved)
    assert [value_of(spec.command, "--port") for spec in specs] == ["9100", "9100"]
    assert "http://127.0.0.1:9100/health" in specs[0].health_command[-1] and specs[1].health_command == ()
    defaults = installer.specifications(plain)
    assert [len(spec.command) for spec in specs] == [len(spec.command) for spec in defaults]
    assert installer.connection(moved) == {**installer.connection(plain), "port": 9100,
                                           "api_url": installer.connection(plain)["api_url"].replace(":8000/", ":9100/")}
    assert serving.describe({"api_port": 9100}, tuple(defaults[0].command)) == ["--api-port 9100 (profile: 8000)"]


def test_api_bind_replaces_the_listen_address_of_the_api_rank_only():
    plain = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    bound = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64, settings={"api_bind": "192.0.2.50"})
    assert bound["id"] != plain["id"] and installer.validate(bound) == bound
    api, worker = installer.specifications(bound)
    assert value_of(api.command, "--host") == "192.0.2.50" and value_of(worker.command, "--host") == "0.0.0.0"
    assert "http://192.0.2.50:8000/health" in api.health_command[-1]
    assert dataclasses.replace(worker, labels={}) == dataclasses.replace(installer.specifications(plain)[1], labels={})
    # The checks, and the URL SparkRing shows without another address, use the listen address.
    assert installer.connection(bound)["api_url"] == "http://192.0.2.50:8000/v1"
    both = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64, settings={"api_bind": "192.0.2.50", "api_port": 9100})
    assert "http://192.0.2.50:9100/health" in installer.specifications(both)[0].health_command[-1]
    assert installer.connection(both)["api_url"] == "http://192.0.2.50:9100/v1"
    base = installer.specifications(plain)[0].command
    assert serving.describe({"api_bind": "192.0.2.50"}, tuple(base)) == ["--api-bind 192.0.2.50 (profile: 0.0.0.0)"]
    composed = installer.rendered(both)
    assert "- 192.0.2.50" in composed["rank0/compose.yaml"] and "- 192.0.2.50" not in composed["rank1/compose.yaml"]
    assert "urlopen('http://192.0.2.50:9100/health'" in composed["rank0/compose.yaml"]


def test_api_endpoint_settings_take_a_port_in_range_and_an_address_a_spark_can_listen_on():
    assert serving.normalized({"api_port": 1024, "api_bind": "10.1.2.3"}) == {"api_bind": "10.1.2.3", "api_port": 1024}
    for bad in (1023, 65536, "8080", True):
        with pytest.raises(ValueError, match="--api-port takes a whole number from 1024 to 65535"):
            serving.normalized({"api_port": bad})
    for bad in ("0.0.0.0", "0.1.2.3", "224.0.0.1", "255.255.255.255", "1.2.3", "010.0.0.1", "spark", 3, "::1"):
        with pytest.raises(ValueError, match="--api-bind takes one IPv4 address of the Spark that serves the API, "
                                             "such as 192.0.2.10; without it the API listens on every address"):
            serving.normalized({"api_bind": bad})


def test_api_port_refuses_the_ports_sparkring_and_the_profile_use():
    args = profile_args(QWEN)
    assert serving.reserved_ports(args) == [(2222, "the port of SparkRing's administration SSH"),
                                            (5255, "the port of SparkRing's image relay"),
                                            (29500, "vLLM's default master port"),
                                            (29638, "the profile's --master-port")]
    for port, reason in serving.reserved_ports(args):
        with pytest.raises(ValueError, match=f"--api-port {port} is {reason}. Choose another port."):
            serving.apply(args, {"api_port": port})
    assert value_of(serving.apply(args, {"api_port": 8001}), "--port") == "8001"
    # The profile's own port is accepted; it names another deployment with the same command.
    assert serving.apply(args, {"api_port": 8000}) == tuple(args)


def test_without_settings_a_container_is_unchanged():
    lock = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    for rank, spec in enumerate(installer.specifications(dict(lock, serving={}))):
        assert serving.container(spec, {}, rank=rank) is spec


def test_compose_renders_the_api_endpoint_as_install_does():
    from runtime.common import compose
    example = compose.read_site(installer.ROOT / "profiles" / QWEN / "compose" / "site.example.yaml")
    plain, _ = compose.build(QWEN, example)
    manifest, files = compose.build(QWEN, example, serving={"api_port": 9100, "api_bind": "192.0.2.50"})
    assert manifest["serving"] == {"api_bind": "192.0.2.50", "api_port": 9100} and manifest["id"] != plain["id"]
    specs, _ = compose.specifications(QWEN, example, image_runtime=manifest["image_runtime"], serving=manifest["serving"])
    assert value_of(specs[0].command, "--host") == "192.0.2.50" and value_of(specs[1].command, "--host") == "0.0.0.0"
    assert "urlopen('http://192.0.2.50:9100/health'" in files["rank0/compose.yaml"]
    assert "'9100'" in files["rank1/compose.yaml"] and "192.0.2.50" not in files["rank1/compose.yaml"]


# Thinking settings write the model's own chat template arguments into vLLM's
# --default-chat-template-kwargs on the API rank (runtime/common/thinking.py).
KWARGS = "--default-chat-template-kwargs"


def chat_defaults(command):
    return json.loads(command[command.index(KWARGS) + 1]) if KWARGS in command else None


def example_site(profile):
    from runtime.common import compose
    return compose.read_site(installer.ROOT / "profiles" / profile.removesuffix("-sparkcache") / "compose/site.example.yaml")


@pytest.mark.parametrize("profile, settings, rendered", [
    ("qwen38-flash-next-tp2", {"reasoning_effort": "low"}, '{"reasoning_effort":"low"}'),
    ("qwen38-flash-next-qad-tp4", {"thinking": "off"}, '{"enable_thinking":false}'),
    ("swift15-qwen38-flash-next-tp2", {"reasoning_effort": "medium"}, '{"reasoning_effort":"medium"}'),
    ("glm53-flash-nvfp4-spark-tp2", {"reasoning_effort": "high"}, '{"reasoning_effort":"high"}'),
    ("glm53-flash-nvfp4-spark-tp4", {"reasoning_effort": "low"}, '{"reasoning_effort":"low"}'),
    ("mimo-v26-flash-mopd-tp2", {"thinking": "off"}, '{"enable_thinking":false}'),
    ("deepseek-v41-flash-tp4", {"reasoning_effort": "max"}, '{"reasoning_effort":"max"}'),
    ("deepseek-v41-flash-tp4", {"reasoning_effort": 60}, '{"reasoning_effort":60}'),
    ("deepseek-v41-flash-tp4", {"thinking": "off"}, '{"thinking":false}'),
])
def test_thinking_settings_set_the_models_arguments_on_the_api_rank_only(profile, settings, rendered):
    from runtime.common import compose
    site_value = example_site(profile)
    plain, _ = compose.build(profile, site_value)
    tuned, files = compose.build(profile, site_value, serving=settings)
    assert tuned["serving"] == settings and tuned["id"] != plain["id"] and "serving" not in plain
    specs, _ = compose.specifications(profile, site_value, serving=settings)
    defaults, _ = compose.specifications(profile, site_value)
    # The API rank's command gains the flag and its JSON object at the end; the
    # headless ranks keep the profile's command.
    assert specs[0].command == (*defaults[0].command, KWARGS, rendered)
    assert [spec.command for spec in specs[1:]] == [spec.command for spec in defaults[1:]]
    assert all(KWARGS not in spec.command for spec in defaults)
    composed = [text for name, text in sorted(files.items()) if name.endswith("compose.yaml")]
    assert f"- '{rendered}'" in composed[0] and not any(KWARGS in text for text in composed[1:])


def test_thinking_settings_are_part_of_the_installer_lock_and_its_api_rank_command():
    plain = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    tuned = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64, settings={"reasoning_effort": "low"})
    assert tuned["serving"] == {"reasoning_effort": "low"} and tuned["id"] != plain["id"]
    assert installer.validate(tuned) == tuned
    commands = [spec.command for spec in installer.specifications(tuned)]
    assert [chat_defaults(command) for command in commands] == [{"reasoning_effort": "low"}, None]
    assert commands[1] == installer.specifications(plain)[1].command
    # Another checkpoint of the profile keeps the profile's thinking behaviour.
    other = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64, "jmni-qad5500-hybrid", settings={"thinking": "off"})
    assert chat_defaults(installer.specifications(other)[0].command) == {"enable_thinking": False}
    composed = [text for name, text in installer.rendered(tuned).items() if name.endswith("compose.yaml")]
    assert len(composed) == 2 and sum("reasoning_effort" in text for text in composed) == 1


@pytest.mark.parametrize("profile, settings, message", [
    ("glm53-flash-nvfp4-spark-tp2", {"thinking": "off"},
     "--thinking off does not apply to glm53-flash-nvfp4-spark-tp2: its model always thinks"),
    ("mimo-v26-flash-mopd-tp2", {"reasoning_effort": "low"},
     "--reasoning-effort does not apply to mimo-v26-flash-mopd-tp2: its model has no effort levels"),
    ("qwen38-flash-next-tp2", {"reasoning_effort": "high"},
     "--reasoning-effort high is not a level that qwen38-flash-next-tp2 accepts; choose low, medium or xhigh"),
    ("deepseek-v41-flash-tp4", {"reasoning_effort": 101}, "or a whole number from 1 to 100"),
    ("qwen38-flash-next-tp2", {"reasoning_effort": "low", "thinking": "off"}, "choose one"),
    ("qwen38-flash-next-tp2-sparkcache", {"thinking": "off"}, "no thinking behaviour is recorded"),
])
def test_thinking_settings_the_model_does_not_accept_are_refused_before_rendering(profile, settings, message):
    from runtime.common import compose
    with pytest.raises(ValueError, match=message):
        compose.build(profile, example_site(profile), serving=settings)


def test_init_refuses_a_thinking_setting_the_model_does_not_accept_before_creating_the_deployment(tmp_path, monkeypatch):
    monkeypatch.setattr(installer.distribution, "identity", lambda root: "1" * 40)
    profile = "glm53-flash-nvfp4-spark-tp2"
    with pytest.raises(ValueError, match="its model always thinks"):
        installer.init(tmp_path / "deployment", profile, site(), image_runtime=installer_image.for_profile(profile),
                       settings={"thinking": "off"})
    assert not (tmp_path / "deployment").exists()


def test_thinking_values_are_words_or_whole_numbers_and_thinking_takes_only_off():
    assert serving.normalized({"reasoning_effort": "low", "thinking": None}) == {"reasoning_effort": "low"}
    assert serving.normalized({"reasoning_effort": "60"}) == serving.normalized({"reasoning_effort": 60}) == {
        "reasoning_effort": 60}
    assert serving.normalized({"thinking": "off"}) == {"thinking": "off"}
    for bad, message in (({"reasoning_effort": "Low"}, "--reasoning-effort takes a level name"),
                         ({"reasoning_effort": ""}, "takes a level name"), ({"reasoning_effort": 0}, "at least 1"),
                         ({"reasoning_effort": True}, "takes a level name"), ({"thinking": "on"}, "--thinking takes off"),
                         ({"thinking": False}, "--thinking takes off")):
        with pytest.raises(ValueError, match=message):
            serving.normalized(bad)
    with pytest.raises(ValueError, match="apply only with the profile and checkpoint"):
        serving.apply(profile_args(QWEN), {"thinking": "off"})


def test_thinking_flags_parse_as_serving_settings():
    import argparse
    parser = argparse.ArgumentParser()
    serving.add_arguments(parser)
    assert serving.from_arguments(parser.parse_args([])) == {}
    args = parser.parse_args(["--reasoning-effort", "low", "--max-images", "2"])
    assert serving.from_arguments(args) == {"max_images": 2, "reasoning_effort": "low"}
    assert serving.from_arguments(parser.parse_args(["--thinking", "off"])) == {"thinking": "off"}
    with pytest.raises(SystemExit):
        parser.parse_args(["--thinking", "on"])
    assert [serving.label(name, value) for name, value in sorted(serving.from_arguments(args).items())] == [
        "--max-images 2", "--reasoning-effort low"]


def test_plans_show_a_thinking_setting_beside_the_models_default():
    args = tuple(profile_args(QWEN))
    assert serving.describe({"reasoning_effort": "low", "max_images": 2}, args, model=(QWEN, None)) == [
        "--max-images 2 (profile: 3)", "--reasoning-effort low (model default: xhigh)"]
    assert serving.describe({"thinking": "off"}, args, model=("mimo-v26-flash-mopd-tp2", None)) == [
        "--thinking off (model default: on)"]
    assert serving.describe({"reasoning_effort": "low"}, args) == ["--reasoning-effort low (model default: not recorded)"]


def test_without_thinking_settings_commands_compose_files_and_locks_are_unchanged():
    from runtime.common import compose
    for profile in compose.SUPPORTED:
        site_value = example_site(profile)
        specs, _ = compose.specifications(profile, site_value)
        assert compose.build(profile, site_value) == compose.build(profile, site_value, serving={})
        assert compose.build(profile, site_value, serving={"thinking": None}) == compose.build(profile, site_value)
        assert all(KWARGS not in spec.command for spec in specs)
        assert all(serving.apply(spec.command, {}) == spec.command for spec in specs)
    lock = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64, settings={"thinking": None, "reasoning_effort": None})
    assert "serving" not in lock and lock == installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    # The other serving settings render as before.
    tuned = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64, settings={"max_images": 8})
    assert all(KWARGS not in spec.command for spec in installer.specifications(tuned))
