"""Profile choices survive every renderer, with checkpoint-bound thinking and sampling defaults."""
import copy
import hashlib
import json

import pytest

from runtime.common import compose, installer, profiles, serving, thinking, toolchain_profiles
from runtime.common.test_installer import site
from runtime.common.test_serving import chat_defaults, example_site


def glm_profiles():
    return [ident for ident in profiles.catalog()
            if (record := thinking.of(ident)) and record["name"] in {"glm53-template", "glm53-flash-template"}]


@pytest.mark.parametrize("profile", glm_profiles())
def test_every_glm_checkpoint_renders_high_only_on_api_rank(profile):
    definition, _ = profiles.load(profile)
    configuration = profiles.read_json(profiles.ROOT / definition["configuration"]["path"])
    assert configuration["serving_defaults"] == {"reasoning_effort": "high"}
    args = configuration["vllm_args"]
    assert args.count("--generation-config") == 1 and args[args.index("--generation-config") + 1] == "auto"
    assert "--override-generation-config" not in args
    for checkpoint in configuration.get("checkpoints") or [None]:
        assert serving.defaults(profile, checkpoint) == {"reasoning_effort": "high"}
        for rank in range(toolchain_profiles.node_count(configuration)):
            spec = toolchain_profiles.container_spec(
                configuration, rank=rank, master="192.0.2.1", host_ip=f"192.0.2.{rank + 1}",
                interface="eth0", image="sha256:" + "1" * 64, model="/models/target", cache="/cache",
                remote=True, checkpoint=checkpoint)
            assert chat_defaults(spec.command) == (None if rank else {"reasoning_effort": "high"})


@pytest.mark.parametrize("choice", [None, {"reasoning_effort": "max"}, {"max_concurrency": 4}])
def test_installer_and_compose_record_and_render_effective_default(choice):
    profile = "glm53-flash-nvfp4-spark-tp2"
    expected = {"reasoning_effort": "high", **(choice or {})}
    lock = installer.make_lock(profile, site(), "1" * 40, "2" * 64, settings=choice)
    assert lock["serving"] == expected
    assert installer.validate(lock) == lock
    commands = [spec.command for spec in installer.specifications(lock)]
    assert [chat_defaults(command) for command in commands] == [{"reasoning_effort": expected["reasoning_effort"]}, None]
    manifest, files = compose.build(profile, example_site(profile), serving=choice)
    assert manifest["serving"] == expected
    assert json.loads(files["rank0/container.json"])["command"] == list(
        compose.specifications(profile, example_site(profile), serving=choice,
                               image_runtime=compose.installer_image_runtime(profile))[0][0].command)
    assert "--default-chat-template-kwargs" in files["rank0/compose.yaml"]
    assert "--default-chat-template-kwargs" not in files["rank1/compose.yaml"]


@pytest.mark.parametrize("value", [None, [], "high", {"max_concurrency": 4}, {"reasoning_effort": None},
                                  {"reasoning_effort": True}, {"reasoning_effort": "High"},
                                  {"thinking": "on"}, {"reasoning_effort": "medium"},
                                  {"reasoning_effort": "high", "thinking": "off"}])
def test_serving_profile_schema_refuses_invalid_defaults(value):
    definition, _ = profiles.load("glm53-flash-nvfp4-spark-tp2")
    configuration = profiles.read_json(profiles.ROOT / definition["configuration"]["path"])
    configuration["serving_defaults"] = value
    with pytest.raises(ValueError):
        serving.profile_defaults(configuration)


def test_profile_resolution_validates_defaults_and_all_checkpoints(monkeypatch):
    original = profiles.read_json
    def changed(path):
        value = original(path)
        if str(path).replace("\\", "/").endswith("glm53-flash-nvfp4-spark-tp2/config.json"):
            value["serving_defaults"] = {"reasoning_effort": "medium"}
        return value
    monkeypatch.setattr(profiles, "read_json", changed)
    with pytest.raises(ValueError, match="is not a level.*low, high or max"):
        profiles.resolve("glm53-flash-nvfp4-spark-tp2")


def test_off_overrides_profile_effort_without_leaving_an_effort_argument(monkeypatch):
    monkeypatch.setattr(serving, "defaults", lambda *args, **kwargs: {"reasoning_effort": "low"})
    values = serving.effective("qwen38-flash-next-tp2", {"thinking": "off"})
    assert values == {"thinking": "off"}
    command = ("serve", "--default-chat-template-kwargs", '{"reasoning_effort":"low","keep":true}')
    changed = serving.apply(command, values, model=("qwen38-flash-next-tp2", None))
    assert chat_defaults(changed) == {"enable_thinking": False, "keep": True}
    with pytest.raises(ValueError, match="always thinks"):
        serving.apply(command, values, model=("glm53-flash-nvfp4-spark-tp2", None))


def test_missing_high_in_selected_checkpoint_is_refused(monkeypatch):
    original = thinking.catalog()
    changed = copy.deepcopy(original)
    changed["behaviours"]["glm53-flash-template"]["levels"].remove("high")
    monkeypatch.setattr(thinking, "catalog", lambda *args, **kwargs: changed)
    with pytest.raises(ValueError, match="high is not a level"):
        serving.defaults("glm53-flash-nvfp4-spark-tp2", "csf")


def test_effort_override_removes_profile_off_choice(monkeypatch):
    monkeypatch.setattr(serving, "defaults", lambda *args, **kwargs: {"thinking": "off"})
    values = serving.effective("qwen38-flash-next-tp2", {"reasoning_effort": "low"})
    assert values == {"reasoning_effort": "low"}
    command = ("serve", "--default-chat-template-kwargs", '{"enable_thinking":false,"keep":true}')
    changed = serving.apply(command, values, model=("qwen38-flash-next-tp2", None))
    assert chat_defaults(changed) == {"reasoning_effort": "low", "keep": True}


def test_profile_defaults_require_a_record_on_the_serving_image():
    definition, _ = profiles.load("glm53-flash-nvfp4-spark-tp2")
    configuration = profiles.read_json(profiles.ROOT / definition["configuration"]["path"])
    configuration["image_extension"] = "native"
    with pytest.raises(ValueError, match="no thinking behaviour is recorded"):
        serving.profile_defaults(configuration)


def test_generated_glm_compose_exports_include_high_on_api_rank_only():
    for profile in set(glm_profiles()) & set(compose.SUPPORTED):
        directory = profiles.ROOT / "profiles" / profile / "compose"
        paths = sorted(directory.glob("compose.rank*.yaml"))
        assert paths
        for rank, path in enumerate(paths):
            text = path.read_text(encoding="utf-8")
            assert ('{\"reasoning_effort\":\"high\"}' in text) == (rank == 0)


def test_pinned_glm_sampling_content_and_recorded_hashes():
    # Exact published generation_config.json contents, verified at the pinned
    # revisions. Different Transformers versions and do_sample account for
    # the three distinct hashes; temperature and top_p agree in all three.
    known = set()
    for version in ("5.16.0", "5.16.1", "5.12.0"):
        content = {"_from_model_config": True}
        if version == "5.16.1":
            content["do_sample"] = True
        content.update(eos_token_id=[154820, 154827, 154829], pad_token_id=154820,
                       temperature=1.0, top_p=0.95, transformers_version=version)
        known.add(hashlib.sha256((json.dumps(content, indent=2) + "\n").encode()).hexdigest())
    observed = set()
    for profile in glm_profiles():
        definition, _ = profiles.load(profile)
        configuration = profiles.read_json(profiles.ROOT / definition["configuration"]["path"])
        for checkpoint in configuration.get("checkpoints") or [None]:
            record = thinking.of(profile, checkpoint)
            repository, revision = record["checkpoint"].split("@")
            pin = profiles.read_json(profiles.ROOT / "profiles/checkpoints" / repository.replace("/", "--") / f"{revision}.json")
            digest = pin["files"]["generation_config.json"]["sha256"]
            assert digest in known
            assert record["source"]["generation_configs"][digest] == {"temperature": 1.0, "top_p": 0.95}
            observed.add(digest)
    assert observed == known
