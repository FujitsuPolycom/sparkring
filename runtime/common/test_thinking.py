"""Thinking records name each installer checkpoint's behaviour, match its pinned chat template and refuse invalid choices."""
import copy
import json

import pytest

from runtime.common import installer, profiles, qwen_flash_next, thinking


def served_checkpoints():
    """``{(profile, checkpoint name or None): "<repository>@<revision>"}`` for every installer profile's checkpoints."""
    result = {}
    for profile in sorted(installer.INSTALLABLE):
        configuration = profiles.read_json(installer.ROOT / installer.setup.selection(profile)["configuration"])
        names = sorted(configuration.get("checkpoints") or {}) or [None]
        for name in names:
            model = qwen_flash_next.checkpoint_settings(configuration, name)["model"]
            result[profile, name] = f"{model['repository']}@{model['revision']}"
    return result


def pins(key):
    repository, revision = key.split("@")
    return profiles.read_json(installer.ROOT / "profiles/checkpoints" / repository.replace("/", "--") / f"{revision}.json")


def test_every_installer_checkpoint_has_a_record_and_every_record_a_checkpoint():
    served = served_checkpoints()
    for (profile, name), key in served.items():
        record = thinking.of(profile, name)
        assert record is not None and record["checkpoint"] == key, (profile, name)
    # No record names a checkpoint that no installer profile serves.
    assert set(thinking.catalog()["checkpoints"]) == set(served.values())


def test_each_record_reads_the_chat_template_that_the_checkpoint_pins():
    data = thinking.catalog()
    for key, name in data["checkpoints"].items():
        source = data["behaviours"][name]["source"]
        files = pins(key)["files"]
        if source["chat_templates"]:
            assert files["chat_template.jinja"]["sha256"] in source["chat_templates"], key
        else:
            # An encoder replaces the chat template: the checkpoint has none,
            # and its tokenizer configuration is the one that was read.
            assert "chat_template.jinja" not in files, key
            assert files["tokenizer_config.json"]["sha256"] == source["tokenizer_config_sha256"], key


def test_records_state_each_models_default_levels_and_switches():
    qwen = thinking.of("qwen38-flash-next-tp2")
    assert (qwen["default"], qwen["level"], qwen["levels"], qwen["effort"], qwen["off"]) == (
        "on", "xhigh", ["low", "medium", "xhigh"], "reasoning_effort", {"enable_thinking": False})
    glm = thinking.of("glm53-flash-nvfp4-spark-tp4")
    assert (glm["default"], glm["level"], glm["levels"], glm["off"]) == ("always", "max", ["low", "high", "max"], None)
    mimo = thinking.of("mimo-v26-flash-mopd-tp4")
    assert (mimo["default"], mimo["level"], mimo["levels"], mimo["effort"], mimo["off"]) == (
        "on", None, [], None, {"enable_thinking": False})
    deepseek = thinking.of("deepseek-v41-flash-tp4")
    assert (deepseek["default"], deepseek["level"], deepseek["range"], deepseek["off"]) == (
        "on", "high", [1, 100], {"thinking": False})
    assert deepseek["source"]["image"] == "dev-20261001-kraken-cuda1342-nccl2323-status034"
    assert [thinking.summary(record) for record in (qwen, glm, mimo, deepseek, None)] == [
        "on · xhigh", "always · max", "on", "on · high", "not recorded"]
    assert [thinking.levels_text(record) for record in (qwen, glm, mimo, deepseek)] == [
        "low, medium or xhigh", "low, high or max", "", "low, high, xhigh or max, or a whole number from 1 to 100"]
    # Swift serves the Qwen chat template; every checkpoint choice keeps its profile's behaviour.
    assert thinking.of("swift15-qwen38-flash-next-tp2")["name"] == qwen["name"]
    assert thinking.of("qwen38-flash-next-qad-tp4", "qad-step5500-mxfp8-attention")["checkpoint"].startswith(
        "sparkring-derived/Qwen3.8-Flash-Next-NVFP4-QAD5500-MXFP8-Attention@")
    assert thinking.of("qwen38-flash-next-tp2", "qad-step-5500") == qwen
    assert thinking.of("glm53-flash-nvfp4-spark-tp4", "nvidia-nvfp4")["name"] == glm["name"]


def test_profiles_off_the_installer_image_have_no_record():
    # The SparkCache variants run the native image, and the managed GLM profiles their own releases.
    for profile in ("qwen38-flash-next-tp2-sparkcache", "glm53-flash-spark-tp4-dcp1", "deepseek-v41-flash-cycle",
                    "mimo-v26-flash-rl-tp2", "unknown"):
        assert thinking.model(profile) is None and thinking.of(profile) is None


def test_smoke_requests_use_the_recorded_arguments():
    """The installer's smoke request turns thinking off, or asks an always-thinking model for its lowest level."""
    from scripts.installer_host import smoke_request
    for profile in sorted(installer.INSTALLABLE):
        record = thinking.of(profile)
        expected = record["off"] if record["off"] is not None else {record["effort"]: record["levels"][0]}
        assert smoke_request({"profile": profile})["chat_template_kwargs"] == expected, profile


@pytest.mark.parametrize("profile, settings, expected", [
    ("qwen38-flash-next-tp2", {"reasoning_effort": "low"}, {"reasoning_effort": "low"}),
    ("qwen38-flash-next-tp2", {"thinking": "off"}, {"enable_thinking": False}),
    ("swift15-qwen38-flash-next-tp4", {"reasoning_effort": "medium"}, {"reasoning_effort": "medium"}),
    ("glm53-flash-nvfp4-spark-tp2", {"reasoning_effort": "high"}, {"reasoning_effort": "high"}),
    ("mimo-v26-flash-mopd-tp2", {"thinking": "off"}, {"enable_thinking": False}),
    ("deepseek-v41-flash-tp4", {"reasoning_effort": "max"}, {"reasoning_effort": "max"}),
    ("deepseek-v41-flash-tp4", {"reasoning_effort": 60}, {"reasoning_effort": 60}),
    ("deepseek-v41-flash-tp4", {"thinking": "off"}, {"thinking": False}),
    ("qwen38-flash-next-tp2", {}, {}),
])
def test_settings_become_the_models_own_arguments(profile, settings, expected):
    assert thinking.arguments(thinking.of(profile), settings, profile) == expected


@pytest.mark.parametrize("profile, settings, message", [
    ("qwen38-flash-next-tp2", {"reasoning_effort": "max"},
     "--reasoning-effort max is not a level that qwen38-flash-next-tp2 accepts; choose low, medium or xhigh "
     "(default: xhigh)"),
    ("qwen38-flash-next-tp2", {"reasoning_effort": 60}, "choose low, medium or xhigh"),
    ("deepseek-v41-flash-tp4", {"reasoning_effort": 101},
     "--reasoning-effort 101 is not a level that deepseek-v41-flash-tp4 accepts; choose low, high, xhigh or max, "
     "or a whole number from 1 to 100 (default: high)"),
    ("deepseek-v41-flash-tp4", {"reasoning_effort": "medium"}, "is not a level"),
    ("glm53-flash-nvfp4-spark-tp4", {"thinking": "off"},
     "--thinking off does not apply to glm53-flash-nvfp4-spark-tp4: its model always thinks, and its chat template "
     "has no switch that turns thinking off. --reasoning-effort low makes it think least."),
    ("mimo-v26-flash-mopd-tp4", {"reasoning_effort": "low"},
     "--reasoning-effort does not apply to mimo-v26-flash-mopd-tp4: its model has no effort levels; --thinking off "
     "turns its thinking off"),
    ("qwen38-flash-next-tp2", {"reasoning_effort": "low", "thinking": "off"},
     "--reasoning-effort sets how hard the model thinks and --thinking off turns thinking off; choose one"),
])
def test_settings_the_model_does_not_accept_are_refused_with_the_valid_choices(profile, settings, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)").replace(".", r"\.")):
        thinking.arguments(thinking.of(profile), settings, profile)


def test_a_profile_without_a_record_refuses_thinking_settings():
    with pytest.raises(ValueError, match="--thinking off does not apply to qwen38-flash-next-tp2-sparkcache: no thinking "
                                         "behaviour is recorded for its checkpoint on its image"):
        thinking.arguments(None, {"thinking": "off"}, "qwen38-flash-next-tp2-sparkcache")


def test_deployment_default_names_the_setting_and_the_models_default():
    qwen = thinking.of("qwen38-flash-next-tp2")
    plain = thinking.deployment_default(qwen, {"max_images": 2})
    assert plain == {"thinking": "on", "level": "xhigh", "chosen": False, "model": {"thinking": "on", "level": "xhigh"}}
    assert thinking.deployment_text(plain) == "on · xhigh (model default)"
    low = thinking.deployment_default(qwen, {"reasoning_effort": "low"})
    assert thinking.deployment_text(low) == "on · low (deployment default; model default: on · xhigh)"
    off = thinking.deployment_default(thinking.of("mimo-v26-flash-mopd-tp2"), {"thinking": "off"})
    assert thinking.deployment_text(off) == "off (deployment default; model default: on)"
    assert thinking.deployment_default(None, {}) is None
    assert thinking.deployment_text(None) == "not recorded for this model"


def write(root, data):
    (root / "profiles").mkdir(exist_ok=True)
    (root / thinking.CATALOG).write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("change, message", [
    (lambda d: d["behaviours"]["glm53-flash-template"].update(off={"enable_thinking": False}),
     "without an off switch always thinks"),
    (lambda d: d["behaviours"]["qwen38-flash-next-template"].update(off=None), "always thinks"),
    (lambda d: d["behaviours"]["qwen38-flash-next-template"].update(level="max"), "level is one of its levels"),
    (lambda d: d["behaviours"]["mimo-v26-flash-template"].update(effort="reasoning_effort"), "effort names the argument"),
    (lambda d: d["behaviours"]["qwen38-flash-next-template"].update(levels=["Low"]), "lowercase words"),
    (lambda d: d["behaviours"]["deepseek-v41-encoder"].update(range=[0, 100]), "range is the lowest"),
    (lambda d: d["behaviours"]["qwen38-flash-next-template"].update(default="sometimes"), "default is one of"),
    (lambda d: d["behaviours"]["qwen38-flash-next-template"]["source"].update(chat_templates=["c3cf"]),
     "source lists the chat templates"),
    (lambda d: d["behaviours"]["qwen38-flash-next-template"].pop("off"), "expected the fields"),
    (lambda d: d["checkpoints"].update({"owner/name@abc": "qwen38-flash-next-template"}), "full revision"),
    (lambda d: d["checkpoints"].update({"owner/name@" + "0" * 40: "missing"}), "name a behaviour"),
    (lambda d: d.update(schema="sparkring-thinking/v2"), "expected sparkring-thinking/v1"),
])
def test_an_invalid_catalog_is_refused(tmp_path, change, message):
    data = copy.deepcopy(thinking.catalog())
    change(data)
    write(tmp_path, data)
    with pytest.raises(ValueError, match=message):
        thinking.catalog(tmp_path)


def test_research_records_join_the_catalog_and_a_name_in_both_files_is_refused(tmp_path):
    main = profiles.read_json(profiles.ROOT / thinking.CATALOG)
    research = profiles.read_json(profiles.ROOT / thinking.RESEARCH)
    merged = thinking.catalog()
    assert set(merged["checkpoints"]) == set(main["checkpoints"]) | set(research["checkpoints"])
    assert not set(research["checkpoints"]) & set(main["checkpoints"])
    # The CSF checkpoint is in the main file: the GLM-5.3-Flash profiles of two and four Sparks list it.
    csf = "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd33efa73c3bb7c95b74eee10cad34f9be"
    assert main["checkpoints"][csf] == merged["checkpoints"][csf] == "glm53-flash-template"
    # A research checkpoint may name a behaviour of the main file.
    other = "example-owner/Example-Research-Model@" + "1" * 40
    write(tmp_path, main)
    (tmp_path / thinking.RESEARCH).write_text(json.dumps(dict(research, checkpoints={
        **research["checkpoints"], other: "glm53-flash-template"})), encoding="utf-8")
    assert thinking.catalog(tmp_path)["checkpoints"][other] == "glm53-flash-template"
    (tmp_path / thinking.RESEARCH).write_text(json.dumps(dict(research, behaviours={
        **research["behaviours"], "glm53-flash-template": main["behaviours"]["glm53-flash-template"]})),
        encoding="utf-8")
    with pytest.raises(ValueError, match="glm53-flash-template appear in both"):
        thinking.catalog(tmp_path)
