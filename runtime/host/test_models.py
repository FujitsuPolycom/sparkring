"""`sparkring models` names each installer profile's thinking default and levels."""
import json

from runtime.host import models


def test_models_lists_each_installer_profiles_thinking_default(capsys):
    assert models.main([]) == 0
    lines = capsys.readouterr().out.splitlines()

    def thinking(profile):
        at = lines.index(f"{profile}  [installer]")
        return lines[at + 2]
    assert thinking("qwen38-flash-next-tp2") == "  Thinking: on · xhigh (levels: low, medium or xhigh)"
    assert thinking("swift15-qwen38-flash-next-tp4") == "  Thinking: on · xhigh (levels: low, medium or xhigh)"
    assert thinking("glm53-flash-nvfp4-spark-tp2") == "  Thinking: always · max (levels: low, high or max)"
    assert thinking("mimo-v26-flash-mopd-tp4") == "  Thinking: on (no levels)"
    assert thinking("deepseek-v41-flash-tp4") == ("  Thinking: on · high (levels: low, high, xhigh or max, "
                                                  "or a whole number from 1 to 100)")
    # A guide profile has no record and no Thinking line.
    at = lines.index("qwen38-flash-next-tp2-sparkcache  [guide]")
    assert not lines[at + 2].startswith("  Thinking:")


def test_models_json_carries_the_thinking_record_or_null(capsys):
    assert models.main(["--json"]) == 0
    rows = {row["profile"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["mimo-v26-flash-mopd-tp2"]["thinking"] == {"default": "on", "level": None, "levels": [], "effort": None,
                                                           "off": {"enable_thinking": False}}
    assert rows["deepseek-v41-flash-tp4"]["thinking"]["range"] == [1, 100]
    assert all(row["thinking"] is not None for row in rows.values() if row["automated"])
    assert rows["glm53-flash-spark-tp4-dcp1"]["thinking"] is None


def test_each_alias_names_an_installer_profile_and_no_catalog_id():
    from runtime.common import installer, profiles
    catalog = profiles.catalog()
    for alias, ident in models.ALIASES.items():
        assert ident in installer.INSTALLABLE
        assert alias not in catalog and alias not in profiles.REPLACED


def test_an_alias_selects_its_profile_id():
    assert models.select("glm53-flash-tp4", 4) == "glm53-flash-nvfp4-spark-tp4"
    assert models.select("glm53-flash-tp2", 2) == "glm53-flash-nvfp4-spark-tp2"
    assert models.canonical("qwen38-flash-next-tp2") == "qwen38-flash-next-tp2"
    import pytest
    with pytest.raises(ValueError, match="requires 2 Sparks"):
        models.select("glm53-flash-tp2", 4)


def test_models_shows_the_checkpoint_each_installation_selects_and_the_aliases(capsys):
    assert models.main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    at = lines.index("glm53-flash-nvfp4-spark-tp4  [installer]")
    assert lines[at + 3] == ("  Checkpoint: csf where the image's vLLM reads it, else nvfp4-spark; "
                             "--checkpoint also takes nvfp4-qad or nvidia-nvfp4")
    assert lines[at + 4] == "  Also selected as: glm53-flash-tp4"
    at = lines.index("glm53-flash-nvfp4-spark-tp2  [installer]")
    assert lines[at + 3].startswith("  Checkpoint: csf where the image's vLLM reads it, else nvfp4-spark")
    assert lines[at + 4] == "  Also selected as: glm53-flash-tp2"
    # A profile without a checkpoints table shows no Checkpoint line.
    at = lines.index("mimo-v26-flash-mopd-tp2  [installer]")
    end = next(number for number in range(at + 1, len(lines)) if not lines[number].startswith("  "))
    assert not any(line.startswith("  Checkpoint:") for line in lines[at:end])


def test_models_json_names_the_checkpoints_and_aliases(capsys):
    assert models.main(["--json"]) == 0
    rows = {row["profile"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["glm53-flash-nvfp4-spark-tp4"]["checkpoints"] == {
        "default": "nvfp4-spark", "preferred": "csf", "others": ["nvfp4-qad", "nvidia-nvfp4"]}
    assert rows["glm53-flash-nvfp4-spark-tp4"]["aliases"] == ["glm53-flash-tp4"]
    assert rows["mimo-v26-flash-mopd-tp2"]["checkpoints"] is None and rows["mimo-v26-flash-mopd-tp2"]["aliases"] == []
    assert rows["glm53-flash-spark-tp4-dcp1"]["checkpoints"] is None
