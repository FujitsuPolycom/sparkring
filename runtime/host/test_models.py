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
