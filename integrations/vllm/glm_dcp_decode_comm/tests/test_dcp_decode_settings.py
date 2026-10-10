"""Settings: the five item flags in every combination, the other settings and the refused ones."""

from __future__ import annotations

import itertools

import pytest

import glm_dcp_decode_comm as pkg
from glm_dcp_decode_comm import runtime

COMBINATIONS = list(itertools.product(("0", "1"), repeat=len(pkg.FLAGS)))


def test_the_plugin_carries_the_five_items():
    assert pkg.FLAGS == {
        "overlap": "GLM_DCP_DECODE_OVERLAP", "wk_overlap": "GLM_DCP_DECODE_WK_OVERLAP",
        "query_pack": "GLM_DCP_DECODE_QUERY_PACK", "selection_reuse": "GLM_DCP_DECODE_SELECTION_REUSE",
        "a2a_fused": "GLM_DCP_DECODE_A2A_FUSED"}


@pytest.mark.parametrize("bits", COMBINATIONS, ids="".join)
def test_every_combination_of_the_item_flags_parses(bits):
    env = dict(zip(pkg.FLAGS.values(), bits))
    values = pkg.settings_from_env(env)
    assert {item: values[item] for item in pkg.FLAGS} == {item: bit == "1" for item, bit in zip(pkg.FLAGS, bits)}
    assert pkg.enabled(values) == ("1" in bits)
    assert values["comm_priority"] == -1 and values["audit"] is False
    settings = runtime.Settings(**{item: values[item] for item in pkg.FLAGS})
    assert settings.query_path == (values["query_pack"] or values["overlap"])
    assert settings.comm_stream == values["overlap"]


@pytest.mark.parametrize("value", ("2", "yes", "true", "-1", "1.0"))
def test_a_malformed_flag_refuses(value):
    with pytest.raises(pkg.PatchRefused, match="must be 0 or 1"):
        pkg.settings_from_env({"GLM_DCP_DECODE_QUERY_PACK": value})


@pytest.mark.parametrize("name", pkg.UNSUPPORTED_SETTINGS)
def test_the_lossy_fp8_settings_are_refused(name):
    assert pkg.settings_from_env({name: "0", "GLM_DCP_DECODE_A2A_FUSED": "1"})["a2a_fused"]
    with pytest.raises(pkg.PatchRefused, match="not carried"):
        pkg.settings_from_env({name: "1" if "TILE" not in name else "128", "GLM_DCP_DECODE_A2A_FUSED": "1"})


def test_audit_needs_an_item_and_keeps_it():
    with pytest.raises(pkg.PatchRefused, match="set at least one item flag"):
        pkg.settings_from_env({"GLM_DCP_DECODE_AUDIT": "1"})
    values = pkg.settings_from_env({"GLM_DCP_DECODE_AUDIT": "1", "GLM_DCP_DECODE_OVERLAP": "1"})
    assert values["audit"] and values["overlap"]


@pytest.mark.parametrize("text,ok", [("-5", True), ("0", True), ("-1", True), ("", True), ("1", False),
                                     ("-6", False), ("high", False)])
def test_the_communication_priority_is_a_whole_number_from_minus_five_to_zero(text, ok):
    env = {"GLM_DCP_DECODE_COMM_PRIORITY": text, "GLM_DCP_DECODE_OVERLAP": "1"}
    if ok:
        assert pkg.settings_from_env(env)["comm_priority"] == int(text or "-1")
    else:
        with pytest.raises(pkg.PatchRefused, match="from -5 to 0"):
            pkg.settings_from_env(env)


def test_registration_with_every_flag_off_patches_nothing(monkeypatch):
    monkeypatch.setattr(pkg, "_REGISTERED", None)
    for name in (*pkg.FLAGS.values(), *pkg.UNSUPPORTED_SETTINGS, pkg.AUDIT):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(pkg, "_package_roots", lambda names: pytest.fail("no tree is read with every flag off"))
    pkg.register()
    assert pkg._REGISTERED is False
    pkg.register()                                       # idempotent
    monkeypatch.setenv("GLM_DCP_DECODE_OVERLAP", "1")
    with pytest.raises(pkg.PatchRefused, match="settings changed after registration"):
        pkg.register()
