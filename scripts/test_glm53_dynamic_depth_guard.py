"""Reject the reported DFlash/DCP4 combination before any remote action."""
import dataclasses
import json
from pathlib import Path

import pytest

import sparkring_generic_launcher as generic
import sparkring_runtime as runtime
from sparkring_site import load_site


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "scripts/config/glm53-flash-tp4-site.example.yaml"
PROFILE = ROOT / "scripts/config/glm53-flash-dflash2-bf16-tp4-dcp1.example.json"
CONTROLS = (
    {"num_speculative_tokens_per_batch_size": [[1, 1, 7], [2, 4, 5], [5, 16, 3]]},
    {"adaptive_speculative_tokens_window": 32, "adaptive_speculative_tokens_initial": 5},
)


def configured(control, *, method="dflash", dcp=4, family="glm53-flash", equals=False):
    site = load_site(SITE)
    site = dataclasses.replace(site, serving=dataclasses.replace(
        site.serving, decode_context_parallel_size=dcp))
    original = runtime.load_runtime_profile(PROFILE)
    config = {"method": method, "num_speculative_tokens": 7, **control}
    arguments = (("--speculative-config=" + json.dumps(config),) if equals
                 else ("--speculative-config", json.dumps(config)))
    profile = dataclasses.replace(original, model_family=family,
                                  extra_vllm_args=arguments)
    return site, profile


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("equals", [False, True])
def test_reported_configs_rejected_by_real_plan_builder(control, equals):
    site, profile = configured(control, equals=equals)
    with pytest.raises(runtime.ProfileError, match="DFlash.*TP4/DCP4"):
        generic.build_actions(site, profile, "plan")


def test_execute_fails_before_remote_actions(monkeypatch, capsys):
    site, profile = configured(CONTROLS[1])
    monkeypatch.setattr(generic, "load_site", lambda *args: site)
    monkeypatch.setattr(generic, "load_profile", lambda *args: profile)
    monkeypatch.setattr(generic, "_require_resolved_inputs", lambda *args: None)
    monkeypatch.setattr(generic, "_validate_site_profile_alignment", lambda *args: None)
    def forbidden(*args):
        pytest.fail("Rejected configuration must not invoke SSH or Docker")
    monkeypatch.setattr(runtime, "execute", forbidden)
    with pytest.raises(SystemExit) as failure:
        generic.main(["start", "--execute", "--site", "site.yaml", "--profile", "profile.json",
                      "--confirmation", profile.confirmation])
    assert failure.value.code == 2
    assert "DFlash" in capsys.readouterr().err


@pytest.mark.parametrize("dcp,method,control", [
    (4, "dflash", {}),
    (4, "dflash", {"num_speculative_tokens_per_batch_size": None,
                   "adaptive_speculative_tokens_window": None}),
    (4, "mtp", {"num_speculative_tokens": 3}),
    (4, "mtp", CONTROLS[1]),
    (1, "dflash", CONTROLS[0]),
    (2, "dflash", CONTROLS[1]),
])
def test_fixed_mtp_and_other_dcp_configs_are_preserved(dcp, method, control):
    site, profile = configured(control, dcp=dcp, method=method)
    assert len(generic.build_actions(site, profile, "plan")) == 4


def test_unrelated_family_is_not_classified_by_profile_name():
    site, profile = configured(CONTROLS[0], family="deepseek")
    assert "glm53" in profile.profile_id
    assert len(generic.build_actions(site, profile, "plan")) == 4


@pytest.mark.parametrize("arguments,rejected", [
    (("-sc", json.dumps({"method": "dflash", **CONTROLS[0]})), True),
    (("-sc=" + json.dumps({"method": "dflash", **CONTROLS[1]}),), True),
    (("-sc.method=dflash", "-sc.adaptive_speculative_tokens_window=32"), True),
    (("--speculative_config=" + json.dumps({"method": "dflash", **CONTROLS[1]}),), True),
    (("--speculative-config.method", "dflash",
      "--speculative-config.adaptive_speculative_tokens_window", "32"), True),
    (("--speculative_config.method=dflash",
      "--speculative_config.num_speculative_tokens_per_batch_size=[[1,1,7]]"), True),
    (("--speculative-config.method=dflash",
      "--speculative-config.num_speculative_tokens_per_batch_size+=1,1,7"), True),
    (("--speculative-config", json.dumps({"method": "dflash", **CONTROLS[1]}),
      "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}'), False),
    (("--speculative-config", '{"method":"mtp","num_speculative_tokens":3}',
      "--speculative-config", json.dumps({"method": "dflash", **CONTROLS[1]})), True),
    # vLLM appends the dotted dictionary after ordinary JSON, replacing it.
    (("--speculative-config", json.dumps({"method": "dflash", **CONTROLS[1]}),
      "--speculative-config.method", "dflash"), False),
    (("--speculative-config.method", "mtp",
      "--speculative-config", json.dumps({"method": "dflash", **CONTROLS[1]})), False),
    (("--speculative-config", '{"method":"mtp","num_speculative_tokens":3}',
      "--speculative-config.method=dflash",
      "--speculative-config.adaptive_speculative_tokens_window=32"), True),
    (("--speculative-config.method=dflash", "--speculative-config.method=mtp",
      "--speculative-config.adaptive_speculative_tokens_window=32"), False),
    (("--speculative-config.method=mtp", "--speculative-config.method=dflash",
      "--speculative-config.adaptive_speculative_tokens_window=32"), True),
    (("--speculative-config.method=dflash",
      "--speculative-config.adaptive_speculative_tokens_window=32",
      "--speculative-config.adaptive_speculative_tokens_window=null"), False),
    (("--speculative-config.method=dflash",
      "--speculative-config.adaptive_speculative_tokens_window=null",
      "--speculative-config.adaptive_speculative_tokens_window=32"), True),
    # Dotted long/short spellings are separate groups in vLLM's parser.
    (("--speculative-config.method=mtp", "-sc.method=dflash",
      "--speculative-config.adaptive_speculative_tokens_window=32"), False),
    (("-sc.method=mtp", "--speculative-config.method=dflash",
      "--speculative-config.adaptive_speculative_tokens_window=32"), True),
    (("--speculative-config", "null"), False),
    (("--hf-overrides", json.dumps({"method": "dflash", **CONTROLS[1]})), False),
])
def test_guard_obeys_speculative_option_precedence(arguments, rejected):
    site, profile = configured({})
    profile = dataclasses.replace(profile, extra_vllm_args=arguments)
    if rejected:
        with pytest.raises(runtime.ProfileError, match="DFlash.*TP4/DCP4"):
            generic.build_actions(site, profile, "plan")
    else:
        assert len(generic.build_actions(site, profile, "plan")) == 4


@pytest.mark.parametrize("arguments", [
    ["-tp", "4"], ["-tp=4"], ["-dcp", "4"], ["-dcp=4"],
    ["--tensor_parallel_size", "4"], ["--decode_context_parallel_size=4"],
])
def test_profile_cannot_override_site_dimensions_through_vllm_aliases(arguments):
    document = json.loads(PROFILE.read_text())
    document["extra_vllm_args"] = arguments
    with pytest.raises(runtime.ProfileError, match="site-owned option"):
        runtime.parse_runtime_profile(document)


def test_non_option_values_are_preserved():
    document = json.loads(PROFILE.read_text())
    arguments = ["--served-model-name", "glm53_dflash_test",
                 "--hf-overrides", '{"some_key":"--tensor_parallel_size=4"}']
    document["extra_vllm_args"] = arguments
    assert runtime.parse_runtime_profile(document).extra_vllm_args == tuple(arguments)
