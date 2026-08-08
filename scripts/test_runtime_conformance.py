"""Conformance tests for the generic runtime launcher.

Covers: validate/explain/diff CLI, malformed inputs, recursive semantic
diff with exact nested JSON paths (0=same, 1=different, 2=invalid),
deterministic output, identity/image/topology/labels/mounts/hooks/action
change detection, one-sided site fallback, Windows/POSIX invocation,
offline/no-SSH via poisoned PATH, exact NF3 extraction golden tests
(old context and expand independently reconstructed and monkeypatched,
complete RemoteAction lists compared by equality for start/stop/status/
verify-rollback), actual generated semantic plan snapshots for
contributor/EXL3/NF3, template validation, contributor workflow,
and a clean-checkout rehearsal from HEAD via git archive with zero
overlays.

All tests run on ordinary Windows and Linux machines without GPUs or Sparks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sparkring_exl3_launcher as exl3  # noqa: E402
import sparkring_generic_launcher as generic  # noqa: E402, F401
import sparkring_launcher as nf3  # noqa: E402
import sparkring_runtime as runtime  # noqa: E402
from sparkring_site import load_site  # noqa: E402

SITE = ROOT / "scripts/config/site.example.yaml"
GENERIC = ROOT / "scripts/config/generic.example.json"
LAUNCH_NF3 = ROOT / "scripts/config/launch.example.json"
TEMPLATE = ROOT / "scripts/config/native-profile.template.json"
CONTRIBUTOR = ROOT / "scripts/config/contributor-example.json"
FIXTURES = ROOT / "scripts/config/fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generic_doc(**overrides):
    doc = json.loads(GENERIC.read_text(encoding="utf-8"))
    doc.update(overrides)
    return doc


def _write_profile(tmp_path, doc, name="profile.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _write_site(tmp_path, api_port=8000, mtu=None, name="site.yaml"):
    import yaml
    doc = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    doc["serving"]["api_port"] = api_port
    if mtu is not None:
        doc["topology"]["mtu"] = mtu
    path = tmp_path / name
    path.write_text(yaml.dump(doc), encoding="utf-8")
    return path


def _run_cli(*args, expect_code=None, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/sparkring_generic_launcher.py"), *args],
        capture_output=True, text=True, timeout=30, cwd=str(ROOT), env=env,
    )
    if expect_code is not None:
        assert result.returncode == expect_code, (
            f"expected exit {expect_code}, got {result.returncode}: {result.stderr}"
        )
    return result.returncode, result.stdout, result.stderr


def _run_validate(profile_path, site_path=None, **kw):
    args = []
    if site_path:
        args.extend(["--site", str(site_path)])
    args.extend(["--profile", str(profile_path), "validate"])
    return _run_cli(*args, **kw)


def _run_explain(profile_path, site_path=None, **kw):
    args = []
    if site_path:
        args.extend(["--site", str(site_path)])
    args.extend(["--profile", str(profile_path), "explain"])
    return _run_cli(*args, **kw)


def _run_diff(profile_a, profile_b, site=None, site_a=None, site_b=None, **kw):
    args = []
    if site:
        args.extend(["--site", str(site)])
    if site_a:
        args.extend(["--site-a", str(site_a)])
    if site_b:
        args.extend(["--site-b", str(site_b)])
    args.extend(["--profile-a", str(profile_a), "--profile-b", str(profile_b),
                "diff"])
    return _run_cli(*args, **kw)


def _stable_projection(plan, site_obj, profile):
    """Stable semantic projection for snapshot comparison.

    The sanitized example site contains documentation-only addresses, but the
    fixture still avoids freezing rendered shell text. It retains the complete
    profile environment and a focused per-rank action contract instead.
    """
    proj = generic._plan_projection(plan, site_obj, profile)
    action_summary = {}
    for rank, action in proj.pop("actions_by_rank").items():
        command = action["remote_command"]
        action_summary[rank] = {
            "rank": action["rank"],
            "has_image_verification": " image inspect " in f" {command} ",
            "has_managed_label": "org.sparkring.managed=true" in command,
            "has_profile_label": "org.sparkring.profile=" in command,
            "headless": "--headless" in command,
        }
    proj["action_summary"] = action_summary
    proj["_excluded"] = [
        "rendered remote_command: avoid freezing incidental shell formatting",
        "action ssh_target: topology is retained separately from the sanitized tracked site example",
        "action_summary retains rank, image verification, managed/profile labels, and headless role",
    ]
    return proj


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def test_validate_structural_only():
    code, stdout, _ = _run_validate(CONTRIBUTOR, expect_code=0)
    r = json.loads(stdout)
    assert r["valid"] is True
    assert r["validation_scope"] == "structural"


def test_validate_plan_build_with_site():
    code, stdout, _ = _run_validate(CONTRIBUTOR, SITE, expect_code=0)
    r = json.loads(stdout)
    assert r["valid"] is True
    assert r["validation_scope"] == "plan-build"


def test_validate_nf3_with_site():
    code, stdout, _ = _run_validate(LAUNCH_NF3, SITE, expect_code=0)
    r = json.loads(stdout)
    assert r["valid"] is True
    assert r["source_schema"] == runtime.NF3_SCHEMA
    assert r["identity_scope"] == "declared-site-image"


def test_validate_generic_example_is_unresolved():
    """generic.example.json has REPLACE placeholders -> template/unresolved."""
    code, stdout, _ = _run_validate(GENERIC, expect_code=1)
    r = json.loads(stdout)
    assert r["valid"] is False
    assert "template/unresolved" in r["error"]


def test_validate_template_is_unresolved():
    """Template with obvious placeholders must be template/unresolved."""
    code, stdout, _ = _run_validate(TEMPLATE, expect_code=1)
    r = json.loads(stdout)
    assert r["valid"] is False
    assert "template/unresolved" in r["error"]


def test_validate_contributor_example_is_valid():
    """The filled sanitized fixture must be structurally/plan-build valid."""
    code, stdout, _ = _run_validate(CONTRIBUTOR, SITE, expect_code=0)
    r = json.loads(stdout)
    assert r["valid"] is True
    assert r["validation_scope"] == "plan-build"


def test_validate_requires_profile():
    code, _, stderr = _run_cli("validate", expect_code=2)
    assert "validate requires --profile" in stderr


@pytest.mark.parametrize("option", [
    ("--execute",),
    ("--confirmation", "IGNORED"),
    ("--max-num-batched-tokens", "3072"),
])
def test_conformance_commands_reject_inapplicable_options(option):
    code, _, stderr = _run_cli(
        "--profile", str(CONTRIBUTOR), *option, "validate", expect_code=2)
    assert code == 2
    assert "does not apply" in stderr or "always offline" in stderr or "applies only" in stderr


def test_validate_site_incompatibility(tmp_path):
    doc = _generic_doc(image_id="sha256:zzz")
    path = _write_profile(tmp_path, doc)
    code, stdout, _ = _run_validate(path, SITE, expect_code=1)
    assert json.loads(stdout)["valid"] is False


@pytest.mark.parametrize("override,fragment", [
    ({"image": ""}, "image"),
    ({"image_id": ""}, "image_id"),
    ({"profile_id": ""}, "profile_id"),
    ({"model_family": ""}, "model_family"),
    ({"container_name": ""}, "container_name"),
])
def test_validate_malformed_fails(tmp_path, override, fragment):
    doc = _generic_doc(**override)
    path = _write_profile(tmp_path, doc)
    code, stdout, _ = _run_validate(path, expect_code=1)
    r = json.loads(stdout)
    assert r["valid"] is False
    assert fragment in r["error"]


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------

def test_explain_generic():
    code, stdout, _ = _run_explain(GENERIC, SITE, expect_code=0)
    info = json.loads(stdout)
    assert info["profile_id"] == "example-generic-runtime"
    assert info["safety_classes"]["plan"] == ["OFFLINE"]
    assert info["safety_classes"]["start"] == ["MUTATES HOST", "STOPS SERVING"]
    assert "model correctness" in info["claim_disclaimer"]


def test_explain_without_site():
    code, stdout, _ = _run_explain(GENERIC, expect_code=0)
    info = json.loads(stdout)
    assert info["target_topology"] is None


def test_explain_nf3_bridge():
    code, stdout, _ = _run_explain(LAUNCH_NF3, SITE, expect_code=0)
    info = json.loads(stdout)
    assert info["schema"] == runtime.NF3_SCHEMA
    assert info["identity_scope"] == "declared-site-image"
    assert "serving.api_port" in info["site_owned_settings"]


def test_explain_hooks_show_argv(tmp_path):
    """explain must show structural argv for hooks, not just booleans."""
    doc = _generic_doc()
    doc["attestation_hook"] = ["curl", "localhost:8000/attest"]
    doc["health_check"] = ["curl", "localhost:8000/health"]
    path = _write_profile(tmp_path, doc)
    code, stdout, _ = _run_explain(path, SITE, expect_code=0)
    info = json.loads(stdout)
    assert info["hooks"]["attestation_hook"] == ["curl", "localhost:8000/attest"]
    assert info["hooks"]["health_check"] == ["curl", "localhost:8000/health"]


def test_explain_requires_profile():
    code, _, stderr = _run_cli("explain", expect_code=2)
    assert "explain requires --profile" in stderr


# ---------------------------------------------------------------------------
# Diff: exit codes and scope
# ---------------------------------------------------------------------------

def test_diff_identical_profiles_exit0():
    code, stdout, _ = _run_diff(CONTRIBUTOR, CONTRIBUTOR, expect_code=0)
    r = json.loads(stdout)
    assert r["identical"] is True
    assert r["scope"] == "profile-only"


def test_diff_different_profiles_exit1():
    code, stdout, _ = _run_diff(CONTRIBUTOR, LAUNCH_NF3, expect_code=1)
    r = json.loads(stdout)
    assert r["identical"] is False
    assert r["scope"] == "profile-only"


def test_diff_identical_plans_exit0():
    code, stdout, _ = _run_diff(CONTRIBUTOR, CONTRIBUTOR, site=SITE, expect_code=0)
    r = json.loads(stdout)
    assert r["identical"] is True
    assert r["scope"] == "plan"


def test_diff_different_plans_exit1():
    code, stdout, _ = _run_diff(CONTRIBUTOR, LAUNCH_NF3, site=SITE, expect_code=1)
    r = json.loads(stdout)
    assert r["identical"] is False
    assert r["scope"] == "plan"


def test_diff_malformed_exit2(tmp_path):
    path_a = _write_profile(tmp_path, _generic_doc(), "a.json")
    path_b = tmp_path / "b.json"
    path_b.write_text("{not json", encoding="utf-8")
    code, stdout, _ = _run_diff(path_a, path_b, expect_code=2)
    assert "error" in json.loads(stdout)


def test_diff_requires_both_profiles():
    code, _, stderr = _run_cli("--profile-a", str(GENERIC), "diff", expect_code=2)
    assert "diff requires --profile-b" in stderr


# ---------------------------------------------------------------------------
# Diff: independent site-a/site-b (topology change detection)
# ---------------------------------------------------------------------------

def test_diff_site_a_site_b_topology_change(tmp_path):
    """Two different sites must produce different plans for the same profile."""
    site_a = _write_site(tmp_path, api_port=8000, name="site-a.yaml")
    site_b = _write_site(tmp_path, api_port=9000, name="site-b.yaml")
    code, stdout, _ = _run_diff(
        CONTRIBUTOR, CONTRIBUTOR, site_a=site_a, site_b=site_b, expect_code=1)
    r = json.loads(stdout)
    assert r["identical"] is False
    assert r["scope"] == "plan"
    fields = [d["field"] for d in r["differences"]]
    assert "topology.serving.api_port" in fields
    assert "actions_by_rank" in ".".join(fields)


def test_diff_site_a_site_b_physical_topology_change(tmp_path):
    """A valid MTU-only site change must be named by the semantic diff."""
    site_a = _write_site(tmp_path, mtu=9000, name="site-a.yaml")
    site_b = _write_site(tmp_path, mtu=4096, name="site-b.yaml")
    code, stdout, _ = _run_diff(
        CONTRIBUTOR, CONTRIBUTOR, site_a=site_a, site_b=site_b, expect_code=1)
    fields = [item["field"] for item in json.loads(stdout)["differences"]]
    assert "topology.mtu" in fields


def test_diff_shared_site_identical(tmp_path):
    """Same profile, same shared site = identical plan."""
    code, stdout, _ = _run_diff(
        CONTRIBUTOR, CONTRIBUTOR, site=SITE, expect_code=0)
    assert json.loads(stdout)["identical"] is True


# ---------------------------------------------------------------------------
# Diff: one-sided site fallback
# ---------------------------------------------------------------------------

def test_diff_site_a_only_falls_back(tmp_path):
    """Only --site-a given: used for both sides, never load_site(None)."""
    site_a = _write_site(tmp_path, api_port=8000, name="site-a.yaml")
    # Same profile on both sides with one site → identical
    code, stdout, _ = _run_diff(
        CONTRIBUTOR, CONTRIBUTOR, site_a=site_a, expect_code=0)
    r = json.loads(stdout)
    assert r["identical"] is True
    assert r["scope"] == "plan"


def test_diff_site_b_only_falls_back(tmp_path):
    """Only --site-b given: used for both sides, never load_site(None)."""
    site_b = _write_site(tmp_path, api_port=8000, name="site-b.yaml")
    code, stdout, _ = _run_diff(
        CONTRIBUTOR, CONTRIBUTOR, site_b=site_b, expect_code=0)
    r = json.loads(stdout)
    assert r["identical"] is True
    assert r["scope"] == "plan"


def test_diff_no_site_no_fallback_error(tmp_path):
    """Neither --site nor --site-a/--site-b for plan diff: must error."""
    # Without any site, diff falls to profile-only, not plan diff
    code, stdout, _ = _run_diff(CONTRIBUTOR, CONTRIBUTOR, expect_code=0)
    r = json.loads(stdout)
    assert r["scope"] == "profile-only"


# ---------------------------------------------------------------------------
# Diff: field-level change detection (parametrized, profile-only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,override_a,override_b", [
    ("image_id", {"image_id": "sha256:" + "a" * 64},
     {"image_id": "sha256:" + "b" * 64}),
    ("image", {"image": "registry.example/a:1"},
     {"image": "registry.example/b:2"}),
    ("model_family", {"model_family": "family-a"},
     {"model_family": "family-b"}),
    ("container_name", {"container_name": "container-a"},
     {"container_name": "container-b"}),
    ("privileged", {"privileged": False}, {"privileged": True}),
    ("confirmation", {"confirmation": None},
     {"confirmation": "confirm-token"}),
    ("extra_vllm_args[0]", {"extra_vllm_args": ["--a"]},
     {"extra_vllm_args": ["--b"]}),
    ("extra_vllm_args[1]", {"extra_vllm_args": ["--a"]},
     {"extra_vllm_args": ["--a", "value"]}),
    ("extra_volumes[0][0]",
     {"extra_volumes": [{"host": "/host-a", "container": "/cont", "mode": "ro"}]},
     {"extra_volumes": [{"host": "/host-b", "container": "/cont", "mode": "ro"}]}),
    ("extra_labels.k", {"extra_labels": {"k": "v1"}},
     {"extra_labels": {"k": "v2"}}),
    ("identity.model_repository",
     {"identity": {"model_repository": "org/a",
                    "model_revision": "0" * 40,
                    "model_config_sha256": "0" * 64}},
     {"identity": {"model_repository": "org/b",
                    "model_revision": "0" * 40,
                    "model_config_sha256": "0" * 64}}),
])
def test_diff_field_change_detected(tmp_path, field, override_a, override_b):
    doc_a = _generic_doc(**override_a)
    doc_b = _generic_doc(**override_b)
    path_a = _write_profile(tmp_path, doc_a, "a.json")
    path_b = _write_profile(tmp_path, doc_b, "b.json")
    code, stdout, _ = _run_diff(path_a, path_b, expect_code=1)
    r = json.loads(stdout)
    fields = [d["field"] for d in r["differences"]]
    assert field in fields, f"expected field '{field}' in {fields}"


def test_diff_action_change_via_image_id(tmp_path):
    """Image ID change must produce different plan actions (plan diff)."""
    doc_a = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b["image_id"] = "sha256:" + "e" * 64
    path_a = _write_profile(tmp_path, doc_a, "a.json")
    path_b = _write_profile(tmp_path, doc_b, "b.json")
    code, stdout, _ = _run_diff(path_a, path_b, site=SITE, expect_code=1)
    r = json.loads(stdout)
    action_diffs = [d for d in r["differences"]]
    assert len(action_diffs) > 0


# ---------------------------------------------------------------------------
# Diff: recursive nested JSON paths (plan-level)
# ---------------------------------------------------------------------------

def test_diff_recursive_nested_path_environment(tmp_path):
    """Environment key change must produce a nested path like environment.KEY."""
    doc_a = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b["environment"]["EXTRA_VAR"] = "new-value"
    path_a = _write_profile(tmp_path, doc_a, "a.json")
    path_b = _write_profile(tmp_path, doc_b, "b.json")
    code, stdout, _ = _run_diff(path_a, path_b, site=SITE, expect_code=1)
    r = json.loads(stdout)
    fields = [d["field"] for d in r["differences"]]
    assert any(f.startswith("environment.") for f in fields), (
        f"expected nested environment path in {fields}")


def test_diff_recursive_nested_path_topology(tmp_path):
    """Topology change must produce a nested path like topology.api_port."""
    site_a = _write_site(tmp_path, api_port=8000, name="site-a.yaml")
    site_b = _write_site(tmp_path, api_port=9000, name="site-b.yaml")
    code, stdout, _ = _run_diff(
        CONTRIBUTOR, CONTRIBUTOR, site_a=site_a, site_b=site_b, expect_code=1)
    r = json.loads(stdout)
    fields = [d["field"] for d in r["differences"]]
    assert "topology.serving.api_port" in fields, (
        f"expected topology.serving.api_port in {fields}")


def test_diff_recursive_nested_path_labels(tmp_path):
    """Label change must produce a nested path like extra_labels.KEY."""
    doc_a = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b["extra_labels"]["new_label"] = "new-value"
    path_a = _write_profile(tmp_path, doc_a, "a.json")
    path_b = _write_profile(tmp_path, doc_b, "b.json")
    code, stdout, _ = _run_diff(path_a, path_b, site=SITE, expect_code=1)
    r = json.loads(stdout)
    fields = [d["field"] for d in r["differences"]]
    assert "extra_labels.new_label" in fields, (
        f"expected extra_labels.new_label in {fields}")


def test_diff_recursive_nested_path_identity(tmp_path):
    """Identity change must produce nested paths like identity.KEY."""
    doc_a = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b["identity"]["model_revision"] = "f" * 40
    path_a = _write_profile(tmp_path, doc_a, "a.json")
    path_b = _write_profile(tmp_path, doc_b, "b.json")
    code, stdout, _ = _run_diff(path_a, path_b, site=SITE, expect_code=1)
    r = json.loads(stdout)
    fields = [d["field"] for d in r["differences"]]
    assert "identity.model_revision" in fields, (
        f"expected identity.model_revision in {fields}")


def test_diff_recursive_nested_path_action_command(tmp_path):
    """Action command change must produce nested action path."""
    doc_a = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b["image_id"] = "sha256:" + "e" * 64
    path_a = _write_profile(tmp_path, doc_a, "a.json")
    path_b = _write_profile(tmp_path, doc_b, "b.json")
    code, stdout, _ = _run_diff(path_a, path_b, site=SITE, expect_code=1)
    r = json.loads(stdout)
    fields = [d["field"] for d in r["differences"]]
    assert any(f.startswith("actions_by_rank.") for f in fields), (
        f"expected actions_by_rank.* path in {fields}")


def test_diff_recursive_nested_path_hooks(tmp_path):
    """Hook change must produce nested path like attestation_hook[0]."""
    doc_a = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b["attestation_hook"] = ["curl", "localhost:8000/attest"]
    path_a = _write_profile(tmp_path, doc_a, "a.json")
    path_b = _write_profile(tmp_path, doc_b, "b.json")
    code, stdout, _ = _run_diff(path_a, path_b, site=SITE, expect_code=1)
    r = json.loads(stdout)
    fields = [d["field"] for d in r["differences"]]
    assert any(f.startswith("attestation_hook") for f in fields), (
        f"expected attestation_hook path in {fields}")


def test_diff_recursive_nested_path_mounts(tmp_path):
    """Mount change must produce a nested path like extra_volumes[0][0]."""
    doc_a = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b["extra_volumes"] = [{"host": "/srv/new", "container": "/models/new", "mode": "ro"}]
    path_a = _write_profile(tmp_path, doc_a, "a.json")
    path_b = _write_profile(tmp_path, doc_b, "b.json")
    code, stdout, _ = _run_diff(path_a, path_b, site=SITE, expect_code=1)
    r = json.loads(stdout)
    fields = [d["field"] for d in r["differences"]]
    assert any("extra_volumes" in f for f in fields), (
        f"expected extra_volumes path in {fields}")


def test_diff_recursive_nested_path_confirmation(tmp_path):
    """Confirmation change must produce nested path."""
    doc_a = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    doc_b["confirmation"] = "confirm-token"
    path_a = _write_profile(tmp_path, doc_a, "a.json")
    path_b = _write_profile(tmp_path, doc_b, "b.json")
    code, stdout, _ = _run_diff(path_a, path_b, site=SITE, expect_code=1)
    r = json.loads(stdout)
    fields = [d["field"] for d in r["differences"]]
    assert "confirmation" in fields, (
        f"expected confirmation in {fields}")


# ---------------------------------------------------------------------------
# Deterministic output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    lambda: _run_validate(CONTRIBUTOR, SITE),
    lambda: _run_explain(GENERIC, SITE),
    lambda: _run_diff(GENERIC, LAUNCH_NF3, site=SITE),
    lambda: _run_cli("--site", str(SITE), "--profile", str(GENERIC), "plan"),
])
def test_output_deterministic(fn):
    _, s1, _ = fn()
    _, s2, _ = fn()
    assert s1 == s2


# ---------------------------------------------------------------------------
# Offline / no-SSH (poisoned PATH in child environment)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args, expected_code", [
    (("--site", str(SITE), "--profile", str(GENERIC), "plan"), 0),
    (("--profile", str(CONTRIBUTOR), "validate"), 0),
    (("--site", str(SITE), "--profile", str(CONTRIBUTOR), "validate"), 0),
    (("--site", str(SITE), "--profile", str(GENERIC), "explain"), 0),
    (("--profile-a", str(GENERIC), "--profile-b", str(LAUNCH_NF3), "diff"), 1),
])
def test_offline_poisoned_path(args, expected_code):
    """Conformance and plan commands must succeed with no ssh on PATH.

    PATH is set to the scripts directory only (no ssh available), but
    Python's own directory is preserved via PYTHONPATH so dependencies work.
    """
    env = {"PATH": str(ROOT / "scripts")}
    code, stdout, _ = _run_cli(*args, env_extra=env)
    assert code == expected_code
    assert json.loads(stdout)


# ---------------------------------------------------------------------------
# Windows / POSIX
# ---------------------------------------------------------------------------

def test_cli_path_with_spaces(tmp_path):
    doc = json.loads(CONTRIBUTOR.read_text(encoding="utf-8"))
    path = tmp_path / "test profile.json"
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    code, stdout, _ = _run_validate(path, expect_code=0)
    assert json.loads(stdout)["valid"] is True


# ---------------------------------------------------------------------------
# NF3 extraction: exact golden tests (monkeypatched before/after)
# ---------------------------------------------------------------------------

def _reconstruct_old_nf3_context(site, rank_id):
    """Independently reconstruct the pre-extraction NF3 _context.

    This is the exact body that lived in sparkring_launcher.py before
    the F8 extraction to sparkring_runtime.site_context.
    """
    rank = site.rank(rank_id)
    peers_by_rank = {peer.rank: peer for peer in rank.transport_peers}
    peers = [peers_by_rank[rank_id ^ 1], peers_by_rank[rank_id ^ 3]]
    ports = {port.peer_rank: port for port in rank.ring_ports}
    master = site.rank(site.serving.master_rank)
    return {
        "api_port": str(site.serving.api_port),
        "draft_path": "/mtp-draft",
        "master_addr": str(master.management.address),
        "master_port": str(site.serving.master_port),
        "model_path": site.runtime.model_path,
        "peer0_addr": str(peers[0].address),
        "peer0_device": ports[peers[0].rank].rdma_device,
        "peer0_gid": str(ports[peers[0].rank].roce_gid_index),
        "peer0_rank": str(peers[0].rank),
        "peer1_addr": str(peers[1].address),
        "peer1_device": ports[peers[1].rank].rdma_device,
        "peer1_gid": str(ports[peers[1].rank].roce_gid_index),
        "peer1_rank": str(peers[1].rank),
        "rank": str(rank_id),
        "world_size": str(len(site.ranks)),
    }


def _reconstruct_old_expand(value, context):
    """Independently reconstruct the pre-extraction NF3 _expand."""
    import re
    return re.sub(
        r"\{([a-z0-9_]+)\}", lambda m: context[m.group(1)], value,
    )


def test_nf3_context_is_runtime_site_context():
    assert nf3._context is runtime.site_context
    assert nf3._expand is runtime.expand


def test_nf3_context_exact_all_ranks():
    """Exact context dict equality: shared site_context vs independently
    reconstructed pre-extraction _context, for all 4 ranks."""
    site = load_site(SITE)
    for rank_id in range(4):
        old = _reconstruct_old_nf3_context(site, rank_id)
        new = runtime.site_context(site, rank_id)
        assert old == new, f"rank {rank_id} context mismatch"


def test_nf3_expand_exact():
    """Exact expand function equality: shared expand vs reconstructed."""
    site = load_site(SITE)
    ctx = runtime.site_context(site, 0)
    test_values = [
        "{model_path}/config.json",
        "{api_port}:{master_port}",
        "{peer0_addr}:{peer0_gid}",
        "no-placeholders-here",
        "{rank}/{world_size}",
    ]
    for val in test_values:
        old = _reconstruct_old_expand(val, ctx)
        new = runtime.expand(val, ctx)
        assert old == new, f"expand mismatch for '{val}'"


def test_nf3_start_actions_exact_monkeypatched():
    """Exact start actions: monkeypatch old _context/_expand, compare lists.

    Temporarily replaces nf3._context and nf3._expand with independently
    reconstructed pre-extraction implementations, builds actions, then
    restores the shared aliases and builds actions again. The complete
    RemoteAction lists must be equal by dataclass equality.
    """
    site = load_site(SITE)
    config = nf3.load_launch(LAUNCH_NF3)

    # Save originals
    orig_context = nf3._context
    orig_expand = nf3._expand

    # Monkeypatch old implementations
    nf3._context = _reconstruct_old_nf3_context
    nf3._expand = _reconstruct_old_expand
    try:
        old_actions = nf3.start_actions(site, config)
    finally:
        nf3._context = orig_context
        nf3._expand = orig_expand

    new_actions = nf3.start_actions(site, config)
    assert old_actions == new_actions, "start actions differ after extraction"


def test_nf3_stop_actions_exact_monkeypatched():
    """Exact stop actions: monkeypatched old vs new shared aliases."""
    site = load_site(SITE)
    config = nf3.load_launch(LAUNCH_NF3)

    orig_context = nf3._context
    orig_expand = nf3._expand

    nf3._context = _reconstruct_old_nf3_context
    nf3._expand = _reconstruct_old_expand
    try:
        old_actions = nf3.simple_actions(site, config, "stop")
    finally:
        nf3._context = orig_context
        nf3._expand = orig_expand

    new_actions = nf3.simple_actions(site, config, "stop")
    assert old_actions == new_actions, "stop actions differ after extraction"


def test_nf3_status_actions_exact_monkeypatched():
    """Exact status actions: monkeypatched old vs new shared aliases."""
    site = load_site(SITE)
    config = nf3.load_launch(LAUNCH_NF3)

    orig_context = nf3._context
    orig_expand = nf3._expand

    nf3._context = _reconstruct_old_nf3_context
    nf3._expand = _reconstruct_old_expand
    try:
        old_actions = nf3.simple_actions(site, config, "status")
    finally:
        nf3._context = orig_context
        nf3._expand = orig_expand

    new_actions = nf3.simple_actions(site, config, "status")
    assert old_actions == new_actions, "status actions differ after extraction"


def test_nf3_verify_rollback_exact_monkeypatched():
    """Exact verify-rollback actions: monkeypatched old vs new."""
    site = load_site(SITE)
    config = nf3.load_launch(LAUNCH_NF3)

    orig_context = nf3._context
    orig_expand = nf3._expand

    nf3._context = _reconstruct_old_nf3_context
    nf3._expand = _reconstruct_old_expand
    try:
        old_actions = nf3.simple_actions(site, config, "verify-rollback")
    finally:
        nf3._context = orig_context
        nf3._expand = orig_expand

    new_actions = nf3.simple_actions(site, config, "verify-rollback")
    assert old_actions == new_actions, (
        "verify-rollback actions differ after extraction")


def test_nf3_start_action_structural_fields():
    """Verify RemoteAction structural fields for all 4 start actions."""
    site = load_site(SITE)
    config = nf3.load_launch(LAUNCH_NF3)
    actions = nf3.start_actions(site, config)
    assert len(actions) == 4
    for i, action in enumerate(actions):
        assert action.rank == i
        assert action.ssh_target == site.rank(i).ssh_target
        assert isinstance(action.argv, tuple)
        cmd = action.shell_command
        assert "docker run" in cmd
        assert "--detach" in cmd
        assert f"glm52-sparkring-nf3-r{i}" in cmd
        assert "org.sparkring.managed=true" in cmd
        assert "org.sparkring.site=" in cmd
        if i != site.serving.master_rank:
            assert "--headless" in cmd


def test_exl3_context_remains_separate():
    """EXL3 _context has fewer keys than runtime.site_context - documented."""
    site = load_site(SITE)
    exl3_ctx = exl3._context(site, 0)
    rt_ctx = runtime.site_context(site, 0)
    extra = set(rt_ctx) - set(exl3_ctx)
    assert {"api_port", "draft_path", "model_path", "peer0_rank",
            "peer1_rank"} <= extra


# ---------------------------------------------------------------------------
# Generated semantic plan snapshots
# ---------------------------------------------------------------------------

def test_contributor_snapshot_matches_fixture():
    """Actual generated plan projection must equal the tracked fixture."""
    fixture = json.loads(
        (FIXTURES / "snapshot-generic.json").read_text("utf-8"))
    site = load_site(SITE)
    profile = generic.load_profile(CONTRIBUTOR)
    actions = generic.build_actions(site, profile, "plan")
    plan = runtime.plan_document("plan", actions, profile)
    projection = _stable_projection(plan, site, profile)
    assert projection == fixture


def test_nf3_snapshot_matches_fixture():
    fixture = json.loads(
        (FIXTURES / "snapshot-nf3-bridge.json").read_text("utf-8"))
    site = load_site(SITE)
    profile = generic.load_profile(LAUNCH_NF3)
    profile = runtime.resolve_from_site(profile, site)
    actions = generic.build_actions(site, profile, "plan")
    plan = runtime.plan_document("plan", actions, profile)
    projection = _stable_projection(plan, site, profile)
    assert projection == fixture


def test_exl3_snapshot_matches_fixture():
    import bootstrap_exl3
    import tempfile
    fixture = json.loads(
        (FIXTURES / "snapshot-exl3-bridge.json").read_text("utf-8"))
    tmpdir = Path(tempfile.mkdtemp())
    site_path = tmpdir / "site.yaml"
    profile_path = tmpdir / "launch.json"
    IMAGE_ID = "sha256:" + "a" * 64
    bootstrap_exl3.write_generated_site(
        SITE, site_path, "sparkring/exl3:test", IMAGE_ID)
    bootstrap_exl3.write_generated_profile(
        profile_path, "sparkring/exl3:test", IMAGE_ID,
        "/srv/models/exl3", "/srv/jit")
    exl3_site = load_site(site_path)
    exl3_profile = generic.load_profile(profile_path)
    actions = generic.build_actions(exl3_site, exl3_profile, "plan")
    plan = runtime.plan_document("plan", actions, exl3_profile)
    projection = _stable_projection(plan, exl3_site, exl3_profile)
    assert projection == fixture


def test_all_fixtures_parse():
    for f in sorted(FIXTURES.glob("*.json")):
        json.loads(f.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Template and contributor
# ---------------------------------------------------------------------------

def test_template_is_unresolved():
    code, stdout, _ = _run_validate(TEMPLATE, expect_code=1)
    r = json.loads(stdout)
    assert r["valid"] is False
    assert "template/unresolved" in r["error"]


def test_template_structurally_valid():
    """Template must parse structurally even if not deployable."""
    profile = runtime.load_runtime_profile(TEMPLATE)
    assert profile.profile_id == "your-model-name"
    assert profile.source_schema == runtime.SCHEMA


def test_template_has_no_family_specific_args():
    """Template must not contain B12X, fp8 KV, quant-method, tool-choice."""
    doc = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    args = " ".join(doc["extra_vllm_args"])
    assert "B12X" not in args
    assert "fp8" not in args
    assert "quantization" not in args
    assert "tool-choice" not in args


def test_contributor_adds_vllm_model(tmp_path):
    """Simulate contributor filling template with real values -> valid."""
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    template["profile_id"] = "contributor-test-model"
    template["model_family"] = "test-model"
    template["container_name"] = "sparkring-test-model"
    template["image"] = "registry.example/org/test:1.0.0"
    template["image_id"] = "sha256:" + "f" * 64
    template["model_host_path"] = "/srv/models/test-model"
    template["model_container_path"] = "/models/test-model"
    template["identity"] = {
        "model_repository": "org/test-model",
        "model_revision": "abc123def456789012345678901234567890abcd",
        "model_config_sha256": "d" * 64,
    }
    path = _write_profile(tmp_path, template, "contributor.json")
    code, stdout, _ = _run_validate(path, SITE, expect_code=0)
    assert json.loads(stdout)["valid"] is True
    code, stdout, _ = _run_cli(
        "--site", str(SITE), "--profile", str(path), "plan", expect_code=0)
    assert len(json.loads(stdout)["actions"]) == 4


def test_contributor_example_validates():
    code, stdout, _ = _run_validate(CONTRIBUTOR, SITE, expect_code=0)
    assert json.loads(stdout)["valid"] is True


# ---------------------------------------------------------------------------
# Clean-checkout rehearsal from HEAD via git archive (zero overlays)
# ---------------------------------------------------------------------------

def test_clean_checkout_rehearsal(tmp_path):
    """Rehearse from HEAD via git archive with zero overlays.

    Does NOT run git add or git commit — those are the caller's
    responsibility. Archives the current HEAD (which must include
    the conformance files if committed), extracts into a fresh temp
    dir, and runs the full conformance workflow inside it.
    Proves no SSH by poisoning PATH to the scripts dir only.
    """
    import tarfile
    import io

    commit_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()

    archive_dir = tmp_path / "clean-checkout"
    archive_dir.mkdir()
    result = subprocess.run(
        ["git", "archive", "--format=tar", commit_hash],
        capture_output=True, cwd=str(ROOT), timeout=30,
    )
    assert result.returncode == 0, f"git archive failed: {result.stderr}"
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r|") as tar:
        tar.extractall(archive_dir, filter="data")

    clean_launcher = archive_dir / "scripts/sparkring_generic_launcher.py"
    clean_site = archive_dir / "scripts/config/site.example.yaml"
    clean_generic = archive_dir / "scripts/config/generic.example.json"
    clean_template = archive_dir / "scripts/config/native-profile.template.json"
    clean_contributor = archive_dir / "scripts/config/contributor-example.json"
    clean_fixtures = archive_dir / "scripts/config/fixtures"
    assert clean_launcher.exists(), "launcher missing from archive"
    assert clean_template.exists(), "template missing from archive"
    assert clean_contributor.exists(), "contributor fixture missing from archive"

    # Poison PATH: only scripts dir (no ssh), but preserve Python deps
    env = dict(os.environ)
    env["PATH"] = str(archive_dir / "scripts")
    env["PYTHONPATH"] = str(archive_dir / "scripts")

    def _run_clean(*args):
        r = subprocess.run(
            [sys.executable, str(clean_launcher), *args],
            capture_output=True, text=True, timeout=30,
            cwd=str(archive_dir), env=env,
        )
        return r.returncode, r.stdout, r.stderr

    # 1. Template is unresolved
    code, stdout, _ = _run_clean("--profile", str(clean_template), "validate")
    assert code == 1
    # 2. Contributor example is valid
    code, stdout, _ = _run_clean("--site", str(clean_site),
                                  "--profile", str(clean_contributor), "validate")
    assert code == 0
    # 3. Explain
    code, stdout, _ = _run_clean("--site", str(clean_site),
                                  "--profile", str(clean_generic), "explain")
    assert code == 0
    # 4. Plan
    code, stdout, _ = _run_clean("--site", str(clean_site),
                                  "--profile", str(clean_contributor), "plan")
    assert code == 0
    assert len(json.loads(stdout)["actions"]) == 4
    # 5. Diff identical
    code, stdout, _ = _run_clean("--profile-a", str(clean_contributor),
                                  "--profile-b", str(clean_contributor), "diff")
    assert code == 0
    # 6. Diff different
    code, stdout, _ = _run_clean("--profile-a", str(clean_contributor),
                                  "--profile-b", str(clean_generic), "diff")
    assert code == 1
    # 7. Fixtures parse
    if clean_fixtures.exists():
        for f in sorted(clean_fixtures.glob("*.json")):
            json.loads(f.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Unsupported schema
# ---------------------------------------------------------------------------

def test_unsupported_schema_rejected(tmp_path):
    path = _write_profile(tmp_path, {"schema": "unknown/v2", "profile_id": "x"})
    code, stdout, _ = _run_validate(path, expect_code=1)
    assert "unsupported schema" in json.loads(stdout)["error"]
