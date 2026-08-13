from __future__ import annotations

import json
from pathlib import Path

import sparkring_exl3_r7_lmcache_canary as canary
import sparkring_generic_launcher as generic
from sparkring_site import load_site


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BASE = ROOT / "scripts/config/contributor-example.json"
SITE = ROOT / "scripts/config/exl3-r7-site.example.yaml"


def _base_document() -> dict:
    document = json.loads(PUBLIC_BASE.read_text(encoding="utf-8"))
    document.update(
        {
            "profile_id": canary.BASE_PROFILE_ID,
            "container_name": "glm52-sparkring-q40-exact-state-canary",
            "image": "sparkring/glm52-exl3-r7-3.5bpw:test",
            "image_id": canary.IMAGE_ID,
            "confirmation": "TEST-ONLY",
        }
    )
    document["environment"].update(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "SPARK_Q40_EXACT_STATE_ATTEST_PATH": (
                "/cache/jit/q40-exact-state-serving-v1-rank{rank}.json"
            ),
        }
    )
    document["extra_volumes"] = [
        {
            "host": "/var/tmp/sparkring-r7-jit",
            "container": "/cache/jit",
            "mode": "rw",
        }
    ]
    return document


def _base_profile():
    return canary.runtime.parse_runtime_profile(
        _base_document(), source="synthetic-public-test-profile"
    )


def _write_base(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "base-profile.json"
    path.write_text(json.dumps(_base_document(), indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(canary, "BASE_PROFILE_SHA256", canary.file_sha256(path))
    monkeypatch.setattr(canary, "SITE_SHA256", canary.file_sha256(SITE))
    return path


def test_derived_profile_changes_only_canary_contract():
    base = _base_profile()
    site = load_site(SITE)

    derived = canary.derive_canary_profile(base, site)

    assert derived.image_id == base.image_id == canary.IMAGE_ID
    assert derived.model_host_path == base.model_host_path
    assert derived.extra_volumes != base.extra_volumes
    assert (canary.CANARY_JIT_HOST_PATH, "/cache/jit", "rw") in (
        derived.extra_volumes
    )
    assert derived.environment["PYTORCH_CUDA_ALLOC_CONF"] == (
        "expandable_segments:False"
    )
    assert derived.environment["SPARK_Q40_EXACT_STATE_ATTEST_PATH"] == (
        base.environment["SPARK_Q40_EXACT_STATE_ATTEST_PATH"]
    )
    index = derived.extra_vllm_args.index("--kv-transfer-config")
    connector = json.loads(derived.extra_vllm_args[index + 1])
    assert connector["kv_connector"] == "LMCacheMPConnector"
    assert connector["kv_load_failure_policy"] == "recompute"
    assert len(
        connector["kv_connector_extra_config"]["lmcache.mp.server_urls"]
    ) == 4


def test_plan_has_bounded_odirect_l2_and_exact_rollback():
    base = _base_profile()
    site = load_site(SITE)
    derived = canary.derive_canary_profile(base, site)

    phases = canary.build_phases(site, base, derived)
    rendered = json.dumps(canary.render_plan(phases, PUBLIC_BASE, SITE, derived))
    server_command = phases["start_servers"][0].shell_command

    assert set(phases) == {
        "precheck",
        "stop_base_engines",
        "start_servers",
        "server_health",
        "clean_canary_receipts",
        "start_canary_engines",
        "canary_ready",
        "stop_canary_engines",
        "stop_servers",
        "restart_base_engines",
        "base_ready",
    }
    assert '"type":"fs_native"' in server_command
    assert '"use_odirect":true' in server_command
    assert '"max_capacity_gb":50' in server_command
    assert "--l1-size-gb 0.5" in server_command
    assert "--l1-init-size-gb 0" in server_command
    receipt_cleanup = phases["clean_canary_receipts"][0].shell_command
    assert canary.CANARY_JIT_HOST_PATH in receipt_cleanup
    assert "q40-exact-state-serving-v1-rank0.json" in receipt_cleanup
    assert "--entrypoint /bin/rm" in receipt_cleanup
    assert canary.IMAGE_ID in receipt_cleanup
    assert canary.BASE_PROFILE_ID in rendered
    assert canary.CONFIRMATION in rendered


def test_main_plan_writes_a_valid_generated_profile(tmp_path, monkeypatch, capsys):
    output = tmp_path / "canary.json"
    base_path = _write_base(tmp_path, monkeypatch)

    assert (
        canary.main(
            [
                "--site",
                str(SITE),
                "--base-profile",
                str(base_path),
                "--output-profile",
                str(output),
                "plan",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    generated = generic.load_profile(output)
    assert plan["maturity"] == "candidate"
    assert generated.profile_id == canary.CANARY_PROFILE_ID
    assert plan["safety"]["start_stops_serving"] is True


def test_precheck_execute_does_not_require_mutation_confirmation(
    tmp_path, monkeypatch, capsys
):
    base_path = _write_base(tmp_path, monkeypatch)
    monkeypatch.setattr(
        canary.runtime,
        "execute",
        lambda actions, timeout: {
            action.rank: {"exit_code": 0, "stdout": "", "stderr": ""}
            for action in actions
        },
    )

    assert (
        canary.main(
            [
                "--site",
                str(SITE),
                "--base-profile",
                str(base_path),
                "--execute",
                "precheck",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert set(result["results"]) == {"precheck"}
