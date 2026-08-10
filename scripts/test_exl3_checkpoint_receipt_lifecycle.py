from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bootstrap_exl3  # noqa: E402
import exl3_checkpoint_receipt_lifecycle as lifecycle  # noqa: E402
import exl3_checkpoint_receipt_sanitize as sanitizer  # noqa: E402
import sparkring_exl3_launcher as exl3  # noqa: E402
from checkpoint_manifest_generator import FileEntry, build_receipt, compute_identity  # noqa: E402
from sparkring_site import load_site  # noqa: E402


IMAGE_ID = "sha256:" + "a" * 64


def generated(tmp_path):
    site_path = tmp_path / "site.yaml"
    profile_path = tmp_path / "profile.json"
    bootstrap_exl3.write_generated_site(
        ROOT / "scripts/config/site.example.yaml",
        site_path,
        "sparkring/exl3:test",
        IMAGE_ID,
    )
    bootstrap_exl3.write_generated_profile(
        profile_path,
        "sparkring/exl3:test",
        IMAGE_ID,
        "/srv/models/exl3",
        "/srv/jit",
    )
    site = load_site(site_path)
    profile = exl3.load_profile(profile_path)
    lifecycle.validate_site_and_profile(site, profile)
    payload = lifecycle.GENERATOR_PATH.read_bytes()
    phases = lifecycle.build_phases(site, profile, payload)
    return site_path, profile_path, site, profile, phases, payload


def receipt(name="model.safetensors", content="c"):
    value = build_receipt(
        [FileEntry(name, 7, content * 64)],
        artifact_root_name="glm52-exl3",
    )
    value["checkpoint_identity_sha256"] = compute_identity(value)
    return value


def rank_results(stdout=""):
    return {
        rank: {"exit_code": 0, "stdout": stdout, "stderr": ""}
        for rank in range(4)
    }


def reserve(tmp_path):
    return lifecycle.reserve_outputs(
        tmp_path / "receipt.json",
        tmp_path / "evidence.json",
        "1" * 32,
    )


class FakeExecutor:
    def __init__(
        self,
        phases,
        *,
        fail_once=(),
        fail_at=None,
        raise_once=None,
        receipts=None,
    ):
        self.names = {id(actions): name for name, actions in phases.items()}
        self.fail_once = set(fail_once)
        self.fail_at = dict(fail_at or {})
        self.raise_once = raise_once
        self.receipts = receipts or {rank: receipt() for rank in range(4)}
        self.calls = []
        self.counts = {}

    def __call__(self, actions, timeout):
        name = self.names[id(actions)]
        self.calls.append((name, timeout))
        self.counts[name] = self.counts.get(name, 0) + 1
        if self.raise_once == name and self.counts[name] == 1:
            raise KeyboardInterrupt(f"interrupt at {name}")
        if (
            name in self.fail_once and self.counts[name] == 1
        ) or self.fail_at.get(name) == self.counts[name]:
            result = rank_results()
            result[2] = {"exit_code": 71, "stdout": "", "stderr": "injected"}
            return result
        if name == "generate_receipts":
            return {
                rank: {
                    "exit_code": 0,
                    "stdout": json.dumps(self.receipts[rank]),
                    "stderr": "",
                }
                for rank in range(4)
            }
        return rank_results()


def test_plan_is_rank_complete_and_generator_is_exact_image_gpu_less(tmp_path, capsys):
    site_path, profile_path, site, profile, phases, payload = generated(tmp_path)
    argv = [
        "--site", str(site_path),
        "--profile", str(profile_path),
        "--receipt-output", str(tmp_path / "receipt.json"),
        "--evidence-output", str(tmp_path / "evidence.json"),
        "plan",
    ]
    assert lifecycle.main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["sensitivity"] == "private-do-not-publish"
    assert plan["safety"]["raw_plan_and_evidence_publishable"] is False
    assert plan["rank_completeness"]["all_phases_exact_rank0_3"] is True
    assert plan["generator_sha256"] == lifecycle.sha256_bytes(payload)
    assert plan["lifecycle"].index("verified_start_prepare") < plan["lifecycle"].index("stop_engines")
    assert plan["lifecycle"].index("reserve_private_outputs") < plan["lifecycle"].index("baseline_preflight")
    for action in phases["generate_receipts"]:
        command = action.shell_command
        assert IMAGE_ID in command
        assert "--network none" in command
        assert "--ipc none" in command
        assert "--read-only" in command
        assert "--cap-drop ALL" in command
        assert "--security-opt no-new-privileges" in command
        assert "docker create" in command and "--interactive" in command
        assert "/srv/models/exl3:/models/glm52-exl3-tr3-3.25bpw:ro" in command
        assert "docker create" in command
        assert "--entrypoint /bin/sh" in command
        assert "--runtime runc" in command
        assert "NVIDIA_VISIBLE_DEVICES=void" in command
        assert "docker start --attach --interactive" in command
        assert "set -- /dev/nvidia*" in command
        assert "checkpoint-manifest-v2.XXXXXX" in command
        assert "/opt/venv/bin/python -S" in command
        assert lifecycle.sha256_bytes(payload) in command
        assert " --gpus " not in command
        assert " --device " not in command
        assert "--privileged" not in command
        assert "org.sparkring.component=checkpoint-manifest-v2-generator" in command
        assert f"org.sparkring.generator-sha256={lifecycle.sha256_bytes(payload)}" in command
        assert "name,image,profile,generator,model_src,model_dst,container_script,require_stopped" in command
        assert "DeviceRequests" in command and "Runtime" in command


def test_helper_ownership_is_closed_over_runtime_devices_and_environment():
    ownership = lifecycle._HELPER_OWNERSHIP  # noqa: SLF001
    assert "hc.get('Runtime')=='runc'" in ownership
    assert "not (hc.get('DeviceRequests') or [])" in ownership
    assert "not (hc.get('Devices') or [])" in ownership
    assert "'NVIDIA_VISIBLE_DEVICES':'void'" in ownership
    assert "doc['Config']['Cmd']==['-c',container_script]" in ownership
    assert "doc['Config']['OpenStdin'] is True" in ownership
    assert "doc['Config']['StdinOnce'] is True" in ownership
    assert "doc['Config']['AttachStdin'] is True" in ownership
    assert "if require_stopped=='1': assert doc['State']['Running'] is False" in ownership


def test_exact_engine_removal_attests_verified_start_contract(tmp_path):
    _site_path, _profile_path, _site, _profile, phases, _payload = generated(tmp_path)
    for action in phases["stop_engines"]:
        command = action.shell_command
        assert lifecycle.verified_start.REMOTE_OUTER_VERIFIED_ENTRYPOINT in command
        assert '"SPARKRING_OUTER_MODEL_VERIFIED":"1"' in command
        assert "entrypoint_override" in command


def test_reserve_outputs_is_dual_exclusive_and_finalize_is_atomic(tmp_path):
    reservation = reserve(tmp_path)
    assert reservation["receipt_path"].exists()
    assert reservation["evidence_path"].exists()
    with pytest.raises(FileExistsError):
        lifecycle.reserve_outputs(
            reservation["receipt_path"], tmp_path / "second-evidence.json", "other"
        )
    summary = lifecycle.finalize_receipt(reservation, receipt())
    assert summary["checkpoint_identity_sha256"] == receipt()["checkpoint_identity_sha256"]
    assert json.loads(reservation["receipt_path"].read_text()) == receipt()


def test_reservation_tamper_is_terminal_before_any_stop(tmp_path):
    _site_path, _profile_path, _site, profile, phases, _payload = generated(tmp_path)
    reservation = reserve(tmp_path)
    reservation["receipt_path"].write_text('{"foreign":true}', encoding="utf-8")
    executor = FakeExecutor(phases)
    with pytest.raises(lifecycle.LifecycleError, match="reservation"):
        lifecycle.execute_transaction(
            phases, profile, reservation, hash_timeout=300, executor=executor
        )
    names = [name for name, _timeout in executor.calls]
    assert names == ["baseline_preflight", "verified_start_prepare", "helper_absent"]
    evidence = json.loads(reservation["evidence_path"].read_text())
    assert evidence["terminal_before_removal"] is True
    assert evidence["restoration"] is None


@pytest.mark.parametrize("failed", ["baseline_preflight", "verified_start_prepare", "helper_absent"])
def test_every_prestop_failure_is_terminal_before_removal(tmp_path, failed):
    _site_path, _profile_path, _site, profile, phases, _payload = generated(tmp_path)
    reservation = reserve(tmp_path)
    executor = FakeExecutor(phases, fail_once={failed})
    with pytest.raises(lifecycle.LifecycleError, match=f"phase {failed}"):
        lifecycle.execute_transaction(
            phases, profile, reservation, hash_timeout=300, executor=executor
        )
    names = [name for name, _timeout in executor.calls]
    assert "stop_engines" not in names
    assert "stop_servers" not in names
    assert json.loads(reservation["evidence_path"].read_text())["execution_state"] == "failed-before-stop"


@pytest.mark.parametrize(
    "failed",
    ["stop_engines", "stop_servers", "quiescence", "generate_receipts", "remove_helper"],
)
def test_failure_at_each_destructive_boundary_restores_then_fails(tmp_path, failed):
    _site_path, _profile_path, _site, profile, phases, _payload = generated(tmp_path)
    reservation = reserve(tmp_path)
    executor = FakeExecutor(phases, fail_once={failed})
    with pytest.raises(lifecycle.LifecycleError, match=f"phase {failed}"):
        lifecycle.execute_transaction(
            phases, profile, reservation, hash_timeout=300, executor=executor
        )
    names = [name for name, _timeout in executor.calls]
    assert names.count("verified_start_prepare") == 2
    assert "start_servers" in names and "start_engines" in names
    evidence = json.loads(reservation["evidence_path"].read_text())
    assert evidence["passed"] is False
    assert evidence["execution_state"] == "failed-restored"
    assert evidence["restoration"]["passed"] is True


def test_keyboard_interrupt_after_stop_restores_then_is_reraised(tmp_path):
    _site_path, _profile_path, _site, profile, phases, _payload = generated(tmp_path)
    reservation = reserve(tmp_path)
    executor = FakeExecutor(phases, raise_once="generate_receipts")
    with pytest.raises(KeyboardInterrupt, match="generate_receipts"):
        lifecycle.execute_transaction(
            phases, profile, reservation, hash_timeout=300, executor=executor
        )
    names = [name for name, _timeout in executor.calls]
    assert "start_servers" in names and "start_engines" in names
    evidence = json.loads(reservation["evidence_path"].read_text())
    assert evidence["execution_state"] == "failed-restored"


def test_private_evidence_hashes_remote_errors_without_embedding_text(tmp_path):
    _site_path, _profile_path, _site, profile, phases, _payload = generated(tmp_path)
    reservation = reserve(tmp_path)
    private_error = "/srv/private/Cody/model operator@rank0 192.0.2.99"

    class ErrorExecutor(FakeExecutor):
        def __call__(self, actions, timeout):
            result = super().__call__(actions, timeout)
            if self.names[id(actions)] == "baseline_preflight":
                result[0] = {
                    "exit_code": 71,
                    "stdout": "",
                    "stderr": private_error,
                }
            return result

    with pytest.raises(lifecycle.LifecycleError, match="baseline_preflight"):
        lifecycle.execute_transaction(
            phases,
            profile,
            reservation,
            hash_timeout=300,
            executor=ErrorExecutor(phases),
        )
    payload = reservation["evidence_path"].read_text(encoding="utf-8")
    evidence = json.loads(payload)
    assert evidence["sensitivity"] == "private-do-not-publish"
    assert private_error not in payload
    result = evidence["phases"][0]["ranks"]["0"]
    assert result["stderr_bytes"] == len(private_error.encode("utf-8"))
    assert result["stderr_sha256"] == lifecycle.sha256_bytes(private_error.encode())
    assert result["error_text_included"] is False


def test_restore_failure_overrides_successful_receipt_generation(tmp_path):
    _site_path, _profile_path, _site, profile, phases, _payload = generated(tmp_path)
    reservation = reserve(tmp_path)
    executor = FakeExecutor(phases, fail_once={"start_engines"})
    with pytest.raises(lifecycle.LifecycleError, match="restoration failed"):
        lifecycle.execute_transaction(
            phases, profile, reservation, hash_timeout=300, executor=executor
        )
    assert json.loads(reservation["receipt_path"].read_text())["manifest_version"] == 2
    evidence = json.loads(reservation["evidence_path"].read_text())
    assert evidence["passed"] is False
    assert evidence["execution_state"] == "failed-restoration-unproven"
    assert evidence["restoration"]["passed"] is False


@pytest.mark.parametrize(
    "name,count",
    [
        ("verified_start_prepare", 2),
        ("remove_helper", 2),
        ("stop_engines", 2),
        ("stop_servers", 2),
        ("quiescence", 2),
        ("start_servers", 1),
        ("server_health", 1),
        ("start_engines", 1),
        ("engine_ready", 1),
        ("baseline_final", 1),
        ("helper_absent", 2),
    ],
)
def test_every_restoration_boundary_failure_overrides_primary_success(
    tmp_path, name, count
):
    _site_path, _profile_path, _site, profile, phases, _payload = generated(tmp_path)
    reservation = reserve(tmp_path)
    executor = FakeExecutor(phases, fail_at={name: count})
    with pytest.raises(lifecycle.LifecycleError, match="restoration failed"):
        lifecycle.execute_transaction(
            phases, profile, reservation, hash_timeout=300, executor=executor
        )
    evidence = json.loads(reservation["evidence_path"].read_text())
    assert evidence["passed"] is False
    assert evidence["execution_state"] == "failed-restoration-unproven"
    assert evidence["restoration"]["passed"] is False


def test_success_requires_receipt_and_full_restored_health(tmp_path):
    _site_path, _profile_path, _site, profile, phases, _payload = generated(tmp_path)
    reservation = reserve(tmp_path)
    executor = FakeExecutor(phases)
    evidence = lifecycle.execute_transaction(
        phases, profile, reservation, hash_timeout=300, executor=executor
    )
    assert evidence["passed"] is True
    assert evidence["execution_state"] == "completed-restored"
    assert evidence["restoration"]["passed"] is True
    assert json.loads(reservation["receipt_path"].read_text()) == receipt()
    names = [name for name, _timeout in executor.calls]
    assert names[-3:] == ["server_health", "baseline_final", "helper_absent"]


def test_rank_receipt_disagreement_restores_and_rejects(tmp_path):
    _site_path, _profile_path, _site, profile, phases, _payload = generated(tmp_path)
    receipts = {rank: receipt() for rank in range(4)}
    receipts[2] = receipt("different.safetensors", "d")
    reservation = reserve(tmp_path)
    executor = FakeExecutor(phases, receipts=receipts)
    with pytest.raises(lifecycle.LifecycleError, match="differs from rank 0"):
        lifecycle.execute_transaction(
            phases, profile, reservation, hash_timeout=300, executor=executor
        )
    evidence = json.loads(reservation["evidence_path"].read_text())
    assert evidence["restoration"]["passed"] is True
    assert json.loads(reservation["receipt_path"].read_text())["state"] == "reserved-incomplete"


@pytest.mark.parametrize(
    "bad,match",
    [
        ("not-json", "malformed JSON"),
        ('{"manifest_version":2,"manifest_version":2}', "duplicate key"),
        ('{"manifest_version":NaN}', "non-finite"),
        ("[]", "checkpoint receipt must be an object"),
    ],
)
def test_malformed_duplicate_and_nonfinite_generator_json_rejected(bad, match):
    result = rank_results(json.dumps(receipt()))
    result[1]["stdout"] = bad
    with pytest.raises(lifecycle.LifecycleError, match=match):
        lifecycle.parse_equal_receipts(result)


def test_generator_stderr_and_incomplete_rank_set_rejected():
    result = rank_results(json.dumps(receipt()))
    result[3]["stderr"] = "warning"
    with pytest.raises(lifecycle.LifecycleError, match="stderr"):
        lifecycle.parse_equal_receipts(result)
    del result[3]
    with pytest.raises(lifecycle.LifecycleError, match="generators failed"):
        lifecycle.parse_equal_receipts(result)


def test_execute_requires_confirmation_before_reserving_outputs(tmp_path):
    site_path, profile_path, _site, _profile, _phases, _payload = generated(tmp_path)
    argv = [
        "--site", str(site_path),
        "--profile", str(profile_path),
        "--receipt-output", str(tmp_path / "receipt.json"),
        "--evidence-output", str(tmp_path / "evidence.json"),
        "--execute",
        "generate",
    ]
    with pytest.raises(SystemExit):
        lifecycle.main(argv)
    assert not (tmp_path / "receipt.json").exists()
    assert not (tmp_path / "evidence.json").exists()


def successful_private_artifacts(tmp_path):
    _site_path, _profile_path, _site, profile, phases, _payload = generated(tmp_path)
    reservation = reserve(tmp_path)
    reservation["evidence"].update(
        {
            "profile_id": profile.profile_id,
            "image_id": profile.image_id,
            "generator_sha256": lifecycle.sha256_bytes(lifecycle.GENERATOR_PATH.read_bytes()),
            "required_ranks": [0, 1, 2, 3],
        }
    )
    lifecycle.checkpoint_evidence(reservation)
    lifecycle.execute_transaction(
        phases,
        profile,
        reservation,
        hash_timeout=300,
        executor=FakeExecutor(phases),
    )
    return reservation["receipt_path"], reservation["evidence_path"]


def test_public_sanitizer_is_closed_and_cannot_copy_private_paths(tmp_path):
    receipt_path, evidence_path = successful_private_artifacts(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    secrets = (
        "operator@private-rank",
        "/srv/private/models/Cody/checkpoint",
        "192.0.2.99",
    )
    evidence["phases"][0]["private_test_values"] = list(secrets)
    evidence["restoration"]["private_test_values"] = list(secrets)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    output = tmp_path / "public.json"
    assert sanitizer.main(
        [
            "--receipt", str(receipt_path),
            "--evidence", str(evidence_path),
            "--output", str(output),
        ]
    ) == 0
    payload = output.read_text(encoding="utf-8")
    public = json.loads(payload)
    sanitizer.validate_public_receipt(public)
    assert public["sensitivity"] == "public-sanitized"
    assert public["transaction"]["restoration_passed"] is True
    assert all(secret not in payload for secret in secrets)
    assert "phases" not in public
    assert "artifact_root_name" not in payload
    assert "files" not in public["checkpoint"]


def test_public_sanitizer_rejects_failure_rank_drift_and_receipt_drift(tmp_path):
    receipt_path, evidence_path = successful_private_artifacts(tmp_path)
    receipt_payload = receipt_path.read_bytes()
    evidence_payload = evidence_path.read_bytes()
    receipt_doc = lifecycle.strict_json_loads(receipt_payload, "receipt")
    evidence_doc = lifecycle.strict_json_loads(evidence_payload, "evidence")
    for mutation, match in (
        (lambda value: value.update(passed=False), "successful"),
        (lambda value: value.update(required_ranks=[0, 1, 2]), "exact ranks"),
    ):
        changed = json.loads(json.dumps(evidence_doc))
        mutation(changed)
        with pytest.raises(sanitizer.SanitizeError, match=match):
            sanitizer.build_public_receipt(
                receipt_doc,
                changed,
                receipt_payload=receipt_payload,
                evidence_payload=json.dumps(changed).encode(),
            )
    changed_receipt = receipt("different.safetensors", "d")
    changed_payload = lifecycle.encoded_pretty(changed_receipt)
    with pytest.raises(sanitizer.SanitizeError, match="do not match"):
        sanitizer.build_public_receipt(
            changed_receipt,
            evidence_doc,
            receipt_payload=changed_payload,
            evidence_payload=evidence_payload,
        )


def test_public_validator_rejects_unknown_fields(tmp_path):
    receipt_path, evidence_path = successful_private_artifacts(tmp_path)
    receipt_payload = receipt_path.read_bytes()
    evidence_payload = evidence_path.read_bytes()
    public = sanitizer.build_public_receipt(
        lifecycle.strict_json_loads(receipt_payload, "receipt"),
        lifecycle.strict_json_loads(evidence_payload, "evidence"),
        receipt_payload=receipt_payload,
        evidence_payload=evidence_payload,
    )
    public["private_path"] = "/must/not/pass"
    with pytest.raises(sanitizer.SanitizeError, match="schema-closed"):
        sanitizer.validate_public_receipt(public)
