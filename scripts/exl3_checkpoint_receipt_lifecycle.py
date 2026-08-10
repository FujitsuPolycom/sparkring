#!/usr/bin/env python3
"""Confirmation-gated four-Spark EXL3 checkpoint-receipt lifecycle.

This is the supported bridge between the canonical EXL3+LMCache deployment and
SparkCache's checkpoint-manifest-v2 identity.  A dry run prints every remote
action.  Execute mode reserves both private outputs before any interruption,
attests the exact live rank-0..3 baseline, prepares the reviewed verified-start
wrapper/reclaim authority, stops only exact-owned containers, proves model/GPU
quiescence, runs the published generator in the pinned image without GPU
access, requires canonical equality across all ranks, writes the receipt, and
restores the canonical stack.  Every failure after interruption attempts the
same exact verified restoration before returning failure.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import exl3_sparkcache_config as sparkcache_config  # noqa: E402
import exl3_verified_start as verified_start  # noqa: E402
import sparkring_exl3_launcher as exl3  # noqa: E402
import sparkring_exl3_lmcache_launcher as lmcache  # noqa: E402
import sparkring_exl3_sparkcache_launcher as sparkcache_launcher  # noqa: E402
from sparkring_site import SiteConfigError, load_site  # noqa: E402


PLAN_SCHEMA = "sparkring-exl3-checkpoint-receipt-plan/v1"
EVIDENCE_SCHEMA = "sparkring-exl3-checkpoint-receipt-evidence/v1"
RECEIPT_RESERVATION_SCHEMA = "sparkring-checkpoint-receipt-reservation/v1"
PRIVATE_SENSITIVITY = "private-do-not-publish"
CONFIRMATION = "GENERATE-EXL3-CHECKPOINT-RECEIPT-ALL-FOUR"
EXPECTED_RANKS = frozenset(range(4))
GENERATOR_PATH = ROOT / "scripts/checkpoint_manifest_generator.py"
GENERATOR_COMPONENT = "checkpoint-manifest-v2-generator"
HELPER_PREFIX = "glm52-sparkring-checkpoint-manifest-v2"


class LifecycleError(RuntimeError):
    """A lifecycle invariant or phase failed."""


def strict_json_loads(payload: str | bytes, label: str) -> Any:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise LifecycleError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise LifecycleError(f"{label} contains non-finite number {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise LifecycleError(f"{label} is malformed JSON: {error}") from error


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise LifecycleError(f"value is not canonical JSON: {error}") from error
    return encoded.encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encoded_pretty(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_replace_owned(
    path: Path, document: dict, run_id: str, expected_sha256: str
) -> str:
    """Checkpoint a run-owned JSON document without overwriting foreign data."""
    try:
        current_payload = path.read_bytes()
        current = strict_json_loads(current_payload, "reserved output")
    except (OSError, LifecycleError) as error:
        raise LifecycleError(f"reserved output is unreadable or malformed: {error}") from error
    if not isinstance(current, dict) or current.get("run_id") != run_id:
        raise LifecycleError("reserved output ownership changed")
    if sha256_bytes(current_payload) != expected_sha256:
        raise LifecycleError("reserved output bytes changed outside this run")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(encoded_pretty(document))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    final_payload = path.read_bytes()
    if strict_json_loads(final_payload, "checkpointed output") != document:
        raise LifecycleError("checkpointed output differs after atomic replacement")
    return sha256_bytes(final_payload)


def reserve_outputs(receipt_path: Path, evidence_path: Path, run_id: str) -> dict:
    """Exclusively reserve both private outputs before any remote mutation."""
    receipt_path = receipt_path.absolute()
    evidence_path = evidence_path.absolute()
    if receipt_path == evidence_path:
        raise LifecycleError("receipt and evidence outputs must be distinct")
    for path in (receipt_path, evidence_path):
        if path.suffix.lower() != ".json":
            raise LifecycleError("private output paths must use a .json suffix")
        path.parent.mkdir(parents=True, exist_ok=True)
    receipt_reservation = {
        "schema": RECEIPT_RESERVATION_SCHEMA,
        "run_id": run_id,
        "state": "reserved-incomplete",
    }
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "sensitivity": PRIVATE_SENSITIVITY,
        "run_id": run_id,
        "lane": "public-functional",
        "maturity": "offline-validated-tool-live-execution-pending",
        "execution_state": "reserved",
        "started_at": utc_now(),
        "phases": [],
        "receipt": None,
        "restoration": None,
        "passed": False,
    }
    created: list[Path] = []
    try:
        for path, document in (
            (receipt_path, receipt_reservation),
            (evidence_path, evidence),
        ):
            with path.open("xb") as output:
                output.write(encoded_pretty(document))
                output.flush()
                os.fsync(output.fileno())
            created.append(path)
    except Exception:
        for path in created:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return {
        "receipt_path": receipt_path,
        "evidence_path": evidence_path,
        "receipt_reservation": receipt_reservation,
        "receipt_reservation_sha256": sha256_bytes(encoded_pretty(receipt_reservation)),
        "evidence_sha256": sha256_bytes(encoded_pretty(evidence)),
        "evidence": evidence,
    }


def assert_receipt_reservation(reservation: dict) -> None:
    path = reservation["receipt_path"]
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise LifecycleError(f"receipt reservation cannot be read: {error}") from error
    if sha256_bytes(payload) != reservation["receipt_reservation_sha256"]:
        raise LifecycleError("receipt reservation changed before finalization")
    try:
        current = strict_json_loads(payload, "receipt reservation")
    except LifecycleError as error:
        raise LifecycleError("receipt reservation became malformed") from error
    if current != reservation["receipt_reservation"]:
        raise LifecycleError("receipt reservation contents changed")


def finalize_receipt(reservation: dict, receipt: dict) -> dict:
    """Atomically replace only this run's reservation with a canonical receipt."""
    identity = sparkcache_config.validate_checkpoint_receipt(receipt)
    assert_receipt_reservation(reservation)
    path = reservation["receipt_path"]
    payload = encoded_pretty(receipt)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        # Recheck immediately before the replacement closes the ownership race.
        assert_receipt_reservation(reservation)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if path.read_bytes() != payload:
        raise LifecycleError("final receipt bytes differ after atomic replacement")
    return {
        "schema": "sparkcache-checkpoint-manifest-v2",
        "checkpoint_identity_sha256": identity,
        "receipt_sha256": sha256_bytes(payload),
        "file_count": receipt["file_count"],
    }


def checkpoint_evidence(reservation: dict) -> None:
    reservation["evidence_sha256"] = _atomic_replace_owned(
        reservation["evidence_path"],
        reservation["evidence"],
        reservation["evidence"]["run_id"],
        reservation["evidence_sha256"],
    )


def validate_site_and_profile(site, profile: exl3.Profile) -> None:
    rank_ids = [rank.id for rank in site.ranks]
    targets = [rank.ssh_target for rank in site.ranks]
    if rank_ids != [0, 1, 2, 3] or set(rank_ids) != EXPECTED_RANKS:
        raise LifecycleError("site must contain ordered exact ranks 0,1,2,3")
    if any(not isinstance(target, str) or not target for target in targets):
        raise LifecycleError("every rank requires an explicit SSH target")
    if len(set(targets)) != 4:
        raise LifecycleError("rank SSH targets must be distinct")
    if site.serving.master_rank != 0:
        raise LifecycleError("canonical EXL3+LMCache baseline requires master rank 0")
    if profile.profile_id != lmcache.PROFILE_ID or profile.engine != "docker":
        raise LifecycleError("baseline must be canonical Docker EXL3+LMCache CS512")
    recipe = json.loads(exl3.RECIPE_PATH.read_text(encoding="utf-8"))
    model = recipe["model"]
    exact_model = {
        "model_repository": model["repository"],
        "model_revision": model["revision"],
        "model_config_sha256": model["config_sha256"],
        "model_index_sha256": model["index_sha256"],
        "model_tier_bitmap_sha256": model["tier_bitmap_sha256"],
        "model_manifest_sha256": model["manifest_sha256"],
        "model_shard_count": model["shard_count"],
        "model_weight_bytes": model["weight_bytes"],
    }
    observed = {key: profile.document[key] for key in exact_model}
    if canonical_bytes(observed) != canonical_bytes(exact_model):
        raise LifecycleError("baseline model identity differs from the published EXL3 recipe")


def helper_name(rank: int) -> str:
    return f"{HELPER_PREFIX}-r{rank}"


def _shell_action(rank, script: str) -> exl3.RemoteAction:
    return exl3.RemoteAction(rank.id, rank.ssh_target, ("sh", "-lc", script))


def _combine_actions(*groups: list[exl3.RemoteAction]) -> list[exl3.RemoteAction]:
    by_group = [{action.rank: action for action in group} for group in groups]
    if any(set(group) != EXPECTED_RANKS for group in by_group):
        raise LifecycleError("combined action group does not contain exact ranks 0,1,2,3")
    result = []
    for rank in sorted(EXPECTED_RANKS):
        actions = [group[rank] for group in by_group]
        if len({action.ssh_target for action in actions}) != 1:
            raise LifecycleError("combined actions disagree on SSH authority")
        scripts = []
        for action in actions:
            if action.argv[:2] != ("sh", "-lc") or len(action.argv) != 3:
                raise LifecycleError("combined action is not a guarded shell action")
            scripts.append(f"({action.argv[-1]})")
        result.append(
            exl3.RemoteAction(rank, actions[0].ssh_target, ("sh", "-lc", " && ".join(scripts)))
        )
    return result


def baseline_preflight_actions(site, profile: exl3.Profile) -> list[exl3.RemoteAction]:
    """Attest exact running engines/servers plus API and server health."""
    return _combine_actions(
        sparkcache_launcher.baseline_final_status_actions(site, profile, {}),
        lmcache.ready_actions(site, profile),
        lmcache.server_health_actions(site, engine=profile.engine),
    )


def helper_absent_actions(site) -> list[exl3.RemoteAction]:
    return [
        _shell_action(
            rank,
            "docker info >/dev/null"
            f" && ids=$(docker ps -aq --filter name=^/{shlex.quote(helper_name(rank.id))}$)"
            " && test -z \"$ids\"",
        )
        for rank in site.ranks
    ]


_HELPER_OWNERSHIP = r"""
import json,subprocess,sys
name,image,profile,generator,model_src,model_dst,container_script,require_stopped=sys.argv[1:]
doc=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
image_doc=json.loads(subprocess.check_output(['docker','image','inspect',image]))[0]
assert require_stopped in ('0','1')
assert doc['Name']=='/'+name and doc['Image']==image
if require_stopped=='1': assert doc['State']['Running'] is False
expected_labels=dict(image_doc['Config'].get('Labels') or {})
expected_labels.update({'org.sparkring.managed':'true','org.sparkring.exl3-profile':profile,'org.sparkring.component':'checkpoint-manifest-v2-generator','org.sparkring.generator-sha256':generator})
assert (doc['Config']['Labels'] or {})==expected_labels
assert doc['Config']['Cmd']==['-c',container_script]
assert doc['Config']['Entrypoint']==['/bin/sh']
assert doc['Config']['OpenStdin'] is True
assert doc['Config']['StdinOnce'] is True
assert doc['Config']['AttachStdin'] is True
actual_items=doc['Config'].get('Env') or []
actual=dict(item.split('=',1) for item in actual_items if '=' in item)
assert len(actual)==len(actual_items)
base_items=image_doc['Config'].get('Env') or []
expected=dict(item.split('=',1) for item in base_items if '=' in item)
expected.update({'NVIDIA_VISIBLE_DEVICES':'void','PYTHONDONTWRITEBYTECODE':'1'})
assert actual==expected
mounts={m['Destination']:(m['Source'],m['RW']) for m in doc.get('Mounts') or []}
assert mounts=={model_dst:(model_src,False)}
hc=doc['HostConfig']
assert hc['NetworkMode']=='none' and hc['IpcMode']=='none'
assert hc['ReadonlyRootfs'] is True and hc['Privileged'] is False
assert not (hc.get('DeviceRequests') or []) and not (hc.get('Devices') or [])
assert hc.get('Runtime')=='runc'
assert hc.get('AutoRemove') is False
assert hc.get('CapDrop')==['ALL']
assert hc.get('SecurityOpt')==['no-new-privileges']
assert hc.get('PidsLimit')==128
assert hc.get('Tmpfs')=={'/tmp':'rw,noexec,nosuid,nodev,size=16m'}
""".strip()


def generator_container_script(model_container_path: str, generator_sha256: str) -> str:
    """Return the exact in-container hash/device barrier and generator command."""
    return (
        "set -eu; "
        "tmp=$(mktemp /tmp/checkpoint-manifest-v2.XXXXXX); "
        "trap 'rm -f \"$tmp\"' EXIT HUP INT TERM; "
        "cat > \"$tmp\"; "
        f"test \"$(sha256sum \"$tmp\" | awk '{{print $1}}')\" = {shlex.quote(generator_sha256)}; "
        "set -- /dev/nvidia*; test \"$1\" = '/dev/nvidia*'; "
        # The runtime image's sitecustomize hook emits loader diagnostics to
        # stdout.  This manifest generator is stdlib-only, so ``-S`` keeps the
        # receipt stream strict JSON and excludes image import hooks.
        f"exec /opt/venv/bin/python -S \"$tmp\" --artifact-root {shlex.quote(model_container_path)}"
    )


def helper_ownership_command(
    name: str,
    profile: exl3.Profile,
    generator_sha256: str,
    *,
    require_stopped: bool = False,
) -> str:
    return shlex.join(
        (
            "python3",
            "-c",
            _HELPER_OWNERSHIP,
            name,
            profile.image_id,
            profile.profile_id,
            generator_sha256,
            profile.model_host_path,
            profile.model_container_path,
            generator_container_script(
                profile.model_container_path, generator_sha256
            ),
            "1" if require_stopped else "0",
        )
    )


def helper_remove_actions(site, profile: exl3.Profile, generator_sha256: str) -> list[exl3.RemoteAction]:
    actions = []
    for rank in site.ranks:
        name = helper_name(rank.id)
        attest = helper_ownership_command(name, profile, generator_sha256)
        script = (
            "docker info >/dev/null || exit 70; "
            f"name={shlex.quote(name)}; "
            "if ! docker inspect \"$name\" >/dev/null 2>&1; then "
            "ids=$(docker ps -aq --filter name=^/$name$) || exit 71; "
            "test -z \"$ids\" || exit 77; exit 0; fi; "
            f"{attest} || exit 78; exec docker rm --force \"$name\""
        )
        actions.append(_shell_action(rank, script))
    return actions


def quiescence_actions(site, profile: exl3.Profile) -> list[exl3.RemoteAction]:
    actions = []
    for rank in site.ranks:
        names = (
            exl3.container_name(profile, rank.id),
            lmcache.server_name(rank.id),
            helper_name(rank.id),
        )
        attest = shlex.join(
            (
                "python3",
                "-c",
                sparkcache_launcher._NO_MODEL_CONTAINERS_ATTEST,  # noqa: SLF001
                profile.model_host_path,
                *names,
            )
        )
        check = (
            f"{attest}"
            " && gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
            " && test -z \"$(printf '%s\\n' \"$gpu_pids\" | sed '/^[[:space:]]*$/d')\""
        )
        script = (
            "docker info >/dev/null"
            f" && for i in $(seq 1 60); do {check} && exit 0; sleep 1; done"
            f" && {check}"
        )
        actions.append(_shell_action(rank, script))
    return actions


def generator_actions(
    site, profile: exl3.Profile, generator_payload: bytes, generator_sha256: str
) -> list[exl3.RemoteAction]:
    encoded = base64.b64encode(generator_payload).decode("ascii")
    actions = []
    for rank in site.ranks:
        name = helper_name(rank.id)
        container_script = generator_container_script(
            profile.model_container_path, generator_sha256
        )
        command = shlex.join(
            (
                "docker",
                "create",
                "--name",
                name,
                "--label",
                "org.sparkring.managed=true",
                "--label",
                f"org.sparkring.exl3-profile={profile.profile_id}",
                "--label",
                f"org.sparkring.component={GENERATOR_COMPONENT}",
                "--label",
                f"org.sparkring.generator-sha256={generator_sha256}",
                "--network",
                "none",
                "--ipc",
                "none",
                "--runtime",
                "runc",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "128",
                "--interactive",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=void",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=16m",
                "--volume",
                f"{profile.model_host_path}:{profile.model_container_path}:ro",
                "--entrypoint",
                "/bin/sh",
                profile.image_id,
                "-c",
                container_script,
            )
        )
        cleanup_attest = helper_ownership_command(name, profile, generator_sha256)
        prestart_attest = helper_ownership_command(
            name, profile, generator_sha256, require_stopped=True
        )
        script = (
            f"test \"$(docker image inspect --format '{{{{.Id}}}}' {shlex.quote(profile.image)})\" = {shlex.quote(profile.image_id)}"
            f" && test -z \"$(docker ps -aq --filter name=^/{shlex.quote(name)}$)\""
            f" && name={shlex.quote(name)} && ("
            "created=0; "
            f"cleanup() {{ if test \"$created\" = 1; then {cleanup_attest} >/dev/null 2>&1"
            " && docker rm --force \"$name\" >/dev/null 2>&1 || true; fi; }; "
            "trap cleanup EXIT HUP INT TERM; "
            f"{command} >/dev/null && created=1"
            f" && {prestart_attest}"
            f" && printf %s {shlex.quote(encoded)} | base64 -d | "
            "docker start --attach --interactive \"$name\")"
        )
        actions.append(_shell_action(rank, script))
    return actions


def build_phases(site, profile: exl3.Profile, generator_payload: bytes) -> dict:
    generator_sha256 = sha256_bytes(generator_payload)
    return {
        "baseline_preflight": baseline_preflight_actions(site, profile),
        "verified_start_prepare": verified_start.prepare_verified_start_actions(site),
        "helper_absent": helper_absent_actions(site),
        "stop_engines": sparkcache_launcher.baseline_exact_remove_actions(
            site, profile, component="engine"
        ),
        "stop_servers": sparkcache_launcher.baseline_exact_remove_actions(
            site, profile, component="lmcache-server"
        ),
        "quiescence": quiescence_actions(site, profile),
        "generate_receipts": generator_actions(
            site, profile, generator_payload, generator_sha256
        ),
        "remove_helper": helper_remove_actions(site, profile, generator_sha256),
        "start_servers": lmcache.server_start_actions(site, profile),
        "server_health": lmcache.server_health_actions(site, engine=profile.engine),
        "start_engines": sparkcache_launcher.baseline_verified_start_actions(site, profile),
        "engine_ready": lmcache.ready_actions(site, profile),
        "baseline_final": sparkcache_launcher.baseline_final_status_actions(site, profile, {}),
    }


def render(actions: list[exl3.RemoteAction]) -> list[dict]:
    return [
        {"rank": action.rank, "ssh_target": action.ssh_target, "remote_command": action.shell_command}
        for action in actions
    ]


def _failure_result(kind: str, detail: str) -> dict[int, dict]:
    return {
        rank: {
            "exit_code": 125,
            "stdout": "",
            "stderr": f"lifecycle_failure[{kind}]: {detail}"[:4096],
        }
        for rank in sorted(EXPECTED_RANKS)
    }


def execute_phase(
    executor: Callable,
    actions: list[exl3.RemoteAction],
    *,
    timeout: int,
) -> dict[int, dict]:
    ranks = [action.rank for action in actions]
    if len(ranks) != 4 or any(type(rank) is not int for rank in ranks) or set(ranks) != EXPECTED_RANKS:
        return _failure_result("action-ranks", repr(ranks))
    try:
        raw = executor(actions, timeout=timeout)
    except Exception as error:
        return _failure_result("executor-exception", f"{type(error).__name__}: {error}")
    if not isinstance(raw, dict) or any(type(rank) is not int for rank in raw) or set(raw) != EXPECTED_RANKS:
        return _failure_result("executor-ranks", repr(list(raw) if isinstance(raw, dict) else type(raw)))
    result = {}
    for rank in sorted(EXPECTED_RANKS):
        item = raw[rank]
        if not isinstance(item, dict) or set(item) != {"exit_code", "stdout", "stderr"}:
            return _failure_result("executor-shape", f"rank {rank}")
        if (
            isinstance(item["exit_code"], bool)
            or not isinstance(item["exit_code"], int)
            or not isinstance(item["stdout"], str)
            or not isinstance(item["stderr"], str)
        ):
            return _failure_result("executor-types", f"rank {rank}")
        result[rank] = dict(item)
    return result


def phase_failed(result: Mapping[int, Mapping[str, Any]]) -> bool:
    return set(result) != EXPECTED_RANKS or any(item.get("exit_code") != 0 for item in result.values())


def evidence_result(result: dict[int, dict]) -> dict:
    return {
        str(rank): {
            "exit_code": item["exit_code"],
            "stdout_bytes": len(item["stdout"].encode("utf-8")),
            "stdout_sha256": sha256_bytes(item["stdout"].encode("utf-8")),
            "stderr_bytes": len(item["stderr"].encode("utf-8")),
            "stderr_sha256": sha256_bytes(item["stderr"].encode("utf-8")),
            # Raw remote text can contain SSH targets, host paths, or daemon
            # diagnostics.  Even private evidence stores only commitments.
            "error_text_included": False,
        }
        for rank, item in sorted(result.items())
    }


def parse_equal_receipts(result: dict[int, dict]) -> tuple[dict, dict]:
    if phase_failed(result):
        raise LifecycleError("one or more checkpoint generators failed")
    receipts = {}
    for rank in sorted(EXPECTED_RANKS):
        item = result[rank]
        if item["stderr"] != "":
            raise LifecycleError(f"rank {rank} generator emitted stderr")
        try:
            receipt = strict_json_loads(item["stdout"], f"rank {rank} generator stdout")
        except LifecycleError as error:
            raise LifecycleError(f"rank {rank} generator stdout is not one JSON document: {error}") from error
        try:
            sparkcache_config.validate_checkpoint_receipt(receipt)
        except sparkcache_config.SparkCacheProfileError as error:
            raise LifecycleError(f"rank {rank} receipt is malformed: {error}") from error
        receipts[rank] = receipt
    baseline = canonical_bytes(receipts[0])
    for rank in (1, 2, 3):
        if canonical_bytes(receipts[rank]) != baseline:
            raise LifecycleError(f"rank {rank} receipt differs from rank 0")
    receipt = receipts[0]
    return receipt, {
        "canonical_receipt_sha256": sha256_bytes(baseline),
        "checkpoint_identity_sha256": receipt["checkpoint_identity_sha256"],
        "file_count": receipt["file_count"],
        "rank_canonical_sha256": {
            str(rank): sha256_bytes(canonical_bytes(value)) for rank, value in receipts.items()
        },
    }


def _phase_timeout(name: str, profile: exl3.Profile, hash_timeout: int) -> int:
    if name == "generate_receipts":
        return hash_timeout
    if name in ("start_engines", "engine_ready", "baseline_final"):
        return int(profile.startup_timeout_seconds) + 60
    if name in ("baseline_preflight", "server_health"):
        return 180
    if name == "quiescence":
        return 120
    return 300


def _record_remote_phase(
    reservation: dict,
    name: str,
    result: dict[int, dict],
    *,
    stage: str,
) -> None:
    evidence = reservation["evidence"]
    evidence["phases"].append(
        {
            "name": name,
            "stage": stage,
            "completed_at": utc_now(),
            "passed": not phase_failed(result),
            "ranks": evidence_result(result),
        }
    )
    evidence["execution_state"] = stage
    checkpoint_evidence(reservation)


def _record_local_phase(
    reservation: dict,
    name: str,
    *,
    passed: bool,
    detail: dict | None = None,
    error: str = "",
    stage: str,
) -> None:
    reservation["evidence"]["phases"].append(
        {
            "name": name,
            "stage": stage,
            "completed_at": utc_now(),
            "passed": passed,
            "detail": detail or {},
            "error": error,
        }
    )
    reservation["evidence"]["execution_state"] = stage
    checkpoint_evidence(reservation)


def _run_required_phase(
    name: str,
    phases: dict,
    profile: exl3.Profile,
    hash_timeout: int,
    executor: Callable,
    reservation: dict,
    *,
    stage: str,
) -> dict[int, dict]:
    result = execute_phase(
        executor,
        phases[name],
        timeout=_phase_timeout(name, profile, hash_timeout),
    )
    _record_remote_phase(reservation, name, result, stage=stage)
    if phase_failed(result):
        raise LifecycleError(f"phase {name} failed")
    return result


def restore_canonical(
    phases: dict,
    profile: exl3.Profile,
    hash_timeout: int,
    executor: Callable,
    reservation: dict,
) -> bool:
    """Make one complete exact restoration attempt and attest final health.

    Cleanup phases continue so a transient earlier failure cannot suppress a
    later exact quiescence proof.  Start is forbidden unless every cleanup and
    quiescence phase passed.  Any start or health failure makes restoration
    false; final health never upgrades an earlier failure to success.
    """
    evidence = reservation["evidence"]
    evidence["restoration"] = {
        "started_at": utc_now(),
        "passed": False,
        "phases": [],
    }
    checkpoint_evidence(reservation)
    cleanup_names = (
        "verified_start_prepare",
        "remove_helper",
        "stop_engines",
        "stop_servers",
        "quiescence",
    )
    cleanup_ok = True
    for name in cleanup_names:
        result = execute_phase(
            executor,
            phases[name],
            timeout=_phase_timeout(name, profile, hash_timeout),
        )
        passed = not phase_failed(result)
        evidence["restoration"]["phases"].append(
            {
                "name": name,
                "passed": passed,
                "ranks": evidence_result(result),
                "completed_at": utc_now(),
            }
        )
        checkpoint_evidence(reservation)
        cleanup_ok = cleanup_ok and passed
        if name == "verified_start_prepare" and not passed:
            evidence["restoration"]["terminal_before_removal"] = True
            checkpoint_evidence(reservation)
            return False
    if not cleanup_ok:
        evidence["restoration"]["start_skipped"] = "cleanup-or-quiescence-failed"
        checkpoint_evidence(reservation)
        return False

    start_names = (
        "start_servers",
        "server_health",
        "start_engines",
        "engine_ready",
        "server_health",
        "baseline_final",
        "helper_absent",
    )
    restored = True
    for name in start_names:
        result = execute_phase(
            executor,
            phases[name],
            timeout=_phase_timeout(name, profile, hash_timeout),
        )
        passed = not phase_failed(result)
        evidence["restoration"]["phases"].append(
            {
                "name": name,
                "passed": passed,
                "ranks": evidence_result(result),
                "completed_at": utc_now(),
            }
        )
        checkpoint_evidence(reservation)
        restored = restored and passed
        # An unhealthy server is not an authority for engine startup.
        if name == "server_health" and not passed and "start_engines" not in [
            item["name"] for item in evidence["restoration"]["phases"]
        ]:
            evidence["restoration"]["start_skipped"] = "server-health-failed"
            checkpoint_evidence(reservation)
            return False
    evidence["restoration"]["passed"] = restored
    evidence["restoration"]["completed_at"] = utc_now()
    checkpoint_evidence(reservation)
    return restored


def execute_transaction(
    phases: dict,
    profile: exl3.Profile,
    reservation: dict,
    *,
    hash_timeout: int,
    executor: Callable = exl3.execute,
) -> dict:
    """Execute the receipt transaction; restore on every post-stop BaseException."""
    interrupted = False
    primary_succeeded = False
    caught: BaseException | None = None
    try:
        # All three gates are terminal before removal.  Output ownership is
        # checked again after the remote prepare and immediately before stop.
        _run_required_phase(
            "baseline_preflight", phases, profile, hash_timeout, executor,
            reservation, stage="pre-stop",
        )
        _run_required_phase(
            "verified_start_prepare", phases, profile, hash_timeout, executor,
            reservation, stage="pre-stop",
        )
        _run_required_phase(
            "helper_absent", phases, profile, hash_timeout, executor,
            reservation, stage="pre-stop",
        )
        assert_receipt_reservation(reservation)
        _record_local_phase(
            reservation,
            "output_reservations_reverified",
            passed=True,
            detail={"receipt_reserved": True, "evidence_reserved": True},
            stage="pre-stop",
        )

        interrupted = True
        _run_required_phase(
            "stop_engines", phases, profile, hash_timeout, executor,
            reservation, stage="interrupted",
        )
        _run_required_phase(
            "stop_servers", phases, profile, hash_timeout, executor,
            reservation, stage="interrupted",
        )
        _run_required_phase(
            "quiescence", phases, profile, hash_timeout, executor,
            reservation, stage="interrupted",
        )
        generated = _run_required_phase(
            "generate_receipts", phases, profile, hash_timeout, executor,
            reservation, stage="interrupted",
        )
        receipt, equality = parse_equal_receipts(generated)
        _record_local_phase(
            reservation,
            "validate_equal_rank_receipts",
            passed=True,
            detail=equality,
            stage="interrupted",
        )
        receipt_summary = finalize_receipt(reservation, receipt)
        reservation["evidence"]["receipt"] = receipt_summary
        _record_local_phase(
            reservation,
            "finalize_checkpoint_manifest_v2",
            passed=True,
            detail=receipt_summary,
            stage="interrupted",
        )
        _run_required_phase(
            "remove_helper", phases, profile, hash_timeout, executor,
            reservation, stage="interrupted",
        )
        primary_succeeded = True
    except BaseException as error:
        caught = error

    restoration_ok = True
    if interrupted:
        try:
            restoration_ok = restore_canonical(
                phases, profile, hash_timeout, executor, reservation
            )
        except BaseException as restore_error:
            restoration_ok = False
            reservation["evidence"]["restoration"] = {
                "passed": False,
                "exception_type": type(restore_error).__name__,
                "exception_sha256": sha256_bytes(str(restore_error).encode("utf-8")),
                "completed_at": utc_now(),
            }
            try:
                checkpoint_evidence(reservation)
            except BaseException:
                pass
            if caught is None:
                caught = restore_error

    evidence = reservation["evidence"]
    evidence["completed_at"] = utc_now()
    evidence["passed"] = primary_succeeded and restoration_ok and caught is None
    if evidence["passed"]:
        evidence["execution_state"] = "completed-restored"
    elif interrupted and restoration_ok:
        evidence["execution_state"] = "failed-restored"
    elif interrupted:
        evidence["execution_state"] = "failed-restoration-unproven"
    else:
        evidence["execution_state"] = "failed-before-stop"
        evidence["terminal_before_removal"] = True
    if caught is not None:
        evidence["error"] = {
            "type": type(caught).__name__,
            "message_sha256": sha256_bytes(str(caught).encode("utf-8")),
            "message_bytes": len(str(caught).encode("utf-8")),
        }
    try:
        checkpoint_evidence(reservation)
    except BaseException as evidence_error:
        if caught is None:
            caught = evidence_error

    # Re-raise only after the restoration/final evidence attempt.  A failed
    # restoration overrides an otherwise successful primary transaction.
    if not restoration_ok:
        raise LifecycleError("canonical EXL3+LMCache restoration failed") from caught
    if caught is not None:
        raise caught
    if not primary_succeeded:
        raise LifecycleError("receipt transaction did not complete")
    return evidence


def build_plan(site, profile: exl3.Profile, phases: dict, generator_sha256: str) -> dict:
    lifecycle = [
        "reserve_private_outputs",
        "baseline_preflight",
        "verified_start_prepare",
        "helper_absent",
        "output_reservations_reverified",
        "stop_engines",
        "stop_servers",
        "quiescence",
        "generate_receipts",
        "validate_equal_rank_receipts",
        "finalize_checkpoint_manifest_v2",
        "remove_helper",
        "restore_canonical",
    ]
    return {
        "schema": PLAN_SCHEMA,
        "sensitivity": PRIVATE_SENSITIVITY,
        "lane": "public-functional",
        "maturity": "offline-validated",
        "command": "generate",
        "dry_run": True,
        "mutates_remote": True,
        "stops_serving": True,
        "confirmation_required": CONFIRMATION,
        "profile_id": profile.profile_id,
        "image_id": profile.image_id,
        "generator_sha256": generator_sha256,
        "required_ranks": [0, 1, 2, 3],
        "lifecycle": lifecycle,
        "phases": {name: render(actions) for name, actions in phases.items()},
        "rank_completeness": {
            "rank_ids": [rank.id for rank in site.ranks],
            "all_phases_exact_rank0_3": all(
                [action.rank for action in actions] == [0, 1, 2, 3]
                for actions in phases.values()
            ),
        },
        "safety": {
            "outputs_reserved_before_remote_action": True,
            "pre_stop_failure_is_terminal": True,
            "automatic_restore_after_any_post_stop_base_exception": True,
            "generator_gpu_access": False,
            "model_mount": "read-only",
            "receipt_requires_all_rank_canonical_equality": True,
            "raw_plan_and_evidence_publishable": False,
            "public_release_requires_closed_sanitizer": True,
        },
        "disclaimer": "A successful plan is not a receipt, live validation, or acceptance.",
    }


def main(argv: list[str] | None = None, *, executor: Callable = exl3.execute) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--receipt-output", required=True, type=Path)
    parser.add_argument("--evidence-output", required=True, type=Path)
    parser.add_argument("--hash-timeout-seconds", type=int, default=7200)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("command", choices=("plan", "generate"))
    args = parser.parse_args(argv)
    if args.hash_timeout_seconds < 300 or args.hash_timeout_seconds > 43200:
        parser.error("--hash-timeout-seconds must be between 300 and 43200")
    if args.command == "plan" and args.execute:
        parser.error("--execute does not apply to plan")
    if args.command == "plan" and args.confirmation:
        parser.error("--confirmation does not apply to plan")
    try:
        site = load_site(args.site)
        profile = exl3.load_profile(Path(args.profile))
        validate_site_and_profile(site, profile)
        generator_payload = GENERATOR_PATH.read_bytes()
        if b"\r\n" in generator_payload or not generator_payload.startswith(b"#!/usr/bin/env python3\n"):
            raise LifecycleError("published checkpoint generator must be LF-only Python")
        generator_sha256 = sha256_bytes(generator_payload)
        phases = build_phases(site, profile, generator_payload)
        plan = build_plan(site, profile, phases, generator_sha256)
    except (
        OSError,
        KeyError,
        json.JSONDecodeError,
        SiteConfigError,
        exl3.ProfileError,
        sparkcache_config.SparkCacheProfileError,
        sparkcache_launcher.LauncherError,
        LifecycleError,
    ) as error:
        parser.error(str(error))

    if args.command == "plan" or not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if args.confirmation != CONFIRMATION:
        parser.error(f"execute requires --confirmation {CONFIRMATION}")

    run_id = secrets.token_hex(16)
    try:
        # This is intentionally before baseline_preflight: no remote action,
        # including read-only SSH, starts until both local outputs are owned.
        reservation = reserve_outputs(
            args.receipt_output, args.evidence_output, run_id
        )
        reservation["evidence"].update(
            {
                "profile_id": profile.profile_id,
                "image_id": profile.image_id,
                "generator_sha256": generator_sha256,
                "required_ranks": [0, 1, 2, 3],
            }
        )
        checkpoint_evidence(reservation)
        evidence = execute_transaction(
            phases,
            profile,
            reservation,
            hash_timeout=args.hash_timeout_seconds,
            executor=executor,
        )
    except (OSError, LifecycleError) as error:
        print(f"checkpoint receipt lifecycle failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema": EVIDENCE_SCHEMA,
                "run_id": run_id,
                "passed": evidence["passed"],
                "execution_state": evidence["execution_state"],
                "receipt": evidence["receipt"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
