#!/usr/bin/env python3
"""Dry-run-first LMCache NVMe canary for the accepted R7 operator profile.

The canary derives its engine profile from an exact, hash-pinned running-profile
receipt. It changes only the container identity, CUDA allocator stability
setting, LMCache connector configuration, and evidence labels. Four rank-local
LMCache MP servers use a 0.5 GB lazy L1 and a bounded 50 GB native-filesystem
L2 tier. Rollback restores the hash-pinned base engine profile and deliberately
preserves the L2 directories for inspection or a persistence replay.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sparkring_generic_launcher as generic  # noqa: E402
import sparkring_runtime as runtime  # noqa: E402
from sparkring_site import SiteConfigError, load_site  # noqa: E402


BASE_PROFILE_SHA256 = "47a9f825b00c0399fcdf5313c2bf3cee00feb775c0a02c5bbb6d888fb609add1"
SITE_SHA256 = "a1c62b5b42c98d75830a8a30ef71c33953fd8d28bd4dce28aecd0d133e81fe4c"
IMAGE_ID = "sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513"
BASE_PROFILE_ID = "glm52-exl3-r7-target-exact-q40-block8"
CANARY_PROFILE_ID = "glm52-exl3-r7-q40-lmcache-nvme-canary"
CANARY_CONTAINER = "glm52-sparkring-q40-lmcache-nvme-canary"
SERVER_PREFIX = "glm52-sparkring-lmcache-r7-nvme-server"
CONFIRMATION = "START-R7-LMCACHE-NVME-CANARY-ALL-FOUR"

SERVER_PORT = 6556
HTTP_PORT = 18081
CHUNK_SIZE = 512
L1_SIZE_GB = 0.5
L2_CAPACITY_GB = 50
L2_HOST_PATH = "/var/lib/sparkring/context-cache/lmcache-r7-nvme-canary"
L2_CONTAINER_PATH = "/cache/lmcache-r7-l2"
CANARY_JIT_HOST_PATH = "/var/tmp/sparkring-r7-jit-lmcache-canary"

INSTALLED_LMCACHE_VERSION = "0.5.2+glm52dcp4.1"
INSTALLED_IMAGE_LABELS = {
    "org.sparkring.lmcache-integration-tree": "a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa",
    "org.sparkring.lmcache-composed-tree": "7dddbfde874d123e5b5785e6e56b4b7baf4baa82",
    "org.sparkring.lmcache-topology-patch-sha256": "6eb5d3c15cd67a62dca5714fabc4d9c12c7175dfb9ab3e971dc6bc280f0aa533",
}


class CanaryError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, role: str) -> None:
    observed = file_sha256(path)
    if observed != expected:
        raise CanaryError(
            f"{role} SHA-256 drifted: observed {observed}, expected {expected}"
        )


def server_name(rank: int) -> str:
    return f"{SERVER_PREFIX}-r{rank}"


def shell_action(rank: Any, script: str) -> runtime.RemoteAction:
    return runtime.RemoteAction(rank.id, rank.ssh_target, ("sh", "-lc", script))


def l2_adapter_json() -> str:
    return json.dumps(
        {
            "type": "fs_native",
            "base_path": L2_CONTAINER_PATH,
            "num_workers": 2,
            "use_odirect": True,
            "max_capacity_gb": L2_CAPACITY_GB,
            "eviction": {
                "eviction_policy": "LRU",
                "trigger_watermark": 0.85,
                "eviction_ratio": 0.15,
            },
        },
        separators=(",", ":"),
    )


def derive_canary_profile(
    base: runtime.RuntimeProfile, site: Any
) -> runtime.RuntimeProfile:
    if base.source_schema != runtime.SCHEMA:
        raise CanaryError("base profile must use sparkring-runtime-profile/v1")
    if base.profile_id != BASE_PROFILE_ID:
        raise CanaryError(f"base profile must be {BASE_PROFILE_ID}")
    if base.image_id != IMAGE_ID:
        raise CanaryError(f"base image must be {IMAGE_ID}")

    document = copy.deepcopy(base.document)
    document["profile_id"] = CANARY_PROFILE_ID
    document["container_name"] = CANARY_CONTAINER
    document["confirmation"] = CONFIRMATION
    document["environment"]["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:False"
    )
    document["environment"]["LMCACHE_DISABLE_BANNER"] = "1"
    for volume in document["extra_volumes"]:
        if volume["container"] == "/cache/jit":
            volume["host"] = CANARY_JIT_HOST_PATH
            break
    else:
        raise CanaryError("base profile has no /cache/jit volume")
    document["extra_labels"].update(
        {
            "org.sparkring.evidence-maturity": "candidate",
            "org.sparkring.experiment": "lmcache-r7-nvme-canary",
            "org.sparkring.lmcache-l1-gb": str(L1_SIZE_GB),
            "org.sparkring.lmcache-l2-gb": str(L2_CAPACITY_GB),
            "org.sparkring.lmcache-l2-type": "fs_native-odirect",
            "org.sparkring.lmcache-version": INSTALLED_LMCACHE_VERSION,
        }
    )
    urls = [
        f"tcp://{rank.management.address}:{SERVER_PORT}" for rank in site.ranks
    ]
    connector = {
        "kv_connector": "LMCacheMPConnector",
        "kv_connector_module_path": (
            "lmcache.integration.vllm.lmcache_mp_connector"
        ),
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "lmcache.mp.server_urls": urls,
            "lmcache.mp.mq_timeout": 10.0,
            "lmcache.mp.heartbeat_interval": 10.0,
        },
    }
    document["extra_vllm_args"].extend(
        [
            "--kv-transfer-config",
            json.dumps(connector, separators=(",", ":")),
        ]
    )
    return runtime.parse_runtime_profile(document, source="derived-r7-lmcache-canary")


def image_guard(profile: runtime.RuntimeProfile) -> str:
    checks = [
        f"test \"$({profile.engine} image inspect --format '{{{{.Id}}}}' "
        f"{shlex.quote(profile.image)})\" = {shlex.quote(IMAGE_ID)}"
    ]
    for label, expected in INSTALLED_IMAGE_LABELS.items():
        checks.append(
            f"test \"$({profile.engine} image inspect --format "
            f"'{{{{index .Config.Labels {json.dumps(label)}}}}}' "
            f"{shlex.quote(IMAGE_ID)})\" = {shlex.quote(expected)}"
        )
    return " && ".join(checks)


def precheck_actions(
    site: Any, base: runtime.RuntimeProfile, canary: runtime.RuntimeProfile
) -> list[runtime.RemoteAction]:
    actions = []
    required_kib = (L2_CAPACITY_GB + 10) * 1024 * 1024
    for rank in site.ranks:
        base_name = runtime.container_name(base, rank.id)
        canary_name = runtime.container_name(canary, rank.id)
        server = server_name(rank.id)
        script = (
            f"{image_guard(base)}"
            f" && test \"$({base.engine} inspect --format '{{{{.Image}}}}' "
            f"{shlex.quote(base_name)})\" = {shlex.quote(IMAGE_ID)}"
            f" && test \"$({base.engine} inspect --format "
            f"'{{{{index .Config.Labels \"org.sparkring.profile\"}}}}' "
            f"{shlex.quote(base_name)})\" = {shlex.quote(BASE_PROFILE_ID)}"
            f" && test -z \"$({base.engine} ps -aq --filter "
            f"name=^/{shlex.quote(canary_name)}$)\""
            f" && test -z \"$({base.engine} ps -aq --filter "
            f"name=^/{shlex.quote(server)}$)\""
            f" && test $(df -Pk {shlex.quote(L2_HOST_PATH.rsplit('/', 1)[0])} "
            f"| awk 'NR==2 {{print $4}}') -ge {required_kib}"
            f" && test -z \"$(ss -ltnH "
            f"'( sport = :{SERVER_PORT} or sport = :{HTTP_PORT} )' 2>/dev/null)\""
        )
        actions.append(shell_action(rank, script))
    return actions


def server_start_actions(
    site: Any, profile: runtime.RuntimeProfile
) -> list[runtime.RemoteAction]:
    actions = []
    adapter = l2_adapter_json()
    for rank in site.ranks:
        name = server_name(rank.id)
        command = [
            profile.engine,
            "run",
            "--detach",
            "--name",
            name,
            "--label",
            "org.sparkring.managed=true",
            "--label",
            f"org.sparkring.profile={CANARY_PROFILE_ID}",
            "--label",
            "org.sparkring.component=lmcache-server",
            "--label",
            f"org.sparkring.rank={rank.id}",
            "--privileged",
            "--gpus",
            "all",
            "--network",
            "host",
            "--ipc",
            "host",
            "--shm-size",
            profile.shm_size,
            "--ulimit",
            "memlock=-1:-1",
            "--volume",
            f"{L2_HOST_PATH}:{L2_CONTAINER_PATH}:rw",
            "--env",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False",
            "--env",
            "LMCACHE_DISABLE_BANNER=1",
            "--entrypoint",
            "/opt/venv/bin/lmcache",
            IMAGE_ID,
            "server",
            "--instance-id",
            f"spark-r{rank.id}-r7-nvme-cs512",
            "--host",
            "0.0.0.0",
            "--port",
            str(SERVER_PORT),
            "--chunk-size",
            str(CHUNK_SIZE),
            "--max-gpu-workers",
            "1",
            "--max-cpu-workers",
            "1",
            "--supported-transfer-mode",
            "lmcache_driven",
            "--l1-size-gb",
            str(L1_SIZE_GB),
            "--l1-init-size-gb",
            "0",
            "--l1-use-lazy",
            "--eviction-policy",
            "LRU",
            "--eviction-trigger-watermark",
            "0.85",
            "--eviction-ratio",
            "0.15",
            "--l2-adapter",
            adapter,
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(HTTP_PORT),
            "--worker-reap-timeout-seconds",
            "120",
            "--worker-registration-grace-seconds",
            "3600",
            "--disable-observability",
        ]
        guard = (
            image_guard(profile)
            + f" && test -z \"$({profile.engine} ps -aq --filter "
            f"name=^/{shlex.quote(name)}$)\""
            + f" && exec {shlex.join(command)}"
        )
        actions.append(shell_action(rank, guard))
    return actions


def server_health_actions(
    site: Any, profile: runtime.RuntimeProfile
) -> list[runtime.RemoteAction]:
    actions = []
    for rank in site.ranks:
        name = server_name(rank.id)
        script = (
            f"test \"$({profile.engine} inspect --format "
            f"'{{{{index .Config.Labels \"org.sparkring.profile\"}}}}' "
            f"{shlex.quote(name)})\" = {shlex.quote(CANARY_PROFILE_ID)}"
            f" && for i in $(seq 1 90); do "
            f"curl -fsS http://127.0.0.1:{HTTP_PORT}/healthcheck >/dev/null "
            "&& break; sleep 2; done"
            f" && curl -fsS http://127.0.0.1:{HTTP_PORT}/healthcheck >/dev/null"
            f" && curl -fsS http://127.0.0.1:{HTTP_PORT}/status"
            f" && test \"$({profile.engine} inspect --format "
            f"'{{{{.State.Running}}}}' {shlex.quote(name)})\" = true"
            f" && test \"$({profile.engine} inspect --format "
            f"'{{{{.State.OOMKilled}}}}' {shlex.quote(name)})\" = false"
        )
        actions.append(shell_action(rank, script))
    return actions


def server_stop_actions(
    site: Any, profile: runtime.RuntimeProfile
) -> list[runtime.RemoteAction]:
    actions = []
    for rank in site.ranks:
        name = server_name(rank.id)
        script = (
            f"listing=$({profile.engine} ps -a --filter "
            f"name=^/{shlex.quote(name)}$ --format '{{{{.Names}}}}')"
            "; test -z \"$listing\" && exit 0"
            f"; test \"$listing\" = {shlex.quote(name)} || exit 74"
            f"; test \"$({profile.engine} inspect --format "
            f"'{{{{index .Config.Labels \"org.sparkring.managed\"}}}}' "
            f"{shlex.quote(name)})\" = true || exit 73"
            f"; test \"$({profile.engine} inspect --format "
            f"'{{{{index .Config.Labels \"org.sparkring.profile\"}}}}' "
            f"{shlex.quote(name)})\" = {shlex.quote(CANARY_PROFILE_ID)} || exit 73"
            f"; exec {profile.engine} rm --force {shlex.quote(name)}"
        )
        actions.append(shell_action(rank, script))
    return actions


def engine_ready_actions(
    site: Any, profile: runtime.RuntimeProfile
) -> list[runtime.RemoteAction]:
    actions = []
    for rank in site.ranks:
        name = runtime.container_name(profile, rank.id)
        api_check = (
            f" && curl -fsS http://127.0.0.1:{site.serving.api_port}/health "
            ">/dev/null"
            if rank.id == site.serving.master_rank
            else ""
        )
        script = (
            "for i in $(seq 1 900); do "
            f"test \"$({profile.engine} inspect --format '{{{{.State.Running}}}}' "
            f"{shlex.quote(name)} 2>/dev/null)\" = true"
            + api_check
            + " && break; sleep 2; done"
            f" && test \"$({profile.engine} inspect --format "
            f"'{{{{.State.Running}}}}' {shlex.quote(name)})\" = true"
            f" && test \"$({profile.engine} inspect --format "
            f"'{{{{.State.OOMKilled}}}}' {shlex.quote(name)})\" = false"
            + api_check
            + f" && ! {profile.engine} logs {shlex.quote(name)} 2>&1 | "
            "grep -E 'Traceback|CUDA out of memory|OutOfMemoryError|OOMKilled'"
        )
        actions.append(shell_action(rank, script))
    return actions


def canary_receipt_cleanup_actions(
    site: Any, profile: runtime.RuntimeProfile
) -> list[runtime.RemoteAction]:
    """Remove one-shot Q40 receipts while preserving the warm JIT cache.

    The exact-state runtime writes a rank-local attestation receipt during
    startup and deliberately refuses to overwrite it. A canary restart must
    remove only that receipt before recreating the engine; compiled artifacts
    in the same isolated mount remain reusable.
    """

    actions = []
    for rank in site.ranks:
        filename = f"q40-exact-state-serving-v1-rank{rank.id}.json"
        receipt = f"{CANARY_JIT_HOST_PATH}/{filename}"
        script = (
            f"test -d {shlex.quote(CANARY_JIT_HOST_PATH)}"
            f" && {profile.engine} run --rm"
            f" --volume {shlex.quote(f'{CANARY_JIT_HOST_PATH}:/cache/jit:rw')}"
            " --entrypoint /bin/rm"
            f" {shlex.quote(profile.image_id)}"
            f" -f -- {shlex.quote(f'/cache/jit/{filename}')}"
            f" && test ! -e {shlex.quote(receipt)}"
        )
        actions.append(shell_action(rank, script))
    return actions


def base_restore_actions(
    site: Any, base: runtime.RuntimeProfile
) -> list[runtime.RemoteAction]:
    """Start a missing base rank or accept its exact running container.

    A forward failure can occur after only some ranks stop. The normal generic
    start action is intentionally non-idempotent, so rollback wraps it with an
    exact-name and exact-profile check before deciding whether a rank needs to
    be recreated.
    """

    actions = []
    for rank, start in zip(site.ranks, runtime.start_actions(site, base)):
        name = runtime.container_name(base, rank.id)
        start_script = start.argv[-1]
        script = (
            f"listing=$({base.engine} ps -a --filter "
            f"name=^/{shlex.quote(name)}$ --format '{{{{.Names}}}}')"
            "; if test -n \"$listing\"; then"
            f" test \"$listing\" = {shlex.quote(name)} || exit 74"
            f"; test \"$({base.engine} inspect --format "
            f"'{{{{index .Config.Labels \"org.sparkring.managed\"}}}}' "
            f"{shlex.quote(name)})\" = true || exit 73"
            f"; test \"$({base.engine} inspect --format "
            f"'{{{{index .Config.Labels \"org.sparkring.profile\"}}}}' "
            f"{shlex.quote(name)})\" = {shlex.quote(BASE_PROFILE_ID)} || exit 73"
            f"; test \"$({base.engine} inspect --format '{{{{.State.Running}}}}' "
            f"{shlex.quote(name)})\" = true || exit 74"
            "; exit 0; fi"
            f"; {start_script}"
        )
        actions.append(shell_action(rank, script))
    return actions


def build_phases(
    site: Any, base: runtime.RuntimeProfile, canary: runtime.RuntimeProfile
) -> dict[str, list[runtime.RemoteAction]]:
    return {
        "precheck": precheck_actions(site, base, canary),
        "stop_base_engines": runtime.stop_actions(site, base),
        "start_servers": server_start_actions(site, canary),
        "server_health": server_health_actions(site, canary),
        "clean_canary_receipts": canary_receipt_cleanup_actions(site, canary),
        "start_canary_engines": runtime.start_actions(site, canary),
        "canary_ready": engine_ready_actions(site, canary),
        "stop_canary_engines": runtime.stop_actions(site, canary),
        "stop_servers": server_stop_actions(site, canary),
        "restart_base_engines": base_restore_actions(site, base),
        "base_ready": engine_ready_actions(site, base),
    }


def render_plan(
    phases: dict[str, list[runtime.RemoteAction]],
    base_path: Path,
    site_path: Path,
    canary: runtime.RuntimeProfile,
) -> dict[str, Any]:
    return {
        "schema": "sparkring-r7-lmcache-nvme-canary-plan/v1",
        "lane": "public-functional",
        "maturity": "candidate",
        "hardware": "four directly cabled NVIDIA DGX Sparks / GB10 GPUs",
        "base_profile": {
            "path": str(base_path),
            "sha256": BASE_PROFILE_SHA256,
            "profile_id": BASE_PROFILE_ID,
            "image_id": IMAGE_ID,
        },
        "site": {"path": str(site_path), "sha256": SITE_SHA256},
        "canary": {
            "profile_id": canary.profile_id,
            "lmcache_version": INSTALLED_LMCACHE_VERSION,
            "chunk_size": CHUNK_SIZE,
            "l1_size_gb_per_rank": L1_SIZE_GB,
            "l1_init_size_gb_per_rank": 0,
            "l2_type": "fs_native",
            "l2_capacity_gb_per_rank": L2_CAPACITY_GB,
            "l2_use_odirect": True,
            "l2_host_path": L2_HOST_PATH,
            "rollback_preserves_l2": True,
        },
        "safety": {
            "start_stops_serving": True,
            "confirmation": CONFIRMATION,
            "rollback_restores_exact_base_profile": True,
        },
        "forward_sequence": [
            "precheck",
            "stop_base_engines",
            "start_servers",
            "server_health",
            "clean_canary_receipts",
            "start_canary_engines",
            "canary_ready",
            "server_health",
        ],
        "rollback_sequence": [
            "stop_canary_engines",
            "stop_servers",
            "restart_base_engines",
            "base_ready",
        ],
        "phases": {
            name: runtime.render(actions) for name, actions in phases.items()
        },
    }


def phase_failed(results: dict[int, dict]) -> bool:
    return any(result.get("exit_code") != 0 for result in results.values())


def run_sequence(
    phases: dict[str, list[runtime.RemoteAction]], sequence: list[str]
) -> tuple[dict[str, Any], str | None]:
    results: dict[str, Any] = {}
    for name in sequence:
        timeout = 1900 if name.endswith("ready") else 240
        results[name] = runtime.execute(phases[name], timeout=timeout)
        if phase_failed(results[name]):
            return results, name
    return results, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True)
    parser.add_argument("--base-profile", required=True)
    parser.add_argument("--output-profile")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "command", choices=("plan", "precheck", "start", "status", "rollback")
    )
    args = parser.parse_args(argv)

    site_path = Path(args.site)
    base_path = Path(args.base_profile)
    try:
        require_hash(site_path, SITE_SHA256, "site")
        require_hash(base_path, BASE_PROFILE_SHA256, "base profile")
        site = load_site(site_path)
        base = generic.load_profile(base_path)
        canary = derive_canary_profile(base, site)
        phases = build_phases(site, base, canary)
    except (
        OSError,
        KeyError,
        json.JSONDecodeError,
        SiteConfigError,
        runtime.ProfileError,
        CanaryError,
    ) as exc:
        parser.error(str(exc))

    if args.output_profile:
        Path(args.output_profile).write_text(
            json.dumps(canary.document, indent=2) + "\n", encoding="utf-8"
        )

    plan = render_plan(phases, base_path, site_path, canary)
    if args.command == "plan" or not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if args.command in ("start", "rollback") and args.confirmation != CONFIRMATION:
        parser.error(f"execute requires --confirmation {CONFIRMATION}")

    if args.command == "precheck":
        results, failed_phase = run_sequence(phases, ["precheck"])
        print(json.dumps({"plan": plan, "results": results}, indent=2))
        return 1 if failed_phase else 0

    if args.command == "status":
        results, failed_phase = run_sequence(
            phases, ["server_health", "canary_ready"]
        )
        print(json.dumps({"plan": plan, "results": results}, indent=2))
        return 1 if failed_phase else 0

    if args.command == "rollback":
        results, failed_phase = run_sequence(
            phases,
            [
                "stop_canary_engines",
                "stop_servers",
                "restart_base_engines",
                "base_ready",
            ],
        )
        print(json.dumps({"plan": plan, "results": results}, indent=2))
        return 1 if failed_phase else 0

    results, failed_phase = run_sequence(
        phases,
        [
            "precheck",
            "stop_base_engines",
            "start_servers",
            "server_health",
            "clean_canary_receipts",
            "start_canary_engines",
            "canary_ready",
            "server_health",
        ],
    )
    if failed_phase is not None:
        rollback_results, rollback_failure = run_sequence(
            phases,
            [
                "stop_canary_engines",
                "stop_servers",
                "restart_base_engines",
                "base_ready",
            ],
        )
        results["automatic_rollback"] = rollback_results
        results["automatic_rollback_failed_phase"] = rollback_failure
    print(
        json.dumps(
            {"plan": plan, "failed_phase": failed_phase, "results": results},
            indent=2,
        )
    )
    return 1 if failed_phase else 0


if __name__ == "__main__":
    raise SystemExit(main())
