#!/usr/bin/env python3
"""Build once and deploy the pinned EXL3 profile on a four-Spark ring."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import posixpath
import shlex
import shutil
import sys
from pathlib import Path

import yaml

from bootstrap_nf3 import (
    checkout_command,
    direct_image_fanout,
    early_fabric_preflight_command,
    fanout_image_archive,
    git_value,
    image_id,
    remote,
    run,
    ssh_bootstrap_verification_command,
    ssh_image_fanout_verification_command,
)
from sparkring_site import SiteConfig, load_site
from verify_ssh_mesh import split_ssh_target


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "recipes/glm52-exl3-tr3-3.25bpw.json"
CONFIRMATION = "BOOTSTRAP-EXL3-ALL-FOUR"
MODEL_CONTAINER_PATH = "/models/glm52-exl3-tr3-3.25bpw"
MODEL_HEADROOM_BYTES = 16 * 1024**3


def load_contract() -> dict:
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    if recipe.get("schema") != "sparkring-recipe/v1" or recipe.get("recipe_id") != "glm52-exl3-tr3-3.25bpw":
        raise RuntimeError("wrong EXL3 recipe contract")
    return recipe


def write_generated_site(source: Path, destination: Path, image: str, digest: str) -> None:
    recipe = load_contract()
    model = recipe["model"]
    serving = recipe["serving"]
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["runtime"].update(
        {
            "container_image": image,
            "container_image_digest": digest,
            "model_path": MODEL_CONTAINER_PATH,
            "model_repo": model["repository"],
            "model_revision": model["revision"],
            "checkpoint_sha256": model["index_sha256"],
        }
    )
    document["serving"].update(
        {
            "tensor_parallel_size": 4,
            "decode_context_parallel_size": 4,
            "mtp_mode": "static",
            "mtp_tokens": 2,
            "max_model_len": serving["max_model_len"],
            "kv_cache_bytes_per_rank": serving["kv_cache_bytes_per_rank"],
            "max_num_seqs": serving["max_num_seqs"],
        }
    )
    document["artifacts"] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def write_generated_profile(destination: Path, image: str, digest: str, model_path: str, jit_cache: str) -> None:
    recipe = load_contract()
    model = recipe["model"]
    serving = recipe["serving"]
    document = {
        "schema": "sparkring-public-exl3-launch/v1",
        "profile_id": "glm52-exl3-tr3-3.25bpw-lmcache-cs512",
        "engine": "docker",
        "container_name": "glm52-sparkring-exl3-lmcache-cs512",
        "image": image,
        "image_id": digest,
        "model_host_path": model_path,
        "model_container_path": MODEL_CONTAINER_PATH,
        "jit_cache_host_path": jit_cache,
        "model_repository": model["repository"],
        "model_revision": model["revision"],
        "model_config_sha256": model["config_sha256"],
        "model_index_sha256": model["index_sha256"],
        "model_tier_bitmap_sha256": model["tier_bitmap_sha256"],
        "model_manifest_sha256": model["manifest_sha256"],
        "model_shard_count": model["shard_count"],
        "model_weight_bytes": model["weight_bytes"],
        "shm_size": "16g",
        "startup_timeout_seconds": 3600,
        "environment": serving["environment"],
        "extra_vllm_args": serving["vllm_args"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def model_capacity_command(model_path: str, repository_bytes: int) -> str:
    required_total = repository_bytes + MODEL_HEADROOM_BYTES
    parent = posixpath.dirname(model_path.rstrip("/")) or "/"
    return "\n".join(
        (
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(parent)}",
            f"free=$(df -PB1 {shlex.quote(parent)} | awk 'NR==2 {{print $4}}')",
            f"existing=$(du -sb {shlex.quote(model_path)} 2>/dev/null | awk '{{print $1}}' || printf 0)",
            f"remaining=$(({repository_bytes} - existing))",
            "if [ \"$remaining\" -lt 0 ]; then remaining=0; fi",
            f"required=$((remaining + {MODEL_HEADROOM_BYTES}))",
            "test \"$free\" -ge \"$required\" || { echo \"insufficient model space: need $required, have $free\" >&2; exit 4; }",
            f"mkdir -p {shlex.quote(model_path)}",
            f"printf 'MODEL_CAPACITY_OK total={required_total} remaining=%s free=%s\\n' \"$remaining\" \"$free\"",
        )
    )


def direct_rsync_target(site: SiteConfig, destination_rank: int, destination_address: str, model_path: str) -> str:
    user, _ = split_ssh_target(site.rank(destination_rank).ssh_target)
    return f"{user}@{destination_address}:{model_path.rstrip('/')}/"


def rsync_command(source: str, destination: str) -> list[str]:
    return [
        "rsync", "--archive", "--partial", "--inplace", "--human-readable",
        "--info=progress2", "--protect-args", f"{source.rstrip('/')}/", destination,
    ]


def remote_bootstrap_dependencies_command() -> str:
    """Fail before expensive work when a follower cannot receive/verify payloads."""
    return "\n".join(
        (
            "set -euo pipefail",
            "for program in docker git python3 rsync sha256sum; do",
            "  command -v \"$program\" >/dev/null || { echo \"missing required program: $program\" >&2; exit 5; }",
            "done",
            "docker version >/dev/null",
            "python3 -c 'import yaml'",
            "printf 'EXL3_BOOTSTRAP_DEPENDENCIES_OK\\n'",
        )
    )


def local_uid_gid() -> tuple[int, int]:
    if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
        raise RuntimeError("containerized EXL3 download requires a POSIX rank-0 host")
    return os.getuid(), os.getgid()


def container_download_command(image: str, model_path: str) -> list[str]:
    """Use the built ARM64 runtime so the host needs no HF Python packages."""
    model_parent = posixpath.dirname(model_path.rstrip("/")) or "/"
    uid, gid = local_uid_gid()
    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{uid}:{gid}",
        "--env",
        "HOME=/tmp",
        "--env",
        "HF_HOME=/tmp/sparkring-huggingface",
        "--env",
        "HF_HUB_OFFLINE=0",
    ]
    if os.environ.get("HF_TOKEN"):
        command.extend(("--env", "HF_TOKEN"))
    command.extend(
        (
            "--volume",
            f"{ROOT}:/opt/sparkring-public:ro",
            "--volume",
            f"{model_parent}:{model_parent}",
            "--entrypoint",
            "/opt/venv/bin/python",
            image,
            "/opt/sparkring-public/scripts/download_exl3.py",
            "download",
            "--model-path",
            model_path,
        )
    )
    return command


def fanout_model(site: SiteConfig, model_path: str, repository_bytes: int, remote_repository: str) -> None:
    first_wave, relay = direct_image_fanout(site)
    for rank in site.ranks:
        if rank.id != 0:
            remote(rank.ssh_target, model_capacity_command(model_path, repository_bytes))
    print("==> Fanning model over direct 200GbE ring (resumable rsync)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        for hop in first_wave:
            target = direct_rsync_target(site, hop.destination_rank, hop.destination_address, model_path)
            futures.append(pool.submit(run, rsync_command(model_path, target)))
        for future in futures:
            future.result()
    relay_user, _ = split_ssh_target(site.rank(relay.destination_rank).ssh_target)
    relay_target = f"{relay_user}@{relay.destination_address}:{model_path.rstrip('/')}/"
    relay_script = shlex.join(rsync_command(model_path, relay_target))
    remote(site.rank(relay.source_rank).ssh_target, relay_script)
    verify = f"python3 {shlex.quote(remote_repository + '/scripts/download_exl3.py')} verify --model-path {shlex.quote(model_path)}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        checks = [pool.submit(remote, rank.ssh_target, verify) for rank in site.ranks if rank.id != 0]
        for check in checks:
            check.result()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("plan", "execute"))
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--model-host-path", default="/srv/models/GLM-5.2-EXL3-TR3-3.25bpw")
    parser.add_argument("--generated-site", type=Path, default=ROOT / ".sparkring/bootstrap-exl3/site.yaml")
    parser.add_argument("--generated-profile", type=Path, default=ROOT / ".sparkring/bootstrap-exl3/launch.json")
    parser.add_argument("--remote-root", default="/var/tmp/sparkring-bootstrap")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args(argv)

    site = load_site(args.site)
    recipe = load_contract()
    commit = git_value("rev-parse", "HEAD")
    dirty = git_value("status", "--porcelain")
    image = f"sparkring/glm52-exl3-tr3-3.25bpw:{commit[:12]}"
    base_image = f"sparkring/glm52-nf3-nvfp4-rope8:{commit[:12]}"
    plan = {
        "schema": "sparkring-exl3-bootstrap-plan/v1",
        "runs_on": "rank0",
        "source_commit": commit,
        "model": f"{recipe['model']['repository']}@{recipe['model']['revision']}",
        "model_path": args.model_host_path,
        "base_image": base_image,
        "image": image,
        "launch_after_prepare": not args.no_launch,
        "steps": [
            "verify SSH mesh and direct 200GbE/RDMA fabric",
            "build or reuse the receipt-gated public NF3 base on rank 0",
            "adopt or resume the exact 81-shard model on rank 0",
            "fan model bytes over the direct 200GbE ring",
            "reconstruct exact EXL3 and LMCache source trees and build one derived ARM64 image",
            "fan the exact image ID over the direct 200GbE ring",
            (
                "generate ignored site/profile files, verify, and launch four ranks"
                if not args.no_launch
                else "generate ignored site/profile files and verify without launching"
            ),
        ],
    }
    print(json.dumps(plan, indent=2))
    if args.action == "plan":
        return 0
    if args.confirmation != CONFIRMATION:
        parser.error(f"execute requires --confirmation {CONFIRMATION}")
    if dirty:
        parser.error("checkout is dirty; bootstrap receipts require a clean commit")
    if os.uname().machine != "aarch64":
        parser.error("execute must run natively on rank 0 (aarch64)")
    for program in ("docker", "git", "rsync"):
        if not shutil.which(program):
            parser.error(f"{program} is required")

    run(ssh_bootstrap_verification_command(args.site), cwd=ROOT)
    run(ssh_image_fanout_verification_command(args.site), cwd=ROOT)
    run(early_fabric_preflight_command(args.site), cwd=ROOT)
    followers = [rank for rank in site.ranks if rank.id != 0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        dependency_checks = [
            pool.submit(
                remote,
                rank.ssh_target,
                remote_bootstrap_dependencies_command(),
            )
            for rank in followers
        ]
        for dependency_check in dependency_checks:
            dependency_check.result()
    remote_repository = f"{args.remote_root}/{commit}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        checkouts = [pool.submit(remote, rank.ssh_target, checkout_command(commit, args.remote_root)) for rank in followers]
        for checkout in checkouts:
            checkout.result()

    environment = os.environ.copy()
    environment["OUTPUT_IMAGE"] = base_image
    run(["bash", str(ROOT / "scripts/build-nf3-nvfp4-rope8-image.sh")], cwd=ROOT, env=environment)
    base_id = image_id(base_image)
    Path(args.model_host_path).parent.mkdir(parents=True, exist_ok=True)
    run(container_download_command(base_image, args.model_host_path), cwd=ROOT)
    fanout_model(site, args.model_host_path, int(recipe["model"]["repository_bytes"]), remote_repository)

    cache = Path(os.environ.get("SPARKRING_BOOTSTRAP_CACHE", Path.home() / ".cache/sparkring/exl3-bootstrap"))
    source_cache = cache / "sources"
    context = cache / "contexts" / commit
    if context.exists():
        run([sys.executable, str(ROOT / "runtime/exl3/verify_build_context.py"), "--context", str(context)], cwd=ROOT)
    else:
        run([sys.executable, str(ROOT / "runtime/exl3/prepare_context.py"), "--source-root", str(source_cache), "--output", str(context)], cwd=ROOT)
    build_env = os.environ.copy()
    build_env.update({"SPARKRING_EXL3_BASE_IMAGE": base_image, "SPARKRING_EXL3_BASE_IMAGE_ID": base_id})
    run(["bash", str(context / "build-image.sh"), str(context), image], cwd=ROOT, env=build_env)
    expected_id = image_id(image)

    archive = cache / f"{image.replace('/', '_').replace(':', '_')}.tar"
    marker = archive.with_suffix(".image-id")
    if not archive.is_file() or not marker.is_file() or marker.read_text(encoding="utf-8").strip() != expected_id:
        archive.parent.mkdir(parents=True, exist_ok=True)
        run(["docker", "save", "--output", str(archive), image])
        marker.write_text(expected_id + "\n", encoding="utf-8")
    fanout_image_archive(site, image, expected_id, archive)

    write_generated_site(args.site, args.generated_site, image, expected_id)
    write_generated_profile(args.generated_profile, image, expected_id, args.model_host_path, str(site.paths.jit_cache_dir))
    launcher = [sys.executable, str(ROOT / "scripts/sparkring_exl3_lmcache_launcher.py"), "--site", str(args.generated_site), "--profile", str(args.generated_profile)]
    run([*launcher, "plan"], cwd=ROOT)
    run([sys.executable, str(ROOT / "scripts/preflight.py"), "--site", str(args.generated_site)], cwd=ROOT)
    if not args.no_launch:
        run(
            [
                *launcher,
                "--execute",
                "--confirmation",
                "START-EXL3-LMCACHE-CS512-ALL-FOUR",
                "start",
            ],
            cwd=ROOT,
        )
    print(f"PASS: EXL3 bootstrap complete; image={image} image_id={expected_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
