#!/usr/bin/env python3
"""One-command, receipt-gated NF3 bootstrap for a four-Spark ring.

Run this on rank 0. ``plan`` is read-only. ``execute`` downloads the pinned
model on every rank, builds one exact image on rank 0, fans that image out,
generates an ignored site file, runs preflight, and starts all four ranks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from sparkring_site import load_site

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = "BOOTSTRAP-NF3-ALL-FOUR"
PUBLIC_REPOSITORY = "https://github.com/FujitsuPolycom/sparkring.git"
PROFILES = {
    "fp8": {
        "image_prefix": "sparkring/glm52-nf3",
        "build_script": "scripts/build-nf3-image.sh",
        "build_step": "build faststart plus thin NF3 image once on rank0",
    },
    "nvfp4-rope8": {
        "image_prefix": "sparkring/glm52-nf3-nvfp4-rope8",
        "build_script": "scripts/build-nf3-nvfp4-rope8-image.sh",
        "build_step": (
            "build the thin NVFP4-latent/FP8-RoPE compatibility layer"
        ),
    },
}


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=capture,
        check=False,
    )
    if capture:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(argv)}"
        )
    return result


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def remote(target: str, command: str) -> None:
    run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            target,
            command,
        ]
    )


def checkout_command(commit: str, remote_root: str) -> str:
    repository = f"{remote_root}/{commit}"
    return "\n".join(
        (
            "set -euo pipefail",
            f"mkdir -p -- {shlex.quote(remote_root)}",
            (
                f"if [ ! -d {shlex.quote(repository + '/.git')} ]; then "
                f"git clone --filter=blob:none {shlex.quote(PUBLIC_REPOSITORY)} "
                f"{shlex.quote(repository)}; fi"
            ),
            (
                f"test \"$(git -C {shlex.quote(repository)} remote get-url origin)\" "
                f"= {shlex.quote(PUBLIC_REPOSITORY)}"
            ),
            (
                f"git -C {shlex.quote(repository)} fetch --depth 1 origin "
                f"{shlex.quote(commit)}"
            ),
            (
                f"git -C {shlex.quote(repository)} checkout --detach --force "
                f"{shlex.quote(commit)}"
            ),
            (
                f"test \"$(git -C {shlex.quote(repository)} rev-parse HEAD)\" "
                f"= {shlex.quote(commit)}"
            ),
        )
    )


def download_command(
    commit: str,
    remote_root: str,
    model_path: str,
    draft_path: str,
) -> str:
    repository = f"{remote_root}/{commit}"
    return checkout_command(commit, remote_root) + "\n" + (
        f"bash {shlex.quote(repository + '/scripts/download-glm52.sh')} "
        f"{shlex.quote(model_path)} {shlex.quote(draft_path)}"
    )


def image_id(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_generated_site(
    source: Path,
    destination: Path,
    image: str,
    digest: str,
) -> None:
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    runtime = document["runtime"]
    runtime["container_image"] = image
    runtime["container_image_digest"] = digest
    runtime["model_path"] = "/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid"
    document["artifacts"] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )


def write_generated_launch(
    source: Path,
    destination: Path,
    profile: str,
) -> None:
    if profile not in PROFILES:
        raise ValueError(f"unknown KV profile: {profile}")
    document = json.loads(source.read_text(encoding="utf-8"))
    environment = document["environment"]
    arguments = document["extra_vllm_args"]
    dtype_index = arguments.index("--kv-cache-dtype") + 1
    environment["VLLM_SPARK_KV_PROFILE"] = profile
    environment["VLLM_SPARK_RUNTIME_ID"] = f"glm52-nf3-{profile}"
    for name in (
        "VLLM_SPARK_KV_CACHE_DTYPE",
        "VLLM_NVFP4_MLA_PER_TOKEN_SCALE",
        "VLLM_SPARK_KV_SCALE_MODE",
    ):
        environment.pop(name, None)
    if profile == "fp8":
        arguments[dtype_index] = "fp8"
    else:
        arguments[dtype_index] = "nvfp4_ds_mla"
        environment.update(
            {
                "VLLM_SPARK_KV_CACHE_DTYPE": "nvfp4_ds_mla",
                "VLLM_NVFP4_MLA_PER_TOKEN_SCALE": "1",
                "VLLM_SPARK_KV_SCALE_MODE": "per-token",
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("plan", "execute"))
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument(
        "--launch-config",
        type=Path,
        default=ROOT / "scripts/config/launch.example.json",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="fp8",
        help="KV storage profile; fp8 is the conservative default",
    )
    parser.add_argument(
        "--generated-site",
        type=Path,
        default=ROOT / ".sparkring/bootstrap/site.yaml",
    )
    parser.add_argument(
        "--remote-root",
        default="/var/tmp/sparkring-bootstrap",
    )
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="prepare and verify all ranks but do not start containers",
    )
    args = parser.parse_args(argv)

    site = load_site(args.site)
    launch = json.loads(args.launch_config.read_text(encoding="utf-8"))
    profile = PROFILES[args.profile]
    commit = git_value("rev-parse", "HEAD")
    dirty = git_value("status", "--porcelain")
    image = f"{profile['image_prefix']}:{commit[:12]}"
    model_path = launch["model_host_path"]
    draft_path = launch["mtp_draft_host_path"]
    plan = {
        "schema": "sparkring-nf3-bootstrap-plan/v1",
        "profile": args.profile,
        "runs_on": "rank0",
        "source_commit": commit,
        "image": image,
        "model_path": model_path,
        "draft_path": draft_path,
        "ranks": [
            {"rank": rank.id, "ssh_target": rank.ssh_target}
            for rank in site.ranks
        ],
        "steps": [
            "verify clean pinned checkout and SSH/cable preflight",
            "resume and hash-verify target plus MTP draft on all four ranks",
            profile["build_step"],
            "save and fan out the exact image ID to ranks 1-3",
            "write ignored generated site config and rerun preflight",
            "launch all four ranks with the pinned C8/Q40 profile",
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
    if not shutil.which("docker"):
        parser.error("docker is required")

    rank0 = site.rank(0)
    addresses = subprocess.run(
        ["ip", "-o", "-4", "addr", "show"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if str(rank0.management.address) not in addresses:
        parser.error(
            "this host does not own rank 0's management address; "
            "run bootstrap on rank 0"
        )

    print("==> Validating site topology before mutation")
    run(
        [
            sys.executable,
            str(ROOT / "scripts/sparkring_site.py"),
            str(args.site),
        ],
        cwd=ROOT,
    )
    print("==> Verifying key-based SSH fanout")
    run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_ssh_mesh.py"),
            "--site",
            str(args.site),
            "--scope",
            "fanout",
        ],
        cwd=ROOT,
    )

    print("==> Downloading/verifying rank 0 model")
    run(
        [
            "bash",
            str(ROOT / "scripts/download-glm52.sh"),
            model_path,
            draft_path,
        ],
        cwd=ROOT,
    )
    followers = [rank for rank in site.ranks if rank.id != 0]
    print("==> Downloading/verifying follower models in parallel")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            rank.id: pool.submit(
                remote,
                rank.ssh_target,
                download_command(
                    commit,
                    args.remote_root,
                    model_path,
                    draft_path,
                ),
            )
            for rank in followers
        }
        for rank_id, future in futures.items():
            future.result()
            print(f"rank {rank_id}: model verified")

    print("==> Building one receipt-gated NF3 image on rank 0")
    environment = dict(os.environ)
    environment["OUTPUT_IMAGE"] = image
    run(
        ["bash", str(ROOT / profile["build_script"])],
        cwd=ROOT,
        env=environment,
    )
    expected_id = image_id(image)

    cache = Path.home() / ".cache/sparkring/nf3-bootstrap"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{image.replace('/', '_').replace(':', '_')}.tar"
    archive_marker = archive.with_suffix(".image-id")
    if (
        not archive.is_file()
        or not archive_marker.is_file()
        or archive_marker.read_text(encoding="utf-8").strip() != expected_id
    ):
        print("==> Exporting exact image once")
        run(["docker", "save", "--output", str(archive), image])
        archive_marker.write_text(expected_id + "\n", encoding="utf-8")

    print("==> Fanning exact image to ranks 1-3")
    for rank in followers:
        remote_archive = f"/var/tmp/{archive.name}"
        run(
            [
                "scp",
                "-o",
                "BatchMode=yes",
                str(archive),
                f"{rank.ssh_target}:{remote_archive}",
            ]
        )
        command = (
            f"set -euo pipefail; docker load --input {shlex.quote(remote_archive)}; "
            f"observed=$(docker image inspect {shlex.quote(image)} "
            "--format '{{.Id}}'); "
            f"test \"$observed\" = {shlex.quote(expected_id)}; "
            f"rm -f -- {shlex.quote(remote_archive)}"
        )
        remote(rank.ssh_target, command)
        print(f"rank {rank.id}: image {expected_id} verified")

    write_generated_site(args.site, args.generated_site, image, expected_id)
    generated_launch = (
        args.generated_site.parent / f"launch.{args.profile}.json"
    )
    write_generated_launch(
        args.launch_config,
        generated_launch,
        args.profile,
    )
    print(f"==> Generated site: {args.generated_site}")
    print(f"==> Generated launch: {generated_launch}")
    run(
        [
            sys.executable,
            str(ROOT / "scripts/preflight.py"),
            "--site",
            str(args.generated_site),
        ],
        cwd=ROOT,
    )
    launcher = [
        sys.executable,
        str(ROOT / "scripts/sparkring_launcher.py"),
        "--site",
        str(args.generated_site),
        "--launch-config",
        str(generated_launch),
    ]
    run([*launcher, "plan"], cwd=ROOT)
    if args.no_launch:
        print("PASS: all four ranks prepared; launch intentionally skipped")
        return 0
    run([*launcher, "--execute", "start"], cwd=ROOT)
    print("PASS: NF3 runtime launched on all four ranks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
