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
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from sparkring_site import SiteConfig, load_site
from verify_ssh_mesh import Hop, image_fanout_hops, split_ssh_target

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = "BOOTSTRAP-NF3-ALL-FOUR"
PUBLIC_REPOSITORY = "https://github.com/FujitsuPolycom/sparkring.git"
# Follower ranks clone the source they build from. A deployment working from a
# fork, a mirror, or an internal host serves that source elsewhere, and every
# rank must agree on which. This environment variable names that origin; the
# clone still verifies the checked-out commit, so a substituted origin cannot
# change which bytes are built.
BOOTSTRAP_REPOSITORY_ENV = "SPARKRING_BOOTSTRAP_REPOSITORY"


def bootstrap_repository(environ=None) -> str:
    """Return the repository URL follower ranks clone."""

    source = os.environ if environ is None else environ
    return str(source.get(BOOTSTRAP_REPOSITORY_ENV, "")).strip() or PUBLIC_REPOSITORY
IMAGE_TRANSFER_HEADROOM_BYTES = 5 * 1024**3
RECIPE_PATH = ROOT / "recipes/glm52-nf3-hybrid.json"
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


def load_nf3_contract() -> dict[str, object]:
    """Load the canonical target, draft, and serving pins for this bootstrap."""

    document = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    if document.get("schema") != "sparkring-recipe/v1":
        raise RuntimeError("NF3 recipe schema is not sparkring-recipe/v1")
    if document.get("recipe_id") != "glm52-nf3-hybrid":
        raise RuntimeError("NF3 recipe id is not glm52-nf3-hybrid")
    return document


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


def remote_output(target: str, command: str) -> str:
    result = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "StrictHostKeyChecking=yes",
            target,
            command,
        ],
        capture=True,
    )
    return result.stdout.strip()


def ssh_bootstrap_verification_command(site_path: Path) -> list[str]:
    """Return the verifier invocation matching this bootstrap's fanout path."""
    return [
        sys.executable,
        str(ROOT / "scripts/verify_ssh_mesh.py"),
        "--site",
        str(site_path),
        "--scope",
        "bootstrap",
    ]


def ssh_image_fanout_verification_command(site_path: Path) -> list[str]:
    """Return the verifier invocation for direct-ring archive payload hops."""
    return [
        sys.executable,
        str(ROOT / "scripts/verify_ssh_mesh.py"),
        "--site",
        str(site_path),
        "--scope",
        "image-fanout",
    ]


def early_fabric_preflight_command(site_path: Path) -> list[str]:
    """Return the image-independent ring/RDMA gate run before model work."""
    return [
        sys.executable,
        str(ROOT / "scripts/preflight.py"),
        "--site",
        str(site_path),
        "--scope",
        "fabric",
        "--no-evidence",
    ]


def direct_image_fanout(site: SiteConfig) -> tuple[tuple[Hop, Hop], Hop]:
    """Return rank0's parallel first wave and the one relay hop."""
    hops = image_fanout_hops(site)
    if (
        len(hops) != 3
        or any(hop.source_rank != 0 for hop in hops[:2])
        or hops[2].source_rank != hops[0].destination_rank
    ):
        raise ValueError("unexpected direct-ring image fanout topology")
    return (hops[0], hops[1]), hops[2]


def direct_ssh_target(site: SiteConfig, hop: Hop) -> str:
    """Use the destination rank's SSH user at its direct-ring address."""
    user, _ = split_ssh_target(site.rank(hop.destination_rank).ssh_target)
    return f"{user}@{hop.destination_address}"


def remote_archive_path(expected_id: str) -> str:
    """Return an identity-bound, exact temporary archive path."""
    match = re.fullmatch(r"sha256:([0-9a-f]{64})", expected_id)
    if match is None:
        raise ValueError(f"invalid Docker image ID: {expected_id!r}")
    return f"/var/tmp/sparkring-image-{match.group(1)}.tar"


def load_archive_command(
    image: str,
    expected_id: str,
    archive: str,
    *,
    keep_archive: bool,
) -> str:
    """Load one archive, attest its image ID, then optionally remove only it."""
    lines = [
        "set -euo pipefail",
        f"test -f {shlex.quote(archive)}",
        f"docker load --input {shlex.quote(archive)}",
        (
            f"observed=$(docker image inspect {shlex.quote(image)} "
            "--format '{{.Id}}')"
        ),
        f"test \"$observed\" = {shlex.quote(expected_id)}",
    ]
    if not keep_archive:
        lines.append(f"rm -f -- {shlex.quote(archive)}")
    return "\n".join(lines)


def remote_image_id(target: str, image: str) -> str:
    command = (
        f"docker image inspect {shlex.quote(image)} "
        "--format '{{.Id}}' 2>/dev/null || true"
    )
    return remote_output(target, command)


def direct_scp_argv(source: str, destination: str) -> list[str]:
    return [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=yes",
        source,
        destination,
    ]


def stage_relay_archive_command(
    image: str,
    expected_id: str,
    archive: str,
) -> str:
    """Create an exact archive on a relay that already has the exact image."""
    partial_prefix = archive + ".partial"
    return "\n".join(
        (
            "set -euo pipefail",
            (
                f"observed=$(docker image inspect {shlex.quote(image)} "
                "--format '{{.Id}}')"
            ),
            f"test \"$observed\" = {shlex.quote(expected_id)}",
            f"partial={shlex.quote(partial_prefix)}.$$",
            "trap 'rm -f -- \"$partial\"' EXIT",
            f"docker save --output \"$partial\" {shlex.quote(image)}",
            f"mv -f -- \"$partial\" {shlex.quote(archive)}",
            "trap - EXIT",
        )
    )


def relay_scp_command(
    archive: str,
    destination: str,
) -> str:
    return shlex.join(
        direct_scp_argv(archive, f"{destination}:{archive}")
    )


def fanout_image_archive(
    site: SiteConfig,
    image: str,
    expected_id: str,
    archive: Path,
) -> None:
    """Fan one exact Docker archive over only the direct-ring data plane.

    Management SSH orchestrates commands and observes image IDs. Archive bytes
    travel only over the three direct-ring SSH hops returned by
    :func:`direct_image_fanout`.
    """
    remote_archive = remote_archive_path(expected_id)
    if not archive.is_file() or archive.stat().st_size <= 0:
        raise RuntimeError(f"image archive is missing or empty: {archive}")
    archive_bytes = archive.stat().st_size
    first_wave, relay_hop = direct_image_fanout(site)
    followers = [rank for rank in site.ranks if rank.id != 0]

    observed: dict[int, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            rank.id: pool.submit(remote_image_id, rank.ssh_target, image)
            for rank in followers
        }
        for rank_id, future in futures.items():
            observed[rank_id] = future.result()

    missing = {
        rank_id for rank_id, value in observed.items()
        if value != expected_id
    }
    for rank in followers:
        if rank.id not in missing:
            print(f"rank {rank.id}: exact image already present; transfer skipped")

    relay_rank = relay_hop.source_rank
    opposite_rank = relay_hop.destination_rank
    # A relay that already owns the image still needs temporary archive space
    # when the opposite rank is missing; it will docker-save an exact archive.
    capacity_ids = set(missing)
    if opposite_rank in missing:
        capacity_ids.add(relay_rank)

    print("==> Checking all destination/relay capacities before image transfer")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(capacity_ids))
    ) as pool:
        futures = {
            rank_id: pool.submit(
                remote,
                site.rank(rank_id).ssh_target,
                image_transfer_capacity_command(archive_bytes),
            )
            for rank_id in sorted(capacity_ids)
        }
        for rank_id, future in futures.items():
            future.result()
            print(f"rank {rank_id}: archive/import capacity verified")

    def deliver_first_wave(hop: Hop) -> None:
        destination_rank = hop.destination_rank
        if destination_rank not in missing:
            return
        destination = direct_ssh_target(site, hop)
        run(
            direct_scp_argv(
                str(archive),
                f"{destination}:{remote_archive}",
            )
        )
        keep = (
            destination_rank == relay_rank
            and opposite_rank in missing
        )
        remote(
            site.rank(destination_rank).ssh_target,
            load_archive_command(
                image,
                expected_id,
                remote_archive,
                keep_archive=keep,
            ),
        )
        print(
            f"rank {destination_rank}: image {expected_id} verified "
            f"over direct edge {hop.edge}"
        )

    print("==> Sending rank0 archive to both direct neighbours in parallel")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(deliver_first_wave, hop) for hop in first_wave]
        for future in futures:
            future.result()

    if opposite_rank in missing:
        relay_rank_config = site.rank(relay_rank)
        if relay_rank not in missing:
            print(
                f"rank {relay_rank}: staging relay archive from its exact image"
            )
            remote(
                relay_rank_config.ssh_target,
                stage_relay_archive_command(
                    image,
                    expected_id,
                    remote_archive,
                ),
            )
        destination = direct_ssh_target(site, relay_hop)
        print(
            f"rank {relay_rank}: relaying archive to rank {opposite_rank} "
            f"over direct edge {relay_hop.edge}"
        )
        remote(
            relay_rank_config.ssh_target,
            relay_scp_command(remote_archive, destination),
        )
        remote(
            site.rank(opposite_rank).ssh_target,
            load_archive_command(
                image,
                expected_id,
                remote_archive,
                keep_archive=False,
            ),
        )
        # Remove only the identity-bound relay archive, and only after the
        # opposite rank has loaded and attested the expected image ID.
        remote(
            relay_rank_config.ssh_target,
            f"rm -f -- {shlex.quote(remote_archive)}",
        )
        print(f"rank {opposite_rank}: image {expected_id} verified")

    final_ids: dict[int, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            rank.id: pool.submit(remote_image_id, rank.ssh_target, image)
            for rank in followers
        }
        for rank_id, future in futures.items():
            final_ids[rank_id] = future.result()
    wrong = {
        rank_id: value for rank_id, value in final_ids.items()
        if value != expected_id
    }
    if wrong:
        raise RuntimeError(
            f"direct-ring image fanout final identity mismatch: {wrong}"
        )


def checkout_command(
    commit: str, remote_root: str, repository_url: str | None = None
) -> str:
    repository = f"{remote_root}/{commit}"
    origin = repository_url or bootstrap_repository()
    return "\n".join(
        (
            "set -euo pipefail",
            f"mkdir -p -- {shlex.quote(remote_root)}",
            (
                f"if [ ! -d {shlex.quote(repository + '/.git')} ]; then "
                f"git clone --filter=blob:none {shlex.quote(origin)} "
                f"{shlex.quote(repository)}; fi"
            ),
            (
                f"test \"$(git -C {shlex.quote(repository)} remote get-url origin)\" "
                f"= {shlex.quote(origin)}"
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


def required_image_transfer_bytes(archive_bytes: int) -> int:
    """Return conservative free space for an archive plus Docker import.

    Followers temporarily hold both the docker-save archive and imported image
    layers. Requiring two archive widths plus fixed headroom is conservative
    when /var/tmp and Docker share a filesystem, and harmless when they do not.
    """

    if not isinstance(archive_bytes, int) or archive_bytes <= 0:
        raise ValueError("image archive size must be a positive integer")
    return archive_bytes * 2 + IMAGE_TRANSFER_HEADROOM_BYTES


def image_transfer_capacity_command(archive_bytes: int) -> str:
    required = required_image_transfer_bytes(archive_bytes)
    return "\n".join(
        (
            "set -euo pipefail",
            "docker_root=$(docker info --format '{{.DockerRootDir}}')",
            "test -n \"$docker_root\"",
            "test -d /var/tmp",
            "test -d \"$docker_root\"",
            "tmp_free=$(df -PB1 /var/tmp | awk 'NR==2 {print $4}')",
            "docker_free=$(df -PB1 \"$docker_root\" | awk 'NR==2 {print $4}')",
            "case \"$tmp_free:$docker_free\" in "
            "*[!0-9:]*|:*) echo 'unable to measure follower free space' >&2; "
            "exit 4;; esac",
            (
                f"if [ \"$tmp_free\" -lt {required} ]; then "
                "echo \"insufficient /var/tmp space: "
                f"need {required}, have $tmp_free\" >&2; exit 4; fi"
            ),
            (
                f"if [ \"$docker_free\" -lt {required} ]; then "
                "echo \"insufficient Docker-root space: "
                f"need {required}, have $docker_free\" >&2; exit 4; fi"
            ),
            (
                "printf 'IMAGE_TRANSFER_CAPACITY_OK "
                f"required={required} tmp_free=%s docker_free=%s\\n' "
                "\"$tmp_free\" \"$docker_free\""
            ),
        )
    )


def write_generated_site(
    source: Path,
    destination: Path,
    image: str,
    digest: str,
    profile: str,
) -> None:
    if profile not in PROFILES:
        raise ValueError(f"unknown KV profile: {profile}")
    recipe = load_nf3_contract()
    model = recipe["model"]
    serving_contract = recipe["serving"]
    if profile not in serving_contract["kv_profiles"]:
        raise RuntimeError(f"NF3 recipe does not define KV profile {profile}")

    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    runtime = document["runtime"]
    runtime["container_image"] = image
    runtime["container_image_digest"] = digest
    runtime["model_path"] = f"/models/{model['install_subdir']}"
    runtime["model_repo"] = model["repository"]
    runtime["model_revision"] = model["revision"]
    runtime["checkpoint_sha256"] = model["index_sha256"]
    document["serving"]["kv_cache_bytes_per_rank"] = serving_contract[
        "kv_cache_bytes_per_rank"
    ]
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
            (
                "save and fan out the exact image ID over the direct ring "
                "(rank0 to both neighbors in parallel, then one relay)"
            ),
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
    print("==> Verifying rank 0 management SSH to itself and every follower")
    run(
        ssh_bootstrap_verification_command(args.site),
        cwd=ROOT,
    )
    print("==> Verifying exact direct-ring SSH image fanout tree")
    run(
        ssh_image_fanout_verification_command(args.site),
        cwd=ROOT,
    )
    print("==> Verifying live 200GbE/RDMA fabric before model work")
    run(
        early_fabric_preflight_command(args.site),
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

    print("==> Fanning exact image over the direct 200GbE ring")
    fanout_image_archive(site, image, expected_id, archive)

    write_generated_site(
        args.site,
        args.generated_site,
        image,
        expected_id,
        args.profile,
    )
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
