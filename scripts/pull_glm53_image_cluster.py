#!/usr/bin/env python3
"""Pull and verify one immutable GLM-5.3 image on every configured rank."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from sparkring_site import SiteConfigError, load_site


RECEIPT_SCHEMA = "sparkring-glm53-cluster-image/v1"
CONFIRMATION = "PULL_GLM53_IMAGE"
IMMUTABLE_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")


class PullError(RuntimeError):
    """Raised when an immutable image cannot be verified on every rank."""


def remote_command(image: str) -> tuple[str, ...]:
    if IMMUTABLE_IMAGE.fullmatch(image) is None:
        raise PullError("image must be an immutable registry reference ending in @sha256:<64 hex>")
    script = (
        f"docker pull --platform linux/arm64 {shlex.quote(image)} >/dev/null"
        f" && docker image inspect {shlex.quote(image)}"
    )
    return ("sh", "-lc", script)


def plan_document(site: Any, image: str) -> dict[str, Any]:
    command = remote_command(image)
    return {
        "schema": "sparkring-glm53-cluster-image-plan/v1",
        "safety": ["MUTATES HOST"],
        "image": image,
        "actions": [
            {
                "rank": rank.id,
                "ssh_target": rank.ssh_target,
                "command": list(command),
            }
            for rank in site.ranks
        ],
    }


def _pull_one(rank: Any, image: str, timeout: int) -> tuple[int, dict[str, Any]]:
    remote = shlex.join(remote_command(image))
    completed = subprocess.run(
        ("ssh", rank.ssh_target, remote),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PullError(f"rank {rank.id} image pull failed: {detail}")
    try:
        documents = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PullError(f"rank {rank.id} image inspection was not JSON") from exc
    if not isinstance(documents, list) or len(documents) != 1:
        raise PullError(f"rank {rank.id} returned an unexpected image inspection")
    document = documents[0]
    if document.get("Architecture") != "arm64" or document.get("Os") != "linux":
        raise PullError(f"rank {rank.id} pulled a non-linux/arm64 image")
    return rank.id, {
        "ssh_target": rank.ssh_target,
        "image_id": document.get("Id"),
        "repo_digests": sorted(document.get("RepoDigests") or []),
        "size_bytes": document.get("Size"),
        "labels": dict(sorted((document.get("Config", {}).get("Labels") or {}).items())),
    }


def pull_cluster(site: Any, image: str, timeout: int) -> dict[str, Any]:
    results: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(site.ranks)) as pool:
        futures = [pool.submit(_pull_one, rank, image, timeout) for rank in site.ranks]
        for future in concurrent.futures.as_completed(futures):
            rank, result = future.result()
            results[rank] = result
    image_ids = {result["image_id"] for result in results.values()}
    if len(image_ids) != 1:
        raise PullError(f"ranks resolved different local image IDs: {sorted(image_ids)}")
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "implemented",
        "image": image,
        "image_id": next(iter(image_ids)),
        "ranks": [
            {"rank": rank, **results[rank]}
            for rank in sorted(results)
        ],
        "limitation": (
            "This receipt proves identical image content on every rank. It does not "
            "qualify model serving, transport, DFlash, or SparkCache behavior."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation")
    args = parser.parse_args()
    try:
        site = load_site(args.site)
        plan = plan_document(site, args.image)
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if args.confirmation != CONFIRMATION:
            parser.error(f"execute requires --confirmation {CONFIRMATION}")
        receipt = pull_cluster(site, args.image, args.timeout)
    except (OSError, SiteConfigError, PullError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
