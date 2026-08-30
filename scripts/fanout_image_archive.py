#!/usr/bin/env python3
"""Distribute one verified image archive through the configured direct fabric.

Planning is the default and performs no remote work. ``--verify`` reads the
archive state on every rank. ``--execute`` seeds one rank, forwards the exact
archive through a direct-link chain, verifies every hop, and imports the image
unless ``--create-only`` is set.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit

from sparkring_site import SiteConfigError, load_site


PLAN_SCHEMA = "sparkring-image-archive-fabric-plan/v1"
RECEIPT_SCHEMA = "sparkring-image-archive-fabric-receipt/v1"
VERIFY_SCHEMA = "sparkring-image-archive-fabric-verification/v1"
CONFIRMATION = "FANOUT_IMAGE_ARCHIVE"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ARCHIVE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,190}\Z")
_IMAGE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,510}\Z")


class FanoutError(RuntimeError):
    """The archive cannot be distributed or verified safely."""


@dataclass(frozen=True)
class ArchivePaths:
    directory: str
    final: str
    partial: str


@dataclass(frozen=True)
class Hop:
    index: int
    edge: str
    source_rank: int
    destination_rank: int
    source_address: str
    destination_address: str


@dataclass(frozen=True)
class Probe:
    state: str
    sha256: str | None = None
    bytes: int | None = None
    detail: str | None = None


Runner = Callable[..., subprocess.CompletedProcess[str]]


def archive_paths(target_directory: str, archive_name: str) -> ArchivePaths:
    if _ARCHIVE_NAME.fullmatch(archive_name) is None:
        raise FanoutError(
            "archive name must be one safe filename without path separators"
        )
    directory = PurePosixPath(target_directory)
    if not directory.is_absolute() or ".." in directory.parts:
        raise FanoutError("target directory must be an absolute normalized path")
    if directory == PurePosixPath("/") or len(directory.parts) < 4:
        raise FanoutError("target directory is too broad")
    final = directory / archive_name
    partial = directory / f".{archive_name}.partial"
    if final.parent != directory or partial.parent != directory:
        raise FanoutError("archive paths must remain inside the target directory")
    return ArchivePaths(str(directory), str(final), str(partial))


def source_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise FanoutError("source URL must use HTTP or HTTPS")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise FanoutError(
            "source URL must not contain credentials, a query, or a fragment"
        )
    return value


def _rank_map(site: Any) -> dict[int, Any]:
    ranks = {rank.id: rank for rank in site.ranks}
    if len(ranks) != 4 or sorted(ranks) != list(range(4)):
        raise FanoutError("direct archive fan-out requires ranks 0, 1, 2, and 3")
    return ranks


def _port_to(rank: Any, destination_rank: int) -> Any:
    matches = [
        port for port in rank.ring_ports if port.peer_rank == destination_rank
    ]
    if len(matches) != 1 or matches[0].peer_address is None:
        raise FanoutError(
            f"rank {rank.id} has no unique direct port to rank {destination_rank}"
        )
    return matches[0]


def fabric_chain(
    site: Any,
    seed_rank: int,
    first_hop_rank: int | None = None,
) -> tuple[Hop, ...]:
    ranks = _rank_map(site)
    if seed_rank not in ranks:
        raise FanoutError(f"seed rank {seed_rank} is not configured")
    neighbours = tuple(sorted(ranks[seed_rank].neighbour_ranks))
    if len(neighbours) != 2:
        raise FanoutError("seed rank must have exactly two direct neighbours")
    if first_hop_rank is None:
        first_hop_rank = neighbours[0]
    if first_hop_rank not in neighbours:
        raise FanoutError("first hop must be a direct neighbour of the seed rank")

    order = [seed_rank, first_hop_rank]
    while len(order) < len(ranks):
        candidates = [
            rank_id
            for rank_id in sorted(ranks[order[-1]].neighbour_ranks)
            if rank_id not in order
        ]
        if len(candidates) != 1:
            raise FanoutError("configured direct links do not form one chainable cycle")
        order.append(candidates[0])

    hops = []
    for index, (source_rank, destination_rank) in enumerate(
        zip(order, order[1:]), start=1
    ):
        port = _port_to(ranks[source_rank], destination_rank)
        hops.append(
            Hop(
                index=index,
                edge=port.edge,
                source_rank=source_rank,
                destination_rank=destination_rank,
                source_address=str(port.address),
                destination_address=str(port.peer_address),
            )
        )
    return tuple(hops)


def _remote_argv(rank: Any, script: str, connect_timeout: int) -> tuple[str, ...]:
    return (
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        rank.ssh_target,
        shlex.join(("sh", "-lc", script)),
    )


def _probe_script(paths: ArchivePaths, expected_sha256: str) -> str:
    final = shlex.quote(paths.final)
    expected = shlex.quote(expected_sha256)
    return (
        f"if [ ! -e {final} ]; then printf 'MISSING\\n'; exit 0; fi; "
        f"if [ ! -f {final} ]; then printf 'CONFLICT non-regular\\n'; exit 0; fi; "
        f"actual=$(sha256sum -- {final} | awk '{{print $1}}'); "
        f"bytes=$(wc -c < {final}); "
        f"if [ \"$actual\" = {expected} ]; then "
        "printf 'EXACT %s %s\\n' \"$actual\" \"$bytes\"; "
        "else printf 'CONFLICT %s %s\\n' \"$actual\" \"$bytes\"; fi"
    )


def _prepare_script(paths: ArchivePaths) -> str:
    return f"install -d -m 0750 -- {shlex.quote(paths.directory)}"


def _download_script(
    paths: ArchivePaths,
    url: str,
    expected_sha256: str,
) -> str:
    directory = shlex.quote(paths.directory)
    final = shlex.quote(paths.final)
    partial = shlex.quote(paths.partial)
    url_arg = shlex.quote(url)
    expected = shlex.quote(expected_sha256)
    return (
        "set -eu; "
        f"install -d -m 0750 -- {directory}; "
        f"test ! -e {final}; "
        f"curl --fail --location --retry 3 --continue-at - --output {partial}"
        f" -- {url_arg}; "
        f"actual=$(sha256sum -- {partial} | awk '{{print $1}}'); "
        f"test \"$actual\" = {expected}; "
        f"sync -f {partial}; ln -- {partial} {final}; rm -- {partial}; "
        f"sync -f {directory}; sha256sum -- {final}"
    )


def _transfer_script(
    paths: ArchivePaths,
    hop: Hop,
    destination_user: str,
    connect_timeout: int,
) -> str:
    source = shlex.quote(paths.final)
    destination = shlex.quote(
        f"{destination_user}@{hop.destination_address}:{paths.partial}"
    )
    remote_shell = shlex.quote(
        "ssh -o BatchMode=yes"
        " -o StrictHostKeyChecking=yes"
        f" -o ConnectTimeout={connect_timeout}"
        f" -b {hop.source_address}"
    )
    return (
        "rsync --archive --partial --append-verify --protect-args"
        f" --timeout={connect_timeout} -e {remote_shell} -- {source} {destination}"
    )


def _promote_script(paths: ArchivePaths, expected_sha256: str) -> str:
    directory = shlex.quote(paths.directory)
    final = shlex.quote(paths.final)
    partial = shlex.quote(paths.partial)
    expected = shlex.quote(expected_sha256)
    return (
        "set -eu; "
        f"test ! -e {final}; test -f {partial}; "
        f"actual=$(sha256sum -- {partial} | awk '{{print $1}}'); "
        f"test \"$actual\" = {expected}; "
        f"sync -f {partial}; ln -- {partial} {final}; rm -- {partial}; "
        f"sync -f {directory}; sha256sum -- {final}"
    )


def _load_script(
    paths: ArchivePaths,
    image: str,
    expected_image_id: str,
) -> str:
    final = shlex.quote(paths.final)
    image_arg = shlex.quote(image)
    expected = shlex.quote(expected_image_id)
    return (
        f"if docker image inspect {image_arg} >/dev/null 2>&1; then "
        f"existing=$(docker image inspect {image_arg} --format '{{{{.Id}}}}'); "
        f"test \"$existing\" = {expected}; "
        "else "
        f"docker image load --input {final} >/dev/null; fi; "
        f"docker image inspect {image_arg}"
        " --format '{{.Id}} {{.Os}}/{{.Architecture}}'; "
        f"image_id=$(docker image inspect {image_arg} --format '{{{{.Id}}}}')"
        f"; test \"$image_id\" = {expected}"
    )


def _destination_user(rank: Any) -> str:
    if "@" not in rank.ssh_target:
        raise FanoutError(f"rank {rank.id} SSH target has no user")
    return rank.ssh_target.split("@", 1)[0]


def plan_document(
    site: Any,
    *,
    url: str,
    expected_sha256: str,
    paths: ArchivePaths,
    seed_rank: int,
    first_hop_rank: int | None,
    create_only: bool,
    image: str | None,
    expected_image_id: str | None,
    connect_timeout: int,
) -> dict[str, Any]:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise FanoutError("expected SHA-256 must be 64 lowercase hex characters")
    if expected_image_id and _IMAGE_ID.fullmatch(expected_image_id) is None:
        raise FanoutError("expected image ID must be sha256:<64 lowercase hex>")
    if not create_only:
        if not image or _IMAGE_REFERENCE.fullmatch(image) is None:
            raise FanoutError(
                "a safe image reference is required unless --create-only is set"
            )
        if not expected_image_id:
            raise FanoutError(
                "expected image ID is required unless --create-only is set"
            )
    ranks = _rank_map(site)
    hops = fabric_chain(site, seed_rank, first_hop_rank)
    actions: list[dict[str, Any]] = []

    seed = ranks[seed_rank]
    actions.append(
        {
            "id": "seed-probe",
            "kind": "probe",
            "rank": seed_rank,
            "command": list(
                _remote_argv(seed, _probe_script(paths, expected_sha256), connect_timeout)
            ),
        }
    )
    actions.append(
        {
            "id": "seed-download",
            "kind": "download-if-missing",
            "rank": seed_rank,
            "command": list(
                _remote_argv(
                    seed,
                    _download_script(paths, url, expected_sha256),
                    connect_timeout,
                )
            ),
        }
    )
    for hop in hops:
        source = ranks[hop.source_rank]
        destination = ranks[hop.destination_rank]
        actions.extend(
            (
                {
                    "id": f"hop-{hop.index}-probe",
                    "kind": "probe",
                    "rank": hop.destination_rank,
                    "command": list(
                        _remote_argv(
                            destination,
                            _probe_script(paths, expected_sha256),
                            connect_timeout,
                        )
                    ),
                },
                {
                    "id": f"hop-{hop.index}-prepare",
                    "kind": "prepare-if-missing",
                    "rank": hop.destination_rank,
                    "command": list(
                        _remote_argv(
                            destination,
                            _prepare_script(paths),
                            connect_timeout,
                        )
                    ),
                },
                {
                    "id": f"hop-{hop.index}-transfer",
                    "kind": "direct-rsync-if-missing",
                    "rank": hop.source_rank,
                    "hop": asdict(hop),
                    "command": list(
                        _remote_argv(
                            source,
                            _transfer_script(
                                paths,
                                hop,
                                _destination_user(destination),
                                connect_timeout,
                            ),
                            connect_timeout,
                        )
                    ),
                },
                {
                    "id": f"hop-{hop.index}-verify",
                    "kind": "verify-and-promote-if-missing",
                    "rank": hop.destination_rank,
                    "command": list(
                        _remote_argv(
                            destination,
                            _promote_script(paths, expected_sha256),
                            connect_timeout,
                        )
                    ),
                },
            )
        )
    if not create_only:
        assert image is not None
        for rank_id in sorted(ranks):
            rank = ranks[rank_id]
            actions.append(
                {
                    "id": f"rank-{rank_id}-load",
                    "kind": "load-image",
                    "rank": rank_id,
                    "command": list(
                        _remote_argv(
                            rank,
                            _load_script(paths, image, expected_image_id),
                            connect_timeout,
                        )
                    ),
                }
            )
    return {
        "schema": PLAN_SCHEMA,
        "status": "implemented",
        "safety": ["MUTATES HOST"],
        "source_url": url,
        "archive": {
            "name": PurePosixPath(paths.final).name,
            "sha256": expected_sha256,
            "target_directory": paths.directory,
        },
        "seed_rank": seed_rank,
        "topology": {
            "link_speed_mbps": int(site.topology.link_speed_mbps),
            "mtu": int(site.topology.mtu),
        },
        "chain": [asdict(hop) for hop in hops],
        "create_only": create_only,
        "image": image,
        "expected_image_id": expected_image_id,
        "actions": actions,
    }


def parse_probe(output: str) -> Probe:
    tokens = output.strip().split()
    if tokens == ["MISSING"]:
        return Probe("missing")
    if len(tokens) >= 2 and tokens[0] == "CONFLICT":
        if len(tokens) == 3 and _SHA256.fullmatch(tokens[1]):
            return Probe("conflict", tokens[1], int(tokens[2]))
        return Probe("conflict", detail=" ".join(tokens[1:]))
    if (
        len(tokens) == 3
        and tokens[0] == "EXACT"
        and _SHA256.fullmatch(tokens[1])
    ):
        return Probe("exact", tokens[1], int(tokens[2]))
    raise FanoutError(f"archive probe returned malformed evidence: {output!r}")


def _run(
    argv: Sequence[str],
    *,
    timeout: int,
    runner: Runner,
    action: str,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = runner(
            tuple(argv),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise FanoutError(
            f"{action} was interrupted; the bounded partial file is resumable"
        ) from error
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise FanoutError(f"{action} failed: {detail}")
    return completed


def _probe_rank(
    rank: Any,
    paths: ArchivePaths,
    expected_sha256: str,
    *,
    timeout: int,
    connect_timeout: int,
    runner: Runner,
) -> Probe:
    completed = _run(
        _remote_argv(rank, _probe_script(paths, expected_sha256), connect_timeout),
        timeout=timeout,
        runner=runner,
        action=f"rank {rank.id} archive probe",
    )
    return parse_probe(completed.stdout)


def verify_cluster(
    site: Any,
    paths: ArchivePaths,
    expected_sha256: str,
    *,
    timeout: int,
    connect_timeout: int,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    ranks = _rank_map(site)
    evidence = []
    for rank_id in sorted(ranks):
        probe = _probe_rank(
            ranks[rank_id],
            paths,
            expected_sha256,
            timeout=timeout,
            connect_timeout=connect_timeout,
            runner=runner,
        )
        evidence.append({"rank": rank_id, **asdict(probe)})
    if any(item["state"] != "exact" for item in evidence):
        raise FanoutError(f"archive verification failed: {evidence}")
    return {
        "schema": VERIFY_SCHEMA,
        "status": "verified",
        "archive": paths.final,
        "sha256": expected_sha256,
        "topology": {
            "link_speed_mbps": int(site.topology.link_speed_mbps),
            "mtu": int(site.topology.mtu),
        },
        "ranks": evidence,
    }


def execute_fanout(
    site: Any,
    plan: dict[str, Any],
    paths: ArchivePaths,
    *,
    timeout: int,
    connect_timeout: int,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    ranks = _rank_map(site)
    expected_sha256 = plan["archive"]["sha256"]
    url = plan["source_url"]
    seed_rank = int(plan["seed_rank"])
    rank_receipts: dict[int, dict[str, Any]] = {}
    hop_receipts = []

    seed_probe = _probe_rank(
        ranks[seed_rank],
        paths,
        expected_sha256,
        timeout=timeout,
        connect_timeout=connect_timeout,
        runner=runner,
    )
    if seed_probe.state == "conflict":
        raise FanoutError(f"seed rank {seed_rank} has a conflicting archive")
    seed_reused = seed_probe.state == "exact"
    if not seed_reused:
        _run(
            _remote_argv(
                ranks[seed_rank],
                _download_script(paths, url, expected_sha256),
                connect_timeout,
            ),
            timeout=timeout,
            runner=runner,
            action=f"rank {seed_rank} archive download",
        )
        seed_probe = _probe_rank(
            ranks[seed_rank],
            paths,
            expected_sha256,
            timeout=timeout,
            connect_timeout=connect_timeout,
            runner=runner,
        )
    if seed_probe.state != "exact":
        raise FanoutError("seed archive was not verified after download")
    rank_receipts[seed_rank] = {
        "rank": seed_rank,
        **asdict(seed_probe),
        "source": "existing" if seed_reused else "download",
    }

    for hop_record in plan["chain"]:
        hop = Hop(**hop_record)
        destination = ranks[hop.destination_rank]
        probe = _probe_rank(
            destination,
            paths,
            expected_sha256,
            timeout=timeout,
            connect_timeout=connect_timeout,
            runner=runner,
        )
        if probe.state == "conflict":
            raise FanoutError(
                f"rank {hop.destination_rank} has a conflicting archive"
            )
        reused = probe.state == "exact"
        if not reused:
            _run(
                _remote_argv(
                    destination, _prepare_script(paths), connect_timeout
                ),
                timeout=timeout,
                runner=runner,
                action=f"hop {hop.index} destination preparation",
            )
            _run(
                _remote_argv(
                    ranks[hop.source_rank],
                    _transfer_script(
                        paths,
                        hop,
                        _destination_user(destination),
                        connect_timeout,
                    ),
                    connect_timeout,
                ),
                timeout=timeout,
                runner=runner,
                action=f"hop {hop.index} direct rsync",
            )
            _run(
                _remote_argv(
                    destination,
                    _promote_script(paths, expected_sha256),
                    connect_timeout,
                ),
                timeout=timeout,
                runner=runner,
                action=f"hop {hop.index} checksum and promotion",
            )
            probe = _probe_rank(
                destination,
                paths,
                expected_sha256,
                timeout=timeout,
                connect_timeout=connect_timeout,
                runner=runner,
            )
        if probe.state != "exact":
            raise FanoutError(f"hop {hop.index} did not produce an exact archive")
        rank_receipts[hop.destination_rank] = {
            "rank": hop.destination_rank,
            **asdict(probe),
            "source": "existing" if reused else f"rank-{hop.source_rank}",
        }
        hop_receipts.append(
            {
                **asdict(hop),
                "status": "reused" if reused else "transferred-and-verified",
                "sha256": probe.sha256,
                "bytes": probe.bytes,
            }
        )

    loaded = []
    if not plan["create_only"]:
        image = plan["image"]
        assert image
        for rank_id in sorted(ranks):
            completed = _run(
                _remote_argv(
                    ranks[rank_id],
                    _load_script(paths, image, plan["expected_image_id"]),
                    connect_timeout,
                ),
                timeout=timeout,
                runner=runner,
                action=f"rank {rank_id} image import",
            )
            loaded.append({"rank": rank_id, "inspection": completed.stdout.strip()})

    return {
        "schema": RECEIPT_SCHEMA,
        "status": "implemented",
        "archive": {
            "path": paths.final,
            "sha256": expected_sha256,
        },
        "seed_rank": seed_rank,
        "topology": plan["topology"],
        "ranks": [rank_receipts[index] for index in sorted(rank_receipts)],
        "hops": hop_receipts,
        "create_only": plan["create_only"],
        "loaded_images": loaded,
        "limitation": (
            "This receipt proves archive identity and optional image import. It"
            " does not qualify model serving or direct-fabric throughput."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True)
    parser.add_argument("--source-url")
    parser.add_argument("--archive-name", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--target-directory", required=True)
    parser.add_argument("--seed-rank", type=int, default=0)
    parser.add_argument("--first-hop-rank", type=int)
    parser.add_argument("--image")
    parser.add_argument("--expected-image-id")
    parser.add_argument("--create-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--connect-timeout", type=int, default=45)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        site = load_site(args.site)
        paths = archive_paths(args.target_directory, args.archive_name)
        if _SHA256.fullmatch(args.expected_sha256) is None:
            raise FanoutError(
                "expected SHA-256 must be 64 lowercase hex characters"
            )
        if args.verify:
            if args.execute:
                parser.error("--verify and --execute are mutually exclusive")
            document = verify_cluster(
                site,
                paths,
                args.expected_sha256,
                timeout=args.timeout,
                connect_timeout=args.connect_timeout,
            )
        else:
            if not args.source_url:
                parser.error("--source-url is required for planning or execution")
            url = source_url(args.source_url)
            plan = plan_document(
                site,
                url=url,
                expected_sha256=args.expected_sha256,
                paths=paths,
                seed_rank=args.seed_rank,
                first_hop_rank=args.first_hop_rank,
                create_only=args.create_only,
                image=args.image,
                expected_image_id=args.expected_image_id,
                connect_timeout=args.connect_timeout,
            )
            if not args.execute:
                document = plan
            else:
                if args.confirmation != CONFIRMATION:
                    parser.error(
                        f"execute requires --confirmation {CONFIRMATION}"
                    )
                document = execute_fanout(
                    site,
                    plan,
                    paths,
                    timeout=args.timeout,
                    connect_timeout=args.connect_timeout,
                )
    except (FanoutError, OSError, SiteConfigError) as error:
        parser.error(str(error))

    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
