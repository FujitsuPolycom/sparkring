#!/usr/bin/env python3
"""Model-independent SparkRing cluster inventory and validator.

The cluster inventory is the first configuration a blank Spark installation
needs. It contains only management access, the direct-cable topology, and the
ConnectX-7/RoCE facts required by Ring Doctor. Runtime, model, cache, and
serving configuration belong to deployment profiles and are deliberately
absent here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml as _yaml
except Exception as _exc:  # pragma: no cover - only without PyYAML
    _yaml = None
    _YAML_IMPORT_ERROR: Exception | None = _exc
else:
    _YAML_IMPORT_ERROR = None

try:
    from .sparkring_site import (
        PreflightOptions,
        Rank,
        SiteConfigError,
        Topology,
        validate_site,
    )
except ImportError:  # Direct execution
    from sparkring_site import (
        PreflightOptions,
        Rank,
        SiteConfigError,
        Topology,
        validate_site,
    )

SCHEMA_ID = "sparkring-cluster/v1"
SCHEMA_VERSION = 1


class ClusterConfigError(ValueError):
    """A cluster inventory is invalid."""


@dataclasses.dataclass(frozen=True)
class ClusterConfig:
    schema_version: int
    name: str
    description: str
    topology: Topology
    ranks: tuple[Rank, ...]
    preflight: PreflightOptions
    source: str | None = None

    def rank(self, rank_id: int) -> Rank:
        for rank in self.ranks:
            if rank.id == rank_id:
                return rank
        raise KeyError(rank_id)

    @property
    def head_rank(self) -> Rank:
        return self.rank(0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cluster": {
                "name": self.name,
                "description": self.description,
            },
            "topology": {
                "mtu": self.topology.mtu,
                "link_speed_mbps": self.topology.link_speed_mbps,
                "edges": [
                    {
                        "id": edge.id,
                        "subnet": str(edge.subnet),
                        "endpoints": list(edge.endpoints),
                    }
                    for edge in self.topology.edges
                ],
            },
            "ranks": [
                {
                    "id": rank.id,
                    "ssh_target": rank.ssh_target,
                    "management": {
                        "interface": rank.management.interface,
                        "address": str(rank.management.address),
                    },
                    "ring_ports": [
                        {
                            "edge": port.edge,
                            "interface": port.interface,
                            "address": str(port.address),
                            "rdma_device": port.rdma_device,
                            "rdma_port": port.rdma_port,
                            "roce_gid_index": port.roce_gid_index,
                        }
                        for port in rank.ring_ports
                    ],
                    "transport_peers": [
                        {"rank": peer.rank, "address": str(peer.address)}
                        for peer in rank.transport_peers
                    ],
                }
                for rank in self.ranks
            ],
            "preflight": {
                "ssh_timeout_seconds": self.preflight.ssh_timeout_seconds,
                "required_free_ports": list(self.preflight.required_free_ports),
            },
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"cluster     : {self.name} (schema {self.schema_version})",
            f"description : {self.description}",
            f"source      : {self.source or '<in-memory>'}",
            f"topology    : closed {len(self.ranks)}-cycle, "
            f"mtu={self.topology.mtu}, "
            f"link={self.topology.link_speed_mbps} Mb/s",
        ]
        for edge in self.topology.edges:
            left, right = edge.endpoints
            left_port = next(
                port for port in self.rank(left).ring_ports if port.edge == edge.id
            )
            right_port = next(
                port for port in self.rank(right).ring_ports if port.edge == edge.id
            )
            lines.append(
                f"  {edge.id:<10} {str(edge.subnet):<18} "
                f"rank{left} {left_port.address} ({left_port.interface}/"
                f"{left_port.rdma_key} gid{left_port.roce_gid_index}) <-> "
                f"rank{right} {right_port.address} ({right_port.interface}/"
                f"{right_port.rdma_key} gid{right_port.roce_gid_index})"
            )
        lines.append("ranks       :")
        for rank in self.ranks:
            lines.append(
                f"  rank{rank.id} {rank.ssh_target:<28} "
                f"mgmt {rank.management.interface}={rank.management.address}"
            )
        return lines


def _expanded_site_document(document: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"schema_version", "cluster", "topology", "ranks", "preflight"}
    unknown = set(document) - allowed
    missing = allowed - set(document)
    if unknown:
        raise ClusterConfigError(
            "cluster inventory has unsupported key(s): " + ", ".join(sorted(unknown))
        )
    if missing:
        raise ClusterConfigError(
            "cluster inventory is missing key(s): " + ", ".join(sorted(missing))
        )
    cluster = document["cluster"]
    if not isinstance(cluster, Mapping):
        raise ClusterConfigError("cluster must be a mapping")
    ranks = document["ranks"]
    rank_count = len(ranks) if isinstance(ranks, list) else 1
    return {
        "schema_version": document["schema_version"],
        "site": dict(cluster),
        "topology": document["topology"],
        "ranks": ranks,
        "runtime": {
            "container_image": "sparkring/cluster-inventory:local",
            "container_image_digest": "sha256:" + "a1" * 32,
            "model_path": "/var/lib/sparkring/cluster-inventory/model",
            "model_repo": "sparkring/cluster-inventory",
            "model_revision": "a1" * 20,
            "checkpoint_sha256": "b2" * 32,
        },
        "serving": {
            "tensor_parallel_size": rank_count,
            "decode_context_parallel_size": 1,
            "mtp_mode": "off",
            "mtp_tokens": 0,
            "max_model_len": 256,
            "kv_cache_bytes_per_rank": 1 << 20,
            "max_num_seqs": 1,
            "master_rank": 0,
            "api_port": 8000,
            "master_port": 29500,
        },
        "paths": {
            "jit_cache_dir": "/var/lib/sparkring/cluster-inventory/jit",
            "context_cache_dir": "/var/lib/sparkring/cluster-inventory/context",
            "evidence_dir": ".sparkring/evidence",
            "min_free_bytes": {"jit_cache": 0, "context_cache": 0},
        },
        "artifacts": [],
        "preflight": document["preflight"],
    }


def validate_cluster(
    document: Any, source: str | None = None
) -> ClusterConfig:
    if not isinstance(document, Mapping):
        raise ClusterConfigError("cluster inventory root must be a mapping")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ClusterConfigError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    try:
        site = validate_site(_expanded_site_document(document), source=source)
    except SiteConfigError as exc:
        raise ClusterConfigError(str(exc)) from exc
    return ClusterConfig(
        schema_version=site.schema_version,
        name=site.name,
        description=site.description,
        topology=site.topology,
        ranks=site.ranks,
        preflight=site.preflight,
        source=source,
    )


def parse_cluster_yaml(text: str, source: str | None = None) -> ClusterConfig:
    if _yaml is None:
        raise ClusterConfigError(
            "PyYAML is required to read cluster inventories "
            f"(import failed with: {_YAML_IMPORT_ERROR})"
        )
    try:
        document = _yaml.safe_load(text)
    except Exception as exc:
        raise ClusterConfigError(f"could not parse YAML: {exc}") from None
    return validate_cluster(document, source=source)


def load_cluster(path: str | Path) -> ClusterConfig:
    resolved = Path(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ClusterConfigError(f"could not read {resolved}: {exc}") from exc
    return parse_cluster_yaml(text, source=str(resolved))


def write_cluster(config: ClusterConfig, path: str | Path) -> None:
    if _yaml is None:  # pragma: no cover
        raise ClusterConfigError("PyYAML is required to write cluster inventories")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a SparkRing cluster inventory")
    parser.add_argument("path")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        cluster = load_cluster(args.path)
    except ClusterConfigError as exc:
        print(f"INVALID CLUSTER: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(cluster.to_dict(), indent=2, sort_keys=True))
    else:
        print("\n".join(cluster.summary_lines()))
        print(f"\nOK: {args.path} is a valid SparkRing cluster inventory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
