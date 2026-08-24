"""Offline contract tests for model-independent cluster inventories."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.sparkring_cluster import (
    ClusterConfigError,
    parse_cluster_yaml,
    validate_cluster,
    write_cluster,
)
from scripts.ring_doctor import _build_parser, _load_inputs
from scripts.test_sparkring_site import six_ring_document

EXAMPLE = Path(__file__).parent / "config" / "exl3-r7-site.example.yaml"


def cluster_document(site_document: dict) -> dict:
    return {
        "schema_version": site_document["schema_version"],
        "cluster": copy.deepcopy(site_document["site"]),
        "topology": copy.deepcopy(site_document["topology"]),
        "ranks": copy.deepcopy(site_document["ranks"]),
        "preflight": copy.deepcopy(site_document["preflight"]),
    }


@pytest.fixture()
def four_document() -> dict:
    return cluster_document(yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")))


def test_four_ring_needs_no_runtime_or_model_configuration(four_document):
    cluster = validate_cluster(four_document, source="cluster.yaml")

    assert cluster.source == "cluster.yaml"
    assert len(cluster.ranks) == 4
    assert cluster.head_rank.id == 0
    assert "runtime" not in cluster.to_dict()
    assert "serving" not in cluster.to_dict()


def test_six_ring_validates_through_same_inventory(four_document):
    site_shape = {
        **four_document,
        "site": four_document["cluster"],
        "runtime": {},
        "serving": {},
        "paths": {},
        "artifacts": [],
    }
    six = six_ring_document(site_shape)
    document = cluster_document(six)

    cluster = validate_cluster(document)

    assert [rank.id for rank in cluster.ranks] == list(range(6))
    assert len(cluster.topology.edges) == 6


def test_yaml_round_trip_preserves_canonical_shape(four_document):
    first = validate_cluster(four_document)
    second = parse_cluster_yaml(yaml.safe_dump(first.to_dict(), sort_keys=False))

    assert second.to_dict() == first.to_dict()


def test_model_fields_are_rejected_from_cluster_inventory(four_document):
    four_document["runtime"] = {"container_image": "not allowed"}

    with pytest.raises(ClusterConfigError, match="unsupported key.*runtime"):
        validate_cluster(four_document)


def test_five_ring_remains_unsupported(four_document):
    four_document["topology"]["edges"].append(
        {
            "id": "extra",
            "subnet": "10.99.0.0/24",
            "endpoints": [0, 1],
        }
    )

    with pytest.raises(ClusterConfigError, match="exactly 4 or 6 edges"):
        validate_cluster(four_document)


def test_ring_doctor_loads_cluster_without_a_deployment_site(four_document, tmp_path):
    cluster = validate_cluster(four_document)
    path = tmp_path / "cluster.yaml"
    write_cluster(cluster, path)

    loaded = _load_inputs(
        _build_parser().parse_args(["--cluster", str(path)])
    )

    assert loaded.cluster is not None
    assert loaded.site is None
    assert loaded.rendezvous_address == cluster.head_rank.management.address
    assert [spec.name for spec in loaded.specs] == [
        "rank0", "rank1", "rank2", "rank3"
    ]
