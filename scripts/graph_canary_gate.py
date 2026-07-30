#!/usr/bin/env python3
"""Offline comparator for eager-versus-CUDA-graph semantic canaries.

The live capture is intentionally separate from this tool.  Capture one
temperature-zero, MTP-off response under eager execution and the same request
under the graph candidate, tokenize both responses with the serving runtime,
and record the small JSON artifacts described in
``docs/CUDAGRAPH_CORRECTNESS_GATE.md``.  This program then performs a
deterministic, GPU-free comparison.

Exit codes:

* 0: the graph candidate matches the eager oracle and any required native
  replay status is healthy;
* 2: an artifact is malformed;
* 3: a correctness or replay-progress gate failed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "sparkring-graph-canary/v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODES = {"eager", "graph"}


class ArtifactError(ValueError):
    """The artifact cannot be interpreted safely."""


def _require_exact_keys(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str],
    where: str,
) -> None:
    unknown = sorted(set(value) - required - optional)
    missing = sorted(required - set(value))
    if unknown:
        raise ArtifactError(f"{where}: unknown key {unknown[0]!r}")
    if missing:
        raise ArtifactError(f"{where}: missing key {missing[0]!r}")


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtifactError(f"{where}: expected integer >= {minimum}")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ArtifactError(f"{where}: expected boolean")
    return value


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ArtifactError(f"{where}: expected lowercase SHA-256")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{path}: root must be an object")
    return value


def validate_artifact(value: dict[str, Any], *, expected_mode: str) -> None:
    _require_exact_keys(
        value,
        required={
            "schema",
            "mode",
            "model_identity_sha256",
            "prompt_sha256",
            "request",
            "response",
            "expect_native_graph",
            "graph_ranks",
        },
        optional={"note"},
        where="artifact",
    )
    if value["schema"] != SCHEMA:
        raise ArtifactError(f"artifact: unsupported schema {value['schema']!r}")
    if value["mode"] not in _MODES or value["mode"] != expected_mode:
        raise ArtifactError(
            f"artifact.mode: expected {expected_mode!r}, got {value['mode']!r}"
        )
    _digest(value["model_identity_sha256"], "artifact.model_identity_sha256")
    _digest(value["prompt_sha256"], "artifact.prompt_sha256")

    request = value["request"]
    if not isinstance(request, dict):
        raise ArtifactError("artifact.request: expected object")
    _require_exact_keys(
        request,
        required={"temperature", "seed", "max_tokens", "mtp_enabled"},
        optional=set(),
        where="artifact.request",
    )
    if request["temperature"] != 0 and request["temperature"] != 0.0:
        raise ArtifactError("artifact.request.temperature: must be exactly 0")
    _integer(request["seed"], "artifact.request.seed")
    _integer(request["max_tokens"], "artifact.request.max_tokens", minimum=1)
    if _boolean(request["mtp_enabled"], "artifact.request.mtp_enabled"):
        raise ArtifactError(
            "artifact.request.mtp_enabled: first graph canary must disable MTP"
        )

    response = value["response"]
    if not isinstance(response, dict):
        raise ArtifactError("artifact.response: expected object")
    _require_exact_keys(
        response,
        required={"http_status", "finish_reason", "output_token_ids"},
        optional={"text_preview"},
        where="artifact.response",
    )
    _integer(response["http_status"], "artifact.response.http_status", minimum=100)
    if (
        not isinstance(response["finish_reason"], str)
        or not response["finish_reason"].strip()
    ):
        raise ArtifactError(
            "artifact.response.finish_reason: expected non-empty string"
        )
    token_ids = response["output_token_ids"]
    if not isinstance(token_ids, list):
        raise ArtifactError("artifact.response.output_token_ids: expected list")
    for index, token in enumerate(token_ids):
        _integer(token, f"artifact.response.output_token_ids[{index}]")

    expect_native = _boolean(
        value["expect_native_graph"], "artifact.expect_native_graph"
    )
    ranks = value["graph_ranks"]
    if not isinstance(ranks, list):
        raise ArtifactError("artifact.graph_ranks: expected list")
    if expected_mode == "eager":
        if expect_native or ranks:
            raise ArtifactError(
                "eager artifact must use expect_native_graph=false and graph_ranks=[]"
            )
        return
    if not expect_native:
        if ranks:
            raise ArtifactError(
                "stock-graph artifact must leave graph_ranks empty"
            )
        return
    if len(ranks) != 4:
        raise ArtifactError(
            "native-graph artifact requires exactly four rank snapshots"
        )
    seen: set[int] = set()
    for index, rank in enumerate(ranks):
        where = f"artifact.graph_ranks[{index}]"
        if not isinstance(rank, dict):
            raise ArtifactError(f"{where}: expected object")
        _require_exact_keys(
            rank,
            required={
                "rank",
                "captured_nodes",
                "before",
                "after",
                "submit_affinity_verified",
                "progress_affinity_verified",
            },
            optional=set(),
            where=where,
        )
        rank_id = _integer(rank["rank"], f"{where}.rank")
        if rank_id in seen:
            raise ArtifactError(f"{where}.rank: duplicate rank {rank_id}")
        seen.add(rank_id)
        _integer(rank["captured_nodes"], f"{where}.captured_nodes")
        for snapshot_name in ("before", "after"):
            snapshot = rank[snapshot_name]
            snapshot_where = f"{where}.{snapshot_name}"
            if not isinstance(snapshot, dict):
                raise ArtifactError(f"{snapshot_where}: expected object")
            _require_exact_keys(
                snapshot,
                required={
                    "published_sequence",
                    "consumed_sequence",
                    "completed_sequence",
                    "overflow_sequence",
                },
                optional=set(),
                where=snapshot_where,
            )
            for field in (
                "published_sequence",
                "consumed_sequence",
                "completed_sequence",
                "overflow_sequence",
            ):
                _integer(snapshot[field], f"{snapshot_where}.{field}")
        _boolean(
            rank["submit_affinity_verified"],
            f"{where}.submit_affinity_verified",
        )
        _boolean(
            rank["progress_affinity_verified"],
            f"{where}.progress_affinity_verified",
        )
    if seen != {0, 1, 2, 3}:
        raise ArtifactError(
            f"native-graph rank set must be [0,1,2,3], got {sorted(seen)}"
        )


def compare_artifacts(
    eager: dict[str, Any],
    graph: dict[str, Any],
    *,
    minimum_output_tokens: int,
) -> list[str]:
    validate_artifact(eager, expected_mode="eager")
    validate_artifact(graph, expected_mode="graph")
    failures: list[str] = []

    for field in ("model_identity_sha256", "prompt_sha256", "request"):
        if eager[field] != graph[field]:
            failures.append(f"{field} differs between eager and graph artifacts")

    for name, artifact in (("eager", eager), ("graph", graph)):
        response = artifact["response"]
        if response["http_status"] != 200:
            failures.append(
                f"{name} HTTP status is {response['http_status']}, expected 200"
            )
        count = len(response["output_token_ids"])
        if count < minimum_output_tokens:
            failures.append(
                f"{name} produced {count} token ids, minimum is "
                f"{minimum_output_tokens}"
            )

    eager_ids = eager["response"]["output_token_ids"]
    graph_ids = graph["response"]["output_token_ids"]
    eager_finish_reason = eager["response"]["finish_reason"]
    graph_finish_reason = graph["response"]["finish_reason"]
    if eager_finish_reason != graph_finish_reason:
        failures.append(
            "graph finish_reason differs from the eager oracle "
            f"(eager={eager_finish_reason!r}, graph={graph_finish_reason!r})"
        )
    if eager_ids != graph_ids:
        divergence = next(
            (
                index
                for index, (left, right) in enumerate(zip(eager_ids, graph_ids))
                if left != right
            ),
            min(len(eager_ids), len(graph_ids)),
        )
        failures.append(
            "graph token ids differ from the eager oracle at output index "
            f"{divergence} (eager={len(eager_ids)} tokens, "
            f"graph={len(graph_ids)} tokens)"
        )

    if graph["expect_native_graph"]:
        ranks = sorted(graph["graph_ranks"], key=lambda item: item["rank"])
        captured_counts = {rank["captured_nodes"] for rank in ranks}
        if len(captured_counts) != 1:
            failures.append(
                "native captured-node counts are not rank-synchronous: "
                + ",".join(
                    f"r{rank['rank']}={rank['captured_nodes']}" for rank in ranks
                )
            )
        published_deltas: list[tuple[int, int]] = []
        for rank in ranks:
            rank_id = rank["rank"]
            before = rank["before"]
            after = rank["after"]
            before_published = before["published_sequence"]
            after_published = after["published_sequence"]
            published_delta = after_published - before_published
            published_deltas.append((rank_id, published_delta))
            if rank["captured_nodes"] <= 0:
                failures.append(f"rank {rank_id} captured_nodes is not positive")
            if not (
                before["published_sequence"]
                == before["consumed_sequence"]
                == before["completed_sequence"]
            ):
                failures.append(
                    f"rank {rank_id} replay was not caught up before request: "
                    f"{before['published_sequence']}/"
                    f"{before['consumed_sequence']}/"
                    f"{before['completed_sequence']}"
                )
            if published_delta <= 0:
                failures.append(
                    f"rank {rank_id} published_sequence did not advance "
                    f"during request ({before_published}->{after_published})"
                )
            if not (
                after["published_sequence"]
                == after["consumed_sequence"]
                == after["completed_sequence"]
            ):
                failures.append(
                    f"rank {rank_id} replay is not caught up after request: "
                    f"{after['published_sequence']}/"
                    f"{after['consumed_sequence']}/"
                    f"{after['completed_sequence']}"
                )
            if (
                before["overflow_sequence"] != 0
                or after["overflow_sequence"] != 0
            ):
                failures.append(
                    f"rank {rank_id} overflow_sequence is nonzero "
                    f"({before['overflow_sequence']}->"
                    f"{after['overflow_sequence']})"
                )
            if not rank["submit_affinity_verified"]:
                failures.append(f"rank {rank_id} submit affinity is unverified")
            if not rank["progress_affinity_verified"]:
                failures.append(f"rank {rank_id} progress affinity is unverified")
        unique_deltas = {delta for _, delta in published_deltas}
        if len(unique_deltas) != 1:
            failures.append(
                "native published-sequence advancement is not rank-synchronous: "
                + ",".join(
                    f"r{rank_id}=+{delta}"
                    for rank_id, delta in published_deltas
                )
            )

    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare eager and CUDA-graph semantic canary artifacts"
    )
    parser.add_argument("--eager", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument(
        "--minimum-output-tokens",
        type=int,
        default=16,
        help="reject suspiciously short outputs before token-id comparison",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.minimum_output_tokens < 1:
        print("--minimum-output-tokens must be positive", file=sys.stderr)
        return 2
    try:
        eager = _load_json(args.eager)
        graph = _load_json(args.graph)
        failures = compare_artifacts(
            eager,
            graph,
            minimum_output_tokens=args.minimum_output_tokens,
        )
    except ArtifactError as exc:
        print(f"graph canary artifact error: {exc}", file=sys.stderr)
        return 2
    if failures:
        print("graph canary: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 3
    print("graph canary: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
