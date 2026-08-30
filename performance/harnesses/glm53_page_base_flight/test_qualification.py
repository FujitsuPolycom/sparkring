from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qualification import (
    BASE_TOKENS,
    FLIGHT_SCHEMA,
    PARTICIPANTS,
    RECEIPT_SCHEMA,
    RESULT_TOKENS,
    TAIL_TOKENS,
    inspect_manifests,
    prompts,
    verify_evidence,
)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_prompt_geometry_is_exact_and_distinct() -> None:
    base, results, unrelated = prompts(list(range(100, 118)))
    assert len(base) == BASE_TOKENS + 1
    assert len(results) == PARTICIPANTS
    assert all(len(result) == RESULT_TOKENS + 1 for result in results)
    assert all(result[:BASE_TOKENS] == base[:BASE_TOKENS] for result in results)
    assert len({hashlib.sha256(_canonical(result)).hexdigest() for result in results}) == 16
    assert len(unrelated) == 4097
    assert unrelated[:4096] != base[:4096]
    assert TAIL_TOKENS == 32768


def test_manifest_inspection_requires_one_shared_base_and_private_deltas(
    tmp_path: Path,
) -> None:
    base_root = {
        "schema": "sparkcache-page-root/v1",
        "committed_tokens": BASE_TOKENS,
        "chunks": [],
    }
    base_root_sha = hashlib.sha256(_canonical(base_root)).hexdigest()
    for index in range(PARTICIPANTS):
        _write(
            tmp_path / f"manifest-{index}.json",
            {
                "schema": "sparkcache-page-delta-manifest/v2",
                "base_committed_tokens": BASE_TOKENS,
                "committed_tokens": RESULT_TOKENS,
                "context_digest": f"{index + 1:064x}",
                "base_context_digest": "a" * 64,
                "base_root": base_root,
                "base_root_sha256": base_root_sha,
                "layout_sha256": "b" * 64,
                "delta_sha256": f"{index + 101:064x}",
                "delta_encoded_bytes": 1234 + index,
            },
        )
    receipt = inspect_manifests(tmp_path, rank=2)
    assert receipt["status"] == "verified"
    assert receipt["rank"] == 2
    assert receipt["result_count"] == 16
    assert receipt["shared_base_root_sha256"] == base_root_sha
    assert len(receipt["result_context_digests"]) == 16
    assert len(receipt["delta_sha256"]) == 16


def test_verdict_requires_one_flight_and_sixteen_result_restores_per_rank(
    tmp_path: Path,
) -> None:
    labels = {
        "org.sparkcache.source-revision": "5a6613e473a713695948e69e0027fd67530028f8",
        "org.sparkcache.source-tree": "5e74b3f9d484064d966ce6392dda1e0f7ff17190",
        "org.sparkcache.source-sha256": "446c5bdd5a3efae8a4c4955cfbb577be1d8672a91d47770db63115cb25889313",
        "org.sparkcache.cuda-placement-library-sha256": "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c",
        "org.sparkcache.feature.page-base-read-flight": "implemented-gpu-free-tested",
        "org.sparkcache.feature.page-base-read-flight-pr": "42",
        "org.sparkcache.cache-namespace-impact": "none",
        "org.sparkcache.diagnostic-fix": (
            "page-header-source-bytes-fix=229d7d6;"
            "parent=sha256:9f485c4408a56c0868c75f3e62b09432b2d908b5e4eb28915e0e6b4c4e4fe99f"
        ),
        "org.sparkcache.page-header-source-bytes-fix": "229d7d6",
    }
    artifact = _write(
        tmp_path / "artifact.json",
        {
            "schema": "sparkcache-diagnostic-image-receipt/v1",
            "image": {
                "id": "sha256:cc2c0e2f812f4b78d5b91f863aaf46fd8e8e505844245aa50911af1fb8e061c0",
                "labels": labels,
            },
        },
    )
    semantic = _write(
        tmp_path / "semantic.json",
        {
            "status": "verified",
            "prompt_sha256": "d" * 64,
            "response_sha256": "e" * 64,
        },
    )
    results = [
        {
            "result_index": index,
            "http_status": 200,
            "prompt_sha256": f"{index + 1:064x}",
            "response_sha256": f"{index + 101:064x}",
        }
        for index in range(PARTICIPANTS)
    ]
    published = _write(tmp_path / "publish.json", {"results": results})
    replayed = _write(
        tmp_path / "replay.json",
        {
            "status": "verified",
            "results": results,
            "unrelated_later_request": {
                "http_status": 200,
                "prompt_sha256": "f" * 64,
                "response_sha256": "0" * 64,
            },
        },
    )
    manifests = [
        _write(
            tmp_path / f"rank-{rank}-manifests.json",
            {"status": "verified", "rank": rank, "result_count": 16},
        )
        for rank in range(4)
    ]
    logs = []
    for rank in range(4):
        lines = [
            "spark-context-cache-page-base-flight:"
            + json.dumps(
                {
                    "schema": FLIGHT_SCHEMA,
                    "participants": 16,
                    "physical_base_reads": 1,
                    "avoided_base_reads": 15,
                    "outcome": "verified",
                    "storage_mode": "block_pages_v1",
                }
            )
        ]
        lines.extend(
            "spark-context-cache-restore-timing:"
            + json.dumps(
                {
                    "schema": "sparkcache-restore-timing/v1",
                    "span_tokens": RESULT_TOKENS,
                    "storage_mode": "block_pages_v1",
                    "outcome": "verified",
                    "digest": f"{index + 1:064x}",
                    "request_id": f"rank-{rank}-request-{index}",
                }
            )
            for index in range(PARTICIPANTS)
        )
        log = tmp_path / f"rank-{rank}.log"
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logs.append(log)
    verdict = verify_evidence(
        artifact,
        semantic,
        published,
        replayed,
        manifests,
        logs,
    )
    assert verdict["schema"] == RECEIPT_SCHEMA
    assert verdict["status"] == "qualified"
    assert verdict["image_id"] == (
        "sha256:cc2c0e2f812f4b78d5b91f863aaf46fd8e8e505844245aa50911af1fb8e061c0"
    )
    assert [item["verified_result_restores"] for item in verdict["rank_evidence"]] == [
        16,
        16,
        16,
        16,
    ]
