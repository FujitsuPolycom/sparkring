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
        "org.sparkcache.source-revision": "9c2f6c8ac36e0aa5d134fbcd81e819db2ce63970",
        "org.sparkcache.source-tree": "e7ac2ef7a3180c5a83771edac44216c3325894e5",
        "org.sparkcache.source-sha256": "834ff02c235e3f3a3594cec31d0a83d981ac8d410d6482d062725fd9b846a95c",
        "org.sparkcache.cuda-placement-library-sha256": "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c",
        "org.sparkcache.feature.page-base-read-flight": "implemented-gpu-free-tested",
        "org.sparkcache.feature.page-base-read-flight-pr": "42",
        "org.sparkcache.cache-namespace-impact": "none",
    }
    artifact = _write(
        tmp_path / "artifact.json",
        {
            "schema": "sparkcache-diagnostic-image-receipt/v1",
            "image": {"id": "sha256:" + "c" * 64, "labels": labels},
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
    assert verdict["image_id"] == "sha256:" + "c" * 64
    assert [item["verified_result_restores"] for item in verdict["rank_evidence"]] == [
        16,
        16,
        16,
        16,
    ]
