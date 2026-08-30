from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

import qualification
from qualification import (
    BASE_CODEWORD,
    BASE_TOKENS,
    FLIGHT_SCHEMA,
    LANE_CODEWORDS,
    PARTICIPANTS,
    RECEIPT_SCHEMA,
    RESULT_TOKENS,
    TAIL_TOKENS,
    UNRELATED_CODEWORD,
    QualificationError,
    inspect_manifests,
    publish,
    prompts,
    replay,
    verify_evidence,
)


def _instructions() -> dict[str, list[int]]:
    words = (BASE_CODEWORD, *LANE_CODEWORDS, UNRELATED_CODEWORD)
    return {word: [900 + index, 999] for index, word in enumerate(words)}


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_prompt_geometry_is_exact_and_distinct() -> None:
    instructions = _instructions()
    base, results, unrelated = prompts(list(range(100, 118)), instructions)
    assert len(base) == BASE_TOKENS + 1
    assert len(results) == PARTICIPANTS
    assert all(len(result) == RESULT_TOKENS + 1 for result in results)
    assert all(result[:BASE_TOKENS] == base[:BASE_TOKENS] for result in results)
    assert all(
        result[-len(instructions[word]) :] == instructions[word]
        for result, word in zip(results, LANE_CODEWORDS, strict=True)
    )
    assert all(result[BASE_TOKENS] == 101 + index for index, result in enumerate(results))
    assert base[-len(instructions[BASE_CODEWORD]) :] == instructions[BASE_CODEWORD]
    assert unrelated[-len(instructions[UNRELATED_CODEWORD]) :] == instructions[
        UNRELATED_CODEWORD
    ]
    assert len({hashlib.sha256(_canonical(result)).hexdigest() for result in results}) == 16
    assert len(unrelated) == 4097
    assert unrelated[:4096] != base[:4096]
    assert TAIL_TOKENS == 32768


def _completion_receipt(
    tokens: list[int],
    response_sha256: str = "a" * 64,
    expected_oracle: str | None = None,
) -> dict:
    receipt = {
        "http_status": 200,
        "prompt_tokens": len(tokens),
        "prompt_sha256": hashlib.sha256(_canonical(tokens)).hexdigest(),
        "response_sha256": response_sha256,
        "completion_tokens": 1,
        "finish_reason": "length",
        "elapsed_seconds": 0.125,
    }
    if expected_oracle is not None:
        receipt.update(
            expected_oracle=expected_oracle,
            observed_oracle=expected_oracle,
            oracle_match=True,
        )
    return receipt


def test_publish_waits_for_all_rank_held_digest_report_before_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = list(range(100, 118))
    instructions = _instructions()
    observed_lengths: list[int] = []
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text(
        "KV Transfer metrics: spark_cache_ranks_reporting=4, "
        "spark_cache_digests_held=3\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(qualification, "discover_token_bank", lambda *_args: bank)
    monkeypatch.setattr(
        qualification, "discover_instruction_tokens", lambda *_args: instructions
    )
    scheduler_attempts = 0

    def fake_completion(
        _endpoint: str,
        _model: str,
        tokens: list[int],
        _timeout: float,
        *,
        expected_oracle: str | None = None,
    ) -> dict:
        nonlocal scheduler_attempts
        observed_lengths.append(len(tokens))
        if len(tokens) == 2:
            scheduler_attempts += 1
            with scheduler_log.open("a", encoding="utf-8") as stream:
                stream.write(
                    "KV Transfer metrics: spark_cache_ranks_reporting=4, "
                    f"spark_cache_digests_held={3 if scheduler_attempts == 1 else 4}\n"
                )
        return _completion_receipt(tokens, expected_oracle=expected_oracle)

    monkeypatch.setattr(qualification, "_completion", fake_completion)

    receipt = publish("http://rank0", "model", scheduler_log, 10.0)

    assert observed_lengths[:4] == [BASE_TOKENS + 1, 2, 2, RESULT_TOKENS + 1]
    assert observed_lengths[3:] == [RESULT_TOKENS + 1] * PARTICIPANTS
    assert receipt["base_readiness"] == {
        "status": "verified",
        "required_ranks": 4,
        "ranks_reporting": 4,
        "digests_held": 4,
        "scheduler_log_line_sha256": hashlib.sha256(
            (
                "KV Transfer metrics: spark_cache_ranks_reporting=4, "
                "spark_cache_digests_held=4"
            ).encode()
        ).hexdigest(),
        "scheduler_steps": [
            {"attempt": 1, **_completion_receipt([bank[-2], bank[-1]])},
            {"attempt": 2, **_completion_receipt([bank[-2], bank[-1]])},
        ],
    }
    assert all(item["oracle_match"] for item in receipt["results"])


def test_publish_rejects_before_private_results_without_all_rank_holds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = list(range(100, 118))
    instructions = _instructions()
    observed_lengths: list[int] = []
    scheduler_log = tmp_path / "scheduler.log"
    scheduler_log.write_text("startup\n", encoding="utf-8")
    monkeypatch.setattr(qualification, "discover_token_bank", lambda *_args: bank)
    monkeypatch.setattr(
        qualification, "discover_instruction_tokens", lambda *_args: instructions
    )

    def fake_completion(
        _endpoint: str,
        _model: str,
        tokens: list[int],
        _timeout: float,
        *,
        expected_oracle: str | None = None,
    ) -> dict:
        observed_lengths.append(len(tokens))
        if len(tokens) == 2:
            with scheduler_log.open("a", encoding="utf-8") as stream:
                stream.write(
                    "KV Transfer metrics: spark_cache_ranks_reporting=4, "
                    "spark_cache_digests_held=3\n"
                )
        return _completion_receipt(tokens, expected_oracle=expected_oracle)

    monkeypatch.setattr(qualification, "_completion", fake_completion)

    with pytest.raises(QualificationError, match="not held on all ranks"):
        publish("http://rank0", "model", scheduler_log, 0.01)
    assert observed_lengths == [BASE_TOKENS + 1, 2]


def test_replay_accepts_raw_hash_drift_when_lane_oracles_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = list(range(100, 118))
    instructions = _instructions()
    _base, results, unrelated = prompts(bank, instructions)
    published_results = [
        {
            "prompt_sha256": _completion_receipt(tokens)["prompt_sha256"],
            "response_sha256": "a" * 64,
            "expected_oracle": word,
            "observed_oracle": word,
            "oracle_match": True,
        }
        for tokens, word in zip(results, LANE_CODEWORDS, strict=True)
    ]
    mismatch_prompt_sha256 = {
        published_results[index]["prompt_sha256"] for index in (2, 11)
    }
    monkeypatch.setattr(qualification, "discover_token_bank", lambda *_args: bank)
    monkeypatch.setattr(
        qualification, "discover_instruction_tokens", lambda *_args: instructions
    )

    def fake_completion(
        _endpoint: str,
        _model: str,
        tokens: list[int],
        _timeout: float,
        *,
        expected_oracle: str | None = None,
    ) -> dict:
        prompt_sha256 = _completion_receipt(tokens)["prompt_sha256"]
        digest = "b" * 64 if prompt_sha256 in mismatch_prompt_sha256 else "a" * 64
        return _completion_receipt(tokens, digest, expected_oracle)

    monkeypatch.setattr(qualification, "_completion", fake_completion)
    receipt = replay(
        "http://rank0",
        "model",
        {
            "prompt_spec_sha256": qualification._prompt_spec_sha256(instructions),
            "results": published_results,
        },
        10.0,
    )

    assert receipt["status"] == "verified"
    assert receipt["response_mismatch_indices"] == [2, 11]
    assert receipt["oracle_mismatch_indices"] == []
    assert len(receipt["results"]) == PARTICIPANTS
    assert [item["result_index"] for item in receipt["results"]] == list(
        range(PARTICIPANTS)
    )
    assert receipt["unrelated_later_request"]["prompt_sha256"] == (
        _completion_receipt(unrelated)["prompt_sha256"]
    )


def test_replay_returns_complete_rejected_receipt_after_oracle_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = list(range(100, 118))
    instructions = _instructions()
    _base, results, _unrelated = prompts(bank, instructions)
    published_results = [
        {
            "prompt_sha256": _completion_receipt(tokens)["prompt_sha256"],
            "response_sha256": "a" * 64,
            "expected_oracle": word,
            "observed_oracle": word,
            "oracle_match": True,
        }
        for tokens, word in zip(results, LANE_CODEWORDS, strict=True)
    ]
    monkeypatch.setattr(qualification, "discover_token_bank", lambda *_args: bank)
    monkeypatch.setattr(
        qualification, "discover_instruction_tokens", lambda *_args: instructions
    )

    def fake_completion(
        _endpoint: str,
        _model: str,
        tokens: list[int],
        _timeout: float,
        *,
        expected_oracle: str | None = None,
    ) -> dict:
        receipt = _completion_receipt(tokens, expected_oracle=expected_oracle)
        if expected_oracle == LANE_CODEWORDS[6]:
            receipt.update(observed_oracle="spark", oracle_match=False)
        return receipt

    monkeypatch.setattr(qualification, "_completion", fake_completion)
    receipt = replay(
        "http://rank0",
        "model",
        {
            "prompt_spec_sha256": qualification._prompt_spec_sha256(instructions),
            "results": published_results,
        },
        10.0,
    )

    assert receipt["status"] == "rejected"
    assert receipt["oracle_mismatch_indices"] == [6]
    assert len(receipt["results"]) == PARTICIPANTS


def test_replay_cli_writes_rejected_receipt_before_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "replay.json"
    published = _write(tmp_path / "publish.json", {"results": []})
    rejected = {
        "schema": RECEIPT_SCHEMA,
        "kind": "replay",
        "status": "rejected",
        "results": [{"result_index": index} for index in range(PARTICIPANTS)],
        "unrelated_later_request": {"http_status": 200},
        "response_mismatch_indices": [4],
    }
    monkeypatch.setattr(qualification, "replay", lambda *_args: rejected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualification.py",
            "replay",
            "--endpoint",
            "http://rank0",
            "--model",
            "model",
            "--publish-receipt",
            str(published),
            "--output",
            str(output),
        ],
    )

    assert qualification.main() == 1
    assert json.loads(output.read_text(encoding="utf-8")) == rejected


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


def test_verdict_records_research_evidence_for_complete_mechanical_inputs(
    tmp_path: Path,
) -> None:
    labels = {
        "org.sparkcache.source-revision": "a1511d26a1fe2b17b24561bc52e376bf7f54b06a",
        "org.sparkcache.source-tree": "4d5b8eb8c5c13793ee7a1e67b2b34bd38fcf4ddb",
        "org.sparkcache.source-sha256": "6651f2823c816fac93779cbca54a8f19c0ed262830953149f3a87d189d1f833b",
        "org.sparkcache.cuda-placement-library-sha256": "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c",
        "org.sparkcache.feature.page-base-read-flight": "implemented-gpu-free-tested",
        "org.sparkcache.feature.page-base-read-flight-pr": "42",
        "org.sparkcache.page-base-read-flight-singleton-later-cohorts": (
            "a1511d26a1fe2b17b24561bc52e376bf7f54b06a"
        ),
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
                "id": "sha256:" + "1" * 64,
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
            "expected_oracle": LANE_CODEWORDS[index],
            "observed_oracle": LANE_CODEWORDS[index],
            "oracle_match": True,
        }
        for index in range(PARTICIPANTS)
    ]
    published = _write(
        tmp_path / "publish.json",
        {
            "schema": RECEIPT_SCHEMA,
            "prompt_spec_sha256": "c" * 64,
            "base_readiness": {
                "status": "verified",
                "scheduler_steps": [{"attempt": 1}],
            },
            "results": results,
        },
    )
    replayed = _write(
        tmp_path / "replay.json",
        {
            "schema": RECEIPT_SCHEMA,
            "status": "verified",
            "prompt_spec_sha256": "c" * 64,
            "oracle_mismatch_indices": [],
            "results": results,
            "unrelated_later_request": {
                "http_status": 200,
                "prompt_sha256": "f" * 64,
                "response_sha256": "0" * 64,
                "expected_oracle": UNRELATED_CODEWORD,
                "observed_oracle": UNRELATED_CODEWORD,
                "oracle_match": True,
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
    assert verdict["kind"] == "research-verdict"
    assert verdict["status"] == "research-only"
    assert verdict["image_id"] == "sha256:" + "1" * 64
    assert [item["verified_result_restores"] for item in verdict["rank_evidence"]] == [
        16,
        16,
        16,
        16,
    ]
