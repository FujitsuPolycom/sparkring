#!/usr/bin/env python3
"""Produce and verify deterministic GLM-5.3 page-base-flight evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


BASE_TOKENS = 98_304
RESULT_TOKENS = 131_072
TAIL_TOKENS = RESULT_TOKENS - BASE_TOKENS
PARTICIPANTS = 16
FLIGHT_SCHEMA = "sparkcache-page-base-restore-flight/v1"
RESTORE_SCHEMA = "sparkcache-restore-timing/v1"
RECEIPT_SCHEMA = "sparkring-glm53-pr42-page-base-flight-qualification/v1"
PROMPT_SEED = (
    "SparkCache deterministic token bank: alpha beta gamma delta epsilon zeta "
    "eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon."
)


class QualificationError(RuntimeError):
    """The deterministic qualification evidence is incomplete or contradictory."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _token_hash(tokens: list[int]) -> str:
    return _sha256(_canonical(tokens))


def _write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _post(endpoint: str, route: str, payload: dict[str, Any], timeout: float) -> dict:
    request = urllib.request.Request(
        endpoint.rstrip("/") + route,
        data=_canonical(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raise QualificationError(f"HTTP {exc.code} from {route}") from exc
    except OSError as exc:
        raise QualificationError(f"request to {route} did not complete: {exc}") from exc
    if status != 200:
        raise QualificationError(f"HTTP {status} from {route}")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise QualificationError(f"non-JSON response from {route}") from exc


def discover_token_bank(endpoint: str, model: str, timeout: float) -> list[int]:
    document = _post(
        endpoint,
        "/tokenize",
        {"model": model, "prompt": PROMPT_SEED},
        timeout,
    )
    observed = document.get("tokens") or document.get("token_ids")
    if not isinstance(observed, list):
        raise QualificationError("tokenize response omits token IDs")
    unique: list[int] = []
    for value in observed:
        if isinstance(value, int) and value >= 0 and value not in unique:
            unique.append(value)
    if len(unique) < PARTICIPANTS + 2:
        raise QualificationError("tokenize response has fewer than 18 distinct tokens")
    return unique[: PARTICIPANTS + 2]


def prompts(token_bank: list[int]) -> tuple[list[int], list[list[int]], list[int]]:
    if len(token_bank) < PARTICIPANTS + 2:
        raise QualificationError("token bank must contain at least 18 distinct IDs")
    common, terminator = token_bank[0], token_bank[-1]
    base = [common] * BASE_TOKENS + [terminator]
    results = [
        [common] * BASE_TOKENS + [token_bank[index + 1]] * TAIL_TOKENS + [terminator]
        for index in range(PARTICIPANTS)
    ]
    unrelated = [token_bank[-2]] * 4096 + [terminator]
    return base, results, unrelated


def _completion(
    endpoint: str,
    model: str,
    token_ids: list[int],
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = _post(
        endpoint,
        "/v1/completions",
        {
            "model": model,
            "prompt": token_ids,
            "max_tokens": 1,
            "temperature": 0,
        },
        timeout,
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise QualificationError("completion response must contain one choice")
    text = choices[0].get("text")
    if not isinstance(text, str):
        raise QualificationError("completion response omits text")
    usage = response.get("usage") or {}
    return {
        "http_status": 200,
        "prompt_tokens": len(token_ids),
        "prompt_sha256": _token_hash(token_ids),
        "response_sha256": _sha256(text.encode()),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choices[0].get("finish_reason"),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def publish(endpoint: str, model: str, timeout: float) -> dict[str, Any]:
    bank = discover_token_bank(endpoint, model, timeout)
    base, results, _unrelated = prompts(bank)
    base_result = _completion(endpoint, model, base, timeout)
    result_receipts = [
        {"result_index": index, **_completion(endpoint, model, prompt, timeout)}
        for index, prompt in enumerate(results)
    ]
    return {
        "schema": RECEIPT_SCHEMA,
        "kind": "publish",
        "status": "observed",
        "model": model,
        "token_bank_sha256": _token_hash(bank),
        "base_publication_tokens": BASE_TOKENS,
        "result_publication_tokens": RESULT_TOKENS,
        "private_tail_tokens": TAIL_TOKENS,
        "base_request": base_result,
        "results": result_receipts,
    }


def semantic(endpoint: str, model: str, timeout: float) -> dict[str, Any]:
    prompt = "The capital of France is"
    response = _post(
        endpoint,
        "/v1/completions",
        {"model": model, "prompt": prompt, "max_tokens": 16, "temperature": 0},
        timeout,
    )
    choice = response["choices"][0]
    text = str(choice.get("text", ""))
    return {
        "schema": RECEIPT_SCHEMA,
        "kind": "semantic",
        "status": "verified" if "paris" in text.lower() else "rejected",
        "model": model,
        "prompt_sha256": _sha256(prompt.encode()),
        "response_sha256": _sha256(text.encode()),
        "semantic_match": "paris" in text.lower(),
    }


def replay(
    endpoint: str,
    model: str,
    publish_receipt: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    bank = discover_token_bank(endpoint, model, timeout)
    _base, results, unrelated = prompts(bank)
    expected = [item["prompt_sha256"] for item in publish_receipt["results"]]
    observed = [_token_hash(item) for item in results]
    if observed != expected:
        raise QualificationError("reconstructed prompt hashes differ from publication")
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARTICIPANTS + 1) as pool:
        futures = [
            pool.submit(_completion, endpoint, model, prompt, timeout)
            for prompt in results
        ]
        time.sleep(0.05)
        unrelated_future = pool.submit(_completion, endpoint, model, unrelated, timeout)
        receipts = [future.result() for future in futures]
        unrelated_receipt = unrelated_future.result()
    for index, item in enumerate(receipts):
        item["result_index"] = index
        if item["response_sha256"] != publish_receipt["results"][index][
            "response_sha256"
        ]:
            raise QualificationError(f"result {index} response differs from publication")
    return {
        "schema": RECEIPT_SCHEMA,
        "kind": "replay",
        "status": "verified",
        "model": model,
        "token_bank_sha256": _token_hash(bank),
        "results": receipts,
        "unrelated_later_request": unrelated_receipt,
    }


def inspect_manifests(root: Path, rank: int) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for path in root.rglob("*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            document.get("base_committed_tokens") == BASE_TOKENS
            and document.get("committed_tokens") == RESULT_TOKENS
            and document.get("schema") in {
                "sparkcache-page-delta-manifest/v1",
                "sparkcache-page-delta-manifest/v2",
            }
        ):
            base_root = document.get("base_root")
            if not isinstance(base_root, dict):
                raise QualificationError("result manifest omits its authenticated base root")
            if _sha256(_canonical(base_root)) != document.get("base_root_sha256"):
                raise QualificationError("result manifest base-root checksum differs")
            selected.append(
                {
                    "context_digest": document["context_digest"],
                    "base_context_digest": document["base_context_digest"],
                    "base_root_sha256": document["base_root_sha256"],
                    "layout_sha256": document["layout_sha256"],
                    "delta_sha256": document["delta_sha256"],
                    "delta_encoded_bytes": document["delta_encoded_bytes"],
                }
            )
    if len(selected) != PARTICIPANTS:
        raise QualificationError(f"rank {rank} has {len(selected)} result manifests, want 16")
    for field in ("base_context_digest", "base_root_sha256", "layout_sha256"):
        if len({item[field] for item in selected}) != 1:
            raise QualificationError(f"rank {rank} result manifests disagree on {field}")
    for field in ("context_digest", "delta_sha256"):
        if len({item[field] for item in selected}) != PARTICIPANTS:
            raise QualificationError(f"rank {rank} result manifests do not have 16 {field} values")
    return {
        "schema": RECEIPT_SCHEMA,
        "kind": "manifest-inspection",
        "status": "verified",
        "rank": rank,
        "storage_mode": "block_pages_v1",
        "base_tokens": BASE_TOKENS,
        "result_tokens": RESULT_TOKENS,
        "result_count": len(selected),
        "shared_base_context_digest": selected[0]["base_context_digest"],
        "shared_base_root_sha256": selected[0]["base_root_sha256"],
        "layout_sha256": selected[0]["layout_sha256"],
        "result_context_digests": sorted(item["context_digest"] for item in selected),
        "delta_sha256": sorted(item["delta_sha256"] for item in selected),
    }


def _records(path: Path, marker: str, schema: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        try:
            document = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError as exc:
            raise QualificationError(f"malformed {schema} record in {path}") from exc
        if document.get("schema") == schema:
            records.append(document)
    return records


def verify_evidence(
    artifact_receipt: Path,
    semantic_receipt: Path,
    publish_receipt: Path,
    replay_receipt: Path,
    manifest_receipts: list[Path],
    rank_logs: list[Path],
) -> dict[str, Any]:
    artifact = json.loads(artifact_receipt.read_text(encoding="utf-8"))
    semantic_document = json.loads(semantic_receipt.read_text(encoding="utf-8"))
    published = json.loads(publish_receipt.read_text(encoding="utf-8"))
    replayed = json.loads(replay_receipt.read_text(encoding="utf-8"))
    labels = artifact.get("labels")
    image_id = artifact.get("image_id")
    if not isinstance(labels, dict):
        image = artifact.get("image")
        if not isinstance(image, dict):
            raise QualificationError("artifact receipt omits image metadata")
        labels = image.get("labels")
        image_id = image.get("id")
    if not isinstance(labels, dict) or not isinstance(image_id, str):
        raise QualificationError("artifact receipt omits labels or image ID")
    expected_labels = {
        "org.sparkcache.source-revision": "9c2f6c8ac36e0aa5d134fbcd81e819db2ce63970",
        "org.sparkcache.source-tree": "e7ac2ef7a3180c5a83771edac44216c3325894e5",
        "org.sparkcache.source-sha256": "834ff02c235e3f3a3594cec31d0a83d981ac8d410d6482d062725fd9b846a95c",
        "org.sparkcache.cuda-placement-library-sha256": "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c",
        "org.sparkcache.feature.page-base-read-flight": (
            "implemented-gpu-free-tested"
        ),
        "org.sparkcache.feature.page-base-read-flight-pr": "42",
        "org.sparkcache.cache-namespace-impact": "none",
    }
    if any(labels.get(name) != value for name, value in expected_labels.items()):
        raise QualificationError("artifact receipt labels differ from PR42")
    contract = artifact.get("page_base_restore_flight_contract")
    if contract is not None and (
        not isinstance(contract, dict) or contract.get("summary_schema") != FLIGHT_SCHEMA
    ):
        raise QualificationError("artifact receipt contains a contradictory contract")
    if semantic_document.get("status") != "verified":
        raise QualificationError("semantic canary is not verified")
    if replayed.get("status") != "verified":
        raise QualificationError("replay receipt is not verified")
    published_results = published.get("results", [])
    replayed_results = replayed.get("results", [])
    if len(published_results) != PARTICIPANTS or len(replayed_results) != PARTICIPANTS:
        raise QualificationError("publication and replay must each contain 16 results")
    if [item["prompt_sha256"] for item in published_results] != [
        item["prompt_sha256"] for item in replayed_results
    ]:
        raise QualificationError("publication and replay prompt hashes differ")
    if [item["response_sha256"] for item in published_results] != [
        item["response_sha256"] for item in replayed_results
    ]:
        raise QualificationError("publication and replay response hashes differ")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_receipts]
    if len(manifests) != 4 or any(item.get("status") != "verified" for item in manifests):
        raise QualificationError("four verified rank manifest receipts are required")
    rank_evidence = []
    if len(rank_logs) != 4:
        raise QualificationError("four bounded rank logs are required")
    for rank, path in enumerate(rank_logs):
        flights = _records(path, "spark-context-cache-page-base-flight:", FLIGHT_SCHEMA)
        if len(flights) != 1:
            raise QualificationError(f"rank {rank} has {len(flights)} flight summaries")
        flight = flights[0]
        required = {
            "participants": 16,
            "physical_base_reads": 1,
            "avoided_base_reads": 15,
            "outcome": "verified",
            "storage_mode": "block_pages_v1",
        }
        if any(flight.get(name) != value for name, value in required.items()):
            raise QualificationError(f"rank {rank} flight summary differs: {flight}")
        restores = [
            item
            for item in _records(
                path,
                "spark-context-cache-restore-timing:",
                RESTORE_SCHEMA,
            )
            if item.get("span_tokens") == RESULT_TOKENS
            and item.get("storage_mode") == "block_pages_v1"
            and item.get("outcome") == "verified"
        ]
        if len(restores) != PARTICIPANTS:
            raise QualificationError(f"rank {rank} has {len(restores)} verified result restores")
        if len({item["digest"] for item in restores}) != PARTICIPANTS:
            raise QualificationError(f"rank {rank} result restore digests are not independent")
        rank_evidence.append(
            {
                "rank": rank,
                "flight_summary": flight,
                "verified_result_restores": len(restores),
                "log_sha256": _sha256(path.read_bytes()),
            }
        )
    inputs = [
        artifact_receipt,
        semantic_receipt,
        publish_receipt,
        replay_receipt,
        *manifest_receipts,
    ]
    return {
        "schema": RECEIPT_SCHEMA,
        "kind": "qualification-verdict",
        "status": "qualified",
        "image_id": image_id,
        "sparkcache_revision": labels["org.sparkcache.source-revision"],
        "semantic_prompt_sha256": semantic_document["prompt_sha256"],
        "semantic_response_sha256": semantic_document["response_sha256"],
        "result_prompt_sha256": [item["prompt_sha256"] for item in replayed_results],
        "result_response_sha256": [item["response_sha256"] for item in replayed_results],
        "unrelated_later_request": replayed["unrelated_later_request"],
        "rank_evidence": rank_evidence,
        "input_sha256": {path.name: _sha256(path.read_bytes()) for path in inputs},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("semantic", "publish", "replay"):
        command = subparsers.add_parser(name)
        command.add_argument("--endpoint", required=True)
        command.add_argument("--model", required=True)
        command.add_argument("--timeout", type=float, default=900.0)
        command.add_argument("--output", type=Path, required=True)
        if name == "replay":
            command.add_argument("--publish-receipt", type=Path, required=True)
    inspect = subparsers.add_parser("inspect-manifests")
    inspect.add_argument("--manifest-root", type=Path, required=True)
    inspect.add_argument("--rank", type=int, choices=range(4), required=True)
    inspect.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact-receipt", type=Path, required=True)
    verify.add_argument("--semantic-receipt", type=Path, required=True)
    verify.add_argument("--publish-receipt", type=Path, required=True)
    verify.add_argument("--replay-receipt", type=Path, required=True)
    verify.add_argument("--manifest-receipt", type=Path, action="append", required=True)
    verify.add_argument("--rank-log", type=Path, action="append", required=True)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "semantic":
        document = semantic(args.endpoint, args.model, args.timeout)
    elif args.command == "publish":
        document = publish(args.endpoint, args.model, args.timeout)
    elif args.command == "replay":
        document = replay(
            args.endpoint,
            args.model,
            json.loads(args.publish_receipt.read_text(encoding="utf-8")),
            args.timeout,
        )
    elif args.command == "inspect-manifests":
        document = inspect_manifests(args.manifest_root, args.rank)
    else:
        if len(args.manifest_receipt) != 4 or len(args.rank_log) != 4:
            parser.error("verify requires exactly four manifest receipts and rank logs")
        document = verify_evidence(
            args.artifact_receipt,
            args.semantic_receipt,
            args.publish_receipt,
            args.replay_receipt,
            args.manifest_receipt,
            args.rank_log,
        )
    _write(args.output, document)
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if document.get("status") in {"observed", "verified", "qualified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
