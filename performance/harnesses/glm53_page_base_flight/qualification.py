#!/usr/bin/env python3
"""Produce and verify GLM-5.3 page-base-flight evidence with stable oracles."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


BASE_TOKENS = 98_304
RESULT_TOKENS = 131_072
TAIL_TOKENS = RESULT_TOKENS - BASE_TOKENS
PARTICIPANTS = 16
TP_RANKS = 4
FLIGHT_SCHEMA = "sparkcache-page-base-restore-flight/v1"
RESTORE_SCHEMA = "sparkcache-restore-timing/v1"
RECEIPT_SCHEMA = "sparkring-glm53-pr42-page-base-flight-qualification/v2"
BASE_CODEWORD = "base"
LANE_CODEWORDS = (
    "red",
    "blue",
    "green",
    "black",
    "white",
    "gold",
    "silver",
    "orange",
    "purple",
    "yellow",
    "brown",
    "gray",
    "pink",
    "cyan",
    "coral",
    "apple",
)
UNRELATED_CODEWORD = "quartz"
INSTRUCTION_TEMPLATE = "Reply with exactly the lowercase word {word}.\nAnswer:"
READINESS_RETRY_SECONDS = 1.0
READINESS_MAX_ATTEMPTS = 8
READINESS_TOTAL_SECONDS = 60.0
PROMPT_SEED = (
    "SparkCache deterministic token bank: alpha beta gamma delta epsilon zeta "
    "eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon."
)


class QualificationError(RuntimeError):
    """The qualification evidence is incomplete or contradictory."""


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


def _instruction(word: str) -> str:
    return INSTRUCTION_TEMPLATE.format(word=word)


def discover_instruction_tokens(
    endpoint: str, model: str, timeout: float
) -> dict[str, list[int]]:
    instructions: dict[str, list[int]] = {}
    for word in (BASE_CODEWORD, *LANE_CODEWORDS, UNRELATED_CODEWORD):
        document = _post(
            endpoint,
            "/tokenize",
            {"model": model, "prompt": _instruction(word)},
            timeout,
        )
        observed = document.get("tokens") or document.get("token_ids")
        if (
            not isinstance(observed, list)
            or len(observed) < 2
            or len(observed) > 64
            or any(not isinstance(value, int) or value < 0 for value in observed)
        ):
            raise QualificationError(
                f"tokenizer returned an invalid instruction for {word!r}"
            )
        instructions[word] = list(observed)
    return instructions


def _prompt_spec_sha256(instructions: dict[str, list[int]]) -> str:
    return _sha256(
        _canonical(
            {
                "instruction_template": INSTRUCTION_TEMPLATE,
                "base_codeword": BASE_CODEWORD,
                "lane_codewords": LANE_CODEWORDS,
                "unrelated_codeword": UNRELATED_CODEWORD,
                "instruction_tokens": instructions,
            }
        )
    )


def prompts(
    token_bank: list[int],
    instructions: dict[str, list[int]],
) -> tuple[list[int], list[list[int]], list[int]]:
    if len(token_bank) < PARTICIPANTS + 2:
        raise QualificationError("token bank must contain at least 18 distinct IDs")
    required_words = {BASE_CODEWORD, *LANE_CODEWORDS, UNRELATED_CODEWORD}
    if set(instructions) != required_words:
        raise QualificationError("instruction-token lanes are incomplete")
    common = token_bank[0]
    base_instruction = instructions[BASE_CODEWORD]
    base_fill = BASE_TOKENS - len(base_instruction) + 1
    if base_fill <= 0:
        raise QualificationError("base instruction exceeds the publication boundary")
    base_prefix = [common] * base_fill + base_instruction[:-1]
    base = base_prefix + [base_instruction[-1]]
    results = []
    for index, word in enumerate(LANE_CODEWORDS):
        instruction = instructions[word]
        private_fill = TAIL_TOKENS - len(instruction) + 1
        if private_fill <= 0:
            raise QualificationError(f"lane instruction {word!r} exceeds its tail")
        results.append(
            base_prefix + [token_bank[index + 1]] * private_fill + instruction
        )
    unrelated_instruction = instructions[UNRELATED_CODEWORD]
    unrelated_fill = 4096 - len(unrelated_instruction) + 1
    if unrelated_fill <= 0:
        raise QualificationError("unrelated instruction exceeds its request boundary")
    unrelated = [token_bank[-2]] * unrelated_fill + unrelated_instruction
    return base, results, unrelated


def _normalize_oracle(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().casefold()


def _completion(
    endpoint: str,
    model: str,
    token_ids: list[int],
    timeout: float,
    *,
    expected_oracle: str | None = None,
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
    receipt = {
        "http_status": 200,
        "prompt_tokens": len(token_ids),
        "prompt_sha256": _token_hash(token_ids),
        "response_sha256": _sha256(text.encode()),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choices[0].get("finish_reason"),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    if expected_oracle is not None:
        observed_oracle = _normalize_oracle(text)
        receipt.update(
            expected_oracle=expected_oracle,
            observed_oracle=observed_oracle,
            oracle_match=observed_oracle == expected_oracle,
        )
    return receipt


def _confirm_base_held_on_all_ranks(
    endpoint: str,
    model: str,
    token_bank: list[int],
    scheduler_log: Path,
    scheduler_log_offset: int,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + min(timeout, READINESS_TOTAL_SECONDS)
    last_counts: tuple[int, int] | None = None
    matched_line: str | None = None
    scheduler_steps: list[dict[str, Any]] = []
    next_trigger = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_trigger and len(scheduler_steps) < READINESS_MAX_ATTEMPTS:
            scheduler_steps.append(
                {
                    "attempt": len(scheduler_steps) + 1,
                    **_completion(
                        endpoint,
                        model,
                        [token_bank[-2], token_bank[-1]],
                        timeout,
                    ),
                }
            )
            next_trigger = time.monotonic() + READINESS_RETRY_SECONDS
        try:
            size = scheduler_log.stat().st_size
            offset = scheduler_log_offset if size >= scheduler_log_offset else 0
            with scheduler_log.open("rb") as stream:
                stream.seek(offset)
                observed = stream.read().decode("utf-8", errors="replace")
        except OSError as exc:
            raise QualificationError(
                f"scheduler log cannot be read: {scheduler_log}"
            ) from exc
        for line in observed.splitlines():
            if "KV Transfer metrics:" not in line:
                continue
            ranks = re.search(r"\bspark_cache_ranks_reporting=(\d+)\b", line)
            held = re.search(r"\bspark_cache_digests_held=(\d+)\b", line)
            if ranks is None or held is None:
                continue
            last_counts = (int(ranks.group(1)), int(held.group(1)))
            if last_counts == (TP_RANKS, TP_RANKS):
                matched_line = line
                break
        if matched_line is not None:
            break
        time.sleep(0.1)
    if matched_line is None:
        suffix = "no complete report observed"
        if last_counts is not None:
            suffix = (
                f"ranks_reporting={last_counts[0]} digests_held={last_counts[1]}"
            )
        raise QualificationError(f"base publication is not held on all ranks: {suffix}")
    return {
        "status": "verified",
        "required_ranks": TP_RANKS,
        "ranks_reporting": TP_RANKS,
        "digests_held": TP_RANKS,
        "scheduler_log_line_sha256": _sha256(matched_line.encode()),
        "scheduler_steps": scheduler_steps,
    }


def publish(
    endpoint: str,
    model: str,
    scheduler_log: Path,
    timeout: float,
) -> dict[str, Any]:
    try:
        scheduler_log_offset = scheduler_log.stat().st_size
    except OSError as exc:
        raise QualificationError(f"scheduler log cannot be read: {scheduler_log}") from exc
    bank = discover_token_bank(endpoint, model, timeout)
    instructions = discover_instruction_tokens(endpoint, model, timeout)
    base, results, _unrelated = prompts(bank, instructions)
    base_result = _completion(
        endpoint,
        model,
        base,
        timeout,
        expected_oracle=BASE_CODEWORD,
    )
    base_readiness = _confirm_base_held_on_all_ranks(
        endpoint,
        model,
        bank,
        scheduler_log,
        scheduler_log_offset,
        timeout,
    )
    result_receipts = [
        {
            "result_index": index,
            **_completion(
                endpoint,
                model,
                prompt,
                timeout,
                expected_oracle=word,
            ),
        }
        for index, (prompt, word) in enumerate(
            zip(results, LANE_CODEWORDS, strict=True)
        )
    ]
    oracle_mismatch_indices = [
        item["result_index"] for item in result_receipts if not item["oracle_match"]
    ]
    return {
        "schema": RECEIPT_SCHEMA,
        "kind": "publish",
        "status": "rejected" if oracle_mismatch_indices else "observed",
        "model": model,
        "token_bank_sha256": _token_hash(bank),
        "prompt_spec_sha256": _prompt_spec_sha256(instructions),
        "base_publication_tokens": BASE_TOKENS,
        "result_publication_tokens": RESULT_TOKENS,
        "private_tail_tokens": TAIL_TOKENS,
        "base_request": base_result,
        "base_readiness": base_readiness,
        "results": result_receipts,
        "oracle_mismatch_indices": oracle_mismatch_indices,
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
    instructions = discover_instruction_tokens(endpoint, model, timeout)
    prompt_spec_sha256 = _prompt_spec_sha256(instructions)
    if publish_receipt.get("prompt_spec_sha256") != prompt_spec_sha256:
        raise QualificationError("prompt specification differs from publication")
    _base, results, unrelated = prompts(bank, instructions)
    expected = [item["prompt_sha256"] for item in publish_receipt["results"]]
    observed = [_token_hash(item) for item in results]
    if observed != expected:
        raise QualificationError("reconstructed prompt hashes differ from publication")
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARTICIPANTS + 1) as pool:
        futures = [
            pool.submit(
                _completion,
                endpoint,
                model,
                prompt,
                timeout,
                expected_oracle=word,
            )
            for prompt, word in zip(results, LANE_CODEWORDS, strict=True)
        ]
        time.sleep(0.05)
        unrelated_future = pool.submit(
            _completion,
            endpoint,
            model,
            unrelated,
            timeout,
            expected_oracle=UNRELATED_CODEWORD,
        )
        receipts = [future.result() for future in futures]
        unrelated_receipt = unrelated_future.result()
    response_mismatch_indices: list[int] = []
    oracle_mismatch_indices: list[int] = []
    for index, item in enumerate(receipts):
        item["result_index"] = index
        if item["response_sha256"] != publish_receipt["results"][index][
            "response_sha256"
        ]:
            response_mismatch_indices.append(index)
        if not item["oracle_match"]:
            oracle_mismatch_indices.append(index)
    if not unrelated_receipt["oracle_match"]:
        oracle_mismatch_indices.append(PARTICIPANTS)
    return {
        "schema": RECEIPT_SCHEMA,
        "kind": "replay",
        "status": "rejected" if oracle_mismatch_indices else "verified",
        "model": model,
        "token_bank_sha256": _token_hash(bank),
        "prompt_spec_sha256": prompt_spec_sha256,
        "results": receipts,
        "unrelated_later_request": unrelated_receipt,
        "response_mismatch_indices": response_mismatch_indices,
        "oracle_mismatch_indices": oracle_mismatch_indices,
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
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise QualificationError("artifact receipt omits an exact image ID")
    expected_labels = {
        "org.sparkcache.source-revision": "a1511d26a1fe2b17b24561bc52e376bf7f54b06a",
        "org.sparkcache.source-tree": "4d5b8eb8c5c13793ee7a1e67b2b34bd38fcf4ddb",
        "org.sparkcache.source-sha256": "6651f2823c816fac93779cbca54a8f19c0ed262830953149f3a87d189d1f833b",
        "org.sparkcache.cuda-placement-library-sha256": "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c",
        "org.sparkcache.feature.page-base-read-flight": (
            "implemented-gpu-free-tested"
        ),
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
    if any(labels.get(name) != value for name, value in expected_labels.items()):
        raise QualificationError("artifact receipt labels differ from PR42")
    contract = artifact.get("page_base_restore_flight_contract")
    if contract is not None and (
        not isinstance(contract, dict) or contract.get("summary_schema") != FLIGHT_SCHEMA
    ):
        raise QualificationError("artifact receipt contains a contradictory contract")
    if semantic_document.get("status") != "verified":
        raise QualificationError("semantic canary is not verified")
    if published.get("schema") != RECEIPT_SCHEMA or replayed.get("schema") != RECEIPT_SCHEMA:
        raise QualificationError("publication and replay require qualification schema v2")
    readiness = published.get("base_readiness")
    if not isinstance(readiness, dict) or readiness.get("status") != "verified":
        raise QualificationError("base publication readiness is not verified")
    scheduler_steps = readiness.get("scheduler_steps")
    if (
        not isinstance(scheduler_steps, list)
        or not scheduler_steps
        or len(scheduler_steps) > READINESS_MAX_ATTEMPTS
        or [item.get("attempt") for item in scheduler_steps]
        != list(range(1, len(scheduler_steps) + 1))
    ):
        raise QualificationError("base readiness scheduler attempts are incomplete")
    if replayed.get("status") != "verified":
        raise QualificationError("replay receipt is not verified")
    if published.get("prompt_spec_sha256") != replayed.get("prompt_spec_sha256"):
        raise QualificationError("publication and replay prompt specifications differ")
    if published.get("oracle_mismatch_indices") or replayed.get(
        "oracle_mismatch_indices"
    ):
        raise QualificationError("publication or replay has an oracle mismatch")
    published_results = published.get("results", [])
    replayed_results = replayed.get("results", [])
    if len(published_results) != PARTICIPANTS or len(replayed_results) != PARTICIPANTS:
        raise QualificationError("publication and replay must each contain 16 results")
    if [item["prompt_sha256"] for item in published_results] != [
        item["prompt_sha256"] for item in replayed_results
    ]:
        raise QualificationError("publication and replay prompt hashes differ")
    for index, word in enumerate(LANE_CODEWORDS):
        for label, item in (
            ("publication", published_results[index]),
            ("replay", replayed_results[index]),
        ):
            if (
                item.get("expected_oracle") != word
                or item.get("observed_oracle") != word
                or item.get("oracle_match") is not True
            ):
                raise QualificationError(
                    f"{label} result {index} does not match oracle {word!r}"
                )
    unrelated = replayed.get("unrelated_later_request")
    if (
        not isinstance(unrelated, dict)
        or unrelated.get("http_status") != 200
        or unrelated.get("expected_oracle") != UNRELATED_CODEWORD
        or unrelated.get("observed_oracle") != UNRELATED_CODEWORD
        or unrelated.get("oracle_match") is not True
    ):
        raise QualificationError("unrelated request oracle is not verified")
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
        elif name == "publish":
            command.add_argument("--scheduler-log", type=Path, required=True)
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
        document = publish(args.endpoint, args.model, args.scheduler_log, args.timeout)
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
