#!/usr/bin/env python3
"""Exact request-payload sealing for ``llm_decode_bench.py``.

The benchmark historically creates a random run marker for every invocation.
That is useful for defeating accidental prefix-cache reuse, but it also means
two performance arms do not receive byte-identical prompts.  This module is a
small, dependency-free integration seam which records and replays the exact
JSON request bodies constructed by the benchmark.

The manifest is deliberately an envelope around immutable ``content``.  Its
SHA-256 is calculated over canonical UTF-8 JSON for that content, while every
individual request payload has its own canonical SHA-256.  Import validates all
commitments before returning any payload and consumes calls in order.  A run is
sealed only after :meth:`PromptManifestSession.finish` proves that the complete
manifest was consumed.

This module is OFFLINE by itself: it only reads or writes the named manifest.
The benchmark integration which calls it owns any network activity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "sparkring-llm-decode-prompt-manifest/v1"
SUMMARY_SCHEMA = "sparkring-llm-decode-prompt-manifest-summary/v1"
HARNESS_NAME = "llm_decode_bench.py"
HARNESS_VERSION = "0.4.31"
WORKLOAD_FIELDS = frozenset(
    {
        "harness",
        "model",
        "contexts",
        "concurrencies",
        "duration_seconds",
        "max_tokens",
        "temperature",
        "reasoning_effort",
        "ignore_eos",
        "unique_context_percent",
        "decode_warmup_seconds",
        "cell_warmup_timeout_seconds",
        "dcp_size",
        "token_targeting",
    }
)
CALL_DESCRIPTOR_FIELDS = frozenset(
    {"phase", "request_kind", "context_tokens", "concurrency", "benchmark_mode"}
)
SAFE_MODEL_IDENTIFIER = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}(?:/[A-Za-z0-9][A-Za-z0-9._+-]{0,127})?"
)


class PromptManifestError(ValueError):
    """A prompt manifest or attempted replay is invalid."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PromptManifestError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise PromptManifestError(f"non-finite JSON number {value!r} is unsupported")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one byte representation used by all commitments."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise PromptManifestError(f"value is not canonical JSON: {error}") from error
    return encoded.encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _encoded_document(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _write_exclusive(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(_encoded_document(document))
        output.flush()
        os.fsync(output.fileno())


def _replace_atomic(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(_encoded_document(document))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PromptManifestError(f"{label} must be a JSON object")
    return value


def _require_payloads(value: Any, concurrency: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != concurrency:
        raise PromptManifestError(
            f"payloads must contain exactly {concurrency} stream entries"
        )
    payloads: list[dict[str, Any]] = []
    for stream_index, item in enumerate(value):
        item = _require_object(item, f"payloads[{stream_index}]")
        if item.get("stream_index") != stream_index:
            raise PromptManifestError(
                f"payloads[{stream_index}] has a non-canonical stream_index"
            )
        payload = _require_object(item.get("payload"), f"payloads[{stream_index}].payload")
        expected = sha256_json(payload)
        if item.get("payload_sha256") != expected:
            raise PromptManifestError(
                f"payloads[{stream_index}] payload_sha256 does not match its payload"
            )
        payloads.append(copy.deepcopy(payload))
    return payloads


def _non_prompt_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return request settings which replay is not allowed to silently change."""

    return {key: copy.deepcopy(value) for key, value in payload.items() if key != "messages"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_workload_projection(value: Any) -> dict[str, Any]:
    """Return the exact non-private benchmark workload or fail closed.

    A manifest can seal arbitrary request bodies, including private prompts.
    ``load_summary`` is intended for sanitized evidence, so it exposes only the
    fixed v0.4.31 benchmark descriptor and rejects extra fields rather than
    accidentally copying a host, path, credential, or operator note.
    """

    workload = _require_object(value, "manifest.content.workload")
    fields = set(workload)
    if fields != WORKLOAD_FIELDS:
        missing = sorted(WORKLOAD_FIELDS - fields)
        unexpected = sorted(fields - WORKLOAD_FIELDS)
        detail = []
        if missing:
            detail.append(f"missing {missing!r}")
        if unexpected:
            detail.append(f"unexpected/private {unexpected!r}")
        raise PromptManifestError(
            "manifest workload is not the safe exact v0.4.31 projection: "
            + "; ".join(detail)
        )
    if workload["harness"] != {"name": HARNESS_NAME, "version": HARNESS_VERSION}:
        raise PromptManifestError(
            f"manifest workload harness must be {HARNESS_NAME} v{HARNESS_VERSION}"
        )
    model = workload["model"]
    if (
        not isinstance(model, str)
        or SAFE_MODEL_IDENTIFIER.fullmatch(model) is None
        or model in {".", ".."}
        or "/." in model
    ):
        raise PromptManifestError(
            "manifest workload model must be a safe served-model identifier, not a path"
        )
    for key in ("contexts", "concurrencies"):
        values = workload[key]
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 1
                for item in values
            )
        ):
            raise PromptManifestError(f"manifest workload {key} must be positive integers")
    for key in ("duration_seconds", "decode_warmup_seconds", "cell_warmup_timeout_seconds"):
        number = workload[key]
        if not _is_number(number) or number < 0:
            raise PromptManifestError(f"manifest workload {key} must be non-negative")
    for key in ("max_tokens", "dcp_size"):
        number = workload[key]
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise PromptManifestError(
                f"manifest workload {key} must be a positive integer"
            )
    for key in ("temperature", "unique_context_percent"):
        if not _is_number(workload[key]):
            raise PromptManifestError(f"manifest workload {key} must be numeric")
    if not 0 <= workload["unique_context_percent"] <= 100:
        raise PromptManifestError(
            "manifest workload unique_context_percent must be between 0 and 100"
        )
    if workload["reasoning_effort"] is not None:
        raise PromptManifestError(
            "manifest workload reasoning_effort must be null for the sealed C8 contract"
        )
    if not isinstance(workload["ignore_eos"], bool):
        raise PromptManifestError("manifest workload ignore_eos must be boolean")
    if workload["token_targeting"] not in {"estimate", "exact"}:
        raise PromptManifestError(
            "manifest workload token_targeting must be 'estimate' or 'exact'"
        )
    return copy.deepcopy(workload)


def _safe_call_descriptor_projection(calls: Any) -> list[dict[str, Any]]:
    if not isinstance(calls, list) or not calls:
        raise PromptManifestError("manifest must contain at least one ordered call")
    projection: list[dict[str, Any]] = []
    for sequence, call_value in enumerate(calls):
        call = _require_object(call_value, f"calls[{sequence}]")
        descriptor = _require_object(call.get("descriptor"), f"calls[{sequence}].descriptor")
        fields = set(descriptor)
        if fields != CALL_DESCRIPTOR_FIELDS:
            missing = sorted(CALL_DESCRIPTOR_FIELDS - fields)
            unexpected = sorted(fields - CALL_DESCRIPTOR_FIELDS)
            detail = []
            if missing:
                detail.append(f"missing {missing!r}")
            if unexpected:
                detail.append(f"unexpected/private {unexpected!r}")
            raise PromptManifestError(
                f"calls[{sequence}] descriptor is not the safe exact projection: "
                + "; ".join(detail)
            )
        if descriptor["phase"] not in {
            "decode-warmup",
            "decode-measured",
            "burst-measured",
        }:
            raise PromptManifestError(f"calls[{sequence}] has invalid phase")
        if descriptor["request_kind"] not in {
            "prefix-scout",
            "decode-stream-template",
        }:
            raise PromptManifestError(f"calls[{sequence}] has invalid request_kind")
        if descriptor["benchmark_mode"] not in {"duration", "request-count"}:
            raise PromptManifestError(f"calls[{sequence}] has invalid benchmark_mode")
        for key in ("context_tokens", "concurrency"):
            number = descriptor[key]
            if not isinstance(number, int) or isinstance(number, bool) or number < 1:
                raise PromptManifestError(f"calls[{sequence}] has invalid {key}")
        projection.append(copy.deepcopy(descriptor))
    return projection


def _validate_payload_semantics(
    calls: list[dict[str, Any]], workload: Mapping[str, Any]
) -> None:
    """Bind every sealed request's public settings to the workload claim."""

    for sequence, call in enumerate(calls):
        descriptor = call["descriptor"]
        payloads = _require_payloads(call["payloads"], descriptor["concurrency"])
        expected: dict[str, Any] = {
            "model": workload["model"],
            "stream": True,
            "max_tokens": (
                1
                if descriptor["request_kind"] == "prefix-scout"
                else workload["max_tokens"]
            ),
            "temperature": workload["temperature"],
            "stream_options": (
                {"include_usage": True}
                if descriptor["request_kind"] == "prefix-scout"
                else {"include_usage": True, "continuous_usage_stats": True}
            ),
        }
        if workload["ignore_eos"]:
            expected["ignore_eos"] = True
        if workload["reasoning_effort"] is not None:
            expected["reasoning_effort"] = workload["reasoning_effort"]
        for stream_index, payload in enumerate(payloads):
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise PromptManifestError(
                    f"calls[{sequence}] stream {stream_index} messages are missing"
                )
            if _non_prompt_payload(payload) != expected:
                raise PromptManifestError(
                    f"calls[{sequence}] stream {stream_index} request settings "
                    "do not match the sealed workload"
                )


def _validate_document(document: Any) -> dict[str, Any]:
    document = _require_object(document, "manifest")
    if document.get("schema") != SCHEMA:
        raise PromptManifestError(f"manifest schema must be {SCHEMA!r}")
    content = _require_object(document.get("content"), "manifest.content")
    commitment = document.get("content_sha256")
    if not isinstance(commitment, str) or commitment != sha256_json(content):
        raise PromptManifestError("manifest content_sha256 does not match content")
    if content.get("complete") is not True:
        raise PromptManifestError("manifest is incomplete and cannot be replayed")
    descriptor = _require_object(content.get("workload"), "manifest.content.workload")
    canonical_json_bytes(descriptor)
    calls = content.get("calls")
    if not isinstance(calls, list) or not calls:
        raise PromptManifestError("manifest must contain at least one ordered call")
    for sequence, call in enumerate(calls):
        call = _require_object(call, f"calls[{sequence}]")
        if call.get("sequence") != sequence:
            raise PromptManifestError(f"calls[{sequence}] has a non-canonical sequence")
        call_descriptor = _require_object(call.get("descriptor"), f"calls[{sequence}].descriptor")
        concurrency = call_descriptor.get("concurrency")
        if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
            raise PromptManifestError(f"calls[{sequence}] has invalid concurrency")
        payloads = _require_payloads(call.get("payloads"), concurrency)
        expected_set = sha256_json(
            [sha256_json(payload) for payload in payloads]
        )
        if call.get("payload_set_sha256") != expected_set:
            raise PromptManifestError(f"calls[{sequence}] payload_set_sha256 is invalid")
    expected_sequence = sha256_json(
        [call["payload_set_sha256"] for call in calls]
    )
    if content.get("payload_sequence_sha256") != expected_sequence:
        raise PromptManifestError("manifest payload_sequence_sha256 is invalid")
    return document


class PromptManifestSession:
    """Ordered exact-payload export or replay session."""

    def __init__(
        self,
        *,
        mode: str,
        path: Path,
        workload: Mapping[str, Any],
        document: dict[str, Any],
    ) -> None:
        self.mode = mode
        self.path = path
        self.workload = copy.deepcopy(dict(workload))
        self.document = document
        self._cursor = 0
        self._finished = False

    @classmethod
    def export(cls, path: str | Path, workload: Mapping[str, Any]) -> "PromptManifestSession":
        output = Path(path)
        canonical_json_bytes(workload)
        _safe_workload_projection(dict(workload))
        content: dict[str, Any] = {
            "complete": False,
            "workload": copy.deepcopy(dict(workload)),
            "calls": [],
            "payload_sequence_sha256": None,
        }
        document = {
            "schema": SCHEMA,
            "content": content,
            "content_sha256": sha256_json(content),
        }
        _write_exclusive(output, document)
        return cls(mode="export", path=output, workload=workload, document=document)

    @classmethod
    def import_(cls, path: str | Path, workload: Mapping[str, Any]) -> "PromptManifestSession":
        source = Path(path)
        try:
            document = json.loads(
                source.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_constant,
            )
        except (OSError, json.JSONDecodeError, PromptManifestError) as error:
            raise PromptManifestError(f"cannot read prompt manifest: {error}") from error
        document = _validate_document(document)
        expected_workload = document["content"]["workload"]
        _safe_workload_projection(expected_workload)
        _safe_call_descriptor_projection(document["content"]["calls"])
        if canonical_json_bytes(dict(workload)) != canonical_json_bytes(expected_workload):
            raise PromptManifestError(
                "current benchmark workload does not exactly match the sealed workload"
            )
        return cls(mode="import", path=source, workload=workload, document=document)

    @property
    def content_sha256(self) -> str:
        return str(self.document["content_sha256"])

    def resolve_call(
        self,
        descriptor: Mapping[str, Any],
        generated_payloads: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Record or replay one ordered benchmark call.

        ``descriptor`` must include the exact concurrency.  During import the
        currently generated payloads are still checked after removing only
        ``messages``.  Thus prompt bytes are replayed, while model, sampling,
        streaming, output limit and all other request settings must remain
        exactly equal to the sealed run.
        """

        if self._finished:
            raise PromptManifestError("prompt manifest session is already finished")
        descriptor = copy.deepcopy(dict(descriptor))
        canonical_json_bytes(descriptor)
        concurrency = descriptor.get("concurrency")
        if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
            raise PromptManifestError("call descriptor concurrency must be a positive integer")
        if len(generated_payloads) != concurrency:
            raise PromptManifestError(
                "generated payload count does not equal descriptor concurrency"
            )
        generated = [copy.deepcopy(dict(payload)) for payload in generated_payloads]
        canonical_json_bytes(generated)

        if self.mode == "export":
            entries = [
                {
                    "stream_index": index,
                    "payload": payload,
                    "payload_sha256": sha256_json(payload),
                }
                for index, payload in enumerate(generated)
            ]
            call = {
                "sequence": self._cursor,
                "descriptor": descriptor,
                "payloads": entries,
                "payload_set_sha256": sha256_json(
                    [entry["payload_sha256"] for entry in entries]
                ),
            }
            self.document["content"]["calls"].append(call)
            self._cursor += 1
            self.document["content_sha256"] = sha256_json(self.document["content"])
            _replace_atomic(self.path, self.document)
            return generated

        calls = self.document["content"]["calls"]
        if self._cursor >= len(calls):
            raise PromptManifestError("benchmark attempted an extra unsealed call")
        call = calls[self._cursor]
        if canonical_json_bytes(call["descriptor"]) != canonical_json_bytes(descriptor):
            raise PromptManifestError(
                f"call {self._cursor} descriptor differs from the sealed call"
            )
        replay = _require_payloads(call["payloads"], concurrency)
        for index, (current, sealed) in enumerate(zip(generated, replay)):
            if canonical_json_bytes(_non_prompt_payload(current)) != canonical_json_bytes(
                _non_prompt_payload(sealed)
            ):
                raise PromptManifestError(
                    f"call {self._cursor} stream {index} non-prompt request settings drifted"
                )
        self._cursor += 1
        return replay

    def finish(self) -> dict[str, Any]:
        """Seal export or prove complete consumption, returning result metadata."""

        if self._finished:
            raise PromptManifestError("prompt manifest session was finished twice")
        content = self.document["content"]
        if self.mode == "export":
            if not content["calls"]:
                raise PromptManifestError("cannot finish an empty prompt manifest")
            content["payload_sequence_sha256"] = sha256_json(
                [call["payload_set_sha256"] for call in content["calls"]]
            )
            content["complete"] = True
            self.document["content_sha256"] = sha256_json(content)
            _replace_atomic(self.path, self.document)
        elif self._cursor != len(content["calls"]):
            raise PromptManifestError(
                f"only {self._cursor} of {len(content['calls'])} sealed calls were consumed"
            )
        self._finished = True
        return {
            "schema": SUMMARY_SCHEMA,
            "manifest_schema": SCHEMA,
            "mode": self.mode,
            "content_sha256": self.document["content_sha256"],
            "payload_sequence_sha256": content["payload_sequence_sha256"],
            "call_count": len(content["calls"]),
            "consumed_call_count": self._cursor,
            "consumed_all": self._cursor == len(content["calls"]),
        }


def load_summary(path: str | Path) -> dict[str, Any]:
    """Validate a complete manifest and return its safe public summary."""

    try:
        document = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, json.JSONDecodeError, PromptManifestError) as error:
        raise PromptManifestError(f"cannot read prompt manifest: {error}") from error
    document = _validate_document(document)
    content = document["content"]
    workload = _safe_workload_projection(content["workload"])
    call_descriptors = _safe_call_descriptor_projection(content["calls"])
    _validate_payload_semantics(content["calls"], workload)
    return {
        "schema": SUMMARY_SCHEMA,
        "manifest_schema": SCHEMA,
        "content_sha256": document["content_sha256"],
        "payload_sequence_sha256": content["payload_sequence_sha256"],
        "call_count": len(content["calls"]),
        "workload": workload,
        "ordered_call_descriptors": call_descriptors,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a sealed llm_decode_bench prompt manifest")
    parser.add_argument("manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        print(json.dumps(load_summary(args.manifest), indent=2, sort_keys=True))
    except PromptManifestError as error:
        print(f"prompt manifest invalid: {error}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
