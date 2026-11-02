"""Deterministic HTTP store/restart/restore qualification mechanics."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

EXPECTED_LONG = "SPARKCACHE_OK:9540"
EXPECTED_SHORT = "SPARKCACHE_CANARY_OK"
CANARY_PROMPT = "Respond with exactly SPARKCACHE_CANARY_OK and no other text."

# This identifier is a frozen compatibility interface used by existing
# DeepSeek-V4 and GLM-5.2 reference files. Its model-specific spelling does
# not constrain the model-neutral qualification mechanics.
SEMANTIC_REFERENCE_SCHEMA = "sparkcache-deepseek-semantic-reference/v1"

RequestFunction = Callable[[str, str, str, int], dict[str, Any]]
ContentReader = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class AssistantCompletion:
    """Normalized assistant evidence and final answer from one response."""

    body: str
    content: str
    finish_reason: str | None

    def require_conclusive(self, check: str) -> str:
        """Return the body unless response metadata cannot prove completion."""

        if self.finish_reason == "length":
            raise SemanticGateInconclusive(
                check,
                "completion token limit reached",
                completion=self,
            )
        if not self.body:
            raise SemanticGateInconclusive(
                check,
                "completion has no non-whitespace assistant body",
                completion=self,
            )
        return self.body


class SemanticGateInconclusive(RuntimeError):
    """A semantic check whose response cannot establish pass or failure."""

    status = "INCONCLUSIVE"

    def __init__(
        self,
        check: str,
        reason: str,
        *,
        completion: AssistantCompletion,
    ) -> None:
        super().__init__(f"{check} inconclusive: {reason}")
        self.check = check
        self.reason = reason
        self.completion = completion

    def as_result(self) -> dict[str, Any]:
        """Return stable JSON-compatible evidence for a semantic-check report."""

        return {
            "assistant_body_present": bool(self.completion.body),
            "check": self.check,
            "finish_reason": self.completion.finish_reason,
            "reason": self.reason,
            "status": self.status,
        }


def build_long_prompt(records: int = 384) -> str:
    """Build a stable multi-thousand-token fact-retrieval prompt."""

    if not 384 <= records <= 32768:
        raise ValueError("records must be in [384, 32768]")
    alpha_index = records * 17 // 384
    omega_index = records * 311 // 384
    index_width = max(4, len(str(records - 1)))
    archive = []
    for index in range(records):
        if index == alpha_index:
            text = "the alpha cache marker has value 3719"
        elif index == omega_index:
            text = "the omega cache marker has value 5821"
        else:
            text = "the archival copper marker is stable and carries no numeric value"
        archive.append(f"Archive record {index:0{index_width}d}: {text}.")
    return (
        "Read the archive below. Retain the two explicitly numeric cache marker "
        "values; all other records are padding.\n\n"
        + "\n".join(archive)
        + "\n\nAdd the alpha and omega cache marker values. Respond with exactly "
        "SPARKCACHE_OK:<sum>, with no other text."
    )


def request_chat(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Send one deterministic OpenAI-compatible chat request.

    ``extra_body`` merges fixed model-specific request fields (for example a
    reasoning toggle) into the JSON body after the deterministic defaults.
    """

    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if extra_body:
        body.update(extra_body)
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:  # noqa: S310
        return json.load(response)


def assistant_completion(response: dict[str, Any]) -> AssistantCompletion:
    """Normalize reasoning and answer fields from one chat response.

    OpenAI-compatible servers expose reasoning under either ``reasoning`` or
    ``reasoning_content``. The normalized body concatenates ``reasoning``,
    ``reasoning_content``, and ``content`` in that order before trimming outer
    whitespace. The completion's ``finish_reason`` remains separate so a
    token-limited response cannot be mistaken for a complete answer.
    """

    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("completion response has no assistant content") from error
    if not isinstance(message, dict):
        raise RuntimeError("completion response assistant message is not an object")
    parts: list[str] = []
    for field in ("reasoning", "reasoning_content", "content"):
        value = message.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise RuntimeError(
                f"completion response assistant {field} is not a string"
            )
        parts.append(value)
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise RuntimeError("completion response finish_reason is not a string")
    content = message.get("content")
    return AssistantCompletion(
        body="".join(parts).strip(),
        content=content.strip() if isinstance(content, str) else "",
        finish_reason=finish_reason,
    )


def assistant_content(response: dict[str, Any]) -> str:
    """Return the normalized final assistant answer."""

    return assistant_completion(response).content


def _conclusive_completion(
    response: dict[str, Any],
    *,
    check: str,
    content_reader: ContentReader,
) -> AssistantCompletion:
    parsed = assistant_completion(response)
    body = content_reader(response)
    if not isinstance(body, str):
        raise RuntimeError("completion content reader did not return a string")
    completion = AssistantCompletion(
        body=parsed.body,
        content=body.strip(),
        finish_reason=parsed.finish_reason,
    )
    completion.require_conclusive(check)
    return completion


def run_semantic_miss(
    endpoint: str,
    model: str,
    reference: Path,
    *,
    long_max_tokens: int = 32,
    records: int = 384,
    request: RequestFunction = request_chat,
    content_reader: ContentReader = assistant_content,
) -> dict[str, Any]:
    """Store the deterministic miss response and its prompt identity."""

    prompt = build_long_prompt(records)
    response = request(endpoint, model, prompt, long_max_tokens)
    completion = _conclusive_completion(
        response,
        check="long semantic miss",
        content_reader=content_reader,
    )
    content = completion.content
    if content != EXPECTED_LONG:
        raise RuntimeError(f"long semantic miss failed: {content!r}")
    result = {
        "schema": SEMANTIC_REFERENCE_SCHEMA,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "records": records,
        "assistant_body": completion.body,
        "content": content,
        "usage": response.get("usage"),
    }
    reference.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def run_semantic_hit(
    endpoint: str,
    model: str,
    reference: Path,
    *,
    long_max_tokens: int = 32,
    short_max_tokens: int = 32,
    records: int | None = None,
    request: RequestFunction = request_chat,
    content_reader: ContentReader = assistant_content,
) -> dict[str, Any]:
    """Require prompt identity, restored output, and a post-restore canary."""

    expected = json.loads(reference.read_text(encoding="utf-8"))
    if expected.get("schema") != SEMANTIC_REFERENCE_SCHEMA:
        raise RuntimeError("semantic reference has an unsupported schema")
    expected_records = expected.get("records", 384)
    if records is not None and records != expected_records:
        raise RuntimeError("semantic reference record count differs")
    prompt = build_long_prompt(int(expected_records))
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if expected.get("prompt_sha256") != digest:
        raise RuntimeError("semantic reference prompt identity differs")
    prime_completion = _conclusive_completion(
        request(endpoint, model, CANARY_PROMPT, short_max_tokens),
        check="manifest-inventory publication canary",
        content_reader=content_reader,
    )
    prime_content = prime_completion.content
    if prime_content != EXPECTED_SHORT:
        raise RuntimeError(
            "manifest-inventory publication canary failed: "
            f"{prime_content!r}"
        )
    completion = _conclusive_completion(
        request(endpoint, model, prompt, long_max_tokens),
        check="long semantic hit",
        content_reader=content_reader,
    )
    content = completion.content
    if content != EXPECTED_LONG or content != expected.get("content"):
        raise RuntimeError(f"long semantic hit failed: {content!r}")
    expected_body = expected.get("assistant_body")
    if expected_body is not None and completion.body != expected_body:
        raise RuntimeError("long semantic hit assistant body differs")
    canary_completion = _conclusive_completion(
        request(endpoint, model, CANARY_PROMPT, short_max_tokens),
        check="post-restore semantic canary",
        content_reader=content_reader,
    )
    canary_content = canary_completion.content
    if canary_content != EXPECTED_SHORT:
        raise RuntimeError(f"post-restore semantic canary failed: {canary_content!r}")
    return {"content": content, "post_restore_canary": canary_content}
