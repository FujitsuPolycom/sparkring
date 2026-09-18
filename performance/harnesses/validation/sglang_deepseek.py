#!/usr/bin/env python3
"""Bounded SGLang authentication, API and optional long-prompt retrieval checks.

Uses no GPU API or lifecycle command. The optional retrieval request sends exact
input IDs from the checkpoint's tokenizer and encoding/encoding.py chat format.
Results describe these synthetic requests; they do not establish general quality.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


MAX_OUTPUT_TOKENS = 32
EXPECTED_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
MODEL_CONFIG_SHA256 = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
TOKENIZER_SHA256 = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"
CHAT_ENCODER_SHA256 = "502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward a bearer credential to a redirect destination."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def request(base, path, key, payload=None, timeout=900):
    headers = {"Accept": "application/json"}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode()
    started = time.monotonic()
    req = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(req, timeout=timeout) as response:
            status = response.status
            # Generated outputs are capped at 32 tokens. Reject unexpectedly
            # large responses instead of recording arbitrary server content.
            raw = response.read(1024 * 1024 + 1)
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read(16384)
    elapsed = time.monotonic() - started
    require(len(raw) <= 1024 * 1024, "API response exceeded the harness size limit")
    try:
        body = json.loads(raw) if raw else None
    except (ValueError, UnicodeError):
        body = None
    return status, body, {"http_status": status, "elapsed_seconds": round(elapsed, 4),
                          "response_sha256": hashlib.sha256(raw).hexdigest()}


def chat_payload(model, content, **extra):
    return {"model": model, "temperature": 0, "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False, "chat_template_kwargs": {"thinking": False},
            "messages": [{"role": "user", "content": content}], **extra}


def completion_details(body):
    require(isinstance(body, dict), "completion body was not a JSON object")
    choices = body.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "expected exactly one completion")
    choice = choices[0]
    text = choice.get("message", {}).get("content")
    require(isinstance(text, str), "completion did not contain text")
    usage = body.get("usage") or {}
    return text.strip(), {"finish_reason": choice.get("finish_reason"),
                          "usage": usage, "output": text[:256]}


def make_long_prompt(model_path, target, context_limit):
    require(target >= 1024, "target prompt must contain at least 1024 tokens")
    require(target + MAX_OUTPUT_TOKENS + 256 <= context_limit,
            "target leaves insufficient space for the output and tokenization margin")
    model = Path(model_path).resolve()
    require(digest(model / "config.json") == MODEL_CONFIG_SHA256,
            "model config differs from the selected DeepSeek checkpoint")
    tokenizer_path = model / "tokenizer.json"
    encoding_root = model / "encoding"
    require((encoding_root / "encoding.py").is_file(), "checkpoint chat encoder is missing")
    require(digest(tokenizer_path) == TOKENIZER_SHA256,
            "tokenizer differs from the selected DeepSeek checkpoint")
    require(digest(encoding_root / "encoding.py") == CHAT_ENCODER_SHA256,
            "chat encoder differs from the selected DeepSeek checkpoint")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sys.path.insert(0, str(encoding_root))
    try:
        encoder = importlib.import_module("encoding")
        require(Path(encoder.__file__).resolve() == (encoding_root / "encoding.py").resolve(),
                "chat encoder resolved outside the selected checkpoint")
        encode_messages = encoder.encode_messages
    finally:
        sys.path.pop(0)
    nonce = secrets.token_hex(16)
    answer = "IRIS-" + secrets.token_hex(8).upper()
    needle = "\nThe retrieval code for record LANTERN is " + answer + ".\n"
    prefix = ("Request nonce: " + nonce + ".\nRead the reference archive. "
              "Its single LANTERN record contains the answer to the final question.\n")
    filler = ("Reference entry: equipment inspections check connectors, inventory counts, "
              "document labels and maintenance intervals. This entry contains no retrieval code.\n")
    suffix = ("\nWhat is the retrieval code for record LANTERN? "
              "Return only the exact code, without explanation or punctuation.\n")
    estimate = max(1, len(tokenizer.encode(filler, add_special_tokens=False).ids))
    repeats = max(1, (target - 256) // estimate)
    started = time.monotonic()
    selected = None
    for _ in range(12):
        content = prefix + filler * (repeats // 2) + needle + filler * (repeats - repeats // 2) + suffix
        formatted = encode_messages([{"role": "user", "content": content}], thinking_mode="chat")
        require(isinstance(formatted, str), "checkpoint chat encoder did not return text")
        ids = tokenizer.encode(formatted, add_special_tokens=False).ids
        if target - 256 <= len(ids) <= target:
            selected = (formatted, ids)
            break
        delta = target - len(ids)
        adjustment = delta // estimate if delta < 0 else max(1, delta // estimate)
        repeats = max(1, repeats + adjustment)
    require(selected is not None, "could not construct a prompt within 256 tokens of the target")
    formatted, ids = selected
    require(len(ids) + MAX_OUTPUT_TOKENS <= context_limit, "constructed request exceeds context limit")
    needle_start = formatted.index(needle)
    needle_token = len(tokenizer.encode(formatted[:needle_start], add_special_tokens=False).ids)
    token_bytes = json.dumps(ids, separators=(",", ":")).encode()
    return ids, answer, {"checkpoint_revision_expected": EXPECTED_REVISION,
        "config_sha256": digest(model / "config.json"),
        "tokenizer_sha256": digest(tokenizer_path),
        "chat_encoder_sha256": digest(encoding_root / "encoding.py"),
        "target_prompt_tokens": target, "input_tokens": len(ids),
        "context_limit": context_limit, "output_budget_tokens": MAX_OUTPUT_TOKENS,
        "needle_token_offset": needle_token, "needle_depth": round(needle_token / len(ids), 6),
        "input_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "request_nonce": nonce, "expected_answer": answer,
        "construction_seconds": round(time.monotonic() - started, 4)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--target-prompt-tokens", type=int, default=640000)
    parser.add_argument("--context-limit", type=int, default=655360)
    args = parser.parse_args(argv)
    base = args.url.rstrip("/")
    if "://" not in base:
        base = "http://" + base
    if base.endswith("/v1"):
        base = base[:-3]
    parsed = urllib.parse.urlparse(base)
    require(parsed.scheme in ("http", "https") and parsed.hostname and not parsed.username
            and not parsed.password and parsed.path in ("", "/") and not parsed.query and not parsed.fragment,
            "URL must identify an HTTP API root without credentials or query parameters")
    require(not args.output.exists(), "output file already exists; choose an unused result path")
    keys = [line.strip() for line in args.key_file.read_text().splitlines() if line.strip()]
    require(len(keys) >= 2 and len(set(keys)) == len(keys), "provide at least two distinct API keys")
    require(all("," not in key and all(33 <= ord(c) <= 126 for c in key) for key in keys),
            "API keys must contain no whitespace, control characters or commas")
    os.umask(0o077)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "sparkring-sglang-bounded-api-check/v1", "started_at": now(),
        "endpoint": base, "model": args.model, "request_timeout_seconds": 900,
        "max_output_tokens": MAX_OUTPUT_TOKENS, "long_context_requested": args.long_context,
        "scope": "Sequential synthetic API checks; no throughput, soak or general quality qualification.",
        "tests": [], "complete": False}
    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    def case(name, check):
        result = {"name": name, "started_at": now()}
        started = time.monotonic()
        try:
            result.update(check() or {})
            result["passed"] = True
        except Exception as error:
            message = str(error)
            for key in keys:
                message = message.replace(key, "[REDACTED]")
            result.update(passed=False, error_type=type(error).__name__, error=message[:512])
        result["case_seconds"] = round(time.monotonic() - started, 4)
        report["tests"].append(result)
        save()
        print(json.dumps({"name": name, "passed": result["passed"],
                          "seconds": result["case_seconds"]}), flush=True)
        return result["passed"]
    def health():
        status, _, details = request(base, "/health", None)
        require(status == 200, "health endpoint returned HTTP " + str(status))
        return details
    def models(key):
        status, body, details = request(base, "/v1/models", key)
        require(status == 200 and isinstance(body, dict), "model listing failed with HTTP " + str(status))
        names = [item.get("id") for item in body.get("data", []) if isinstance(item, dict)]
        require(args.model in names, "expected model identifier absent from model listing")
        return {**details, "model_ids": names}
    def denial(key, generation):
        payload = chat_payload(args.model, "Reply with the number 42.") if generation else None
        path = "/v1/chat/completions" if generation else "/v1/models"
        status, _, details = request(base, path, key, payload)
        require(status in (401, 403), "unauthorized request returned HTTP " + str(status))
        return {**details, "path": path}
    short_outputs = []
    def short(key):
        status, body, details = request(base, "/v1/chat/completions", key,
            chat_payload(args.model, "What is 19 + 23? Reply only with the number."))
        require(status == 200, "generation returned HTTP " + str(status))
        text, completion = completion_details(body)
        require(text == "42", "arithmetic answer was not exactly 42")
        require(completion["finish_reason"] == "stop", "arithmetic response did not finish normally")
        short_outputs.append(text)
        return {**details, **completion}
    def structured():
        schema = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True,
            "schema": {"type": "object", "properties": {"answer": {"type": "integer"}},
                       "required": ["answer"], "additionalProperties": False}}}
        status, body, details = request(base, "/v1/chat/completions", keys[0],
            chat_payload(args.model, "Return an object whose answer is the integer 42.", response_format=schema))
        require(status == 200, "structured generation returned HTTP " + str(status))
        text, completion = completion_details(body)
        require(json.loads(text) == {"answer": 42}, "structured output differs from the expected object")
        require(completion["finish_reason"] == "stop", "structured response did not finish normally")
        return {**details, **completion, "protocol": "response_format.json_schema"}
    save()
    ready = case("health", health)
    for index, key in enumerate(keys[:2], 1):
        ready = case("model_identity_key_" + str(index), lambda key=key: models(key)) and ready
    invalid = "invalid-harness-key-" + secrets.token_hex(16)
    for label, key in (("missing", None), ("invalid", invalid)):
        for generation in (False, True):
            ready = case("deny_" + label + ("_generation" if generation else "_models"),
                         lambda key=key, generation=generation: denial(key, generation)) and ready
    if ready:
        for index, key in enumerate(keys[:2], 1):
            case("deterministic_generation_key_" + str(index), lambda key=key: short(key))
        case("structured_output", structured)
    else:
        report["generation_skipped"] = "Health, model identity or authentication admission failed."
    if args.long_context and ready and all(test["passed"] for test in report["tests"]):
        def long_context():
            ids, expected, construction = make_long_prompt(args.model_path, args.target_prompt_tokens, args.context_limit)
            # Persist the count/hash before the long request; never save the corpus or its IDs.
            report["long_prompt"] = construction
            save()
            status, body, details = request(base, "/generate", keys[0],
                {"input_ids": ids, "stream": False,
                 "sampling_params": {"temperature": 0, "max_new_tokens": MAX_OUTPUT_TOKENS}})
            require(status == 200 and isinstance(body, dict), "long retrieval returned HTTP " + str(status))
            text = body.get("text")
            metadata = body.get("meta_info") or {}
            observed = {**details, **construction, "output": str(text)[:256],
                "server_prompt_tokens": metadata.get("prompt_tokens"),
                "server_completion_tokens": metadata.get("completion_tokens"),
                "server_cached_tokens": metadata.get("cached_tokens"),
                "finish_reason": metadata.get("finish_reason")}
            report["long_response"] = observed
            save()
            require(metadata.get("prompt_tokens") == len(ids), "server prompt count differs from exact submitted IDs")
            require(isinstance(text, str) and text.strip() == expected, "long retrieval did not return the exact code")
            return observed
        case("near_context_exact_retrieval", long_context)
    elif args.long_context:
        report["long_context_skipped"] = "Short API checks failed."
    report["complete"] = True
    report["finished_at"] = now()
    report["passed"] = all(test["passed"] for test in report["tests"])
    report["short_outputs_equal"] = len(short_outputs) == 2 and len(set(short_outputs)) == 1
    save()
    print(json.dumps({"passed": report["passed"], "tests": len(report["tests"]),
                      "output": str(args.output)}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print("Validation setup failed: " + str(error), file=sys.stderr)
        sys.exit(2)
