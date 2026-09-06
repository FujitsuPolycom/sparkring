#!/usr/bin/env python3
"""Run token-guarded exact/revision/join retrieval cases; never change a serving stack."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parent))
from needle_fixtures import FIXTURE_VERSION, REFERENCE_SOURCE_SHA256, build_case_document  # noqa: E402

SCHEMA = "sparkring-retrieval-validation/v1"


def token_count(raw):
    match = re.fullmatch(r"([1-9][0-9]*)([kKmM]?)", raw.strip())
    if not match:
        raise ValueError("Token counts must be positive integers, optionally suffixed k or m")
    value = int(match[1]) * {"": 1, "k": 1024, "m": 1024 * 1024}[match[2].lower()]
    if value > 16 * 1024 * 1024:
        raise ValueError("Token count exceeds the harness allocation bound of 16M")
    return value


def base_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/", "/v1", "/v1/")):
        raise ValueError("Base URL must be HTTP(S), without credentials, query, fragment or path except /v1")
    return f"{parsed.scheme}://{parsed.netloc}"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def post_json(url, payload, api_key, timeout):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise RuntimeError("HTTP response exceeds the 8 MiB receipt bound")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise RuntimeError("Expected a JSON object from the endpoint")
        return result
    except HTTPError as error:
        # Error bodies may echo authorization headers; do not put them in receipts.
        raise RuntimeError(f"HTTP {error.code} from the requested endpoint") from error
    except (URLError, TimeoutError, OSError, ValueError) as error:
        raise RuntimeError(f"Endpoint request failed ({type(error).__name__})") from error


def messages_for(document):
    return [{"role": "system", "content": "You perform exact long-context retrieval. Follow the final user question exactly."},
            {"role": "user", "content": document}]


def run_case(config, target, position, mode, index, repetition, *, post=post_json):
    seed = config["seed"] + index
    document, expected, _ = build_case_document(target, position, mode, seed)
    messages = messages_for(document)
    result = {"type": "case", "case_index": index, "repetition": repetition,
              "fixture_seed": seed, "context_target_tokens": target, "position_percent": position,
              "mode": mode, "expected": expected, "prompt_sha256": hashlib.sha256(document.encode()).hexdigest(),
              "passed": False, "response": "", "error": None}
    started = time.monotonic()
    try:
        tokenized = post(config["base_url"] + "/tokenize",
                         {"model": config["model"], "messages": messages,
                          "add_generation_prompt": True, "chat_template_kwargs": config["chat_template_kwargs"]},
                         config["api_key"], config["timeout"])
        count = tokenized.get("count")
        if type(count) is not int or count <= 0:
            raise RuntimeError("Tokenization must provide a positive integer count; no approximate fallback")
        result["tokenized_prompt_tokens"] = count
        if count + config["max_tokens"] > config["context_limit"]:
            raise RuntimeError("Tokenized prompt plus output budget exceeds the configured model context limit")
        payload = {"model": config["model"], "messages": messages, "max_tokens": config["max_tokens"],
                   "temperature": config["temperature"], "top_p": 1, "seed": seed,
                   "chat_template_kwargs": config["chat_template_kwargs"]}
        chat_started = time.monotonic()
        data = post(config["base_url"] + "/v1/chat/completions", payload, config["api_key"], config["timeout"])
        result["chat_elapsed_seconds"] = time.monotonic() - chat_started
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise RuntimeError("Completion must contain exactly one choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Completion message is missing")
        response = message.get("content") or ""
        if not isinstance(response, str):
            raise RuntimeError("Completion content must be text")
        response = response.strip()
        finish = choice.get("finish_reason")
        result.update(response=response, finish_reason=finish, usage=data.get("usage"),
                      answer_matches=response.count(expected) == 1, exact_output=response == expected,
                      output_budget_exhausted=finish == "length")
        if finish != "stop":
            raise RuntimeError("Output budget exhausted" if finish == "length" else "Completion did not finish normally")
        result["passed"] = result["answer_matches"]
    except (RuntimeError, ValueError, OSError, TypeError) as error:
        result["error"] = str(error)
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def emit(stream, record):
    line = json.dumps(record, ensure_ascii=True)
    stream.write(line + "\n")
    stream.flush()
    os.fsync(stream.fileno())
    print(line, flush=True)


def execute(config, output, *, post=post_json):
    total = len(config["contexts"]) * len(config["positions"]) * len(config["modes"]) * config["repetitions"]
    safe_config = {key: value for key, value in config.items() if key != "api_key"}
    with Path(output).open("x", encoding="utf-8") as stream:
        emit(stream, {"type": "start", "schema": SCHEMA, "fixture_version": FIXTURE_VERSION,
                      "reference_source_sha256": REFERENCE_SOURCE_SHA256,
                      "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "fixtures_sha256": hashlib.sha256(Path(__file__).with_name("needle_fixtures.py").read_bytes()).hexdigest(),
                      "config": safe_config, "planned_cases": total, "started_unix": time.time()})
        passed = errors = completed = 0
        interrupted = False
        try:
            index = 0
            for target in config["contexts"]:
                for position in config["positions"]:
                    for mode in config["modes"]:
                        index += 1
                        for repetition in range(1, config["repetitions"] + 1):
                            result = run_case(config, target, position, mode, index, repetition, post=post)
                            emit(stream, result)
                            completed += 1
                            passed += result["passed"]
                            errors += result["error"] is not None
        except KeyboardInterrupt:
            interrupted = True
        success = completed == total and passed == total and not interrupted
        emit(stream, {"type": "summary", "passed": passed, "completed": completed, "planned": total,
                      "errors": errors, "failed_answers": completed - passed - errors,
                      "interrupted": interrupted, "success": success})
    return 130 if interrupted else 0 if success else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-limit", required=True, type=token_count)
    parser.add_argument("--contexts", default="64k,128k,256k")
    parser.add_argument("--positions", default="5,50,95")
    parser.add_argument("--modes", default="exact,revision,join")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--chat-template-kwargs", default="{}", help="JSON object passed identically to tokenization and chat")
    parser.add_argument("--repetitions", type=int, default=1, help="Repeat each identical prompt; does not reset or assert cache state")
    parser.add_argument("--output", type=Path, required=True, help="Absent JSONL receipt path; parent directory must exist")
    args = parser.parse_args()
    try:
        contexts = [token_count(item) for item in args.contexts.split(",")]
        positions = [int(item) for item in args.positions.split(",")]
        modes = args.modes.split(",")
        kwargs = json.loads(args.chat_template_kwargs)
        if not contexts or not positions or not modes or not set(modes) <= {"exact", "revision", "join"}:
            raise ValueError("Nonempty contexts, positions and exact/revision/join modes are required")
        if not all(1 <= position <= 99 for position in positions):
            raise ValueError("Positions must be between 1 and 99")
        if not isinstance(kwargs, dict):
            raise ValueError("Chat template kwargs must be a JSON object")
        if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
            raise ValueError("Temperature must be finite and between zero and two")
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600:
            raise ValueError("Timeout must be finite and between zero and 3600 seconds")
        if not 1 <= args.max_tokens < args.context_limit or not 1 <= args.repetitions <= 10:
            raise ValueError("Output budget must fit context; repetitions must be between one and ten")
        if len(contexts) * len(positions) * len(modes) * args.repetitions > 300:
            raise ValueError("At most 300 cases are allowed per receipt")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env):
            raise ValueError("Invalid API-key environment variable name")
        config = {"base_url": base_url(args.base_url), "model": args.model, "contexts": contexts,
                  "positions": positions, "modes": modes, "temperature": args.temperature, "seed": args.seed,
                  "max_tokens": args.max_tokens, "context_limit": args.context_limit, "timeout": args.timeout,
                  "chat_template_kwargs": kwargs, "repetitions": args.repetitions,
                  "api_key_env": args.api_key_env, "api_key": os.getenv(args.api_key_env, "")}
        if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
            raise ValueError("Receipt path must be absent beneath an existing directory")
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    return execute(config, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
