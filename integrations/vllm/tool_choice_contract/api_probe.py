"""Bounded named/required tool API probes for an explicitly selected endpoint."""

import argparse
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from contract import StreamChoice


def decoded_response(body, stream):
    if not stream:
        data = json.loads(body)
        return data.get("error"), data.get("choices", []), True
    errors, choices, done = [], {}, False
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            done = True
            continue
        data = json.loads(line[6:])
        if "error" in data:
            errors.append(data["error"])
        for choice in data.get("choices", []):
            state = choices.setdefault(choice["index"], {"accumulator": StreamChoice(), "finish_reason": None})
            message = state["accumulator"].append(choice.get("delta", {}))
            if message:
                raise ValueError(message)
            if choice.get("finish_reason") is not None:
                state["finish_reason"] = choice["finish_reason"]
    return errors[0] if errors else None, [{"index": index, "finish_reason": state["finish_reason"],
        "message": {"tool_calls": list(state["accumulator"].calls.values())}}
        for index, state in choices.items()], done


def token_limit_rejected(status, error, done, stream):
    """Whether a truncated request received the policy's token-limit error.

    Nonstreamed, that is HTTP 400; streamed, an SSE error with code 400 after
    HTTP 200 headers, followed by ``[DONE]``.
    """
    return bool(error and error.get("type") == "BadRequestError" and error.get("code") == 400
                and status == (200 if stream else 400) and done)


def payload(model, mode, stream, max_tokens, thinking="off"):
    """One request; ``thinking="default"`` leaves the chat template's reasoning setting unchanged."""
    query = {"model": model, "messages": [{"role": "user", "content":
        "Call lookup with key exactly cedar. Do not answer in text."}],
        "tools": [{"type": "function", "function": {"name": "lookup",
            "description": "Look up a key.", "parameters": {"type": "object",
                "properties": {"key": {"type": "string"}}, "required": ["key"],
                "additionalProperties": False}}}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}} if mode == "named" else "required",
        "parallel_tool_calls": False, "stream": stream, "temperature": 0, "max_tokens": max_tokens}
    if thinking == "off":
        query["chat_template_kwargs"] = {"enable_thinking": False}
    return query


def run(base_url, model, output, policy, *, truncated_tokens=1, positive_tokens=128, thinking="off"):
    records = []
    output.mkdir(parents=True, exist_ok=False)
    for mode in ("named", "required"):
        for stream in (False, True):
            for negative in (False, True):
                case = f"{mode}-{'stream' if stream else 'full'}-{'truncated' if negative else 'positive'}"
                max_tokens = truncated_tokens if negative else positive_tokens
                headers = {"Content-Type": "application/json"}
                if os.environ.get("VLLM_API_KEY"):
                    headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
                req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                    data=json.dumps(payload(model, mode, stream, max_tokens, thinking)).encode(), headers=headers)
                start = time.monotonic()
                try:
                    with urllib.request.urlopen(req, timeout=90) as response:
                        status, body = response.status, response.read().decode()
                except urllib.error.HTTPError as error:
                    status, body = error.code, error.read().decode()
                (output / (case + ".txt")).write_text(body, encoding="utf-8")
                error, choices, done = decoded_response(body, stream and status == 200)
                if negative and policy == "enabled":
                    passed = token_limit_rejected(status, error, done, stream)
                elif negative:
                    passed = status == 200 and not error and any(
                        not c.get("message", {}).get("tool_calls") for c in choices)
                else:
                    passed = status == 200 and not error and done and len(choices) == 1
                    calls = choices[0].get("message", {}).get("tool_calls", []) if choices else []
                    passed = passed and len(calls) == 1 and calls[0]["function"]["name"] == "lookup"
                    if passed:
                        passed = json.loads(calls[0]["function"]["arguments"]) == {"key": "cedar"}
                records.append({"case": case, "max_tokens": max_tokens, "http_status": status,
                    "passed": bool(passed), "error_type": error.get("type") if error else None,
                    "error_message": error.get("message") if error else None,
                    "elapsed_seconds": round(time.monotonic() - start, 3)})
                print(json.dumps(records[-1]), flush=True)
    result = {"schema": "sparkring-tool-contract-api-probe/v1", "model": model,
              "policy": policy, "thinking": thinking, "passed": all(row["passed"] for row in records),
              "cases": records,
              "scope": "Eight requests. A truncated budget is a truncation probe, not a deterministic parser fault injector."}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Explicit /v1 endpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New private evidence directory")
    parser.add_argument("--policy", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--truncated-max-tokens", type=int, default=1,
                        help="max_tokens of the requests expected to end before a complete call")
    parser.add_argument("--positive-max-tokens", type=int, default=128,
                        help="max_tokens of the requests expected to return the call")
    parser.add_argument("--thinking", choices=("off", "default"), default="off",
                        help="off sends enable_thinking=false; default leaves the served chat template's setting")
    args = parser.parse_args()
    result = run(args.base_url, args.model, args.output, args.policy, truncated_tokens=args.truncated_max_tokens,
                 positive_tokens=args.positive_max_tokens, thinking=args.thinking)
    raise SystemExit(0 if result["passed"] else 1)
