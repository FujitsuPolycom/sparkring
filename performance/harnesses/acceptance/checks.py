"""API readiness, functional checks and the concurrent correctness screen.

All requests go through an injected client with `get_json(url, timeout=)` and
`post_json(url, body, timeout=)`; see runners.HttpClient.
"""
from __future__ import annotations

import base64
import concurrent.futures
import json
import random
import re
import struct
import time
import zlib

from .runners import HttpError


def wait_ready(http, base_url, *, expected, timeout, interval, clock, sleep, log):
    """Poll `/v1/models` until it lists a model; the listed names must include `expected`."""
    start = clock()
    attempts, error = 0, None
    while True:
        attempts += 1
        try:
            served = [row["id"] for row in http.get_json(base_url + "/models", timeout=30).get("data", [])]
        except (HttpError, KeyError, TypeError, AttributeError) as exc:
            served, error = [], str(exc)
        if served:
            result = {"served_models": served, "expected": expected, "seconds": round(clock() - start, 1),
                      "attempts": attempts, "ok": expected in served}
            if not result["ok"]:
                result["error"] = f"the API serves {', '.join(served)}, not the profile's {expected}"
            return result
        if clock() - start >= timeout:
            return {"served_models": [], "expected": expected, "seconds": round(clock() - start, 1),
                    "attempts": attempts, "ok": False, "error": f"no model listed after {timeout} s: {error}"}
        if attempts == 1:
            log(f"readiness: waiting for {base_url}/models")
        sleep(interval)


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def two_color_png(width=64, height=64):
    """A PNG whose left half is red and right half is blue."""
    row = b"\x00" + b"".join(b"\xff\x00\x00" if x < width // 2 else b"\x00\x00\xff" for x in range(width))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b"")


class Chat:
    """Greedy chat requests against one served model with the profile's thinking switch."""

    def __init__(self, http, base_url, model, *, thinking_off, thinking_on, timeout=600):
        self.http, self.url, self.model = http, base_url + "/chat/completions", model
        self.thinking_off, self.thinking_on, self.timeout = thinking_off, thinking_on, timeout

    def __call__(self, content, *, max_tokens=256, thinking=False, **extra):
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens, "temperature": 0,
                **(self.thinking_on if thinking else self.thinking_off), **extra}
        return self.http.post_json(self.url, body, timeout=self.timeout)["choices"][0]["message"]


def _text(message):
    return message.get("content") or ""


def _calls(message):
    return message.get("tool_calls") or []


def check_count(chat):
    reply = _text(chat("Count from 1 to 20, comma separated. Output only the numbers."))
    return ", ".join(str(i) for i in range(1, 21)) in reply, reply


def check_arithmetic(chat):
    reply = _text(chat("What is 17*23? Reply with the number only."))
    return reply.strip() == "391", reply


def check_code(chat):
    reply = _text(chat("Write a Python function is_prime(n). Code only.", max_tokens=400))
    return "def is_prime" in reply, reply[:120]


def check_tool_call(chat):
    calls = _calls(chat("What is the weather in Paris?", tools=[WEATHER]))
    try:
        ok = bool(calls) and calls[0]["function"]["name"] == "get_weather" \
            and json.loads(calls[0]["function"]["arguments"]).get("city") == "Paris"
    except (KeyError, TypeError, ValueError, AttributeError):
        ok = False
    return ok, calls


def check_forced_tool_call(chat):
    calls = _calls(chat("Tell me about Rome.", tools=[WEATHER],
                        tool_choice={"type": "function", "function": {"name": "get_weather"}}))
    try:
        ok = bool(calls) and calls[0]["function"]["name"] == "get_weather"
    except (KeyError, TypeError):
        ok = False
    return ok, calls


def check_image(chat):
    image = "data:image/png;base64," + base64.b64encode(two_color_png()).decode()
    reply = _text(chat([{"type": "image_url", "image_url": {"url": image}},
                        {"type": "text", "text": "This image has two colored halves. "
                                                 "Name the color of the left half and the right half."}]))
    return bool(re.search(r"(?i)red", reply)) and bool(re.search(r"(?i)blue", reply)), reply


def check_thinking_on(chat):
    message = chat("What is 17*23? Reply with the number only.", max_tokens=2048, thinking=True)
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return bool(reasoning.strip()) and "391" in _text(message), \
        {"reasoning_chars": len(reasoning), "content": message.get("content")}


# (name, required profile feature, check). Every check but `thinking on`
# sends the profile's thinking-off settings.
FUNCTIONAL_CHECKS = (
    ("count", None, check_count),
    ("arithmetic", None, check_arithmetic),
    ("code", None, check_code),
    ("tool call", "tools", check_tool_call),
    ("forced tool call", "tools", check_forced_tool_call),
    ("image", "image", check_image),
    ("thinking on", "reasoning", check_thinking_on),
)
FEATURE_REASON = {"tools": "the profile does not enable automatic tool choice",
                  "image": "the profile accepts no image input",
                  "reasoning": "the profile sets no reasoning parser"}


def run_functional(chat, *, features, skip=(), log=lambda line: None):
    """Run each check once; a request error fails that check only."""
    rows = []
    for name, feature, check in FUNCTIONAL_CHECKS:
        if name in skip:
            rows.append({"name": name, "status": "SKIP", "detail": "skipped on request"})
        elif feature and feature not in features:
            rows.append({"name": name, "status": "SKIP", "detail": FEATURE_REASON[feature]})
        else:
            try:
                ok, detail = check(chat)
            except (HttpError, KeyError, IndexError, TypeError, AttributeError) as error:
                ok, detail = False, f"request failed: {error}"
            rows.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        log(functional_line(rows[-1]))
    return {"checks": rows, "passed": sum(r["status"] == "PASS" for r in rows),
            "failed": sum(r["status"] == "FAIL" for r in rows), "skipped": sum(r["status"] == "SKIP" for r in rows),
            "ok": not any(r["status"] == "FAIL" for r in rows)}


def functional_line(row):
    if row["status"] == "SKIP":
        return f"SKIP {row['name']}: {row['detail']}"
    return f"{row['status']} {row['name']}: {row['detail']!r}"[:300]


def functional_text(result):
    return "".join(functional_line(row) + "\n" for row in result["checks"]) + f"{result['failed']} failed\n"


# Correctness screen: 24 short questions with known answers plus 8 questions
# about a code hidden in about 6K tokens of filler per round. Question IDs,
# prompts, filler seeds and shuffling reproduce the screen published with the
# DeepSeek-V4.1-Flash and Swift records, so screens of different profiles
# send identical requests.
QUESTIONS = (
    ("a1", "What is 347 * 29? Reply with the number only.", r"\b10063\b"),
    ("a2", "What is 2^20? Reply with the number only.", r"\b1048576\b"),
    ("a3", "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Number only.", r"\b72\b"),
    ("a4", "If x + 2x + 3x = 48, what is x? Number only.", r"\b8\b"),
    ("a5", "How many prime numbers are there below 50? Number only.", r"\b15\b"),
    ("a7", "What is the remainder when 1000 is divided by 7? Number only.", r"\b6\b"),
    ("a8", "Compute 15% of 240. Number only.", r"\b36\b"),
    ("f1", "What is the capital of Australia? One word.", r"Canberra"),
    ("f2", "Which element has the chemical symbol 'Fe'? One word.", r"[Ii]ron"),
    ("f3", "Who wrote 'Pride and Prejudice'? Name only.", r"Austen"),
    ("f4", "What is the largest planet in our solar system? One word.", r"Jupiter"),
    ("f5", "In what year did the Berlin Wall fall? Number only.", r"1989"),
    ("f6", "中国的首都是哪里？只回答城市名。", r"北京"),
    ("f7", "水的化学式是什么？只回答化学式。", r"H\s*2\s*O|H₂O"),
    ("c1", "What does this Python print? print(sorted([3,1,2], reverse=True)[1])  Answer with the output only.",
     r"^\s*`*2`*\s*$"),
    ("c2", "What does this Python print? print(len({1,2,2,3,3,3}))  Answer with the output only.", r"^\s*`*3`*\s*$"),
    ("c3", "What does this Python print? print('abc'[::-1])  Answer with the output only.", r"cba"),
    ("c4", "Write a Python function is_palindrome(s) that returns True if s reads the same backwards. Code only.",
     r"def is_palindrome"),
    ("c5", "Write a Python function fib(n) returning the n-th Fibonacci number iteratively. Code only.", r"def fib"),
    ("l1", "All cats are mammals. Tom is a cat. Is Tom a mammal? Answer yes or no.", r"(?i)\byes\b"),
    ("l2", "If today is Wednesday, what day will it be in 10 days? One word.", r"Saturday"),
    ("l4", "How many letters 'r' are in the word 'strawberry'? Number only.", r"\b3\b"),
    ("t1", "Translate to English: '我今天很高兴见到你。' Answer with the translation only.", r"(?i)happy|glad|pleased"),
    ("t2", "Explain in two sentences why the sky is blue.", r"(?i)scatter"),
)
FILLER = ("the river keeps moving past old stone bridges while merchants count copper coins and children chase "
          "gulls along the harbor wall as clouds gather over distant hills and lanterns flicker in narrow streets").split()
# One word repeated at least 8 times in a row.
DEGENERATE = re.compile(r"(\b\S+\b)(?:\s+\1\b){7,}")
STRESS_THREADS = 16


def needle_question(seed):
    rng = random.Random(seed)
    code = str(10000 + seed * 7919 % 89999)
    parts = []
    for index in range(60):
        parts.append(" ".join(rng.choice(FILLER) for _ in range(70)) + ".")
        if index == 30:
            parts.append(f"Important: the vault access code is {code}.")
    return (f"L{seed}", "\n".join(parts) + "\n\nWhat is the vault access code mentioned above? Reply with the number only.",
            r"\b" + code + r"\b")


def round_items(number):
    items = list(QUESTIONS) + [needle_question(number * 8 + j + 1) for j in range(8)]
    random.Random(number).shuffle(items)
    return items


def score(qid, pattern, text):
    return {"id": qid, "ok": bool(re.search(pattern, text.strip(), re.M)), "degenerate": bool(DEGENERATE.search(text)),
            "error": None, "text": text[:240]}


def run_stress(chat, *, rounds, clock=time.monotonic):
    """Send `rounds` rounds of 32 requests through 16 threads with thinking off.

    Categories are exclusive: a failed request is an error; a response with
    a repeated word is degenerate; any other response that misses the
    expected answer is wrong.
    """
    def ask(item):
        qid, prompt, pattern = item
        try:
            text = _text(chat(prompt, max_tokens=300))
        except (HttpError, KeyError, IndexError, TypeError, AttributeError) as error:
            return {"id": qid, "ok": False, "degenerate": False, "error": str(error)[:200], "text": ""}
        return score(qid, pattern, text)

    start = clock()
    responses = []
    for number in range(rounds):
        with concurrent.futures.ThreadPoolExecutor(STRESS_THREADS) as pool:
            for row in pool.map(ask, round_items(number)):
                responses.append({**row, "round": number})
    seconds = round(clock() - start, 1)
    errors = [r for r in responses if r["error"]]
    degenerate = [r for r in responses if not r["error"] and r["degenerate"]]
    wrong = [r for r in responses if not r["error"] and not r["degenerate"] and not r["ok"]]
    summary = {"rounds": rounds, "n": len(responses), "degenerate": len(degenerate), "wrong": len(wrong),
               "errors": len(errors), "seconds": seconds,
               "degenerate_items": [[r["round"], r["id"], r["text"][:120]] for r in degenerate],
               "wrong_ids": sorted({r["id"] for r in wrong}),
               "wrong_samples": [[r["id"], r["text"][:80]] for r in wrong[:8]],
               "error_samples": [[r["id"], r["error"]] for r in errors[:8]],
               "ok": not degenerate and not errors}
    return summary, responses
