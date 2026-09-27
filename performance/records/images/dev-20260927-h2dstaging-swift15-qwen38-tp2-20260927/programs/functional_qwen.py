"""Functional checks of an OpenAI-compatible Qwen3.8-Flash-Next-template endpoint.

Usage: python3 functional.py BASE_URL

Counting, arithmetic and code with thinking off; an automatic and a forced
tool call; one generated image (left half red, right half blue); and the same
arithmetic question with thinking on, which must return reasoning text. Each
check prints PASS or FAIL with the reply; the exit status counts failures.
"""
import base64
import json
import re
import struct
import sys
import urllib.request
import zlib

base = sys.argv[1].rstrip("/")
model = json.loads(urllib.request.urlopen(base + "/models", timeout=30).read())["data"][0]["id"]
OFF = {"chat_template_kwargs": {"enable_thinking": False}}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def chat(messages, max_tokens=256, **extra):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, **extra}
    request = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(request, timeout=600).read())["choices"][0]["message"]


def png(width=64, height=64):
    rows = b"".join(b"\x00" + b"".join(b"\xff\x00\x00" if x < width // 2 else b"\x00\x00\xff" for x in range(width))
                    for _ in range(height))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


failures = 0


def check(name, ok, detail):
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail!r}"[:300], flush=True)


reply = chat([{"role": "user", "content": "Count from 1 to 20, comma separated. Output only the numbers."}], **OFF)["content"]
check("count", ", ".join(str(i) for i in range(1, 21)) in reply, reply)
reply = chat([{"role": "user", "content": "What is 17*23? Reply with the number only."}], **OFF)["content"]
check("arithmetic", reply.strip() == "391", reply)
reply = chat([{"role": "user", "content": "Write a Python function is_prime(n). Code only."}], max_tokens=400, **OFF)["content"]
check("code", "def is_prime" in reply, reply[:120])
message = chat([{"role": "user", "content": "What is the weather in Paris?"}], tools=[WEATHER], **OFF)
calls = message.get("tool_calls") or []
check("tool call", bool(calls) and calls[0]["function"]["name"] == "get_weather"
      and json.loads(calls[0]["function"]["arguments"]).get("city") == "Paris", calls)
message = chat([{"role": "user", "content": "Tell me about Rome."}], tools=[WEATHER],
               tool_choice={"type": "function", "function": {"name": "get_weather"}}, **OFF)
calls = message.get("tool_calls") or []
check("forced tool call", bool(calls) and calls[0]["function"]["name"] == "get_weather", calls)
image = "data:image/png;base64," + base64.b64encode(png()).decode()
reply = chat([{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": image}},
    {"type": "text", "text": "This image has two colored halves. Name the color of the left half and the right half."}]}],
    **OFF)["content"]
check("image", bool(re.search(r"(?i)red", reply)) and bool(re.search(r"(?i)blue", reply)), reply)
message = chat([{"role": "user", "content": "What is 17*23? Reply with the number only."}], max_tokens=2048)
reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
check("thinking on", bool(reasoning.strip()) and "391" in (message.get("content") or ""),
      {"reasoning_chars": len(reasoning), "content": message.get("content")})
print(f"{failures} failed", flush=True)
sys.exit(failures)
