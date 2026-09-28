"""In-memory stand-ins for the harness tests: a chat server, Node A over SSH and the benchmark.

FakeModel answers the functional checks and the correctness-screen
questions correctly and records every request. FakeNodeA plays the
installer's run directory. FakeBench writes an llm-inference-bench matrix.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import threading

from performance.harnesses.acceptance.checks import QUESTIONS
from performance.harnesses.acceptance.runners import HttpError, Result

ANSWERS = {"a1": "10063", "a2": "1048576", "a3": "72", "a4": "8", "a5": "15", "a7": "6", "a8": "36", "f1": "Canberra",
           "f2": "Iron", "f3": "Jane Austen", "f4": "Jupiter", "f5": "1989", "f6": "北京", "f7": "H2O", "c1": "2",
           "c2": "3", "c3": "cba", "c4": "def is_palindrome(s):\n    return s == s[::-1]",
           "c5": "def fib(n):\n    a, b = 0, 1", "l1": "Yes", "l2": "Saturday", "l4": "3", "t1": "I am happy to see you.",
           "t2": "Air molecules scatter blue light more. So the sky looks blue."}
PROMPTS = {prompt: ANSWERS[qid] for qid, prompt, _ in QUESTIONS}


class FakeModel:
    def __init__(self, served="Model-TP2", *, overrides=None, fail_models=0):
        self.served, self.overrides = served, overrides or {}
        self.fail_models = fail_models
        self.requests, self.lock = [], threading.Lock()

    def get_json(self, url, *, timeout):
        with self.lock:
            self.requests.append(("GET", url, None))
            if self.fail_models:
                self.fail_models -= 1
                raise HttpError(f"{url}: connection refused")
        return {"data": [{"id": self.served}]}

    def post_json(self, url, body, *, timeout):
        with self.lock:
            self.requests.append(("POST", url, body))
        content = body["messages"][0]["content"]
        text = content if isinstance(content, str) else " ".join(p.get("text", "") for p in content)
        for key, reply in self.overrides.items():
            if key in text:
                if isinstance(reply, Exception):
                    raise reply
                return {"choices": [{"message": reply}]}
        return {"choices": [{"message": self.answer(text, body)}]}

    def answer(self, text, body):
        if text in PROMPTS:
            return {"content": PROMPTS[text]}
        code = re.search(r"vault access code is (\d+)", text)
        if code:
            return {"content": code.group(1)}
        if text.startswith("Count from 1 to 20"):
            return {"content": ", ".join(str(i) for i in range(1, 21))}
        if "17*23" in text:
            if "chat_template_kwargs" in body:
                return {"content": "391"}
            return {"content": "\n\n391", "reasoning_content": "17*23 = 17*20 + 17*3 = 391."}
        if "is_prime" in text:
            return {"content": "```python\ndef is_prime(n):\n    return n > 1\n```"}
        if "Paris" in text:
            return {"content": None, "tool_calls": [
                {"function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]}
        if "Rome" in text and body.get("tool_choice"):
            return {"content": None, "tool_calls": [
                {"function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}]}
        if "two colored halves" in text:
            return {"content": "The left half is red and the right half is blue."}
        return {"content": "I do not know."}


class FakeNodeA:
    """Node A's side of the detached installer: launch, poll and read."""

    def __init__(self, stdout, stderr, *, exit_code=0, polls_before_exit=1):
        self.stdout, self.stderr, self.exit_code = stdout, stderr, exit_code
        self.polls_before_exit = polls_before_exit
        self.launched, self.scripts = [], []

    def run(self, script, *, timeout):
        self.scripts.append(script)
        if "command.sh" in script:
            self.launched.append(script)
            return Result(0, "launched\n", "")
        if 'if [ ! -d "$dir" ]' in script:
            if not self.launched:
                return Result(0, "missing\n", "")
            if self.polls_before_exit > 0:
                self.polls_before_exit -= 1
                return Result(0, "running\nNode 0: Prepare pinned image\n", "")
            return Result(0, f"exit {self.exit_code}\nModel ready\n", "")
        if script.startswith("cat ") and "stdout.json" in script:
            return Result(0, self.stdout, "")
        if script.startswith("cat ") and "stderr.log" in script:
            return Result(0, self.stderr, "")
        raise AssertionError(f"unexpected script: {script}")


def bench_matrix(server, *, model="Model-TP2", note=None):
    """A reduced llm-inference-bench 0.6.2 matrix, including sections a record drops."""
    def cell(concurrency, rate, steps, accept):
        return {"concurrency": concurrency, "context_tokens": 0, "aggregate_tps": rate, "server_steps_per_s": steps,
                "server_spec_accept_length": accept, "num_errors": 0, "failure_reason": ""}
    metadata = {"version": "0.6.2", "model": model, "server": server, "temperature": 1.0}
    if note:
        metadata["note"] = note
    return {"metadata": metadata,
            "startup_diagnostics": {"hostname": "client-box", "server_url": server},
            "prefill": {"8192": {"tok_per_sec": 2675.0}, "65536": {"tok_per_sec": 2749.0},
                        "131072": {"tok_per_sec": 2040.0}},
            "results": [cell(1, 27.0853, 12.64, 2.1429), cell(8, 139.0267, 42.90, 3.2407),
                        cell(16, 205.5873, 64.20, 3.2023)],
            "summary_table": {"0": {"1": 27.0853, "8": 139.0267, "16": 205.5873}},
            "burst_results": [], "burst_summary_table": {}, "methodology": {}}


class FakeBench:
    """Stands in for runners.run_process running llm_decode_bench.py."""

    def __init__(self, server, *, note=None, exit_code=0):
        self.server, self.note, self.exit_code = server, note, exit_code
        self.calls = []

    def __call__(self, argv, *, log_path, cwd=None, timeout=None):
        self.calls.append(argv)
        Path(log_path).write_text("benchmark log\n", encoding="utf-8")
        if self.exit_code == 0:
            output = Path(argv[argv.index("--output") + 1])
            model = argv[argv.index("--model") + 1]
            output.write_text(json.dumps(bench_matrix(self.server, model=model, note=self.note)), encoding="utf-8")
        return self.exit_code
