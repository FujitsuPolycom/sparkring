"""Compare exact-length completion prefill using shared, disjoint token fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI base URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if (args.output.exists() or args.trials < 1 or any(size < 128 for size in args.sizes)
            or len(set(args.sizes)) != len(args.sizes)):
        parser.error("Use a fresh output, positive trial count and context sizes of at least 128")
    endpoint = args.base_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        parser.error("base-url must end in /v1")

    def post(path, payload):
        request = urllib.request.Request(endpoint[:-3] + path,
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    identity = {"schema": "sparkring-prefill-fixtures/v1", "model": args.model,
                "fixture_id": args.fixture_id, "sizes": args.sizes, "trials": args.trials}
    if not args.fixtures.exists():
        body = post("/tokenize", {"model": args.model, "prompt":
            "Record wind speed, cloud coverage and temperature each hour. " * 256})["tokens"]
        cases = []
        for size in args.sizes:
            for trial in range(args.trials):
                prefix = post("/tokenize", {"model": args.model, "prompt":
                    f"{args.fixture_id}/{size}/{trial}. Observations follow. "})["tokens"]
                if len(prefix) >= size or not body:
                    raise ValueError("Tokenized prefix or body cannot fill the requested case")
                tokens = prefix + (body * ((size // len(body)) + 1))[:size-len(prefix)]
                cases.append({"size": size, "trial": trial, "prompt": tokens})
        args.fixtures.parent.mkdir(parents=True, exist_ok=True)
        with args.fixtures.open("x", encoding="utf-8") as stream:
            json.dump({**identity, "cases": cases}, stream)
    raw = args.fixtures.read_bytes()
    fixture = json.loads(raw)
    if {key: fixture.get(key) for key in identity} != identity:
        raise ValueError("Existing fixtures have different model or workload settings")
    expected_cases = [(size, trial) for size in args.sizes for trial in range(args.trials)]
    if [(case.get("size"), case.get("trial")) for case in fixture["cases"]] != expected_cases:
        raise ValueError("Fixture cases differ from the complete requested size/trial matrix")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {**identity, "fixture_sha256": hashlib.sha256(raw).hexdigest(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "results": []}
    for case in fixture["cases"]:
        if len(case["prompt"]) != case["size"]:
            raise ValueError("Fixture token count differs from its declared size")
        start = time.monotonic()
        response = post("/v1/completions", {"model": args.model, "prompt": case["prompt"],
                                           "temperature": 0, "max_tokens": 1})
        elapsed = time.monotonic() - start
        usage = response["usage"]
        if usage["prompt_tokens"] != case["size"] or usage["completion_tokens"] != 1:
            raise ValueError("Server token accounting differs from the fixture")
        row = {"size": case["size"], "trial": case["trial"], "elapsed_seconds": elapsed,
               "prefill_tokens_per_second": case["size"] / elapsed, "usage": usage}
        record["results"].append(row)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
