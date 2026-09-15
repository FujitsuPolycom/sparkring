"""Measure exact-length cold Qwen prefills with reusable, disjoint token fixtures."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="API origin, without /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1 or any(n < 128 or n >= 262144 for n in args.lengths):
        parser.error("Use positive trial count and input lengths in [128,262143]")
    if not args.prepare_only and (args.output is None or args.output.exists()):
        parser.error("A non-existing --output file is required for measurement")

    def post(path, payload):
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/" + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)

    if not args.fixtures.exists():
        text = "Prefill transport attribution record. " + (
            "The weather station samples temperature, pressure, and wind each hour. "
            * 2400
        )
        seed = post("tokenize", {"model": args.model, "prompt": text})["tokens"][:16384]
        if len(seed) != 16384:
            raise ValueError("Tokenizer returned fewer than 16384 fixture seed tokens")
        source = seed * ((max(args.lengths) + len(seed) - 1) // len(seed))
        cases = []
        for size in args.lengths:
            for trial in range(args.trials):
                prefix = post(
                    "tokenize",
                    {
                        "model": args.model,
                        "prompt": f"HC kernel qualification length {size}, trial {trial}. Weather data follows.",
                    },
                )["tokens"]
                cases.append(
                    {
                        "size": size,
                        "trial": trial,
                        "payload": {
                            "model": args.model,
                            "prompt": prefix + source[len(prefix) : size],
                            "max_tokens": 1,
                            "temperature": 0,
                        },
                    }
                )
        args.fixtures.parent.mkdir(parents=True, exist_ok=True)
        args.fixtures.write_text(json.dumps(cases))
    fixture_bytes = args.fixtures.read_bytes()
    cases = json.loads(fixture_bytes)
    if [(c["size"], c["trial"]) for c in cases] != [
        (n, t) for n in args.lengths for t in range(args.trials)
    ]:
        raise ValueError("Fixture lengths/trials differ from the requested comparison")
    if any(
        c["payload"]["model"] != args.model or len(c["payload"]["prompt"]) != c["size"]
        for c in cases
    ):
        raise ValueError("Fixture model or token count differs")
    digest = hashlib.sha256(fixture_bytes).hexdigest()
    if args.prepare_only:
        print(json.dumps({"fixture_sha256": digest, "cases": len(cases)}))
        return
    results = []
    for case in cases:
        start = time.monotonic()
        response = post("v1/completions", case["payload"])
        elapsed = time.monotonic() - start
        usage = response["usage"]
        if usage["prompt_tokens"] != case["size"] or usage["completion_tokens"] != 1:
            raise ValueError("Server token counts differ from the fixture")
        result = {
            "input_tokens": case["size"],
            "trial": case["trial"],
            "elapsed_seconds": elapsed,
            "tokens_per_second": case["size"] / elapsed,
        }
        results.append({**result, "response": response})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"fixture_sha256": digest, "samples": results}, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
