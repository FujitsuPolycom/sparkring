#!/usr/bin/env python3
"""Teacher-forced log-likelihood of fixed texts, for comparing two deployments of one model.

``measure`` sends each text file (its first ``--chars`` characters) as a
completion prompt with ``prompt_logprobs`` 0 and stores, per prompt token,
the token ID, its log-probability under the served model and its rank.
``compare`` reads two such results, requires identical token IDs, and reports
per file and overall: the mean negative log-likelihood (NLL) per token of
each, their difference, the mean absolute per-token log-probability
difference, and the share of tokens each ranks first.

    python3 logprob_probe.py measure --base http://127.0.0.1:8000 --out a.json FILE...
    python3 logprob_probe.py compare a.json b.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def measure(args):
    results = []
    for name in args.files:
        raw = Path(name).read_bytes()
        text = raw.decode("utf-8")[: args.chars]
        body = {"model": args.model, "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
        request = urllib.request.Request(args.base + "/v1/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        response = json.loads(urllib.request.urlopen(request, timeout=600).read())
        tokens = []
        for entry in response["choices"][0]["prompt_logprobs"][1:]:
            (token, row), = entry.items()
            tokens.append([int(token), row["logprob"], row["rank"]])
        results.append({"file": name, "file_sha256": hashlib.sha256(raw).hexdigest(), "chars": len(text),
                        "tokens": tokens})
    Path(args.out).write_text(json.dumps({"label": args.label, "texts": results}) + "\n", encoding="utf-8")


def compare(args):
    first, second = (json.loads(Path(name).read_text(encoding="utf-8")) for name in (args.first, args.second))
    rows, total = [], {"n": 0, "nll_a": 0.0, "nll_b": 0.0, "abs": 0.0, "top_a": 0, "top_b": 0}
    for a, b in zip(first["texts"], second["texts"], strict=True):
        if a["file_sha256"] != b["file_sha256"] or [t[0] for t in a["tokens"]] != [t[0] for t in b["tokens"]]:
            raise SystemExit(f"{a['file']}: the two results scored different tokens")
        n = len(a["tokens"])
        nll_a = -sum(t[1] for t in a["tokens"])
        nll_b = -sum(t[1] for t in b["tokens"])
        absdiff = sum(abs(x[1] - y[1]) for x, y in zip(a["tokens"], b["tokens"]))
        top_a = sum(t[2] == 1 for t in a["tokens"])
        top_b = sum(t[2] == 1 for t in b["tokens"])
        rows.append({"file": a["file"], "tokens": n, "nll_a": nll_a / n, "nll_b": nll_b / n,
                     "nll_b_minus_a": (nll_b - nll_a) / n, "mean_abs_logprob_diff": absdiff / n,
                     "top1_a": top_a / n, "top1_b": top_b / n})
        for key, value in (("n", n), ("nll_a", nll_a), ("nll_b", nll_b), ("abs", absdiff), ("top_a", top_a),
                           ("top_b", top_b)):
            total[key] += value
    n = total["n"]
    summary = {"a": first["label"], "b": second["label"], "tokens": n, "nll_a": total["nll_a"] / n,
               "nll_b": total["nll_b"] / n, "nll_b_minus_a": (total["nll_b"] - total["nll_a"]) / n,
               "mean_abs_logprob_diff": total["abs"] / n, "top1_a": total["top_a"] / n,
               "top1_b": total["top_b"] / n, "files": rows}
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    actions = parser.add_subparsers(dest="action", required=True)
    measured = actions.add_parser("measure")
    measured.add_argument("--base", default="http://127.0.0.1:8000")
    measured.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4-QAD-TP2")
    measured.add_argument("--chars", type=int, default=6000)
    measured.add_argument("--label", default="")
    measured.add_argument("--out", required=True)
    measured.add_argument("files", nargs="+")
    compared = actions.add_parser("compare")
    compared.add_argument("first")
    compared.add_argument("second")
    args = parser.parse_args()
    measure(args) if args.action == "measure" else compare(args)


if __name__ == "__main__":
    main()
