"""Tables of a campaign: decode per cell and arm, TTFT per prompt length, and output agreement.

``python -m performance.harnesses.serving_ab.report RESULTS_DIR LABEL [LABEL ...]`` prints them for the named
starts (directories of RESULTS_DIR); :func:`summary` returns the same as JSON. A label's arm is its prefix:
``S+`` for labels starting ``S+``, otherwise the first letter (``S``, ``N``, ``P``); a warm-up label ``W-<arm>``
counts as its arm. Ratios are against arm ``N`` when the starts include it.

Every start's row also shows its run receipt: the measurement method its decode result records
(receipt/method.py), its key resolved settings and digest (``resolved.json``, receipt/resolved.py) and, when
the start's directory holds the verdict of ``receipt check`` as ``check.json``, the declared change and
whether the arm differed from its control in exactly that change. A ratio between two arms is refused,
and the tables say why, when the arms' decode results record different measurement settings or when a
start of either arm failed its check.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from performance.harnesses.serving_ab.receipt import method as methods  # noqa: E402
from performance.harnesses.serving_ab.receipt.resolved import KEY_SETTINGS, canonical  # noqa: E402

ARM_ORDER = ["S", "S+", "N", "P"]


def arm_of(label: str) -> str:
    """The arm of a start label: ``S+2`` and the warm-up ``W-S+`` are ``S+``, ``N1`` is ``N``."""
    label = label.removeprefix("W-")
    return "S+" if label.startswith("S+") else label[0]


def decode_cell(r: dict) -> tuple[float, float | None, float | None]:
    """Aggregate tok/s, engine steps/s and accept length of one decode cell.

    The benchmark's steps/s is aggregate: the tokens per second of all streams over the accept length, from
    the server's speculative-decoding counters. A model served without speculation reports no drafts; one
    step then emits one token per request, so steps/s is the aggregate tok/s and the accept length is 1.
    """
    steps, accept = r.get("server_steps_per_s"), r.get("server_spec_accept_length")
    if not steps and not r.get("server_spec_drafts"):
        steps, accept = r["aggregate_tps"], 1.0
    return r["aggregate_tps"], steps, accept


def receipt_of(d: Path) -> dict:
    """The start's key resolved settings, digest, declared change and check verdict, where recorded."""
    out: dict = {"digest": None, "settings": {}, "declared": None, "control": None, "check": None, "problems": []}
    path = d / "resolved.json"
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        settings = document.get("settings") or {}
        out["digest"] = document.get("digest")
        out["settings"] = {k: settings[k] for k in KEY_SETTINGS if k in settings}
        if document.get("error"):
            out["error"] = document["error"]
    path = d / "check.json"
    if path.exists():
        verdict = json.loads(path.read_text(encoding="utf-8"))
        out.update(declared=verdict.get("declared"), control=verdict.get("control"),
                   check="passed" if verdict.get("passed") else "FAILED",
                   problems=[p.get("text", str(p)) for p in verdict.get("problems") or []])
    return out


def method_of(d: Path, decode: dict) -> dict:
    """The decode result's recorded method settings, and its method's name (method.json, else identified)."""
    name = None
    path = d / "method.json"
    if path.exists():
        name = json.loads(path.read_text(encoding="utf-8")).get("method")
    return {"name": name or methods.identify(decode) or "unrecorded", "signature": methods.signature(decode)}


def load(root: Path, label: str) -> dict:
    d = root / label
    out = {"label": label, "arm": arm_of(label)}
    ready = d / "ready.json"
    out["ready_s"] = json.loads(ready.read_text())["ready_seconds"] if ready.exists() else None
    decode = json.loads((d / "decode.json").read_text())
    out["method"] = method_of(d, decode)
    out["receipt"] = receipt_of(d)
    out["decode"] = {(r["context_tokens"], r["concurrency"]): decode_cell(r) for r in decode["results"]}
    out["ttft"] = {}
    for line in (d / "prefill.jsonl").read_text().splitlines():
        record = json.loads(line)
        if "ttft_seconds" in record:
            out["ttft"].setdefault(record["target_tokens"], []).append(
                (record["ttft_seconds"], record.get("cached_tokens_reported")))
    out["fingerprints"] = json.loads((d / "fingerprint.json").read_text())["fingerprints"]
    out["logprobs"] = json.loads((d / "logprobs.json").read_text())["positions"]
    return out


def _view(position):
    ranked = {token: value["rank"] for token, value in position.items()}
    top = min(ranked, key=ranked.get)
    prompt = next((token for token, rank in ranked.items() if rank != 1), top)
    return position[prompt]["logprob"], top


def logprob_compare(a, b) -> dict:
    n = min(len(a), len(b))
    diffs, agree = [], 0
    for x, y in zip(a[:n], b[:n]):
        lx, tx = _view(x)
        ly, ty = _view(y)
        diffs.append(abs(lx - ly))
        agree += tx == ty
    diffs.sort()
    return {"tokens": n, "mean": statistics.mean(diffs), "p99": diffs[int(0.99 * (n - 1))], "max": diffs[-1],
            "rank1_agreement": agree / n, "identical": diffs[-1] == 0.0}


def fingerprint_compare(a, b) -> dict:
    out = {}
    for name in a:
        ta, tb = a[name]["tokens"], b[name]["tokens"]
        first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if first is None and len(ta) != len(tb):
            first = min(len(ta), len(tb))
        out[name] = "identical" if first is None else f"diverges at token {first}"
    return out


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def comparability(runs: list[dict], arms: list[str], base: str | None) -> dict:
    """Per arm: whether its ratio to ``base`` counts, and every reason it does not."""
    out = {}
    for arm in arms:
        reasons = []
        group = [r for r in runs if r["arm"] in (arm, base)] if base else [r for r in runs if r["arm"] == arm]
        for r in group:
            if r["receipt"].get("check") == "FAILED":
                reasons.append(f"{r['label']} failed its arm check: " + "; ".join(r["receipt"]["problems"]))
        first = group[0]["method"]["signature"] if group else {}
        if len({canonical(r["method"]["signature"]) for r in group}) > 1:
            fields = sorted({f for r in group for f, v in r["method"]["signature"].items()
                             if canonical(v) != canonical(first.get(f))})
            reasons.append("measured differently (" + ", ".join(f"{r['label']} {r['method']['name']}" for r in group)
                           + "; differing: " + ", ".join(fields) + ")")
        out[arm] = {"base": base, "refused": bool(reasons), "reasons": reasons}
    return out


def summary(root: Path, labels: list[str]) -> dict:
    runs = [load(root, label) for label in labels]
    arms = sorted({r["arm"] for r in runs},
                  key=lambda a: (ARM_ORDER.index(a) if a in ARM_ORDER else len(ARM_ORDER), a))
    cells = []
    for ctx, conc in sorted({key for r in runs for key in r["decode"]}):
        row = {"context": ctx, "streams": conc, "starts": {}, "steps_per_s": {}, "tok_per_s": {}}
        for r in runs:
            value = r["decode"].get((ctx, conc))
            row["starts"][r["label"]] = value
        for arm in arms:
            mine = [r["decode"][(ctx, conc)] for r in runs if r["arm"] == arm and (ctx, conc) in r["decode"]]
            row["steps_per_s"][arm] = {"median": _median([v[1] for v in mine]),
                                       "range": [min(v[1] for v in mine), max(v[1] for v in mine)] if mine else None}
            row["tok_per_s"][arm] = {"median": _median([v[0] for v in mine])}
        cells.append(row)
    ttft = []
    for ctx in sorted({c for r in runs for c in r["ttft"]}):
        row = {"prompt_tokens": ctx, "starts": {}, "arms": {},
               "cached_tokens": sorted({s[1] for r in runs for s in r["ttft"].get(ctx, [])}, key=str)}
        for r in runs:
            samples = r["ttft"].get(ctx)
            row["starts"][r["label"]] = statistics.median(s[0] for s in samples) if samples else None
        for arm in arms:
            row["arms"][arm] = _median([row["starts"][r["label"]] for r in runs if r["arm"] == arm])
        ttft.append(row)
    pairs = []
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            pairs.append({"a": a["label"], "b": b["label"],
                          "fingerprints": fingerprint_compare(a["fingerprints"], b["fingerprints"]),
                          "logprobs": logprob_compare(a["logprobs"], b["logprobs"])})
    base = "N" if "N" in arms else None
    return {"starts": [{"label": r["label"], "arm": r["arm"], "ready_s": r["ready_s"], "method": r["method"],
                        "receipt": r["receipt"]} for r in runs],
            "arms": arms, "comparisons": comparability(runs, arms, base), "decode": cells, "ttft": ttft,
            "outputs": pairs}


def _short(value) -> str:
    if isinstance(value, dict) and "enum" in value:
        return value["enum"].split(".")[-1]
    if value is None:
        return "-"
    if value is True or value is False:
        return "on" if value else "off"
    return str(value)


RECEIPT_COLUMNS = (("target KV", "target.kv_cache_dtype"), ("block", "target.block_size"),
                   ("draft KV", "speculative.kv_cache_dtype"), ("async", "scheduler.async_scheduling"),
                   ("compile", "compilation.mode"), ("cudagraph", "compilation.cudagraph_mode"),
                   ("seqs", "scheduler.max_num_seqs"), ("batched", "scheduler.max_num_batched_tokens"))


def receipt_table(doc: dict) -> list[str]:
    """One row per start: method, digest, key resolved settings, declared change, check verdict."""
    lines = ["Run receipts: decode method, resolved settings (resolved.json) and declared change (check.json)",
             "| start | arm | method | digest | " + " | ".join(name for name, _ in RECEIPT_COLUMNS)
             + " | declared change | check |",
             "|---|---|---|---|" + "---|" * len(RECEIPT_COLUMNS) + "---|---|"]
    for start in doc["starts"]:
        receipt = start.get("receipt") or {}
        settings = receipt.get("settings") or {}
        declared = receipt.get("declared")
        change = ("; ".join(f"{k}={_short(v)}" for k, v in declared.items()) + f" vs {receipt.get('control')}"
                  if declared else "-")
        digest = (receipt.get("digest") or "-").removeprefix("sha256:")[:12]
        cells = " | ".join(_short(settings.get(key)) if key in settings else "?" for _, key in RECEIPT_COLUMNS)
        lines.append(f"| {start['label']} | {start['arm']} | {(start.get('method') or {}).get('name', '-')} | "
                     f"{digest} | {cells} | {change} | {receipt.get('check') or 'not checked'} |")
    return lines


def _method_text(doc: dict) -> str:
    names = sorted({(s.get("method") or {}).get("name", "unrecorded") for s in doc["starts"]})
    signature = ((doc["starts"][0].get("method") or {}).get("signature") or {}) if doc["starts"] else {}
    if len(names) == 1 and signature.get("temperature") is not None:
        return (f"method {names[0]}: temperature {signature['temperature']:g}, at most {signature['max_tokens']} "
                f"tokens, {signature['duration_per_test']:g} s cells")
    return "methods " + ", ".join(names)


def tables(doc: dict) -> str:
    arms, base = doc["arms"], "N" if "N" in doc["arms"] else None
    labels = [s["label"] for s in doc["starts"]]
    comparisons = doc.get("comparisons") or {}

    def refused(arm):
        return (comparisons.get(arm) or {}).get("refused")

    def ratio(numerator, denominator, arm):
        return " | REFUSED" if refused(arm) else f" | {numerator / denominator:.3f}"

    lines = ["Starts: " + ", ".join(f"{s['label']} (ready {s['ready_s']} s)" for s in doc["starts"]), ""]
    for arm in arms:
        for reason in (comparisons.get(arm) or {}).get("reasons") or []:
            lines.append(f"REFUSED: arm {arm}" + (f" against {base}" if base and arm != base else "") + f": {reason}")
    if any(refused(a) for a in arms):
        lines.append("")
    lines += receipt_table(doc) + [""]
    lines += [f"Decode engine steps/s, arm median (range); {_method_text(doc)}",
              "| context | streams | " + " | ".join(arms) + "".join(f" | {a}/{base}" for a in arms if base and a != base) + " |",
              "|---|---|" + "---|" * (len(arms) + (len(arms) - 1 if base else 0))]
    for row in doc["decode"]:
        st = row["steps_per_s"]
        cells = " | ".join(f"{st[a]['median']:.2f} ({st[a]['range'][0]:.2f}-{st[a]['range'][1]:.2f})" if st[a]["range"] else "-"
                           for a in arms)
        ratios = "".join(ratio(st[a]["median"], st[base]["median"], a) for a in arms
                         if base and a != base and st[a]["median"] and st[base]["median"])
        lines.append(f"| {row['context'] // 1024}k | {row['streams']} | {cells}{ratios} |")
    lines += ["", "Decode aggregate tok/s / steps/s / accept length per start",
              "| context | streams | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for row in doc["decode"]:
        lines.append(f"| {row['context'] // 1024}k | {row['streams']} | " + " | ".join(
            f"{v[0]:.1f} / {v[1]:.2f} / {v[2]:.2f}" if v else "-" for v in (row["starts"][label] for label in labels)) + " |")
    lines += ["", "TTFT seconds, median of the samples",
              "| prompt | " + " | ".join(labels) + " | " + " | ".join(arms) + "".join(
                  f" | {a}/{base}" for a in arms if base and a != base) + " | cached tokens |",
              "|---|" + "---|" * (len(labels) + len(arms) + (len(arms) - 1 if base else 0) + 1)]
    for row in doc["ttft"]:
        per = " | ".join(f"{row['starts'][label]:.3f}" if row["starts"][label] is not None else "-" for label in labels)
        am = " | ".join(f"{row['arms'][a]:.3f}" if row["arms"][a] is not None else "-" for a in arms)
        ratios = "".join(ratio(row["arms"][a], row["arms"][base], a) for a in arms
                         if base and a != base and row["arms"][a] and row["arms"][base])
        lines.append(f"| {row['prompt_tokens'] // 1024}k | {per} | {am}{ratios} | {row['cached_tokens']} |")
    lines += ["", "Outputs: identical fingerprints of 4; prompt-logprob mean |d| / max |d| / rank-1 agreement"]
    for p in doc["outputs"]:
        same = sum(v == "identical" for v in p["fingerprints"].values())
        lp = p["logprobs"]
        lines.append(f"  {p['a']} vs {p['b']}: {same}/4; {lp['mean']:.4f} / {lp['max']:.4f} / {lp['rank1_agreement']:.4f}"
                     + (" IDENTICAL" if lp["identical"] else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    print(tables(summary(Path(sys.argv[1]), sys.argv[2:])))
