"""Draft an evidence record and the README profile-table values from saved step results.

The record follows the images records in performance/records/images/:
status line, Conditions, Measurement, Result, Conclusion and Limitations,
with the raw files in a directory of the same name. The prose is a draft
that states only what the saved results show; a person edits it before
committing.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import json
import os
import re
import shlex

from . import throughput

DASH = "—"
BENCH_URL = "https://github.com/local-inference-lab/llm-inference-bench"
TOPIC = re.compile(r"[a-z0-9][a-z0-9.-]*")
DATE = re.compile(r"20\d{6}")
STATUSES = ("implemented", "qualified", "research-only")


def image_short(image):
    """`dev-20260927-b12xcache-cuda1342-nccl2323-status032` -> `dev-20260927-b12xcache`."""
    return re.sub(r"-cuda\d.*$", "", image)


def record_name(image, topic, date):
    if not TOPIC.fullmatch(topic):
        raise ValueError(f"Record topic must match {TOPIC.pattern}: {topic!r}")
    if not DATE.fullmatch(date):
        raise ValueError(f"Record date must be YYYYMMDD: {date!r}")
    return f"{image_short(image)}-{topic}-{date}"


def half_up(value, digits=0):
    return Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def decimal(value, digits):
    return DASH if value is None else f"{half_up(value, digits):.{digits}f}"


def whole(value):
    return DASH if value is None else f"{int(half_up(value)):,}"


def readme_values(summary):
    """Decode 1 / 8 / 16 and 64K prefill as the README profile table prints them.

    One stream keeps one decimal; 8 and 16 streams and prefill are whole
    numbers; missing values print as a dash.
    """
    decode = summary["decode"]
    rate = [decode.get(str(level), {}).get("aggregate_tps") for level in throughput.CONCURRENCY]
    return f"{decimal(rate[0], 1)} / {whole(rate[1])} / {whole(rate[2])}", whole(summary["prefill"].get("65536"))


def readme_line(profile, summary):
    decode, prefill = readme_values(summary)
    return f"README values for `{profile.id}` (port {profile.port}): decode {decode}; prefill 64K {prefill}"


def throughput_row(summary):
    decode, prefill = summary["decode"], summary["prefill"]
    levels = [decode.get(str(level), {}) for level in throughput.CONCURRENCY]
    return "| " + " | ".join([
        " / ".join(decimal(cell.get("aggregate_tps"), 1) for cell in levels),
        " / ".join(decimal(cell.get("server_steps_per_s"), 1) for cell in levels),
        " / ".join(decimal(cell.get("server_spec_accept_length"), 2) for cell in levels),
        " / ".join(whole(prefill.get(str(length))) for length in throughput.PREFILL),
    ]) + " |"


def sanitize_text(text, *, replace, names=(), users=()):
    """Replace known addresses and names, number any other private address, then refuse leftovers.

    `replace` maps exact strings (Node A's API host, the SSH alias) to
    placeholders. Other private IPv4 addresses become `ADDRESS_1`,
    `ADDRESS_2`, ... in order of appearance. `names` must then be absent as
    whole tokens and `users` absent as accounts (`user@`, `/home/user`);
    otherwise PrivateDataError is raised.
    """
    for value, placeholder in sorted(replace.items(), key=lambda item: -len(item[0])):
        if value:
            text = text.replace(value, placeholder)
    numbered = {}

    def number(match):
        return numbered.setdefault(match.group(0), f"ADDRESS_{len(numbered) + 1}")
    text = throughput.PRIVATE_ADDRESS.sub(number, text)
    findings = throughput.private_findings(text, names=names) + throughput.account_forms(text, users)
    if findings:
        raise throughput.PrivateDataError("refusing to write; the text still contains " + "; ".join(sorted(set(findings))))
    return text


def _cluster(profile):
    return "directly cabled Spark pair" if profile.nodes == 2 else f"{_count(profile.nodes)}-Spark ring"


def _count(number):
    return {2: "two", 3: "three", 4: "four"}.get(number, str(number))


def _functional_phrase(functional):
    if not functional:
        return None
    rows = functional["checks"]
    run = [r for r in rows if r["status"] != "SKIP"]
    skipped = [r["name"] for r in rows if r["status"] == "SKIP"]
    if functional["failed"]:
        text = f"{functional['passed']} of {len(run)} functional checks passed"
    else:
        text = f"all {len(run)} functional checks passed"
    return text + (f" ({', '.join(skipped)} skipped)" if skipped else "")


def _stress_phrase(stress):
    if not stress:
        return None
    if stress["degenerate"] or stress["errors"]:
        return (f"the {stress['n']}-request correctness screen returned {stress['degenerate']} degenerate and "
                f"{stress['errors']} failed responses")
    return f"a {stress['n']}-request correctness screen returned no degenerate or failed response"


def _link(directory, name, label):
    return f"[{label}]({directory}/{name})"


def _install_flags(arguments):
    """The installer's further arguments as a record prints them.

    A development image lock is a file on Node A, so its path is shown as
    `LOCK`; the conditions name the image that served.
    """
    shown, lock_path = [], False
    for argument in arguments:
        if lock_path:
            argument, lock_path = "LOCK", False
        elif argument == "--image-lock":
            lock_path = True
        elif argument.startswith("--image-lock="):
            argument = "--image-lock=LOCK"
        shown.append(shlex.quote(argument))
    return "".join(f" {argument}" for argument in shown)


def render(*, profile, name, record_dir, repo_root, files, install=None, source=None, install_arguments=(),
           functional=None, stress=None, summary=None, status=None, client, harness_revision):
    """Return the record's Markdown.

    `files` maps roles to file names in the record directory;
    `install_arguments` are the arguments the installer received after
    `--profile ID --yes --json`.
    """
    status = status or profile.status
    if status not in STATUSES:
        raise ValueError(f"Record status must be one of {', '.join(STATUSES)}")
    try:
        root = os.path.relpath(repo_root, record_dir).replace(os.sep, "/")
    except ValueError:
        raise ValueError("The record directory must be on the repository's drive") from None
    cluster = _cluster(profile)
    runs = summary["runs"] if summary else 0
    version = (", ".join(summary["versions"]) if summary else "") or "of an unrecorded version"
    parts = [p for p in (_functional_phrase(functional), _stress_phrase(stress)) if p]
    parts.append(f"measured on one {'pair' if profile.nodes == 2 else f'{_count(profile.nodes)}-Spark ring'}")
    if summary:
        parts.append("single-run timing" if runs == 1 else f"median of {runs} benchmark runs")
    if status != "qualified":
        parts.append("not serving-qualified")
    installed = bool(install and install.get("ok"))
    lines = [
        "<!-- Drafted by performance/harnesses/acceptance/accept_profile.py; review the prose before committing. -->",
        f"# {profile.title} with the installer image",
        "",
        f"Status: **{status}; " + "; ".join(parts) + "**.",
        "",
    ]
    if installed:
        revision = (install.get("source_revision") or "")[:12] or "unknown"
        flags = _install_flags(install_arguments)
        lines.append(f"`install.sh --profile {profile.id} --yes --json{flags}`, from {source.describe()}, installed "
                     f"source commit `{revision}` on one {cluster}, which then served `{profile.repository}` "
                     f"revision `{profile.revision[:12]}` as `{profile.served_model_name}`.")
    else:
        lines.append(f"This run installed nothing; it measured the `{profile.id}` deployment already serving "
                     f"`{profile.repository}` revision `{profile.revision[:12]}` as `{profile.served_model_name}` "
                     f"on one {cluster}.")
    lines += ["", "## Conditions", "",
              f"- **Profile and image:** `{profile.id}` on installer image `{profile.image}`, selected by "
              f"[`release.json`]({root}/{profile.release}); API port {profile.port}."]
    if installed:
        only = ("; standard output held one `sparkring-install-result/v1` document and nothing else"
                if install.get("only_document") else "")
        ready = install.get("api_ready_seconds")
        ready_text = (f" Node 0's API readiness step took {ready:g} s" if ready is not None
                      else " The installer output names no API readiness time")
        phases = f" ({_link(name, files['install_phases'], 'installer phases')})" if "install_phases" in files else ""
        lines.append(f"- **Installation:** result state `complete` with exit status 0{only}.{ready_text}{phases}.")
    else:
        lines.append("- **Installation:** none in this run.")
    lines += [f"- **Cluster:** one {cluster} (`{profile.topology}`).",
              f"- **Client:** {client} sent every request.",
              f"- **Harness:** [`accept_profile.py`]({root}/performance/harnesses/acceptance/accept_profile.py) "
              f"at commit `{harness_revision}`.",
              "", "## Measurement", ""]
    if functional:
        skipped = [f"{r['name']} ({r['detail']})" for r in functional["checks"] if r["status"] == "SKIP"]
        lines.append("- **Functional checks:** counting, arithmetic and code with the profile's thinking-off "
                     f"request settings (`{json.dumps(profile.thinking_off)}`), an automatic and a forced tool call, "
                     "a description of a generated two-color image, then the arithmetic question with "
                     + (f"the request settings `{json.dumps(profile.thinking_on)}`" if profile.thinking_on
                        else "the chat template's default thinking")
                     + ", which must return reasoning text. Each check passes or fails on the reply's content."
                     + (" Skipped: " + "; ".join(skipped) + "." if skipped else ""))
    if stress:
        lines.append(f"- **Correctness screen:** {stress['rounds']} rounds of 32 requests (24 short questions with "
                     "known answers and 8 questions about a code hidden in about 6K tokens) through 16 threads at "
                     "temperature 0 with the thinking-off settings. A failed request is an error; a response in "
                     "which one word repeats 8 or more times in a row is degenerate; any other response that misses "
                     "the expected answer is wrong.")
    if summary:
        repeat = "Each cell ran once." if runs == 1 else \
            f"The benchmark ran {runs} times; the tables give each value's median and the sum of request errors."
        lines.append(f"- **Throughput:** [llm-inference-bench]({BENCH_URL}) `llm_decode_bench.py` {version} at "
                     "temperature 1.0 with exact token targeting, 1, 8 and 16 concurrent streams, no added context, "
                     "20 s per cell after a 5 s warm-up, up to 2,048 output tokens with end of sequence ignored. "
                     "Decode is the aggregate output rate from stream usage. Prefill is a cold scout-only prompt's "
                     "length divided by its time to first token. Steps per second and tokens per step come from "
                     f"vLLM's speculative-decoding counters. {repeat}")
    if not (functional or stress or summary):
        lines.append("- No functional, correctness or throughput step ran.")
    lines += ["", "## Result", ""]
    correctness = []
    if functional:
        failed = [r["name"] for r in functional["checks"] if r["status"] == "FAIL"]
        run = functional["passed"] + functional["failed"]
        correctness.append(f"{functional['passed']} of {run} functional checks passed"
                           + (f"; failed: {', '.join(failed)}" if failed else "")
                           + f" ({_link(name, files['functional'], 'output')}).")
    if stress:
        wrong = f", to questions {', '.join('`' + q + '`' for q in stress['wrong_ids'])}" if stress["wrong_ids"] else ""
        correctness.append(f"The screen returned {stress['n']} responses in {stress['seconds']:g} s: "
                           f"{stress['degenerate']} degenerate, {stress['errors']} failed and {stress['wrong']} "
                           f"wrong{wrong} ({_link(name, files['stress'], 'summary')}).")
    if correctness:
        lines.append("**Correctness.** " + " ".join(correctness))
    if summary:
        matrices = files.get("matrices", [])
        if len(matrices) == 1:
            links = _link(name, matrices[0], "matrix")
        else:
            links = "matrices: " + ", ".join(_link(name, m, f"run {i + 1}") for i, m in enumerate(matrices))
        errors = sum(cell["num_errors"] for cell in summary["decode"].values())
        decode, prefill = readme_values(summary)
        lines += ["", f"**Throughput** ({links}):", "",
                  "| Decode 1 / 8 / 16 streams (tok/s) | Steps/s | Tokens/step | Prefill 8K / 64K / 128K (tok/s) |",
                  "|---|---|---|---|", throughput_row(summary), "",
                  f"Benchmark request errors: {errors}."]
        for item in summary["invalid"]:
            lines[-1] += f" Run {item['run']}, {item['concurrency']} streams: {item['reason']}."
        lines += ["", f"README profile table values: decode 1 / 8 / 16 users {decode} tok/s, prefill 64K {prefill} tok/s."]
    lines += ["", "## Conclusion", ""]
    claims = [p for p in (_functional_phrase(functional), _stress_phrase(stress)) if p]
    subject = f"`install.sh` installed `{profile.id}`, which" if installed else f"the `{profile.id}` deployment"
    lines.append(f"On one {cluster}, {subject} served `{profile.served_model_name}`"
                 + (": " + "; ".join(claims) if claims else "") + ".")
    lines += ["", "## Limitations", ""]
    if summary:
        lines.append(f"- The benchmark ran {'once' if runs == 1 else f'{runs} times'} on one {cluster}. At "
                     "temperature 1.0 the tokens accepted per speculative step follow the sampled text, so decode "
                     "rates vary between runs.")
        lines.append("- Prefill is one cold prompt per length and run.")
    else:
        lines.append("- Throughput was not measured.")
    if functional or stress:
        lines.append("- The functional checks and the screen test correctness, not output quality against a "
                     "reference checkpoint.")
    if functional and functional["skipped"]:
        lines.append("- Skipped functional checks: " + ", ".join(
            r["name"] for r in functional["checks"] if r["status"] == "SKIP") + ".")
    if not installed:
        lines.append("- The installation of the measured deployment is not part of this record.")
    return "\n".join(lines) + "\n"
