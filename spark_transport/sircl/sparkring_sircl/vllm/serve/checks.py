"""Known-answer prompts, a long-prompt timing probe and the evaluation of SIRCL receipts.

The first two prompts and their pass rules are the repository's acceptance
checks ``count`` and ``arithmetic``
(``performance/harnesses/acceptance/checks.py``), so answers compare with the
profile's published functional checks on the four-Spark cycle. Every request
is greedy (temperature 0) and carries the profile's request settings
(``smoke`` in its ``config.json``; ``reasoning_effort: low`` for
GLM-5.3-Flash).

Receipt evaluation reads every rank's receipts (``sircl-vllm-receipt/v1``,
:mod:`sparkring_sircl.vllm.receipt`) and fails when:

- a rank has no tensor-parallel receipt, or a receipt's state is not
  ``ready``;
- a group whose NCCL policy is ``none`` built PyNccl or has any decision row
  with backend ``nccl``;
- any decision row has backend ``refuse``;
- the tensor-parallel group of a rank ran no all-reduce on SIRCL.

With a source overlay the receipts must also show ``vllm`` (and ``b12x``,
once loaded) imported from it (:func:`import_findings`). :func:`overlay_findings`
judges preflight's facts of every rank's overlay directory.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import plan as plan_mod
from .profile import ServingProfile

MAX_TOKENS = 256


@dataclasses.dataclass(frozen=True)
class Prompt:
    name: str
    content: str
    expectation: str
    accepts: Callable[[str], bool]


PROMPTS = (
    Prompt("count", "Count from 1 to 20, comma separated. Output only the numbers.",
           "contains 1, 2, ..., 20", lambda reply: ", ".join(str(i) for i in range(1, 21)) in reply),
    Prompt("arithmetic", "What is 17*23? Reply with the number only.",
           "exactly 391", lambda reply: reply.strip() == "391"),
    Prompt("capital", "What is the capital of France? Reply with one word.",
           "the word Paris", lambda reply: re.fullmatch(r"\W*paris\W*", reply.strip(), re.IGNORECASE) is not None),
)


def chat_body(profile: ServingProfile, content: str, *, max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    return {"model": profile.served_model_name, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0, **profile.request_settings}


@dataclasses.dataclass(frozen=True)
class Answer:
    name: str
    ok: bool
    content: str
    reasoning_chars: int
    seconds: float | None
    usage: Mapping[str, Any]
    error: str = ""

    def describe(self, expectation: str = "") -> str:
        if self.error:
            return f"{self.name}: FAILED ({self.error})"
        verdict = "passed" if self.ok else f"FAILED (expected {expectation})"
        timing = f" in {self.seconds:.2f} s" if self.seconds is not None else ""
        return (f"{self.name}: {verdict}{timing}; reply {self.content[:160]!r}; reasoning {self.reasoning_chars} "
                f"characters; usage {dict(self.usage)}")


def split_timing(output: str) -> tuple[str, float | None]:
    """Response body and curl's ``time_total`` (the last line of ``commands.api_post`` output)."""
    body, _, last = output.rstrip("\n").rpartition("\n")
    try:
        return body, float(last)
    except ValueError:
        return output, None


def read_answer(name: str, output: str, accepts: Callable[[str], bool]) -> Answer:
    body, seconds = split_timing(output)
    try:
        document = json.loads(body)
        message = document["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError):
        return Answer(name, False, "", 0, seconds, {}, error=f"not a chat completion: {body[:300]!r}")
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return Answer(name, bool(accepts(content)), content, len(reasoning), seconds, document.get("usage") or {})


# -- long prompt ----------------------------------------------------------------------------------

FILLER = "Record {index}: the quick brown fox jumps over the lazy dog near the river bank.\n"
# Tokens of one filler line with the GLM tokenizer, rounded down; the probe
# reports the server's own count (usage.prompt_tokens).
FILLER_TOKENS = 18


def long_prompt_body(profile: ServingProfile, tokens: int, nonce: str) -> dict[str, Any]:
    """A prompt of about ``tokens`` tokens that no earlier request shares a prefix with (``nonce`` first)."""
    lines = max(1, tokens // FILLER_TOKENS)
    text = (f"Session {nonce}. Read the records below.\n"
            + "".join(FILLER.format(index=index) for index in range(lines))
            + "Reply with the single word OK.")
    return chat_body(profile, text, max_tokens=1)


# -- receipts -------------------------------------------------------------------------------------


def parse_receipts(output: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Receipts from ``commands.receipts`` output: (records, problems)."""
    records, problems = [], []
    for line in output.splitlines():
        name, separator, text = line.partition("\t")
        if not separator:
            continue
        try:
            records.append(json.loads(text))
        except ValueError:
            problems.append(f"{name} is not valid JSON")
    return records, problems


def _rows(record: Mapping[str, Any], backend: str) -> list[Mapping[str, Any]]:
    return [row for row in record.get("decisions") or [] if row.get("backend") == backend]


def evaluate_receipts(receipts: Mapping[int, Sequence[Mapping[str, Any]]], world: int, *,
                      dcp: int = 1) -> tuple[list[str], list[str]]:
    """(problems, summary lines) for the receipts of global ranks ``0..world-1``; with decode-context
    parallelism ``dcp`` above 1 every rank also holds a decode-context-parallel receipt with a session."""
    problems, lines = [], []
    for rank in range(world):
        records = list(receipts.get(rank, ()))
        tensor_parallel = [record for record in records if str(record.get("group", "")).split(":")[0] == "tp"]
        if not tensor_parallel:
            problems.append(f"rank {rank}: no tensor-parallel receipt")
        if dcp > 1 and not any(str(record.get("group", "")).split(":")[0] == "dcp" and record.get("session")
                               for record in records):
            problems.append(f"rank {rank}: no decode-context-parallel receipt with a SIRCL session (--dcp-size {dcp})")
        for record in records:
            group = record.get("group")
            where = f"rank {rank} group {group}"
            if record.get("state") != "ready":
                problems.append(f"{where}: state {record.get('state')}")
            if record.get("nccl") == "none":
                if record.get("pynccl") != "skipped":
                    problems.append(f"{where}: PyNccl was built on a group NCCL may not run")
                nccl = _rows(record, "nccl")
                if nccl:
                    problems.append(f"{where}: {sum(int(row.get('calls', 0)) for row in nccl)} calls reached "
                                    f"NCCL ({[row.get('collective') for row in nccl]})")
            refused = _rows(record, "refuse")
            if refused:
                problems.append(f"{where}: refused collectives {[row.get('collective') for row in refused]}")
            decisions = ", ".join(f"{row.get('collective')}/{row.get('backend')}/{row.get('method')}"
                                  f"={row.get('calls')}" for row in record.get("decisions") or []) or "none"
            stats = record.get("session_stats") or {}
            lines.append(f"rank {rank} {group}: nccl {record.get('nccl')}, pynccl {record.get('pynccl')}, "
                         f"session {record.get('session')}, state {record.get('state')}, decisions {decisions}"
                         + (f", session stats {json.dumps(stats, sort_keys=True)[:400]}" if stats else ""))
        for record in tensor_parallel:
            if not any(row.get("collective") == "all_reduce" for row in _rows(record, "sircl")):
                problems.append(f"rank {rank} group {record.get('group')}: no all-reduce ran on SIRCL")
    return problems, lines


def import_findings(receipts: Mapping[int, Sequence[Mapping[str, Any]]],
                    overlay: str | None = None) -> tuple[list[str], list[str]]:
    """(lines, problems) about the directories every rank imported ``vllm`` and ``b12x`` from, as its
    tensor-parallel receipt states them. With a source overlay (its container path), a rank whose ``vllm``, or
    whose loaded ``b12x``, comes from elsewhere is a problem."""
    lines, problems = [], []
    seen: dict[tuple[Any, Any], list[int]] = {}
    for rank, records in sorted(receipts.items()):
        record = next((item for item in records if str(item.get("group", "")).split(":")[0] == "tp"), None)
        if record is None:
            continue
        found = (record.get("vllm"), record.get("b12x"))
        seen.setdefault(found, []).append(rank)
        if overlay is None:
            continue
        inside = overlay.rstrip("/") + "/"
        if not str(found[0] or "").startswith(inside):
            problems.append(f"rank {rank}: vllm was imported from {found[0] or 'an unstated directory'}, not the "
                            f"source overlay {overlay}")
        if found[1] is not None and not str(found[1]).startswith(inside):
            problems.append(f"rank {rank}: b12x was imported from {found[1]}, not the source overlay {overlay}")
    for (vllm, b12x), ranks in seen.items():
        lines.append(f"imports of rank(s) {','.join(str(rank) for rank in ranks)}: vllm from {vllm or 'unstated'}, "
                     f"b12x from {b12x or 'not loaded when the receipt was written'}")
    return lines, problems


def large_blocks_findings(receipts: Mapping[int, Sequence[Mapping[str, Any]]],
                          expected: int | None = None) -> tuple[list[str], list[str]]:
    """(lines, problems) about the two-shot and large-message grid cap each rank's tensor-parallel session
    used: ``large_blocks`` of its receipt's session statistics (the session's ``stats()``). With ``expected``
    (``--large-blocks``), a rank whose session states another value is a problem; a rank whose statistics
    state none is reported and not judged."""
    lines, problems = [], []
    used: dict[Any, list[int]] = {}
    for rank, records in sorted(receipts.items()):
        record = next((item for item in records if str(item.get("group", "")).split(":")[0] == "tp"), None)
        if record is None:
            continue
        stats = record.get("session_stats")
        value = stats.get("large_blocks") if isinstance(stats, Mapping) else None
        value = value if isinstance(value, int) and not isinstance(value, bool) else None
        used.setdefault(value, []).append(rank)
        if expected is not None and value is not None and value != expected:
            problems.append(f"rank {rank}: the session's two-shot and large-message grid cap is {value} blocks, "
                            f"--large-blocks asked for {expected}")
    source = f"--large-blocks {expected}" if expected is not None else "the session's default"
    for value, ranks in used.items():
        who = ",".join(str(rank) for rank in ranks)
        if value is None:
            lines.append(f"large_blocks of rank(s) {who}: the receipts' session statistics state none ({source} "
                         "not confirmed)")
        else:
            lines.append(f"large_blocks of rank(s) {who}: {value} (two-shot and large-message grid cap; "
                         f"{source})")
    return lines, problems


def overlay_findings(rows: Sequence[tuple[str, str, Mapping[str, str]]]) -> tuple[list[str], list[str]]:
    """(lines, blockers) for the source overlay of every rank: (where, host directory, facts of
    ``commands.overlay_facts``). Each directory must hold :data:`plan.OVERLAY_FILES`, and the tree its
    ``OVERLAY.json`` states must be the same on every Spark."""
    lines, blockers = [], []
    trees: dict[str, str] = {}
    for where, directory, values in rows:
        if values.get("overlay") != "present":
            blockers.append(f"{where}: overlay directory {directory} is missing or unreadable")
            continue
        missing = [name for name in plan_mod.OVERLAY_FILES if values.get(f"file:{name}") != "present"]
        if missing:
            blockers.append(f"{where}: overlay directory {directory} lacks {', '.join(missing)}")
            continue
        try:
            record = base64.b64decode(values.get("record", ""), validate=True)
        except (ValueError, binascii.Error):
            record = b""
        tree, basis = plan_mod.overlay_identity(record, values.get("record_sha256", ""))
        trees[where] = tree
        lines.append(f"{where}: overlay {directory} holds {', '.join(plan_mod.OVERLAY_FILES)}; tree {tree} ({basis})")
        metadata = values.get("metadata", "").split()
        if metadata:
            lines.append(f"{where}: overlay {directory} carries distribution metadata {', '.join(metadata)}; the "
                         "image's entrypoint reads installed distribution versions through the module path, so "
                         "a version it pins may then differ")
    if len(set(trees.values())) > 1:
        blockers.append("the overlay trees differ between Sparks: "
                        + "; ".join(f"{where}: {tree}" for where, tree in trees.items()))
    return lines, blockers


def nccl_free_findings(receipts: Mapping[int, Sequence[Mapping[str, Any]]],
                       world: int) -> tuple[list[str], list[str]]:
    """(lines, problems) of ``--require-no-nccl`` for the receipts of global ranks ``0..world-1``: every group
    of every rank must show NCCL policy ``none``, PyNccl ``skipped`` and no decision row with backend ``nccl``."""
    lines, problems = [], []
    groups: set[str] = set()
    for rank in range(world):
        records = list(receipts.get(rank, ()))
        if not records:
            problems.append(f"rank {rank}: no receipt, so its groups' use of NCCL is unknown")
        for record in records:
            group = str(record.get("group"))
            groups.add(group)
            where = f"rank {rank} group {group}"
            if record.get("nccl") != "none":
                problems.append(f"{where}: NCCL policy {record.get('nccl')} ({record.get('nccl_reason') or 'no reason'}"
                                "): NCCL may run on this group")
            if record.get("pynccl") != "skipped":
                problems.append(f"{where}: PyNccl {record.get('pynccl')}: vLLM built an NCCL communicator")
            rows = _rows(record, "nccl")
            if rows:
                problems.append(f"{where}: {sum(int(row.get('calls', 0)) for row in rows)} calls on NCCL ("
                                + ", ".join(f"{row.get('collective')}/{row.get('method')}={row.get('calls')}"
                                            for row in rows) + ")")
    if not problems:
        lines.append(f"NCCL-free receipts: every rank's groups ({', '.join(sorted(groups))}) show nccl=none, "
                     "pynccl=skipped and no NCCL decision row")
    return lines, problems


def tuning_findings(receipts: Mapping[int, Sequence[Mapping[str, Any]]],
                    expected: Mapping[str, str | None]) -> tuple[list[str], list[str]]:
    """(lines, problems) about the tuning table each session decides from: a receipt's ``tuning`` (the table's
    hash) and its session statistics' ``tuning`` (``stats()["tuning"]``: the key, the ops decided by
    ``collective/mode/choice``, the choices the session could not run and the named tables that did not
    match). ``expected`` maps a group kind to the hash of the table the plan matched (None: the rules); a
    session of such a kind with another table is a problem."""
    lines, problems = [], []
    for rank, records in sorted(receipts.items()):
        for record in records:
            group = str(record.get("group", ""))
            kind = group.split(":")[0]
            stats = record.get("session_stats")
            info = stats.get("tuning") if isinstance(stats, Mapping) else None
            info = info if isinstance(info, Mapping) else {}
            table = record.get("tuning") or info.get("table")
            if kind in expected and expected[kind] != table:
                problems.append(f"rank {rank} group {group}: tuning table {table or 'none (the rules choose)'}, "
                                f"the plan matched {expected[kind] or 'none'}")
            if table is None and not info:
                continue
            parts = [f"table {table}" if table else "rules"]
            key = info.get("key")
            if isinstance(key, Mapping):
                parts.append(f"key {key.get('shape')}, {key.get('world')} ranks, {key.get('lanes')} lanes, "
                             f"{key.get('max_relays')} relays")
            decisions = info.get("decisions") if isinstance(info.get("decisions"), Mapping) else {}
            parts.append("decisions " + (", ".join(f"{name}={count}" for name, count in decisions.items())
                                         or "none yet"))
            unusable = info.get("unusable") if isinstance(info.get("unusable"), Mapping) else {}
            if unusable:
                parts.append("choices the session could not run (the rules carried them) "
                             + ", ".join(f"{name}={count}" for name, count in unusable.items()))
            unmatched = info.get("unmatched") if isinstance(info.get("unmatched"), Mapping) else {}
            if unmatched:
                parts.append("tables that do not match it " + "; ".join(
                    f"{path} ({(list(why) or ['?'])[0]})" for path, why in unmatched.items()))
            lines.append(f"tuning, rank {rank} group {group}: " + "; ".join(parts))
    return lines, problems


def nccl_log_findings(found: Mapping[int, Sequence[str]], *, debug: bool) -> tuple[list[str], list[str]]:
    """(lines, problems) of the container log scan of ``--require-no-nccl``: ``found`` holds each rank's log
    lines that match :data:`plan.NCCL_LOG_PATTERNS`. A line with a :data:`plan.NCCL_INIT_PATTERNS` pattern is
    a communicator NCCL or PyNccl created, a problem; any other NCCL line is NCCL library activity without a
    communicator, reported on its own line."""
    lines, problems, activity = [], [], []
    for rank, matched in sorted(found.items()):
        hits = [line for line in matched if any(pattern in line for pattern in plan_mod.NCCL_INIT_PATTERNS)]
        other = [line for line in matched if line not in hits
                 and any(pattern in line for pattern in plan_mod.NCCL_LIBRARY_PATTERNS)]
        if hits:
            problems.append(f"rank {rank}: its log shows {len(hits)} NCCL communicator line(s), first: "
                            f"{hits[0].strip()[-240:]}")
        if other:
            activity.append(f"NCCL library activity without a communicator: rank {rank}, {len(other)} line(s), "
                            f"first: {other[0].strip()[-240:]}")
    if not problems:
        lines.append("NCCL log scan: no rank's log shows an NCCL communicator"
                     + ("" if debug else "; NCCL logs its own initialization only with NCCL_DEBUG=INFO and INIT "
                                         "(--nccl-debug), so only vLLM's PyNccl line was in view"))
    return lines + activity, problems
