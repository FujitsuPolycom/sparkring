#!/usr/bin/env python3
"""Check that every runtime input is pinned by an immutable identity.

`runtime/runtime-lock.json` names the inputs a runtime build consumes. An
identity that can move — a branch, a floating tag, an abbreviated commit —
makes the resulting image unreproducible without announcing it, so the
default mode reads the lock and the tree and reports any input whose
identity is not immutable, plus any in-tree artifact whose recorded
SHA-256 no longer matches its bytes.

Safety class: OFFLINE by default; it reads the checkout and nothing else.
`--check-remote` additionally contacts the public sources the lock names
to report whether each pinned identity still resolves. That mode reaches
the network but mutates nothing, and it contacts no configured Spark.

Retrievability is a separate property from immutability. A commit is
immutable the moment it is written and stays a valid identity forever;
whether anyone still serves those bytes is a fact about the world that
changes without warning. `--check-remote` is how that fact gets observed
before a build needs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "runtime" / "runtime-lock.json"

FULL_HEX40 = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Fields naming a source whose identity must be immutable, mapped to the
# shape that identity has to take.
COMMIT_FIELDS = (
    ("vllm", "commit"),
    ("sparkinfer", "commit"),
    ("flashinfer", "commit"),
    ("nccl", "commit"),
    ("deep_gemm", "commit_full"),
)
IMAGE_FIELDS = (
    ("base_image", "builder"),
    ("base_image", "runtime"),
)

# A reference that names something a maintainer can move.
MUTABLE = re.compile(
    r"^(latest|main|master|stable|dev|nightly|head|HEAD|v?[0-9]+(\.[0-9]+)*)$"
)


@dataclass(frozen=True)
class Finding:
    """One input that fails a property this checker asserts."""

    subject: str
    detail: str


def load_lock(path: Path = LOCK) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get(lock: Mapping[str, Any], *keys: str) -> Any:
    node: Any = lock
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def check_immutable_identities(lock: Mapping[str, Any]) -> list[Finding]:
    """Every external source is named by an identity that cannot move."""

    findings: list[Finding] = []

    for section, field in COMMIT_FIELDS:
        value = _get(lock, section, field)
        if value is None:
            findings.append(Finding(f"{section}.{field}", "absent from the lock"))
            continue
        if not FULL_HEX40.match(str(value)):
            findings.append(
                Finding(
                    f"{section}.{field}",
                    f"{value!r} is not a full 40-character commit; an "
                    "abbreviated or symbolic reference can resolve to "
                    "different bytes later",
                )
            )
        if MUTABLE.match(str(value)):
            findings.append(
                Finding(f"{section}.{field}", f"{value!r} is a movable reference")
            )

    for section, field in IMAGE_FIELDS:
        entry = _get(lock, section, field)
        if not isinstance(entry, Mapping):
            findings.append(Finding(f"{section}.{field}", "absent from the lock"))
            continue
        digest = str(entry.get("digest", ""))
        if not DIGEST.match(digest):
            findings.append(
                Finding(
                    f"{section}.{field}.digest",
                    f"{digest!r} is not a sha256 content digest; a tag "
                    "can be repointed at a different image",
                )
            )
        if "tag" in entry:
            findings.append(
                Finding(
                    f"{section}.{field}",
                    "carries a tag; a build that resolves a tag is not "
                    "pinned even when a digest sits beside it",
                )
            )

    revision = _get(lock, "model", "revision")
    if revision is None:
        findings.append(Finding("model.revision", "absent from the lock"))
    elif not FULL_HEX40.match(str(revision)):
        findings.append(
            Finding(
                "model.revision",
                f"{revision!r} is not an immutable commit hash; a branch "
                "name resolves to different weights over time",
            )
        )

    return findings


def _tracked_artifacts(lock: Mapping[str, Any]) -> Iterable[tuple[str, str, str]]:
    """Yield (subject, repo-relative path, expected sha256) from the lock."""

    for index, overlay in enumerate(lock.get("overlays") or []):
        if isinstance(overlay, Mapping) and "path" in overlay:
            yield (
                f"overlays[{index}]",
                str(overlay["path"]),
                str(overlay.get("sha256", "")),
            )
    for index, patch in enumerate(_get(lock, "nccl", "patches") or []):
        if isinstance(patch, Mapping) and "path" in patch:
            yield (
                f"nccl.patches[{index}]",
                str(patch["path"]),
                str(patch.get("sha256", "")),
            )
    for index, item in enumerate(lock.get("public_runtime_inputs") or []):
        if isinstance(item, Mapping) and "path" in item:
            yield (
                f"public_runtime_inputs[{index}]",
                str(item["path"]),
                str(item.get("sha256", "")),
            )


def check_tracked_artifacts(
    lock: Mapping[str, Any], root: Path = ROOT
) -> list[Finding]:
    """Every in-tree artifact the lock hashes still has those bytes."""

    findings: list[Finding] = []
    for subject, relative, expected in _tracked_artifacts(lock):
        if not SHA256.match(expected):
            findings.append(
                Finding(subject, f"{relative}: recorded hash {expected!r} is malformed")
            )
            continue
        path = root / relative
        if not path.is_file():
            findings.append(Finding(subject, f"{relative}: absent from the tree"))
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            findings.append(
                Finding(
                    subject,
                    f"{relative}: recorded {expected[:12]}, tree has {actual[:12]}",
                )
            )
    return findings


def _fetch_once(repository: str, commit: str, timeout: int) -> bool:
    """Attempt the fetch a build performs, restricted to history."""

    with tempfile.TemporaryDirectory() as scratch:
        init = subprocess.run(
            ["git", "init", "--quiet", scratch],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if init.returncode != 0:
            return False
        fetched = subprocess.run(
            ["git", "-C", scratch, "fetch", "--depth", "1", repository, commit],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return fetched.returncode == 0


def _git_commit_resolves(
    repository: str, commit: str, timeout: int = 300, attempts: int = 3
) -> bool:
    """Whether a server serves one commit to a client, retried.

    A repository answering at all proves nothing about a particular commit, so
    this performs the fetch a build performs. That request is also refused for
    reasons unrelated to the commit — rate limiting and transient transport
    failures both look like a refusal — and a durability check that cries wolf
    gets ignored. A single success is conclusive; only repeated refusals are
    reported.
    """

    for attempt in range(attempts):
        if _fetch_once(repository, commit, timeout):
            return True
        if attempt + 1 < attempts:
            time.sleep(5 * (attempt + 1))
    return False


def check_remote_sources(lock: Mapping[str, Any]) -> list[Finding]:
    """Report pinned git sources whose server no longer answers.

    Network-touching. A failure here does not mean the lock is wrong; it
    means the bytes it names are no longer being served, which is the
    condition a mirror exists to survive.
    """

    findings: list[Finding] = []
    for section, field in COMMIT_FIELDS:
        repository = _get(lock, section, "repository")
        commit = _get(lock, section, field)
        if not repository or not commit:
            continue
        try:
            if not _git_commit_resolves(str(repository), str(commit)):
                findings.append(
                    Finding(
                        f"{section}.repository",
                        f"{repository} did not answer for {str(commit)[:12]}",
                    )
                )
        except (OSError, subprocess.SubprocessError) as error:
            findings.append(
                Finding(f"{section}.repository", f"{repository}: {error!r}")
            )
    return findings


def run(check_remote: bool = False, root: Path = ROOT) -> list[Finding]:
    lock = load_lock(root / "runtime" / "runtime-lock.json")
    findings = check_immutable_identities(lock)
    findings += check_tracked_artifacts(lock, root)
    if check_remote:
        findings += check_remote_sources(lock)
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_runtime_inputs",
        description=(
            "Report runtime inputs that are not pinned by an immutable "
            "identity, and in-tree artifacts whose recorded hash no longer "
            "matches."
        ),
    )
    parser.add_argument(
        "--check-remote",
        action="store_true",
        help=(
            "also contact the public sources the lock names and report "
            "whether each pinned identity still resolves (network)"
        ),
    )
    arguments = parser.parse_args(argv)

    findings = run(check_remote=arguments.check_remote)
    if not findings:
        scope = "identities and tree artifacts"
        if arguments.check_remote:
            scope += " and remote resolution"
        print(f"runtime inputs: PASS ({scope})")
        return 0
    for finding in findings:
        print(f"FAIL {finding.subject}: {finding.detail}")
    print(f"{len(findings)} runtime input finding(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
