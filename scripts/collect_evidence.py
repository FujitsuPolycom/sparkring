#!/usr/bin/env python3
"""Gather a SparkRing evidence bundle that is safe to attach in public.

Collects the artifacts a public-lane bug report or result needs -- site config,
preflight JSON, the acceptance gate's ``result.json``, the runtime manifest,
version pins, and log tails -- and **redacts** them before writing anything.

Redaction is mandatory, not optional. Two mechanisms run together:

1. **Literal redaction.** Every identifier-shaped value in your site config
   (SSH targets, users, hostnames, URLs, fabric and management addresses,
   local model/cache paths, site id) becomes a blocklisted literal and is
   replaced everywhere -- including inside log tails, which is where those
   strings actually leak from.
2. **Class redaction.** IPv4/IPv6 literals, ``user@host`` targets and email
   addresses, home/user paths, and non-allowlisted fully-qualified hostnames
   are replaced by pattern, so identifiers that never appeared in your config
   are caught too.

Then every emitted file is re-scanned. If a blocklisted literal or a
high-confidence identifier pattern survives, the bundle is **not written** and
the tool exits non-zero. A bundle that cannot be proven clean is not produced.

Image digests are kept by default: they are content addresses and are what
makes one report comparable to another. Pass ``--redact-digests`` if your
registry path is itself sensitive.

Redaction is mechanical. Read the bundle before you post it -- only you know
what else about your site is identifying.

Usage::

    python scripts/collect_evidence.py \\
        --site my-site.json \\
        --acceptance-result evidence/acceptance/<run-id>/result.json \\
        --preflight preflight.json \\
        --runtime-manifest runtime-manifest.json \\
        --log serve-rank0.log --log serve-rank1.log \\
        --out evidence-bundle
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

BUNDLE_SCHEMA = "sparkring-evidence-bundle/v1"
REDACTION_RULESET = "sparkring-redaction/v1"
DEFAULT_LOG_TAIL_LINES = 200

# Public infrastructure that is already named throughout this repository.
# Redacting these would destroy the report without protecting anything.
DOMAIN_ALLOWLIST = frozenset(
    {
        "github.com",
        "raw.githubusercontent.com",
        "huggingface.co",
        "hf.co",
        "nvidia.com",
        "nvcr.io",
        "developer.nvidia.com",
        "nvidia.github.io",
        "download.pytorch.org",
        "pytorch.org",
        "flashinfer.ai",
        "pypi.org",
        "files.pythonhosted.org",
        "docker.com",
        "download.docker.com",
        "ubuntu.com",
        "localhost",
        "example.com",
    }
)

# Last labels that mean "this is a filename", not "this is a hostname".
FILE_SUFFIXES = frozenset(
    {
        "bz2", "c", "cfg", "cmake", "conf", "cpp", "cu", "cuh", "csv", "d",
        "deb", "diff", "env", "gz", "h", "hpp", "html", "in", "ini", "j2",
        "json", "list", "lock", "log", "md", "ninja", "o", "out", "patch",
        "pem", "pid", "png", "ps1", "py", "pyc", "rst", "service", "sh",
        "so", "sql", "svg", "tar", "template", "tmp", "toml", "ts", "tsv",
        "txt", "whl", "xml", "yaml", "yml", "zip",
    }
)

# Keys whose values are site identifiers even when they do not look like one.
# Repo-relative tool paths (lock_path, verify_script, qualifier, ...) are
# deliberately NOT here: they are public and redacting them would make the
# report unreadable. Absolute paths are still caught by the path rule.
IDENTIFIER_KEYS = frozenset(
    {
        "site_id", "host", "hostname", "hosts", "node", "nodes", "user",
        "username", "ssh", "ssh_target", "ssh_user", "target", "left",
        "right", "peer", "peer0", "peer1", "container", "container_name",
        "api_base", "base_url", "url", "master_addr", "left_ip", "right_ip",
        "model_path", "model_dir", "cache_dir", "jit_cache",
    }
)

SECRET_KEY = re.compile(
    r"(?i)(token|secret|password|passwd|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|credential|authorization|auth[_-]?header)"
)

IPV4 = re.compile(
    r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b"
)
# Requires at least three colons so clock times ("10:11:12") are never touched.
IPV6 = re.compile(
    r"\b(?:[0-9A-Fa-f]{1,4}:){3,7}[0-9A-Fa-f]{1,4}\b|(?<![\w:])::[0-9A-Fa-f:]{2,}"
)
USER_AT_HOST = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9][A-Za-z0-9.\-]*\b")
USER_PATH = re.compile(
    r"(?:/home/|/root/|/Users/|(?<![\w])[A-Za-z]:[\\/]Users[\\/])"
    r"[A-Za-z0-9._\-/\\]*"
)
DOTTED_TOKEN = re.compile(
    r"\b[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)+\b"
)
SHA256_REF = re.compile(r"\bsha256:[0-9a-f]{64}\b")

PLACEHOLDER_IPV4 = "<redacted-ipv4>"
PLACEHOLDER_IPV6 = "<redacted-ipv6>"
PLACEHOLDER_USER_AT_HOST = "<redacted-user-at-host>"
PLACEHOLDER_PATH = "<redacted-path>"
PLACEHOLDER_HOSTNAME = "<redacted-hostname>"
PLACEHOLDER_SECRET = "<redacted-secret>"
PLACEHOLDER_DIGEST = "<redacted-digest>"
PLACEHOLDER_REDACTED_TEXT = "<redacted-free-text>"


class RedactionError(Exception):
    """A blocklisted identifier survived redaction. Nothing is written."""


class EvidenceError(Exception):
    """Input problem; the bundle is not produced."""


# ---------------------------------------------------------------------------
# Redaction engine
# ---------------------------------------------------------------------------


def looks_like_hostname(token: str) -> bool:
    """True when a dotted token is a hostname rather than a file or version."""
    labels = token.split(".")
    if len(labels) < 2:
        return False
    last = labels[-1].lower()
    if not last.isalpha() or len(last) < 2:
        return False
    if last in FILE_SUFFIXES:
        return False
    lowered = token.lower()
    if lowered in DOMAIN_ALLOWLIST:
        return False
    return not any(
        lowered == allowed or lowered.endswith("." + allowed)
        for allowed in DOMAIN_ALLOWLIST
    )


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _classify(literal: str) -> str:
    if "@" in literal:
        return "target"
    if _is_ip(literal):
        return "address"
    if "://" in literal:
        return "url"
    if literal.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", literal):
        return "path"
    if "." in literal:
        return "host"
    return "identifier"


def _collect_literals(node: Any, key: str | None, found: set[str]) -> None:
    if isinstance(node, dict):
        for child_key, value in node.items():
            _collect_literals(value, str(child_key), found)
        return
    if isinstance(node, (list, tuple)):
        for value in node:
            _collect_literals(value, key, found)
        return
    if not isinstance(node, str):
        return
    value = node.strip()
    if len(value) < 3:
        # Literals shorter than three characters would match inside unrelated
        # words and shred the report without protecting anything.
        return

    if key is not None and SECRET_KEY.search(key):
        # A secret is blocklisted as a literal too, not merely dropped from the
        # config: the same value is usually echoed in a log tail somewhere.
        found.add(value)
        return

    identifier = False
    if "@" in value:
        identifier = True
        user, _, host = value.partition("@")
        if len(user) >= 3:
            found.add(user)
        if len(host) >= 3:
            found.add(host)
    elif _is_ip(value):
        identifier = True
    elif "://" in value:
        identifier = True
        parts = urlsplit(value)
        if parts.hostname and len(parts.hostname) >= 2:
            found.add(parts.hostname)
        if parts.netloc:
            found.add(parts.netloc)
    elif value.startswith("/") and value.count("/") >= 2:
        identifier = True
    elif re.match(r"^[A-Za-z]:[\\/]", value):
        identifier = True
    elif "." in value and looks_like_hostname(value):
        identifier = True
    elif key is not None and key.lower() in IDENTIFIER_KEYS:
        identifier = True

    if identifier:
        found.add(value)


@dataclass
class Redactor:
    """Literal + class redaction with a mandatory residue self-check."""

    literals: dict[str, str] = field(default_factory=dict)
    redact_digests: bool = False
    generic_hostnames: bool = True

    @classmethod
    def from_site(
        cls,
        site: Any,
        *,
        redact_digests: bool = False,
        extra_literals: Iterable[str] = (),
        generic_hostnames: bool = True,
    ) -> "Redactor":
        found: set[str] = set()
        _collect_literals(site, None, found)
        for literal in extra_literals:
            if literal and len(literal.strip()) >= 3:
                found.add(literal.strip())
        # Drop anything that is only a placeholder or clearly not identifying.
        candidates = sorted(
            (literal for literal in found if not literal.startswith("<")),
            key=lambda value: (-len(value), value),
        )
        counters: dict[str, int] = {}
        mapping: dict[str, str] = {}
        for literal in candidates:
            kind = _classify(literal)
            counters[kind] = counters.get(kind, 0) + 1
            mapping[literal] = f"<redacted-{kind}-{counters[kind]}>"
        return cls(
            literals=mapping,
            redact_digests=redact_digests,
            generic_hostnames=generic_hostnames,
        )

    # -- application --------------------------------------------------------

    def text(self, value: str) -> str:
        redacted = value
        # Longest literals first so a host is not half-replaced inside a URL.
        for literal in sorted(self.literals, key=len, reverse=True):
            if literal in redacted:
                redacted = redacted.replace(literal, self.literals[literal])
            lowered = literal.lower()
            if lowered != literal and lowered in redacted:
                redacted = redacted.replace(lowered, self.literals[literal])
        redacted = USER_AT_HOST.sub(PLACEHOLDER_USER_AT_HOST, redacted)
        redacted = USER_PATH.sub(PLACEHOLDER_PATH, redacted)
        redacted = IPV4.sub(PLACEHOLDER_IPV4, redacted)
        redacted = IPV6.sub(PLACEHOLDER_IPV6, redacted)
        if self.redact_digests:
            redacted = SHA256_REF.sub(PLACEHOLDER_DIGEST, redacted)
        if self.generic_hostnames:
            redacted = DOTTED_TOKEN.sub(self._maybe_hostname, redacted)
        return redacted

    @staticmethod
    def _maybe_hostname(match: "re.Match[str]") -> str:
        token = match.group(0)
        if looks_like_hostname(token):
            return PLACEHOLDER_HOSTNAME
        return token

    def obj(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            # Keys are redacted too: a site can legitimately key a map by
            # hostname, and an un-redacted key would fail the self-check with
            # no way for the operator to proceed.
            return {
                self.text(str(child_key)): self.obj(child_value, str(child_key))
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [self.obj(item, key) for item in value]
        if isinstance(value, str):
            if key is not None and SECRET_KEY.search(key):
                return PLACEHOLDER_SECRET
            return self.text(value)
        return value

    # -- verification -------------------------------------------------------

    def residues(self, value: str) -> list[str]:
        """Blocklisted identifiers that survived. Empty means publishable."""
        found: list[str] = []
        lowered = value.lower()
        for literal in self.literals:
            if literal and literal.lower() in lowered:
                found.append(f"site literal {literal!r}")
        for match in IPV4.findall(value):
            found.append(f"ipv4 {match!r}")
        for match in USER_AT_HOST.findall(value):
            found.append(f"user@host {match!r}")
        for match in USER_PATH.findall(value):
            found.append(f"user path {match!r}")
        if self.generic_hostnames:
            for match in DOTTED_TOKEN.findall(value):
                if looks_like_hostname(match):
                    found.append(f"hostname {match!r}")
        return sorted(set(found))


# ---------------------------------------------------------------------------
# Bundle assembly
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON from {path}: {exc}") from exc


def load_site_document(path: Path) -> Any:
    """Load the site config through scripts/sparkring_site.py when available.

    That module owns the schema (and the YAML format). When it is unavailable
    the collector degrades to a plain JSON read of an already-normalised
    document -- redaction is identical either way, since it works on the
    values, not on the schema.
    """
    if not path.is_file():
        raise EvidenceError(f"site config not found: {path}")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import sparkring_site  # type: ignore

        return sparkring_site.load_site(str(path)).to_dict()
    except ImportError:
        print(
            "collect-evidence: NOTE: scripts/sparkring_site.py is unavailable; "
            "reading the site config as plain JSON.",
            file=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001 - fall back, but say why
        print(
            f"collect-evidence: NOTE: sparkring_site could not load {path} "
            f"({exc}); falling back to a plain JSON read. Redaction is "
            "unaffected.",
            file=sys.stderr,
        )
    return read_json(path)


def tail_lines(path: Path, count: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise EvidenceError(f"cannot read log {path}: {exc}") from exc
    lines = text.splitlines()
    return "\n".join(lines[-count:]) + ("\n" if lines else "")


def git_commit(repo_root: Path) -> str:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def versions_document(repo_root: Path, lock: Any) -> dict:
    lock = lock if isinstance(lock, dict) else {}
    toolchain = lock.get("toolchain") if isinstance(lock.get("toolchain"), dict) else {}
    model = lock.get("model") if isinstance(lock.get("model"), dict) else {}
    return {
        "collected_at": utc_now_iso(),
        "collector_host": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "repo": {"git_commit": git_commit(repo_root)},
        "runtime_lock": {
            "schema": lock.get("schema"),
            "runtime_id": lock.get("runtime_id"),
            "vllm": lock.get("vllm"),
            "sparkinfer": lock.get("sparkinfer"),
            "flashinfer": lock.get("flashinfer"),
            "deep_gemm": lock.get("deep_gemm"),
            "nccl": lock.get("nccl"),
            "toolchain": toolchain,
            "base_image": lock.get("base_image"),
            "model": {
                "repository": model.get("repository"),
                "revision": model.get("revision"),
                "config_sha256": model.get("config_sha256"),
            },
        },
    }


README_TEXT = """SparkRing evidence bundle
=========================

This bundle was produced by scripts/collect_evidence.py for a public-lane
result or bug report. Every file in it has been passed through the redaction
ruleset {ruleset} and re-scanned: no site literal from the site config, no
IPv4/IPv6 literal, no user@host target and no user/home path survived that
scan.

Redaction is mechanical. Read these files before posting them -- only you know
what else about your site is identifying.

Contents
--------
{contents}

Reading the result
------------------
`acceptance-result.redacted.json` carries two INDEPENDENT verdicts:

  functional_verdict   PASS | FAIL | BASELINE-RECORDED | NOT-RUN
  performance_verdict  IN-BAND | OUT-OF-BAND | BASELINE-RECORDED | NOT-MEASURED

Report them separately. A performance result is never a functional result, and
reference-lane historical throughput numbers are never public-lane results --
see docs/PUBLIC_FUNCTIONAL_TARGET.md section 4.
"""


@dataclass
class BundleFile:
    name: str
    text: str
    source: str | None
    note: str | None = None


def build_files(
    *,
    site: Any,
    redactor: Redactor,
    site_path: Path,
    acceptance_result: Path | None,
    preflight: Path | None,
    runtime_manifest: Path | None,
    runtime_lock: Path | None,
    logs: Sequence[Path],
    log_tail: int,
    repo_root: Path,
) -> list[BundleFile]:
    files: list[BundleFile] = []

    redacted_site = redactor.obj(site)
    if isinstance(redacted_site, dict):
        # Keep correlation without disclosure: a stable pseudonym for the site
        # name, and no free-text description at all.
        name = None
        if isinstance(site.get("site"), dict):
            name = site["site"].get("name")
        elif site.get("site_id"):
            name = site.get("site_id")
        if name:
            digest = hashlib.sha256(str(name).encode("utf-8")).hexdigest()
            pseudonym = f"site-{digest[:12]}"
            if isinstance(redacted_site.get("site"), dict):
                redacted_site["site"]["name"] = pseudonym
                redacted_site["site"]["description"] = PLACEHOLDER_REDACTED_TEXT
            if "site_id" in redacted_site:
                redacted_site["site_id"] = pseudonym
        redacted_site.pop("source", None)
    files.append(
        BundleFile(
            "site-config.redacted.json",
            json.dumps(redacted_site, indent=2, sort_keys=True) + "\n",
            str(site_path.name),
        )
    )

    if acceptance_result is not None:
        files.append(
            BundleFile(
                "acceptance-result.redacted.json",
                json.dumps(
                    redactor.obj(read_json(acceptance_result)), indent=2, sort_keys=True
                )
                + "\n",
                acceptance_result.name,
            )
        )

    if preflight is not None:
        files.append(
            BundleFile(
                "preflight.redacted.json",
                json.dumps(redactor.obj(read_json(preflight)), indent=2, sort_keys=True)
                + "\n",
                preflight.name,
            )
        )

    if runtime_manifest is not None:
        files.append(
            BundleFile(
                "runtime-manifest.redacted.json",
                json.dumps(
                    redactor.obj(read_json(runtime_manifest)), indent=2, sort_keys=True
                )
                + "\n",
                runtime_manifest.name,
            )
        )

    lock = read_json(runtime_lock) if runtime_lock is not None else {}
    files.append(
        BundleFile(
            "versions.json",
            json.dumps(
                redactor.obj(versions_document(repo_root, lock)), indent=2, sort_keys=True
            )
            + "\n",
            str(runtime_lock.name) if runtime_lock else None,
        )
    )

    for log in logs:
        files.append(
            BundleFile(
                f"logs/{log.name}.tail.txt",
                redactor.text(tail_lines(log, log_tail)),
                log.name,
                note=f"last {log_tail} lines",
            )
        )

    return files


def verify_files(files: Sequence[BundleFile], redactor: Redactor) -> None:
    problems: list[str] = []
    for entry in files:
        residues = redactor.residues(entry.text)
        if residues:
            problems.append(f"{entry.name}: {', '.join(residues[:8])}")
    if problems:
        raise RedactionError(
            "redaction self-check failed; the bundle was NOT written because it "
            "still contains blocklisted identifiers:\n  - " + "\n  - ".join(problems)
        )


def write_bundle(
    out_dir: Path, files: Sequence[BundleFile], redactor: Redactor, force: bool
) -> dict:
    if out_dir.exists():
        if not force:
            raise EvidenceError(
                f"output directory {out_dir} already exists; pass --force to replace it"
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    manifest_entries = []
    for entry in files:
        path = out_dir / entry.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry.text, encoding="utf-8")
        manifest_entries.append(
            {
                "name": entry.name,
                "sha256": hashlib.sha256(entry.text.encode("utf-8")).hexdigest(),
                "bytes": len(entry.text.encode("utf-8")),
                "source": entry.source,
                "note": entry.note,
            }
        )

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "generated_at": utc_now_iso(),
        "redaction": {
            "ruleset": REDACTION_RULESET,
            "site_literals_redacted": len(redactor.literals),
            "digests_redacted": redactor.redact_digests,
            "generic_hostname_redaction": redactor.generic_hostnames,
            "classes": [
                "ipv4",
                "ipv6",
                "user@host and email",
                "home/user paths",
                "site-config literals (hosts, users, urls, addresses, paths)",
                "non-allowlisted hostnames",
                "secret-shaped keys",
            ],
            "self_check": "passed",
        },
        "files": manifest_entries,
        "warning": (
            "Redaction is mechanical. Review every file before posting it "
            "publicly."
        ),
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contents = "\n".join(f"  {entry['name']}" for entry in manifest_entries)
    (out_dir / "README.txt").write_text(
        README_TEXT.format(ruleset=REDACTION_RULESET, contents=contents),
        encoding="utf-8",
    )
    return manifest


def discover_preflight(explicit: str | None, acceptance_result: Path | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise EvidenceError(f"preflight JSON not found: {path}")
        return path
    candidates = [Path("preflight.json"), Path("preflight-result.json")]
    if acceptance_result is not None:
        candidates.insert(0, acceptance_result.parent / "preflight.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def collect(args: argparse.Namespace) -> tuple[int, dict]:
    site_path = Path(args.site)
    site = load_site_document(site_path)

    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parent.parent
    )

    acceptance_result = Path(args.acceptance_result) if args.acceptance_result else None
    if acceptance_result is not None and not acceptance_result.is_file():
        raise EvidenceError(f"acceptance result not found: {acceptance_result}")

    runtime_manifest = Path(args.runtime_manifest) if args.runtime_manifest else None
    if runtime_manifest is not None and not runtime_manifest.is_file():
        raise EvidenceError(f"runtime manifest not found: {runtime_manifest}")

    runtime_lock = Path(args.runtime_lock) if args.runtime_lock else (
        repo_root / "runtime" / "runtime-lock.json"
    )
    if not runtime_lock.is_file():
        runtime_lock = None

    logs = [Path(entry) for entry in (args.log or [])]
    for log in logs:
        if not log.is_file():
            raise EvidenceError(f"log not found: {log}")

    preflight = discover_preflight(args.preflight, acceptance_result)

    redactor = Redactor.from_site(
        site,
        redact_digests=args.redact_digests,
        extra_literals=args.redact_token or (),
        generic_hostnames=not args.no_generic_hostnames,
    )
    files = build_files(
        site=site,
        redactor=redactor,
        site_path=site_path,
        acceptance_result=acceptance_result,
        preflight=preflight,
        runtime_manifest=runtime_manifest,
        runtime_lock=runtime_lock,
        logs=logs,
        log_tail=args.log_tail_lines,
        repo_root=repo_root,
    )
    verify_files(files, redactor)
    manifest = write_bundle(Path(args.out), files, redactor, args.force)
    return 0, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_evidence",
        description=(
            "Assemble a redacted SparkRing evidence bundle that is safe to "
            "attach to a public issue."
        ),
    )
    parser.add_argument("--site", required=True, help="site config JSON")
    parser.add_argument("--out", required=True, help="output bundle directory")
    parser.add_argument("--acceptance-result", default=None, help="result.json")
    parser.add_argument("--preflight", default=None, help="preflight JSON")
    parser.add_argument("--runtime-manifest", default=None, help="runtime-manifest.json")
    parser.add_argument(
        "--runtime-lock",
        default=None,
        help="runtime-lock.json (default: <repo>/runtime/runtime-lock.json)",
    )
    parser.add_argument(
        "--log", action="append", default=[], help="log file to tail (repeatable)"
    )
    parser.add_argument(
        "--log-tail-lines", type=int, default=DEFAULT_LOG_TAIL_LINES,
        help=f"lines of each log to keep (default {DEFAULT_LOG_TAIL_LINES})",
    )
    parser.add_argument(
        "--redact-token",
        action="append",
        default=[],
        help="additional literal to redact everywhere (repeatable)",
    )
    parser.add_argument(
        "--redact-digests",
        action="store_true",
        help="also redact sha256: image digests (kept by default)",
    )
    parser.add_argument(
        "--no-generic-hostnames",
        action="store_true",
        help=(
            "disable pattern-based hostname redaction (site literals, addresses "
            "and user@host targets are still redacted)"
        ),
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output directory"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code, manifest = collect(args)
    except RedactionError as exc:
        print(f"collect-evidence: REDACTION FAILURE: {exc}", file=sys.stderr)
        return 4
    except EvidenceError as exc:
        print(f"collect-evidence: ERROR: {exc}", file=sys.stderr)
        return 3
    print(
        f"collect-evidence: wrote {len(manifest['files'])} redacted file(s) to "
        f"{args.out} ({manifest['redaction']['site_literals_redacted']} site "
        "literals blocklisted). Review before posting."
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
