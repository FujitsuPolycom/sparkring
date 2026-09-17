"""Scan tracked text for site/credential shapes without printing matched content."""
from pathlib import Path
import json
import math
import re
import subprocess
import sys


RULES = {
    "private-lan-address": r"192\.168\.[0-9]{1,3}\.[0-9]{1,3}|(?:^|[^0-9.])172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}",
    "private-ssh-target": r"[A-Za-z0-9_.-]+@(?:192\.168\.|10\.[0-9]|172\.(?:1[6-9]|2[0-9]|3[01])\.)",
    "private-workspace": r"Documents[\\/]sparkring",
    "private-key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "ssh-public-key": r"ssh-(?:rsa|ed25519|dss) AAAA",
    "credential-assignment": r"(?:pass(?:word|wd)|secret|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|bearer)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9/+_.-]{12,}",
    "provider-token": r"AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN\s*[:=]|xox[baprs]-|gh[pousr]_[A-Za-z0-9]{20,}",
}
EXCLUDES = {"scripts/check_release_safety.py"}


def credential_scan_text(text):
    """Normalize numeric Bearer token scores in single-line panel captures only.

    Original bytes remain the input for every other rule. Requiring one line
    preserves diagnostic line numbers; ambiguous or malformed JSON stays raw.
    """
    if len(text.splitlines()) != 1 or not text.lstrip().startswith('{'):
        return text

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON member')
            result[key] = value
        return result

    class JsonNumber(str):
        """Retain numeric spelling so unrelated assignments cannot shrink."""

    try:
        panel = json.loads(text, object_pairs_hook=unique, parse_float=JsonNumber)
        if not isinstance(panel, dict) or not {'model', 'label', 'k', 'panel', 'n_records', 'wall_s', 'records'} <= panel.keys():
            return text
        rows = panel['records']
        if not isinstance(rows, list) or panel['n_records'] != len(rows):
            return text
        for row in rows:
            if not isinstance(row, dict) or not {'w', 'p', 'top', 'argmax'} <= row.keys():
                return text
            top = row['top']
            if not isinstance(top, dict) or not all(
                type(value) in (int, JsonNumber) and math.isfinite(float(value)) and float(value) <= 0
                for value in top.values()
            ):
                return text
        for row in rows:
            for token, score in row['top'].items():
                if token.strip().lower() == 'bearer' and type(score) is JsonNumber:
                    row['top'][token] = 0
        # Unrelated fractional numbers serialize as strings with their exact
        # lexemes. This keeps credential-shaped values visible to the scanner.
        return json.dumps(panel, ensure_ascii=False)
    except (ValueError, TypeError, OverflowError):
        return text


def findings(text):
    credential_text = credential_scan_text(text)
    for number, line in enumerate(text.splitlines(), 1):
        for name, pattern in RULES.items():
            scanned = credential_text if name == 'credential-assignment' and credential_text != text else line
            if re.search(pattern, scanned, re.I):
                yield number, name


def main(root):
    try:
        paths = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, check=True,
            capture_output=True,
        ).stdout.decode("utf-8", "surrogateescape").split("\0")
        count = 0
        for relative in filter(None, paths):
            if relative in EXCLUDES:
                continue
            data = (root / relative).read_bytes()
            if b"\0" in data:
                continue
            for line, rule in findings(data.decode("utf-8", errors="replace")):
                # JSON escapes control characters in paths, preventing log injection.
                print(json.dumps({"path": relative, "line": line, "rule": rule}))
                count += 1
        print(f"Release-safety findings: {count}; matched content is withheld.")
        return int(count > 0)
    except (OSError, subprocess.SubprocessError):
        # Exception strings can contain command output or sensitive paths.
        print("Release-safety scan failed; tracked content could not be fully inspected.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
