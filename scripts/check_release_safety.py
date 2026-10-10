"""Scan tracked text for site/credential shapes without printing matched content.

Besides the shapes in RULES, the scan looks for the operator's own site values
when a local, untracked file lists them: one host name or MAC address per line
(``#`` starts a comment), in ``scripts/config/site-values.txt`` or the file
``--site-values`` names. Host names match as whole tokens and MAC addresses
also in their EUI-64 IPv6 link-local form. The values never enter the tree, so
CI, which has no such file, applies only RULES.

    python scripts/check_release_safety.py ROOT [--site-values FILE]
"""
from pathlib import Path
import argparse
import ipaddress
import json
import math
import re
import subprocess


RULES = {
    "private-lan-address": r"192\.168\.[0-9]{1,3}\.[0-9]{1,3}|(?:^|[^0-9.])172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}",
    "private-ssh-target": r"[A-Za-z0-9_.-]+@(?:192\.168\.|10\.[0-9]|172\.(?:1[6-9]|2[0-9]|3[01])\.)",
    "private-workspace": r"Documents[\\/]sparkring",
    # A local Windows user path: a drive letter or WSL mount followed by Users, or an AppData directory,
    # with any number of separators, so JSON-escaped backslashes match too.
    "local-user-path": r"[A-Za-z]:(?:\\|/)+Users(?:\\|/)|/mnt/[a-z]/Users/|AppData(?:\\|/)",
    "private-key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "ssh-public-key": r"ssh-(?:rsa|ed25519|dss) AAAA",
    "credential-assignment": r"(?:pass(?:word|wd)|secret|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|bearer)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9/+_.-]{12,}",
    "provider-token": r"AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN\s*[:=]|xox[baprs]-|gh[pousr]_[A-Za-z0-9]{20,}",
}
EXCLUDES = {"scripts/check_release_safety.py"}
# The untracked file of site values (.gitignore lists it).
SITE_VALUES = Path("scripts/config/site-values.txt")
# Lines that keep a site value on purpose: path -> {line: reason}. A file listed here is bound by SHA-256
# elsewhere, so editing it would change a frozen input.
SITE_VALUE_ALLOWED = {
    "performance/records/transport/eager-width-validation-20260817.md": {
        182: "frozen preserved input (runtime/releases/preserved-inputs.json)"},
}
_MAC = re.compile(r"[0-9a-f]{2}(?::[0-9a-f]{2}){5}")


def site_patterns(path):
    """One compiled pattern per value of the site-value file at ``path``; none when the file is absent.

    A MAC address also yields its EUI-64 interface identifier as an IPv6 link-local address prints it
    (``fe80::4ebb:47ff:fe2c:931e`` for ``4c:bb:47:2c:93:1e``)."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    patterns = []
    for line in lines:
        value = line.split("#", 1)[0].strip().lower()
        if not value:
            continue
        if _MAC.fullmatch(value):
            octets = [int(part, 16) for part in value.split(":")]
            if not any(octets):
                continue
            octets[0] ^= 0x02
            interface = bytes(octets[:3] + [0xFF, 0xFE] + octets[3:])
            eui = str(ipaddress.IPv6Address(bytes((0xFE, 0x80)) + bytes(6) + interface)).removeprefix("fe80::")
            patterns.append(re.compile(re.escape(value) + "|" + re.escape(value.replace(":", "-")) + "|"
                                       + re.escape(eui) + r"(?![0-9a-f])", re.I))
        else:
            patterns.append(re.compile(r"(?<![\w.-])" + re.escape(value) + r"(?![\w-])", re.I))
    return patterns


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


def findings(text, site=()):
    credential_text = credential_scan_text(text)
    for number, line in enumerate(text.splitlines(), 1):
        for name, pattern in RULES.items():
            scanned = credential_text if name == 'credential-assignment' and credential_text != text else line
            if re.search(pattern, scanned, re.I):
                yield number, name
        if any(pattern.search(line) for pattern in site):
            yield number, "site-value"


def main(root, site_values=None):
    site = site_patterns(site_values if site_values is not None else root / SITE_VALUES)
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
            allowed = SITE_VALUE_ALLOWED.get(relative, {})
            for line, rule in findings(data.decode("utf-8", errors="replace"), site):
                if rule == "site-value" and line in allowed:
                    continue
                # JSON escapes control characters in paths, preventing log injection.
                print(json.dumps({"path": relative, "line": line, "rule": rule}))
                count += 1
        print(f"Release-safety findings: {count}; matched content is withheld. Site values: {len(site)}.")
        return int(count > 0)
    except (OSError, subprocess.SubprocessError):
        # Exception strings can contain command output or sensitive paths.
        print("Release-safety scan failed; tracked content could not be fully inspected.")
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--site-values", type=Path, help="the untracked file of site host names and MAC addresses")
    options = parser.parse_args()
    raise SystemExit(main(options.root.resolve(), options.site_values))
