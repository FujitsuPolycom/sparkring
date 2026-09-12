"""Check tracked inline Markdown links and ATX heading anchors; not visual QA."""
from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from urllib.parse import unquote
from pathlib import Path

LINK = re.compile(r"(?<!\\)\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
SKIP = re.compile(r"^(https?:|mailto:|ftp:|tel:|data:|#!)", re.IGNORECASE)

def slug(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_~]", "", text)
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"[^\w\- ]+", "", text).strip().replace(" ", "-")

def anchors(path: Path) -> set[str]:
    result, fenced = set(), False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced and (match := HEADING.match(line)):
            base = slug(match.group(1))
            anchor, suffix = base, 0
            while anchor in result:
                suffix += 1
                anchor = f"{base}-{suffix}"
            result.add(anchor)
    return result

def formatting_warnings(text: str) -> list[int]:
    """Locate top-level lists touching paragraphs or tables, outside code fences."""
    result, fenced, previous = [], False, ""
    for number, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced and re.match(r"^(?:[-+*] |\d+\. )", line):
            if previous.strip() and not re.match(r"^(?:[-+*] |\d+\. |\s)", previous):
                result.append(number)
        previous = line
    return result

def main() -> int:
    root = Path(sys.argv[1]).resolve()
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "*.md"], cwd=root, check=True,
        capture_output=True,
    ).stdout.decode("utf-8", "surrogateescape").split("\0")
    cache, failures, checked = {}, [], 0
    for relative in filter(None, tracked):
        source = root / relative
        fenced = False
        for number, line in enumerate(source.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if FENCE.match(line):
                fenced = not fenced
                continue
            if fenced:
                continue
            for target in LINK.findall(re.sub(r"`[^`]*`", "", line)):
                if SKIP.match(target):
                    continue
                checked += 1
                path, _, fragment = target.partition("#")
                path = unquote(path)
                fragment = unicodedata.normalize("NFC", unquote(fragment)).lower()
                destination = source if not path else (source.parent / path).resolve()
                if not destination.is_relative_to(root):
                    failures.append(f"{relative}:{number}: target escapes repository {path!r}")
                elif not destination.exists():
                    failures.append(f"{relative}:{number}: missing target {path!r}")
                elif fragment and destination.suffix.lower() == ".md":
                    if destination not in cache:
                        cache[destination] = anchors(destination)
                    if fragment.lower() not in cache[destination]:
                        failures.append(f"{relative}:{number}: missing anchor #{fragment} in {destination.relative_to(root)}")
    for failure in failures:
        print(f"::error::{failure}")
    front_page = root / "README.md"
    if front_page.exists():
        for number in formatting_warnings(front_page.read_text(encoding="utf-8")):
            print(f"::warning file=README.md,line={number}::Separate the list from preceding content with a blank line.")
    print(f"checked {checked} repo-relative links")
    return int(bool(failures))

if __name__ == "__main__":
    raise SystemExit(main())
