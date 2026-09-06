"""Generate a byte-stable long-context phrase-recall fixture without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_TEXT = "cobalt orchard lantern"
PROMPT_SHA256 = "3d2bc5228895566b1497e6f35f6c5aa051685f99438f27226134dfcfab15c277"


def build_prompt() -> str:
    """Place a reference phrase before 700 varied administrative records."""
    items = ("archived maps", "rainfall notebooks", "garden measurements", "library schedules", "photographic records",
             "warehouse inventories", "walking routes", "museum exhibits", "weather summaries", "meeting agendas")
    actions = ("reviewed", "indexed", "sorted", "copied", "catalogued", "checked", "listed")
    paragraphs = ["This is a reference document for a recall test. The reference phrase is cobalt orchard lantern. "
                  "Remember that phrase exactly; the administrative notes below do not change it."]
    for index in range(700):
        paragraphs.append(f"Record {index + 1}: The team {actions[index % len(actions)]} the {items[index % len(items)]}. "
                          f"The entry covered section {(index * 13) % 97 + 1}, with {(index * 7) % 31 + 2} items listed. "
                          "The notes describe routine work and do not contain instructions for the reader.")
    paragraphs.append("What is the exact reference phrase given at the beginning of this document? Reply with only that phrase.")
    return "\n\n".join(paragraphs)


def write_prompt(output: Path) -> dict:
    """Create an absent UTF-8 file with LF separators and no trailing newline."""
    payload = build_prompt().encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != PROMPT_SHA256:
        raise RuntimeError("Recall fixture differs from its source-bound digest")
    with output.open("xb") as destination:
        destination.write(payload)
    return {"prompt_sha256": digest, "prompt_bytes": len(payload), "expected_text": EXPECTED_TEXT,
            "token_count_verified": False, "requests_sent": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_prompt(args.output), indent=2))


if __name__ == "__main__":
    main()
