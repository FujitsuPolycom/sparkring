"""Expose canonical deployment identities without ambiguous family shortcuts."""
import argparse
import json

from runtime.common import installer, profiles


def catalog():
    rows = []
    for ident in profiles.catalog():
        definition, _ = profiles.load(ident)
        resolved = profiles.resolve(ident)
        rows.append({"profile": ident, "title": definition["title"],
                     "nodes": resolved.get("serving", {}).get("node_count"),
                     "model": resolved.get("model", {}), "guide": definition["guide"],
                     "automated": ident in installer.SUPPORTED})
    return rows


def select(value, nodes):
    if value in ("qwen", "glm", "deepseek"):
        raise ValueError(f"'{value}' names a model family. Run 'sparkring models' and select the exact profile.")
    matches = [row for row in catalog() if row["profile"] == value]
    if not matches:
        raise ValueError("Unknown profile. Run 'sparkring models' for exact model/version/topology choices.")
    row = matches[0]
    if not row["automated"]:
        raise ValueError("This profile uses its own guide: " + row["guide"])
    if row["nodes"] != nodes:
        raise ValueError(f"{value} requires {row['nodes']} Sparks; this cluster has {nodes}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring models")
    parser.add_argument("action", nargs="?", choices=("list",), default="list")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rows = catalog()
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            mode = "installer" if row["automated"] else "guide"
            print(f"{row['profile']}  [{mode}]\n  {row['title']}")
        print("Install a profile with: sudo sparkring install --profile PROFILE")
    return 0
