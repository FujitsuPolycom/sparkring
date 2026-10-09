"""Expose canonical deployment identities without ambiguous family shortcuts."""
import argparse
import json

from runtime.common import installer, profiles, qwen_flash_next, thinking

THINKING_FIELDS = ("default", "level", "levels", "range", "effort", "off")
# Other names that select an installer profile. A deployment, its records and its status keep the
# profile's ID, so every command replaces an alias with that ID before it reads anything else. The IDs
# of the GLM-5.3-Flash profiles of two and four Sparks name their NVFP4-Spark checkpoint, which they
# install only on an image whose vLLM cannot read their CSF checkpoint
# (profiles/glm53-checkpoints.md#default-checkpoint-by-profile); an alias names the model and the
# number of Sparks alone.
ALIASES = {"glm53-flash-tp2": "glm53-flash-nvfp4-spark-tp2", "glm53-flash-tp4": "glm53-flash-nvfp4-spark-tp4"}


def canonical(value):
    """The profile ID that ``value`` selects: the ID that an alias of ALIASES stands for, else ``value``."""
    return ALIASES.get(value, value)


def checkpoints(definition):
    """The checkpoint names of a profile definition's serving configuration, or None without a checkpoints table.

    The result is ``{"default", "preferred", "others"}``: ``preferred`` is the
    checkpoint that ``sparkring install`` selects instead of ``default`` on an
    image whose vLLM reads it (image_lock.preferred_checkpoint), or None, and
    ``others`` are the further names that ``--checkpoint`` takes.
    """
    source = definition["configuration"]
    if source["format"] != "serving-profile":
        return None
    configuration = profiles.read_json(profiles.local_path(source["path"]))
    default, names = qwen_flash_next.checkpoint_names(configuration)
    if default is None:
        return None
    preferred = qwen_flash_next.preferred_checkpoint(configuration)
    return {"default": default, "preferred": preferred,
            "others": [name for name in names if name not in (default, preferred)]}


def catalog():
    rows = []
    aliases = {}
    for alias, ident in ALIASES.items():
        aliases.setdefault(ident, []).append(alias)
    for ident in profiles.catalog():
        definition, _ = profiles.load(ident)
        resolved = profiles.resolve(ident)
        record = thinking.of(ident)
        automated = ident in installer.INSTALLABLE
        rows.append({"profile": ident, "title": definition["title"],
                     "nodes": resolved.get("serving", {}).get("node_count"),
                     "model": resolved.get("model", {}), "guide": definition["guide"],
                     "automated": automated, "aliases": sorted(aliases.get(ident, ())),
                     # The checkpoints an installation selects among (checkpoints()), or None.
                     "checkpoints": checkpoints(definition) if automated else None,
                     # The default checkpoint's thinking record (runtime/common/thinking.py), or None.
                     "thinking": None if record is None else {key: record[key] for key in THINKING_FIELDS if key in record}})
    return rows


def thinking_text(record):
    """A profile's thinking default and levels in one line: ``on · xhigh (levels: low, medium or xhigh)``."""
    levels = thinking.levels_text(record)
    return thinking.summary(record) + (f" (levels: {levels})" if levels else " (no levels)")


def checkpoint_text(record):
    """A checkpoints() record in one line, such as
    ``csf where the image's vLLM reads it, else nvfp4-spark; --checkpoint also takes nvfp4-qad``."""
    text = (f"{record['preferred']} where the image's vLLM reads it, else {record['default']}"
            if record["preferred"] else record["default"])
    if record["others"]:
        text += "; --checkpoint also takes " + (" or ".join(record["others"]) if len(record["others"]) < 3
                                               else ", ".join(record["others"][:-1]) + " or " + record["others"][-1])
    return text


def select(value, nodes):
    if value in ("qwen", "glm", "deepseek"):
        raise ValueError(f"'{value}' names a model family. Run 'sparkring models' and select the exact profile.")
    value = canonical(value)
    matches = [row for row in catalog() if row["profile"] == value]
    if not matches:
        replaced = profiles.replacement_message(value)
        if replaced:
            raise ValueError(replaced + ". Run 'sparkring models' for exact model/version/topology choices.")
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
            if row["thinking"] is not None:
                print("  Thinking: " + thinking_text(row["thinking"]))
            if row["checkpoints"] is not None:
                print("  Checkpoint: " + checkpoint_text(row["checkpoints"]))
            if row["aliases"]:
                print("  Also selected as: " + ", ".join(row["aliases"]))
        print("Install a profile with: sudo sparkring install --profile PROFILE")
    return 0
