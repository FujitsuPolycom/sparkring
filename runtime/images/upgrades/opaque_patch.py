"""Describe binary patch assets without exposing opaque payloads as source text.

Binary assets remain in the approved carried patch and installed-source checks.
The LLM receives identities and presence/size facts, not a claim that it has
inspected their contents. Their paths are protected against proposal edits.
"""

import copy
from pathlib import Path
import re
import shlex

from runtime.images.upgrades.contracts import beneath, sha, require, relative


def binary_fragments(patch):
    result = []
    for part in re.split(rb"(?=^diff --git )", patch, flags=re.MULTILINE):
        if b"\nGIT binary patch\n" not in part:
            continue
        fields = shlex.split(part.splitlines()[0].decode("utf-8"))
        require(
            len(fields) == 4 and fields[:2] == ["diff", "--git"],
            "Binary patch header is not an admitted Git diff",
        )
        paths = []
        for value, prefix in zip(fields[2:], ("a/", "b/")):
            require(value.startswith(prefix), "Binary patch path has no Git prefix")
            name = value[len(prefix) :]
            relative(name)
            # beneath performs the shared path/symlink checks when roots exist.
            if name not in paths:
                paths.append(name)
        result.append(
            {
                "paths": paths,
                "patch_sha256": sha(part),
                "patch_bytes": len(part),
                "raw": part,
            }
        )
    return result


def protected_paths(patch):
    """Keep opaque carried assets outside LLM-proposed source edits."""
    return sorted({name for part in binary_fragments(patch) for name in part["paths"]})


def compact(request, *, roots):
    original = request["carried_patch"].encode()
    fragments = binary_fragments(original)
    if not fragments:
        return copy.deepcopy(request), []
    by_sha = {row["patch_sha256"]: row for row in fragments}
    text = []
    for part in re.split(rb"(?=^diff --git )", original, flags=re.MULTILINE):
        if sha(part) not in by_sha:
            text.append(part)
    value = copy.deepcopy(request)
    value["carried_patch"] = b"".join(text).decode()
    value["complete_carried_patch_sha256"] = sha(original)
    value["opaque_patch_assets"] = []
    protected = set()
    for row in fragments:
        record = {key: item for key, item in row.items() if key != "raw"}
        record["source_files"] = {}
        for name in row["paths"]:
            protected.add(name)
            record["source_files"][name] = {}
            for role, root in roots.items():
                path = beneath(Path(root), name, exists=False)
                record["source_files"][name][role] = (
                    {
                        "present": True,
                        "bytes": path.stat().st_size,
                        "sha256": sha(path.read_bytes()),
                    }
                    if path.is_file()
                    else {"present": False}
                )
        value["opaque_patch_assets"].append(record)
    if isinstance(value.get("feedback"), list):
        for row in value["feedback"]:
            raw = row.get("patch", "").encode()
            identifier = sha(raw)
            if identifier in by_sha:
                del row["patch"]
                row["opaque_patch_sha256"] = identifier
    value["opaque_asset_policy"] = (
        "Binary patch payloads are not shown. The complete approved patch remains hash-bound and is not modified by this view. "
        "Use the asset identities and file-presence facts for compatibility analysis; do not claim content inspection. "
        "Proposal edits to these paths are forbidden. An unresolved asset migration must remain unresolved."
    )
    return value, sorted(protected)
