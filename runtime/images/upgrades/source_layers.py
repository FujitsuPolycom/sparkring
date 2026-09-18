"""Reconstruct a carried patch from an approved source-extension descriptor.

The output extends an existing baseline, not the upstream candidate. Exact
preimages and postimages are required before this feature set enters a policy.
No source oracle, compatibility binding or qualification is inferred.
"""

from pathlib import Path
import re

from .contracts import relative, require, sha
from .sources import apply_patch, complete_patch, copy_snapshot, git, inventory


def compose_extension(baseline, output, component, descriptor, patch):
    """Return one complete baseline patch without modifying its input snapshot.

    The baseline's Git index must still describe its unpatched upstream tree,
    as produced by the upgrade runner. Its working tree contains carried patches.
    """
    require(component in ('vllm', 'b12x'), 'Unsupported source component')
    require(descriptor.get('schema') == 'sparkring-source-extension/v1', 'Unknown extension schema')
    require(sha(patch) == descriptor['patch']['sha256'], 'Extension patch differs from descriptor')
    baseline = Path(baseline).resolve()
    output = Path(output).resolve()
    require(not output.is_relative_to(baseline), 'Output cannot be inside the protected baseline')
    records = {name: value for name, value in descriptor['sources'].items()
               if name.startswith(component + '/')}
    require(records, 'Extension has no files for this component')
    for name, row in records.items():
        relative(name)
        path = baseline / name
        require(not path.is_symlink(), 'Extension preimage is a symlink')
        actual = sha(path.read_bytes()) if path.is_file() else None
        require(actual == row['parent_sha256'], 'Extension preimage differs: ' + name)
    fragments = []
    selected = set()
    for fragment in re.split(rb'(?=^diff --git )', patch, flags=re.MULTILINE):
        if not fragment.strip():
            continue
        match = re.match(rb'diff --git a/(\S+) b/(\S+)\r?\n', fragment)
        require(match and match[1] == match[2], 'Only same-path extension patches are admitted')
        name = match[1].decode('utf-8')
        require(name in descriptor['sources'], 'Patch changes an undeclared source file')
        if name.startswith(component + '/'):
            require(name not in selected, 'Duplicate extension patch file')
            selected.add(name)
            fragments.append(fragment)
    require(selected == set(records), 'Extension patch and source inventory differ')
    original_tree = git(baseline, 'write-tree').decode().strip()
    target = copy_snapshot(baseline, output)
    before = inventory(target)
    applied, feedback = apply_patch(target, b''.join(fragments))
    require(applied, 'Extension application failed: ' + str(feedback))
    for name, row in records.items():
        path = target / name
        require(path.is_file() and sha(path.read_bytes()) == row['sha256'],
                'Extension postimage differs: ' + name)
    after = inventory(target)
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    require(changed <= records.keys(), 'Extension changed unregistered files')
    result = complete_patch(target, original_tree)
    return result, {'schema': 'sparkring-carried-source-layer/v1', 'component': component,
                    'source_extension': descriptor['id'], 'patch_sha256': sha(result),
                    'files_verified': sorted(records), 'serving_qualified': False}
