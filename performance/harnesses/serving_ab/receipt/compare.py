"""Compare two ``resolved.json`` documents, and check an arm against its control and its declared change.

An arm declaration (``serving-ab-arm/v1``, JSON) names the arm, its control and the settings it changes, as
keys of ``resolved.json``'s ``settings`` and the values the arm must have::

    {"schema": "serving-ab-arm/v1", "arm": "WD", "control": "W",
     "changes": {"speculative.kv_cache_dtype": "auto"}}

A change's value may also be ``{"from": CONTROL_VALUE, "to": ARM_VALUE}``. A key ending in ``.*`` with the
value ``null`` declares every difference under that prefix (``"compilation.*": null`` for an arm that
replaces the compilation configuration); it asserts no value.

:func:`check` fails, listing every finding, when

- a setting differs between arm and control and no declared change covers it;
- a declared change did not take effect: the arm lacks the key or holds another value, or the control
  already holds the declared value, or the control does not hold a declared ``from`` value.

It warns, without failing, when a declared change leaves its effective value (``derived.*_effective``)
unchanged, when either document's ranks disagree on a setting, and it reports derived and observed
differences (KV capacity, resolved block size, kernels chosen) as consequences.

A value matches a declared value when their JSON forms are equal, when both are scalars with equal text
(``16`` and ``"16"``, ``True`` and ``"true"``), or when the setting is an enum member and the declared
value is its full name, member name or value (``CompilationMode.NONE``, ``NONE`` or ``0``).
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .resolved import canonical

SCHEMA = "serving-ab-arm/v1"
MISSING = "<absent>"


def load(path) -> dict:
    """A ``resolved.json`` from its path or from a directory holding one."""
    path = Path(path)
    if path.is_dir():
        path = path / "resolved.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_declaration(path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} declaration")
    if not isinstance(document.get("changes"), dict):
        raise ValueError(f"{path}: 'changes' must map setting keys to values")
    return document


def differences(a: Mapping, b: Mapping, *, ignore: Sequence[str] = ()) -> dict:
    """``{key: (a value, b value)}`` for every key whose value differs; absent keys show as :data:`MISSING`."""
    out = {}
    for key in sorted(set(a) | set(b)):
        if any(fnmatch.fnmatchcase(key, pattern) for pattern in ignore):
            continue
        left, right = a.get(key, MISSING), b.get(key, MISSING)
        if (key in a) != (key in b) or canonical(left) != canonical(right):
            out[key] = (left, right)
    return out


def matches(actual, declared) -> bool:
    if canonical(actual) == canonical(declared):
        return True
    if isinstance(actual, dict) and "enum" in actual:
        names = {actual["enum"], actual["enum"].split(".")[-1]}
        return declared in names or canonical(actual.get("value")) == canonical(declared)
    scalar = (str, int, float, bool, type(None))
    if isinstance(actual, scalar) and isinstance(declared, scalar):
        return str(actual).lower() == str(declared).lower()
    return False


def show(value) -> str:
    if isinstance(value, dict) and "enum" in value:
        return f"{value['enum']} ({canonical(value.get('value'))})"
    return value if value == MISSING else canonical(value)


def _covered(key: str, changes: Mapping) -> bool:
    if key in changes:
        return True
    return any(pattern.endswith(".*") and key.startswith(pattern[:-1]) for pattern in changes)


def diff(a: Mapping, b: Mapping, *, ignore: Sequence[str] = ()) -> dict:
    """Every difference of two documents: ``settings`` (failing in a check), ``derived`` and ``observed``."""
    return {part: differences(a.get(part) or {}, b.get(part) or {}, ignore=ignore)
            for part in ("settings", "derived", "observed")}


def check(arm: Mapping, control: Mapping, declaration: Mapping, *, ignore: Sequence[str] = ()) -> dict:
    """The verdict of an arm against its control (see the module docstring)."""
    changes = dict(declaration.get("changes") or {})
    found = diff(arm, control, ignore=ignore)
    problems, warnings = [], []
    arm_settings, control_settings = arm.get("settings") or {}, control.get("settings") or {}
    for key, (mine, theirs) in found["settings"].items():
        if not _covered(key, changes):
            problems.append({"kind": "undeclared", "key": key, "arm": show(mine), "control": show(theirs),
                             "text": f"{key} differs and is not declared: arm {show(mine)}, control {show(theirs)}"})
    for key, value in changes.items():
        if key.endswith(".*"):
            prefix = key[:-1]
            if value is not None:
                problems.append({"kind": "declaration", "key": key,
                                 "text": f"{key}: a prefix declaration takes no value"})
            elif not any(k.startswith(prefix) for k in found["settings"]):
                problems.append({"kind": "no-effect", "key": key,
                                 "text": f"{key}: declared, but nothing under {prefix[:-1]} differs from the control"})
            continue
        target, origin = (value.get("to"), value.get("from")) if isinstance(value, dict) and set(value) <= {"from", "to"} \
            else (value, None)
        has_from = isinstance(value, dict) and "from" in value
        if key not in arm_settings:
            problems.append({"kind": "no-effect", "key": key,
                             "text": f"{key}: declared {show(target)}, but the arm's resolved settings lack it"})
            continue
        if not matches(arm_settings[key], target):
            problems.append({"kind": "no-effect", "key": key, "arm": show(arm_settings[key]),
                             "text": f"{key}: declared {show(target)}, but the arm resolved {show(arm_settings[key])}"})
        elif key in control_settings and canonical(control_settings[key]) == canonical(arm_settings[key]):
            problems.append({"kind": "no-effect", "key": key,
                             "text": f"{key}: declared {show(target)}, but the control already has it"})
        if has_from and not matches(control_settings.get(key, MISSING), origin):
            problems.append({"kind": "control", "key": key,
                             "text": f"{key}: declared from {show(origin)}, but the control resolved "
                                     f"{show(control_settings.get(key, MISSING))}"})
        effective = key + "_effective"
        derived_arm, derived_control = arm.get("derived") or {}, control.get("derived") or {}
        if effective in derived_arm and canonical(derived_arm[effective]) == canonical(derived_control.get(effective)):
            warnings.append(f"{key}: the change leaves {effective} at {show(derived_arm[effective])} "
                            f"({(arm.get('sources') or {}).get(effective, 'derived')})")
    for name, document in (("arm", arm), ("control", control)):
        for mismatch in document.get("rank_mismatches") or []:
            warnings.append(f"{name}: rank {mismatch['rank']} resolved {mismatch['key']} = {show(mismatch['value'])}, "
                            f"rank 0 {show(mismatch['rank0'])}")
    return {
        "schema": "serving-ab-arm-check/v1",
        "arm": declaration.get("arm"), "control": declaration.get("control"),
        "declared": changes, "arm_digest": arm.get("digest"), "control_digest": control.get("digest"),
        "passed": not problems, "problems": problems, "warnings": warnings,
        "consequences": {part: {k: [show(x), show(y)] for k, (x, y) in found[part].items()}
                         for part in ("derived", "observed")},
    }


def format_diff(found: Mapping, *, labels=("A", "B")) -> str:
    lines = []
    for part in ("settings", "derived", "observed"):
        rows = found.get(part) or {}
        lines.append(f"{part}: {len(rows)} difference(s)")
        for key, (left, right) in rows.items():
            lines.append(f"  {key}: {labels[0]} {show(left)} | {labels[1]} {show(right)}")
    return "\n".join(lines)


def format_check(result: Mapping) -> str:
    lines = [f"arm {result.get('arm')} against control {result.get('control')}: "
             f"{'PASSED' if result['passed'] else 'FAILED'}",
             "declared: " + (", ".join(f"{k} = {show(v)}" for k, v in result["declared"].items()) or "nothing")]
    lines += [f"  FAIL {p['text']}" for p in result["problems"]]
    lines += [f"  WARN {w}" for w in result["warnings"]]
    for part in ("derived", "observed"):
        for key, (left, right) in result["consequences"][part].items():
            lines.append(f"  note {part} {key}: arm {left} | control {right}")
    return "\n".join(lines)
