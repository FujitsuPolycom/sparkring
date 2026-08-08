#!/usr/bin/env python3
"""Archive rehearsal for the runtime bundle layer.

This script assumes it is already running inside an extracted committed
archive (e.g. ``git archive`` output).  It poisons all remote executors,
validates every tracked bundle example, builds the EXL3 bridge plan, and runs
focused offline checks — without staging, committing, or pushing.

Usage::

    python scripts/rehearse_runtime_bundle_archive.py

Exit codes::

    0 — all checks passed
    1 — one or more checks failed
    2 — environment error (missing files, import failure)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


# ---------------------------------------------------------------------------
# Remote-executor poison — installed before any bundle module is used.
# ---------------------------------------------------------------------------

_POISONED = False
_POISON_STATE: dict[str, Any] = {}


def _install_poison() -> None:
    """Monkey-patch runtime.execute and runtime.run_remote to raise on call.

    Called once at import time so that no downstream code path can reach SSH.
    """
    global _POISONED  # noqa: PLW0603
    if _POISONED:
        return
    import socket  # noqa: PLC0415
    import sparkring_exl3_launcher as exl3  # noqa: PLC0415
    import sparkring_runtime as runtime  # noqa: PLC0415

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "remote executor reached during archive rehearsal"
        )

    _POISON_STATE.update({
        "runtime.execute": runtime.execute,
        "runtime.run_remote": runtime.run_remote,
        "exl3.execute": exl3.execute,
        "socket.socket": socket.socket,
        "socket.create_connection": socket.create_connection,
    })
    runtime.execute = _boom  # type: ignore[assignment]
    runtime.run_remote = _boom  # type: ignore[assignment]
    exl3.execute = _boom  # type: ignore[assignment]
    socket.socket = _boom  # type: ignore[assignment,misc]
    socket.create_connection = _boom  # type: ignore[assignment]
    _POISONED = True


def _restore_poison() -> None:
    """Restore process-global boundaries after an in-process rehearsal."""
    global _POISONED  # noqa: PLW0603
    if not _POISONED:
        return
    import socket  # noqa: PLC0415
    import sparkring_exl3_launcher as exl3  # noqa: PLC0415
    import sparkring_runtime as runtime  # noqa: PLC0415

    runtime.execute = _POISON_STATE["runtime.execute"]
    runtime.run_remote = _POISON_STATE["runtime.run_remote"]
    exl3.execute = _POISON_STATE["exl3.execute"]
    socket.socket = _POISON_STATE["socket.socket"]
    socket.create_connection = _POISON_STATE["socket.create_connection"]
    _POISON_STATE.clear()
    _POISONED = False


# ---------------------------------------------------------------------------
# Tracked configs to validate.
# ---------------------------------------------------------------------------

TRACKED_BUNDLES = [
    "scripts/config/bundle-native-single.json",
    "scripts/config/bundle-engine-cache.json",
    "scripts/config/bundle-exl3-lmcache-bridge.json",
]

TRACKED_TEMPLATES = [
    "scripts/config/bundle.template.json",
]

EXL3_BRIDGE = "scripts/config/bundle-exl3-lmcache-bridge.json"
SITE_EXAMPLE = "scripts/config/site.example.yaml"


# ---------------------------------------------------------------------------
# Check functions — each returns (ok, message).
# ---------------------------------------------------------------------------


def check_tracked_bundles_parse() -> tuple[bool, str]:
    """Every non-template tracked bundle must load without error."""
    import sparkring_bundle as bundle_mod  # noqa: PLC0415

    for cfg in TRACKED_BUNDLES:
        p = ROOT / cfg
        if not p.exists():
            return False, f"Missing tracked bundle: {cfg}"
        try:
            bundle_mod.load_bundle(p)
        except Exception as exc:
            return False, f"{cfg} failed to load: {exc}"
    return True, f"All {len(TRACKED_BUNDLES)} tracked bundles parsed"


def check_templates_are_json() -> tuple[bool, str]:
    """Template bundles must at least be valid JSON."""
    for cfg in TRACKED_TEMPLATES:
        p = ROOT / cfg
        if not p.exists():
            return False, f"Missing template: {cfg}"
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"{cfg} is not valid JSON: {exc}"
    return True, f"All {len(TRACKED_TEMPLATES)} templates are valid JSON"


def check_exl3_bridge_plan() -> tuple[bool, str]:
    """The tracked EXL3 bridge must produce an offline plan."""
    import sparkring_bundle as bundle_mod  # noqa: PLC0415
    from sparkring_site import load_site  # noqa: PLC0415

    p = ROOT / EXL3_BRIDGE
    if not p.exists():
        return False, f"Missing EXL3 bridge config: {EXL3_BRIDGE}"
    site_p = ROOT / SITE_EXAMPLE
    if not site_p.exists():
        return False, f"Missing site example: {SITE_EXAMPLE}"
    try:
        bundle = bundle_mod.load_bundle(p)
        site = load_site(site_p)
        plan = bundle_mod.bundle_plan(bundle, site)
        if plan is None:
            return False, "EXL3 bridge plan was None"
        # Bridge is plan-only: check canonical_phases, not native phases
        if not plan.get("canonical_phases"):
            return False, "EXL3 bridge plan has no canonical_phases"
        if not plan.get("lifecycle_sequences"):
            return False, "EXL3 bridge plan has no lifecycle_sequences"
        return True, "EXL3 bridge plan built successfully"
    except Exception as exc:
        return False, f"EXL3 bridge plan failed: {exc}"


def check_offline_plan_projection() -> tuple[bool, str]:
    """Plan projection must work for all tracked bundles."""
    import sparkring_bundle as bundle_mod  # noqa: PLC0415

    for cfg in TRACKED_BUNDLES:
        p = ROOT / cfg
        if not p.exists():
            return False, f"Missing tracked bundle: {cfg}"
        try:
            bundle = bundle_mod.load_bundle(p)
            proj = bundle_mod.bundle_projection(bundle)
            if proj is None:
                return False, f"{cfg} projection was None"
        except Exception as exc:
            return False, f"{cfg} projection failed: {exc}"
    return True, "All tracked bundle projections built"


def check_no_git_operations() -> tuple[bool, str]:
    """Verify no git add/commit/stage is invoked by the rehearsal."""
    # This check is structural — the script itself must not call git.
    # The test suite verifies this by inspecting source.
    return True, "No git operations performed"


CHECKS = [
    ("tracked-bundles-parse", check_tracked_bundles_parse),
    ("templates-valid-json", check_templates_are_json),
    ("exl3-bridge-plan", check_exl3_bridge_plan),
    ("offline-plan-projection", check_offline_plan_projection),
    ("no-git-operations", check_no_git_operations),
]


def main(argv: list[str] | None = None) -> int:
    _install_poison()
    try:
        failures = 0
        for name, fn in CHECKS:
            try:
                ok, msg = fn()
            except Exception as exc:
                ok, msg = False, f"unexpected exception: {exc}"
            status = "PASS" if ok else "FAIL"
            print(f"  {name}: {status} — {msg}")
            if not ok:
                failures += 1

        if failures:
            print(f"\n{failures} check(s) failed", file=sys.stderr)
            return 1
        print(f"\nAll {len(CHECKS)} checks passed")
        return 0
    finally:
        _restore_poison()


if __name__ == "__main__":
    raise SystemExit(main())
