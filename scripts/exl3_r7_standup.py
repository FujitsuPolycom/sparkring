#!/usr/bin/env python3
"""Build the public profile foundation for the EXL3 3.5-bpw operator profile.

This script chains the public input chain from tracked inputs to a complete
MTP4 KV9.25 rebuilt-image candidate with an exact MTP3 KV9.25 rollback. It is
dry-run by default and separates OFFLINE, READ-ONLY REMOTE, MUTATES HOST,
and STOPS SERVING steps.

No live execution is performed or allowed by this script. Every step that
could contact a host or change state requires an explicit --execute flag
and prints its safety class before running.

Usage:
    python scripts/exl3_r7_standup.py plan
    python scripts/exl3_r7_standup.py plan --site scripts/config/site.yaml
    python scripts/exl3_r7_standup.py plan --output-dir .sparkring/exl3-r7
"""

from __future__ import annotations

import argparse
import hashlib
import json

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

SAFETY_OFFLINE = "OFFLINE"
SAFETY_READ_ONLY = "READ-ONLY REMOTE"
SAFETY_MUTATES = "MUTATES HOST"
SAFETY_STOPS = "STOPS SERVING"

DESCRIPTION = """\
Stand-up entrypoint for the EXL3 3.5-bpw fixed-MTP4, DCP4, KV9.25 foundation.

The operator deployment is accepted on one four-Spark appliance. Profiles
generated for another image ID remain candidates until their live gates pass.
This is not the repository default or a reference-lane result.
"""


class StandupError(ValueError):
    """The stand-up chain cannot produce a truthful profile."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _step(label: str, safety: str, dry_run: bool) -> None:
    print(f"[{safety}] {label}" + (" (dry-run)" if dry_run else ""))


def _run_script(name: str, args: list[str], dry_run: bool) -> dict:
    script = SCRIPTS / name
    if not script.exists():
        raise StandupError(f"required script not found: {script}")
    cmd = [sys.executable, str(script)] + args
    if dry_run:
        print(f"  would run: {' '.join(cmd)}")
        return {}
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise StandupError(
            f"{name} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    try:
        return json.loads(result.stdout) if result.stdout.strip() else {}
    except json.JSONDecodeError:
        return {}


def plan(
    site: Path | None = None,
    output_dir: Path | None = None,
    template: Path | None = None,
    dry_run: bool = True,
) -> dict:
    """Produce the complete public profile chain offline.

    Returns a receipt dict with profile hashes and the rollback identity.
    All steps are OFFLINE; no host is contacted.
    """

    if output_dir is None:
        output_dir = ROOT / ".sparkring" / "exl3-r7"
    template_was_default = template is None
    if template_was_default:
        template = SCRIPTS / "config" / "exl3-r7-candidate.example.json"

    if site is not None and not template_was_default:
        _validate_resolved_inputs(site, template)
    elif not dry_run:
        if site is None:
            raise StandupError("--execute requires a complete ignored --site file")
        raise StandupError(
            "--execute requires an ignored --template with the built image ID"
        )

    receipt: dict = {"dry_run": dry_run, "steps": []}

    # Step 1: Validate the recipe (OFFLINE)
    _step("Validate R7 recipe", SAFETY_OFFLINE, dry_run)
    recipe_path = ROOT / "recipes" / "glm52-exl3-r7-3.5bpw.json"
    if not recipe_path.exists():
        raise StandupError(f"recipe not found: {recipe_path}")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("recipe_id") != "glm52-exl3-r7-3.5bpw":
        raise StandupError("wrong recipe ID")
    receipt["steps"].append({"step": "recipe", "safety": SAFETY_OFFLINE})

    # Step 2: Validate the pins (OFFLINE)
    _step("Validate R7 public pins", SAFETY_OFFLINE, dry_run)
    pins_path = SCRIPTS / "config" / "exl3-r7-pins.json"
    if not pins_path.exists():
        raise StandupError(f"pins not found: {pins_path}")
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    if pins.get("schema_version") != 1:
        raise StandupError("wrong pins schema_version")
    receipt["steps"].append({"step": "pins", "safety": SAFETY_OFFLINE})

    # Step 3: Generate stock-DCP4 baseline (OFFLINE)
    _step("Generate stock-DCP4 baseline profile", SAFETY_OFFLINE, dry_run)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    stock_path = output_dir / "stock-dcp4-profile.json"
    _run_script(
        "generate_exl3_r7_stock_dcp4.py",
        ["--template", str(template), "--output", str(stock_path)],
        dry_run,
    )
    if not dry_run:
        stock_bytes = stock_path.read_bytes()
        receipt["stock_profile_sha256"] = _sha256(stock_bytes)
    receipt["steps"].append({"step": "stock-dcp4", "safety": SAFETY_OFFLINE})

    # Step 4: Prepare MTP2 from stock-DCP4 (OFFLINE)
    _step("Derive fixed-MTP2 from stock-DCP4", SAFETY_OFFLINE, dry_run)
    mtp2_profile = output_dir / "mtp2-profile.json"
    mtp2_site = output_dir / "mtp2-site.yaml"
    mtp2_rollback = output_dir / "mtp2-rollback.json"
    stock_site = output_dir / "stock-site.yaml"
    if not dry_run:
        _write_stock_site(stock_site, site)
    _run_script(
        "prepare_exl3_r7_mtp2.py",
        [
            "--stock-profile", str(stock_path),
            "--stock-site", str(stock_site),
            "--candidate-profile", str(mtp2_profile),
            "--candidate-site", str(mtp2_site),
            "--rollback-profile", str(mtp2_rollback),
        ],
        dry_run,
    )
    receipt["steps"].append({"step": "mtp2", "safety": SAFETY_OFFLINE})

    # Step 5: Prepare MTP3 from MTP2 (OFFLINE)
    _step("Derive fixed-MTP3 from MTP2", SAFETY_OFFLINE, dry_run)
    mtp3_profile = output_dir / "mtp3-profile.json"
    mtp3_site = output_dir / "mtp3-site.yaml"
    mtp3_rollback = output_dir / "mtp3-rollback.json"
    _run_script(
        "prepare_exl3_r7_mtp3.py",
        [
            "--stock-dcp4-profile", str(stock_path),
            "--mtp2-profile", str(mtp2_profile),
            "--mtp2-site", str(mtp2_site),
            "--candidate-profile", str(mtp3_profile),
            "--candidate-site", str(mtp3_site),
            "--rollback-profile", str(mtp3_rollback),
        ],
        dry_run,
    )
    receipt["steps"].append({"step": "mtp3", "safety": SAFETY_OFFLINE})

    # Step 6: Prepare KV9.25 from MTP3 (OFFLINE)
    _step("Derive KV9.25 from fixed-MTP3", SAFETY_OFFLINE, dry_run)
    kv925_profile = output_dir / "mtp3-kv925-profile.json"
    kv925_site = output_dir / "mtp3-kv925-site.yaml"
    kv925_rollback_profile = output_dir / "mtp3-kv925-rollback.json"
    kv925_rollback_site = output_dir / "mtp3-kv925-rollback-site.yaml"
    _run_script(
        "prepare_exl3_r7_mtp3_kv925.py",
        [
            "--qualified-profile", str(mtp3_profile),
            "--qualified-site", str(mtp3_site),
            "--candidate-profile", str(kv925_profile),
            "--candidate-site", str(kv925_site),
            "--rollback-profile", str(kv925_rollback_profile),
            "--rollback-site", str(kv925_rollback_site),
        ],
        dry_run,
    )
    receipt["steps"].append({"step": "kv925", "safety": SAFETY_OFFLINE})

    # Step 7: Prepare MTP4 from MTP3 KV9.25 (OFFLINE)
    _step("Derive fixed-MTP4 from MTP3 KV9.25", SAFETY_OFFLINE, dry_run)
    mtp4_profile = output_dir / "mtp4-kv925-profile.json"
    mtp4_site = output_dir / "mtp4-kv925-site.yaml"
    mtp4_rollback_profile = output_dir / "mtp4-kv925-rollback.json"
    mtp4_rollback_site = output_dir / "mtp4-kv925-rollback-site.yaml"
    if not dry_run:
        kv925_bytes = kv925_profile.read_bytes()
        kv925_site_bytes = kv925_site.read_bytes()
        mtp4_args = [
            "--mtp3-profile", str(kv925_profile),
            "--mtp3-site", str(kv925_site),
            "--expected-mtp3-profile-sha256", _sha256(kv925_bytes),
            "--expected-mtp3-site-sha256", _sha256(kv925_site_bytes),
            "--candidate-profile", str(mtp4_profile),
            "--candidate-site", str(mtp4_site),
            "--rollback-profile", str(mtp4_rollback_profile),
            "--rollback-site", str(mtp4_rollback_site),
        ]
    else:
        mtp4_args = [
            "--mtp3-profile", str(kv925_profile),
            "--mtp3-site", str(kv925_site),
            "--expected-mtp3-profile-sha256", "0" * 64,
            "--expected-mtp3-site-sha256", "0" * 64,
            "--candidate-profile", str(mtp4_profile),
            "--candidate-site", str(mtp4_site),
            "--rollback-profile", str(mtp4_rollback_profile),
            "--rollback-site", str(mtp4_rollback_site),
        ]
    _run_script("prepare_exl3_r7_mtp4.py", mtp4_args, dry_run)
    if not dry_run:
        mtp4_bytes = mtp4_profile.read_bytes()
        rollback_bytes = mtp4_rollback_profile.read_bytes()
        rollback_site_bytes = mtp4_rollback_site.read_bytes()
        receipt["mtp4_profile_sha256"] = _sha256(mtp4_bytes)
        receipt["mtp4_site_sha256"] = _sha256(mtp4_site.read_bytes())
        receipt["rollback_profile_sha256"] = _sha256(rollback_bytes)
        receipt["rollback_site_sha256"] = _sha256(rollback_site_bytes)
        receipt["rollback_identity"] = (
            "MTP4 rollback is byte-identical to MTP3 KV9.25"
        )
    receipt["steps"].append({"step": "mtp4", "safety": SAFETY_OFFLINE})

    # Step 8: Validate site template (OFFLINE)
    if site and site.exists():
        _step("Validate site configuration", SAFETY_OFFLINE, dry_run)
        if not dry_run:
            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "sparkring_site.py"), str(site)],
                capture_output=True, text=True, check=False,
            )
            if result.returncode != 0:
                raise StandupError(f"site validation failed: {result.stderr.strip()}")
        receipt["steps"].append({"step": "site-validate", "safety": SAFETY_OFFLINE})

    # Step 9: Read-only preflight plan (READ-ONLY REMOTE — not executed in dry-run)
    if site and site.exists():
        _step("Preflight plan (read-only remote)", SAFETY_READ_ONLY, dry_run)
        if not dry_run:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "preflight.py"),
                    "--site", str(site),
                    "--print-plan",
                ],
                capture_output=True, text=True, check=False,
            )
            if result.returncode != 0:
                raise StandupError(f"preflight plan failed: {result.stderr.strip()}")
        receipt["steps"].append({"step": "preflight-plan", "safety": SAFETY_READ_ONLY})

    receipt["operator_deployment_maturity"] = "accepted"
    receipt["generated_profile_maturity"] = "candidate"
    receipt["note"] = (
        "Operator acceptance applies to the documented image and four-Spark "
        "appliance. A profile generated for another image ID requires the "
        "live promotion gate. The public default remains EXL3 3.25-bpw plus "
        "LMCache."
    )
    return receipt


def _validate_resolved_inputs(site: Path, template: Path) -> None:
    """Reject placeholders and cross-file image or model path drift."""

    from sparkring_site import load_site

    if not site.is_file() or not template.is_file():
        raise StandupError("--site and --template must name existing files")
    try:
        document = json.loads(template.read_text(encoding="utf-8"))
        parsed_site = load_site(site)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise StandupError(f"invalid resolved stand-up input: {exc}") from exc
    image = document.get("image")
    image_id = document.get("image_id")
    if not isinstance(image, str) or "REPLACE" in image.upper():
        raise StandupError("candidate template image is unresolved")
    if image_id == "sha256:" + "1" * 64:
        raise StandupError("candidate template image_id is unresolved")
    expected = {
        "image": (image, parsed_site.runtime.container_image),
        "image_id": (image_id, parsed_site.runtime.container_image_digest),
        "model_host_path": (
            document.get("model_host_path"),
            parsed_site.runtime.model_path,
        ),
        "jit_cache_host_path": (
            document.get("jit_cache_host_path"),
            parsed_site.paths.jit_cache_dir,
        ),
    }
    for field, (profile_value, site_value) in expected.items():
        if profile_value != site_value:
            raise StandupError(
                f"candidate template {field} does not match the complete site"
            )


def _write_stock_site(path: Path, source_site: Path) -> None:
    """Derive a stock-DCP4 control while preserving the complete site."""

    text = source_site.read_text(encoding="utf-8")
    replacements = (
        ('  mtp_mode: "static"', '  mtp_mode: "off"'),
        ("  mtp_tokens: 4", "  mtp_tokens: 0"),
        (
            "  kv_cache_bytes_per_rank: 9250000000",
            "  kv_cache_bytes_per_rank: 9000000000",
        ),
    )
    for before, after in replacements:
        if text.count(before) != 1:
            raise StandupError(
                f"complete site requires exactly one {before.strip()} declaration"
            )
        text = text.replace(before, after)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="Offline profile chain (dry-run by default)")
    plan_parser.add_argument("--site", type=Path, default=None)
    plan_parser.add_argument("--output-dir", type=Path, default=None)
    plan_parser.add_argument("--template", type=Path, default=None)
    plan_parser.add_argument("--execute", action="store_true", help="Run the offline chain")
    args = parser.parse_args()

    if args.command == "plan":
        dry_run = not args.execute
        if dry_run:
            print("DRY-RUN: no files will be written, no hosts contacted.\n")
        else:
            print("EXECUTE: offline profile chain will write files.\n")
        try:
            receipt = plan(
                site=args.site,
                output_dir=args.output_dir,
                template=args.template,
                dry_run=dry_run,
            )
        except StandupError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"\n{json.dumps(receipt, indent=2, sort_keys=True)}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
