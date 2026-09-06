"""Execute reviewed deployment plans with phase barriers and resumable receipts."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Callable


def plan_digest(plan: dict) -> str:
    payload = {k: v for k, v in plan.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def seal_plan(plan: dict) -> dict:
    return {**plan, "sha256": plan_digest(plan)}


def validate_plan(plan: dict) -> None:
    if plan.get("schema") != "sparkring-deploy-plan/v1" or plan.get(
        "sha256"
    ) != plan_digest(plan):
        raise ValueError("Plan changed; generate and review it again")
    names = set()
    for phase in plan.get("phases", []):
        if not phase.get("id") or phase["id"] in names:
            raise ValueError("Phase IDs must be unique")
        names.add(phase["id"])
        hosts = set()
        for action in phase["actions"]:
            if action["host"] in hosts:
                raise ValueError("Each phase may contain one action per host")
            hosts.add(action["host"])
            if not action.get("argv") or not all(
                isinstance(a, str) and "\x00" not in a for a in action["argv"]
            ):
                raise ValueError("Actions require an argument array")
            if action.get("risk") not in (
                "read-only",
                "mutates-host",
                "driver-reload",
                "starts-model",
                "stops-model",
                "hardware-test",
            ):
                raise ValueError("Action risk must be explicit")
            if action["risk"] != "read-only" and not action.get("verify"):
                raise ValueError("Changing actions require a verification command")


class CommandRunner:
    """Run local commands or safely quoted commands through configured SSH aliases."""

    def __call__(self, host: str, argv: list[str], timeout: float = 120) -> dict:
        import re

        if host != "controller" and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host
        ):
            raise ValueError("Invalid SSH target")
        command = (
            argv
            if host == "controller"
            else [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                host,
                shlex.join(argv),
            ]
        )
        started = time.monotonic()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "seconds": time.monotonic() - started,
                "uncertain": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "returncode": 124,
                "stdout": "",
                "stderr": "Command timed out; remote work may still be running",
                "seconds": time.monotonic() - started,
                "uncertain": True,
            }


def save_receipt(path: Path, receipt: dict) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def verified(result: dict, expected: dict) -> bool:
    if result["returncode"] != 0 or result.get("uncertain"):
        return False
    if "stdout" in expected and result["stdout"].strip() != expected["stdout"]:
        return False
    if "json" in expected:
        try:
            document = json.loads(result["stdout"])
        except (TypeError, ValueError):
            return False
        if not matches(document, expected["json"]):
            return False
    return True


def matches(value, expected):
    """Match explicit object fields and list records; an empty list means empty."""
    if isinstance(expected, dict):
        return isinstance(value, dict) and all(
            k in value and matches(value[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(value, list):
            return False
        return (
            value == []
            if expected == []
            else all(
                any(matches(item, wanted) for item in value) for wanted in expected
            )
        )
    return type(value) is type(expected) and value == expected


def execute_plan(plan: dict, receipt_path: Path, approval: str, **options) -> dict:
    """Hold a receipt lock so two controllers cannot apply the same plan at once."""
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    lock = receipt_path.with_name(receipt_path.name + ".lock")
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            "Deployment receipt is locked; inspect the owning process before recovery"
        ) from exc
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        return _execute_plan(plan, receipt_path, approval, **options)
    finally:
        lock.unlink()


def _execute_plan(
    plan: dict,
    receipt_path: Path,
    approval: str,
    *,
    runner: Callable | None = None,
    resume: bool = False,
    allow_driver_reload: bool = False,
    allow_model_actions: bool = False,
    allow_hardware_tests: bool = False,
) -> dict:
    """Apply one reviewed plan; never retry an uncertain mutation automatically."""
    validate_plan(plan)
    if approval != plan["sha256"]:
        raise ValueError("Approval must name the exact plan SHA-256")
    for phase in plan["phases"]:
        for action in phase["actions"]:
            if action["risk"] == "driver-reload" and not allow_driver_reload:
                raise ValueError(
                    "Plan contains a driver reload; explicit driver-reload authorization required"
                )
            if (
                action["risk"] in ("starts-model", "stops-model")
                and not allow_model_actions
            ):
                raise ValueError(
                    "Plan contains model actions; explicit model-action authorization required"
                )
            if action["risk"] == "hardware-test" and not allow_hardware_tests:
                raise ValueError(
                    "Plan contains GPU/RDMA tests; explicit hardware-test authorization required"
                )
    run = runner or CommandRunner()
    if receipt_path.exists():
        if not resume:
            raise ValueError("Receipt exists; use resume after inspecting it")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("plan_sha256") != plan["sha256"]:
            raise ValueError("Receipt belongs to a different plan")
    else:
        if resume:
            raise ValueError("No receipt exists to resume")
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt = {
            "schema": "sparkring-deploy-receipt/v1",
            "plan_sha256": plan["sha256"],
            "actions": {},
            "complete": False,
        }
    # A resumed receipt describes an in-progress recheck until every phase passes.
    receipt["complete"] = False
    save_receipt(receipt_path, receipt)
    for phase in plan["phases"]:
        for action in phase["actions"]:
            if action["risk"] == "driver-reload" and not allow_driver_reload:
                raise ValueError(
                    "Plan contains a driver reload; explicit driver-reload authorization required"
                )
            if (
                action["risk"] in ("starts-model", "stops-model")
                and not allow_model_actions
            ):
                raise ValueError(
                    "Plan contains model actions; explicit model-action authorization required"
                )
        # Preflight every host before any mutation in this phase.
        for action in phase["actions"]:
            key = phase["id"] + ":" + action["host"]
            old = receipt["actions"].get(key)
            if old and old["state"] in ("running", "uncertain"):
                raise ValueError(
                    f"{key}: prior outcome is uncertain; inspect host state before creating a recovery plan"
                )
            if old and old["state"] == "succeeded":
                continue
            for check in action.get("checks", []):
                result = run(action["host"], check["argv"], check.get("timeout", 120))
                if not verified(result, check):
                    receipt["failure"] = {
                        "action": key,
                        "stage": "preflight",
                        "result": result,
                    }
                    save_receipt(receipt_path, receipt)
                    raise RuntimeError(f"{key}: preflight failed")
        todo = []
        for action in phase["actions"]:
            key = phase["id"] + ":" + action["host"]
            old = receipt["actions"].get(key)
            if old and old["state"] == "succeeded":
                check = action.get("verify") or {
                    "argv": action["argv"],
                    "timeout": action.get("timeout", 120),
                }
                result = run(action["host"], check["argv"], check.get("timeout", 120))
                old["resume_verification"] = result
                if not verified(result, check):
                    receipt["failure"] = {
                        "action": key,
                        "stage": "resume_verification",
                        "result": result,
                    }
                    save_receipt(receipt_path, receipt)
                    raise RuntimeError(
                        f"{key}: completed action no longer verifies; review drift"
                    )
                save_receipt(receipt_path, receipt)
                continue
            receipt["actions"][key] = {"state": "running"}
            todo.append((key, action))
        save_receipt(receipt_path, receipt)

        def apply(item):
            key, action = item
            result = run(action["host"], action["argv"], action.get("timeout", 120))
            state = (
                "uncertain"
                if result.get("uncertain")
                else "failed"
                if result["returncode"]
                else "succeeded"
            )
            if state == "failed" and action["risk"] != "read-only":
                state = "uncertain"
            if state == "succeeded" and action.get("verify"):
                check = action["verify"]
                validation = run(
                    action["host"], check["argv"], check.get("timeout", 120)
                )
                if not verified(validation, check):
                    state = "uncertain"
                result["verification"] = validation
            return key, {"state": state, "result": result}

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for key, result in pool.map(apply, todo):
                receipt["actions"][key] = result
                save_receipt(receipt_path, receipt)
        if any(receipt["actions"][key]["state"] != "succeeded" for key, _ in todo):
            raise RuntimeError(
                f"{phase['id']}: phase failed; later phases were not started"
            )
    receipt["complete"] = True
    receipt.pop("failure", None)
    save_receipt(receipt_path, receipt)
    return receipt
