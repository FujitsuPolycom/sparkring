"""Build standalone managed-runtime plans from prepared SparkRing site inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import importlib.util
import secrets
import sys

try:
    from .deploy_engine import plan_digest, seal_plan
except ImportError:
    from deploy_engine import plan_digest, seal_plan

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "runtime/glm53-spark-mtp3-mesh"


def probe_readiness(launch, output_root, *, wait=None, load=None):
    """Sample readiness afresh and preserve each result under an unused filename."""
    if wait is None or load is None:
        spec = importlib.util.spec_from_file_location(
            "deploy_readiness", PROFILE / "wait_managed_ready.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        wait, load = module.wait, module.load_launch
    output_root = Path(output_root)
    if any(p.is_symlink() for p in (output_root, *output_root.parents)):
        raise ValueError("Readiness output cannot contain symlinks")
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    result = wait(load(Path(launch)), 900)
    output = output_root / ("ready-" + secrets.token_hex(16) + ".json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    output.chmod(0o600)
    result = {**result, "receipt": str(output)}
    print(json.dumps(result))
    return result


def _action(host, argv, risk, verify, *, checks=(), timeout=120):
    return {
        "host": host,
        "argv": argv,
        "risk": risk,
        "verify": verify,
        "checks": list(checks),
        "timeout": timeout,
    }


def _checked_path(value, label):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"/srv/sparkring/[A-Za-z0-9_-]+", value)
        or ".." in Path(value).parts
    ):
        raise ValueError(f"{label} must be a dedicated path under /srv/sparkring")
    return value


def verify_test_receipt(path, action):
    """Reject failed or incomplete readiness and native-test receipts."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Test receipt cannot be a symlink")
    if action == "ready":
        result = json.loads(path.read_text())
        if (
            result.get("schema") != "sparkring-managed-model-readiness/v1"
            or result.get("ready") is not True
        ):
            raise ValueError("Model readiness did not pass")
    elif action == "native-check":
        plan = json.loads((path / "plan.json").read_text())
        cells = plan.get("cells", [])
        if plan.get("schema") != "sparkring-mtp3-native-plan/v1" or not cells:
            raise ValueError("Native-test plan is missing its test cells")
        for cell in cells:
            results = json.loads((path / f"q{cell['rows']}.json").read_text())
            if (
                len(results) != 4
                or any(row.get("returncode") != 0 for row in results)
                or "EVIDENCE_JSON " not in results[0].get("stdout", "")
            ):
                raise ValueError("Native-test results are incomplete or failed")
    else:
        raise ValueError("Unsupported test receipt")
    return {"passed": True}


def build_runtime_plan(preparation, action):
    """Use installed, reviewed source and artifacts; preparation and model start are separate."""
    if preparation.get("schema") != "sparkring-deploy-preparation/v1":
        raise ValueError("Expected a saved deployment preparation document")
    spec = preparation["spec"]
    workspace = _checked_path(spec["workspace"], "workspace")
    hosts = spec["hosts"]
    if [h["rank"] for h in hosts] != list(range(4)):
        raise ValueError("Expected exactly four ordered ranks")
    for host in hosts:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host["host"]):
            raise ValueError("Invalid host")
    if len({h["host"] for h in hosts}) != 4:
        raise ValueError("Expected four distinct hosts")
    source = workspace + "/source"
    launch = workspace + "/launch"
    receipt = source + "/runtime/glm53-spark-mtp3-mesh/image-receipt.json"
    managed = source + "/runtime/glm53-spark-mtp3-mesh"
    installed = (
        "/opt/sparkring/managed-mesh/runtime/glm53-spark-mtp3-mesh/managed_service.py"
    )
    config = "/etc/sparkring/managed-mesh/service.json"
    public = json.loads((PROFILE / "public-image.json").read_text())
    image_id = public["config_image_id"]
    phases = []
    capabilities = preparation.get("lifecycle_capabilities", [])
    if (
        not isinstance(capabilities, list)
        or not all(isinstance(item, str) for item in capabilities)
        or len(set(capabilities)) != len(capabilities)
    ):
        raise ValueError("Lifecycle capabilities must be distinct operation names")
    memory_operations = {"memory-idle", "memory-prepare", "memory-check"}
    memory_supported = memory_operations.intersection(capabilities)
    if memory_supported and memory_supported != memory_operations:
        raise ValueError("Memory preparation requires all three lifecycle capabilities")

    def service(operation, risk="read-only"):
        if operation == "model-quiesce":
            risk = "mutates-host"
            verification = {
                "argv": [
                    "sudo",
                    "-n",
                    "python3",
                    "-c",
                    "import json,sys; from pathlib import Path; root=Path(json.loads(Path(sys.argv[1]).read_text())['state_dir']); status=root/'status.json'; intent=root/'model-intent.json'; assert not status.exists() or (json.loads(intent.read_text())['active'] is False and json.loads(intent.read_text())['generation']==json.loads(status.read_text())['generation'])",
                    config,
                ]
            }
        else:
            verification = {
                "argv": [
                    "sudo",
                    "-n",
                    "python3",
                    installed,
                    "model-stopped",
                    "--config",
                    config,
                ]
            }
        return [
            _action(
                h["host"],
                ["sudo", "-n", "python3", installed, operation, "--config", config],
                risk,
                verification,
            )
            for h in hosts
        ]

    def mesh_up_phases():
        return [
            {"id": "require-stopped-models", "actions": service("model-stopped")},
            {
                "id": "start-mesh-supervisors",
                "actions": [
                    _action(
                        h["host"],
                        ["sudo", "-n", "systemctl", "start", "sparkring-mesh.service"],
                        "mutates-host",
                        {
                            "argv": [
                                "sudo",
                                "-n",
                                "python3",
                                installed,
                                "gate",
                                "--config",
                                config,
                                "--timeout",
                                "60",
                            ]
                        },
                        timeout=90,
                    )
                    for h in hosts
                ],
            },
        ]

    if action == "create":
        from scripts.deploy_suite import require_verified_network

        require_verified_network(preparation)
        phases.append(
            {
                "id": "create-stopped-containers",
                "actions": [
                    _action(
                        h["host"],
                        [
                            "env",
                            "SPARKRING_CREATE_ONLY=1",
                            "bash",
                            launch + "/launch-rank.sh",
                            str(h["rank"]),
                            launch + f"/rank{h['rank']}.env",
                        ],
                        "mutates-host",
                        {
                            "argv": [
                                "docker",
                                "inspect",
                                "--format",
                                "{{.State.Status}}",
                                spec["owner"] + f"-r{h['rank']}",
                            ],
                            "stdout": "created",
                        },
                        checks=[
                            {"argv": ["docker", "ps", "-q"], "stdout": ""},
                            {
                                "argv": [
                                    "docker",
                                    "image",
                                    "inspect",
                                    "--format",
                                    "{{.Id}}",
                                    public["public_reference"],
                                ],
                                "stdout": image_id,
                            },
                        ],
                        timeout=180,
                    )
                    for h in hosts
                ],
            }
        )
    elif action == "install":
        epoch = preparation.get("epoch")
        if not isinstance(epoch, str) or not re.fullmatch(r"[0-9a-f]{32}", epoch):
            raise ValueError("A shared persisted epoch is required before installation")
        phases.append(
            {
                "id": "install-managed-services",
                "actions": [
                    _action(
                        h["host"],
                        [
                            "sudo",
                            "-n",
                            "python3",
                            managed + "/managed_install.py",
                            "--launch",
                            launch,
                            "--image-receipt",
                            receipt,
                            "--rank",
                            str(h["rank"]),
                            "--epoch",
                            epoch,
                            "--key-file",
                            workspace + "/private/health.key",
                            "--apply",
                        ],
                        "mutates-host",
                        {
                            "argv": [
                                "sudo",
                                "-n",
                                "python3",
                                installed,
                                "model-stopped",
                                "--config",
                                config,
                            ]
                        },
                        checks=[
                            {
                                "argv": [
                                    "docker",
                                    "inspect",
                                    "--format",
                                    "{{.State.Running}}",
                                    spec["owner"] + f"-r{h['rank']}",
                                ],
                                "stdout": "false",
                            }
                        ],
                        timeout=180,
                    )
                    for h in hosts
                ],
            }
        )
    elif action == "up":
        phases.extend(mesh_up_phases())
    elif action == "start":
        if memory_supported:
            phases.append(
                {
                    "id": "all-rank-start-stop-barrier",
                    "actions": service("model-stopped"),
                }
            )
            for operation in ("memory-idle", "memory-prepare", "memory-check"):
                phases.append(
                    {
                        "id": operation,
                        "actions": [
                            _action(
                                h["host"],
                                [
                                    "sudo",
                                    "-n",
                                    "python3",
                                    installed,
                                    operation,
                                    "--config",
                                    config,
                                ],
                                "mutates-host"
                                if operation == "memory-prepare"
                                else "read-only",
                                {
                                    "argv": [
                                        "sudo",
                                        "-n",
                                        "python3",
                                        installed,
                                        "memory-check",
                                        "--config",
                                        config,
                                    ]
                                }
                                if operation == "memory-prepare"
                                else {},
                                timeout=180,
                            )
                            for h in hosts
                        ],
                    }
                )
        phases.append(
            {
                "id": "require-mesh-ready",
                "actions": [
                    _action(
                        h["host"],
                        [
                            "sudo",
                            "-n",
                            "python3",
                            installed,
                            "gate",
                            "--config",
                            config,
                            "--timeout",
                            "60",
                        ],
                        "read-only",
                        {},
                    )
                    for h in hosts
                ],
            }
        )
        phases.append(
            {
                "id": "start-managed-models",
                "actions": [
                    _action(
                        h["host"],
                        [
                            "sudo",
                            "-n",
                            "systemctl",
                            "start",
                            "sparkring-mesh-model.service",
                        ],
                        "starts-model",
                        {
                            "argv": [
                                "docker",
                                "inspect",
                                "--format",
                                "{{.State.Running}}",
                                spec["owner"] + f"-r{h['rank']}",
                            ],
                            "stdout": "true",
                        },
                        timeout=90,
                    )
                    for h in hosts
                ],
            }
        )
    elif action in ("stop", "down", "recover"):
        phases.append({"id": "quiesce-models", "actions": service("model-quiesce")})
        phases.append(
            {
                "id": "stop-managed-models",
                "actions": [
                    _action(
                        h["host"],
                        [
                            "sudo",
                            "-n",
                            "systemctl",
                            "stop",
                            "sparkring-mesh-model.service",
                        ],
                        "stops-model",
                        {
                            "argv": [
                                "sudo",
                                "-n",
                                "python3",
                                "-c",
                                "import subprocess; p=subprocess.run(['systemctl','show','sparkring-mesh-model.service','--property=LoadState,ActiveState'],check=True,capture_output=True,text=True); fields=dict(line.split('=',1) for line in p.stdout.splitlines()); assert fields.get('LoadState')=='loaded' and fields.get('ActiveState') in ('inactive','failed')",
                            ]
                        },
                        timeout=120,
                    )
                    for h in hosts
                ],
            }
        )
        phases.append(
            {
                "id": "stop-pinned-containers",
                "actions": service("stop-model", "stops-model"),
            }
        )
        phases.append(
            {"id": "all-rank-stop-barrier", "actions": service("model-stopped")}
        )
        if action in ("down", "recover"):
            phases.append(
                {
                    "id": "stop-mesh",
                    "actions": [
                        _action(
                            h["host"],
                            [
                                "sudo",
                                "-n",
                                "systemctl",
                                "stop",
                                "sparkring-mesh.service",
                            ],
                            "mutates-host",
                            {
                                "argv": [
                                    "sudo",
                                    "-n",
                                    "python3",
                                    installed,
                                    "model-stopped",
                                    "--config",
                                    config,
                                ]
                            },
                        )
                        for h in hosts
                    ],
                }
            )
            for operation in ("cleanup", "reset-units"):
                phases.append(
                    {"id": operation, "actions": service(operation, "mutates-host")}
                )
            if action == "recover":
                phases.extend(mesh_up_phases())
    elif action in ("ready", "native-check"):
        controller_launch = preparation.get("controller_launch")
        if not controller_launch or not Path(controller_launch).is_absolute():
            raise ValueError("A staged controller launch copy is required")
        output = str(Path(controller_launch).parent / (action + "-receipt.json"))
        if action == "ready":
            argv = [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0,sys.argv[1]); from scripts.deploy_runtime import probe_readiness; result=probe_readiness(sys.argv[2],sys.argv[3]); raise SystemExit(0 if result.get('ready') is True else 1)",
                str(ROOT),
                controller_launch,
                str(Path(controller_launch).parent / "ready-results"),
            ]
            risk = "read-only"
        else:
            argv = [
                sys.executable,
                str(PROFILE / "qualification/run_native.py"),
                "--launch",
                controller_launch,
                "--image-receipt",
                str(PROFILE / "image-receipt.json"),
                "--output",
                output,
                "--execute-authorized",
            ]
            risk = "hardware-test"
        phases.append(
            {
                "id": action,
                "actions": [
                    _action(
                        "controller",
                        argv,
                        risk,
                        {}
                        if action == "ready"
                        else {
                            "argv": [
                                sys.executable,
                                "-c",
                                "import sys; sys.path.insert(0, sys.argv[1]); from scripts.deploy_runtime import verify_test_receipt; import json; print(json.dumps(verify_test_receipt(sys.argv[2], sys.argv[3])))",
                                str(ROOT),
                                output,
                                action,
                            ],
                            "json": {"passed": True},
                        },
                        timeout=1800,
                    )
                ],
            }
        )
    elif action in ("status", "logs"):
        phases.append(
            {
                "id": action,
                "actions": [
                    _action(
                        h["host"],
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "{{json .State}}",
                            spec["owner"] + f"-r{h['rank']}",
                        ]
                        if action == "status"
                        else [
                            "docker",
                            "logs",
                            "--tail",
                            "100",
                            spec["owner"] + f"-r{h['rank']}",
                        ],
                        "read-only",
                        {},
                    )
                    for h in hosts
                ],
            }
        )
    else:
        raise ValueError("Unsupported managed-runtime action")
    # Every host mutation rechecks the source and launch copy used by the plan.
    # Recovery uses the installed lifecycle controller; it must remain available
    # even when staged source or model inputs have been removed or damaged.
    if action in ("create", "install", "up", "start"):
        if not preparation.get("source"):
            raise ValueError("Stage runtime files before creating this plan")
        for phase in phases:
            for item in phase["actions"]:
                item["checks"].insert(
                    0,
                    {
                        "argv": [
                            "python3",
                            source + "/scripts/deploy_stage.py",
                            "verify-host",
                            "--workspace",
                            workspace,
                            "--preparation-sha256",
                            plan_digest(preparation),
                        ],
                        "json": {"verified": True},
                    },
                )
    return seal_plan(
        {
            "schema": "sparkring-deploy-plan/v1",
            "operation": action,
            "profile": spec["profile"],
            "preparation_sha256": hashlib.sha256(
                json.dumps(preparation, sort_keys=True).encode()
            ).hexdigest(),
            "phases": phases,
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "create",
            "install",
            "up",
            "start",
            "stop",
            "down",
            "recover",
            "status",
            "logs",
            "ready",
            "native-check",
        ),
    )
    parser.add_argument("--preparation", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_runtime_plan(json.loads(args.preparation.read_text()), args.action),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
