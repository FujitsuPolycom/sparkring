"""Protected gate execution and separately authorized build/hardware actions."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import platform
import socket

from .contracts import Refused, Uncertain, beneath, check_policy, read, require
from .io import command, write_json


def expand(values, context):
    def substitute(match):
        key = match.group(1)
        require(key in context, "Unknown command placeholder: " + key)
        return str(context[key])

    return [re.sub(r"\{([a-z_]+)\}", substitute, value) for value in values]


def lease_valid(path, policy, gate, deadline):
    require(path is not None, "Hardware gate requires an explicit operator lease")
    lease = read(path)
    require(
        lease.get("schema") == "sparkring-upgrade-hardware-lease/v1"
        and lease.get("policy_sha256") == policy["_digest"]
        and lease.get("exclusive") is True,
        "Hardware lease does not authorize this exact policy",
    )
    now = time.time()
    require(active_lease(lease, now), "Hardware lease is not active")
    require(
        gate.get("resources")
        and set(gate["resources"]) <= set(lease.get("resources", [])),
        "Hardware resources are not leased",
    )
    require(
        gate["id"] in lease.get("gate_ids", []),
        "Hardware gate is not authorized by lease",
    )
    return min(deadline, time.monotonic() + lease["expires_at"] - now)


def builder_lease_valid(path, policy):
    require(path is not None, "Execution requires an explicit builder lease")
    value = read(path)
    require(
        value.get("schema") == "sparkring-upgrade-builder-lease/v1"
        and value.get("policy_sha256") == policy["_digest"]
        and value.get("host") == socket.gethostname()
        and value.get("exclusive") is True,
        "Builder lease does not authorize this policy and host",
    )
    now = time.time()
    require(active_lease(value, now), "Builder lease is not active")
    return value


def active_lease(value, now):
    start, end = value.get("not_before", now), value.get("expires_at")
    return (
        all(type(x) in (int, float) and math.isfinite(x) for x in (start, end))
        and start <= now < end
    )


def validate_gate(value, gate, context):
    require(
        isinstance(value, dict) and value.get("schema") == "sparkring-upgrade-gate/v1",
        "Gate receipt schema is invalid",
    )
    require(
        value.get("gate") == gate["id"]
        and value.get("input_sha256") == context["input_sha"]
        and value.get("variant") == context["variant"]
        and value.get("subject_sha256") == context["subject_sha256"],
        "Gate receipt identity differs",
    )
    require(
        value.get("outcome") in ("passed", "failed"),
        "Gate must explicitly pass or fail",
    )
    require(
        type(value.get("assertions")) is int
        and value["assertions"] > 0
        and type(value.get("skipped")) is int
        and value["skipped"] == 0,
        "A zero-test or skipped gate cannot establish acceptance",
    )
    for metric, threshold in gate.get("metrics", {}).items():
        values = value.get("measurements", {}).get(metric, [])
        require(
            isinstance(values, list)
            and len(values) >= threshold.get("min_samples", 3)
            and all(
                type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values
            ),
            "Performance gate lacks finite repeated measurements: " + metric,
        )
    if (
        gate.get("metrics")
        and gate.get("stage") in ("image", "hardware")
        and context["variant"] != "control"
    ):
        require(
            gate.get("baseline_image"), "Performance control image is not specified"
        )
        baseline = validate_gate(
            value.get("baseline"),
            gate,
            {**context, "variant": "control", "subject_sha256": gate["baseline_image"]},
        )
        require(baseline["outcome"] == "passed", "Performance control did not pass")
    return value


def acceptable(value, gate, baseline=None):
    if value["outcome"] != "passed":
        return False
    baseline = baseline or value.get("baseline")
    if (
        gate.get("metrics")
        and gate.get("stage") in ("image", "hardware")
        and not baseline
    ):
        return False
    if baseline:
        for metric, threshold in gate.get("metrics", {}).items():
            before = statistics.median(baseline["measurements"][metric])
            after = statistics.median(value["measurements"][metric])
            tolerance = threshold["max_regression_fraction"]
            if threshold["direction"] == "higher" and after < before * (1 - tolerance):
                return False
            if threshold["direction"] == "lower" and after > before * (1 + tolerance):
                return False
    return True


class Executor:
    def __init__(
        self,
        policy,
        *,
        execute=False,
        build=False,
        hardware_lease=None,
        builder_lease=None,
        publish=False,
    ):
        self.policy = policy
        self.execute = execute
        self.build_enabled = build
        self.hardware_lease = hardware_lease
        self.publish_enabled = publish
        self.builder_lease = builder_lease

    def _run(self, argv, output, deadline, env=None, *, mutating=False):
        check_policy(self.policy)
        lease = builder_lease_valid(self.builder_lease, self.policy)
        deadline = min(deadline, time.monotonic() + lease["expires_at"] - time.time())
        remaining = min(
            self.policy["budgets"]["command_seconds"], deadline - time.monotonic()
        )
        require(remaining > 0, "Run time budget exhausted")
        result = command(
            argv,
            seconds=remaining,
            limit=self.policy["budgets"]["output_bytes"],
            env=env,
        )
        metadata = {k: v for k, v in result.items() if k not in ("stdout", "stderr")}
        output.mkdir(parents=True, exist_ok=True)
        try:
            for key in ("stdout", "stderr"):
                # A gate can write its output mount. Exclusive creation refuses
                # preplanted files and symlinks instead of following them on host.
                with (output / (key + ".log")).open("xb") as stream:
                    stream.write(result[key])
            write_json(output / "execution.json", metadata)
        except OSError as error:
            failure = Uncertain if result["uncertain"] or mutating else Refused
            raise failure(
                "Cannot safely record execution output; inspect owned work"
            ) from error
        if result["uncertain"] or (mutating and result["returncode"]):
            raise Uncertain(
                "Execution outcome uncertain; inspect owned jobs before retrying"
            )
        check_policy(self.policy)
        require(
            result["returncode"] == 0,
            "Gate or action command failed; inspect its retained log",
        )

    def gate(self, gate, source, output, context, deadline):
        context = {**context, "baseline_image": gate.get("baseline_image", "")}
        deadline = min(deadline, time.monotonic() + gate.get("timeout_seconds", 1800))
        require(
            self.execute,
            "Execution is disabled; run with --execute after reviewing the plan",
        )
        output = Path(output)
        output.mkdir(parents=True)
        if gate["stage"] == "hardware":
            deadline = lease_valid(self.hardware_lease, self.policy, gate, deadline)
        image = gate.get("image", "")
        if image == "candidate":
            image = context.get("image_id", "")
        if gate["executor"] == "docker":
            require(
                sys.platform == "linux",
                "Docker gates require a dedicated Linux builder; Windows supports planning and the simulated trial",
            )
            require(
                re.fullmatch(r"(?:[A-Za-z0-9_./:-]+@)?sha256:[a-f0-9]{64}", image),
                "Gate image must be immutable",
            )
            container_name = (
                "sr-upgrade-"
                + context["run_id"].lower()
                + "-"
                + gate["id"]
                + "-"
                + output.name
            )
            write_json(
                output / "owned-resource.json",
                {
                    "container": container_name,
                    "run_id": context["run_id"],
                    "daemon": "unix:///var/run/docker.sock",
                },
            )
            argv = [
                "docker",
                "--host",
                "unix:///var/run/docker.sock",
                "run",
                "--name",
                container_name,
                "--label",
                "sparkring.upgrade.run=" + context["run_id"],
                "--rm",
                "--pull",
                "never",
                "--network",
                "none",
                "--read-only",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "256",
                "--memory",
                str(gate.get("memory_bytes", 4 * 1024**3)),
                "--cpus",
                str(gate.get("cpus", 2)),
                "--tmpfs",
                "/tmp:rw,nosuid,size=1073741824",
                "--mount",
                f"type=bind,src={Path(source).resolve()},dst=/source,readonly",
                "--mount",
                f"type=bind,src={output.resolve()},dst=/out",
            ]
            for item in gate.get("inputs", []):
                path = beneath(self.policy["_root"], item["path"])
                argv += [
                    "--mount",
                    f"type=bind,src={path},dst=/oracles/{item['path']},readonly",
                ]
            if gate["stage"] == "oracle":
                baseline = Path(context["baseline_path"]).resolve()
                argv += ["--mount", f"type=bind,src={baseline},dst=/baseline,readonly"]
            inner = {
                **context,
                "source": "/source",
                "result": "/out/result.json",
                "policy_root": "/oracles",
            }
            for name, value in (
                ("SPARKRING_UPGRADE_INPUT", context["input_sha"]),
                ("SPARKRING_UPGRADE_VARIANT", context["variant"]),
                ("SPARKRING_UPGRADE_SUBJECT", context["subject_sha256"]),
                ("SPARKRING_UPGRADE_BASELINE_IMAGE", context["baseline_image"]),
                ("SPARKRING_UPGRADE_GATE", gate["id"]),
            ):
                argv += ["--env", name + "=" + value]
            if gate["stage"] == "hardware":
                argv += ["--gpus", "all"]
            else:
                argv += ["--runtime", "runc", "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES=",
                         "--env", "OMP_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1"]
            inner_argv = expand(gate["argv"], inner)
            argv += ["--entrypoint", inner_argv[0], image, *inner_argv[1:]]
        else:
            require(
                gate["stage"] == "hardware",
                "Untrusted source tests cannot execute on the controller",
            )
            argv = expand(
                gate["argv"],
                {
                    **context,
                    "source": source,
                    "result": output / "result.json",
                    "lease": self.hardware_lease,
                    "policy_root": self.policy["_root"],
                },
            )
        self._run(argv, output, deadline, mutating=gate["stage"] == "hardware")
        path = output / "result.json"
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size <= self.policy["budgets"]["output_bytes"],
            "Gate receipt missing or oversized",
        )
        return validate_gate(read(path), gate, context)

    def action(self, kind, context, output, deadline):
        step = self.policy.get(kind)
        require(step, f"No trusted {kind} recipe configured")
        require(
            self.execute
            and (self.build_enabled if kind == "build" else self.publish_enabled),
            f"{kind} permission is disabled",
        )
        if kind == "build":
            machine = {"aarch64": "arm64", "x86_64": "amd64"}.get(
                platform.machine(), platform.machine()
            )
            require(
                sys.platform == "linux"
                and self.policy.get("platform") == "linux/" + machine,
                "Build on an explicitly leased Linux host of the target architecture",
            )
        require(
            kind != "publish"
            or self.policy.get("permissions", {}).get("candidate_publication") is True,
            "Policy does not authorize candidate publication",
        )
        output = Path(output)
        output.mkdir(parents=True)
        expanded = {
            **context,
            "result": output / "result.json",
            "policy_root": self.policy["_root"],
        }
        argv = expand(step["argv"], expanded)
        if argv[0] == "python":
            argv[0] = sys.executable
        extra = {
            key: os.environ[key]
            for key in step.get("env_names", [])
            if key in os.environ
        }
        self._run(argv, output, deadline, extra, mutating=True)
        path = output / "result.json"
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size <= self.policy["budgets"]["output_bytes"],
            "Action receipt missing or oversized",
        )
        result = read(path)
        require(
            result.get("schema") == f"sparkring-upgrade-{kind}/v1"
            and result.get("input_sha256") == context["input_sha"],
            "Action receipt identity differs",
        )
        if kind == "build":
            require(
                re.fullmatch(r"sha256:[a-f0-9]{64}", result.get("image_id", "")),
                "Build must identify immutable image bytes",
            )
            require(
                result.get("installed_verified") is True
                and result.get("platform") == self.policy["platform"],
                "Built image has not passed installed/platform verification",
            )
            require(
                set(self.policy.get("required_features", []))
                <= set(result.get("features", [])),
                "Built image omits required capabilities",
            )
        else:
            require(
                result.get("channel") == "candidate"
                and result.get("image_id") == context["image_id"],
                "Publication must retain the candidate image identity",
            )
        return result
