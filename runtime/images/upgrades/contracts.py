"""Trusted policy validation and deterministic artifact identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re


class Refused(ValueError):
    """A required safety or evidence condition cannot be established."""


class Uncertain(Refused):
    """An owned external action may still be running or have incomplete effects."""


def require(condition, message):
    if not condition:
        raise Refused(message)


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            Refused("Nonfinite JSON number")
        ),
    )


def relative(value):
    require(
        isinstance(value, str) and value and "\\" not in value and ":" not in value,
        "Expected a relative POSIX path",
    )
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and str(path) == value
        and ".." not in path.parts
        and value != "."
        and not any(ord(c) < 32 for c in value),
        "Unsafe relative path",
    )
    return path


def beneath(root, value, *, exists=True):
    path = Path(root).joinpath(*relative(value).parts)
    require(
        path.resolve().is_relative_to(Path(root).resolve()), "Path escapes its owner"
    )
    require(
        not any(
            p.is_symlink() for p in (path, *path.parents) if p != Path(root).parent
        ),
        "Symlinked policy inputs are not admitted",
    )
    if exists:
        require(path.is_file(), f"Missing policy input: {value}")
    return path


def oid(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def identifier(value, pattern=r"[a-z][a-z0-9-]{0,47}"):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def argv_valid(value):
    return (
        isinstance(value, list)
        and value
        and all(isinstance(x, str) and x and "\x00" not in x for x in value)
    )


def controller_inputs():
    root = Path(__file__).resolve().parents[3]
    files = [
        *Path(__file__).parent.glob("*.py"),
        root / "scripts/image_upgrade.py",
        root / "runtime/images/candidate_image.py",
        root / "runtime/images/Dockerfile.candidate",
    ]
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in sorted(files)}


def load_policy(path):
    path = Path(path).resolve()
    policy = read(path)
    require(isinstance(policy, dict), "Policy must be an object")
    require(
        set(policy)
        <= {
            "schema",
            "name",
            "sources",
            "budgets",
            "gates",
            "build",
            "publish",
            "permissions",
            "platform",
            "required_features",
            "agent",
            "local_sources",
            "foundation",
        },
        "Unknown policy field",
    )
    require(
        policy.get("schema") == "sparkring-image-upgrade/v1",
        "Unsupported upgrade policy schema",
    )
    require(identifier(policy.get("name")), "Invalid policy name")
    require(
        isinstance(policy.get("sources"), list) and policy["sources"],
        "At least one source is required",
    )
    require(isinstance(policy.get("budgets"), dict), "Explicit budgets are required")
    for field in (
        "run_seconds",
        "command_seconds",
        "source_bytes",
        "output_bytes",
        "agent_attempts",
    ):
        value = policy["budgets"].get(field)
        require(
            type(value) is int and value > 0,
            f"Positive integer budget required: {field}",
        )
    require(
        policy["budgets"]["agent_attempts"] <= 5,
        "At most five reconciliation attempts are allowed",
    )
    for field in ("state_bytes", "min_free_bytes"):
        if field in policy["budgets"]:
            require(
                type(policy["budgets"][field]) is int and policy["budgets"][field] > 0,
                "Invalid storage budget",
            )
    require(
        set(policy["budgets"])
        <= {
            "run_seconds",
            "command_seconds",
            "source_bytes",
            "output_bytes",
            "agent_attempts",
            "state_bytes",
            "min_free_bytes",
        },
        "Unknown budget field",
    )
    inputs = {path.name: sha(path.read_bytes())}
    inputs.update(
        {"controller/" + key: value for key, value in controller_inputs().items()}
    )
    seen = set()
    for source in policy["sources"]:
        require(
            isinstance(source, dict)
            and set(source)
            <= {
                "id",
                "repository",
                "ref",
                "baseline",
                "editable_paths",
                "native_paths",
                "protected_paths",
                "excluded_paths",
                "contracts",
                "patch",
                "patch_sha256",
            },
            "Unknown source fields",
        )
        name = source.get("id", "")
        require(
            identifier(name, r"[a-z][a-z0-9_]{0,31}") and name not in seen,
            "Invalid or duplicate source id",
        )
        seen.add(name)
        require(oid(source.get("baseline")), "A full baseline commit is required")
        repository = source.get("repository", "")
        require(
            isinstance(repository, str)
            and (
                re.fullmatch(
                    r"https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?",
                    repository,
                )
                or (
                    policy.get("local_sources") is True
                    and Path(repository).is_absolute()
                )
            ),
            "Repository is not admitted",
        )
        ref = source.get("ref", "")
        require(
            isinstance(ref, str)
            and (oid(ref) or re.fullmatch(r"refs/(heads|tags)/[A-Za-z0-9_./-]+", ref))
            and ".." not in ref,
            "Use a full commit or explicit branch/tag ref",
        )
        for key in ("editable_paths", "native_paths"):
            require(
                isinstance(source.get(key), list) and source[key],
                f"Explicit {key} required",
            )
            for item in source[key]:
                relative(item)
        require(
            isinstance(source.get("contracts"), list) and source["contracts"],
            "Behavior contracts are required",
        )
        for contract in source["contracts"]:
            require(
                isinstance(contract, dict)
                and set(contract) == {"id", "kind", "oracle", "invariant"},
                "Unknown contract fields",
            )
            require(identifier(contract["id"]), "Invalid contract id")
            require(
                isinstance(contract.get("invariant"), str)
                and contract["invariant"].strip(),
                "Contract invariant is required",
            )
            require(
                contract.get("kind") in ("correctness", "optimization"),
                "Unknown contract kind",
            )
            require(contract.get("oracle"), "Contract must name an independent oracle")
        require(
            isinstance(source.get("protected_paths", []), list),
            "Protected paths must be a list",
        )
        for item in source.get("protected_paths", []):
            relative(item)
        require(
            isinstance(source.get("excluded_paths", []), list),
            "Excluded paths must be a list",
        )
        for item in source.get("excluded_paths", []):
            relative(item)
            require(
                item in (".claude", ".agents"),
                "Only agent-tool metadata directories may be excluded from source snapshots",
            )
        if source.get("patch"):
            file = beneath(path.parent, source["patch"])
            require(
                sha(file.read_bytes()) == source.get("patch_sha256"),
                "Patch identity differs",
            )
            inputs[source["patch"]] = sha(file.read_bytes())
    gates = policy.get("gates", [])
    require(isinstance(gates, list), "gates must be a list")
    gate_ids = set()
    for gate in gates:
        require(
            isinstance(gate, dict)
            and set(gate)
            <= {
                "id",
                "stage",
                "executor",
                "image",
                "argv",
                "inputs",
                "metrics",
                "baseline_image",
                "resources",
                "memory_bytes",
                "cpus",
            },
            "Unknown gate fields",
        )
        require(
            identifier(gate.get("id")) and gate["id"] not in gate_ids,
            "Invalid or duplicate gate id",
        )
        gate_ids.add(gate["id"])
        require(
            gate.get("stage") in ("oracle", "image", "hardware"), "Unknown gate stage"
        )
        require(argv_valid(gate.get("argv")), "Gate command must be an argument array")
        require(
            gate.get("executor") in ("docker", "operator"),
            "Gate executor must be docker or operator",
        )
        if gate["executor"] == "operator":
            require(
                gate["stage"] == "hardware",
                "Host execution is reserved for explicitly authorized hardware adapters",
            )
        if gate["executor"] == "docker":
            image = gate.get("image")
            require(
                isinstance(image, str)
                and (
                    re.fullmatch(r"(?:[A-Za-z0-9_./:-]+@)?sha256:[a-f0-9]{64}", image)
                    or (image == "candidate" and gate["stage"] != "oracle")
                ),
                "Docker gate requires an immutable image",
            )
        if gate["stage"] == "hardware":
            require(
                isinstance(gate.get("resources"), list)
                and gate["resources"]
                and all(isinstance(x, str) and x for x in gate["resources"]),
                "Hardware gate must name resources",
            )
        require(
            type(gate.get("memory_bytes", 1)) is int
            and gate.get("memory_bytes", 1) > 0,
            "Invalid gate memory limit",
        )
        require(
            type(gate.get("cpus", 1)) in (int, float)
            and 0 < gate.get("cpus", 1) <= 256,
            "Invalid gate CPU limit",
        )
        require(isinstance(gate.get("inputs", []), list), "Gate inputs must be a list")
        for item in gate.get("inputs", []):
            require(
                isinstance(item, dict) and set(item) == {"path", "sha256"},
                "Invalid oracle input",
            )
            file = beneath(path.parent, item["path"])
            require(
                sha(file.read_bytes()) == item["sha256"],
                "Oracle input identity differs",
            )
            inputs[item["path"]] = item["sha256"]
        require(isinstance(gate.get("metrics", {}), dict), "Metrics must be an object")
        if gate.get("metrics") and gate["stage"] != "oracle":
            reference = gate.get("baseline_image")
            require(
                isinstance(reference, str)
                and re.fullmatch(
                    r"(?:[A-Za-z0-9_./:-]+@)?sha256:[a-f0-9]{64}", reference
                ),
                "Image/hardware performance gates require an immutable baseline image",
            )
        for threshold in gate.get("metrics", {}).values():
            require(
                isinstance(threshold, dict)
                and set(threshold)
                <= {"direction", "max_regression_fraction", "min_samples"},
                "Invalid metric threshold",
            )
            require(
                threshold.get("direction") in ("higher", "lower"),
                "Unknown performance metric direction",
            )
            tolerance = threshold.get("max_regression_fraction")
            require(
                type(tolerance) in (int, float) and 0 <= tolerance < 1,
                "Invalid regression tolerance",
            )
            require(
                type(threshold.get("min_samples", 3)) is int
                and threshold.get("min_samples", 3) >= 3,
                "Performance gates require at least three measurements",
            )
    if policy.get("agent"):
        require(
            isinstance(policy["agent"], dict)
            and set(policy["agent"])
            <= {
                "endpoint",
                "model",
                "key_env",
                "json_mode",
                "allow_plaintext",
                "timeout_seconds",
                "max_tokens",
                "context_bytes",
            },
            "Unknown agent field; use key_env, not an embedded key",
        )
        for field in ("endpoint", "model"):
            require(
                isinstance(policy["agent"].get(field), str) and policy["agent"][field],
                "Agent endpoint and model must be explicit strings",
            )
        for field in ("timeout_seconds", "max_tokens", "context_bytes"):
            require(
                type(policy["agent"].get(field, 1)) is int
                and policy["agent"].get(field, 1) > 0,
                "Invalid agent budget",
            )
    require(
        isinstance(policy.get("permissions", {}), dict)
        and set(policy.get("permissions", {})) <= {"candidate_publication"},
        "Unknown permission field",
    )
    require(
        type(policy.get("permissions", {}).get("candidate_publication", False)) is bool,
        "Publication permission must be boolean",
    )
    for source in policy["sources"]:
        for contract in source["contracts"]:
            require(contract["oracle"] in gate_ids, "Contract oracle is not defined")
            require(
                next(g for g in gates if g["id"] == contract["oracle"])["stage"]
                == "oracle",
                "Contract requires an oracle-stage gate",
            )
    for key in ("build", "publish"):
        step = policy.get(key)
        if step:
            require(
                isinstance(step, dict)
                and set(step)
                <= {"argv", "inputs", "env_names", "supports_native_rebuild"},
                "Unknown action fields",
            )
            require(
                argv_valid(step.get("argv")), f"{key} requires a fixed argument array"
            )
            require(
                isinstance(step.get("env_names", []), list)
                and all(
                    identifier(x, r"[A-Z][A-Z0-9_]*") for x in step.get("env_names", [])
                ),
                "Invalid action environment allowlist",
            )
            for item in step.get("inputs", []):
                file = beneath(path.parent, item["path"])
                require(
                    sha(file.read_bytes()) == item["sha256"],
                    f"{key} input hash differs",
                )
                inputs[item["path"]] = item["sha256"]
    policy["_root"] = str(path.parent)
    policy["_path"] = str(path)
    policy["_inputs"] = inputs
    policy["_digest"] = sha(encoded(inputs))
    return policy


def check_policy(policy):
    require(
        load_policy(policy["_path"])["_digest"] == policy["_digest"],
        "Trusted policy or oracle changed during the run",
    )


def allowed(path, prefixes):
    return any(
        path == prefix or path.startswith(prefix.rstrip("/") + "/")
        for prefix in prefixes
    )
