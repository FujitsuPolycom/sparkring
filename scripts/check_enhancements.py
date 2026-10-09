"""Check one deployment against SparkRing's catalog of speed enhancements.

The catalog, ``performance/enhancements.json`` (schema
``sparkring-enhancements/v1``), lists each enhancement once: what it does, the
models and tensor-parallel (TP) and decode-context-parallel (DCP) sizes it
applies to, the exact settings that enable it, the images that carry it, its
status and its measured gain with the conditions and evidence of each
measurement. ``performance/enhancements.md`` is its generated per-model view
(``scripts/generate_enhancements.py``).

This program reads one deployment: a profile ID, optionally one of its
checkpoints, and optionally the configuration that actually ran, given as a
launch record (the serving-configuration shape: ``environment`` and
``vllm_args``), a serving A/B runner campaign's ``plan.json``
(``serving-ab-plan/v1``, one arm and rank), or a container environment dump
and argument dump. For every catalog entry that applies to the deployment's
model and TP/DCP size it reports one status:

- ``enabled``: the configuration turns it on, or the image does for an entry
  without a switch;
- ``missing``: the image carries it and the configuration leaves it off;
- ``refused-here``: the model code or the installer refuses it at this size;
- ``needs-port``: the deployment's image does not carry it (no image does,
  or only another image does).

Missing and needs-port entries are sorted by measured gain, largest first.
Entries whose gain is unmeasured follow, and an entry whose ``superseded_by``
entry is enabled comes last. Entries listed for other sizes of the model are
named at the end. Host settings and other switches that no configuration
records are reported enabled only when ``--enabled ID`` names them.

Safety class: OFFLINE. Only repository files and the files named on the
command line are read; no host is contacted.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.common import profiles as profile_catalog  # noqa: E402

CATALOG = "performance/enhancements.json"
SCHEMA = "sparkring-enhancements/v1"
MODEL_NAMES = "profiles/model-names.json"
STATUS_ORDER = ("missing", "needs-port", "refused-here", "enabled")
TRANSPORTS = ("prepared", "sircl", "libsircl")
# Topologies whose ranks reach each other through relays: only SIRCL ring sessions run them.
RELAYED = profile_catalog.RELAYED
# Variables that SIRCL's adapter and serve launcher set on every container they run on SIRCL ring sessions. A
# profile's own SIRCL session settings (schedules, link sizes, the one-shot limit) take effect only where its
# deployment runs SIRCL, so they do not show which transport a deployment uses.
SIRCL_MARKERS = ("SIRCL_MODE", "SIRCL_FABRIC", "SIRCL_RANK_POSITIONS")

TOP_KEYS = {"schema", "purpose", "related", "categories", "sides", "statuses", "images", "checks",
            "enhancements", "not_adopted"}
ENTRY_KEYS = {"id", "title", "description", "category", "sides", "models", "parallelism", "enable", "detect",
              "provided_by", "status", "gain", "measurements", "evidence", "profiles"}
ENTRY_OPTIONAL = {"requires", "superseded_by", "trade_off", "transports"}
ENABLE_KEYS = {"environment", "arguments", "speculative", "checkpoint", "plugin", "host", "launcher", "note"}
MEASUREMENT_KEYS = {"image", "tp", "dcp", "streams", "context", "result", "evidence"}
MEASUREMENT_OPTIONAL = {"checkpoint"}
IMAGE_KEYS = {"label", "release", "image_id", "short_id", "parent", "transports", "description", "evidence"}
NOT_ADOPTED_KEYS = {"id", "title", "models", "setting", "result", "evidence"}
CHECK_KEYS = {"id", "when_any", "unless_any", "message"}
SUBJECTS = ("env", "arg", "model", "transport", "image")
OPERATORS = ("equals", "in", "at_least", "at_most", "has_item", "lacks_item", "contains", "present", "absent")
ID = re.compile(r"[a-z0-9][a-z0-9.-]{1,79}")
PRIVATE_RECORD = re.compile(r"private record \d{4}-\d{2}(-\d{2})?(\.\.\d{2})?: \S.*")
SOURCE_REFERENCE = re.compile(r"source: \S+ \S.*")
# Shapes of private data that the catalog must never hold: local drive and home paths, mounted Windows
# drives and private IPv4 addresses.
PRIVATE_SHAPES = re.compile(r"[A-Za-z]:\\|AppData|/mnt/[a-z]/|/home/[a-z]|/Users/[A-Za-z]|scratchpad|"
                            r"(?<![0-9.])(?:10\.[0-9]{1,3}|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))"
                            r"\.[0-9]{1,3}\.[0-9]{1,3}(?![0-9])")


class CatalogError(ValueError):
    pass


def read_json(path: Path) -> Any:
    return profile_catalog.read_json(path)


def load_catalog(root: Path = ROOT) -> dict:
    return read_json(Path(root) / CATALOG)


def model_names(root: Path = ROOT) -> dict[str, str]:
    """``{repository: model name}`` of ``profiles/model-names.json``."""
    return read_json(Path(root) / MODEL_NAMES)["models"]


# -- the deployment ---------------------------------------------------------------------------------


@dataclasses.dataclass
class Deployment:
    """The settings one check reads. ``source`` is ``profile`` when the profile's own configuration is read,
    which the installer renders with its default SIRCL tuning row; otherwise the configuration that ran."""

    profile: str
    checkpoint: str | None
    model_repository: str
    model_revision: str
    model_name: str
    tp: int
    dcp: int
    topology: str | None
    image: str | None
    transport: str
    environment: dict[str, str]
    arguments: dict[str, Any]
    source: str
    enabled: frozenset[str] = frozenset()
    notes: tuple[str, ...] = ()


def parse_arguments(tokens: list[str]) -> dict[str, Any]:
    """``{flag: value}`` of a vLLM argument list; a flag without a value maps to True.

    A token after a flag is its value unless it starts with ``--``; ``--flag=value`` is accepted. Tokens
    that are neither a flag nor a flag's value (a ``serve`` subcommand, a model path) are skipped.
    """
    result: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = str(tokens[index])
        if token.startswith("--"):
            flag, equals, value = token.partition("=")
            if equals:
                result[flag] = value
            elif index + 1 < len(tokens) and not str(tokens[index + 1]).startswith("--"):
                result[flag] = str(tokens[index + 1])
                index += 1
            else:
                result[flag] = True
        index += 1
    return result


def integer_argument(arguments: dict[str, Any], flag: str, default: int) -> int:
    value = arguments.get(flag)
    try:
        return int(value) if value not in (None, True) else default
    except ValueError:
        raise CatalogError(f"{flag} {value!r} is not an integer") from None


def profile_configuration(profile_id: str, root: Path = ROOT) -> tuple[dict, dict]:
    """``(profile document, serving configuration)`` of a catalog profile."""
    paths = profile_catalog.catalog(root)
    if profile_id not in paths:
        raise CatalogError(profile_catalog.replacement_message(profile_id) or f"Unknown profile: {profile_id}")
    document = read_json(paths[profile_id])
    source = document["configuration"]
    if source.get("format") != "serving-profile":
        raise CatalogError(f"{profile_id} has a {source.get('format')} configuration; the checker reads "
                           "serving-profile configurations (environment and vllm_args)")
    config = read_json(profile_catalog.local_path(source["path"], root))
    if not isinstance(config.get("environment"), dict) or not isinstance(config.get("vllm_args"), list):
        raise CatalogError(f"{source['path']} has no environment and vllm_args")
    return document, config


def with_checkpoint(config: dict, checkpoint: str | None) -> dict:
    """The configuration with ``checkpoint``'s pinned settings applied (``qwen_flash_next.checkpoint_settings``)."""
    if checkpoint is None:
        return config
    from runtime.common import qwen_flash_next
    try:
        return qwen_flash_next.checkpoint_settings(config, checkpoint)
    except ValueError as error:
        raise CatalogError(str(error)) from None


def image_for_release(catalog: dict, release: str | None) -> str | None:
    for key, image in catalog["images"].items():
        if release and image.get("release") == release:
            return key
    return None


def image_for_id(catalog: dict, image_id: str | None) -> str | None:
    if not image_id:
        return None
    digest = str(image_id).removeprefix("sha256:")
    for key, image in catalog["images"].items():
        if (image.get("image_id") or "").removeprefix("sha256:") == digest or (
                len(digest) >= 12 and image["short_id"] and digest.startswith(image["short_id"])):
            return key
    return None


def infer_transport(environment: dict[str, str], topology: str | None, image: dict | None) -> tuple[str, str]:
    """``(transport, reason)`` of a deployment whose transport was not named."""
    plugins = [part.strip() for part in environment.get("VLLM_PLUGINS", "").split(",")]
    if "libsircl" in environment.get("VLLM_NCCL_SO_PATH", "") or "libsircl" in plugins:
        return "libsircl", "vLLM's NCCL library is libsircl"
    if "sircl" in plugins or any(key in environment for key in SIRCL_MARKERS):
        return "sircl", "the environment runs SIRCL's adapter"
    if topology in RELAYED:
        return "sircl", f"only SIRCL ring sessions run topology {topology}"
    if image and "sircl" in image["transports"]:
        return "sircl", "the installer runs SIRCL wherever the image and the fabric carry it"
    return "prepared", "the image carries no SIRCL layer" if image else "no image identified"


def read_environment_dump(path: Path) -> dict[str, str]:
    """A container environment: ``KEY=VALUE`` lines (``env``), a JSON list of them (``docker inspect``'s
    ``Config.Env``) or a JSON object."""
    text = Path(path).read_text(encoding="utf-8-sig")
    stripped = text.strip()
    if stripped.startswith(("[", "{")):
        value = json.loads(stripped)
        if isinstance(value, dict):
            return {str(key): str(item) for key, item in value.items()}
        lines = [str(item) for item in value]
    else:
        lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    result = {}
    for line in lines:
        key, equals, value = line.partition("=")
        if not equals:
            raise CatalogError(f"{path}: {line!r} is not KEY=VALUE")
        result[key.strip()] = value
    return result


def read_arguments_dump(path: Path) -> list[str]:
    """A container's arguments: a JSON list (``docker inspect``'s ``Config.Cmd`` or ``Args``) or one shell
    command line."""
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if text.startswith("["):
        return [str(item) for item in json.loads(text)]
    return shlex.split(text.replace("\\\n", " "))


def plan_settings(path: Path, arm: str | None, rank: int) -> tuple[dict[str, str], list[str], dict]:
    """``(environment, arguments, plan)`` of one arm's rank command in a serving A/B runner plan."""
    from performance.harnesses.serving_ab import spec
    plan = read_json(Path(path))
    if plan.get("schema") != "serving-ab-plan/v1":
        raise CatalogError(f"{path} is not a serving-ab-plan/v1 plan")
    arm = arm or plan["arms"][0]
    if arm not in plan["commands"]:
        raise CatalogError(f"{path} has no arm {arm}; its arms are {', '.join(plan['commands'])}")
    tokens = plan["commands"][arm][rank]
    return spec.environment(tokens), list(tokens[spec.image_index(tokens) + 1:]), plan


def deployment(profile_id: str, *, catalog: dict, checkpoint: str | None = None, image: str | None = None,
               launch_record: Path | None = None, plan: Path | None = None, arm: str | None = None,
               rank: int = 0, environment_dump: Path | None = None, arguments_dump: Path | None = None,
               tp: int | None = None, dcp: int | None = None, transport: str | None = None,
               enabled: tuple[str, ...] = (), root: Path = ROOT) -> Deployment:
    """The deployment that one check reads; the profile supplies whatever the other inputs leave out."""
    document, config = profile_configuration(profile_id, root)
    default_name = model_names(root).get(config["model"]["repository"])
    config = with_checkpoint(config, checkpoint)
    environment = dict(config["environment"])
    tokens = list(config["vllm_args"])
    model = dict(config["model"])
    source, notes, plan_image = "profile", [], None
    if sum(item is not None for item in (launch_record, plan, environment_dump or arguments_dump)) > 1:
        raise CatalogError("Give one of a launch record, a plan or a container dump")
    if launch_record is not None:
        record = read_json(Path(launch_record))
        if not isinstance(record.get("environment"), dict) or not isinstance(record.get("vllm_args"), list):
            raise CatalogError(f"{launch_record}: a launch record has environment (an object) and vllm_args (a list)")
        environment = {str(key): str(value) for key, value in record["environment"].items()}
        tokens = [str(token) for token in record["vllm_args"]]
        model = record.get("model") or model
        source = "record"
    elif plan is not None:
        environment, tokens, document_plan = plan_settings(Path(plan), arm, rank)
        if document_plan.get("profile") != profile_id:
            notes.append(f"the plan's profile is {document_plan.get('profile')}, not {profile_id}")
        model = document_plan.get("model") or model
        plan_image = document_plan.get("image")
        source = "plan"
    elif environment_dump is not None or arguments_dump is not None:
        if environment_dump is not None:
            environment = read_environment_dump(Path(environment_dump))
        if arguments_dump is not None:
            tokens = read_arguments_dump(Path(arguments_dump))
        source = "dump"
    arguments = parse_arguments(tokens)
    names = model_names(root)
    model_name = names.get(model["repository"]) or default_name
    if model_name is None:
        raise CatalogError(f"{model['repository']} is not in {MODEL_NAMES}")
    if image is not None and image not in catalog["images"]:
        raise CatalogError(f"Unknown image {image}; the catalog's images are {', '.join(catalog['images'])}")
    image = image or image_for_id(catalog, plan_image) or image_for_release(catalog, document.get("release"))
    if image is None:
        notes.append("no catalog image matches the deployment; image availability is not checked")
    if transport is None:
        transport, reason = infer_transport(environment, config.get("topology"),
                                            catalog["images"].get(image) if image else None)
        notes.append(f"transport {transport}: {reason} (--transport names it)")
    elif transport not in TRANSPORTS:
        raise CatalogError(f"--transport takes {', '.join(TRANSPORTS)}")
    unknown = sorted(set(enabled) - {entry["id"] for entry in catalog["enhancements"]})
    if unknown:
        raise CatalogError(f"--enabled names unknown entries: {', '.join(unknown)}")
    return Deployment(
        profile=profile_id, checkpoint=checkpoint, model_repository=model["repository"],
        model_revision=str(model.get("revision") or ""), model_name=model_name,
        tp=tp if tp is not None else integer_argument(arguments, "--tensor-parallel-size", 1),
        dcp=dcp if dcp is not None else integer_argument(arguments, "--decode-context-parallel-size", 1),
        topology=config.get("topology"), image=image, transport=transport, environment=environment,
        arguments=arguments, source=source, enabled=frozenset(enabled), notes=tuple(notes))


# -- conditions -------------------------------------------------------------------------------------


def _json_path(value: Any, path: str) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return str(actual).lower() == str(expected).lower()
    return str(actual) == str(expected)


def _items(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare(actual: Any, operator: str, operand: Any) -> bool:
    if operator == "present":
        return actual is not None
    if operator == "absent":
        return actual is None
    if operator == "lacks_item":
        return actual is None or str(operand) not in _items(actual)
    if actual is None:
        return False
    if operator == "equals":
        return _same(actual, operand)
    if operator == "in":
        return any(_same(actual, item) for item in operand)
    if operator in ("at_least", "at_most"):
        number = _number(actual)
        if number is None:
            return False
        return number >= float(operand) if operator == "at_least" else number <= float(operand)
    if operator == "has_item":
        return str(operand) in _items(actual)
    if operator == "contains":
        return str(operand) in str(actual)
    raise CatalogError(f"Unknown operator {operator}")


def condition_operator(condition: dict) -> tuple[str, Any]:
    found = [key for key in OPERATORS if key in condition]
    if len(found) != 1:
        raise CatalogError(f"A condition takes one operator of {', '.join(OPERATORS)}: {condition}")
    return found[0], condition[found[0]]


def holds(condition: dict, view: Deployment) -> bool:
    """Whether one detection or check condition holds for the deployment."""
    if "transport" in condition:
        return view.transport == condition["transport"]
    if "image" in condition:
        return view.image in condition["image"]
    operator, operand = condition_operator(condition)
    if "env" in condition:
        actual = view.environment.get(condition["env"])
        if actual is None and "unset_means" in condition:
            actual = condition["unset_means"]
        if actual is None and "installer_unset_means" in condition and view.source == "profile":
            actual = condition["installer_unset_means"]
    elif "arg" in condition:
        actual = view.arguments.get(condition["arg"])
        if actual is None and "unset_means" in condition:
            actual = condition["unset_means"]
        if "json" in condition and actual is not None:
            actual = _json_path(actual, condition["json"])
    elif "model" in condition:
        actual = view.model_repository
    else:
        raise CatalogError(f"A condition names one subject of {', '.join(SUBJECTS)}: {condition}")
    return compare(actual, operator, operand)


# -- evaluation -------------------------------------------------------------------------------------


def _size_in(value: Any, size: int) -> bool:
    if value == "any":
        return True
    return size in (value if isinstance(value, list) else [value])


def size_rule(rules: list[dict], tp: int, dcp: int) -> dict | None:
    return next((rule for rule in rules if _size_in(rule.get("tp", "any"), tp)
                 and _size_in(rule.get("dcp", "any"), dcp)), None)


def checkpoint_matches(pattern: str, repository: str, revision: str) -> bool:
    name, _, prefix = pattern.partition("@")
    return name == repository and revision.startswith(prefix)


def model_spec(entry: dict, view: Deployment) -> dict | None:
    """The entry's model record that matches the deployment: a checkpoint-specific record before a general
    one of the same model."""
    records = sorted(entry["models"], key=lambda record: not record.get("checkpoints"))
    for record in records:
        if record["name"] != view.model_name:
            continue
        patterns = record.get("checkpoints")
        if not patterns or any(checkpoint_matches(pattern, view.model_repository, view.model_revision)
                               for pattern in patterns):
            return record
    return None


def ancestry(catalog: dict, key: str | None) -> list[str]:
    chain = []
    while key and key not in chain:
        chain.append(key)
        key = catalog["images"][key].get("parent")
    return chain


def provided(entry: dict, catalog: dict, image: str | None) -> bool | None:
    """Whether the deployment's image carries the entry; None when the image is not identified."""
    source = entry["provided_by"]
    if source.get("host") or "any" in source.get("images", ()):
        return True
    if not source.get("images"):
        return False
    if image is None:
        return None
    return any(key in source["images"] for key in ancestry(catalog, image))


def detected(entry: dict, view: Deployment) -> bool:
    if entry["id"] in view.enabled:
        return True
    conditions = entry["detect"]
    return conditions is not None and all(holds(condition, view) for condition in conditions)


def gain_of(entry: dict, record: dict, tp: int) -> tuple[float | None, str | None]:
    gain = record.get("gain") or entry.get("gain")
    if not gain:
        return None, None
    percent = (gain.get("by_tp") or {}).get(str(tp), gain.get("percent"))
    return percent, gain.get("summary")


def applies_to_transport(entry: dict, view: Deployment) -> bool:
    return view.transport in entry.get("transports", TRANSPORTS)


def enable_text(entry: dict) -> str:
    """One line of the settings that enable an entry."""
    enable = entry["enable"]
    parts = [f"{key}={value}" for key, value in enable.get("environment", {}).items()]
    for flag, value in enable.get("arguments", {}).items():
        parts.append(flag if value is True else f"{flag} {value if isinstance(value, str) else json.dumps(value)}")
    if enable.get("speculative"):
        parts.append("--speculative-config keys " + json.dumps(enable["speculative"], separators=(",", ":")))
    for key in ("checkpoint", "plugin", "launcher", "host", "note"):
        if enable.get(key):
            parts.append(f"{key}: {enable[key]}" if key in ("plugin", "host") else enable[key])
    return "; ".join(parts)


def check_warnings(catalog: dict, view: Deployment) -> list[str]:
    """``<id>: <message>`` of every catalog check (``checks``) whose conditions hold for ``view``."""
    return [f"{check['id']}: {check['message']}" for check in catalog["checks"]
            if any(holds(condition, view) for condition in check["when_any"])
            and not any(holds(condition, view) for condition in check["unless_any"])]


def setting_warnings(environment: dict[str, str], tokens: list[str], *, catalog: dict | None = None,
                     root: Path = ROOT) -> list[str]:
    """The catalog checks' warnings for one container's environment and vLLM arguments.

    The installer's plan and the serving A/B runner's plan call it with the rank-0 container they render, so a
    configuration that a check describes as slow (quantized dense linears without --linear-backend) is reported
    before anything starts. The container's settings are read as they are, without the installer's defaults."""
    catalog = catalog if catalog is not None else load_catalog(root)
    view = Deployment(profile="", checkpoint=None, model_repository="", model_revision="", model_name="", tp=1,
                      dcp=1, topology=None, image=None, transport="", environment=dict(environment),
                      arguments=parse_arguments([str(token) for token in tokens]), source="record")
    return check_warnings(catalog, view)


def evaluate(catalog: dict, view: Deployment) -> dict:
    """The report of one deployment: every applicable entry with its status, the warnings of the catalog's
    checks and the entries listed only for other sizes of the model."""
    entries = {entry["id"]: entry for entry in catalog["enhancements"]}
    applicable = {}
    for entry in catalog["enhancements"]:
        record = model_spec(entry, view)
        if record is not None and applies_to_transport(entry, view):
            applicable[entry["id"]] = record
    on = {key: detected(entries[key], view) for key in applicable}
    rows, other_sizes = [], []
    for key, record in applicable.items():
        entry = entries[key]
        parallelism = entry["parallelism"]
        refused = size_rule(parallelism.get("refused", []), view.tp, view.dcp)
        supported = size_rule(parallelism["supported"], view.tp, view.dcp)
        carried = provided(entry, catalog, view.image)
        notes = []
        if refused is not None:
            status = "refused-here"
            notes.append(f"{refused['by']}: {refused['text']}")
            if on[key]:
                notes.append("this configuration enables it, so startup fails")
        elif supported is None:
            other_sizes.append(key)
            continue
        elif carried is False:
            status = "needs-port"
            source = entry["provided_by"]
            notes.append(source.get("port") or "carried by " + ", ".join(
                catalog["images"][image]["label"] for image in source["images"]))
            if on[key] and entry["detect"]:
                notes.append("this configuration sets it, but the image does not carry the code")
        elif on[key]:
            status = "enabled"
        else:
            status = "missing"
        if carried is None:
            notes.append("image not identified")
        if status in ("missing", "needs-port", "enabled"):
            for required in entry.get("requires", ()):
                if required in on and not on[required]:
                    notes.append(f"requires {required}, which is off")
        superseded = entry.get("superseded_by")
        superseded_on = bool(superseded and on.get(superseded))
        if superseded_on and status in ("missing", "needs-port"):
            notes.append(f"{superseded} is enabled and takes its place for eligible batches")
        if entry.get("trade_off") and status in ("missing", "needs-port"):
            notes.append("trade-off: " + entry["trade_off"])
        percent, summary = gain_of(entry, record, view.tp)
        rows.append({"id": key, "title": entry["title"], "status": status,
                     "entry_status": record.get("status") or entry["status"], "category": entry["category"],
                     "sides": entry["sides"], "gain_percent": percent, "gain": summary,
                     "enable": enable_text(entry), "notes": notes, "superseded": superseded_on})

    def order(row: dict) -> tuple:
        ranked = row["status"] in ("missing", "needs-port")
        percent = row["gain_percent"]
        return (STATUS_ORDER.index(row["status"]), ranked and row["superseded"],
                ranked and percent is None, -(percent or 0) if ranked else 0, row["id"])

    rows.sort(key=order)
    warnings = check_warnings(catalog, view)
    for row in rows:
        if row["status"] == "refused-here" and len(row["notes"]) > 1:
            warnings.append(f"{row['id']} is enabled but refused at TP{view.tp}/DCP{view.dcp}")
    return {"schema": "sparkring-enhancement-check/v1", "profile": view.profile, "checkpoint": view.checkpoint,
            "model": {"name": view.model_name, "repository": view.model_repository,
                      "revision": view.model_revision},
            "tp": view.tp, "dcp": view.dcp, "image": view.image, "transport": view.transport,
            "source": view.source, "notes": list(view.notes), "entries": rows, "warnings": warnings,
            "other_sizes": sorted(other_sizes)}


def render(report: dict, catalog: dict) -> str:
    image = catalog["images"].get(report["image"]) if report["image"] else None
    revision = report["model"]["revision"][:12]
    lines = [f"{report['profile']}" + (f" checkpoint {report['checkpoint']}" if report["checkpoint"] else "")
             + f": {report['model']['name']} ({report['model']['repository']}"
             + (f"@{revision}" if revision else "") + f") at TP{report['tp']}/DCP{report['dcp']}",
             "image: " + (f"{report['image']} ({image['short_id'] or image['label']})" if image else "not identified")
             + f"; transport: {report['transport']}; settings read from the {report['source']}"]
    lines += [f"note: {note}" for note in report["notes"]]
    counts = {status: sum(row["status"] == status for row in report["entries"]) for status in STATUS_ORDER}
    lines.append(", ".join(f"{counts[status]} {status}" for status in STATUS_ORDER))
    lines.append("")
    lines.append(f"{'status':<13} {'gain':>7}  entry")
    for row in report["entries"]:
        percent = row["gain_percent"]
        gain = "-" if percent is None else f"{percent:+.1f}%"
        lines.append(f"{row['status']:<13} {gain:>7}  {row['id']} ({row['entry_status']}; "
                     f"{', '.join(row['sides'])}): {row['title']}")
        if row["status"] in ("missing", "needs-port"):
            if row["gain"]:
                lines.append(f"{'':23}measured: {row['gain']}")
            lines.append(f"{'':23}enable: {row['enable']}")
        for note in row["notes"]:
            lines.append(f"{'':23}{note}")
    if report["warnings"]:
        lines += ["", "warnings:"] + [f"- {warning}" for warning in report["warnings"]]
    if report["other_sizes"]:
        lines += ["", f"listed for other sizes of {report['model']['name']}: {', '.join(report['other_sizes'])}"]
    return "\n".join(lines)


# -- validation -------------------------------------------------------------------------------------


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _evidence_problem(reference: Any, root: Path) -> str | None:
    if not isinstance(reference, str) or not reference:
        return "an evidence reference is a nonempty string"
    if PRIVATE_RECORD.fullmatch(reference) or SOURCE_REFERENCE.fullmatch(reference):
        return None
    path = reference.split("#", 1)[0]
    if "\\" in path or path.startswith("/") or not (root / path).is_file():
        return (f"evidence {reference!r} is neither a repository file, 'private record DATE: label' nor "
                "'source: PACKAGE@COMMIT PATH'")
    return None


def _condition_problem(condition: Any, images: dict) -> str | None:
    if not isinstance(condition, dict):
        return f"condition {condition!r} is not an object"
    subjects = [key for key in SUBJECTS if key in condition]
    if len(subjects) != 1:
        return f"condition {condition} names one subject of {', '.join(SUBJECTS)}"
    subject = subjects[0]
    if subject == "transport":
        return None if set(condition) == {"transport"} and condition["transport"] in TRANSPORTS else \
            f"condition {condition}: transport takes one of {', '.join(TRANSPORTS)}"
    if subject == "image":
        return None if set(condition) == {"image"} and isinstance(condition["image"], list) and set(
            condition["image"]) <= set(images) else f"condition {condition}: image takes catalog image keys"
    allowed = {subject, "unset_means"} | ({"installer_unset_means"} if subject == "env" else set()) | (
        {"json"} if subject == "arg" else set())
    operators = [key for key in OPERATORS if key in condition]
    if len(operators) != 1 or not set(condition) - set(operators) <= allowed:
        return f"condition {condition}: one operator of {', '.join(OPERATORS)} and only {', '.join(sorted(allowed))}"
    operator = operators[0]
    if operator in ("present", "absent") and condition[operator] is not True:
        return f"condition {condition}: {operator} takes true"
    if operator == "in" and not isinstance(condition["in"], list):
        return f"condition {condition}: in takes a list"
    if subject == "model" and condition["model"] != "repository":
        return f"condition {condition}: model takes 'repository'"
    return None


def _size_problem(rule: Any, refused: bool) -> str | None:
    keys = {"tp", "dcp"} | ({"by", "text"} if refused else set())
    if not isinstance(rule, dict) or not {"tp", "dcp"} <= set(rule) <= keys or (refused and set(rule) != keys):
        return f"size rule {rule!r} takes tp and dcp" + (", by and text" if refused else "")
    for key in ("tp", "dcp"):
        value = rule[key]
        if value != "any" and not (isinstance(value, list) and value and all(
                isinstance(item, int) and item > 0 for item in value)):
            return f"size rule {rule!r}: {key} is 'any' or a list of sizes"
    return None


def validate(catalog: dict, root: Path = ROOT, *, profiles: bool = True) -> list[str]:
    """Problems of the catalog document; with ``profiles``, also every entry whose ``profiles`` list differs
    from the profiles whose configurations turn it on."""
    problems: list[str] = []
    if catalog.get("schema") != SCHEMA or set(catalog) != TOP_KEYS:
        return [f"{CATALOG} has schema {SCHEMA} and exactly the keys {', '.join(sorted(TOP_KEYS))}"]
    for text in _strings(catalog):
        if PRIVATE_SHAPES.search(text):
            problems.append(f"a catalog string holds a private path or address: {text[:60]!r}")
    for key, path in catalog["related"].items():
        if not (root / path).is_file():
            problems.append(f"related {key}: {path} is not a repository file")
    names = set(model_names(root).values())
    images = catalog["images"]
    for key, image in images.items():
        if set(image) != IMAGE_KEYS:
            problems.append(f"image {key} has exactly the keys {', '.join(sorted(IMAGE_KEYS))}")
            continue
        if image["release"] is not None and not (root / image["release"]).is_file():
            problems.append(f"image {key}: release {image['release']} is not a repository file")
        if image["image_id"] is not None and not re.fullmatch(r"sha256:[0-9a-f]{64}", image["image_id"]):
            problems.append(f"image {key}: image_id is sha256:<64 hex> or null")
        if (image["short_id"] is None) != (image["image_id"] is None) or (image["short_id"] is not None and (
                not re.fullmatch(r"[0-9a-f]{12}", image["short_id"])
                or not image["image_id"].startswith("sha256:" + image["short_id"]))):
            problems.append(f"image {key}: short_id is the image ID's first 12 hex digits, null without an image ID")
        if image["parent"] is not None and image["parent"] not in images:
            problems.append(f"image {key}: parent {image['parent']} is not a catalog image")
        if not set(image["transports"]) <= set(TRANSPORTS):
            problems.append(f"image {key}: transports are of {', '.join(TRANSPORTS)}")
        problems += [f"image {key}: {problem}" for reference in image["evidence"]
                     if (problem := _evidence_problem(reference, root))]
    seen: set[str] = set()
    ids = {entry.get("id") for entry in catalog["enhancements"]}
    for entry in catalog["enhancements"]:
        key = entry.get("id")
        where = f"enhancement {key}"
        if not isinstance(key, str) or not ID.fullmatch(key) or key in seen:
            problems.append(f"{where}: ids are unique lowercase names")
        seen.add(key)
        if not ENTRY_KEYS <= set(entry) <= ENTRY_KEYS | ENTRY_OPTIONAL:
            problems.append(f"{where}: keys are {', '.join(sorted(ENTRY_KEYS))} and optionally "
                            f"{', '.join(sorted(ENTRY_OPTIONAL))}")
            continue
        if entry["category"] not in catalog["categories"]:
            problems.append(f"{where}: category {entry['category']} is not in categories")
        if not entry["sides"] or not set(entry["sides"]) <= set(catalog["sides"]):
            problems.append(f"{where}: sides are of {', '.join(catalog['sides'])}")
        if entry["status"] not in catalog["statuses"]:
            problems.append(f"{where}: status {entry['status']} is not in statuses")
        if not entry["models"]:
            problems.append(f"{where}: lists at least one model")
        for record in entry["models"]:
            if record.get("name") not in names or not set(record) <= {"name", "checkpoints", "gain", "status"}:
                problems.append(f"{where}: model {record.get('name')} is a name of {MODEL_NAMES} with optional "
                                "checkpoints, gain and status")
            if record.get("status") is not None and record["status"] not in catalog["statuses"]:
                problems.append(f"{where}: model {record.get('name')} status is not in statuses")
        parallelism = entry["parallelism"]
        if not isinstance(parallelism, dict) or not {"supported"} <= set(parallelism) <= {"supported", "refused"} \
                or not parallelism["supported"]:
            problems.append(f"{where}: parallelism has supported and optionally refused size rules")
        else:
            problems += [f"{where}: {problem}" for rule in parallelism["supported"]
                         if (problem := _size_problem(rule, False))]
            problems += [f"{where}: {problem}" for rule in parallelism.get("refused", [])
                         if (problem := _size_problem(rule, True))]
        enable = entry["enable"]
        if not isinstance(enable, dict) or not enable or not set(enable) <= ENABLE_KEYS:
            problems.append(f"{where}: enable takes {', '.join(sorted(ENABLE_KEYS))}")
        if entry["detect"] is not None:
            problems += [f"{where}: {problem}" for condition in entry["detect"]
                         if (problem := _condition_problem(condition, images))]
        for field in ("requires",):
            for other in entry.get(field, ()):
                if other not in ids or other == key:
                    problems.append(f"{where}: {field} names {other}, which is not another entry")
        if entry.get("superseded_by") is not None and entry["superseded_by"] not in ids - {key}:
            problems.append(f"{where}: superseded_by names no other entry")
        if not set(entry.get("transports", TRANSPORTS)) <= set(TRANSPORTS):
            problems.append(f"{where}: transports are of {', '.join(TRANSPORTS)}")
        source = entry["provided_by"]
        if not isinstance(source, dict) or not set(source) <= {"images", "host", "port"}:
            problems.append(f"{where}: provided_by takes images, host and port")
        elif not source.get("host"):
            listed = source.get("images")
            if not isinstance(listed, list) or not set(listed) <= set(images) | {"any"}:
                problems.append(f"{where}: provided_by images are catalog image keys or 'any'")
            elif not listed and not source.get("port"):
                problems.append(f"{where}: an entry that no image carries describes the port it needs")
        gains = [entry["gain"]] + [record.get("gain") for record in entry["models"]]
        for gain in gains:
            if gain is None:
                continue
            if not isinstance(gain, dict) or not {"percent", "summary"} <= set(gain) <= {"percent", "summary", "by_tp"} \
                    or (gain["percent"] is not None and not isinstance(gain["percent"], (int, float))):
                problems.append(f"{where}: gain has percent (a number or null), summary and optionally by_tp")
        for measurement in entry["measurements"]:
            if not MEASUREMENT_KEYS <= set(measurement) <= MEASUREMENT_KEYS | MEASUREMENT_OPTIONAL:
                problems.append(f"{where}: a measurement has {', '.join(sorted(MEASUREMENT_KEYS))}")
                continue
            if measurement["image"] is not None and not isinstance(measurement["image"], str):
                problems.append(f"{where}: a measurement's image is a string or null")
            problem = _evidence_problem(measurement["evidence"], root)
            if problem:
                problems.append(f"{where}: {problem}")
        if not entry["evidence"]:
            problems.append(f"{where}: cites at least one evidence reference")
        problems += [f"{where}: {problem}" for reference in entry["evidence"]
                     if (problem := _evidence_problem(reference, root))]
    for item in catalog["not_adopted"]:
        key = item.get("id")
        if set(item) != NOT_ADOPTED_KEYS or not ID.fullmatch(str(key)) or key in seen:
            problems.append(f"not_adopted {key}: unique id and exactly {', '.join(sorted(NOT_ADOPTED_KEYS))}")
            continue
        seen.add(key)
        if not item["models"] or not set(item["models"]) <= names:
            problems.append(f"not_adopted {key}: models are names of {MODEL_NAMES}")
        problems += [f"not_adopted {key}: {problem}" for reference in item["evidence"]
                     if (problem := _evidence_problem(reference, root))]
    for check in catalog["checks"]:
        if set(check) != CHECK_KEYS:
            problems.append(f"check {check.get('id')}: exactly {', '.join(sorted(CHECK_KEYS))}")
            continue
        problems += [f"check {check['id']}: {problem}" for condition in check["when_any"] + check["unless_any"]
                     if (problem := _condition_problem(condition, images))]
    if profiles and not problems:
        computed = profiles_setting(catalog, root)
        for entry in catalog["enhancements"]:
            expected = computed.get(entry["id"], [])
            if sorted(entry["profiles"]) != expected:
                problems.append(f"enhancement {entry['id']}: profiles {sorted(entry['profiles'])} differ from the "
                                f"profiles whose configurations turn it on: {expected}")
    return problems


def serving_profiles(root: Path = ROOT) -> list[str]:
    """Catalog profiles whose configuration is a serving profile with environment and vllm_args."""
    found = []
    for profile_id in sorted(profile_catalog.catalog(root)):
        try:
            profile_configuration(profile_id, root)
        except CatalogError:
            continue
        found.append(profile_id)
    return found


def profiles_setting(catalog: dict, root: Path = ROOT) -> dict[str, list[str]]:
    """``{entry id: sorted profile labels}``: ``PROFILE`` when the profile's own configuration turns the
    entry on, ``PROFILE:CHECKPOINT`` when only that checkpoint's settings do. An entry without a switch
    (``detect`` empty or null) has none. A default that applies when the configuration leaves the switch
    unset counts."""
    found: dict[str, set[str]] = {}
    for profile_id in serving_profiles(root):
        _, config = profile_configuration(profile_id, root)
        variants = [None] + [name for name in sorted(config.get("checkpoints") or {})
                             if name != config.get("checkpoint")]
        default_on: set[str] = set()
        for checkpoint in variants:
            view = deployment(profile_id, catalog=catalog, checkpoint=checkpoint, root=root)
            for entry in catalog["enhancements"]:
                if not entry["detect"] or model_spec(entry, view) is None or not applies_to_transport(entry, view):
                    continue
                if detected(entry, view):
                    if checkpoint is None:
                        default_on.add(entry["id"])
                        found.setdefault(entry["id"], set()).add(profile_id)
                    elif entry["id"] not in default_on:
                        found.setdefault(entry["id"], set()).add(f"{profile_id}:{checkpoint}")
    return {key: sorted(value) for key, value in found.items()}


# -- command line -----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profile", nargs="?", help="profile ID (python scripts/profiles.py list)")
    parser.add_argument("--checkpoint", help="one of the profile's checkpoints (its checkpoints table)")
    parser.add_argument("--image", help="catalog image key; default: the plan's image or the profile's release")
    parser.add_argument("--launch-record", type=Path,
                        help="JSON with the environment and vllm_args that ran (the serving-profile shape)")
    parser.add_argument("--plan", type=Path, help="a serving A/B runner campaign's plan.json")
    parser.add_argument("--arm", help="the plan's arm (default: its first)")
    parser.add_argument("--rank", type=int, default=0, help="the plan's rank (default 0)")
    parser.add_argument("--env-dump", type=Path, help="container environment: KEY=VALUE lines or docker inspect's "
                                                      "Config.Env JSON list")
    parser.add_argument("--args-dump", type=Path, help="container arguments: a JSON list or one command line")
    parser.add_argument("--tp", type=int, help="tensor-parallel size (default: --tensor-parallel-size)")
    parser.add_argument("--dcp", type=int, help="decode-context-parallel size (default: "
                                                "--decode-context-parallel-size, else 1)")
    parser.add_argument("--transport", choices=TRANSPORTS, help="default: inferred from the settings and image")
    parser.add_argument("--enabled", action="append", default=[], metavar="ID",
                        help="an entry that is on outside the configuration, such as a host setting")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--validate", action="store_true",
                        help="validate the catalog and its profiles lists, then exit")
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog()
        if args.validate:
            problems = validate(catalog)
            for problem in problems:
                print(problem, file=sys.stderr)
            print(f"{CATALOG}: {len(catalog['enhancements'])} enhancements, "
                  f"{len(catalog['not_adopted'])} not adopted, {len(problems)} problems")
            return 1 if problems else 0
        if not args.profile:
            parser.error("name a profile, or give --validate")
        view = deployment(args.profile, catalog=catalog, checkpoint=args.checkpoint, image=args.image,
                          launch_record=args.launch_record, plan=args.plan, arm=args.arm, rank=args.rank,
                          environment_dump=args.env_dump, arguments_dump=args.args_dump, tp=args.tp,
                          dcp=args.dcp, transport=args.transport, enabled=tuple(args.enabled))
        report = evaluate(catalog, view)
    except (CatalogError, ValueError, OSError, KeyError) as error:
        print(f"check_enhancements: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=1) if args.json else render(report, catalog))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
