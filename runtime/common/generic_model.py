"""Serve a Hugging Face model with vLLM on a SparkRing pair, four-Spark ring or ring half.

A generic deployment runs a model that no installer profile pins: a public
Hugging Face repository at one commit. Its template profile, ``generic-vllm-tp2``
or ``generic-vllm-tp4`` for the number of Sparks it uses, holds SparkRing's own
part of the command and environment: the parallel layout, the API and master
ports, the fabric settings that every installer profile of that size shares
(NCCL on the cabled ports, RoCE all-reduce) and conservative memory defaults.

The request (``request``) adds the model, the name it is served as, and the
vLLM arguments the operator names. ``apply`` lays it over the template. The
installer records the request in the deployment's selection card
(``selection``), and every Spark reads it from the deployment lock that each
host operation carries, so no Spark needs a file for the model.

The model's files are pinned when the installation is planned (``resolve``):
the Hub's tree listing at the commit gives each file's size and SHA-256
(scripts/pin_checkpoint.py), and the request carries that manifest, which
installer.checkpoint_pins validates as it validates a manifest that ships with
SparkRing. Checkpoint search, download, copies over the cables and file
verification then work as they do for a profile's checkpoint.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import urllib.error

PROFILES = {2: "generic-vllm-tp2", 4: "generic-vllm-tp4"}
# The placeholder model of the template profiles, which a request replaces.
TEMPLATE_REPOSITORY = "sparkring-generic/template"
SCHEMA = "sparkring-generic-model/v1"
KEYS = {"schema", "model", "served_model_name", "arguments", "context_length", "remote_code", "pins"}
MODEL_KEYS = {"repository", "revision", "config_sha256", "index_sha256"}
# The longest context the template serves without an operator's choice. A
# model whose configuration names a shorter maximum gets that maximum: vLLM
# refuses a --max-model-len above it.
DEFAULT_CONTEXT = 32768
# vLLM options that SparkRing sets for every generic deployment, each with the
# SparkRing option that changes it, or None when it follows from the cluster.
OWNED = {
    "--host": "--api-bind", "--port": "--api-port", "--served-model-name": "--name",
    "--model": None, "--tensor-parallel-size": None, "-tp": None, "--pipeline-parallel-size": None, "-pp": None,
    "--data-parallel-size": None, "-dp": None, "--decode-context-parallel-size": None, "-dcp": None,
    "--nnodes": None, "--node-rank": None, "--master-addr": None, "--master-port": None,
    "--distributed-executor-backend": None, "--headless": None, "--download-dir": None,
}
OWNED_PREFIXES = ("--data-parallel-",)
MAX_ARGUMENTS = 64
# The longest pin manifest a request carries, as compact JSON. Every host
# operation receives the deployment lock, which holds the request, as one
# base64 command-line argument, and Linux limits one argument to 128 KiB; the
# rest of a lock takes about 10 KiB. 64 KiB holds about 200 files.
PINS_LIMIT = 64 * 1024
SERVED_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
REPOSITORY_PART = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?"


def is_generic(profile_id):
    """Whether ``profile_id`` is a generic template profile."""
    return profile_id in PROFILES.values()


def profile_for(nodes):
    """The template profile for a deployment on ``nodes`` Sparks."""
    if nodes not in PROFILES:
        raise ValueError("A generic model runs on two or four Sparks")
    return PROFILES[nodes]


def parse_model(text):
    """``(repository, revision)`` of ``OWNER/NAME[@REVISION]``; ``revision`` is None for the default branch."""
    repository, separator, revision = str(text).partition("@")
    if (not re.fullmatch(REPOSITORY_PART + "/" + REPOSITORY_PART, repository)
            or "--" in repository or ".." in repository):
        raise ValueError(f"--model {text}: name a Hugging Face repository as OWNER/NAME, optionally @REVISION")
    if separator and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", revision):
        raise ValueError(f"--model {text}: the revision after @ is a branch, tag or commit id")
    return repository, revision or None


def _flag(token):
    """The option name of ``token`` (``--max-model-len`` for ``--max-model-len=8192``), or None for a value."""
    if not token.startswith("-") or re.fullmatch(r"-\d.*", token):
        return None
    return token.split("=", 1)[0]


def arguments(values):
    """The vLLM arguments ``values``, checked: options with their values, none of SparkRing's own.

    Each item is an option (``--name`` or ``--name=value``) or the value of the
    option before it. A value without an option, such as a model path, is
    refused, as is an option that SparkRing sets (OWNED) and every argument
    with a control character. The result is the arguments as given.
    """
    values = [str(value) for value in values]
    if len(values) > MAX_ARGUMENTS:
        raise ValueError(f"Give at most {MAX_ARGUMENTS} vLLM arguments")
    previous = None
    for value in values:
        if not value or len(value) > 4096 or any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("Each vLLM argument is nonempty text of at most 4096 characters without control characters")
        flag = _flag(value)
        if flag is None:
            if previous is None:
                raise ValueError(f"vLLM argument {value!r} follows no option; SparkRing sets the model itself")
            previous = None
            continue
        if flag in OWNED or flag.startswith(OWNED_PREFIXES):
            instead = OWNED.get(flag)
            raise ValueError(f"SparkRing sets vLLM's {flag} itself" + (f"; use {instead}" if instead else ""))
        previous = None if "=" in value else flag
    return values


def merge(template, extra):
    """The template's vLLM arguments with each option of ``extra`` in place of the template's, or added.

    An option that the template sets with a value takes ``extra``'s value; one
    that the template sets alone (a switch such as ``--enable-prefix-caching``)
    stays; any other option is appended with its value, in ``extra``'s order.
    ``--name=value`` is written as ``--name value``.
    """
    result = list(template)
    pairs, index = [], 0
    while index < len(extra):
        flag, separator, inline = extra[index].partition("=")
        if separator:
            pairs.append((flag, inline))
            index += 1
        elif index + 1 < len(extra) and _flag(extra[index + 1]) is None:
            pairs.append((flag, extra[index + 1]))
            index += 2
        else:
            pairs.append((flag, None))
            index += 1
    for flag, value in pairs:
        if flag in result:
            position = result.index(flag)
            if value is None:
                continue
            if position + 1 < len(result) and _flag(result[position + 1]) is None:
                result[position + 1] = value
            else:
                result.insert(position + 1, value)
        else:
            result += [flag] if value is None else [flag, value]
    return result


def _context(config):
    """The template's context for a model whose ``config.json`` is ``config``."""
    for section in (config, config.get("text_config") or {}, config.get("llm_config") or {}):
        maximum = section.get("max_position_embeddings") if isinstance(section, dict) else None
        if type(maximum) is int and maximum > 0:
            return min(DEFAULT_CONTEXT, maximum)
    return DEFAULT_CONTEXT


def request(repository, revision, pins, config, *, served_model_name=None, extra=()):
    """The generic request for ``repository`` at the commit ``revision``.

    ``pins`` is the revision's pin manifest and ``config`` its ``config.json``
    bytes. The served name defaults to the repository's name. ``remote_code``
    records whether the configuration names custom model code (``auto_map``),
    which vLLM runs only with ``--trust-remote-code``.
    """
    try:
        document = json.loads(config)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"{repository}@{revision}: config.json is not JSON ({error})") from None
    if not isinstance(document, dict):
        raise ValueError(f"{repository}@{revision}: config.json is not a JSON object")
    if hashlib.sha256(config).hexdigest() != pins["files"]["config.json"]["sha256"]:
        raise ValueError(f"{repository}@{revision}: config.json differs from its pinned SHA-256")
    if len(json.dumps(pins, separators=(",", ":"))) > PINS_LIMIT:
        raise ValueError(f"{repository}@{revision} lists {len(pins['files'])} files; a generic deployment carries "
                         f"the list of its files in its deployment lock, which holds about 200 files "
                         f"({PINS_LIMIT // 1024} KiB)")
    served = served_model_name or repository.split("/", 1)[1]
    value = {"schema": SCHEMA,
             "model": {"repository": repository, "revision": revision,
                       "config_sha256": pins["files"]["config.json"]["sha256"],
                       "index_sha256": pins["files"][pins["index"]]["sha256"]},
             "served_model_name": served, "arguments": arguments(extra), "context_length": _context(document),
             "remote_code": "auto_map" in document, "pins": pins}
    check(value)
    return value


def resolve(text, *, served_model_name=None, extra=(), hub=None):
    """Read ``OWNER/NAME[@REVISION]`` from the Hugging Face Hub and return its request (``request``).

    The revision, by default the repository's default branch, resolves to its
    commit, whose files are pinned (scripts/pin_checkpoint.manifest). Requests
    are anonymous, so a gated or private repository is refused.
    """
    from scripts import pin_checkpoint
    hub = hub or pin_checkpoint.Hub()
    repository, revision = parse_model(text)
    arguments(extra)
    named = repository + (f"@{revision}" if revision else "")
    try:
        commit = hub.commit(repository, revision or "main")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"{named}: the Hub named no commit for it")
        downloaded = {}
        pins = pin_checkpoint.manifest(hub, repository, commit, keep=downloaded)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise ValueError(f"{named} is gated or private on Hugging Face; SparkRing installs public models "
                             "without a token") from None
        if error.code == 404:
            raise ValueError(f"{named} was not found on Hugging Face") from None
        raise ValueError(f"{named}: Hugging Face answered {error.code} {error.reason}") from None
    except urllib.error.URLError as error:
        raise ValueError(f"Hugging Face could not be reached for {named}: {error.reason}") from None
    except ValueError as error:
        raise ValueError(f"{named} cannot be served as a generic model: {error}") from None
    return request(repository, commit, pins, downloaded["config.json"], served_model_name=served_model_name,
                   extra=extra)


def check(value):
    """``value`` when it is a well-formed generic request; its pins are checked by installer.checkpoint_pins."""
    if not isinstance(value, dict) or set(value) != KEYS or value["schema"] != SCHEMA:
        raise ValueError("A generic request holds exactly " + ", ".join(sorted(KEYS)))
    model = value["model"]
    if (not isinstance(model, dict) or set(model) != MODEL_KEYS
            or not re.fullmatch(r"[0-9a-f]{40}", str(model["revision"]))
            or not all(re.fullmatch(r"[0-9a-f]{64}", str(model[key])) for key in ("config_sha256", "index_sha256"))):
        raise ValueError("A generic request's model names a repository, a commit and the SHA-256 of config.json "
                         "and its weight index")
    parse_model(model["repository"])
    if not isinstance(value["served_model_name"], str) or not re.fullmatch(SERVED_NAME, value["served_model_name"]):
        raise ValueError("--name is 1 to 128 letters, digits, dots, hyphens and underscores, starting with a "
                         "letter or digit")
    if not isinstance(value["arguments"], list) or arguments(value["arguments"]) != value["arguments"]:
        raise ValueError("A generic request's vLLM arguments are a list of text")
    if type(value["context_length"]) is not int or not 1 <= value["context_length"] <= DEFAULT_CONTEXT:
        raise ValueError(f"A generic request's context length is 1 to {DEFAULT_CONTEXT} tokens")
    if not isinstance(value["remote_code"], bool) or not isinstance(value["pins"], dict):
        raise ValueError("A generic request records remote_code as true or false and its pin manifest")
    return value


def apply(profile, value):
    """The template ``profile`` serving the request ``value``.

    The model and served name replace the template's, the template's
    ``--max-model-len`` takes the request's context length, and the request's
    vLLM arguments then replace or extend the template's (``merge``).
    """
    check(value)
    result = copy.deepcopy(profile)
    result["model"] = dict(value["model"])
    result["served_model_name"] = value["served_model_name"]
    args = result["vllm_args"]
    args[args.index("--max-model-len") + 1] = str(value["context_length"])
    result["vllm_args"] = merge(args, value["arguments"])
    return result


def selection(card, value):
    """The selection card of a generic deployment: the template's card naming the requested model.

    ``card`` is the template profile's card (setup.selection). The card keeps
    the request under ``generic``, from which installer.checkpoint_contract and
    installer.checkpoint_pins read the model and its pins.
    """
    if not is_generic(card["profile"]):
        raise ValueError(f"{card['profile']} is not a generic template profile")
    check(value)
    return {**card, "model_repository": value["model"]["repository"], "model_revision": value["model"]["revision"],
            "generic": value}


def notes(value):
    """Lines that a plan prints about a generic request."""
    lines = [f"Generic model: {value['model']['repository']} at {value['model']['revision'][:12]}, served as "
             f"{value['served_model_name']}, context {value['context_length']} tokens."]
    if value["arguments"]:
        lines.append("vLLM arguments: " + " ".join(value["arguments"]))
    if value["remote_code"] and "--trust-remote-code" not in value["arguments"]:
        lines.append("Warning: this model's configuration names its own model code (auto_map); vLLM runs it only "
                     "with --trust-remote-code after --.")
    return lines
