"""Read the serving facts of one installer profile from its repository files.

An installer profile has `profiles/<id>/profile.json` (catalog record, release)
and a `sparkring-serving-profile/v1` `config.json` (served name, vLLM
arguments, optional `smoke` request settings). Everything the harness needs to
address and exercise the deployment comes from these two files and the
profile's release selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.common import profiles, toolchain_profiles  # noqa: E402

TOPOLOGY_NODES = {"direct-pair-2": 2, "direct-cycle-4": 4}
# The installer's smoke request uses the same default for a profile without
# `smoke` (scripts/installer_host.py smoke_request).
DEFAULT_THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass(frozen=True)
class ProfileInfo:
    id: str
    title: str
    status: str
    served_model_name: str
    port: int
    nodes: int
    topology: str
    repository: str
    revision: str
    image: str
    release: str
    features: frozenset = field(default_factory=frozenset)
    # Request fields merged into requests that ask for a direct answer. They
    # come from the profile's `smoke` settings, which the installer also uses:
    # `thinking` for DeepSeek, `enable_thinking` for Qwen and MiMo, and low
    # `reasoning_effort` for GLM, whose template always reasons.
    thinking_off: dict = field(default_factory=dict)
    # Request fields for the reasoning check; empty means the template default.
    thinking_on: dict = field(default_factory=dict)
    # The listed checkpoint installed with `--checkpoint`; None for the profile's default.
    checkpoint: str | None = None


def _option(args, name):
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def features(args):
    """Request features the served model accepts, from its vLLM arguments."""
    result = set()
    limits = _option(args, "--limit-mm-per-prompt")
    if limits:
        try:
            if int(json.loads(limits).get("image", 0)) > 0:
                result.add("image")
        except (ValueError, TypeError, AttributeError):
            pass
    if "--enable-auto-tool-choice" in args:
        result.add("tools")
    if _option(args, "--reasoning-parser"):
        result.add("reasoning")
    return frozenset(result)


def image_name(release_path, root=ROOT):
    """The installer image's name: the lock beside the release, or the release ID."""
    lock = (root / release_path).parent / "installer-image.json"
    if lock.is_file():
        return profiles.read_json(lock)["name"]
    return profiles.read_json(root / release_path)["id"]


def load(profile_id, root=ROOT, checkpoint=None):
    """The profile's serving facts; ``checkpoint`` selects a checkpoint the profile lists.

    The listed checkpoint's served name, repository, revision and vLLM
    arguments replace the default's, as `sparkring install --checkpoint`
    applies them (toolchain_profiles.checkpoint_settings). Naming the default
    checkpoint, or an alias of it, is the same as naming none.
    """
    profile, _ = profiles.load(profile_id, root)
    source = profile["configuration"]
    if source["format"] != "serving-profile":
        raise ValueError(f"{profile_id} is not an installer profile: its configuration is a {source['format']}")
    config = profiles.read_json(profiles.local_path(source["path"], root))
    if checkpoint is not None:
        default, _ = toolchain_profiles.checkpoint_names(config)
        checkpoint = toolchain_profiles.checkpoint_name(config, checkpoint)
        config = toolchain_profiles.checkpoint_settings(config, checkpoint)
        if checkpoint == default:
            checkpoint = None
    args = [str(value) for value in config.get("vllm_args", [])]
    port = _option(args, "--port")
    if port is None:
        raise ValueError(f"{profile_id}: config.json sets no --port")
    topology = config["topology"]
    nodes = TOPOLOGY_NODES.get(topology) or int(_option(args, "--nnodes") or 0)
    if not nodes:
        raise ValueError(f"{profile_id}: node count unknown for topology {topology}")
    return ProfileInfo(
        id=profile_id, title=profile["title"], status=profile["status"],
        served_model_name=config["served_model_name"], port=int(port), nodes=nodes, topology=topology,
        repository=config["model"]["repository"], revision=config["model"]["revision"],
        image=image_name(profile["release"], root), release=profile["release"], features=features(args),
        thinking_off=config.get("smoke", DEFAULT_THINKING_OFF), thinking_on={}, checkpoint=checkpoint)
