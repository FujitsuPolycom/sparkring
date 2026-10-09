"""Per-rank container commands of each arm, from one base container per rank.

The base of rank ``r`` is the installer's own container for the profile
(:func:`runtime.common.qwen_flash_next.render`, a ``docker create`` command), so Compose
exports are not needed and research-only profiles render the same way. Every arm applies the
same site substitutions (:func:`common`) and then only its own transport part:

- ``S``: SIRCL ring sessions on every collective, NCCL off. The part is what
  ``python -m sparkring_sircl.vllm.serve bundle --nccl never`` produces for the rank: its mounts
  and environment, ``PYTHONPATH`` prepended with the staged tree and ``sircl`` added to
  ``VLLM_PLUGINS``, plus the runner's ``--sircl-env`` variables. ``--roce-slot`` sets
  ``VLLM_ENABLE_ROCE_ALLREDUCE=1`` so that SIRCL's ``roce_slot`` shim takes vLLM's RoCE slot;
  otherwise the bundle's ``0`` stays.
- ``S+``: ``S`` with ``SIRCL_FUSED_NORM=1`` (the adapter's documented switch for the fused
  all-reduce + residual add + RMSNorm).
- ``N``: vLLM's own communicator and PyNccl over the image's NCCL. RoCEnante and the four-rank
  adapter are off (``VLLM_ENABLE_ROCE_ALLREDUCE=0``, ``SPARKRING_TRANSPORT_PROFILE`` and its
  manifest empty, ``SPARK_TP4_ENABLED=0``), and the NCCL variables are those
  ``bundle --nccl auto`` gives the rank: the RDMA devices facing its peers (``NCCL_IB_HCA``) and,
  on a whole cycle, ``NCCL_ALGO=Ring`` with ``NCCL_SKIP_TREE_CONNECT=1``. The profile's other NCCL
  settings stay. Refused where NCCL may not run (a group whose Sparks do not all share cables).
- ``P``: the profile's prepared transport, the base container as the installer renders it.
  Refused unless the placement is one the prepared transport runs (a whole pair site, a whole
  four-Spark ring or one of its halves).

The functions here are pure: they take token lists and bundle documents and return token lists.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence

ARMS = ("S", "S+", "N", "P")
SIRCL_ARMS = ("S", "S+")
# Logging only: every arm prints each NCCL communicator it creates, which the verification counts.
LOGGING = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT"}
NO_PREPARED = {"VLLM_ENABLE_ROCE_ALLREDUCE": "0", "SPARKRING_TRANSPORT_PROFILE": "",
               "SPARKRING_TRANSPORT_MANIFEST_SHA256": "", "SPARK_TP4_ENABLED": "0"}
NCCL_FROM_AUTO = ("NCCL_IB_HCA", "NCCL_ALGO", "NCCL_SKIP_TREE_CONNECT")


class SpecError(ValueError):
    pass


def slug(arm: str) -> str:
    return {"S": "s", "S+": "sp", "N": "n", "P": "p"}[arm]


def image_index(tokens: Sequence[str]) -> int:
    return next(i for i, token in enumerate(tokens) if token.startswith("sha256:"))


def environment(tokens: Sequence[str]) -> dict[str, str]:
    return {tokens[i + 1].split("=", 1)[0]: tokens[i + 1].split("=", 1)[1]
            for i in range(image_index(tokens)) if tokens[i] == "--env"}


def mounts(tokens: Sequence[str]) -> list[str]:
    return [tokens[i + 1] for i in range(image_index(tokens)) if tokens[i] == "--mount"]


def set_arg(tokens: list[str], flag: str, value: str) -> str | None:
    """Replace the value of a vLLM argument after the image, or append the flag; returns the old value."""
    at = image_index(tokens)
    for i in range(at + 1, len(tokens) - 1):
        if tokens[i] == flag:
            old, tokens[i + 1] = tokens[i + 1], value
            return old
    tokens += [flag, value]
    return None


def arg(tokens: Sequence[str], flag: str) -> str | None:
    at = image_index(tokens)
    return next((tokens[i + 1] for i in range(at + 1, len(tokens) - 1) if tokens[i] == flag), None)


def set_env(tokens: list[str], key: str, value: str) -> None:
    for i in range(image_index(tokens)):
        if tokens[i] == "--env" and tokens[i + 1].split("=", 1)[0] == key:
            tokens[i + 1] = f"{key}={value}"
            return
    at = image_index(tokens)
    tokens[at:at] = ["--env", f"{key}={value}"]


def drop_env(tokens: list[str], prefix: str, keep: Sequence[str] = ()) -> None:
    i = 0
    while i < image_index(tokens):
        if tokens[i] == "--env" and tokens[i + 1].startswith(prefix) and tokens[i + 1].split("=", 1)[0] not in keep:
            del tokens[i:i + 2]
            continue
        i += 1


def common(base: Sequence[str], *, docker: Sequence[str], run: str, arm: str, rank: int, model: str, cache: str,
           host_ip: str, seccomp: str | None = None) -> list[str]:
    """The site substitutions every arm shares.

    ``seccomp`` is the host path of runtime/common/loader-seccomp.json, the policy that admits the io_uring
    calls of the B12X weight loader (Docker's default policy refuses them); the runner copies it to every
    Spark and every arm runs under it.
    """
    tokens = list(base)
    if tokens[:3] != ["docker", "create", "--name"]:
        raise SpecError(f"rank {rank}: the base is not a docker create command: {tokens[:3]}")
    tokens = [*docker, "run", "-d", "--name", f"ab-{run}-{slug(arm)}-r{rank}", "--label", f"serving-ab={run}",
              "--label", f"serving-ab.arm={arm}", "--label", f"serving-ab.rank={rank}", *tokens[4:]]
    for i in range(image_index(tokens)):
        if tokens[i] == "--mount":
            fields = dict(item.split("=", 1) for item in tokens[i + 1].split(",") if "=" in item)
            if fields.get("dst") == "/models/target":
                tokens[i + 1] = tokens[i + 1].replace(f"src={fields['src']},", f"src={model},")
            elif fields.get("dst") == "/cache":
                tokens[i + 1] = tokens[i + 1].replace(f"src={fields['src']},", f"src={cache},")
    if seccomp:
        at = image_index(tokens)
        tokens[at:at] = ["--security-opt", f"seccomp={seccomp}"]
    set_env(tokens, "VLLM_HOST_IP", host_ip)
    for key, value in LOGGING.items():
        set_env(tokens, key, value)
    return tokens


def sircl_part(tokens: list[str], part: Mapping, *, fused_norm: bool, roce_slot: bool,
               extra_env: Mapping[str, str]) -> list[str]:
    at = image_index(tokens)
    tokens[at:at] = [item for mount in part["mounts"] for item in ("--mount", mount["option"])]
    for key, value in sorted(part["environment"].items()):
        set_env(tokens, key, value)
    env = environment(tokens)
    set_env(tokens, "PYTHONPATH", ":".join(p for p in (part["pythonpath_prepend"], env.get("PYTHONPATH", "")) if p))
    plugins = [p for p in env.get("VLLM_PLUGINS", "").split(",") if p]
    if part["vllm_plugins_add"] not in plugins:
        plugins.append(part["vllm_plugins_add"])
    set_env(tokens, "VLLM_PLUGINS", ",".join(plugins))
    for key, value in sorted(extra_env.items()):
        set_env(tokens, key, value)
    if fused_norm:
        set_env(tokens, "SIRCL_FUSED_NORM", "1")
    if roce_slot:
        set_env(tokens, "VLLM_ENABLE_ROCE_ALLREDUCE", "1")
    if part.get("vllm_arguments"):
        raise SpecError(f"the bundle adds vLLM arguments this runner does not merge: {part['vllm_arguments']}")
    return tokens


def nccl_part(tokens: list[str], auto_part: Mapping) -> list[str]:
    for key, value in NO_PREPARED.items():
        set_env(tokens, key, value)
    for key in NCCL_FROM_AUTO:
        if key in auto_part["environment"]:
            set_env(tokens, key, auto_part["environment"][key])
    drop_env(tokens, "SIRCL_", keep=("SIRCL_ENABLED",))
    return tokens


def prepared_allowed(ring_size: int, positions: Sequence[int]) -> bool:
    """The placements the prepared transport runs: a whole pair site, a whole four-Spark ring or a half."""
    group = tuple(positions)
    if ring_size == 2:
        return group == (0, 1)
    if ring_size == 4:
        return group in ((0, 1, 2, 3), (0, 1), (2, 3))
    return False


def render_arm(arm: str, bases: Sequence[Sequence[str]], *, docker: Sequence[str], run: str,
               models: Sequence[str], caches: Mapping[str, str], host_ips: Sequence[str], bundle: Mapping | None,
               bundle_auto: Mapping | None, ring_size: int, positions: Sequence[int], roce_slot: bool = False,
               extra_env: Mapping[str, str] | None = None, seccomp: str | None = None) -> list[list[str]]:
    """Every rank's command of one arm."""
    if arm not in ARMS:
        raise SpecError(f"unknown arm {arm!r}; arms are {', '.join(ARMS)}")
    if arm == "N" and (bundle_auto is None or bundle_auto.get("nccl") in (None, "none")):
        raise SpecError("arm N needs a group NCCL may run on (every pair of consecutive ranks cabled): "
                        f"bundle --nccl auto gives this group NCCL policy {bundle_auto and bundle_auto.get('nccl')!r}")
    if arm == "P" and not prepared_allowed(ring_size, positions):
        raise SpecError("arm P: the prepared transport runs a whole pair site, a whole four-Spark ring or one of "
                        f"its halves; positions {list(positions)} on a ring of {ring_size} are none of these")
    out = []
    for rank, base in enumerate(bases):
        tokens = common(base, docker=docker, run=run, arm=arm, rank=rank, model=models[rank],
                        cache=caches[arm], host_ip=host_ips[rank], seccomp=seccomp)
        if arm in SIRCL_ARMS:
            part = bundle["ranks"][rank]
            if part["rank"] != rank or part["lan_address"] != host_ips[rank]:
                raise SpecError(f"bundle rank {part['rank']} at {part['lan_address']} does not match rank {rank}")
            tokens = sircl_part(tokens, part, fused_norm=arm == "S+", roce_slot=roce_slot, extra_env=extra_env or {})
        elif arm == "N":
            tokens = nccl_part(tokens, bundle_auto["ranks"][rank])
        out.append(tokens)
    return out


def diff(a: Sequence[str], b: Sequence[str]) -> dict:
    """What differs between two ranks' commands: environment, mounts and every other token."""
    ea, eb = environment(a), environment(b)

    def rest(tokens):
        out, i = [], 0
        while i < len(tokens):
            if tokens[i] in ("--env", "--mount", "--name", "--label"):
                i += 2
                continue
            out.append(tokens[i])
            i += 1
        return out
    return {"environment": {key: [ea.get(key), eb.get(key)] for key in sorted(set(ea) | set(eb))
                            if ea.get(key) != eb.get(key)},
            "mounts_only_first": [m for m in mounts(a) if m not in mounts(b)],
            "mounts_only_second": [m for m in mounts(b) if m not in mounts(a)],
            "other_tokens_equal": rest(a) == rest(b)}


def shell(tokens: Sequence[str]) -> str:
    return shlex.join(tokens)
