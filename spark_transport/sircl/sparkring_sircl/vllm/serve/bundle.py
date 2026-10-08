"""SIRCL for vLLM containers that another launcher starts: what each ``docker run`` must add, and the checks.

The serve launcher (:mod:`.cli`) starts SparkRing profiles itself. A launcher
that keeps its own image, model, plugins and vLLM arguments (for example a
GLM-5.3 deployment on the whole ring) uses this module instead:

- ``bundle`` prints, per rank, everything its ``docker run`` must add for
  SIRCL to carry the collectives of every vLLM group: the mounts (the staged
  ``sparkring_sircl`` tree and native build cache, read-only, and a per-run
  directory for SIRCL's receipts), the environment (the ``SIRCL_*`` settings
  that place the groups on the ring, ``VLLM_HOST_IP`` and the socket
  interfaces from the site file, and the transports that must stay off), the
  two values it merges with its own (the staged tree before ``PYTHONPATH``,
  ``sircl`` added to ``VLLM_PLUGINS``), and the container requirements SIRCL
  relies on. SIRCL needs no seccomp policy or added capability. With
  ``--reasoning-effort`` it also gives the chat template defaults the vLLM
  command of global rank 0, which serves the API, must add. With
  ``--overlay`` every rank also mounts a source overlay read-only at
  ``/opt/sparkring-overlay`` and puts it before the staged tree on
  ``PYTHONPATH``; ``--vllm-arg``, ``--drop-vllm-arg`` and ``--speculative-set``
  list the vLLM argument changes every rank's command must make;
  ``--b12x-cache-dir`` sets ``B12X_COMPILE_CACHE_DIR``. OFFLINE.
- ``bundle --stage`` also copies the package tree, creates the run
  directories and builds the native library in the serving image on every
  Spark, as ``stage`` does for a profile, after checking every rank's overlay
  directory as ``preflight`` does. MUTATES HOST: files under the site's
  remote directory and one short-lived container per Spark. Its progress
  lines go to standard error, so standard output holds the JSON document
  alone (``bundle --stage > bundle.json`` works).
- ``bundle-check`` reads, after start, each named container's state and SIRCL
  log lines and every rank's receipts, and applies the receipt checks of
  ``check`` (with ``--overlay``, also that ``vllm`` and ``b12x`` came from the
  overlay). READ-ONLY REMOTE.

Global rank ``r`` of the launch must run on the Spark at the bundle's
position ``r`` (``SIRCL_RANK_POSITIONS``); the receipts are named after the
global rank.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...ring import remote
from ...ring.site import Site, SiteError
from .. import catalog, fabric, guard
from ..fabric import NcclPolicy
from . import checks, commands, probe, staging
from . import profile as profile_mod
from . import plan as plan_mod
from .plan import (BUILD_TARGET, DISABLED_TRANSPORTS, OVERLAY_TARGET, RECEIPT_TARGET, RUN_TARGET, SESSION_MODULE,
                   SOURCE_TARGET, Mount, ServePlanError, VllmEdits)
from .sitefile import ServeSite

SCHEMA = "sircl-vllm-bundle/v1"
# NCCL settings under which NCCL's ring algorithm may run on a cycle of Sparks.
RING_SETTINGS = {"NCCL_ALGO": "Ring", "NCCL_SKIP_TREE_CONNECT": "1"}
SESSION_GROUPS = ("tp", "tp,dcp")
REQUIREMENTS = (
    "--network host: vLLM's CPU process groups and SIRCL's setup exchange use the site's LAN addresses",
    "the GPU and the Spark's RDMA devices (--gpus all, /dev/infiniband): SIRCL's sessions open the RoCE "
    "functions of every route",
    "--ulimit memlock=-1:-1: SIRCL registers pinned host memory with the RDMA devices",
    "no other transport in vLLM's RoCE all-reduce slot and no plugin that replaces CudaCommunicator's "
    "collectives, vLLM's DCP combine or the RoCE slot class: SIRCL refuses to share a group",
    "no SIRCL_* variable from the launcher's own environment: the bundle sets every SIRCL_* variable it checks; "
    "one the launcher adds reaches the tensor-parallel session unchecked (for example SIRCL_LINK_SLOTS, which "
    "--link-slots sets), and SIRCL's adapter removes the schedule and link variables from every "
    "decode-context-parallel session",
)
# Where NCCL may not run, launcher settings that would create NCCL communicators outside SIRCL's groups or
# issue collectives SIRCL's tripwire cannot see; the bundle does not see the launcher's own vLLM arguments.
NCCL_FREE_REQUIREMENTS = (
    f"{plan_mod.SPLIT_GROUP} unset or 0: with it vLLM binds the default process group to the GPU, which "
    "creates an NCCL communicator over every rank at startup; SIRCL's general plugin refuses it under "
    "SIRCL_NCCL=never",
    "no --load-format instanttensor: that loader receives the world group's NCCL process group "
    "(vllm/model_executor/model_loader/weight_utils.py, instanttensor_weights_iterator)",
    "no --enable-eplb: expert load balancing calls torch.distributed all_reduce and all_gather through names "
    "bound at import (vllm/distributed/eplb/eplb_state.py, rebalance_execute.py), which SIRCL's tripwire "
    "cannot wrap",
)
# With --tuning-table: the staged tables (bundle --stage writes them; otherwise the launcher copies them).
TUNING_REQUIREMENT = ("the tuning tables at the host paths of tuning.tables on every Spark, inside the run "
                      "directory the run mount maps (bundle --stage writes them): SIRCL_TUNING_TABLE names "
                      "their container paths, and a session refuses setup when a named table is missing")
RECEIPT = "SIRCL receipt"
REGISTERED = "SIRCL vLLM adapter registered"
ACTIVATED = "Platform plugin sircl is activated"
Runner = Callable[..., remote.Result]


@dataclasses.dataclass(frozen=True)
class BundleOptions:
    positions: tuple[int, ...]
    run_id: str | None = None                 # None: bundle-<first>-<last>
    image: str | None = None                  # serving image the stage step builds in (None: the site's)
    session_groups: str = "tp"                # SIRCL_GROUPS
    capacity: int = plan_mod.DEFAULT_CAPACITY
    dispatch: int | None = None               # None: the capacity
    oneshot_max: int | None = None            # SIRCL_ONESHOT_MAX_BYTES; None: the session's default
    large_blocks: int | None = None           # SIRCL_LARGE_BLOCKS; None: the session's default
    gather: int = plan_mod.DEFAULT_GATHER
    nccl_mode: str = plan_mod.DEFAULT_NCCL_MODE   # SIRCL_NCCL
    large_allreduce: str = "auto"             # SIRCL_LARGE_ALLREDUCE
    startup_wait: float = plan_mod.DEFAULT_STARTUP_WAIT_S
    serving_wait: float = plan_mod.DEFAULT_SERVING_WAIT_S
    spin_limit: int | None = None
    gid_index: int | None = None              # None: the site's, else 3
    fused_norm: bool = False                  # SIRCL_FUSED_NORM (research-only)
    large_schedule: str | None = None         # SIRCL_LARGE_SCHEDULE; None: the session's default
    gather_schedule: str | None = None        # SIRCL_GATHER_SCHEDULE
    scatter_schedule: str | None = None       # SIRCL_SCATTER_SCHEDULE
    chain_min: int | None = None              # SIRCL_CHAIN_MIN_BYTES; None: the session's defaults
    ring_min: int | None = None               # SIRCL_RING_MIN_BYTES; None: the session's defaults
    link_sizes: Mapping[str, int] = dataclasses.field(default_factory=dict)  # plan.LinkSize.attribute -> bytes
    link_slots: int | None = None             # SIRCL_LINK_SLOTS; None: the session's default
    ring_gather_stagger: int | None = None    # SIRCL_RING_GATHER_STAGGER; None: the session's default
    tuning_tables: tuple[str, ...] = ()       # --tuning-table PATH (SIRCL_TUNING_TABLE)
    reasoning_effort: str | None = None       # --reasoning-effort, checked against the checkpoint below
    repository: str | None = None             # --repository: the checkout holding profiles/thinking.json
    checkpoint: str | None = None             # --checkpoint (--checkpoint-id) REPOSITORY@REVISION the launcher serves
    thinking_behaviour: str | None = None     # --thinking-behaviour NAME of profiles/thinking.json
    extra_env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    overlay: str | None = None                # --overlay HOSTDIR: every Spark without its own
    overlays: Mapping[int, str] = dataclasses.field(default_factory=dict)   # --overlay N=HOSTDIR
    vllm_edits: VllmEdits | None = None       # --vllm-arg, --drop-vllm-arg, --speculative-set
    b12x_cache_dir: str | None = None         # --b12x-cache-dir: B12X_COMPILE_CACHE_DIR
    require_no_nccl: bool = False             # --require-no-nccl
    nccl_debug: bool = False                  # --nccl-debug
    dcp_size: int = 1                         # --dcp-size: vLLM's decode-context parallelism, for the group map


@dataclasses.dataclass(frozen=True)
class BundleRank:
    rank: int
    position: int
    host: str
    ssh: str
    docker: str
    lan_address: str
    sudo: str
    mounts: tuple[Mount, ...]
    environment: dict[str, str]               # set exactly; PYTHONPATH and VLLM_PLUGINS are merged instead
    directories: tuple[str, ...]              # created on the Spark before the container starts
    vllm_arguments: tuple[str, ...] = ()      # to add to the launcher's vLLM command (rank 0 only)
    pythonpath_prepend: str = SOURCE_TARGET   # before the launcher's PYTHONPATH (an overlay, then the tree)

    def docker_args(self) -> list[str]:
        """``--mount`` and ``--env`` arguments to add to the launcher's ``docker run``."""
        args: list[str] = []
        for mount in self.mounts:
            args += ["--mount", mount.option()]
        for key, value in sorted(self.environment.items()):
            args += ["--env", f"{key}={value}"]
        return args


@dataclasses.dataclass(frozen=True)
class BundlePlan:
    run_id: str
    site: Site
    positions: tuple[int, ...]
    group: fabric.GroupTopology
    nccl_policy: NcclPolicy
    nccl_reason: str
    options: BundleOptions
    staged_digest: str
    library: str
    image: str
    ranks: tuple[BundleRank, ...]
    chat_template_defaults: Mapping[str, Any] | None = None   # what --reasoning-effort adds on rank 0
    thinking: profile_mod.ThinkingBehaviour | None = None     # the checkpoint's thinking behaviour
    thinking_source: str = ""                                 # profiles/thinking.json or --thinking-behaviour
    overlay: Mapping[int, str] = dataclasses.field(default_factory=dict)   # position -> overlay HOSTDIR
    tuning: plan_mod.TuningPlan = dataclasses.field(default_factory=plan_mod.TuningPlan)   # --tuning-table

    @property
    def remote_root(self) -> str:
        return f"{self.site.remote_dir}/serve"

    @property
    def source_dir(self) -> str:
        return f"{self.remote_root}/src/{self.staged_digest}"

    @property
    def build_dir(self) -> str:
        return f"{self.site.remote_dir}/build-cache"

    @property
    def run_dir(self) -> str:
        return f"{self.remote_root}/runs/{self.run_id}"

    @property
    def receipt_dir(self) -> str:
        return f"{self.run_dir}/receipts"

    def carriers(self) -> tuple[list[plan_mod.GroupCarrier], list[tuple[str, str]]]:
        """The process groups vLLM builds when tensor parallelism spans every bundle rank, and their carriers."""
        allowed = self.nccl_policy is not NcclPolicy.NONE
        groups = plan_mod.group_carriers(self.group.layout, self.positions, nccl_mode=self.options.nccl_mode,
                                         environment=RING_SETTINGS, dcp=self.options.dcp_size,
                                         session_groups=self.options.session_groups.split(","))
        dispatch = self.options.capacity if self.options.dispatch is None else self.options.dispatch
        return groups, plan_mod.collective_carriers(nccl_allowed=allowed, large=self.options.large_allreduce,
                                                    dispatch=dispatch, gather=self.options.gather)

    def render_text(self) -> str:
        """The bundle as plan text: the group, its NCCL policy, the groups vLLM builds and what carries them."""
        lines = [f"bundle {self.run_id}: SIRCL for a vLLM launch on Sparks {list(self.positions)} of the "
                 f"{self.site.size}-Spark ring (global rank r on position r)",
                 f"  group {self.group.fabric.describe()} with NCCL policy {self.nccl_policy.value} ({self.nccl_reason}); "
                 f"session groups {self.options.session_groups}; SIRCL_LARGE_ALLREDUCE={self.options.large_allreduce}",
                 f"  tensor-parallel session: {plan_mod.oneshot_text(self.options.oneshot_max, self.derived_oneshot_max)}"
                 f"; {plan_mod.POST_ORDER_TEXT}",
                 *plan_mod.session_lines(self.sessions()),
                 *self.tuning.lines(),
                 *plan_mod.carrier_lines(*self.carriers()),
                 plan_mod.nccl_free_text(required=self.options.require_no_nccl,
                                         debug=self.options.nccl_debug or self.options.require_no_nccl,
                                         nccl_allowed=self.nccl_policy is not NcclPolicy.NONE),
                 "  the map assumes tensor parallelism over every bundle rank with pipeline, data and prefill-"
                 "context parallelism 1"]
        return "\n".join(lines)

    def sessions(self) -> list[plan_mod.SessionSettings]:
        """The tensor-parallel session over every bundle rank and, with --session-groups tp,dcp and --dcp-size
        above 1, the decode-context-parallel sessions, as they will run."""
        return bundle_sessions(self.group, self.positions, self.options)

    @property
    def derived_oneshot_max(self) -> int:
        """The one-shot limit the tensor-parallel sessions derive when --oneshot-max is not given."""
        return plan_mod.derived_oneshot_max(self.group, self.options.capacity)

    def check_command(self, site_path: str = "SITE") -> str:
        options = self.options
        extra = (["--large-blocks", str(options.large_blocks)] if options.large_blocks is not None else [])
        extra += (["--require-no-nccl"] if options.require_no_nccl else
                  ["--nccl-debug"] if options.nccl_debug else [])
        return " ".join(["python -m sparkring_sircl.vllm.serve bundle-check", "--site", shlex.quote(site_path),
                         "--positions", ",".join(str(p) for p in self.positions), "--run-id", self.run_id,
                         "--nccl", options.nccl_mode, *extra, "--container", shlex.quote("NAME-{rank}")])

    def to_json(self, site_path: str = "SITE") -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "positions": list(self.positions),
            "fabric": self.group.fabric.describe(),
            "nccl": self.nccl_policy.value,
            "nccl_reason": self.nccl_reason,
            "nccl_mode": self.options.nccl_mode,
            "session_groups": self.options.session_groups,
            "session_defaults": {"oneshot_max": self.options.oneshot_max,
                                 "oneshot_max_derived": self.derived_oneshot_max,
                                 "post_order": plan_mod.DEFAULT_POST_ORDER},
            "staged_digest": self.staged_digest,
            "library": self.library,
            "image": self.image,
            "remote": {"source": self.source_dir, "build": self.build_dir, "run": self.run_dir,
                       "receipts": self.receipt_dir},
            "container": {"source": SOURCE_TARGET, "build": BUILD_TARGET, "run": RUN_TARGET,
                          "receipts": RECEIPT_TARGET},
            "merge": {
                **({key: f"set to {value}; the launcher must not override it (NCCL's ring on this cycle)"
                    for key, value in RING_SETTINGS.items()} if self.nccl_policy is NcclPolicy.RING else {}),
                "PYTHONPATH": f"prepend {self.ranks[0].pythonpath_prepend} (':' before the launcher's value, if any)",
                "VLLM_PLUGINS": "add sircl to the launcher's comma-separated list; leave the variable unset when "
                                "the launcher does not set it (vLLM then loads every installed plugin)",
                **({plan_mod.CHAT_TEMPLATE_FLAG: "rank 0 only, the rank that serves the API: add the keys of its "
                                                 "vllm_arguments value to the launcher's own value, or the argument "
                                                 "itself when the launcher sets none"}
                   if self.chat_template_defaults is not None else {}),
                **({"vllm_arguments": "every rank: replace or add each argument of vllm_edits.set (a null value is "
                                      "a switch), remove each of vllm_edits.remove with its value, and set the "
                                      "fields of vllm_edits.speculative_config in the JSON object "
                                      "--speculative-config gives"} if self.options.vllm_edits else {}),
            },
            "vllm_edits": _edits_json(self.options.vllm_edits),
            "overlay": ({"target": OVERLAY_TARGET, "sources": {str(position): path for position, path
                                                               in sorted(self.overlay.items())},
                         "required_files": list(plan_mod.OVERLAY_FILES)} if self.overlay else None),
            "serving": {
                "reasoning_effort": self.options.reasoning_effort,
                "default_chat_template_kwargs": (dict(self.chat_template_defaults)
                                                 if self.chat_template_defaults is not None else None),
                "checkpoint": self.options.checkpoint,
                "thinking_behaviour": self.thinking.id if self.thinking else None,
                "template_level": self.thinking.level if self.thinking else None,
                "thinking_source": self.thinking_source or None,
            },
            "requirements": list(REQUIREMENTS) + ([TUNING_REQUIREMENT] if self.tuning.tables else []) + (
                list(NCCL_FREE_REQUIREMENTS)
                                                  if self.nccl_policy is NcclPolicy.NONE else []),
            "nccl_free": {"required": self.options.require_no_nccl,
                          "nccl_debug": self.options.nccl_debug or self.options.require_no_nccl,
                          "nccl_allowed": self.nccl_policy is not NcclPolicy.NONE,
                          "log_patterns": list(plan_mod.NCCL_INIT_PATTERNS),
                          "library_patterns": list(plan_mod.NCCL_LIBRARY_PATTERNS)},
            "sessions": [session.to_json() for session in self.sessions()],
            "tuning": self.tuning.to_json(self.run_dir),
            "carriers": {"groups": [dataclasses.asdict(row) for row in self.carriers()[0]],
                         "collectives": [{"collective": name, "carrier": carrier}
                                         for name, carrier in self.carriers()[1]]},
            "security_options": [],
            "route_maps": {str(rank): fabric.format_routes(self.group.route_map(rank))
                           for rank in range(len(self.positions))},
            "relays": plan_mod.relay_rows(self.group),
            "check": self.check_command(site_path),
            "ranks": [{
                "rank": launch.rank, "position": launch.position, "host": launch.host, "ssh": launch.ssh,
                "lan_address": launch.lan_address,
                "mounts": [{**dataclasses.asdict(mount), "option": mount.option()} for mount in launch.mounts],
                "environment": launch.environment,
                "pythonpath_prepend": launch.pythonpath_prepend,
                "vllm_plugins_add": "sircl",
                "directories": list(launch.directories),
                "docker_args": launch.docker_args(),
                "vllm_arguments": list(launch.vllm_arguments),
            } for launch in self.ranks],
        }


def default_run_id(positions: Sequence[int]) -> str:
    return f"bundle-{positions[0]}-{positions[-1]}"


def bundle_sessions(group: fabric.GroupTopology, positions: Sequence[int],
                    options: BundleOptions) -> list[plan_mod.SessionSettings]:
    """The sessions of a launch with tensor parallelism over every bundle rank: the tensor-parallel session
    with the bundle's schedule and link options, and with ``tp,dcp`` session groups and a DCP size above 1 one
    session per decode-context-parallel group with the session defaults (SIRCL's adapter builds those without
    the tensor-parallel session's variables, ``settings.TP_SESSION_VARIABLES``)."""
    schedules = {attribute: getattr(options, attribute) for attribute, _ in plan_mod.SCHEDULE_VARIABLES
                 if getattr(options, attribute) is not None}
    rows = [plan_mod.session_settings(group, name="tp", groups=f"one group of ranks 0-{len(positions) - 1}",
                                      scoped=True, schedules=schedules, link_sizes=options.link_sizes,
                                      link_slots=options.link_slots, chain_min=options.chain_min,
                                      ring_min=options.ring_min,
                                      ring_gather_stagger=options.ring_gather_stagger)]
    size = options.dcp_size
    if "dcp" in options.session_groups.split(",") and size > 1 and len(positions) % size == 0:
        layout = group.layout
        subgroups = [fabric.describe_group(layout, list(positions[start:start + size]), parent=list(positions))
                     for start in range(0, len(positions), size)]
        rows.append(plan_mod.session_settings(subgroups[0], name="dcp", scoped=False,
                                              groups=f"{len(subgroups)} groups of {size} consecutive ranks"))
    return rows


def bundle_tuning_sessions(group: fabric.GroupTopology, positions: Sequence[int], options: BundleOptions,
                           policy: NcclPolicy) -> list[tuple[str, fabric.GroupTopology, NcclPolicy]]:
    """The sessions a tuning table may serve: the tensor-parallel session (``policy``) and, with ``tp,dcp`` session
    groups and a DCP size above 1, the first decode-context-parallel group's (every DCP group has its shape)."""
    rows = [("tp", group, policy)]
    size = options.dcp_size
    if "dcp" in options.session_groups.split(",") and size > 1 and len(positions) % size == 0:
        sub = fabric.describe_group(group.layout, list(positions[:size]), parent=list(positions))
        sub_policy, _ = guard.effective_policy(sub.nccl_policy, sub.nccl_reason, nccl_mode=options.nccl_mode,
                                               environ=RING_SETTINGS)
        rows.append(("dcp", sub, sub_policy))
    return rows


def build_bundle(site: ServeSite | Site, options: BundleOptions, *, staged_digest: str,
                 library: str) -> BundlePlan:
    """Every rank's additions for the group of Sparks ``options.positions``, in rank order."""
    serve_site = site if isinstance(site, ServeSite) else ServeSite.of(site)
    ring = serve_site.site
    positions = tuple(int(p) for p in options.positions)
    if len(positions) < 2:
        raise ServePlanError("a bundle needs at least two Sparks")
    if len(set(positions)) != len(positions) or any(not 0 <= p < ring.size for p in positions):
        raise ServePlanError(f"positions {list(positions)} must be distinct Sparks of the {ring.size}-Spark ring")
    layout = fabric.Layout.ring(ring.size)
    try:
        group = fabric.describe_group(layout, positions)
    except fabric.FabricError as error:
        raise ServePlanError(str(error)) from None
    run_id = options.run_id or default_run_id(positions)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", run_id):
        raise ServePlanError(f"run id {run_id!r} must be 1-40 lowercase letters, digits and dashes")
    if options.session_groups not in SESSION_GROUPS:
        raise ServePlanError(f"the session groups must be one of {SESSION_GROUPS}")
    if options.nccl_mode not in plan_mod.NCCL_MODES or options.large_allreduce not in plan_mod.LARGE_MODES:
        raise ServePlanError(f"SIRCL_NCCL must be one of {plan_mod.NCCL_MODES} and SIRCL_LARGE_ALLREDUCE one of "
                             f"{plan_mod.LARGE_MODES}")
    dispatch = options.capacity if options.dispatch is None else options.dispatch
    for name, value in (("capacity", options.capacity), ("dispatch ceiling", dispatch),
                        ("gather capacity", options.gather)):
        if value < 16 or value % 16:
            raise ServePlanError(f"the SIRCL {name} must be a positive multiple of 16 bytes")
    if dispatch > options.capacity:
        raise ServePlanError("the dispatch ceiling cannot exceed the all-reduce capacity")
    oneshot = plan_mod.oneshot_environment(options.oneshot_max, dispatch)
    large_blocks = plan_mod.large_blocks_environment(options.large_blocks)
    if options.spin_limit is not None and not 1 <= options.spin_limit < 1 << 32:
        raise ServePlanError("the spin limit must be a positive 32-bit poll count")
    for name, seconds in (("startup", options.startup_wait), ("serving", options.serving_wait)):
        if not 1e-6 <= seconds <= plan_mod.MAX_WAIT_S:
            raise ServePlanError(f"the {name} wait must be between 1e-6 and {plan_mod.MAX_WAIT_S:.0f} seconds")
    plan_mod.check_extra_env(options.extra_env)
    schedules = plan_mod.schedule_environment(options)
    thinking, thinking_source, chat_defaults = _reasoning(options)
    overlays = plan_mod.overlay_paths(positions, options.overlay, options.overlays)
    b12x_cache = plan_mod.check_b12x_cache_dir(options.b12x_cache_dir)
    ring_min = plan_mod.ring_min_environment(options.ring_min)
    chain_min = plan_mod.chain_min_environment(options.chain_min)
    links = plan_mod.link_environment(options.link_sizes)
    links.update(plan_mod.link_slots_environment(options.link_slots))
    links.update(plan_mod.ring_gather_stagger_environment(options.ring_gather_stagger))
    gid = options.gid_index if options.gid_index is not None else ring.gid_index
    gid = 3 if gid is None else gid
    if not 0 <= gid <= 255:
        raise ServePlanError("the GID index must be in 0-255")
    # On a cycle NCCL's ring may run only with NCCL_ALGO=Ring and
    # NCCL_SKIP_TREE_CONNECT=1; the bundle sets both whenever it lets NCCL run there.
    policy, reason = guard.effective_policy(group.nccl_policy, group.nccl_reason, nccl_mode=options.nccl_mode,
                                            environ=RING_SETTINGS)
    if options.require_no_nccl and policy is not NcclPolicy.NONE:
        raise ServePlanError(f"--require-no-nccl: NCCL may run on {group.fabric.describe()} ({reason}); use "
                             "--nccl never so that SIRCL carries every collective")
    if not isinstance(options.dcp_size, int) or options.dcp_size < 1 or len(positions) % options.dcp_size:
        raise ServePlanError(f"--dcp-size {options.dcp_size} must divide the bundle's {len(positions)} ranks")
    if options.dcp_size > 1 and "dcp" not in options.session_groups.split(","):
        raise ServePlanError(f"--dcp-size {options.dcp_size}: decode-context-parallel groups need a SIRCL session, "
                             "and SIRCL's communicator refuses them at startup without one; use --session-groups "
                             "tp,dcp")
    plan_mod.session_problems(bundle_sessions(group, positions, options))
    nccl_logging = plan_mod.nccl_debug_environment(options.nccl_debug or options.require_no_nccl, options.extra_env)
    nccl_free = plan_mod.nccl_free_problems(options.extra_env, nccl_mode=options.nccl_mode,
                                            required=options.require_no_nccl)
    if nccl_free:
        raise ServePlanError("NCCL communicators would be created outside SIRCL's groups: " + "; ".join(nccl_free))
    ring_settings = RING_SETTINGS if policy is NcclPolicy.RING else {}
    clashing = sorted(key for key, value in ring_settings.items() if options.extra_env.get(key, value) != value)
    if clashing:
        raise ServePlanError(f"--env {clashing}: NCCL's ring on this cycle needs "
                             + ", ".join(f"{key}={value}" for key, value in ring_settings.items()))
    run_dir = f"{ring.remote_dir}/serve/runs/{run_id}"
    common = {
        "SIRCL_MODE": "custom",
        "SIRCL_FABRIC": layout.describe(),
        "SIRCL_RANK_POSITIONS": ",".join(str(p) for p in positions),
        "SIRCL_GROUPS": options.session_groups,
        "SIRCL_NCCL": options.nccl_mode,
        "SIRCL_LARGE_ALLREDUCE": options.large_allreduce,
        "SIRCL_SESSION_MODULE": SESSION_MODULE,
        "SIRCL_RECEIPT_DIR": RECEIPT_TARGET,
        "SIRCL_BUILD_CACHE_DIR": BUILD_TARGET,
        "SIRCL_NATIVE_LIBRARY": f"{BUILD_TARGET}/{library}",
        "SIRCL_ALLREDUCE_CAPACITY_BYTES": str(options.capacity),
        "SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES": str(dispatch),
        "SIRCL_ALLGATHER_MAX_BYTES": str(options.gather),
        "SIRCL_STARTUP_WAIT_S": f"{options.startup_wait:g}",
        "SIRCL_SERVING_WAIT_S": f"{options.serving_wait:g}",
        "SIRCL_GID_INDEX": str(gid),
        "SIRCL_FUSED_NORM": "1" if options.fused_norm else "0",
        **DISABLED_TRANSPORTS,
    }
    if options.spin_limit is not None:
        common["SIRCL_SPIN_LIMIT"] = str(options.spin_limit)
    common.update(oneshot)
    common.update(large_blocks)
    common.update(schedules)
    common.update(chain_min)
    common.update(ring_min)
    common.update(links)
    common.update(ring_settings)
    common.update(nccl_logging)
    tuning = plan_mod.tuning_plan(options.tuning_tables, bundle_tuning_sessions(group, positions, options, policy),
                                  large=options.large_allreduce,
                                  shared={"ep": "tp"} if policy is NcclPolicy.NONE else {})
    common.update(tuning.environment())
    if b12x_cache is not None:
        common[plan_mod.B12X_CACHE_VARIABLE] = b12x_cache
    ranks = []
    for rank, position in enumerate(positions):
        host = ring.host(position)
        environment = dict(common)
        environment["VLLM_HOST_IP"] = host.lan_address
        environment["GLOO_SOCKET_IFNAME"] = ring.lan_interface
        environment["NCCL_SOCKET_IFNAME"] = ring.lan_interface
        if policy.allows("all_reduce"):
            environment["NCCL_IB_HCA"] = plan_mod.nccl_hca_value(None, plan_mod.rank_devices(group, rank))
        environment.update(options.extra_env)
        ranks.append(BundleRank(
            rank=rank, position=position, host=host.name, ssh=host.ssh, docker=host.docker,
            lan_address=host.lan_address, sudo=serve_site.sudo.get(position, ""),
            mounts=(*((Mount(overlays[position], OVERLAY_TARGET, True),) if overlays else ()),
                    Mount(f"{ring.remote_dir}/serve/src/{staged_digest}", SOURCE_TARGET, True),
                    Mount(f"{ring.remote_dir}/build-cache", BUILD_TARGET, True),
                    Mount(run_dir, RUN_TARGET, False)),
            environment=environment, directories=(run_dir, f"{run_dir}/receipts"),
            vllm_arguments=((plan_mod.CHAT_TEMPLATE_FLAG, json.dumps(dict(chat_defaults), separators=(",", ":")))
                            if rank == 0 and chat_defaults is not None else ()),
            pythonpath_prepend=f"{OVERLAY_TARGET}:{SOURCE_TARGET}" if overlays else SOURCE_TARGET,
        ))
    return BundlePlan(run_id=run_id, site=ring, positions=positions, group=group, nccl_policy=policy,
                      nccl_reason=reason, options=options, staged_digest=staged_digest, library=library,
                      image=options.image or ring.image, ranks=tuple(ranks), chat_template_defaults=chat_defaults,
                      thinking=thinking, thinking_source=thinking_source, overlay=overlays, tuning=tuning)


def _edits_json(edits: VllmEdits | None) -> dict[str, Any] | None:
    """The vLLM argument changes every rank's command must make, or None without any."""
    if not edits:
        return None
    return {"set": [[flag, value] for flag, value in edits.set], "remove": list(edits.drop),
            "speculative_config": dict(edits.speculative)}


def _reasoning(options: BundleOptions) -> tuple[profile_mod.ThinkingBehaviour | None, str,
                                                dict[str, str] | None]:
    """The served checkpoint's thinking behaviour, where it comes from, and the chat template defaults
    ``--reasoning-effort`` sets: the behaviour ``--repository``'s profiles/thinking.json records for
    ``--checkpoint``, or the one ``--thinking-behaviour`` names. ``--checkpoint`` alone only names the
    checkpoint in the bundle record."""
    revision = name = None
    if options.checkpoint is not None:
        name, revision = plan_mod.parse_checkpoint(options.checkpoint, "--checkpoint")
    if options.reasoning_effort is None and options.thinking_behaviour is None:
        if options.repository:
            raise ServePlanError("--repository gives the profiles/thinking.json --reasoning-effort and "
                                 "--thinking-behaviour read; give it with one of them")
        return None, "", None
    if not options.repository or (options.checkpoint is None and options.thinking_behaviour is None):
        raise ServePlanError(f"--reasoning-effort needs --repository (the SparkRing checkout with "
                             f"{profile_mod.THINKING_RELATIVE}) and --checkpoint REPOSITORY@REVISION (the "
                             "checkpoint the launcher serves) or --thinking-behaviour NAME"
                             if options.reasoning_effort is not None else
                             f"--thinking-behaviour needs --repository (the SparkRing checkout with "
                             f"{profile_mod.THINKING_RELATIVE})")
    root = Path(options.repository)
    if not (root / profile_mod.THINKING_RELATIVE).is_file():
        raise ServePlanError(f"--repository {options.repository} has no {profile_mod.THINKING_RELATIVE}")
    recorded = None
    if name is not None:
        try:
            recorded = profile_mod.thinking_behaviour(root, name, revision)
        except profile_mod.ProfileError as error:
            raise ServePlanError(str(error)) from None
    checkpoint = options.checkpoint or "the served checkpoint"
    behaviour, source = plan_mod.served_thinking(root, checkpoint, recorded, options.thinking_behaviour)
    return behaviour, source, plan_mod.reasoning_kwargs(behaviour, options.reasoning_effort, checkpoint)


# -- remote steps ----------------------------------------------------------------------------------


def stage(plan: BundlePlan, tree: staging.StagedTree, run: Runner, out: Callable[[str], None]) -> int:
    """Copy the package tree, create the run directories and build the native library on every Spark."""
    failures = 0
    tar = tree.tar()
    rows = []
    for launch in plan.ranks:
        mount = commands.overlay_mount(launch)
        if mount is not None:
            answer = run(launch.ssh, commands.overlay_facts(mount.source, launch.sudo), timeout=60)
            rows.append((f"bundle {plan.run_id} rank {launch.rank} ({launch.host})", mount.source,
                         _facts(answer.stdout)))
    if rows:
        lines, blockers = checks.overlay_findings(rows)
        for line in lines:
            out(line)
        for blocker in blockers:
            out(f"BLOCKER: {blocker}")
        failures += bool(blockers)
    for launch in plan.ranks:
        where = f"bundle {plan.run_id} rank {launch.rank} ({launch.host})"
        unpacked = run(launch.ssh, commands.stage_tree(plan.source_dir, plan.staged_digest), timeout=300,
                       input_bytes=tar)
        if not unpacked.ok:
            out(f"{where}: staging the package tree failed: {(unpacked.stderr or unpacked.stdout).strip()}")
            failures += 1
            continue
        made = run(launch.ssh, commands.make_directories(launch.directories, launch.sudo), timeout=60)
        if not made.ok:
            out(f"{where}: creating {list(launch.directories)} failed: {(made.stderr or made.stdout).strip()}")
            failures += 1
            continue
        written = []
        for table, command in plan_mod.write_tuning_tables(plan.tuning.tables, plan.run_dir, launch.sudo):
            answer = run(launch.ssh, command, timeout=60, input_bytes=table.data)
            if not answer.ok:
                out(f"{where}: writing tuning table {table.hash} failed: "
                    f"{(answer.stderr or answer.stdout).strip()}")
                failures += 1
                break
            written.append(" ".join(filter(None, (table.hash, answer.stdout.strip()))))
        else:
            if written:
                out(f"{where}: tuning tables " + ", ".join(written))
        if len(written) != len(plan.tuning.tables):
            continue
        built = run(launch.ssh, commands.stage_container(plan, launch, image=plan.image), timeout=900)
        record = probe.parse(built.stdout)
        if record is None:
            tail = (built.stdout + built.stderr).strip().splitlines()[-5:]
            out(f"{where}: the stage container failed (exit code {built.returncode}): {tail}")
            failures += 1
            continue
        blockers, notes = probe.evaluate(record, staged_root=SOURCE_TARGET, library=plan.library,
                                         overlay=OVERLAY_TARGET if plan.overlay else None)
        modules = record.get("modules") or {}
        out(f"{where}: package tree {plan.staged_digest} {unpacked.stdout.strip()}, library {plan.library} built "
            f"in {plan.image}, sparkring_sircl from {record.get('package')}, vllm from {modules.get('vllm')}, "
            f"b12x from {modules.get('b12x')}")
        for note in notes:
            out(f"{where}: {note}")
        for line in catalog.stage_lines(record.get("shim_status")):
            out(f"{where}: {line}")
        for blocker in blockers:
            out(f"BLOCKER: {where}: {blocker}")
        failures += bool(blockers)
    out("bundle staged" if not failures else f"bundle staging failed: {failures} problem(s)")
    return 0 if not failures else 1


def _facts(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("\t")
        if separator:
            values[key.strip()] = value.strip()
    return values


def container_names(values: Sequence[str], plan: BundlePlan) -> dict[int, str]:
    """Each rank's container: ``N=NAME`` per position, or one name with ``{rank}`` and ``{position}``."""
    named: dict[int, str] = {}
    template = None
    for value in values:
        key, separator, rest = value.partition("=")
        if separator and key.strip().isdigit():
            position = int(key)
            if position not in plan.positions:
                raise ServePlanError(f"--container names Spark {position}, which serves no rank of the bundle")
            named[position] = rest
        elif template is None:
            template = value
        else:
            raise ServePlanError("--container is given twice without a Spark position")
    result = {}
    for launch in plan.ranks:
        name = named.get(launch.position)
        if name is None and template is not None:
            name = template.replace("{rank}", str(launch.rank)).replace("{position}", str(launch.position))
        if not name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ServePlanError(f"rank {launch.rank} (Spark {launch.position}) has no valid container name")
        result[launch.rank] = name
    return result


def check(plan: BundlePlan, containers: Mapping[int, str], run: Runner, out: Callable[[str], None]) -> int:
    """Each rank's container state, SIRCL log lines and receipts, with the receipt checks of ``check``."""
    problems: list[str] = []
    records: dict[int, list[dict[str, Any]]] = {}
    for launch in plan.ranks:
        name = containers[launch.rank]
        where = f"rank {launch.rank} ({launch.host}, container {name})"
        state = run(launch.ssh, commands.container_state(name, launch.docker), timeout=60).stdout.split()
        if not state or state[0] != "running":
            problems.append(f"{where}: container state {' '.join(state) or 'unknown'}")
        found = run(launch.ssh, commands.log_lines(name, launch.docker, (RECEIPT, REGISTERED, ACTIVATED)),
                    timeout=120).stdout.splitlines()
        lines = [line[line.index(RECEIPT):] for line in found if RECEIPT in line]
        out(f"{where}: {len(lines)} receipt line(s), {sum(REGISTERED in line for line in found)} registration "
            f"line(s), {sum(ACTIVATED in line for line in found)} platform activation line(s)")
        for line in lines:
            out(f"  {line}")
        if not any("group=tp" in line for line in lines):
            problems.append(f"{where}: no SIRCL receipt line for the tensor-parallel group; the sircl plugins did "
                            "not run in this rank's worker (PYTHONPATH, VLLM_PLUGINS)")
        answer = run(launch.ssh, commands.receipts(plan.receipt_dir, launch.rank), timeout=60)
        parsed, bad = checks.parse_receipts(answer.stdout)
        records[launch.rank] = parsed
        problems += [f"{where}: {item}" for item in bad]
    found_problems, lines = checks.evaluate_receipts(records, len(plan.ranks))
    imports, wrong = checks.import_findings(records, OVERLAY_TARGET if plan.overlay else None)
    grids, unequal = checks.large_blocks_findings(records, plan.options.large_blocks)
    tuned, untuned = checks.tuning_findings(records, plan.tuning.expected())
    for line in lines + imports + grids + tuned:
        out(line)
    problems += found_problems + wrong + unequal + untuned
    if plan.options.require_no_nccl:
        free, nccl = checks.nccl_free_findings(records, len(plan.ranks))
        logged = {launch.rank: run(launch.ssh, commands.log_lines(containers[launch.rank], launch.docker,
                                                                  plan_mod.NCCL_LOG_PATTERNS),
                                   timeout=120).stdout.splitlines() for launch in plan.ranks}
        scan, created = checks.nccl_log_findings(logged, debug=plan.options.nccl_debug
                                                 or plan.options.require_no_nccl)
        for line in free + scan:
            out(line)
        problems += nccl + created
    for problem in problems:
        out(f"PROBLEM: {problem}")
    out(f"bundle {plan.run_id}: check passed" if not problems else
        f"bundle {plan.run_id}: check failed: {len(problems)} problem(s)")
    return 0 if not problems else 1


# -- command line ----------------------------------------------------------------------------------


def add_arguments(sub: argparse._SubParsersAction) -> None:
    for name in ("bundle", "bundle-check"):
        command = sub.add_parser(name)
        command.add_argument("--site", required=True, type=Path, help="site description (sircl-ring-site/v1)")
        command.add_argument("--positions", required=True,
                             help="ring positions of global ranks 0..N-1, consecutive along the ring: 0-7, 0-3, "
                                  "7,0,1,2 or 7-2")
        command.add_argument("--run-id", help="names the receipt directory (default bundle-<first>-<last>)")
        plan_mod.add_overlay_argument(command)
        plan_mod.add_large_blocks_argument(command)
        plan_mod.add_nccl_free_arguments(command, debug=False)
        command.add_argument("--nccl-debug", dest="nccl_debug", action="store_true",
                             help="bundle: NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS=INIT on every rank; bundle-check: "
                                  "the containers run with them, so a clean log scan covers NCCL's own lines")
        command.add_argument("--nccl", choices=plan_mod.NCCL_MODES, default=plan_mod.DEFAULT_NCCL_MODE,
                             type=plan_mod.nccl_mode_value,
                             help=plan_mod.NCCL_MODE_HELP + " (bundle-check: the bundle's value)")
        plan_mod.add_tuning_argument(command)
        command.add_argument("--session-groups", choices=SESSION_GROUPS, default="tp",
                             help="SIRCL_GROUPS: vLLM groups that get a SIRCL session (dcp needs the session's "
                                  "scatter collectives; bundle-check: the bundle's value)")
        command.add_argument("--dcp-size", dest="dcp_size", type=int, default=1, metavar="N",
                             help="vLLM's decode-context parallelism, for the group map, the sessions and their "
                                  "tuning tables (default 1; bundle-check: the bundle's value)")
        if name == "bundle-check":
            command.add_argument("--container", action="append", required=True, metavar="NAME",
                                 help="each rank's container: N=NAME for Spark N (repeatable), or one name with "
                                      "{rank} or {position}")
            continue
        command.add_argument("--image", help="serving image the stage step builds the native library in "
                                             "(default: the site's image)")
        command.add_argument("--capacity", type=int, default=plan_mod.DEFAULT_CAPACITY)
        command.add_argument("--dispatch", type=int, help="dispatch ceiling (default: the capacity)")
        command.add_argument("--gather", type=int, default=plan_mod.DEFAULT_GATHER)
        plan_mod.add_oneshot_argument(command)
        command.add_argument("--large-allreduce", choices=plan_mod.LARGE_MODES, default="auto")
        command.add_argument("--startup-wait", type=float, default=plan_mod.DEFAULT_STARTUP_WAIT_S, metavar="SECONDS")
        command.add_argument("--serving-wait", type=float, default=plan_mod.DEFAULT_SERVING_WAIT_S, metavar="SECONDS")
        command.add_argument("--spin-limit", type=int)
        command.add_argument("--gid-index", type=int)
        plan_mod.add_schedule_arguments(command)
        plan_mod.add_minimum_arguments(command)
        plan_mod.add_link_arguments(command)
        plan_mod.add_reasoning_argument(command)
        command.add_argument("--repository", help="with --reasoning-effort or --thinking-behaviour: the SparkRing "
                                                  "checkout whose profiles/thinking.json gives the checkpoint's "
                                                  "effort levels")
        plan_mod.add_checkpoint_arguments(command, checkpoint_flags=("--checkpoint", "--checkpoint-id"))
        plan_mod.add_vllm_arguments(command)
        plan_mod.add_b12x_cache_argument(command)
        command.add_argument("--fused-norm", choices=("off", "on"), default="off",
                             help="SIRCL_FUSED_NORM: on runs vLLM's post-all-reduce RMSNorm helper as one fused "
                                  "SIRCL kernel where it is bit-identical (research-only; setup refuses it "
                                  "where it cannot be)")
        command.add_argument("--env", action="append", metavar="KEY=VALUE",
                             help="add one variable to every rank's environment (repeatable)")
        command.add_argument("--text", action="store_true",
                             help="print the groups vLLM builds and what carries them instead of the JSON document")
        command.add_argument("--stage", action="store_true",
                             help="also stage the tree and build the native library on every Spark (MUTATES HOST)")


def plan_from_args(args: argparse.Namespace) -> BundlePlan:
    site = ServeSite.load(args.site)
    positions = plan_mod.parse_positions(args.positions, site.site.size)
    overlay, overlays = plan_mod.split_paths(args.overlay or [], positions, "--overlay")
    options = BundleOptions(positions=positions, run_id=args.run_id, overlay=overlay, overlays=overlays,
                            large_blocks=args.large_blocks, require_no_nccl=args.require_no_nccl,
                            nccl_debug=args.nccl_debug, nccl_mode=args.nccl,
                            tuning_tables=tuple(args.tuning_table or ()), session_groups=args.session_groups,
                            dcp_size=args.dcp_size)
    if args.command == "bundle":
        options = dataclasses.replace(
            options, image=args.image, capacity=args.capacity,
            dispatch=args.dispatch, oneshot_max=args.oneshot_max, gather=args.gather, link_slots=args.link_slots,
            ring_gather_stagger=args.ring_gather_stagger,
            large_allreduce=args.large_allreduce,
            startup_wait=args.startup_wait, serving_wait=args.serving_wait, spin_limit=args.spin_limit,
            gid_index=args.gid_index, fused_norm=args.fused_norm == "on",
            large_schedule=args.large_schedule, gather_schedule=args.gather_schedule,
            scatter_schedule=args.scatter_schedule, ring_min=args.ring_min, chain_min=args.chain_min,
            link_sizes=plan_mod.link_values(args), reasoning_effort=args.reasoning_effort,
            repository=args.repository, checkpoint=args.checkpoint_id, thinking_behaviour=args.thinking_behaviour,
            extra_env=plan_mod.parse_env(args.env or []), vllm_edits=plan_mod.edits_from_args(args),
            b12x_cache_dir=args.b12x_cache_dir)
    return build_bundle(site, options, staged_digest=staging.staged_tree().digest, library=staging.library_name())


def _to_stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def main(args: argparse.Namespace, *, run: Runner, out: Callable[[str], None] = print,
         progress: Callable[[str], None] = _to_stderr) -> int:
    """``bundle``: the JSON document on ``out`` (standard output) and stage progress on ``progress``
    (standard error); ``bundle-check``: its report on ``out``."""
    try:
        plan = plan_from_args(args)
        containers = container_names(args.container, plan) if args.command == "bundle-check" else {}
    except (SiteError, ServePlanError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.command == "bundle-check":
        return check(plan, containers, run, out)
    out(plan.render_text() if args.text else json.dumps(plan.to_json(str(args.site)), indent=1))
    if args.stage:
        return stage(plan, staging.staged_tree(), run, progress)
    return 0
