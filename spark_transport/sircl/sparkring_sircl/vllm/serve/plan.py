"""The launch plan: the profile's containers with SIRCL in front of every collective.

:func:`build_plan` turns a site (:mod:`.sitefile`), a serving profile
(:mod:`.profile`) and the operator's options into one :class:`RankLaunch` per
tensor-parallel rank of one serving instance: the Spark, its Docker command,
the container name, mounts, environment, labels and the exact ``docker run``
command. :func:`build_plans` plans several instances on disjoint groups of
Sparks at once, each with its own run id and ports. Nothing is contacted; the
CPU tests compare plans directly.

A group is consecutive Sparks along the ring, in rank order, for example
positions 0,1,2,3 or 7,0,1,2 (across the ring's 7-0 cable). The group's
cabling decides what NCCL may do (:mod:`sparkring_sircl.vllm.fabric`): on a
pair every pair of ranks shares a cable and NCCL may run; on four Sparks of
the ring ranks 0 and 3 share no cable and NCCL may not run at all.

What changes against the profile's rendered container, and why:

- addresses and interface: ``--master-addr``, ``VLLM_HOST_IP``,
  ``GLOO_SOCKET_IFNAME`` and ``NCCL_SOCKET_IFNAME`` come from the site file;
- ports: ``--port`` is the instance's API port (also in the health check),
  ``--master-port`` the instance's torch.distributed port;
- SIRCL: ``PYTHONPATH`` names the staged package tree, ``VLLM_PLUGINS`` gains
  ``sircl`` (platform and general plugin), and the ``SIRCL_*`` settings place
  the group on the site's ring (:mod:`sparkring_sircl.vllm.settings`).
  ``SIRCL_NATIVE_LIBRARY`` names the library the stage step built from the
  staged source, so a missing build fails at session setup instead of
  compiling on a read-only mount;
- flag waits: ``SIRCL_STARTUP_WAIT_S`` and ``SIRCL_SERVING_WAIT_S`` set how
  long a session waits for a late peer in the startup regime (setup, warm-up,
  graph capture, profiling, sleep and wake-up) and in the serving regime (from
  the first step after warm-up); ``SIRCL_SPIN_LIMIT`` is set only when
  ``--spin-limit`` is given, and time-limited waits ignore it;
- SparkRing's own transports are disabled: vLLM's RoCE all-reduce slot
  (``VLLM_ENABLE_ROCE_ALLREDUCE=0``), the RoCEnante bundle selection
  (``SPARKRING_TRANSPORT_PROFILE`` and its manifest digest empty; an empty
  profile makes the selector hook return without selecting) and SIRCL's
  four-rank adapter (``SPARK_TP4_ENABLED=0``, which turns off every
  four-rank startup hook, with ``VLLM_SPARK_TP4_MODE`` and
  ``VLLM_SPARK_TP4_VOCAB_MODE`` empty);
- GLM-5.3-Flash's mHC prefill sharding (``VLLM_GLM53_MHC_PREFILL_SHARD``)
  keeps the profile's value. On a group NCCL may not run, SIRCL's communicator
  carries its reduce-scatters and all-gathers through the ``mhc_prefill_shard``
  shim (:mod:`sparkring_sircl.vllm.mhc`); ``--mhc-prefill-shard off`` sets it
  to 0;
- on a group NCCL may all-reduce (a pair): ``NCCL_IB_HCA`` names the RDMA
  devices that face the rank's peers in the derived route map. The profile's
  value names the devices of a two-Spark pair cabled port to port; on the ring
  the second rank faces its partner through its port 1;
- mounts: the model directory (read-only) and the cache directory of each
  Spark, the staged package tree (read-only), the native build cache
  (read-only) and a per-run directory for SIRCL's receipts. Bind mounts use
  ``--mount``, which fails when a source directory is missing instead of
  creating an empty one;
- seccomp: the profile's loader policy, staged under the site's remote
  directory (Docker reads it on the Spark);
- operator settings: ``--env KEY=VALUE`` adds or replaces one variable on
  every rank (profiler switches, experiment settings, vLLM and B12X
  runtime switches); it may not name a variable the launcher owns
  (:func:`check_extra_env`);
- vLLM arguments: ``--vllm-arg FLAG=VALUE`` (or ``FLAG`` for a switch)
  replaces or adds one argument of every rank's command,
  ``--drop-vllm-arg FLAG`` removes one, and ``--speculative-set KEY=VALUE``
  sets one field of the JSON object ``--speculative-config`` gives. Arguments
  the launcher sets per rank or reads for its own checks are refused
  (:func:`owned_argument`); the recipe checks below run on the edited
  arguments;
- a source overlay: ``--overlay HOSTDIR`` mounts a host directory read-only at
  ``/opt/sparkring-overlay`` on every rank and puts it first on
  ``PYTHONPATH``, before the staged package tree, so the ``vllm`` and ``b12x``
  packages it holds replace the image's (the serving entrypoint puts only its
  add-on and toolchain directories before ``PYTHONPATH``);
- the served checkpoint: ``--checkpoint-id REPOSITORY@REVISION`` names a
  checkpoint the profile does not pin, for the plan record and the thinking
  lookup, and ``--thinking-behaviour NAME`` names its behaviour in the
  repository's ``profiles/thinking.json`` when that record does not list it;
- B12X's compile cache: ``--b12x-cache-dir CONTAINERPATH`` sets
  ``B12X_COMPILE_CACHE_DIR``, which the profiles place under the ``/cache``
  mount;
- names and labels: ``sircl-serve-<run id>-r<rank>`` and
  ``sircl-serve=<run id>``, so the stop command removes exactly these
  containers. No ``io.sparkring.*`` label is set, so SparkRing's installer
  does not mistake them for its own deployment.

Every other ``B12X_*``, ``CUTE_*`` and cache variable stays as the profile
sets it unless ``--env`` names it: B12X keys its compiled kernels on them, so
changing one recompiles every kernel at startup.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ... import latency_model, protocol
from ... import tuning as tuning_mod
from ... import routes as routes_mod
from ...ring.site import Site
from ...routes import Layout as SessionLayout
from .. import fabric, guard, pins
from ..fabric import NcclPolicy
from . import profile as profile_mod
from .profile import HEADLESS, ServingProfile
from .sitefile import ServeSite

LABEL = "sircl-serve"
CONTAINER_PREFIX = "sircl-serve"
CONTAINER_ROOT = "/sircl"
SOURCE_TARGET = f"{CONTAINER_ROOT}/src"
BUILD_TARGET = f"{CONTAINER_ROOT}/build-cache"
RUN_TARGET = f"{CONTAINER_ROOT}/run"
RECEIPT_TARGET = f"{RUN_TARGET}/receipts"
MODEL_TARGET = "/models/target"
CACHE_TARGET = "/cache"
# A source overlay (--overlay): a host directory of Python packages, mounted read-only and first on PYTHONPATH.
# Preflight requires the files of OVERLAY_FILES in it and compares the tree hash its OVERLAY.json states (the
# first of OVERLAY_TREE_KEYS that is a top-level string; without one, the file's own SHA-256) across Sparks.
OVERLAY_TARGET = "/opt/sparkring-overlay"
OVERLAY_RECORD = "OVERLAY.json"
OVERLAY_FILES = ("vllm/__init__.py", "b12x/__init__.py", OVERLAY_RECORD)
OVERLAY_PACKAGES = ("vllm", "b12x")
OVERLAY_TREE_KEYS = ("tree_sha256", "tree_hash", "tree_digest", "tree", "sha256", "digest")
# B12X's compile cache directory in the containers; the profiles place it under /cache (--cache-path).
B12X_CACHE_VARIABLE = "B12X_COMPILE_CACHE_DIR"
SPECULATIVE_FLAG = "--speculative-config"
SESSION_MODULE = "sparkring_sircl.oneshot"
# Session sizes of the four-Spark path configuration of the ring harness
# (ranks 0-3 of the eight-Spark ring, image aba309e4610c): every all-reduce up
# to 131,072 bytes and every all-gather up to 155,648 bytes was bit-exact with
# no RDMA error counter change.
DEFAULT_CAPACITY = 131072
DEFAULT_GATHER = 155648
# The sessions' auto all-reduce runs a message of at most SIRCL_ONESHOT_MAX_BYTES one-shot and a larger one
# two-shot (when the session has the two-shot all-reduce). Unset, a session with a layout and the two-shot
# all-reduce takes the latency model's limit for its layout, lane count and posting order
# (latency_model.oneshot_limit, at most DEFAULT_ONESHOT_MAX_BYTES and the capacity), and any other session
# DEFAULT_ONESHOT_MAX_BYTES (oneshot/runtime.py, _oneshot_limit). --oneshot-max sets it, up to the dispatch
# ceiling. Every session the launcher builds has a layout.
ONESHOT_VARIABLE = "SIRCL_ONESHOT_MAX_BYTES"
DEFAULT_ONESHOT_MAX_BYTES = 131072
# SIRCL_POST_ORDER unset: a session with a layout posts its lanes to the peers with the most relays first
# (oneshot/runtime.py; sparkring_sircl.posting). The launcher leaves the variable unset.
DEFAULT_POST_ORDER = "farthest"
POST_ORDER_TEXT = ("lanes posted to the peers with the most relays first (SIRCL_POST_ORDER unset: the "
                   "session's order on a layout)")
ONESHOT_MEANING = ("the largest all-reduce the sessions' auto algorithm runs one-shot, larger ones running "
                   "two-shot: a multiple of 16 bytes up to the dispatch ceiling, 0 for two-shot only")
# The largest grid of the sessions' two-shot and large-message launches (two-shot all-reduce, large all-reduce
# pieces, tiled all-gathers, scatter ops): SIRCL_LARGE_BLOCKS, a power of two from 1 to MAX_LARGE_BLOCKS
# (the check in oneshot/runtime.py); unset, DEFAULT_LARGE_BLOCKS there. --large-blocks sets it.
LARGE_BLOCKS_VARIABLE = "SIRCL_LARGE_BLOCKS"
DEFAULT_LARGE_BLOCKS = 32
MAX_LARGE_BLOCKS = 1024
LARGE_BLOCKS_MEANING = ("the largest grid, in blocks, of the sessions' two-shot and large-message launches "
                        f"(two-shot all-reduces among them): a power of two from 1 to {MAX_LARGE_BLOCKS}")
# Flag-wait limits of the session's two regimes (the core's defaults) and the
# largest limit its command ring holds (32-bit microseconds).
DEFAULT_STARTUP_WAIT_S = 600.0
DEFAULT_SERVING_WAIT_S = 20.0
MAX_WAIT_S = 0xFFFFFFFF / 1e6
DEFAULT_API_PORT = 8017
# One default for serve and bundle: never, so NCCL carries no collective of a multi-rank group. --nccl auto,
# the opt-in, lets NCCL run where the cabling allows it (every collective on a pair, NCCL's ring algorithm on
# a whole ring, nothing on a path) as the adapter's rules decide; topology is another name for auto. A tuning
# table chooses only SIRCL's settings in every mode: its NCCL marks are measurements and route no call.
NCCL_MODES = ("never", "auto", "topology")
DEFAULT_NCCL_MODE = "never"
NCCL_MODE_HELP = ("SIRCL_NCCL: never (default) makes SIRCL carry every collective; auto lets NCCL run where "
                  "the cabling allows it: every collective on a pair, NCCL's ring algorithm on a whole "
                  "ring, nothing on a path; topology is another name for auto")


def nccl_mode_value(text: str) -> str:
    """An ``--nccl`` value with ``topology`` spelled ``auto``."""
    return "auto" if text == "topology" else text
LARGE_MODES = ("auto", "sircl", "nccl")
MHC_MODES = ("profile", "off")
# Session schedules of all_reduce_large, all_gather_large and reduce_scatter; unset keeps the session's
# default (auto, auto and pieces). auto never selects the ring.
SCHEDULES = ("auto", "chain", "ring", "pieces")
SCHEDULE_VARIABLES = (("large_schedule", "SIRCL_LARGE_SCHEDULE"), ("gather_schedule", "SIRCL_GATHER_SCHEDULE"),
                      ("scatter_schedule", "SIRCL_SCATTER_SCHEDULE"))
# Every collective with a chain or ring schedule has a minimum of its own (MIN_COLLECTIVES of
# oneshot/runtime.py), a size of its all-reduce message, all-gather output or reduce-scatter input: auto runs
# a chain op from its chain minimum, a ring schedule a ring op from its ring minimum, and a smaller one runs as
# auto does. Unset, the minimums are DEFAULT_CHAIN_MINS and DEFAULT_RING_MINS there, the sizes from which the
# chain and the ring beat two-shot pieces, tiles and scatter ops on Sparks 0-3. SIRCL_CHAIN_MIN_BYTES
# (--chain-min) and SIRCL_RING_MIN_BYTES (--ring-min) set one size for all three.
# The session's schedules when their variables are unset (oneshot/runtime.py, _configure).
DEFAULT_SCHEDULES = {"large_schedule": "auto", "gather_schedule": "auto", "scatter_schedule": "pieces"}
SCHEDULE_LABELS = {"large_schedule": "large all-reduces", "gather_schedule": "large all-gathers",
                   "scatter_schedule": "reduce-scatters"}
MIN_COLLECTIVES = ("reduce", "gather", "scatter")
MIN_MEASURES = {"reduce": "all-reduce message", "gather": "all-gather output", "scatter": "reduce-scatter input"}
DEFAULT_CHAIN_MINS = {"reduce": 8 << 20, "gather": 8 << 20, "scatter": 4 << 20}
DEFAULT_RING_MINS = {"reduce": 4 << 20, "gather": 8 << 20, "scatter": 4 << 20}
CHAIN_MIN_VARIABLE = "SIRCL_CHAIN_MIN_BYTES"
CHAIN_MIN_MEANING = ("the smallest collective auto runs as a chain op where the group's ranks form a chain of "
                     "cable neighbors (an all-reduce's message, an all-gather's output, a reduce-scatter's "
                     "input), one size for all three: a non-negative byte count, 0 for every size")
RING_MIN_VARIABLE = "SIRCL_RING_MIN_BYTES"
RING_MIN_MEANING = ("the smallest collective a ring schedule runs as a ring op (an all-reduce's message, an "
                    "all-gather's output, a reduce-scatter's input), one size for all three, smaller ones running "
                    "as under auto: a non-negative byte count, 0 for every size")
# The session's link slot and piece when unset (DEFAULT_LINK_SLOT_BYTES and DEFAULT_LINK_CHUNK_BYTES of
# oneshot/runtime.py; an unset piece is the smaller of its default and the slot), the largest slot it
# chooses for configured pieces without SIRCL_LINK_SLOT_BYTES (MAX_AUTO_LINK_SLOT_BYTES there), and the
# slot alignment of its link area (SLOT_ALIGNMENT of protocol.py).
DEFAULT_LINK_SLOT_BYTES = 512 << 10
DEFAULT_LINK_CHUNK_BYTES = 512 << 10
MAX_AUTO_LINK_SLOT_BYTES = 1 << 20
LINK_SLOT_ALIGNMENT = 4096
# Receive and own slots per link (protocol.default_link_slots of the session's ranks, at least
# DEFAULT_LINK_SLOTS as in oneshot/runtime.py; 2 to protocol.LINK_MAX_SLOTS), and
# the staggers of the ring reduce-scatter and of the ring all-gather (also the all-gather part of the ring
# all-reduce) the session chooses when its slots hold them (DEFAULT_RING_STAGGER and
# DEFAULT_RING_GATHER_STAGGER there; otherwise 0). A stagger of s rounds needs s x (ranks - 1) + 2 link
# slots (protocol.ring_stagger_slots).
LINK_SLOTS_VARIABLE = "SIRCL_LINK_SLOTS"
DEFAULT_LINK_SLOTS = 8
DEFAULT_RING_STAGGER = 1
RING_GATHER_STAGGER_VARIABLE = "SIRCL_RING_GATHER_STAGGER"
DEFAULT_RING_GATHER_STAGGER = 1
RING_GATHER_STAGGER_MEANING = ("rounds the tensor-parallel session's ring all-gather (also the all-gather part "
                               "of its ring all-reduce) lets a forward leave after the piece it passes on "
                               f"arrived: 0 to {protocol.MAX_RING_STAGGER}, needing rounds x (ranks - 1) + 2 "
                               "link slots")
LINK_SLOTS_MEANING = ("the receive and own slots of each link of the tensor-parallel session's chain and ring "
                      f"ops: 2 to {protocol.LINK_MAX_SLOTS}")
MAX_LINK_SLOT_BYTES = 1 << 31


@dataclasses.dataclass(frozen=True)
class LinkSize:
    """A size of the sessions' link collectives (chain and ring ops) that the launcher sets when asked."""

    attribute: str      # key of ``Options.link_sizes`` and ``BundleOptions.link_sizes``; argparse destination
    variable: str       # the session's environment variable
    label: str          # its name in plan text
    piece: bool         # a piece, at most the link slot (else the link slot itself)
    meaning: str
    default: str        # the session's value when the variable is unset

    @property
    def flag(self) -> str:
        return "--" + self.attribute.replace("_", "-")


# Link sizes the launcher sets (plan, preflight, stage, start, wait, status, check, logs, collect, stop and
# bundle options): the slot, the link piece, and the pieces of the three link collectives, each of which
# takes the link piece when unset. Without --link-slot the sessions' slot is 524,288 bytes, or the largest
# piece set rounded up to 4,096 bytes when that lies above 524,288 and at most 1,048,576 bytes.
LINK_SIZES = (
    LinkSize("link_slot", "SIRCL_LINK_SLOT_BYTES", "link slot", False,
             "the sessions' link slot, the largest piece of a chain or ring collective: a multiple of "
             f"{LINK_SLOT_ALIGNMENT} bytes up to 2 GiB",
             f"{DEFAULT_LINK_SLOT_BYTES:,} B, or the largest piece set rounded up to {LINK_SLOT_ALIGNMENT:,} B "
             f"when that is at most {MAX_AUTO_LINK_SLOT_BYTES:,} B"),
    LinkSize("link_chunk", "SIRCL_LINK_CHUNK_BYTES", "link piece", True,
             "the piece of every chain or ring collective without a piece of its own: a multiple of 16 bytes "
             "up to the link slot", f"the smaller of {DEFAULT_LINK_CHUNK_BYTES:,} B and the link slot"),
    LinkSize("gather_link_chunk", "SIRCL_GATHER_LINK_CHUNK_BYTES", "all-gather link piece", True,
             "the piece of chain and ring all-gathers: a multiple of 16 bytes up to the link slot",
             "the link piece"),
    LinkSize("scatter_link_chunk", "SIRCL_SCATTER_LINK_CHUNK_BYTES", "reduce-scatter link piece", True,
             "the piece of chain and ring reduce-scatters: a multiple of 16 bytes up to the link slot",
             "the link piece"),
    LinkSize("reduce_link_chunk", "SIRCL_REDUCE_LINK_CHUNK_BYTES", "all-reduce link piece", True,
             "the piece of ring all-reduces: a multiple of 16 bytes up to the link slot", "the link piece"),
)
MHC_SHARD = "VLLM_GLM53_MHC_PREFILL_SHARD"
DISABLED_TRANSPORTS = {
    "VLLM_ENABLE_ROCE_ALLREDUCE": "0",
    "SPARKRING_TRANSPORT_PROFILE": "",
    "SPARKRING_TRANSPORT_MANIFEST_SHA256": "",
    "SPARK_TP4_ENABLED": "0",
    "VLLM_SPARK_TP4_MODE": "",
    "VLLM_SPARK_TP4_VOCAB_MODE": "",
}
# Variables the launcher owns; a profile that sets one is refused.
OWNED = ("SIRCL_LAYOUT", "SIRCL_PEER_ROUTES", "VLLM_DISABLE_PYNCCL", "SIRCL_LARGE_ALGORITHM",
         "SIRCL_ALLREDUCE_ALGORITHM")
REASONS = {
    "VLLM_HOST_IP": "the Spark's wired-LAN address (site file)",
    "GLOO_SOCKET_IFNAME": "the site's wired-LAN interface",
    "NCCL_SOCKET_IFNAME": "the site's wired-LAN interface",
    "VLLM_PLUGINS": "adds the sircl platform and general plugin",
    "PYTHONPATH": "the staged sparkring_sircl tree and its entry points",
    "VLLM_ENABLE_ROCE_ALLREDUCE": "no RoCEnante slot: SIRCL's communicator builds its own",
    "SPARKRING_TRANSPORT_PROFILE": "no RoCEnante bundle selection",
    "SPARKRING_TRANSPORT_MANIFEST_SHA256": "no RoCEnante bundle selection",
    "SPARK_TP4_ENABLED": "SIRCL's four-rank adapter stays off (it serves four-Spark cycles only)",
    "VLLM_SPARK_TP4_MODE": "four-rank all-reduce adapter off",
    "VLLM_SPARK_TP4_VOCAB_MODE": "four-rank vocabulary adapter off",
    MHC_SHARD: "mHC prefill sharding turned off (--mhc-prefill-shard off)",
    "SIRCL_STARTUP_WAIT_S": "flag-wait limit of the startup regime: setup, warm-up, graph capture, profiling, "
                            "sleep and wake-up",
    "SIRCL_SERVING_WAIT_S": "flag-wait limit of the serving regime, from the first step after warm-up",
    "SIRCL_SPIN_LIMIT": "--spin-limit: poll budget of waits without a time limit; time-limited flag waits "
                        "ignore it",
    "SIRCL_NATIVE_LIBRARY": "the library the stage step built from the staged source",
    "NCCL_IB_HCA": "NCCL runs on this group: the RDMA devices facing the rank's peers",
    "SIRCL_LARGE_SCHEDULE": "--large-schedule: the tensor-parallel session's schedule of all-reduces above the "
                            "dispatch ceiling",
    "SIRCL_GATHER_SCHEDULE": "--gather-schedule: the tensor-parallel session's schedule of all-gathers above the "
                             "gather op size",
    "SIRCL_SCATTER_SCHEDULE": "--scatter-schedule: the tensor-parallel session's schedule of reduce-scatters",
    **{size.variable: f"{size.flag}: {size.meaning}" for size in LINK_SIZES},
    LINK_SLOTS_VARIABLE: f"--link-slots: {LINK_SLOTS_MEANING}",
    RING_GATHER_STAGGER_VARIABLE: f"--ring-gather-stagger: {RING_GATHER_STAGGER_MEANING}",
    "SIRCL_TUNING_TABLE": "--tuning-table: the measured tuning tables staged in the run directory; each "
                          "session takes the one whose key matches its group shape, sizes and build",
    ONESHOT_VARIABLE: f"--oneshot-max: {ONESHOT_MEANING}",
    LARGE_BLOCKS_VARIABLE: f"--large-blocks: {LARGE_BLOCKS_MEANING}",
    CHAIN_MIN_VARIABLE: f"--chain-min: {CHAIN_MIN_MEANING}",
    RING_MIN_VARIABLE: f"--ring-min: {RING_MIN_MEANING}",
    B12X_CACHE_VARIABLE: "--b12x-cache-dir: the directory B12X compiles its kernels into",
    "NCCL_DEBUG": "--nccl-debug: NCCL logs every communicator it creates, which check --require-no-nccl scans for",
    "NCCL_DEBUG_SUBSYS": "--nccl-debug: NCCL's initialization subsystem only",
}
OVERLAY_PYTHONPATH_REASON = (f"the source overlay (--overlay, {OVERLAY_TARGET}) first, then the staged "
                             "sparkring_sircl tree and its entry points")
PER_RANK = {"VLLM_HOST_IP": "<each Spark's LAN address>",
            "NCCL_IB_HCA": "<per rank: the devices facing its peers>"}
# Variables --env may not set, with the option that controls each instead.
LAUNCHER_KEYS = {
    "VLLM_PLUGINS": "the launcher adds the sircl plugins",
    "PYTHONPATH": "the launcher names the staged package tree; --overlay puts a source overlay before it",
    B12X_CACHE_VARIABLE: "use --b12x-cache-dir, which checks the directory against the containers' mounts",
    "VLLM_HOST_IP": "the site file gives each Spark's address",
    "GLOO_SOCKET_IFNAME": "the site file gives the interface",
    "NCCL_SOCKET_IFNAME": "the site file gives the interface",
    "NCCL_IB_HCA": "the launcher derives each rank's devices",
    MHC_SHARD: "use --mhc-prefill-shard",
    **{key: "the launcher keeps SparkRing's own transports off" for key in DISABLED_TRANSPORTS},
}
ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# SIRCL_* variables with an option of their own, named when --env tries to set one.
OPTION_VARIABLES = {
    "SIRCL_ALLREDUCE_CAPACITY_BYTES": "--capacity",
    "SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES": "--dispatch",
    ONESHOT_VARIABLE: "--oneshot-max",
    LARGE_BLOCKS_VARIABLE: "--large-blocks",
    "SIRCL_ALLGATHER_MAX_BYTES": "--gather",
    "SIRCL_SPIN_LIMIT": "--spin-limit",
    "SIRCL_STARTUP_WAIT_S": "--startup-wait",
    "SIRCL_SERVING_WAIT_S": "--serving-wait",
    "SIRCL_NCCL": "--nccl",
    "SIRCL_LARGE_ALLREDUCE": "--large-allreduce",
    "SIRCL_GID_INDEX": "--gid-index",
    **{variable: "--" + attribute.replace("_", "-") for attribute, variable in SCHEDULE_VARIABLES},
    CHAIN_MIN_VARIABLE: "--chain-min",
    RING_MIN_VARIABLE: "--ring-min",
    **{size.variable: size.flag for size in LINK_SIZES},
    LINK_SLOTS_VARIABLE: "--link-slots",
    RING_GATHER_STAGGER_VARIABLE: "--ring-gather-stagger",
    "SIRCL_TUNING_TABLE": "--tuning-table",
}


class ServePlanError(ValueError):
    """The requested launch cannot be planned."""


@dataclasses.dataclass(frozen=True)
class Options:
    positions: tuple[int, ...]
    model_paths: Mapping[int, str] = dataclasses.field(default_factory=dict)  # --model-path N=PATH
    model_path: str | None = None            # --model-path PATH: every Spark without a closer value
    cache_paths: Mapping[int, str] = dataclasses.field(default_factory=dict)  # --cache-path N=PATH
    cache_path: str | None = None            # --cache-path PATH
    api_port: int = DEFAULT_API_PORT
    master_port: int | None = None           # None: the profile's
    run_id: str | None = None                # None: tp<N>-<first>-<last>
    capacity: int = DEFAULT_CAPACITY
    dispatch: int | None = None              # None: the capacity
    oneshot_max: int | None = None           # SIRCL_ONESHOT_MAX_BYTES; None: the session's default
    large_blocks: int | None = None          # SIRCL_LARGE_BLOCKS; None: the session's default
    reasoning_effort: str | None = None      # --reasoning-effort; None: the chat template's default
    gather: int = DEFAULT_GATHER
    spin_limit: int | None = None            # SIRCL_SPIN_LIMIT; None: unset (the session's poll budget)
    startup_wait: float = DEFAULT_STARTUP_WAIT_S   # SIRCL_STARTUP_WAIT_S
    serving_wait: float = DEFAULT_SERVING_WAIT_S   # SIRCL_SERVING_WAIT_S
    gid_index: int | None = None             # None: the site's, else the profile's NCCL_IB_GID_INDEX
    nccl_mode: str = DEFAULT_NCCL_MODE       # SIRCL_NCCL
    dcp_size: int = 1                        # --dcp-size: vLLM's decode-context parallelism, a session per DCP group
    large_allreduce: str = "auto"            # SIRCL_LARGE_ALLREDUCE
    mhc_prefill_shard: str = "profile"       # VLLM_GLM53_MHC_PREFILL_SHARD: the profile's value, or off
    large_schedule: str | None = None        # SIRCL_LARGE_SCHEDULE; None: the session's default
    gather_schedule: str | None = None       # SIRCL_GATHER_SCHEDULE
    scatter_schedule: str | None = None      # SIRCL_SCATTER_SCHEDULE
    chain_min: int | None = None             # SIRCL_CHAIN_MIN_BYTES; None: the session's defaults
    ring_min: int | None = None              # SIRCL_RING_MIN_BYTES; None: the session's defaults
    link_sizes: Mapping[str, int] = dataclasses.field(default_factory=dict)  # LinkSize.attribute -> bytes
    link_slots: int | None = None            # SIRCL_LINK_SLOTS; None: the session's default
    ring_gather_stagger: int | None = None   # SIRCL_RING_GATHER_STAGGER; None: the session's default
    tuning_tables: tuple[str, ...] = ()      # --tuning-table PATH (SIRCL_TUNING_TABLE)
    extra_env: Mapping[str, str] = dataclasses.field(default_factory=dict)  # --env KEY=VALUE, every rank
    overlay: str | None = None               # --overlay HOSTDIR: every Spark without its own
    overlays: Mapping[int, str] = dataclasses.field(default_factory=dict)   # --overlay N=HOSTDIR
    vllm_edits: VllmEdits | None = None      # --vllm-arg, --drop-vllm-arg, --speculative-set
    checkpoint_id: str | None = None         # --checkpoint-id REPOSITORY@REVISION; None: the profile's
    thinking_behaviour: str | None = None    # --thinking-behaviour NAME of profiles/thinking.json
    b12x_cache_dir: str | None = None        # --b12x-cache-dir: B12X_COMPILE_CACHE_DIR; None: the profile's
    nccl_debug: bool = False                 # --nccl-debug: NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT
    require_no_nccl: bool = False            # --require-no-nccl


@dataclasses.dataclass(frozen=True)
class VllmEdits:
    """Changes to the vLLM arguments of every rank: ``--vllm-arg``, ``--drop-vllm-arg``, ``--speculative-set``."""

    set: tuple[tuple[str, str | None], ...] = ()      # (flag, value); None adds or keeps a switch
    drop: tuple[str, ...] = ()
    speculative: tuple[tuple[str, Any], ...] = ()     # (field of --speculative-config, JSON value)

    def __bool__(self) -> bool:
        return bool(self.set or self.drop or self.speculative)


@dataclasses.dataclass(frozen=True)
class ArgumentChange:
    """One vLLM argument of every rank's command against the profile's: the words after the flag (``()`` for
    a switch), None where the command does not give it."""

    flag: str
    before: tuple[str, ...] | None
    after: tuple[str, ...] | None
    option: str

    def describe(self) -> str:
        def words(value: tuple[str, ...] | None, absent: str) -> str:
            return absent if value is None else " ".join((self.flag, *value))

        if self.flag == SPECULATIVE_FLAG and self.before and self.after:
            fields = _json_field_changes(self.before[0], self.after[0])
            if fields is not None:
                return f"{self.flag}: {fields}  ({self.option})"
        if self.before == self.after:
            return f"{words(self.after, 'unset')}: as the profile gives it  ({self.option})"
        return f"{words(self.before, 'unset')} -> {words(self.after, 'removed')}  ({self.option})"


def _json_field_changes(before: str, after: str) -> str | None:
    """``key old -> new`` for every field two JSON objects differ in, or None when either is not an object."""
    try:
        old, new = json.loads(before), json.loads(after)
    except ValueError:
        return None
    if not isinstance(old, dict) or not isinstance(new, dict):
        return None
    parts = [f"{key} {json.dumps(old[key]) if key in old else 'unset'} -> "
             f"{json.dumps(new[key]) if key in new else 'removed'}"
             for key in sorted(set(old) | set(new)) if old.get(key, ...) != new.get(key, ...)]
    return "; ".join(parts) or "unchanged"


@dataclasses.dataclass(frozen=True)
class Mount:
    source: str
    target: str
    read_only: bool

    def option(self) -> str:
        return f"type=bind,src={self.source},dst={self.target}" + (",readonly" if self.read_only else "")


@dataclasses.dataclass(frozen=True)
class RankLaunch:
    rank: int
    position: int
    host: str
    ssh: str
    lan_address: str
    docker: str
    container: str
    image: str
    entrypoint: tuple[str, ...]
    command: tuple[str, ...]
    environment: dict[str, str]
    mounts: tuple[Mount, ...]
    labels: dict[str, str]
    run_options: tuple[str, ...]           # docker run options other than name, labels, mounts and env
    health: tuple[str, ...] | None
    directories: tuple[str, ...]           # created on the Spark before the container starts
    model_source: str = "given"            # where the model directory comes from
    sudo: str = ""                         # prefix of host-side file operations ("" runs them plainly)
    sudo_source: str = "default"

    def mount(self, target: str) -> Mount:
        return next(mount for mount in self.mounts if mount.target == target)

    def argv(self) -> list[str]:
        """``docker run`` arguments after the Docker command (the site's ``docker`` words)."""
        parts = ["run", "-d", "--name", self.container]
        for key, value in sorted(self.labels.items()):
            parts += ["--label", f"{key}={value}"]
        parts += ["--entrypoint", self.entrypoint[0], *self.run_options]
        for mount in self.mounts:
            parts += ["--mount", mount.option()]
        for key, value in sorted(self.environment.items()):
            parts += ["--env", f"{key}={value}"]
        return parts + [self.image, *self.entrypoint[1:], *self.command]

    def shell(self) -> str:
        """The full command for the Spark's shell: Docker command words, then the quoted arguments."""
        return " ".join([*(shlex.quote(word) for word in self.docker.split()),
                         *(shlex.quote(part) for part in self.argv())])


@dataclasses.dataclass(frozen=True)
class EnvironmentChange:
    name: str
    before: str | None
    after: str | None
    reason: str


@dataclasses.dataclass(frozen=True)
class ServePlan:
    run_id: str
    profile: ServingProfile
    site: Site
    positions: tuple[int, ...]
    group: fabric.GroupTopology
    staged_digest: str
    library: str
    api_port: int
    master_port: int
    capacity: int
    dispatch: int
    gather: int
    spin_limit: int | None
    ranks: tuple[RankLaunch, ...]
    changes: tuple[EnvironmentChange, ...]
    nccl_policy: NcclPolicy = NcclPolicy.NONE   # the policy the adapter applies (cabling, SIRCL_NCCL)
    nccl_policy_reason: str = ""
    nccl_mode: str = DEFAULT_NCCL_MODE
    large_allreduce: str = "auto"
    mhc_prefill_shard: bool = False          # mHC prefill row ownership runs (every rank's setting)
    startup_wait: float = DEFAULT_STARTUP_WAIT_S
    serving_wait: float = DEFAULT_SERVING_WAIT_S
    link_sizes: Mapping[str, int] = dataclasses.field(default_factory=dict)   # the LinkSize values set
    link_slots: int | None = None            # --link-slots; None: the session's default
    ring_gather_stagger: int | None = None   # --ring-gather-stagger; None: the session's default
    schedules: Mapping[str, str] = dataclasses.field(default_factory=dict)   # the schedules set, by attribute
    tuning: TuningPlan = dataclasses.field(default_factory=lambda: TuningPlan())   # --tuning-table
    oneshot_max: int | None = None           # --oneshot-max; None: the session's default
    large_blocks: int | None = None          # --large-blocks; None: the session's default
    ring_min: int | None = None              # --ring-min; None: the session's defaults
    chain_min: int | None = None             # --chain-min; None: the session's defaults
    reasoning_effort: str | None = None      # --reasoning-effort; None: the chat template's default
    chat_template_defaults: Mapping[str, Any] | None = None   # what it adds to rank 0's command
    argument_changes: tuple[ArgumentChange, ...] = ()          # --vllm-arg, --drop-vllm-arg, --speculative-set
    vllm_arguments: tuple[str, ...] = ()     # the recipe's vLLM arguments after them; () before any edit
    overlay: Mapping[int, str] = dataclasses.field(default_factory=dict)   # position -> --overlay HOSTDIR
    checkpoint_id: str | None = None         # --checkpoint-id; None: the profile's checkpoint
    thinking: profile_mod.ThinkingBehaviour | None = None      # the served checkpoint's thinking behaviour
    thinking_source: str = ""                # where it comes from: profiles/thinking.json or --thinking-behaviour
    b12x_cache_dir: str | None = None        # --b12x-cache-dir; None: the profile's B12X_COMPILE_CACHE_DIR
    nccl_debug: bool = False                 # NCCL logs its communicators (--nccl-debug, --require-no-nccl)
    require_no_nccl: bool = False            # --require-no-nccl
    dcp_size: int = 1                        # --dcp-size: decode-context parallelism, a SIRCL session per group

    @property
    def nccl_allowed(self) -> bool:
        """NCCL may run some collective of the tensor-parallel group."""
        return self.nccl_policy is not NcclPolicy.NONE

    @property
    def derived_oneshot_max(self) -> int:
        """The one-shot limit the tensor-parallel sessions derive when --oneshot-max is not given."""
        return derived_oneshot_max(self.group, self.capacity)

    def sessions(self) -> list["SessionSettings"]:
        """The instance's SIRCL sessions as they will run: the tensor-parallel group's and, with ``--dcp-size``
        above 1, one per decode-context-parallel group (with the session defaults, as SIRCL's adapter builds
        them)."""
        rows = [session_settings(self.group, name="tp", groups=f"one group of ranks 0-{len(self.ranks) - 1}",
                                 scoped=True, schedules=self.schedules, link_sizes=self.link_sizes,
                                 link_slots=self.link_slots, chain_min=self.chain_min, ring_min=self.ring_min,
                                 ring_gather_stagger=self.ring_gather_stagger,
                                 table_settings=self.tuning.settings_of("tp"))]
        if self.dcp_size > 1:
            subgroups = dcp_groups(self.group.layout, self.positions, self.dcp_size)
            rows.append(session_settings(subgroups[0], name="dcp", scoped=False,
                                         groups=dcp_groups_text(len(self.positions), self.dcp_size),
                                         table_settings=self.tuning.settings_of("dcp")))
        return rows

    def carriers(self) -> tuple[list[GroupCarrier], list[tuple[str, str]]]:
        groups = group_carriers(self.group.layout, self.positions, nccl_mode=self.nccl_mode,
                                environment=self.profile.ranks[0].environment,
                                dcp=recipe_dcp(self.recipe_arguments),
                                session_groups=SESSION_GROUPS_DCP if self.dcp_size > 1 else ("tp",))
        collectives = collective_carriers(nccl_allowed=self.nccl_allowed, large=self.large_allreduce,
                                          dispatch=self.dispatch, gather=self.gather)
        return groups, collectives

    @property
    def recipe_arguments(self) -> tuple[str, ...]:
        """The vLLM arguments every rank runs after the head of its command: the profile's recipe with the
        plan's argument edits."""
        return self.vllm_arguments or self.profile.recipe_arguments

    @property
    def max_num_batched_tokens(self) -> int:
        value = _recipe_value(self.recipe_arguments, "--max-num-batched-tokens")
        try:
            return int(value) if value is not None else self.profile.max_num_batched_tokens
        except ValueError:
            return self.profile.max_num_batched_tokens

    @property
    def checkpoint(self) -> str:
        """``REPOSITORY@REVISION`` of the served checkpoint."""
        return self.checkpoint_id or f"{self.profile.model_repository}@{self.profile.model_revision}"

    @property
    def foreign_checkpoint(self) -> bool:
        """The plan serves a checkpoint other than the profile's, which the profile's manifest does not describe."""
        return self.checkpoint != f"{self.profile.model_repository}@{self.profile.model_revision}"

    @property
    def b12x_cache(self) -> str | None:
        """``B12X_COMPILE_CACHE_DIR`` in the containers (every rank's), or None when unset."""
        return self.ranks[0].environment.get(B12X_CACHE_VARIABLE) if self.ranks else None

    @property
    def required_shims(self) -> tuple[str, ...]:
        """The pinned shims serving needs on this group: the decode-context-parallel output combine and its B12X
        transport with ``--dcp-size`` above 1, and prefill row ownership on a group without NCCL."""
        dcp = ("dcp_all_to_all", "dcp_b12x_transport") if self.dcp_size > 1 else ()
        if self.nccl_policy.allows("all_reduce"):
            return dcp
        return (dcp + (("mhc_prefill_shard",) if self.mhc_prefill_shard else ())
                + (("qwen_hc_prefill_shard",) if self.hc_prefill_mode == "shard" else ()))

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

    @property
    def seccomp_path(self) -> str:
        return seccomp_path(self.site, self.profile)

    @property
    def api_rank(self) -> RankLaunch:
        return self.ranks[0]

    @property
    def nccl_reason(self) -> str:
        return self.nccl_policy_reason or self.group.nccl_reason

    @property
    def large_on_nccl(self) -> bool:
        """Eager all-reduces above the dispatch ceiling go to NCCL (the planner's ``prefers_nccl``)."""
        return self.nccl_policy.allows("all_reduce") and self.large_allreduce != "sircl"

    @property
    def mhc_world(self) -> int | None:
        """The tensor-parallel size when SIRCL carries mHC row ownership for full prefill chunks, else None."""
        if not self.mhc_prefill_shard or self.nccl_policy.allows("all_reduce"):
            return None
        if self.max_num_batched_tokens not in MHC_CHUNK_ROWS or len(self.ranks) not in MHC_WORLDS:
            return None
        return len(self.ranks)

    @property
    def hc_prefill_mode(self) -> str | None:
        """Qwen3.8's hyper-connection prefill row ownership mode on every rank, or None when unset."""
        return self.ranks[0].environment.get(HC_PREFILL)

    def prefill(self, prompt_tokens: int = 16384) -> PrefillEstimate | None:
        """Prefill collectives and compute, or None when NCCL carries them or the model is unknown.

        The model shapes and compute rates describe the profiles' own
        checkpoints, so a plan serving another checkpoint has no estimate."""
        shape = MODEL_SHAPES.get(self.profile.model_repository)
        if shape is None or self.large_on_nccl or self.foreign_checkpoint:
            return None
        return prefill_estimate(prompt_tokens, chunk_tokens=self.max_num_batched_tokens,
                                hidden=shape.hidden, layers=shape.layers,
                                draft_allreduces=shape.draft_allreduces,
                                prefill_rate=PROFILE_PREFILL_RATES.get(self.profile.id),
                                mhc_world=self.mhc_world,
                                compute_per_token=PATH_COMPUTE_SECONDS_PER_TOKEN.get(self.profile.id))

    def to_json(self) -> dict:
        estimate = self.prefill()
        return {
            "schema": "sircl-serve-plan/v1",
            "run_id": self.run_id,
            "profile": self.profile.id,
            "profile_topology": self.profile.topology,
            "dcp_size": self.dcp_size,
            "release": self.profile.release,
            "image_id": self.profile.image_id,
            "image_reference": self.profile.image_reference,
            "profile_sources": self.profile.sources,
            "served_model_name": self.profile.served_model_name,
            "checkpoint": {
                "id": self.checkpoint,
                "source": "--checkpoint-id" if self.checkpoint_id else "profile",
                "profile_checkpoint": f"{self.profile.model_repository}@{self.profile.model_revision}",
                "manifest": not self.foreign_checkpoint,   # preflight checks the profile's file list and pins
            },
            "overlay": ({"target": OVERLAY_TARGET, "sources": {str(position): path for position, path
                                                               in sorted(self.overlay.items())},
                         "pythonpath": self.ranks[0].environment.get("PYTHONPATH"),
                         "required_files": list(OVERLAY_FILES)} if self.overlay else None),
            "vllm_arguments": {
                "changes": [dataclasses.asdict(change) for change in self.argument_changes],
                "recipe": list(self.recipe_arguments),
            },
            "nccl_free": {"required": self.require_no_nccl, "nccl_debug": self.nccl_debug,
                          "nccl_allowed": self.nccl_allowed, "log_patterns": list(NCCL_INIT_PATTERNS),
                          "library_patterns": list(NCCL_LIBRARY_PATTERNS)},
            "carriers": {"groups": [dataclasses.asdict(row) for row in self.carriers()[0]],
                         "collectives": [{"collective": name, "carrier": carrier}
                                         for name, carrier in self.carriers()[1]]},
            "sessions": [session.to_json() for session in self.sessions()],
            "tuning": self.tuning.to_json(self.run_dir),
            "b12x_cache": {"variable": B12X_CACHE_VARIABLE, "value": self.b12x_cache,
                           "source": ("--b12x-cache-dir" if self.b12x_cache_dir else
                                      "profile" if self.b12x_cache is not None else "unset"),
                           "mount": cache_mount(self.b12x_cache)},
            "positions": list(self.positions),
            "fabric": self.group.fabric.describe(),
            "nccl": self.nccl_policy.value,
            "nccl_reason": self.nccl_reason,
            "nccl_mode": self.nccl_mode,
            "large_allreduce": self.large_allreduce,
            "mhc_prefill_shard": self.mhc_prefill_shard,
            "serving": {
                "reasoning_effort": self.reasoning_effort,
                "default_chat_template_kwargs": (dict(self.chat_template_defaults)
                                                 if self.chat_template_defaults is not None else None),
                "thinking_behaviour": self.thinking.id if self.thinking else None,
                "template_level": self.thinking.level if self.thinking else None,
                "thinking_source": self.thinking_source or None,
            },
            "hc_prefill_mode": self.hc_prefill_mode,
            "route_maps": {str(rank): fabric.format_routes(self.group.route_map(rank))
                           for rank in range(len(self.positions))},
            "relays": relay_rows(self.group),
            "relay_factor": self.group.relay_factor(),
            "max_relays": self.group.max_relays(),
            "staged_digest": self.staged_digest,
            "library": self.library,
            "api_port": self.api_port,
            "master_port": self.master_port,
            "sizes": {"capacity": self.capacity, "dispatch": self.dispatch, "oneshot_max": self.oneshot_max,
                      "oneshot_max_derived": self.derived_oneshot_max, "post_order": DEFAULT_POST_ORDER,
                      "link_slots": self.link_slots, "ring_gather_stagger": self.ring_gather_stagger,
                      "large_blocks": self.large_blocks,
                      "gather": self.gather, "spin_limit": self.spin_limit, "chain_min": self.chain_min,
                      "ring_min": self.ring_min,
                      **{size.attribute: self.link_sizes.get(size.attribute) for size in LINK_SIZES}},
            "waits": {"startup_s": self.startup_wait, "serving_s": self.serving_wait},
            "remote": {"source": self.source_dir, "build": self.build_dir, "run": self.run_dir,
                       "seccomp": self.seccomp_path},
            "prefill_estimate_16k": dataclasses.asdict(estimate) if estimate else None,
            "changes": [dataclasses.asdict(change) for change in self.changes],
            "ranks": [{
                "rank": launch.rank, "position": launch.position, "host": launch.host, "ssh": launch.ssh,
                "lan_address": launch.lan_address, "docker": launch.docker, "container": launch.container,
                "model_source": launch.model_source, "sudo": launch.sudo, "sudo_source": launch.sudo_source,
                "mounts": [dataclasses.asdict(mount) for mount in launch.mounts],
                "directories": list(launch.directories),
                "environment": launch.environment, "labels": launch.labels, "argv": launch.argv(),
            } for launch in self.ranks],
        }


def seccomp_path(site: Site, profile: ServingProfile) -> str:
    return f"{site.remote_dir}/serve/loader-seccomp-{profile.seccomp_sha256[:16]}.json"


def default_run_id(positions: Sequence[int]) -> str:
    return f"tp{len(positions)}-{positions[0]}-{positions[-1]}"


def _run_id(text: str | None, positions: Sequence[int]) -> str:
    value = text or default_run_id(positions)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", value):
        raise ServePlanError(f"run id {value!r} must be 1-40 lowercase letters, digits and dashes")
    return value


def _path(value: str, what: str) -> str:
    if (not value.startswith("/") or "\0" in value or "," in value or "=" in value or re.search(r"\s", value)
            or ".." in value.split("/")):
        raise ServePlanError(f"{what} {value!r} must be an absolute path without commas, '=', spaces or '..'")
    return value.rstrip("/") or "/"


def _replace_flag(arguments: list[str], flag: str, value: str) -> None:
    positions = [index for index, item in enumerate(arguments) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ServePlanError(f"the profile command must hold {flag} exactly once with a value")
    arguments[positions[0] + 1] = value


def _run_options(container, seccomp: str, health: tuple[str, ...] | None) -> tuple[str, ...]:
    options = ["--platform", container.platform, "--pull", container.pull_policy,
               "--restart", container.restart, "--gpus", "all", "--network", container.network_mode,
               "--ipc", container.ipc_mode, "--ulimit", f"memlock={container.memlock}:{container.memlock}"]
    if container.memory is not None:
        options += ["--memory", str(container.memory)]
    if container.memory_swap is not None:
        options += ["--memory-swap", str(container.memory_swap)]
    options += ["--security-opt", f"seccomp={seccomp}"]
    if container.init:
        options += ["--init"]
    check = container.health
    if health and check is not None:
        options += ["--health-cmd", shlex.join(health), "--health-interval", check.interval,
                    "--health-timeout", check.timeout, "--health-start-period", check.start_period,
                    "--health-retries", str(check.retries)]
    else:
        options += ["--no-healthcheck"]
    for device in container.devices:
        options += ["--device", device]
    return tuple(options)


def resolve_model_paths(positions: Sequence[int], *, site_paths: Mapping[int, str],
                        per_spark: Mapping[int, str], shared: str | None) -> dict[int, tuple[str, str]]:
    """Each position's model directory and its source.

    ``--model-path N=PATH`` (``per_spark``) wins, then the site file's
    ``model_path`` of that Spark, then ``--model-path PATH`` (``shared``).
    """
    resolved, missing = {}, []
    for position in positions:
        if position in per_spark:
            resolved[position] = (per_spark[position], f"--model-path {position}=PATH")
        elif position in site_paths:
            resolved[position] = (site_paths[position], "site file model_path")
        elif shared is not None:
            resolved[position] = (shared, "--model-path")
        else:
            missing.append(position)
    if missing:
        raise ServePlanError(f"no model directory for Sparks {missing}: give model_path in their site entries, "
                             "--model-path PATH or --model-path N=PATH")
    return resolved


def rank_devices(group: fabric.GroupTopology, rank: int) -> tuple[str, ...]:
    """The RDMA devices of ``rank``'s lanes to every peer, in route-map order without repeats."""
    devices: list[str] = []
    for _, lanes in sorted(group.route_map(rank).items()):
        for device in lanes:
            if device not in devices:
                devices.append(device)
    return tuple(devices)


def nccl_hca_value(profile_value: str | None, devices: Sequence[str]) -> str:
    """``NCCL_IB_HCA`` naming ``devices``, in the style of the profile's value (``=`` prefix, ``:port``)."""
    text = (profile_value or "=").strip()
    exact = text.startswith("=")
    entries = [item for item in text.lstrip("=^").split(",") if item]
    with_port = any(":" in item for item in entries)
    return ("=" if exact else "") + ",".join(f"{device}:1" if with_port else device for device in devices)


def relay_rows(group: fabric.GroupTopology) -> list[dict]:
    """Every pair of ranks whose lanes cross relays: the relay positions along the lanes."""
    rows = []
    members = group.members
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            lanes = group.lanes(a, b)
            if not lanes or not lanes[0].relays:
                continue
            rows.append({"ranks": [a, b], "positions": [members[a], members[b]], "lanes": len(lanes),
                         "relays": list(lanes[0].relays)})
    return rows


HC_PREFILL = "VLLM_QWEN3_8_HC_PREFILL_MODE"   # Qwen3.8's hyper-connection prefill row ownership


def _recipe_value(arguments: Sequence[str], flag: str) -> str | None:
    """The value after ``flag`` (or of ``flag=value``) in the recipe's vLLM arguments, if given."""
    for index, item in enumerate(arguments):
        if item == flag and index + 1 < len(arguments):
            return arguments[index + 1]
        if item.startswith(flag + "="):
            return item[len(flag) + 1:]
    return None


def relay_conflicts(arguments: Sequence[str], environment: Mapping[str, str]) -> list[str]:
    """Recipe settings whose collectives bypass SIRCL's communicator, refused on a group NCCL may not run.

    The same conditions SIRCL's communicator refuses when it builds such a
    tensor-parallel group (``communicator._config_conflicts``), read from the
    recipe's vLLM arguments and the containers' environment (with ``--env``),
    so a plan fails before any container starts. Each entry names what the
    operator can change. Prefill row ownership is not among them: SIRCL's
    communicator carries GLM-5.3-Flash's mHC rows and Qwen3.8's
    hyper-connection rows through pinned shims.
    """
    found = []
    try:
        compilation = json.loads(_recipe_value(arguments, "--compilation-config") or "{}")
    except json.JSONDecodeError:
        compilation = {}
    passes = compilation.get("pass_config") if isinstance(compilation, dict) else None
    passes = passes if isinstance(passes, dict) else {}
    for name, why in (("fuse_gemm_comms", "fuses GEMMs with symmetric-memory reduce-scatter and all-gather"),
                      ("fuse_allreduce_rms", "fuses all-reduce and RMSNorm through FlashInfer or B12X PCIe "
                                             "transports")):
        if passes.get(name):
            found.append(f"the recipe's --compilation-config pass_config.{name} {why}, outside the device "
                         "communicator; the launcher keeps the recipe's vLLM arguments")
    if "--enable-batch-sharded-sampling" in arguments:
        found.append("--enable-batch-sharded-sampling exchanges logits with torch all_to_all_single of uneven "
                     "splits on the NCCL group (v1/worker/gpu/sample/batch_shard.py)")
    backend = _recipe_value(arguments, "--all2all-backend")
    if ("--enable-expert-parallel" in arguments or "-ep" in arguments) and backend not in (
            None, "naive", "allgather_reducescatter"):
        found.append(f"the {backend} all-to-all backend connects every pair of ranks itself")
    return found


def add_schedule_arguments(command: object) -> None:
    """``--large-schedule``, ``--gather-schedule`` and ``--scatter-schedule`` on an argparse (sub)parser."""
    for flag, what in (("--large-schedule", "all-reduces above the dispatch ceiling (all_reduce_large)"),
                       ("--gather-schedule", "all-gathers above the gather op size (all_gather_large)"),
                       ("--scatter-schedule", "reduce-scatters")):
        command.add_argument(flag, choices=SCHEDULES,
                             help=f"the sessions' schedule of {what}: auto, chain, ring or pieces "
                                  "(default: the session's; auto never selects the ring)")


def schedule_environment(options: object) -> dict[str, str]:
    """``SIRCL_*_SCHEDULE`` settings for the schedules ``options`` name (attributes of :data:`SCHEDULE_VARIABLES`)."""
    result = {}
    for attribute, variable in SCHEDULE_VARIABLES:
        value = getattr(options, attribute, None)
        if value is None:
            continue
        if value not in SCHEDULES:
            raise ServePlanError(f"--{attribute.replace('_', '-')} must be one of {SCHEDULES}, got {value!r}")
        result[variable] = value
    return result


def add_link_arguments(command: object) -> None:
    """One ``BYTES`` option per :data:`LINK_SIZES` entry (``--link-slot``, ``--link-chunk``) and ``--link-slots
    N`` (:data:`LINK_SLOTS_VARIABLE`) on an argparse (sub)parser."""
    for size in LINK_SIZES:
        command.add_argument(size.flag, dest=size.attribute, type=int, metavar="BYTES",
                             help=f"{size.variable}: {size.meaning} (default: the session's, {size.default})")
    command.add_argument("--link-slots", dest="link_slots", type=int, metavar="N",
                         help=f"{LINK_SLOTS_VARIABLE}: {LINK_SLOTS_MEANING} (default: the session's, twice its "
                              f"ranks and at least {DEFAULT_LINK_SLOTS}: 16 on the ring of eight)")
    command.add_argument("--ring-gather-stagger", dest="ring_gather_stagger", type=int, metavar="N",
                         help=f"{RING_GATHER_STAGGER_VARIABLE}: {RING_GATHER_STAGGER_MEANING} (default: the "
                              f"session's, {DEFAULT_RING_GATHER_STAGGER} when the link slots hold it, else 0)")


def link_slots_environment(link_slots: int | None) -> dict[str, str]:
    """``SIRCL_LINK_SLOTS`` for ``--link-slots``: 2 to ``protocol.LINK_MAX_SLOTS``; ``{}`` when unset."""
    if link_slots is None:
        return {}
    if not isinstance(link_slots, int) or isinstance(link_slots, bool):
        raise ServePlanError(f"--link-slots must be a slot count, got {link_slots!r}")
    if not 2 <= link_slots <= protocol.LINK_MAX_SLOTS:
        raise ServePlanError(f"--link-slots must be 2 to {protocol.LINK_MAX_SLOTS}, got {link_slots}")
    return {LINK_SLOTS_VARIABLE: str(link_slots)}


def ring_gather_stagger_environment(stagger: int | None) -> dict[str, str]:
    """``SIRCL_RING_GATHER_STAGGER`` for ``--ring-gather-stagger``: 0 to ``protocol.MAX_RING_STAGGER``; ``{}``
    when unset. Whether the link slots hold it is the session's check (:func:`session_settings`)."""
    if stagger is None:
        return {}
    if not isinstance(stagger, int) or isinstance(stagger, bool):
        raise ServePlanError(f"--ring-gather-stagger must be a number of rounds, got {stagger!r}")
    if not 0 <= stagger <= protocol.MAX_RING_STAGGER:
        raise ServePlanError(f"--ring-gather-stagger must be 0 to {protocol.MAX_RING_STAGGER}, got {stagger}")
    return {RING_GATHER_STAGGER_VARIABLE: str(stagger)}


def link_values(args: object) -> dict[str, int]:
    """The link sizes an argparse namespace sets, keyed by ``LinkSize.attribute``."""
    return {size.attribute: value for size in LINK_SIZES
            if (value := getattr(args, size.attribute, None)) is not None}


def link_slot(sizes: Mapping[str, int]) -> int:
    """The link slot the sessions use for ``sizes`` (``LinkSize.attribute`` to bytes): ``--link-slot`` when
    set; otherwise 524,288 bytes, or the largest piece set rounded up to 4,096 bytes when that lies above
    524,288 and at most 1,048,576 bytes (the session's rule without SIRCL_LINK_SLOT_BYTES)."""
    slot = sizes.get("link_slot")
    if slot is not None:
        return slot
    largest = max((value for attribute, value in sizes.items() if attribute != "link_slot"), default=0)
    wanted = -(-largest // LINK_SLOT_ALIGNMENT) * LINK_SLOT_ALIGNMENT
    return wanted if DEFAULT_LINK_SLOT_BYTES < wanted <= MAX_AUTO_LINK_SLOT_BYTES else DEFAULT_LINK_SLOT_BYTES


def link_environment(sizes: Mapping[str, int]) -> dict[str, str]:
    """``SIRCL_*LINK*`` settings for ``sizes`` (``LinkSize.attribute`` to bytes), checked as a session checks
    them: the slot a positive multiple of 4096 bytes up to 2 GiB, every piece a positive multiple of 16 bytes
    up to the slot the sessions use (:func:`link_slot`)."""
    known = {size.attribute: size for size in LINK_SIZES}
    unknown = sorted(set(sizes) - set(known))
    if unknown:
        raise ServePlanError(f"unknown link sizes {unknown}; the launcher sets {sorted(known)}")
    for attribute, value in sizes.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ServePlanError(f"{known[attribute].flag} must be a byte count, got {value!r}")
    slot_size = known["link_slot"]
    slot = sizes.get("link_slot")
    if slot is not None and not (0 < slot <= MAX_LINK_SLOT_BYTES and slot % LINK_SLOT_ALIGNMENT == 0):
        raise ServePlanError(f"{slot_size.flag} must be a positive multiple of {LINK_SLOT_ALIGNMENT} bytes up to "
                             f"{MAX_LINK_SLOT_BYTES} (2 GiB), got {slot}")
    limit = link_slot(sizes)
    environment = {}
    for size in LINK_SIZES:
        value = sizes.get(size.attribute)
        if value is None:
            continue
        if size.piece and (value < 16 or value % 16):
            raise ServePlanError(f"{size.flag} must be a positive multiple of 16 bytes, got {value}")
        if size.piece and value > limit:
            slot_text = (f"{slot_size.flag} {slot}" if slot is not None else
                         f"without {slot_size.flag} the sessions' slot holds pieces up to "
                         f"{MAX_AUTO_LINK_SLOT_BYTES} bytes; set {slot_size.flag}")
            raise ServePlanError(f"{size.flag} {value} exceeds the link slot of {limit} bytes ({slot_text})")
        environment[size.variable] = str(value)
    return environment


def link_size_text(sizes: Mapping[str, int]) -> str:
    """The link sizes in plan text: each set size with its option, otherwise the value the sessions use."""
    slot = link_slot(sizes)
    piece = sizes.get("link_chunk", min(DEFAULT_LINK_CHUNK_BYTES, slot))
    parts = []
    for size in LINK_SIZES:
        if size.attribute in sizes:
            parts.append(f"{size.label} {sizes[size.attribute]:,} B ({size.flag})")
        elif size.attribute == "link_slot":
            parts.append(f"{size.label} {slot:,} B (the session's default)")
        elif size.attribute == "link_chunk":
            parts.append(f"{size.label} {piece:,} B (the session's default)")
        else:
            parts.append(f"{size.label} {piece:,} B (the link piece)")
    return ", ".join(parts)


def minimums_text(minimums: Mapping[str, int]) -> str:
    """Per-collective minimums in plan text: ``8,388,608 B of all-reduce message, ...``."""
    return ", ".join(f"{minimums[collective]:,} B of {MIN_MEASURES[collective]}" for collective in MIN_COLLECTIVES)


def add_minimum_arguments(command: object) -> None:
    """``--chain-min BYTES`` and ``--ring-min BYTES`` (:data:`CHAIN_MIN_VARIABLE`, :data:`RING_MIN_VARIABLE`) on
    an argparse (sub)parser."""
    command.add_argument("--chain-min", dest="chain_min", type=int, metavar="BYTES",
                         help=f"{CHAIN_MIN_VARIABLE}: {CHAIN_MIN_MEANING} (default: the session's, "
                              f"{minimums_text(DEFAULT_CHAIN_MINS)})")
    command.add_argument("--ring-min", dest="ring_min", type=int, metavar="BYTES",
                         help=f"{RING_MIN_VARIABLE}: {RING_MIN_MEANING} (default: the session's, "
                              f"{minimums_text(DEFAULT_RING_MINS)})")


def _minimum_environment(option: str, variable: str, value: int | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, int) or isinstance(value, bool):
        raise ServePlanError(f"{option} must be a byte count, got {value!r}")
    if value < 0:
        raise ServePlanError(f"{option} must not be negative, got {value}")
    return {variable: str(value)}


def chain_min_environment(chain_min: int | None) -> dict[str, str]:
    """``SIRCL_CHAIN_MIN_BYTES`` for ``--chain-min``: a non-negative byte count; ``{}`` when unset."""
    return _minimum_environment("--chain-min", CHAIN_MIN_VARIABLE, chain_min)


def ring_min_environment(ring_min: int | None) -> dict[str, str]:
    """``SIRCL_RING_MIN_BYTES`` for ``--ring-min``: a non-negative byte count; ``{}`` when unset."""
    return _minimum_environment("--ring-min", RING_MIN_VARIABLE, ring_min)


def chain_min_text(chain_min: int | None) -> str:
    """The sessions' chain minimums in plan text: the value set with its option, otherwise the defaults."""
    if chain_min is None:
        return (f"auto schedules run chain ops from {minimums_text(DEFAULT_CHAIN_MINS)} (the session's defaults) "
                "where the ranks form a chain of cable neighbors")
    return (f"auto schedules run chain ops from {chain_min:,} B of every collective (--chain-min) where the ranks "
            "form a chain of cable neighbors")


def ring_min_text(ring_min: int | None) -> str:
    """The sessions' ring minimums in plan text: the value set with its option, otherwise the defaults."""
    if ring_min is None:
        return (f"ring schedules run ring ops from {minimums_text(DEFAULT_RING_MINS)} (the session's defaults), "
                "smaller collectives as under auto")
    return (f"ring schedules run ring ops from {ring_min:,} B of every collective (--ring-min), smaller collectives "
            "as under auto")


def add_oneshot_argument(command: object) -> None:
    """``--oneshot-max BYTES`` (:data:`ONESHOT_VARIABLE`) on an argparse (sub)parser."""
    command.add_argument("--oneshot-max", dest="oneshot_max", type=int, metavar="BYTES",
                         help=f"{ONESHOT_VARIABLE}: {ONESHOT_MEANING} (default: the session's, the latency "
                              "model's limit for the group's layout, which plan shows)")


def oneshot_environment(oneshot_max: int | None, dispatch: int) -> dict[str, str]:
    """``SIRCL_ONESHOT_MAX_BYTES`` for ``--oneshot-max``: a non-negative multiple of 16 bytes no larger
    than the dispatch ceiling ``dispatch``; ``{}`` when unset (the session's default applies)."""
    if oneshot_max is None:
        return {}
    if not isinstance(oneshot_max, int) or isinstance(oneshot_max, bool):
        raise ServePlanError(f"--oneshot-max must be a byte count, got {oneshot_max!r}")
    if oneshot_max < 0 or oneshot_max % 16:
        raise ServePlanError(f"--oneshot-max must be a non-negative multiple of 16 bytes, got {oneshot_max}")
    if oneshot_max > dispatch:
        raise ServePlanError(f"--oneshot-max {oneshot_max} exceeds the dispatch ceiling of {dispatch} bytes "
                             "(--dispatch, default the capacity)")
    return {ONESHOT_VARIABLE: str(oneshot_max)}


def add_large_blocks_argument(command: object) -> None:
    """``--large-blocks N`` (:data:`LARGE_BLOCKS_VARIABLE`) on an argparse (sub)parser."""
    command.add_argument("--large-blocks", dest="large_blocks", type=int, metavar="N",
                         help=f"{LARGE_BLOCKS_VARIABLE}: {LARGE_BLOCKS_MEANING} (default: the session's, "
                              f"{DEFAULT_LARGE_BLOCKS})")


def large_blocks_environment(large_blocks: int | None) -> dict[str, str]:
    """``SIRCL_LARGE_BLOCKS`` for ``--large-blocks``: a power of two from 1 to :data:`MAX_LARGE_BLOCKS`;
    ``{}`` when unset (the session's default applies)."""
    if large_blocks is None:
        return {}
    if not isinstance(large_blocks, int) or isinstance(large_blocks, bool):
        raise ServePlanError(f"--large-blocks must be a block count, got {large_blocks!r}")
    if not 1 <= large_blocks <= MAX_LARGE_BLOCKS or large_blocks & (large_blocks - 1):
        raise ServePlanError(f"--large-blocks must be a power of two from 1 to {MAX_LARGE_BLOCKS}, got "
                             f"{large_blocks}")
    return {LARGE_BLOCKS_VARIABLE: str(large_blocks)}


def large_blocks_text(large_blocks: int | None) -> str:
    """The sessions' grid cap in plan text: the value set with its option, otherwise the default."""
    if large_blocks is None:
        return (f"two-shot and large-message launches on grids of up to {DEFAULT_LARGE_BLOCKS} blocks (the "
                "session's default)")
    return f"two-shot and large-message launches on grids of up to {large_blocks} blocks (--large-blocks)"


def derived_oneshot_max(group: fabric.GroupTopology, capacity: int) -> int:
    """The one-shot limit a session of ``group`` derives without ``SIRCL_ONESHOT_MAX_BYTES``.

    The latency model's limit for the session's layout, lane count and default posting order, at most
    :data:`DEFAULT_ONESHOT_MAX_BYTES` and the capacity: the rule of ``oneshot/runtime.py``
    (``_oneshot_limit``) for a session with a layout and the two-shot all-reduce.
    """
    layout = SessionLayout.parse(group.session_layout())
    return latency_model.oneshot_limit(layout, group.lane_count, DEFAULT_POST_ORDER,
                                       cap=min(capacity, DEFAULT_ONESHOT_MAX_BYTES))


def oneshot_text(oneshot_max: int | None, derived: int | None = None) -> str:
    """The sessions' one-shot limit in plan text: the value set with its option, otherwise the limit the
    sessions derive (``derived``), otherwise the default of a session without a layout."""
    if oneshot_max is None and derived is not None:
        return (f"one-shot all-reduces up to {derived:,} B and two-shot above (the session's limit: the latency "
                "model's for this layout, lane count and posting order)")
    if oneshot_max is None:
        return (f"one-shot all-reduces up to {DEFAULT_ONESHOT_MAX_BYTES:,} B and two-shot above (the session's "
                "default)")
    if oneshot_max == 0:
        return "two-shot all-reduces only (--oneshot-max 0)"
    return f"one-shot all-reduces up to {oneshot_max:,} B and two-shot above (--oneshot-max)"


# vLLM's argument for the chat template's defaults; --reasoning-effort sets the effort level in it on the
# rank that serves the API, as SparkRing's installer does (docs/operations/install-reference.md, Thinking).
CHAT_TEMPLATE_FLAG = "--default-chat-template-kwargs"


def add_reasoning_argument(command: object) -> None:
    """``--reasoning-effort LEVEL`` on an argparse (sub)parser."""
    command.add_argument("--reasoning-effort", dest="reasoning_effort", metavar="LEVEL",
                         help="the effort level of requests that name none: one of the levels the repository's "
                              "profiles/thinking.json records for the checkpoint, written into vLLM's "
                              f"{CHAT_TEMPLATE_FLAG} on the rank that serves the API (default: the chat "
                              "template's own)")


def reasoning_kwargs(behaviour: profile_mod.ThinkingBehaviour | None, level: str | None,
                     checkpoint: str) -> dict[str, str] | None:
    """The chat template defaults ``--reasoning-effort LEVEL`` sets, ``{<effort argument>: LEVEL}``, checked
    against the checkpoint's thinking behaviour; None when no level is given."""
    if level is None:
        return None
    if behaviour is None:
        raise ServePlanError(f"--reasoning-effort: the repository's {profile_mod.THINKING_RELATIVE} records no "
                             f"thinking behaviour for the checkpoint {checkpoint}; --thinking-behaviour names one")
    if not behaviour.levels or behaviour.effort is None:
        raise ServePlanError(f"--reasoning-effort: the checkpoint {checkpoint} ({behaviour.id}) has no effort "
                             "levels")
    if level not in behaviour.levels:
        raise ServePlanError(f"--reasoning-effort {level!r} is not an effort level of the checkpoint {checkpoint} "
                             f"({behaviour.id}); it accepts {', '.join(behaviour.levels)}")
    return {behaviour.effort: level}


def with_chat_template_kwargs(command: list[str], kwargs: Mapping[str, Any]) -> None:
    """Set ``kwargs`` in ``command``'s chat template defaults: merged into the JSON object its
    ``--default-chat-template-kwargs`` gives, otherwise appended as that argument."""
    places = [index for index, item in enumerate(command)
              if item == CHAT_TEMPLATE_FLAG or item.startswith(CHAT_TEMPLATE_FLAG + "=")]
    if not places:
        command += [CHAT_TEMPLATE_FLAG, json.dumps(dict(kwargs), separators=(",", ":"))]
        return
    if len(places) > 1 or (command[places[0]] == CHAT_TEMPLATE_FLAG and places[0] + 1 >= len(command)):
        raise ServePlanError(f"the profile command must give {CHAT_TEMPLATE_FLAG} at most once, with a value")
    index = places[0]
    inline = command[index] != CHAT_TEMPLATE_FLAG
    text = command[index].split("=", 1)[1] if inline else command[index + 1]
    try:
        current = json.loads(text)
    except ValueError:
        current = None
    if not isinstance(current, dict):
        raise ServePlanError(f"the profile command's {CHAT_TEMPLATE_FLAG} {text!r} is not a JSON object")
    merged = json.dumps({**current, **kwargs}, separators=(",", ":"))
    if inline:
        command[index] = f"{CHAT_TEMPLATE_FLAG}={merged}"
    else:
        command[index + 1] = merged


def thinking_text(plan: "ServePlan") -> str:
    """The plan's line about thinking: the served checkpoint's behaviour and the serving default."""
    profile = plan.profile
    behaviour = plan.thinking
    if behaviour is None:
        return (f"  thinking: {profile_mod.THINKING_RELATIVE} records no behaviour for {plan.checkpoint}; "
                "--thinking-behaviour names one")
    label = behaviour.id + (" by --thinking-behaviour" if plan.thinking_source == "--thinking-behaviour" else "")
    if not behaviour.levels or behaviour.effort is None:
        return f"  thinking ({label}): the model has no effort levels"
    if plan.chat_template_defaults is None:
        return (f"  thinking ({label}): requests that name no {behaviour.effort} run at the chat "
                f"template's default, {behaviour.level}; --reasoning-effort sets one of "
                f"{', '.join(behaviour.levels)}")
    return (f"  thinking ({label}): requests that name no {behaviour.effort} run at "
            f"{plan.reasoning_effort} (--reasoning-effort; the chat template's own default is {behaviour.level}); "
            f"the launcher's checks keep the profile's request settings {json.dumps(profile.request_settings)}")


# -- vLLM argument edits ---------------------------------------------------------------------------

VLLM_FLAG = re.compile(r"--[a-z0-9][a-z0-9-]*")
SPECULATIVE_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# vLLM arguments the launcher sets on each rank or reads for its own checks; --vllm-arg and --drop-vllm-arg
# refuse them with the reason.
OWNED_ARGUMENTS = {
    "--port": "the launcher sets each instance's API port (--api-port)",
    "--master-port": "the launcher sets each instance's torch.distributed port (--master-port)",
    "--master-addr": "the launcher sets rank 0's LAN address from the site file",
    "--node-rank": "the launcher sets each rank's index in its group",
    "--nnodes": "the launcher runs one rank per Spark of the group",
    "--headless": "the launcher sets it on every rank but 0, which serves the API",
    "--tensor-parallel-size": "the group's size: the profile's tensor parallelism, one rank per Spark",
    "--distributed-executor-backend": "the launcher starts one process per rank (mp) on every Spark",
    "--served-model-name": "the launcher's readiness test and checks ask for the profile's served model name",
    "--model": f"the checkpoint is mounted at {MODEL_TARGET} (--model-path)",
    CHAT_TEMPLATE_FLAG: "use --reasoning-effort, which the launcher writes into rank 0's value",
}


def owned_argument(flag: str) -> str | None:
    """Why the launcher owns the vLLM argument ``flag``, or None when an edit may change it."""
    if flag in OWNED_ARGUMENTS:
        return OWNED_ARGUMENTS[flag]
    if flag.endswith("-parallel-size") or flag.startswith("--data-parallel"):
        return "the launcher serves tensor parallelism only, one rank per Spark"
    return None


def _flag(text: str, option: str) -> str:
    """A long vLLM option, with underscores written as dashes as vLLM's parser reads them."""
    flag = "--" + text[2:].replace("_", "-") if text.startswith("--") else text
    if not VLLM_FLAG.fullmatch(flag):
        raise ServePlanError(f"{option} {text!r}: give a long vLLM option such as --quantization")
    reason = owned_argument(flag)
    if reason is not None:
        raise ServePlanError(f"{option} {flag}: the launcher owns this argument ({reason})")
    return flag


def vllm_edits(set_values: Sequence[str] = (), drop_values: Sequence[str] = (),
               speculative_values: Sequence[str] = ()) -> VllmEdits:
    """``--vllm-arg FLAG=VALUE`` or ``FLAG`` (a switch), ``--drop-vllm-arg FLAG`` and ``--speculative-set
    KEY=VALUE`` (VALUE as JSON when it parses, else the text), each checked."""
    changes: list[tuple[str, str | None]] = []
    for item in set_values:
        text, separator, value = item.partition("=")
        flag = _flag(text, "--vllm-arg")
        if separator and not value:
            raise ServePlanError(f"--vllm-arg {item!r}: give a value after '=', or {flag} alone for a switch")
        changes.append((flag, value if separator else None))
    drops = [_flag(item, "--drop-vllm-arg") for item in drop_values]
    fields: list[tuple[str, Any]] = []
    for item in speculative_values:
        key, separator, value = item.partition("=")
        if not separator or not SPECULATIVE_KEY.fullmatch(key):
            raise ServePlanError(f"--speculative-set {item!r} is not KEY=VALUE with a field name as KEY")
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = value
        fields.append((key, parsed))
    for what, names in (("--vllm-arg", [flag for flag, _ in changes]), ("--drop-vllm-arg", drops),
                        ("--speculative-set", [key for key, _ in fields])):
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ServePlanError(f"{what} names {', '.join(repeated)} more than once")
    both = sorted({flag for flag, _ in changes} & set(drops))
    if both:
        raise ServePlanError(f"{', '.join(both)}: given to both --vllm-arg and --drop-vllm-arg")
    if fields and SPECULATIVE_FLAG in drops:
        raise ServePlanError(f"--speculative-set edits {SPECULATIVE_FLAG}, which --drop-vllm-arg removes")
    return VllmEdits(tuple(changes), tuple(drops), tuple(fields))


def _places(arguments: Sequence[str], flag: str) -> list[int]:
    return [index for index, item in enumerate(arguments) if item == flag or item.startswith(flag + "=")]


def _value_words(arguments: Sequence[str], index: int, flag: str) -> tuple[str, ...]:
    """The words after ``flag`` at ``index``: its inline or following value, ``()`` for a switch (the next
    word is another option, or there is none)."""
    item = arguments[index]
    if item != flag:
        return (item[len(flag) + 1:],)
    following = arguments[index + 1] if index + 1 < len(arguments) else None
    return () if following is None or following.startswith("--") else (following,)


def _given(arguments: Sequence[str], flag: str) -> tuple[str, ...] | None:
    places = _places(arguments, flag)
    return _value_words(arguments, places[0], flag) if places else None


def edit_arguments(arguments: Sequence[str], edits: VllmEdits | None) -> tuple[list[str], list[ArgumentChange]]:
    """``arguments`` with ``edits`` applied, and one :class:`ArgumentChange` per argument they name.

    A value replaces the one the arguments give (inline or as the next word)
    or is appended with its flag; a switch is appended unless present. A
    flag the arguments give more than once, a value for a flag given as a
    switch, a switch for a flag given with a value, and a removal of a flag
    they do not give are refused. ``--speculative-set`` fields apply last,
    to the one ``--speculative-config`` JSON object.
    """
    result = list(arguments)
    if not edits:
        return result, []
    options: dict[str, list[str]] = {}
    for flag, value in edits.set:
        options.setdefault(flag, []).append("--vllm-arg")
        places = _places(result, flag)
        if len(places) > 1:
            raise ServePlanError(f"--vllm-arg {flag}: the profile's command gives it {len(places)} times")
        inline = value is not None and value.startswith("-")
        if not places:
            result += [flag] if value is None else [f"{flag}={value}"] if inline else [flag, value]
            continue
        index = places[0]
        current = _value_words(result, index, flag)
        if value is None:
            if current:
                raise ServePlanError(f"--vllm-arg {flag}: the profile's command gives it the value {current[0]!r}; "
                                     f"give {flag}=VALUE")
        elif not current:
            raise ServePlanError(f"--vllm-arg {flag}={value}: the profile's command gives {flag} as a switch "
                                 "without a value; --drop-vllm-arg removes it")
        elif result[index] != flag or inline:
            result[index:index + 2 if result[index] == flag else index + 1] = [f"{flag}={value}"]
        else:
            result[index + 1] = value
    for flag in edits.drop:
        options.setdefault(flag, []).append("--drop-vllm-arg")
        places = _places(result, flag)
        if not places:
            raise ServePlanError(f"--drop-vllm-arg {flag}: the profile's command does not give it")
        for index in reversed(places):
            del result[index:index + 1 + (len(_value_words(result, index, flag)) if result[index] == flag else 0)]
    if edits.speculative:
        options.setdefault(SPECULATIVE_FLAG, []).append("--speculative-set")
        places = _places(result, SPECULATIVE_FLAG)
        if len(places) != 1 or not _value_words(result, places[0], SPECULATIVE_FLAG):
            raise ServePlanError(f"--speculative-set needs exactly one {SPECULATIVE_FLAG} with a value in the "
                                 f"command (the profile's, or one from --vllm-arg); it gives {len(places)}")
        index = places[0]
        text = _value_words(result, index, SPECULATIVE_FLAG)[0]
        try:
            config = json.loads(text)
        except ValueError:
            config = None
        if not isinstance(config, dict):
            raise ServePlanError(f"--speculative-set: {SPECULATIVE_FLAG} {text!r} is not a JSON object")
        config.update(edits.speculative)
        merged = json.dumps(config, separators=(",", ":"))
        if result[index] == SPECULATIVE_FLAG:
            result[index + 1] = merged
        else:
            result[index] = f"{SPECULATIVE_FLAG}={merged}"
    changes = [ArgumentChange(flag, _given(arguments, flag), _given(result, flag), ", ".join(dict.fromkeys(names)))
               for flag, names in options.items()]
    return result, changes


def add_vllm_arguments(command: object) -> None:
    """``--vllm-arg``, ``--drop-vllm-arg`` and ``--speculative-set`` on an argparse (sub)parser."""
    command.add_argument("--vllm-arg", dest="vllm_arg", action="append", default=[], metavar="FLAG=VALUE",
                         help="replace or add one vLLM argument of every rank (repeatable): FLAG=VALUE, or FLAG "
                              "alone for a switch, for example --vllm-arg --quantization=nvfp4_csf; arguments the "
                              "launcher sets per rank (ports, addresses, ranks, parallel sizes, the served model "
                              "name) are refused")
    command.add_argument("--drop-vllm-arg", dest="drop_vllm_arg", action="append", default=[], metavar="FLAG",
                         help="remove one vLLM argument (and its value) from every rank's command (repeatable)")
    command.add_argument("--speculative-set", dest="speculative_set", action="append", default=[],
                         metavar="KEY=VALUE",
                         help=f"set one field of the JSON object {SPECULATIVE_FLAG} gives (repeatable); VALUE "
                              "is JSON when it parses, else text: moe_backend=marlin, num_speculative_tokens=2")


def edits_from_args(args: object) -> VllmEdits:
    return vllm_edits(getattr(args, "vllm_arg", None) or (), getattr(args, "drop_vllm_arg", None) or (),
                      getattr(args, "speculative_set", None) or ())


# Options whose value is a vLLM option, so it starts with "--" (--vllm-arg --quantization=nvfp4_csf).
DASH_VALUE_OPTIONS = ("--vllm-arg", "--drop-vllm-arg")


def join_dash_values(argv: Sequence[str]) -> list[str]:
    """``--vllm-arg --flag=value`` as ``--vllm-arg=--flag=value``, which argparse reads as one option and its
    value instead of two options."""
    result: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in DASH_VALUE_OPTIONS and index + 1 < len(argv) and argv[index + 1].startswith("-"):
            result.append(f"{item}={argv[index + 1]}")
            index += 2
            continue
        result.append(item)
        index += 1
    return result


def vllm_edit_text(changes: Sequence[ArgumentChange]) -> list[str]:
    if not changes:
        return []
    return ["  vLLM arguments edited on every rank:"] + [f"    {change.describe()}" for change in changes]


# -- a source overlay, the served checkpoint, B12X's cache ---------------------------------------------


def add_overlay_argument(command: object) -> None:
    command.add_argument("--overlay", action="append", metavar="HOSTDIR",
                         help=f"a host directory of Python packages, mounted read-only at {OVERLAY_TARGET} on "
                              "every rank and first on PYTHONPATH, so the vllm and b12x packages it holds replace "
                              "the image's: HOSTDIR for every Spark, or N=HOSTDIR for Spark N; it must hold "
                              + ", ".join(OVERLAY_FILES))


def overlay_paths(positions: Sequence[int], shared: str | None, per_spark: Mapping[int, str]) -> dict[int, str]:
    """Each position's overlay directory; every rank or none has one (all ranks import one vLLM)."""
    if shared is None and not per_spark:
        return {}
    resolved = {position: per_spark.get(position, shared) for position in positions}
    missing = [position for position, path in resolved.items() if path is None]
    if missing:
        raise ServePlanError(f"--overlay names no directory for Sparks {missing}; every rank must import the same "
                             "vLLM: give --overlay HOSTDIR, or N=HOSTDIR for every Spark")
    return {position: _path(path, "overlay directory") for position, path in resolved.items()}


def overlay_identity(record: bytes | None, file_sha256: str) -> tuple[str, str]:
    """(what identifies an overlay tree, how it was read) from its ``OVERLAY.json``: the first top-level string
    of :data:`OVERLAY_TREE_KEYS`, else the file's own SHA-256."""
    try:
        document = json.loads(record) if record else None
    except ValueError:
        document = None
    if isinstance(document, dict):
        for key in OVERLAY_TREE_KEYS:
            if isinstance(document.get(key), str) and document[key]:
                return document[key], f"{OVERLAY_RECORD} {key}"
    return file_sha256, f"SHA-256 of {OVERLAY_RECORD} (it states no tree hash)"


def overlay_text(plan: "ServePlan") -> list[str]:
    if not plan.overlay:
        return []
    sources = sorted(set(plan.overlay.values()))
    where = (sources[0] if len(sources) == 1 else
             ", ".join(f"{path} on Spark {position}" for position, path in sorted(plan.overlay.items())))
    lines = [f"  source overlay: {where} (read-only) at {OVERLAY_TARGET} on every rank, first on PYTHONPATH, so "
             "import vllm and import b12x resolve to it (the serving entrypoint puts only its add-on and toolchain "
             "directories before PYTHONPATH); preflight checks " + ", ".join(OVERLAY_FILES) + " on every Spark and "
             f"compares the tree hash {OVERLAY_RECORD} states; stage's probe checks where vllm and b12x resolve"]
    for shim in plan.required_shims:
        lines.append(f"    the {shim} shim this group needs installs only where the overlay's vLLM files match a "
                     "pinned build; stage's probe refuses otherwise"
                     + (" (--mhc-prefill-shard off serves without it)" if shim == "mhc_prefill_shard" else ""))
    return lines


def add_checkpoint_arguments(command: object, *, checkpoint_flags: Sequence[str] = ("--checkpoint-id",)) -> None:
    command.add_argument(*checkpoint_flags, dest="checkpoint_id", metavar="REPOSITORY@REVISION",
                         help="the checkpoint the model directory holds, for the plan record and the thinking "
                              "lookup, when it is not the one the profile pins")
    command.add_argument("--thinking-behaviour", dest="thinking_behaviour", metavar="NAME",
                         help="the thinking behaviour of the checkpoint, by its name in the repository's "
                              "profiles/thinking.json, for a checkpoint that record does not list")


def parse_checkpoint(text: str, option: str = "--checkpoint-id") -> tuple[str, str]:
    """(repository, revision) of ``REPOSITORY@REVISION``."""
    name, separator, revision = text.partition("@")
    if not separator or not name or not revision or "@" in revision or re.search(r"\s", text):
        raise ServePlanError(f"{option} {text!r} is not REPOSITORY@REVISION")
    return name, revision


def served_thinking(root: Any, checkpoint: str, recorded: profile_mod.ThinkingBehaviour | None,
                    name: str | None) -> tuple[profile_mod.ThinkingBehaviour | None, str]:
    """The thinking behaviour of the served checkpoint and where it comes from: ``--thinking-behaviour``'s, which
    must agree with the record's when ``profiles/thinking.json`` lists the checkpoint, else the record's."""
    if name is None:
        return recorded, profile_mod.THINKING_RELATIVE if recorded is not None else ""
    try:
        named = profile_mod.named_thinking_behaviour(root, name)
    except profile_mod.ProfileError as error:
        raise ServePlanError(f"--thinking-behaviour: {error}") from None
    if recorded is not None and recorded.id != named.id:
        raise ServePlanError(f"--thinking-behaviour {name}: {profile_mod.THINKING_RELATIVE} records {recorded.id} "
                             f"for the checkpoint {checkpoint}")
    return named, "--thinking-behaviour"


def add_b12x_cache_argument(command: object) -> None:
    command.add_argument("--b12x-cache-dir", dest="b12x_cache_dir", metavar="CONTAINERPATH",
                         help=f"{B12X_CACHE_VARIABLE} on every rank: where B12X compiles its kernels, an absolute "
                              f"path in the container; under {CACHE_TARGET} it lands in each Spark's cache "
                              "directory (--cache-path), for example /cache/b12x-overlay for a fresh cache "
                              "(default: the profile's)")


# Mount targets a compile cache cannot be written in (read-only mounts).
READ_ONLY_TARGETS = (MODEL_TARGET, SOURCE_TARGET, BUILD_TARGET, OVERLAY_TARGET)


def _under(path: str, target: str) -> bool:
    return path == target or path.startswith(target.rstrip("/") + "/")


def check_b12x_cache_dir(path: str | None) -> str | None:
    """``--b12x-cache-dir``: an absolute container path outside the read-only mounts."""
    if path is None:
        return None
    value = _path(path, "--b12x-cache-dir")
    if value == "/":
        raise ServePlanError("--b12x-cache-dir must name a directory below /")
    if any(_under(value, target) for target in READ_ONLY_TARGETS):
        raise ServePlanError(f"--b12x-cache-dir {value} lies in a read-only mount ("
                             + ", ".join(READ_ONLY_TARGETS) + "); B12X writes its compiled kernels there")
    return value


def cache_mount(path: str | None) -> str | None:
    """The writable mount a container path lies in (the cache or the run directory), or None."""
    if path is None:
        return None
    return next((target for target in (CACHE_TARGET, RUN_TARGET) if _under(path, target)), None)


def b12x_cache_text(plan: "ServePlan") -> str:
    """Where B12X compiles its kernels on each Spark, and whether an overlay shares that cache."""
    value = plan.b12x_cache
    if value is None:
        return (f"  B12X compile cache: {B12X_CACHE_VARIABLE} unset (B12X's default location in the container's "
                "own file system)")
    source = "--b12x-cache-dir" if plan.b12x_cache_dir else "the profile's"
    mount = cache_mount(value)
    if mount == CACHE_TARGET:
        sources = sorted({launch.mount(CACHE_TARGET).source for launch in plan.ranks})
        relative = value[len(CACHE_TARGET):]
        place = (f"the {CACHE_TARGET} mount, on each Spark at {sources[0]}{relative}" if len(sources) == 1 else
                 f"the {CACHE_TARGET} mount, on each Spark at <its cache directory>{relative} (mounts below)")
    elif mount == RUN_TARGET:
        place = f"the run directory, on each Spark at {plan.run_dir}{value[len(RUN_TARGET):]} (new for every run id)"
    else:
        place = "the container's own file system, discarded with the container: every start compiles again"
    text = f"  B12X compile cache: {B12X_CACHE_VARIABLE}={value} ({source}), in {place}"
    if plan.overlay and not plan.b12x_cache_dir:
        text += ("; with the overlay, its b12x reuses kernels the image's b12x compiled there: --b12x-cache-dir "
                 f"{CACHE_TARGET}/<new directory> or a new --cache-path keeps them apart")
    return text


# -- NCCL-free serving -----------------------------------------------------------------------------

# --nccl-debug (and --require-no-nccl): NCCL logs every communicator it creates (INIT), which check scans for.
NCCL_DEBUG_SETTINGS = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT"}
# Log lines that show an NCCL communicator was created in a container: NCCL's own initialization lines, which it
# prints only with NCCL_DEBUG=INFO and INIT among NCCL_DEBUG_SUBSYS, and the line vLLM's PyNccl constructor logs
# whatever NCCL_DEBUG says (vllm/distributed/device_communicators/pynccl.py, "vLLM is using nccl==").
NCCL_INIT_PATTERNS = ("Init START", "Init COMPLETE", "ncclCommInitRank", "ncclCommSplit", "NCCL version",
                      "vLLM is using nccl")
# NCCL's own log lines ("<host>:<pid>:<tid> [<device>] NCCL INFO ..."). Without a communicator line they show NCCL
# reading its settings without creating a communicator, for example "NCCL INFO ENV/Plugin: Could not find:
# libnccl-env.so", seen at the exit of vLLM workers that failed at setup, next to torch's ProcessGroupNCCL
# warning that destroy_process_group() was not called; check reports them and does not count them as
# communicators.
NCCL_LIBRARY_PATTERNS = ("NCCL INFO", "NCCL WARN")
NCCL_LOG_PATTERNS = NCCL_INIT_PATTERNS + NCCL_LIBRARY_PATTERNS
# vLLM's switch that binds the default process group to the GPU (torch.distributed.init_process_group with
# device_id), which creates an NCCL communicator over every rank at startup and one per group through
# split_group (vllm/distributed/parallel_state.py, _init_process_group_for_split_group and
# _create_subgroups_split_group).
SPLIT_GROUP = "VLLM_DISTRIBUTED_USE_SPLIT_GROUP"
NCCL_DEBUG_REASON = "--nccl-debug: NCCL logs every communicator it creates, which check --require-no-nccl scans for"


def add_nccl_free_arguments(command: object, *, debug: bool = True) -> None:
    """``--require-no-nccl`` (and ``--nccl-debug``) on an argparse (sub)parser."""
    command.add_argument("--require-no-nccl", dest="require_no_nccl", action="store_true",
                         help="serve without NCCL: plan and start refuse a group NCCL may run on and settings "
                              "that create NCCL communicators; check fails unless every receipt shows nccl=none, "
                              "pynccl=skipped and no NCCL decision row and no container log shows an NCCL "
                              "communicator (implies --nccl-debug)")
    if debug:
        command.add_argument("--nccl-debug", dest="nccl_debug", action="store_true",
                             help="NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS=INIT on every rank, so NCCL logs every "
                                  "communicator it creates")


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() not in ("", "0", "false", "no", "off")


def nccl_free_problems(environment: Mapping[str, str], *, nccl_mode: str, required: bool,
                       arguments: Sequence[str] = ()) -> list[str]:
    """Settings that create NCCL communicators outside SIRCL's groups, or hand vLLM's world group to code SIRCL
    does not see, refused with SIRCL_NCCL=never or --require-no-nccl (SIRCL's general plugin refuses
    VLLM_DISTRIBUTED_USE_SPLIT_GROUP the same way under SIRCL_NCCL=never)."""
    if nccl_mode != "never" and not required:
        return []
    found = []
    if _recipe_value(arguments, "--load-format") == "instanttensor":
        found.append("--load-format instanttensor hands the world group's NCCL process group to the InstantTensor "
                     "loader (vllm/model_executor/model_loader/weight_utils.py, instanttensor_weights_iterator), "
                     "which no SIRCL session carries; use another load format")
    if "--enable-eplb" in arguments:
        found.append("--enable-eplb: expert load balancing calls torch.distributed all_reduce and all_gather "
                     "through names bound at import (vllm/distributed/eplb/eplb_state.py, rebalance_execute.py), "
                     "which SIRCL's tripwire cannot wrap, on the expert-parallel device group; leave it off")
    if truthy(environment.get(SPLIT_GROUP)):
        found.append(f"{SPLIT_GROUP}={environment[SPLIT_GROUP]} binds vLLM's default process group to the GPU, "
                     "which creates an NCCL communicator over every rank at startup and one per group "
                     "(vllm/distributed/parallel_state.py, _init_process_group_for_split_group, "
                     f"_create_subgroups_split_group); leave {SPLIT_GROUP} unset or 0")
    return found


def nccl_debug_environment(debug: bool, extra_env: Mapping[str, str]) -> dict[str, str]:
    """The NCCL logging settings of ``--nccl-debug``, refused next to ``--env`` values of the same variables."""
    if not debug:
        return {}
    clashing = sorted(key for key in NCCL_DEBUG_SETTINGS if key in extra_env)
    if clashing:
        raise ServePlanError(f"--env {', '.join(clashing)}: --nccl-debug and --require-no-nccl set "
                             + ", ".join(f"{key}={value}" for key, value in NCCL_DEBUG_SETTINGS.items()))
    return dict(NCCL_DEBUG_SETTINGS)


@dataclasses.dataclass(frozen=True)
class GroupCarrier:
    """One kind of process group vLLM builds, and what carries its collectives."""

    name: str            # vLLM's group name
    groups: str          # how many groups, of which ranks
    nccl: str            # the NCCL policy SIRCL's adapter applies to it: all, ring or none
    nccl_reason: str     # why (the cabling, then SIRCL_NCCL and NCCL's ring settings)
    carrier: str


ONE_RANK = "one rank per group: vLLM builds no device communicator and the group issues no collective"


# Decode-context parallelism in profile serving (--dcp-size N): vLLM's DCP groups of N consecutive ranks of
# the tensor-parallel group, each with a SIRCL session (SIRCL_GROUPS tp,dcp, as the bundle builds them), whose
# output combine SIRCL's communicator carries through the dcp_all_to_all shim (and dcp_b12x_transport under
# B12X). DCP_MODELS are the checkpoints whose attention the served images run with decode-context parallelism
# (GLM-5.3-Flash's multi-head latent attention with DeepSeek-V3.2's sparse indexer), DCP_ATTENTION_BACKENDS the
# attention backends that run it for them.
DCP_FLAG = "--decode-context-parallel-size"
DCP_MODELS = {"local-inference-lab/GLM-5.3-Flash-NVFP4-Spark": "GLM-5.3-Flash",
              "local-inference-lab/GLM-5.3-Flash-NVFP4": "GLM-5.3-Flash",
              "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD": "GLM-5.3-Flash",
              "nvidia/GLM-5.3-Flash-NVFP4": "GLM-5.3-Flash"}
DCP_ATTENTION_BACKENDS = ("B12X",)
SESSION_GROUPS_DCP = ("tp", "dcp")
# vLLM interleaves the KV cache over the DCP ranks in blocks of --cp-kv-cache-interleave-size tokens.
# GLM-5.3-Flash's attention under decode-context parallelism needs that size divisible by 4 (its model code
# refuses other values at startup); --dcp-size sets DCP_INTERLEAVE of the model where the recipe leaves the
# flag unset and refuses a value that is not a multiple of it.
INTERLEAVE_FLAG = "--cp-kv-cache-interleave-size"
DCP_INTERLEAVE = {"GLM-5.3-Flash": 4}


def attention_backend(arguments: Sequence[str], environment: Mapping[str, str]) -> str | None:
    """The attention backend the recipe names (``--attention-backend``, else ``VLLM_ATTENTION_BACKEND``), or
    None: vLLM's choice."""
    return _recipe_value(arguments, "--attention-backend") or (environment.get("VLLM_ATTENTION_BACKEND") or None)


def dcp_problem(dcp: object, tensor_parallel: int, checkpoint: str, arguments: Sequence[str],
                environment: Mapping[str, str]) -> str | None:
    """Why profile serving cannot run decode-context parallelism ``dcp`` (``--dcp-size``) for the served
    ``checkpoint`` (``repository@revision``) with the recipe's ``arguments`` and ``environment``, or None."""
    if not isinstance(dcp, int) or isinstance(dcp, bool) or dcp < 1 or tensor_parallel % dcp:
        return f"--dcp-size {dcp} must be a positive divisor of the profile's tensor parallelism {tensor_parallel}"
    if dcp == 1:
        return None
    repository = checkpoint.split("@")[0]
    if repository not in DCP_MODELS:
        return (f"--dcp-size {dcp}: the served checkpoint {repository} is not one whose attention the served "
                f"images run with decode-context parallelism ({', '.join(sorted(set(DCP_MODELS.values())))}: "
                f"{', '.join(DCP_MODELS)})")
    backend = attention_backend(arguments, environment)
    if backend not in DCP_ATTENTION_BACKENDS:
        return (f"--dcp-size {dcp}: the recipe's attention backend {backend or '(unset: vLLM chooses)'} does not "
                f"run {DCP_MODELS[repository]}'s attention with decode-context parallelism; "
                f"{' or '.join(DCP_ATTENTION_BACKENDS)} does (--vllm-arg --attention-backend=B12X)")
    return None


def dcp_interleave(dcp: int, checkpoint: str, arguments: Sequence[str]) -> str | None:
    """The ``--cp-kv-cache-interleave-size`` value ``--dcp-size`` adds for the served ``checkpoint`` where the
    recipe's ``arguments`` leave it unset, or None. Raises :class:`ServePlanError` for a given value that is
    not a multiple of the model's (:data:`DCP_INTERLEAVE`)."""
    model = DCP_MODELS.get(checkpoint.split("@")[0], "")
    multiple = DCP_INTERLEAVE.get(model)
    if dcp <= 1 or multiple is None:
        return None
    given = _recipe_value(arguments, INTERLEAVE_FLAG)
    if given is None:
        return str(multiple)
    if not given.isdigit() or int(given) % multiple:
        raise ServePlanError(f"--dcp-size {dcp}: {model}'s attention under decode-context parallelism needs "
                             f"{INTERLEAVE_FLAG} divisible by {multiple}; the recipe gives {given} "
                             f"(--vllm-arg {INTERLEAVE_FLAG}={multiple})")
    return None


def mhc_dcp_problem(tensor: int, dcp: int, mhc: bool) -> str | None:
    """Why mHC prefill row ownership cannot start at tensor parallelism ``tensor`` with decode-context
    parallelism ``dcp`` in any pinned vLLM build (:func:`pins.mhc_builds`), or None."""
    if not mhc or dcp <= 1 or pins.mhc_builds(tensor, dcp):
        return None
    sizes = sorted({pair for build in pins.SUPPORTED for pair in pins.mhc_admits(build.name)})
    return (f"--dcp-size {dcp}: GLM-5.3-Flash's mHC prefill row ownership ({MHC_SHARD}=1, the profile's) starts "
            "only at " + ", ".join(f"TP{t}/DCP{d}" for t, d in sizes) + " in the pinned vLLM builds, and the "
            f"model refuses TP{tensor} with DCP {dcp} at startup; serve it with --mhc-prefill-shard off")


def dcp_groups_text(world: int, dcp: int) -> str:
    """Plan text of vLLM's decode-context-parallel groups of ``dcp`` consecutive ranks among ``world``."""
    return f"{world // dcp} groups of {dcp} consecutive ranks" if world > dcp else f"one group of ranks 0-{dcp - 1}"


def dcp_groups(layout: fabric.Layout, positions: Sequence[int], dcp: int) -> list[fabric.GroupTopology]:
    """vLLM's decode-context-parallel groups of ``dcp`` consecutive ranks over ``positions``, as SIRCL's adapter
    places them (inside the tensor-parallel group)."""
    positions = tuple(positions)
    try:
        return [fabric.describe_group(layout, list(positions[start:start + dcp]), parent=list(positions))
                for start in range(0, len(positions), dcp)]
    except fabric.FabricError as error:
        raise ServePlanError(f"--dcp-size {dcp}: {error}") from None


def recipe_dcp(arguments: Sequence[str]) -> int:
    """vLLM's decode-context parallelism in the recipe's arguments (1 when not given or not a number)."""
    for flag in ("--decode-context-parallel-size", "-dcp"):
        value = _recipe_value(arguments, flag)
        if value is not None:
            try:
                return max(1, int(value))
            except ValueError:
                return 1
    return 1


def group_carriers(layout: fabric.Layout, positions: Sequence[int], *, nccl_mode: str,
                   environment: Mapping[str, str] | None = None, dcp: int = 1,
                   session_groups: Sequence[str] = ("tp",)) -> list[GroupCarrier]:
    """The process groups vLLM builds for tensor parallelism over the ranks at ``positions`` (pipeline,
    data and prefill-context parallelism 1; decode-context parallelism ``dcp``), each with the NCCL policy
    SIRCL's adapter applies to it and what carries its collectives.

    vLLM's groups: vllm/distributed/parallel_state.py, init_world_group and initialize_model_parallel.
    Policies follow ``adapter.GroupPlacement``: a session group's placement, otherwise the cabling of its
    positions, then ``guard.effective_policy`` with ``nccl_mode`` and NCCL's settings in ``environment``.
    A group without a session shares the session of tp's ranks only where its policy is none.
    """
    env = dict(environment or {})

    def applied(raw: NcclPolicy, reason: str) -> tuple[NcclPolicy, str]:
        return guard.effective_policy(raw, reason, nccl_mode=nccl_mode, environ=env)

    positions = tuple(positions)
    world = len(positions)
    every = f"one group of ranks 0-{world - 1}"
    whole, whole_reason = applied(*fabric.nccl_policy_of(layout, positions))
    top = fabric.describe_group(layout, positions)
    tp, tp_reason = applied(top.nccl_policy, top.nccl_reason)
    rows = [
        GroupCarrier("world", every, whole.value, whole_reason,
                     "no device communicator (vLLM builds the world group without one); its gloo group carries "
                     "vLLM's control messages, and " +
                     ("SIRCL's tripwire refuses NCCL calls on its device group and on the default group"
                      if whole is NcclPolicy.NONE else
                      "direct torch.distributed calls on its device group or the default group run on NCCL")),
        GroupCarrier("tp", every, tp.value, tp_reason,
                     "SIRCL's communicator and the group's SIRCL session (collectives below)"),
        GroupCarrier("ep", every + ", built for mixture-of-experts models", whole.value, whole_reason,
                     "SIRCL's communicator; the same ranks as tp, so it shares tp's session, which also carries the "
                     "torch.distributed calls it issues directly (online quantization's amax all-reduces)"
                     if whole is NcclPolicy.NONE else
                     "NCCL: SIRCL's communicator without a session (only a group NCCL may not run shares tp's), "
                     "so NCCL carries its collectives, PyNccl's and the torch.distributed calls it issues directly "
                     "(online quantization's amax all-reduces at startup)"),
    ]
    if dcp > 1:
        session = "dcp" in session_groups
        policies = []
        refused = ""
        for start in range(0, world, dcp):
            members = positions[start:start + dcp]
            try:
                if session:
                    placed = fabric.describe_group(layout, members, parent=positions)
                    policies.append(applied(placed.nccl_policy, placed.nccl_reason))
                else:
                    policies.append(applied(*fabric.nccl_policy_of(layout, members)))
            except fabric.FabricError as error:
                refused = str(error)
                policies.append((NcclPolicy.NONE, refused))
        kinds = {policy for policy, _ in policies}
        value = "/".join(sorted(kind.value for kind in kinds))
        reason = (policies[0][1] if len(set(policies)) == 1 else
                  "; ".join(f"ranks {start}-{start + dcp - 1}: {policy.value} ({why})"
                            for start, (policy, why) in zip(range(0, world, dcp), policies)))
        if refused:
            carrier = f"refused at startup: {refused}"
        elif session:
            carrier = ("SIRCL's communicator and a session of its own per group (SIRCL_GROUPS tp,dcp): "
                       "all-gather, reduce-scatter and all-to-all, with the dcp_all_to_all and dcp_b12x_transport "
                       "shims" + ("" if kinds == {NcclPolicy.NONE} else
                                  "; NCCL may run what the collective map sends to it"))
        elif NcclPolicy.NONE in kinds:
            carrier = ("refused at startup: a decode-context-parallel group NCCL may not run needs a session "
                       "(SIRCL_GROUPS tp,dcp)")
        else:
            carrier = "NCCL: the groups have no SIRCL session (SIRCL_GROUPS tp), so NCCL carries their collectives"
        rows.append(GroupCarrier("dcp", dcp_groups_text(world, dcp), value, reason,
                                 carrier))
    else:
        rows.append(GroupCarrier("dcp", "decode-context parallelism 1", "-", "one rank", ONE_RANK))
    for name, what in (("pp", "pipeline"), ("dp", "data"), ("pcp", "prefill-context")):
        rows.append(GroupCarrier(name, f"{what} parallelism 1", "-", "one rank", ONE_RANK))
    return rows


def collective_carriers(*, nccl_allowed: bool, large: str, dispatch: int, gather: int) -> list[tuple[str, str]]:
    """What carries each collective class of a group with a SIRCL session (planner.py's rules)."""
    nccl_large = nccl_allowed and large != "sircl"
    above = ("eager calls above it go to NCCL where the group's NCCL policy allows the collective "
             f"(SIRCL_LARGE_ALLREDUCE={large}); captured calls stay on the session's large-message ops"
             if nccl_large else "larger ones the session's large-message ops")
    rows = [
        ("all-reduce", f"one session op up to {dispatch:,} B; {above}"),
        ("all-gather", f"one session op up to {gather:,} B; {above}"),
        ("reduce-scatter", "the session's reduce-scatter where its dtype is prepared, else an all-reduce and this "
                           "rank's slice" + ("; eager calls above the session's op size go to NCCL where allowed"
                                             if nccl_large else "")),
        ("broadcast, gather", ("NCCL for eager calls where allowed, " if nccl_large else "")
         + "an all-gather of the bytes on the session"),
        ("all-to-all", "the session's all-to-all where prepared, else an all-gather and each rank's chunks"),
        ("send, recv", "SIRCL point-to-point channels between every pair of the group's ranks (SIRCL_P2P_GROUPS, "
                       "default pp,tp,dcp; pipeline parallelism 1 issues none); "
         + ("NCCL between ranks that share a cable where the group has no channels" if nccl_allowed else
            "refused where the group has no channels")),
        ("direct torch.distributed calls", "carried on the session: " + ", ".join(guard.CARRIED)
         + "; other direct calls " + ("go to NCCL where allowed" if nccl_allowed else "are refused")),
        ("PyNccl", "built (NCCL may run here)" if nccl_allowed else "not built: no PyNccl communicator, no warm-up"),
    ]
    return rows


def carrier_lines(groups: Sequence[GroupCarrier], collectives: Sequence[tuple[str, str]]) -> list[str]:
    lines = ["  process groups vLLM builds and what carries them:"]
    lines += [f"    {row.name} ({row.groups}): {row.carrier}" if row.nccl == "-" else
              f"    {row.name} ({row.groups}): NCCL policy {row.nccl} ({row.nccl_reason}); {row.carrier}"
              for row in groups]
    lines.append("  collectives of the groups with a session:")
    lines += [f"    {name}: {carrier}" for name, carrier in collectives]
    return lines


def nccl_free_text(*, required: bool, debug: bool, nccl_allowed: bool) -> str:
    if required:
        return ("  NCCL-free (--require-no-nccl): no group lets NCCL run, settings that create NCCL communicators "
                "are refused, NCCL logs every communicator it creates (NCCL_DEBUG=INFO, INIT), and check fails "
                "unless every receipt shows nccl=none, pynccl=skipped and no NCCL decision row and no container "
                "log shows an NCCL communicator")
    if nccl_allowed:
        return "  NCCL: may run on this group; --nccl never and --require-no-nccl serve without it"
    return ("  NCCL: no group lets NCCL run; --require-no-nccl also proves it with check" +
            (" (NCCL_DEBUG=INFO, INIT: --nccl-debug)" if debug else ""))


# -- SIRCL sessions: schedules and link collectives ---------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SessionSettings:
    """One kind of SIRCL session of an instance and its large-message schedules and link collectives as the
    session runs them, from its layout and the launcher's options (oneshot/runtime.py, ``_configure`` and
    ``_plan_chain``)."""

    name: str                         # the vLLM group whose session it is: tp or dcp
    groups: str                       # how many groups, of which ranks
    scoped: bool                      # the launcher's schedule and link options reach it
    schedules: Mapping[str, str]      # schedule by attribute of DEFAULT_SCHEDULES
    given: tuple[str, ...]            # attributes of the schedules an option set
    ranks_on_fabric: tuple[int, int]  # Sparks of its fabric that host one of its ranks, Sparks of the fabric
    chain_order: tuple[int, ...] | None
    chain_area: bool                  # the session reserves the chain area (all_reduce_large chain ops)
    link_area: bool                   # the session reserves the link area (chain and ring link collectives)
    ring: str                         # whether its ring is available to ring schedules
    link_slots: int
    link_slots_given: bool
    ring_stagger: int                 # the ring reduce-scatter's stagger (the session's rule)
    ring_gather_stagger: int          # the ring all-gather's stagger
    ring_gather_stagger_given: bool
    link_sizes: Mapping[str, int]     # the LinkSize values set
    chain_min: int | None
    ring_min: int | None
    problems: tuple[str, ...]         # settings the session refuses at setup
    # The settings of the tuning table the session takes (tuning.SETTINGS), which it applies where the
    # launcher leaves them unset
    table_settings: Mapping[str, int] = dataclasses.field(default_factory=dict)

    @property
    def covers_fabric(self) -> bool:
        return self.ranks_on_fabric[0] == self.ranks_on_fabric[1]

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "groups": self.groups, "scoped": self.scoped,
                "schedules": {key.removesuffix("_schedule"): value for key, value in self.schedules.items()},
                "covers_fabric": self.covers_fabric, "chain_order": (list(self.chain_order)
                                                                if self.chain_order is not None else None),
                "chain_area": self.chain_area, "link_area": self.link_area, "ring": self.ring,
                "link_slots": self.link_slots, "link_slot": link_slot(self.link_sizes),
                "ring_stagger": self.ring_stagger, "ring_gather_stagger": self.ring_gather_stagger,
                "link_sizes": dict(self.link_sizes), "chain_min": self.chain_min, "ring_min": self.ring_min,
                "table_settings": dict(self.table_settings)}


def session_settings(group: fabric.GroupTopology, *, name: str, groups: str, scoped: bool,
                     schedules: Mapping[str, str] | None = None, link_sizes: Mapping[str, int] | None = None,
                     link_slots: int | None = None, chain_min: int | None = None,
                     ring_min: int | None = None, ring_gather_stagger: int | None = None,
                     table_settings: Mapping[str, int] | None = None) -> SessionSettings:
    """The schedules and link collectives of ``group``'s session, and the settings it would refuse.

    The rules of oneshot/runtime.py for a session built with the layout the adapter passes
    (``GroupTopology.session_layout``), the default thread count and hairpin queue: chain and ring ops need
    every Spark of the session's fabric to host one of its ranks; a chain schedule needs the ranks to form a
    chain of cable neighbors (``routes.chain_order``); a ring schedule needs a ring no relay queue carries
    twice (``routes.ring_window``); the link geometry must be valid (``protocol.LinkLayout``) whether or not
    the session reserves a link area, and a ring stagger the launcher sets needs ``stagger x (ranks - 1) +
    2`` link slots (``protocol.ring_stagger_slots``); unset, each stagger is its default where the slots
    hold it, else 0. ``table_settings`` are those of the tuning table the session takes
    (``tuning.SETTINGS``): the session applies each one the launcher leaves unset.
    """
    table = dict(table_settings or {})
    given = {key: value for key, value in (schedules or {}).items() if value is not None}
    effective = {key: given.get(key, default) for key, default in DEFAULT_SCHEDULES.items()}
    sizes = dict(link_sizes or {})
    layout = SessionLayout.parse(group.session_layout())
    maps = [group.route_map(rank) for rank in range(len(group.members))]
    hosted = len(set(layout.positions)), len(layout.fabric.positions)
    covers = sorted(layout.positions) == list(layout.fabric.positions)
    order = routes_mod.chain_order(layout, maps) if covers else None
    large, gather, scatter = (effective[key] for key in DEFAULT_SCHEDULES)
    chain_area = covers and large != "pieces"
    # A tuning table that chooses a link schedule gives the session a link area; a table the tune command
    # built records the link slots exactly when it chooses one (tuning.table_settings).
    link_area = covers and (gather != "pieces" or scatter != "pieces" or large == "ring"
                            or LINK_SLOTS_VARIABLE in table)
    problems = []
    flags = {key: "--" + key.replace("_", "-") for key in DEFAULT_SCHEDULES}
    ring_problems: list[str] = []
    window = 0
    if link_area and order is not None:
        window, ring_problems = routes_mod.ring_window(layout, maps, order, chunk=protocol.LINK_WINDOW_CHUNK,
                                                       queue_bytes=routes_mod.DEFAULT_HAIRPIN_QUEUE)
    for key, value in effective.items():
        if value not in ("chain", "ring"):
            continue
        if not covers:
            problems.append(f"{flags[key]} {value}: the {name} session's ranks occupy {hosted[0]} of the "
                            f"{hosted[1]} Sparks of its fabric, and a session runs chain and ring ops only when "
                            "every Spark of its fabric hosts one of its ranks")
        elif order is None:
            problems.append(f"{flags[key]} {value}: the {name} session's ranks do not form a chain of cable "
                            "neighbors joined by direct lanes")
        elif value == "ring" and ring_problems:
            problems.append(f"{flags[key]} ring: the {name} session's ring cannot run: " + "; ".join(ring_problems))
    slots = (table.get(LINK_SLOTS_VARIABLE, protocol.default_link_slots(len(group.members))) if link_slots is None
             else link_slots)
    slot = link_slot(sizes) if "link_slot" in sizes else max(link_slot(sizes), table.get("SIRCL_LINK_SLOT_BYTES", 0))
    try:
        protocol.LinkLayout(group.lane_count, slots, slot)
    except protocol.ProtocolError as error:
        problems.append(f"the {name} session's link slots ({slots}, --link-slots) and link slot ({slot:,} B): "
                        f"{error}")
    world = len(group.members)

    def held(stagger: int) -> bool:
        return slots >= protocol.ring_stagger_slots(world, stagger)

    ring_stagger = DEFAULT_RING_STAGGER if held(DEFAULT_RING_STAGGER) else 0
    if ring_gather_stagger is None:
        gather_stagger = DEFAULT_RING_GATHER_STAGGER if held(DEFAULT_RING_GATHER_STAGGER) else 0
    else:
        gather_stagger = ring_gather_stagger
        if not held(ring_gather_stagger):
            problems.append(f"--ring-gather-stagger {ring_gather_stagger}: the {name} session's {world} ranks need "
                            f"{protocol.ring_stagger_slots(world, ring_gather_stagger)} link slots for it, and it "
                            f"has {slots} (--link-slots)")
    if not link_area:
        ring = "none (no link area)"
    elif order is None or ring_problems:
        ring = "cannot run: " + ("; ".join(ring_problems) or "the ranks form no chain of cable neighbors")
    elif window:
        ring = f"available, relayed ring lanes keep up to {window:,} B unacknowledged"
    else:
        ring = "available, every ring edge is a cable"
    return SessionSettings(name=name, groups=groups, scoped=scoped, schedules=effective,
                           given=tuple(given), ranks_on_fabric=hosted, chain_order=order, chain_area=chain_area,
                           link_area=link_area, ring=ring, link_slots=slots, link_slots_given=link_slots is not None,
                           ring_stagger=ring_stagger, ring_gather_stagger=gather_stagger,
                           ring_gather_stagger_given=ring_gather_stagger is not None,
                           link_sizes=sizes, chain_min=chain_min, ring_min=ring_min, problems=tuple(problems),
                           table_settings=table)


def session_lines(sessions: Sequence[SessionSettings]) -> list[str]:
    """Plan text: each session's schedules, chain and link areas, ring, link sizes and minimums."""
    lines = ["  SIRCL sessions (the schedule and link options apply to the tensor-parallel session; other "
             "sessions keep the session defaults):"]
    for session in sessions:
        title = {"tp": "tensor-parallel session", "dcp": "decode-context-parallel sessions"}.get(
            session.name, f"{session.name} sessions")
        schedules = ", ".join(
            f"{SCHEDULE_LABELS[key]} {value} (" + ("--" + key.replace("_", "-") if key in session.given
                                                 else "the session's default") + ")"
            for key, value in session.schedules.items())
        text = f"    {title} ({session.name}, {session.groups}): schedules: {schedules}; "
        if not (session.chain_area or session.link_area):
            reason = (f"its ranks occupy {session.ranks_on_fabric[0]} of the {session.ranks_on_fabric[1]} Sparks of "
                      "its fabric" if not session.covers_fabric else "its schedules name no chain or link op")
            text += (f"no chain or link area: {reason}, so every large-message op runs in pieces and the link sizes "
                     "and chain and ring minimums do not apply")
        else:
            order = ("-".join(str(rank) for rank in session.chain_order) if session.chain_order is not None else
                     "none (the ranks form no chain of cable neighbors)")
            text += (f"chain order {order}; chain area {'yes' if session.chain_area else 'no'}, link area "
                     f"{'yes' if session.link_area else 'no'}; ring: {session.ring}; "
                     + link_size_text(session.link_sizes) + f", link slots {session.link_slots} ("
                     + ("--link-slots" if session.link_slots_given else "the tuning table's"
                        if LINK_SLOTS_VARIABLE in session.table_settings else "the session's default") + "); "
                     + f"ring staggers: reduce-scatter {session.ring_stagger} (the session's default), all-gather "
                     + f"{session.ring_gather_stagger} (" + ("--ring-gather-stagger" if session.ring_gather_stagger_given
                                                          else "the session's default") + "); "
                     + chain_min_text(session.chain_min) + "; " + ring_min_text(session.ring_min))
        if not session.scoped:
            text += ("; SIRCL's adapter builds these sessions without the tensor-parallel session's schedule and "
                     "link variables (settings.TP_SESSION_VARIABLES)")
        lines.append(text)
    return lines


def session_problems(sessions: Sequence[SessionSettings]) -> None:
    """Refuse settings a session would refuse at setup, naming the session and the setting."""
    problems = [problem for session in sessions for problem in session.problems]
    if problems:
        raise ServePlanError("SIRCL's sessions would refuse these settings at setup on every rank: "
                             + "; ".join(problems))


# -- measured tuning tables ---------------------------------------------------------------------------

# Measured tuning tables (sparkring_sircl.tuning, schema sircl-tuning-table/v1). --tuning-table PATH (repeatable)
# stages each table's bytes in the run directory under the table's hash, and SIRCL_TUNING_TABLE names every
# staged table; each session takes the one whose key matches its own group shape, sizes and build
# (oneshot/runtime.py), and its setup agreement refuses ranks with different tables.
TUNING_VARIABLE = "SIRCL_TUNING_TABLE"
TUNING_DIRECTORY = "tuning"
TUNING_MEANING = ("the measured tuning tables staged in the run directory; each session takes the one whose key "
                  "matches its group shape, sizes and build, and the rules choose where none does")
SESSION_TITLES = {"tp": "tensor-parallel session", "dcp": "decode-context-parallel sessions"}


@dataclasses.dataclass(frozen=True)
class StagedTable:
    """A tuning table the launcher stages on every rank."""

    source: str                       # the path --tuning-table named
    hash: str                         # the document's hash (tuning.document_hash), which receipts name
    sha256: str                       # of the file's bytes, checked where it is written
    key: Mapping[str, Any]
    sessions: tuple[str, ...]         # the sessions of the launch whose facts it matches
    data: bytes = dataclasses.field(repr=False, compare=False, default=b"")
    decisions: tuple[Mapping[str, Any], ...] = dataclasses.field(repr=False, compare=False, default=())
    # The table's settings (tuning.SETTINGS), which every session that takes it applies where its
    # environment leaves them unset
    settings: Mapping[str, int] = dataclasses.field(compare=False, default_factory=dict)

    @property
    def file_name(self) -> str:
        return f"{self.hash}.json"

    @property
    def container_path(self) -> str:
        return f"{RUN_TARGET}/{TUNING_DIRECTORY}/{self.file_name}"

    def host_path(self, run_dir: str) -> str:
        return f"{run_dir}/{TUNING_DIRECTORY}/{self.file_name}"

    def to_json(self, run_dir: str) -> dict[str, Any]:
        return {"source": self.source, "hash": self.hash, "sha256": self.sha256, "key": dict(self.key),
                "sessions": list(self.sessions), "host_path": self.host_path(run_dir),
                "container_path": self.container_path, "settings": dict(self.settings)}


@dataclasses.dataclass(frozen=True)
class SessionTable:
    """The tuning table one kind of session of the launch takes (None: its rules choose) and what the table
    sends to NCCL on its group."""

    name: str                         # tp or dcp
    facts: Mapping[str, Any]          # tuning.facts_for_layout of the session's layout and lanes
    table: StagedTable | None
    mismatches: Mapping[str, tuple[str, ...]]   # table hash -> key fields that differ (tables it does not take)
    nccl: str                         # plan text: the sizes and modes the table sends to NCCL

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "facts": dict(self.facts), "table": self.table.hash if self.table else None,
                "mismatches": {key: list(value) for key, value in self.mismatches.items()}, "nccl": self.nccl}


@dataclasses.dataclass(frozen=True)
class TuningPlan:
    """The launch's tuning tables, the table each session takes, and the sessions that share one."""

    tables: tuple[StagedTable, ...] = ()
    sessions: tuple[SessionTable, ...] = ()
    shared: Mapping[str, str] = dataclasses.field(default_factory=dict)   # group kind -> session it shares

    def environment(self) -> dict[str, str]:
        if not self.tables:
            return {}
        return {TUNING_VARIABLE: ",".join(table.container_path for table in self.tables)}

    def settings_of(self, name: str) -> dict[str, int]:
        """The settings of the table session ``name`` takes (directly or through the session it shares);
        empty when it takes none."""
        owner = self.shared.get(name, name)
        row = next((row for row in self.sessions if row.name == owner), None)
        return dict(row.table.settings) if row is not None and row.table is not None else {}

    def conflicts(self, environment: Mapping[str, str]) -> list[str]:
        """The launcher's settings of the tensor-parallel session (``environment``, the variables it sets on
        every rank; its link options reach that session alone) below the settings of the table that session
        takes: those would leave the table's choices that need more unable to run."""
        conflicts = tuning_mod.settings_conflicts(self.settings_of("tp"), environment)
        if not conflicts:
            return []
        flags = {name: flag for flag, name in (("--link-slots", LINK_SLOTS_VARIABLE),
                                               ("--link-slot", "SIRCL_LINK_SLOT_BYTES"))}
        named = [f"{item} ({flags.get(item.split('=')[0], 'set by the launcher')})" for item in conflicts]
        table = next(row.table for row in self.sessions if row.name == "tp")
        return [f"--tuning-table {table.source} (table {table.hash}): the tensor-parallel session takes it, and "
                f"its choices need more than " + ", ".join(named) + "; drop those options so the session applies "
                "the table's settings, or name a table tuned under them"]

    def expected(self) -> dict[str, str | None]:
        """Group kind -> the hash of the table its session takes (None: the rules), as check compares it."""
        tables = {row.name: (row.table.hash if row.table else None) for row in self.sessions}
        tables.update({kind: tables.get(owner) for kind, owner in self.shared.items()})
        return tables

    def lines(self) -> list[str]:
        if not self.tables:
            return ["  tuning tables: none (--tuning-table); every session's rules choose"]
        lines = [f"  tuning tables ({TUNING_VARIABLE}=" + ",".join(t.container_path for t in self.tables) + "):"]
        for table in self.tables:
            key = table.key
            lines.append(f"    table {table.hash} from {table.source}: {key.get('shape')}, {key.get('world')} ranks, "
                         f"{key.get('lanes')} lanes, {key.get('max_relays')} relays, native {key.get('native')}, "
                         f"kernels {key.get('kernels')}, sircl {key.get('sircl')}; taken by the "
                         + " and the ".join(SESSION_TITLES.get(name, name) for name in table.sessions))
            if table.settings:
                lines.append("      settings its sessions apply where the launcher leaves them unset: "
                             + ", ".join(f"{name}={value:,}" for name, value in table.settings.items()))
        for row in self.sessions:
            title = SESSION_TITLES.get(row.name, row.name)
            if row.table is None:
                why = "; ".join(f"table {key}: {', '.join(fields)}" for key, fields in row.mismatches.items())
                lines.append(f"    {title} ({row.name}): rules (no table matches; {why})")
                continue
            lines.append(f"    {title} ({row.name}): table {row.table.hash}; {row.nccl}")
            for entry in row.table.decisions:
                intervals = "; ".join(
                    f"from {int(item['from']):,} B {tuning_mod.Choice.from_json(item['choice']).label()}"
                    + (" (NCCL faster)" if item.get("nccl") else "") for item in entry["intervals"])
                until = f"; nothing above {int(entry['until']):,} B" if "until" in entry else ""
                lines.append(f"      {entry['collective']} {entry['mode']}: {intervals}{until}")
        for kind, owner in self.shared.items():
            lines.append(f"    {kind}: shares the {SESSION_TITLES.get(owner, owner)} and its table")
        return lines

    def to_json(self, run_dir: str) -> dict[str, Any]:
        return {"variable": TUNING_VARIABLE, "tables": [table.to_json(run_dir) for table in self.tables],
                "sessions": [row.to_json() for row in self.sessions], "shared": dict(self.shared),
                "expected": self.expected()}


def add_tuning_argument(command: object) -> None:
    """``--tuning-table PATH`` (repeatable; :data:`TUNING_VARIABLE`) on an argparse (sub)parser."""
    command.add_argument("--tuning-table", dest="tuning_table", action="append", metavar="PATH",
                         help=f"{TUNING_VARIABLE}: a measured tuning table (schema {tuning_mod.SCHEMA}), staged on "
                              "every rank; each session takes the table whose key matches it (repeatable: one "
                              "table per group shape)")


def tuned_nccl_text(decisions: Sequence[Mapping[str, Any]], policy: NcclPolicy, large: str,
                    nccl_mode: str = "auto") -> str:
    """Plan text: what the table decides about NCCL on a group of ``policy``. A table chooses SIRCL's
    settings only, under every ``SIRCL_NCCL`` mode and ``large`` (``SIRCL_LARGE_ALLREDUCE``): the adapter
    passes no tuned backend to ``planner.Policy``, so the table's NCCL marks route no call."""
    if policy is NcclPolicy.NONE:
        return "SIRCL carries every size: NCCL may not run on this group"
    return "the table chooses SIRCL's settings only: its NCCL marks are measurements and route no call"


def tuning_plan(paths: Sequence[str], sessions: Sequence[tuple[str, fabric.GroupTopology, NcclPolicy]], *,
                large: str, shared: Mapping[str, str] | None = None, nccl_mode: str = DEFAULT_NCCL_MODE) -> TuningPlan:
    """Load ``paths`` (``--tuning-table``) and match every table against the launch's sessions (name, group,
    effective NCCL policy): each session takes the table whose key equals its facts
    (``tuning.facts_for_layout`` of its layout and lanes, with this tree's native and kernel hashes and
    version). Refuses a malformed table, a table no session takes and two different tables that match one
    session."""
    loaded: dict[str, tuple[str, bytes, tuning_mod.Table]] = {}
    for path in dict.fromkeys(str(item) for item in paths):
        try:
            data = Path(path).read_bytes()
            table = tuning_mod.Table(json.loads(data.decode("utf-8")), path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ServePlanError(f"--tuning-table {path}: {error}") from None
        loaded.setdefault(table.hash, (path, data, table))
    if not loaded:
        return TuningPlan(shared={})
    facts = {name: tuning_mod.facts_for_layout(group.session_layout(), group.lane_count)
             for name, group, _ in sessions}
    mismatches = {digest: {name: tuple(table.mismatches(own)) for name, own in facts.items()}
                  for digest, (_, _, table) in loaded.items()}
    takers = {digest: tuple(name for name, fields in rows.items() if not fields) for digest, rows in mismatches.items()}
    for digest, (path, _, _) in loaded.items():
        if not takers[digest]:
            why = "; ".join(f"{SESSION_TITLES.get(name, name)}: {', '.join(fields)}"
                            for name, fields in mismatches[digest].items())
            raise ServePlanError(f"--tuning-table {path} (table {digest}) matches no session of this launch ({why}); "
                                 "a table applies to the group shape, sizes and build it was measured on")
    for name, _, _ in sessions:
        hits = [digest for digest in loaded if name in takers[digest]]
        if len(hits) > 1:
            raise ServePlanError(f"--tuning-table: tables {', '.join(f'{loaded[d][0]} ({d})' for d in hits)} all match "
                                 f"the {SESSION_TITLES.get(name, name)}; name one table per group shape and build")
    tables = tuple(StagedTable(source=path, hash=digest, sha256=hashlib.sha256(data).hexdigest(),
                               key=dict(table.key), sessions=takers[digest], data=data,
                               decisions=tuple(table.document.get("decisions", ())), settings=dict(table.settings))
                   for digest, (path, data, table) in loaded.items())
    rows = []
    for name, _, policy in sessions:
        taken = next((table for table in tables if name in table.sessions), None)
        rows.append(SessionTable(
            name=name, facts=facts[name], table=taken,
            mismatches={digest: fields for digest, by_session in mismatches.items()
                        for session, fields in by_session.items() if session == name and fields},
            nccl=tuned_nccl_text(taken.decisions, policy, large, nccl_mode) if taken is not None else ""))
    return TuningPlan(tables=tables, sessions=tuple(rows), shared=dict(shared or {}))


def write_tuning_tables(tables: Sequence[StagedTable], run_dir: str, sudo: str) -> list[tuple[StagedTable, str]]:
    """``(table, remote command)`` writing each table's bytes (the command's stdin) to its host path once."""
    from .commands import privileged, write_once

    return [(table, privileged(sudo, write_once(table.host_path(run_dir), table.sha256))) for table in tables]


def micro_batching(arguments: Sequence[str]) -> str | None:
    """Why the recipe's vLLM micro-batching cannot run with SIRCL's sessions, or None when it is off.

    ``--enable-dbo`` or ``--ubatch-size`` above 1 runs each micro-batch's
    forward in its own thread on its own streams; SIRCL's communicator refuses
    it on every group with a session (``communicator._session_conflicts``),
    and the launcher gives every tensor-parallel group a session.
    """
    size = _recipe_value(arguments, "--ubatch-size")
    try:
        ubatches = int(size) if size is not None else 0
    except ValueError:
        ubatches = 0
    if "--enable-dbo" in arguments or ubatches > 1:
        return ("vLLM's micro-batching (--enable-dbo, or --ubatch-size above 1) issues collectives from one "
                "thread per micro-batch, and SIRCL's sessions need one issuing order on every rank")
    return None


def check_extra_env(extra: Mapping[str, str]) -> None:
    """Refuse ``--env`` keys that are malformed or that the launcher owns."""
    for key in extra:
        if not ENV_KEY.fullmatch(key):
            raise ServePlanError(f"--env {key!r} is not an environment variable name")
        if key.startswith("SIRCL_"):
            option = OPTION_VARIABLES.get(key)
            raise ServePlanError(f"--env {key}: the launcher owns every SIRCL_* variable; "
                                 + (f"use {option} " if option else "")
                                 + "(its SIRCL options: " + ", ".join(dict.fromkeys(OPTION_VARIABLES.values()))
                                 + "; bundle also --session-groups, --fused-norm)")
        if key in LAUNCHER_KEYS:
            raise ServePlanError(f"--env {key}: the launcher owns this variable ({LAUNCHER_KEYS[key]})")


def parse_env(values: Sequence[str]) -> dict[str, str]:
    """``KEY=VALUE`` items of ``--env``; a later item for the same key wins."""
    result: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator:
            raise ServePlanError(f"--env {item!r} is not KEY=VALUE")
        result[key.strip()] = value
    check_extra_env(result)
    return result


def build_plan(site: ServeSite | Site, profile: ServingProfile, options: Options, *, staged_digest: str,
               library: str) -> ServePlan:
    serve_site = site if isinstance(site, ServeSite) else ServeSite.of(site)
    ring = serve_site.site
    positions = tuple(options.positions)
    tp = profile.tensor_parallel
    if len(positions) != tp:
        raise ServePlanError(f"the profile serves {tp} ranks; {len(positions)} positions were given")
    if len(set(positions)) != len(positions) or any(not 0 <= p < ring.size for p in positions):
        raise ServePlanError(f"positions {list(positions)} must be distinct Sparks of the {ring.size}-Spark ring")
    layout = fabric.Layout.ring(ring.size)
    try:
        group = fabric.describe_group(layout, positions)
    except fabric.FabricError as error:
        raise ServePlanError(str(error)) from None
    run_id = _run_id(options.run_id, positions)
    if not re.fullmatch(r"[0-9a-f]{16}", staged_digest):
        raise ServePlanError(f"staged digest {staged_digest!r} is not 16 hexadecimal digits")
    if not re.fullmatch(r"roce_proxy-[0-9a-f]{16}\.so", library):
        raise ServePlanError(f"native library name {library!r} is not roce_proxy-<16 hex>.so")
    if not 1024 <= options.api_port <= 65535:
        raise ServePlanError("the API port must be between 1024 and 65535")
    master_port = profile.master_port if options.master_port is None else int(options.master_port)
    if master_port == options.api_port or not 1024 <= master_port <= 65535:
        raise ServePlanError("the master port must be a port between 1024 and 65535 other than the API port")
    dispatch = options.capacity if options.dispatch is None else options.dispatch
    for name, value in (("capacity", options.capacity), ("dispatch ceiling", dispatch),
                        ("gather capacity", options.gather)):
        if value < 16 or value % 16:
            raise ServePlanError(f"the SIRCL {name} must be a positive multiple of 16 bytes")
    if dispatch > options.capacity:
        raise ServePlanError("the dispatch ceiling cannot exceed the all-reduce capacity")
    oneshot = oneshot_environment(options.oneshot_max, dispatch)
    large_blocks = large_blocks_environment(options.large_blocks)
    profile_checkpoint = f"{profile.model_repository}@{profile.model_revision}"
    checkpoint = options.checkpoint_id or profile_checkpoint
    recorded = profile.thinking
    if checkpoint != profile_checkpoint:
        name, revision = parse_checkpoint(checkpoint)
        try:
            recorded = profile_mod.thinking_behaviour(profile.repository, name, revision)
        except profile_mod.ProfileError as error:
            raise ServePlanError(str(error)) from None
    thinking, thinking_source = served_thinking(profile.repository, checkpoint, recorded, options.thinking_behaviour)
    chat_defaults = reasoning_kwargs(thinking, options.reasoning_effort, checkpoint)
    recipe, argument_changes = edit_arguments(profile.recipe_arguments, options.vllm_edits)
    dcp = options.dcp_size
    problem = dcp_problem(dcp, tp, checkpoint, recipe, {**profile.ranks[0].environment, **options.extra_env})
    if problem:
        raise ServePlanError(problem)
    if dcp > 1:
        before = _recipe_value(recipe, DCP_FLAG)
        try:
            _replace_flag(recipe, DCP_FLAG, str(dcp))
        except ServePlanError:
            raise ServePlanError(f"--dcp-size {dcp}: the profile's recipe must give {DCP_FLAG} once with a "
                                 "value") from None
        argument_changes.append(ArgumentChange(DCP_FLAG, (before,) if before is not None else None, (str(dcp),),
                                               "--dcp-size"))
    interleave = dcp_interleave(dcp, checkpoint, recipe)
    if interleave is not None:
        recipe += [INTERLEAVE_FLAG, interleave]
        argument_changes.append(ArgumentChange(INTERLEAVE_FLAG, None, (interleave,), "--dcp-size"))
    b12x_cache = check_b12x_cache_dir(options.b12x_cache_dir)
    overlays = overlay_paths(positions, options.overlay, options.overlays)
    if options.spin_limit is not None and not 1 <= options.spin_limit < 1 << 32:
        raise ServePlanError("the spin limit must be a positive 32-bit poll count")
    for name, seconds in (("startup", options.startup_wait), ("serving", options.serving_wait)):
        if not 1e-6 <= seconds <= MAX_WAIT_S:
            raise ServePlanError(f"the {name} wait must be between 1e-6 and {MAX_WAIT_S:.0f} seconds")
    if options.nccl_mode not in NCCL_MODES or options.large_allreduce not in LARGE_MODES:
        raise ServePlanError(f"SIRCL_NCCL must be one of {NCCL_MODES} and SIRCL_LARGE_ALLREDUCE one of "
                             f"{LARGE_MODES}")
    if options.mhc_prefill_shard not in MHC_MODES:
        raise ServePlanError(f"--mhc-prefill-shard must be one of {MHC_MODES}")
    schedules = schedule_environment(options)
    ring_min = ring_min_environment(options.ring_min)
    chain_min = chain_min_environment(options.chain_min)
    links = link_environment(options.link_sizes)
    links.update(link_slots_environment(options.link_slots))
    links.update(ring_gather_stagger_environment(options.ring_gather_stagger))
    check_extra_env(options.extra_env)
    policy, policy_reason = guard.effective_policy(group.nccl_policy, group.nccl_reason,
                                                   nccl_mode=options.nccl_mode,
                                                   environ=profile.ranks[0].environment)
    if options.large_allreduce == "nccl" and not policy.allows("all_reduce"):
        raise ServePlanError(f"SIRCL_LARGE_ALLREDUCE=nccl needs NCCL, but NCCL may not all-reduce on this group "
                             f"({policy_reason})")
    if options.require_no_nccl and policy is not NcclPolicy.NONE:
        raise ServePlanError(f"--require-no-nccl: NCCL may run on {group.fabric.describe()} ({policy_reason}); "
                             "add --nccl never so that SIRCL carries every collective")
    nccl_logging = nccl_debug_environment(options.nccl_debug or options.require_no_nccl, options.extra_env)
    nccl_free = nccl_free_problems({**profile.ranks[0].environment, **options.extra_env},
                                   nccl_mode=options.nccl_mode, required=options.require_no_nccl,
                                   arguments=recipe)
    if nccl_free:
        raise ServePlanError("NCCL communicators would be created outside SIRCL's groups: " + "; ".join(nccl_free))
    batching = micro_batching(recipe)
    if batching:
        raise ServePlanError(f"SIRCL's communicator refuses this profile: {batching}")
    if policy is NcclPolicy.NONE:
        conflicts = relay_conflicts(recipe, {**profile.ranks[0].environment, **options.extra_env})
        if conflicts:
            raise ServePlanError(f"SIRCL's communicator refuses this profile on {group.fabric.describe()}, where "
                                 f"NCCL may not run ({policy_reason}): " + "; ".join(conflicts))
    models = resolve_model_paths(positions, site_paths=serve_site.model_paths, per_spark=options.model_paths,
                                 shared=options.model_path)
    gid = options.gid_index if options.gid_index is not None else ring.gid_index
    if gid is None:
        gid = int(profile.ranks[0].environment.get("NCCL_IB_GID_INDEX", "3"))
    if not 0 <= gid <= 255:
        raise ServePlanError("the GID index must be in 0-255")
    remote_root = f"{ring.remote_dir}/serve"
    seccomp = seccomp_path(ring, profile)
    run_dir = f"{remote_root}/runs/{run_id}"
    master = ring.host(positions[0]).lan_address
    common = {
        "PYTHONPATH": f"{OVERLAY_TARGET}:{SOURCE_TARGET}" if overlays else SOURCE_TARGET,
        "SIRCL_MODE": "custom",
        "SIRCL_FABRIC": layout.describe(),
        "SIRCL_RANK_POSITIONS": ",".join(str(p) for p in positions),
        "SIRCL_GROUPS": ",".join(SESSION_GROUPS_DCP) if dcp > 1 else "tp",
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
    if b12x_cache is not None:
        common[B12X_CACHE_VARIABLE] = b12x_cache
    common.update(nccl_logging)
    sessions = [("tp", group, policy)]
    if dcp > 1:
        first = dcp_groups(layout, positions, dcp)[0]
        dcp_policy, dcp_reason = guard.effective_policy(first.nccl_policy, first.nccl_reason,
                                                        nccl_mode=options.nccl_mode,
                                                        environ=profile.ranks[0].environment)
        sessions.append(("dcp", first, dcp_policy))
        if dcp_policy is NcclPolicy.NONE and policy is not NcclPolicy.NONE:
            conflicts = relay_conflicts(recipe, {**profile.ranks[0].environment, **options.extra_env})
            if conflicts:
                raise ServePlanError(f"SIRCL's communicator refuses this profile's decode-context-parallel groups, "
                                     f"where NCCL may not run ({dcp_reason}): " + "; ".join(conflicts))
    tuning = tuning_plan(options.tuning_tables, sessions, large=options.large_allreduce,
                         shared={"ep": "tp"} if policy is NcclPolicy.NONE else {}, nccl_mode=options.nccl_mode)
    conflicts = tuning.conflicts(common)
    if conflicts:
        raise ServePlanError("; ".join(conflicts))
    common.update(tuning.environment())
    reasons = {**REASONS, **({"PYTHONPATH": OVERLAY_PYTHONPATH_REASON} if overlays else {})}
    launches: list[RankLaunch] = []
    changes: dict[str, EnvironmentChange] = {}
    for rank, position in enumerate(positions):
        container = profile.ranks[rank]
        host = ring.host(position)
        environment = dict(container.environment)
        owned = [key for key in OWNED if key in environment]
        if owned:
            raise ServePlanError(f"the profile sets {owned}; the launcher must own these variables")
        overrides = dict(common)
        overrides["VLLM_HOST_IP"] = host.lan_address
        overrides["GLOO_SOCKET_IFNAME"] = ring.lan_interface
        overrides["NCCL_SOCKET_IFNAME"] = ring.lan_interface
        plugins = [item for item in environment.get("VLLM_PLUGINS", "").split(",") if item]
        overrides["VLLM_PLUGINS"] = ",".join(plugins + ([] if "sircl" in plugins else ["sircl"]))
        if policy.allows("all_reduce"):
            overrides["NCCL_IB_HCA"] = nccl_hca_value(environment.get("NCCL_IB_HCA"), rank_devices(group, rank))
        if options.mhc_prefill_shard == "off":
            overrides[MHC_SHARD] = "0"
        for key, value in overrides.items():
            before = environment.get(key)
            if before != value and key not in changes:
                reason = reasons.get(key, "SIRCL setting" if key.startswith("SIRCL_") else "launcher setting")
                changes[key] = EnvironmentChange(key, before, PER_RANK.get(key, value), reason)
            environment[key] = value
        for key, value in options.extra_env.items():
            if key not in changes:
                changes[key] = EnvironmentChange(key, environment.get(key), value, "--env")
            environment[key] = value
        command, _ = edit_arguments(container.command, options.vllm_edits)
        _replace_flag(command, "--master-addr", master)
        _replace_flag(command, "--port", str(options.api_port))
        _replace_flag(command, "--master-port", str(master_port))
        if dcp > 1:
            _replace_flag(command, DCP_FLAG, str(dcp))
        if interleave is not None:
            command += [INTERLEAVE_FLAG, interleave]
        if rank == 0 and chat_defaults is not None:
            with_chat_template_kwargs(command, chat_defaults)
        if (HEADLESS in command) != bool(rank):
            raise ServePlanError(f"rank {rank}: --headless must be set on every rank but 0")
        model, model_source = models[position]
        cache = options.cache_paths.get(position, options.cache_path or f"{remote_root}/cache/{profile.id}")
        cache_source = _path(cache, "cache directory")
        mounts = (
            Mount(_path(model, "model directory"), MODEL_TARGET, True),
            Mount(cache_source, CACHE_TARGET, False),
            *((Mount(overlays[position], OVERLAY_TARGET, True),) if overlays else ()),
            Mount(f"{remote_root}/src/{staged_digest}", SOURCE_TARGET, True),
            Mount(f"{ring.remote_dir}/build-cache", BUILD_TARGET, True),
            Mount(run_dir, RUN_TARGET, False),
        )
        health = None
        if container.health is not None:
            documented, requested = f":{profile.api_port}/", f":{options.api_port}/"
            if not any(documented in item for item in container.health.command):
                raise ServePlanError(f"rank {rank}: the health check does not name port {profile.api_port}")
            health = tuple(item.replace(documented, requested) for item in container.health.command)
        launches.append(RankLaunch(
            rank=rank, position=position, host=host.name, ssh=host.ssh, lan_address=host.lan_address,
            docker=host.docker, container=f"{CONTAINER_PREFIX}-{run_id}-r{rank}", image=profile.image_id,
            entrypoint=container.entrypoint, command=tuple(command), environment=environment, mounts=mounts,
            labels={LABEL: run_id, f"{LABEL}.rank": str(rank), f"{LABEL}.position": str(position)},
            run_options=_run_options(container, seccomp, health), health=health,
            directories=(cache_source, run_dir, f"{run_dir}/receipts"), model_source=model_source,
            sudo=serve_site.sudo.get(position, ""), sudo_source=serve_site.sudo_sources.get(position, "default"),
        ))
    shards = {launch.environment.get(MHC_SHARD, "0").strip() not in ("", "0") for launch in launches}
    if len(shards) != 1:
        raise ServePlanError(f"the profile's ranks disagree on {MHC_SHARD}")
    problem = mhc_dcp_problem(tp, dcp, next(iter(shards)))
    if problem:
        raise ServePlanError(problem)
    if len({launch.environment.get(HC_PREFILL) for launch in launches}) != 1:
        raise ServePlanError(f"the profile's ranks disagree on {HC_PREFILL}")
    given_schedules = {attribute: getattr(options, attribute) for attribute, _ in SCHEDULE_VARIABLES
                       if getattr(options, attribute) is not None}
    rows = [session_settings(group, name="tp", groups="", scoped=True, schedules=given_schedules,
                             link_sizes=options.link_sizes, link_slots=options.link_slots,
                             ring_gather_stagger=options.ring_gather_stagger, table_settings=tuning.settings_of("tp"))]
    if dcp > 1:
        rows.append(session_settings(dcp_groups(layout, positions, dcp)[0], name="dcp", groups="", scoped=False,
                                     table_settings=tuning.settings_of("dcp")))
    session_problems(rows)
    return ServePlan(
        run_id=run_id, profile=profile, site=ring, positions=positions, group=group,
        staged_digest=staged_digest, library=library, api_port=options.api_port, master_port=master_port,
        capacity=options.capacity, dispatch=dispatch, gather=options.gather, spin_limit=options.spin_limit,
        ranks=tuple(launches), changes=tuple(changes[key] for key in sorted(changes)),
        nccl_policy=policy, nccl_policy_reason=policy_reason, nccl_mode=options.nccl_mode,
        large_allreduce=options.large_allreduce, mhc_prefill_shard=shards.pop(),
        startup_wait=float(options.startup_wait), serving_wait=float(options.serving_wait),
        link_sizes=dict(options.link_sizes), link_slots=options.link_slots, schedules=given_schedules,
        ring_gather_stagger=options.ring_gather_stagger,
        tuning=tuning,
        oneshot_max=options.oneshot_max, ring_min=options.ring_min,
        chain_min=options.chain_min, large_blocks=options.large_blocks, reasoning_effort=options.reasoning_effort,
        chat_template_defaults=chat_defaults, nccl_debug=bool(nccl_logging), require_no_nccl=options.require_no_nccl,
        argument_changes=tuple(argument_changes), vllm_arguments=tuple(recipe) if argument_changes else (),
        dcp_size=dcp,
        overlay=overlays, checkpoint_id=options.checkpoint_id if checkpoint != profile_checkpoint else None,
        thinking=thinking, thinking_source=thinking_source, b12x_cache_dir=b12x_cache,
    )


def build_plans(site: ServeSite | Site, profile: ServingProfile, groups: Sequence[Sequence[int]],
                options: Options, *, staged_digest: str, library: str) -> tuple[ServePlan, ...]:
    """One instance per group, on disjoint Sparks, with consecutive API and master ports.

    Instance ``k`` listens on ``options.api_port + k`` and rendezvouses on the
    master port (the profile's unless given) ``+ k``. With one group the run
    id is ``options.run_id`` or ``tp<N>-<first>-<last>``; with several,
    ``options.run_id`` (default ``tp<N>``) is the prefix of
    ``<prefix>-<first>-<last>``.
    """
    groups = tuple(tuple(int(p) for p in group) for group in groups)
    if not groups:
        raise ServePlanError("no group of Sparks given")
    used = [position for group in groups for position in group]
    shared = sorted({position for position in used if used.count(position) > 1})
    if shared:
        raise ServePlanError(f"Sparks {shared} appear in more than one group; every Spark serves one instance")
    master = profile.master_port if options.master_port is None else int(options.master_port)
    plans = []
    for index, positions in enumerate(groups):
        if len(groups) == 1:
            run_id = options.run_id
        else:
            run_id = f"{options.run_id or f'tp{len(positions)}'}-{positions[0]}-{positions[-1]}"
        instance = dataclasses.replace(options, positions=positions, run_id=run_id,
                                       api_port=options.api_port + index, master_port=master + index)
        plans.append(build_plan(site, profile, instance, staged_digest=staged_digest, library=library))
    ports = [port for plan in plans for port in (plan.api_port, plan.master_port)]
    if len(set(ports)) != len(ports):
        raise ServePlanError(f"the instances' API and master ports overlap: {ports}; choose --api-port and "
                             "--master-port further apart")
    return tuple(plans)


# -- expected cost of split prefill all-reduces ----------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ModelShape:
    hidden: int
    layers: int
    draft_allreduces: int


# GLM-5.3-Flash (model type glm5_next): hidden size 4,096 and 45 decoder
# layers, the defaults of transformers_utils/configs/glm5_next.py:20,23 in the
# serving image's vLLM. Each forward of T tokens all-reduces [T, 4096] BF16
# once for the embedding and twice per decoder layer (attention output
# projection; MoE or dense MLP output). The one-layer MTP draft runs over the
# same T tokens during prefill and all-reduces its embedding, attention output
# and MoE output (adapter SURVEY.md, section 3).
MODEL_SHAPES = {"local-inference-lab/GLM-5.3-Flash-NVFP4-Spark": ModelShape(4096, 45, 3)}
# vLLM shards mHC rows of full chunks only: a 4,096- or 8,192-token ceiling at
# TP2 or TP4 (mhc_prefill_sharding.py:108-139 and pure_prefill_metadata).
MHC_CHUNK_ROWS = (4096, 8192)
MHC_WORLDS = (2, 4)
# Prefill rates of the profiles with their own transports, medians of two runs
# on installer image dev-20261004-kraken (profile.json evidence scope): 64K-token
# prompts at 3,525 tokens/s on a four-Spark ring (TP4) and 2,403 tokens/s on
# two Sparks of a four-Spark ring (TP2). They include those deployments'
# transfer time, so they bound the compute time from above.
PROFILE_PREFILL_RATES = {"glm53-flash-nvfp4-spark-tp4": 3525.0, "glm53-flash-nvfp4-spark-tp2": 2403.0}
GLM53_FLASH_HIDDEN = 4096
GLM53_FLASH_LAYERS = 45
GLM53_FLASH_DRAFT_ALLREDUCES = 3
CYCLE_PREFILL_TOKENS_PER_SECOND = PROFILE_PREFILL_RATES["glm53-flash-nvfp4-spark-tp4"]
# One-shot all-reduce ops of 131,072 bytes on Sparks 0-3 of the eight-Spark
# ring (ring harness, image aba309e4610c): 48-50 us per op at p50, eager and in
# graph replay. Large-message ops move the same bytes in fewer, larger ops, so
# the estimate uses this rate only as an upper bound on their time.
ONESHOT_OP_BYTES = 131072
ONESHOT_OP_US = (48.0, 50.0)
# Compute per prompt token with mHC row ownership off, measured with this
# launcher on Sparks 0-3 of the eight-Spark ring (image aba309e4610c, one-shot
# ops of at most 131,072 bytes): a 17,055-token prompt answered in 9.37 s; its
# 100,204 one-shot ops at 48-50 us leave 4.36-4.56 s of compute.
PATH_COMPUTE_SECONDS_PER_TOKEN = {"glm53-flash-nvfp4-spark-tp4": 0.26e-3}
# Mixing time of the mHC streams of one 8,192-row chunk on every row,
# inferred from an eight-way sharding measurement of 8K-token prefill (3,594
# to 4,563 tokens/s: 0.484 s saved per chunk by mixing 7/8 fewer rows). Row
# ownership over W ranks saves (1 - 1/W) of it.
MHC_FULL_CHUNK_SECONDS = 0.55
MHC_REFERENCE_ROWS = 8192
# The most link chunks one rank's chunk of a chain reduce-scatter may span: the
# session's arrival counters per chunk owner (PIECE_COUNTERS in
# oneshot/_links_cute.py).
LINK_PIECE_COUNTERS = 65536


def _at_least(value: object, smallest: int) -> bool:
    """``value`` is an integer of at least ``smallest`` (a boolean is not)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= smallest


def _runs_as(schedule: object, ring: bool) -> object:
    """A session schedule as the session runs it at every size: ``ring`` without a ring runs as ``auto``."""
    return "auto" if schedule == "ring" and not ring else schedule


def _minimums(stats: Mapping[str, Any], per_collective: str, shared: str) -> dict[str, int]:
    """A session's minimum of each collective (:data:`MIN_COLLECTIVES`): its per-collective map
    (``chain_mins``, ``ring_mins``), else the size it states for all three (``chain_min_bytes``,
    ``ring_min_bytes``), else 0."""
    single = stats.get(shared)
    base = single if _at_least(single, 0) else 0
    mapping = stats.get(per_collective)
    mapping = mapping if isinstance(mapping, Mapping) else {}
    return {collective: mapping[collective] if _at_least(mapping.get(collective), 0) else base
            for collective in MIN_COLLECTIVES}


def _link_clause(collective: str, ring_from: int | None, chain_min: int | None, pieces: str) -> str:
    text = pieces
    if chain_min is not None:
        text = f"one chain op per {collective} from {chain_min:,} B, else " + text
    if ring_from is not None:
        text = f"one ring op per {collective} from {ring_from:,} B, else " + text
    return text


@dataclasses.dataclass(frozen=True)
class SessionOps:
    """How a session cuts collectives into ops, read from its statistics (a receipt's ``session_stats``).

    Every byte threshold is a size of the whole message: an all-reduce's
    tensor, a reduce-scatter's input, an all-gather's output. Each collective
    has a chain and a ring minimum of its own: the session's ``chain_mins``
    and ``ring_mins`` (keys ``reduce``, ``gather``, ``scatter``), else the size
    it states for all three (``chain_min_bytes``, ``ring_min_bytes``), else 0.
    A ``ring`` schedule runs a collective as a ring op when the session's ring
    runs and the message is at least the collective's ring minimum
    (``ring_min``, ``gather_ring_min``, ``scatter_ring_min``); otherwise it
    runs as ``auto``.

    - All-reduce: one one-shot op up to the dispatch ceiling. Above it, under
      a ``ring`` schedule from ``ring_min`` bytes (``reduce_ring``), one ring
      op for the largest prefix of W chunks of whole 16-byte packs, its
      remainder in ops of at most ``large_piece`` bytes; otherwise one chain
      op from ``chain_min`` bytes when the session chains
      (``chain_available`` and a schedule other than ``pieces``), else ops of
      at most ``large_piece`` bytes. A tail below 16 bytes adds one op.
    - Reduce-scatter (prefill row ownership): the session's ``reduce_scatter``
      takes the whole message when the session has one (``scatter_op_bytes``
      in its statistics); otherwise the adapter carries it as an all-reduce
      and keeps this rank's rows, and ``scatter_piece`` is None. The session
      runs one ring op under a ``ring`` schedule from ``scatter_ring_min``
      bytes; one
      chain op when its links run (``link_available``), the schedule counts as
      ``auto`` or ``chain``, the input is at least ``scatter_chain_min`` bytes
      and each rank's chunk spans at most :data:`LINK_PIECE_COUNTERS` pieces of
      the reduce-scatter's link piece (``link_chunks["scatter"]``, else
      ``link_chunk_bytes``); otherwise ops of at most ``scatter_piece`` bytes
      of input (``scatter_op_bytes``, each rank's share capped at the
      session's relay-safe size).
    - All-gather (along the first dimension): a shard of at most
      ``gather_piece`` bytes is one op. ``all_gather_large`` runs one ring op
      under a ``ring`` schedule from ``gather_ring_min`` bytes; one chain op
      when the
      links run, the schedule counts as ``auto`` or ``chain`` and the output
      is at least ``gather_chain_min`` bytes; otherwise ops of at most
      ``gather_piece`` bytes of whole rows.

    A ring or chain op of a reduce-scatter or all-gather needs each rank's
    share to be a nonzero multiple of 16 bytes and the whole message to be
    below 2 GiB. Its chain threshold is the collective's chain minimum under
    ``auto`` (and below its ring minimum under ``ring``) and the smallest such
    message (16 bytes per rank) under ``chain``. The link pieces of
    all-gathers and ring all-reduces do not change how many ops they take.
    Nothing here assumes a piece size or a schedule: each comes from the
    session.
    """

    dispatch: int
    large_piece: int
    gather_piece: int
    chain_min: int | None = None
    reduce_ring: bool = False
    gather_ring: bool = False
    world: int | None = None
    gather_chain_min: int | None = None
    scatter_piece: int | None = None
    scatter_ring: bool = False
    scatter_chain_min: int | None = None
    scatter_link_chunk: int | None = None
    ring_min: int = 0                       # the all-reduce's ring minimum
    gather_ring_min: int = 0
    scatter_ring_min: int = 0

    @classmethod
    def from_stats(cls, stats: object) -> "SessionOps | None":
        if not isinstance(stats, Mapping):
            return None
        values = [stats.get(name) for name in ("dispatch_limit_bytes", "large_piece_bytes", "gather_piece_bytes")]
        if not all(_at_least(value, 16) for value in values):
            return None
        world = stats.get("world_size") if _at_least(stats.get("world_size"), 1) else None
        ring = stats.get("ring_available") is True and world is not None
        links = stats.get("link_available") is True and world is not None
        chain_mins = _minimums(stats, "chain_mins", "chain_min_bytes")
        ring_mins = _minimums(stats, "ring_mins", "ring_min_bytes")

        def link_chain_min(schedule: object, collective: str) -> int | None:
            """The smallest message a link schedule carries in one chain op where no ring op takes it."""
            if not links or schedule not in ("auto", "chain", "ring"):
                return None
            return 16 * world if schedule == "chain" else max(16 * world, chain_mins[collective])

        large = _runs_as(stats.get("large_schedule", "pieces"), ring)
        chain_min = None
        if stats.get("chain_available") is True and large != "pieces":
            chain_min = 16 if large == "chain" else max(16, chain_mins["reduce"])
        gather = _runs_as(stats.get("gather_schedule", "pieces"), ring)
        scatter_piece = scatter_chain_min = scatter_link_chunk = None
        scatter_ring = False
        operation = stats.get("scatter_op_bytes")
        if world is not None and _at_least(operation, 16 * world):
            share = operation // world // 16 * 16
            safe = stats.get("relay_safe_bytes")
            if _at_least(safe, 16):
                share = min(share, safe // 16 * 16)
            scatter_piece = share * world
            scatter = _runs_as(stats.get("scatter_schedule", "pieces"), ring)
            scatter_ring = scatter == "ring"
            scatter_chain_min = link_chain_min(scatter, "scatter")
            pieces = stats.get("link_chunks")
            piece = pieces.get("scatter") if isinstance(pieces, Mapping) else None
            if not _at_least(piece, 16):
                piece = stats.get("link_chunk_bytes")
            scatter_link_chunk = piece if _at_least(piece, 16) else None
        return cls(values[0], values[1], values[2], chain_min, large == "ring", gather == "ring", world,
                   link_chain_min(gather, "gather"), scatter_piece, scatter_ring, scatter_chain_min,
                   scatter_link_chunk, ring_mins["reduce"], ring_mins["gather"], ring_mins["scatter"])

    def _ring_from(self, ring: bool, ring_min: int) -> int | None:
        """The smallest message a running ring schedule carries as a ring op, or None."""
        return max(ring_min, 16 * self.world) if ring and self.world else None

    @staticmethod
    def _chain_below(ring: bool, chain_min: int | None, ring_min: int) -> int | None:
        """``chain_min`` where chain ops can take a message: always without a ring, below ``ring_min`` with."""
        return chain_min if chain_min is not None and (not ring or chain_min < ring_min) else None

    def allreduce_ops(self, nbytes: int) -> int:
        if nbytes <= self.dispatch:
            return 1
        aligned = nbytes // 16 * 16
        tail = int(aligned < nbytes)
        if self.reduce_ring and self.world and nbytes >= self.ring_min:
            span = 16 * self.world
            ring_bytes = aligned // span * span
            if ring_bytes:
                return 1 + math.ceil((aligned - ring_bytes) / self.large_piece) + tail
            return math.ceil(aligned / self.large_piece) + tail
        if self.chain_min is not None and aligned and aligned >= self.chain_min:
            return 1 + tail
        return math.ceil(aligned / self.large_piece) + tail

    def _link_message(self, whole: int) -> bool:
        """A ring or chain op can carry a whole message of ``whole`` bytes."""
        return self.world is not None and 0 < whole < 1 << 31 and whole % (16 * self.world) == 0

    def scatter_ops(self, nbytes: int) -> int:
        """Ops of a reduce-scatter of ``nbytes`` of input on a session with a reduce-scatter."""
        if self.scatter_piece is None:
            raise ValueError("the session has no reduce-scatter; the adapter carries reduce-scatters as all-reduces")
        if self._link_message(nbytes):
            if self.scatter_ring and nbytes >= self.scatter_ring_min:
                return 1
            chunk = nbytes // self.world
            piece = self.scatter_link_chunk
            if (self.scatter_chain_min is not None and nbytes >= self.scatter_chain_min
                    and (piece is None or -(-chunk // piece) <= LINK_PIECE_COUNTERS)):
                return 1
        return -(-nbytes // self.scatter_piece)

    def gather_ops(self, rows: int, row_bytes: int) -> int:
        """Ops of an all-gather of ``rows`` rows of ``row_bytes`` bytes from every rank."""
        whole = rows * row_bytes * (self.world or 0)
        if self._link_message(whole):
            if self.gather_ring and whole >= self.gather_ring_min:
                return 1
            if self.gather_chain_min is not None and whole >= self.gather_chain_min:
                return 1
        if row_bytes <= self.gather_piece:
            return math.ceil(rows / (self.gather_piece // row_bytes))
        return rows * math.ceil(row_bytes / self.gather_piece)

    def describe(self) -> str:
        """One clause per collective, separated by semicolons: all-reduce, reduce-scatter, all-gather."""
        reduce = _link_clause("all-reduce", self._ring_from(self.reduce_ring, self.ring_min),
                              self._chain_below(self.reduce_ring, self.chain_min, self.ring_min),
                              f"all-reduce pieces of {self.large_piece:,} B above {self.dispatch:,} B")
        scatter = ("reduce-scatters carried as all-reduces" if self.scatter_piece is None else
                   _link_clause("reduce-scatter", self._ring_from(self.scatter_ring, self.scatter_ring_min),
                                self._chain_below(self.scatter_ring, self.scatter_chain_min, self.scatter_ring_min),
                                f"reduce-scatter pieces of {self.scatter_piece:,} B"))
        gather = _link_clause("all-gather", self._ring_from(self.gather_ring, self.gather_ring_min),
                              self._chain_below(self.gather_ring, self.gather_chain_min, self.gather_ring_min),
                              f"all-gather pieces of {self.gather_piece:,} B")
        return "; ".join((reduce, scatter, gather))


@dataclasses.dataclass(frozen=True)
class PrefillEstimate:
    """The collectives and compute of one prompt's prefill.

    Every chunk reduces its ``[rows, hidden]`` activations
    ``reductions_per_chunk`` times. In a chunk whose mHC rows are sharded the
    layers' reductions are reduce-scatters, and as many all-gathers of row
    shards follow. ``allreduces`` lists (count, bytes) of the all-reduces, the
    full chunk's size first; ``scatters`` lists (count, bytes of input) of the
    reduce-scatters; ``gathers`` lists (count, rows, row bytes) of the
    all-gather shards. How many session ops they take depends on the session
    (:class:`SessionOps`, from the receipts), so the time per op is priced
    from a measurement (:meth:`implied_op_seconds`); before one, the measured
    one-shot rate bounds the collectives' time from above.
    """

    prompt_tokens: int
    chunk_tokens: int
    chunks: int
    reductions_per_chunk: int
    allreduces: tuple[tuple[int, int], ...]
    scatters: tuple[tuple[int, int], ...]
    gathers: tuple[tuple[int, int, int], ...]
    compute_seconds: float | None          # measured rate on the path, else the profile's bound
    sharded_chunks: int = 0                # chunks whose mHC rows are sharded
    mhc_saving_seconds: float = 0.0        # mixing time row ownership saves, already in compute_seconds
    compute_measured: bool = False

    @property
    def allreduce_calls(self) -> int:
        return sum(count for count, _ in self.allreduces)

    @property
    def scatter_calls(self) -> int:
        return sum(count for count, _ in self.scatters)

    @property
    def gather_calls(self) -> int:
        return sum(count for count, _, _ in self.gathers)

    def ops(self, session: SessionOps) -> tuple[int, int, int]:
        """(all-reduce, reduce-scatter, all-gather) ops this prefill takes on ``session``.

        On a session without a reduce-scatter the adapter carries each
        reduce-scatter as an all-reduce, so its ops count as all-reduce ops.
        """
        reduce_ops = sum(count * session.allreduce_ops(nbytes) for count, nbytes in self.allreduces)
        scatter_ops = 0
        for count, nbytes in self.scatters:
            if session.scatter_piece is None:
                reduce_ops += count * session.allreduce_ops(nbytes)
            else:
                scatter_ops += count * session.scatter_ops(nbytes)
        gather_ops = sum(count * session.gather_ops(rows, row_bytes) for count, rows, row_bytes in self.gathers)
        return reduce_ops, scatter_ops, gather_ops

    @property
    def oneshot_ops(self) -> int:
        """Ops if every collective were cut into one-shot ops of ``ONESHOT_OP_BYTES`` (a reduce-scatter
        as an all-reduce of its input)."""
        return (sum(count * math.ceil(nbytes / ONESHOT_OP_BYTES) for count, nbytes in self.allreduces + self.scatters)
                + sum(count * math.ceil(rows * row_bytes / ONESHOT_OP_BYTES)
                      for count, rows, row_bytes in self.gathers))

    @property
    def collective_bound_seconds(self) -> tuple[float, float]:
        """The collectives' time at the measured one-shot rate: an upper bound for large-message ops."""
        return self.oneshot_ops * ONESHOT_OP_US[0] / 1e6, self.oneshot_ops * ONESHOT_OP_US[1] / 1e6

    @property
    def bound_seconds(self) -> tuple[float, float] | None:
        """Time to first token with the collectives at their one-shot bound."""
        if self.compute_seconds is None:
            return None
        low, high = self.collective_bound_seconds
        return self.compute_seconds + low, self.compute_seconds + high

    def implied_op_seconds(self, measured_seconds: float, session: SessionOps) -> float | None:
        """Mean time per session op implied by a measured prefill, compute at its estimate."""
        total = sum(self.ops(session))
        if self.compute_seconds is None or not total:
            return None
        return (measured_seconds - self.compute_seconds) / total

    def describe(self) -> str:
        full = self.allreduces[0][1] if self.allreduces else 0
        text = (f"{self.prompt_tokens:,}-token prompt: {self.chunks} prefill chunk(s) of up to "
                f"{self.chunk_tokens:,} tokens, ")
        if self.sharded_chunks:
            _, rows, row_bytes = self.gathers[0]
            text += (f"{self.reductions_per_chunk} reductions per chunk ({full / 2**20:g} MiB in a full chunk); "
                     f"with mHC row ownership in {self.sharded_chunks} chunk(s) they are {self.allreduce_calls:,} "
                     f"all-reduces and {self.scatter_calls:,} reduce-scatters (the session's reduce-scatter where it "
                     f"has one, otherwise all-reduces and this rank's rows), plus {self.gather_calls:,} all-gathers "
                     f"of {rows * row_bytes / 2**20:g} MiB shards")
        else:
            text += (f"{self.allreduce_calls:,} all-reduces ({self.reductions_per_chunk} per chunk, "
                     f"{full / 2**20:g} MiB in a full chunk)")
        text += ("; the session's large-message ops carry them (op sizes in the receipts' session statistics; "
                 "check --long-prompt prices each op)")
        low, high = self.collective_bound_seconds
        bound = (f"at most {low:.1f}-{high:.1f} s at the one-shot rate ({ONESHOT_OP_US[0]:g}-{ONESHOT_OP_US[1]:g} "
                 f"us per {ONESHOT_OP_BYTES:,} B)")
        if self.compute_seconds is None:
            return text + f"; collectives {bound}; compute unknown for this profile"
        compute = (f"compute about {self.compute_seconds:.1f} s (measured on the path)" if self.compute_measured
                   else f"compute at most {self.compute_seconds:.1f} s")
        if self.mhc_saving_seconds:
            compute += f", {self.mhc_saving_seconds:.1f} s less mHC mixing with row ownership"
        first, last = self.bound_seconds
        return text + f"; {compute}; collectives {bound}; time to first token at most {first:.1f}-{last:.1f} s"


def prefill_estimate(prompt_tokens: int, *, chunk_tokens: int, hidden: int = GLM53_FLASH_HIDDEN,
                     layers: int = GLM53_FLASH_LAYERS, draft_allreduces: int = GLM53_FLASH_DRAFT_ALLREDUCES,
                     element_bytes: int = 2, prefill_rate: float | None = CYCLE_PREFILL_TOKENS_PER_SECOND,
                     mhc_world: int | None = None, compute_per_token: float | None = None) -> PrefillEstimate:
    """The collectives and compute of one prompt's prefill in chunks of ``chunk_tokens``.

    Without mHC row ownership a chunk all-reduces ``[rows, hidden]`` once for
    the embedding, twice per layer and ``draft_allreduces`` times in the MTP
    draft. With it (``mhc_world`` ranks, chunks of exactly ``chunk_tokens``
    rows) the ``2 * layers`` layer reductions are reduce-scatters of
    ``[rows, hidden]`` (the session's reduce-scatter where it has one,
    otherwise all-reduces and this rank's rows), and ``2 * layers``
    all-gathers of ``[rows / W, hidden]`` shards follow.

    Compute is ``compute_per_token`` per prompt token when given (measured
    with row ownership off), else the time at the profile's ``prefill_rate``
    (an upper bound); row ownership subtracts ``(1 - 1/W)`` of the mixing time
    of every sharded chunk.
    """
    if prompt_tokens < 1 or chunk_tokens < 1:
        raise ValueError("prompt and chunk sizes must be positive")
    chunks = math.ceil(prompt_tokens / chunk_tokens)
    per_chunk = 1 + 2 * layers + draft_allreduces
    reduces: dict[int, int] = {}
    scatters: dict[int, int] = {}
    gathers: dict[tuple[int, int], int] = {}
    sharded = 0
    for index in range(chunks):
        rows = min(chunk_tokens, prompt_tokens - index * chunk_tokens)
        nbytes = rows * hidden * element_bytes
        if mhc_world and rows == chunk_tokens:
            reduces[nbytes] = reduces.get(nbytes, 0) + per_chunk - 2 * layers
            scatters[nbytes] = scatters.get(nbytes, 0) + 2 * layers
            shard = (rows // mhc_world, hidden * element_bytes)
            gathers[shard] = gathers.get(shard, 0) + 2 * layers
            sharded += 1
        else:
            reduces[nbytes] = reduces.get(nbytes, 0) + per_chunk
    saving = 0.0
    if sharded:
        saving = sharded * MHC_FULL_CHUNK_SECONDS * chunk_tokens / MHC_REFERENCE_ROWS * (1 - 1 / mhc_world)
    if compute_per_token is not None:
        compute = prompt_tokens * compute_per_token - saving
    elif prefill_rate is not None:
        compute = prompt_tokens / prefill_rate - saving
    else:
        compute = None
    return PrefillEstimate(
        prompt_tokens, chunk_tokens, chunks, per_chunk,
        tuple((count, nbytes) for nbytes, count in sorted(reduces.items(), reverse=True)),
        tuple((count, nbytes) for nbytes, count in sorted(scatters.items(), reverse=True)),
        tuple((count, rows, row_bytes) for (rows, row_bytes), count in gathers.items()),
        compute, sharded, saving, compute_per_token is not None)


# -- text ------------------------------------------------------------------------------------------


def documented_model_dir(profile: ServingProfile) -> str:
    return next(mount.source for mount in profile.ranks[0].mounts if mount.target == MODEL_TARGET)


def checkpoint_text(plan: ServePlan) -> str:
    """The plan's line about the served checkpoint and what preflight checks of it."""
    profile = plan.profile
    if plan.foreign_checkpoint:
        return (f"  checkpoint {plan.checkpoint} (--checkpoint-id), not the profile's "
                f"{profile.model_repository}@{profile.model_revision}: the profile's manifest does not describe it, "
                "so preflight checks that every model directory holds config.json and model.safetensors.index.json "
                "and that their SHA-256 digests, file count and bytes agree across the Sparks")
    return (f"  checkpoint {profile.model_repository} at {profile.model_revision}: {len(profile.checkpoint_files)} "
            f"files, {profile.checkpoint_bytes / 2**30:.1f} GiB. The profile's Compose files mount "
            f"{documented_model_dir(profile)} (documentation path); SparkRing's installer keeps it at "
            f"/srv/sparkring/<cluster>/checkpoints/{profile.checkpoint_dir_name}")


def transport_summary(plan: ServePlan) -> str:
    if not plan.nccl_policy.allows("all_reduce"):
        return "every collective runs on SIRCL"
    if plan.large_on_nccl:
        return (f"collectives up to the dispatch ceiling and every captured one run on SIRCL; NCCL carries "
                f"larger eager all-reduces (SIRCL_LARGE_ALLREDUCE={plan.large_allreduce})")
    return "every all-reduce runs on SIRCL (SIRCL_LARGE_ALLREDUCE=sircl); NCCL may carry other collectives"


def relay_lines(plan: ServePlan) -> list[str]:
    rows = relay_rows(plan.group)
    if not rows:
        return ["  relays: none; every lane is one cable between its two ranks"]
    lines = ["  relays (Sparks a lane crosses between its two ranks):"]
    for row in rows:
        (a, b), (pa, pb) = row["ranks"], row["positions"]
        through = ", ".join(str(p) for p in row["relays"])
        noun = "position" if len(row["relays"]) == 1 else "positions"
        lines.append(f"    ranks {a}-{b} (positions {pa}-{pb}): {row['lanes']} lanes through {noun} {through}")
    lines.append(f"    at most {plan.group.max_relays()} relays per lane; relay-load factor "
                 f"{plan.group.relay_factor():g}")
    return lines


def render_text(plan: ServePlan) -> str:
    profile = plan.profile
    group = plan.group
    lines = [
        f"serve run {plan.run_id}: profile {profile.id} ({profile.served_model_name}) on Sparks "
        f"{list(plan.positions)} of the {plan.site.size}-Spark ring",
        f"  image {profile.image_id} ({profile.image_reference}, release {profile.release})",
        f"  the profile is qualified on topology {profile.topology or 'unspecified'}; this group is "
        f"{group.fabric.describe()} with NCCL policy {plan.nccl_policy.value} ({plan.nccl_reason}); "
        f"{transport_summary(plan)}",
        f"  SIRCL sizes: all-reduce capacity {plan.capacity} B, dispatch ceiling {plan.dispatch} B, "
        f"{oneshot_text(plan.oneshot_max, plan.derived_oneshot_max)}, {large_blocks_text(plan.large_blocks)}, "
        f"all-gather capacity "
        f"{plan.gather} B; larger all-reduces and "
        f"all-gathers run in the session's large-message ops, "
        f"whose sizes the session chooses (receipts: large_piece, gather_piece); " + POST_ORDER_TEXT,
        *session_lines(plan.sessions()),
        *plan.tuning.lines(),
        f"  SIRCL flag waits: startup regime up to {plan.startup_wait:g} s (setup, warm-up, graph capture, "
        f"profiling, sleep and wake-up), serving regime up to {plan.serving_wait:g} s from the first step after "
        f"warm-up; " + (f"SIRCL_SPIN_LIMIT={plan.spin_limit} (--spin-limit; time-limited waits ignore it)"
                        if plan.spin_limit is not None else "SIRCL_SPIN_LIMIT unset (the session's poll budget)"),
        f"  API http://{plan.api_rank.lan_address}:{plan.api_port}/v1 (rank 0 on {plan.api_rank.host}); "
        f"torch.distributed master {plan.api_rank.lan_address}:{plan.master_port}",
        f"  remote: package {plan.source_dir}, native library {plan.build_dir}/{plan.library}, "
        f"receipts {plan.receipt_dir}, seccomp policy {plan.seccomp_path}",
        checkpoint_text(plan),
    ]
    for rank in range(len(plan.positions)):
        lines.append(f"  derived route map, rank {rank} (position {plan.positions[rank]}): "
                     f"{fabric.format_routes(group.route_map(rank))}")
    lines += relay_lines(plan)
    lines.append("  environment changes against the profile's containers:")
    for change in plan.changes:
        before = "unset" if change.before is None else repr(change.before)
        lines.append(f"    {change.name}: {before} -> {change.after!r}  ({change.reason})")
    defaults = (f"; {CHAT_TEMPLATE_FLAG} '{json.dumps(dict(plan.chat_template_defaults), separators=(',', ':'))}' "
                "added on rank 0 (--reasoning-effort)" if plan.chat_template_defaults is not None else "")
    lines.append(f"  vLLM arguments changed: --port {profile.api_port} -> {plan.api_port}; --master-addr -> "
                 f"{plan.api_rank.lan_address}; --master-port {profile.master_port} -> {plan.master_port}{defaults}")
    lines += vllm_edit_text(plan.argument_changes)
    lines.append(thinking_text(plan))
    lines += overlay_text(plan)
    lines.append(b12x_cache_text(plan))
    if plan.mhc_prefill_shard:
        carrier = ("SIRCL's reduce-scatter and all-gather (shim mhc_prefill_shard)"
                   if not plan.nccl_policy.allows("all_reduce") else "vLLM's PyNccl path (NCCL may run here)")
        lines.append(f"  mHC prefill sharding: on ({MHC_SHARD} of the profile), carried by {carrier}")
    else:
        lines.append("  mHC prefill sharding: off")
    hc_mode = plan.hc_prefill_mode
    if hc_mode == "shard":
        carrier = ("SIRCL's reduce-scatter and all-gather (shim qwen_hc_prefill_shard)"
                   if not plan.nccl_policy.allows("all_reduce") else "vLLM's PyNccl path (NCCL may run here)")
        lines.append(f"  hyper-connection prefill row ownership: shard ({HC_PREFILL}), carried by {carrier}")
    elif hc_mode is not None:
        lines.append(f"  hyper-connection prefill row ownership: {hc_mode} ({HC_PREFILL}); its collectives go "
                     "through the communicator")
    lines += carrier_lines(*plan.carriers())
    lines.append(nccl_free_text(required=plan.require_no_nccl, debug=plan.nccl_debug,
                                nccl_allowed=plan.nccl_allowed))
    estimate = plan.prefill()
    if estimate is not None:
        lines.append(f"  expected prefill: {estimate.describe()}")
    elif plan.large_on_nccl:
        lines.append("  expected prefill: all-reduces above the dispatch ceiling run on NCCL over the group's "
                     "cables, as in the profile's own deployment")
    elif plan.foreign_checkpoint:
        lines.append(f"  expected prefill: not estimated for {plan.checkpoint}; the launcher's model shapes and "
                     "prefill rates describe the profiles' own checkpoints")
    for launch in plan.ranks:
        sudo = f"\"{launch.sudo}\"" if launch.sudo else "none"
        lines.append(f"  rank {launch.rank}: {launch.host} ({launch.ssh}, LAN {launch.lan_address}), container "
                     f"{launch.container}, docker command \"{launch.docker}\", host file operations through "
                     f"{sudo} ({launch.sudo_source})")
        if "NCCL_IB_HCA" in {change.name for change in plan.changes}:
            lines.append(f"    NCCL_IB_HCA={launch.environment['NCCL_IB_HCA']}")
        for mount in launch.mounts:
            source = f", {launch.model_source}" if mount.target == MODEL_TARGET else ""
            lines.append(f"    mount {mount.source} -> {mount.target} ({'ro' if mount.read_only else 'rw'}{source})")
        lines.append(f"    $ {launch.shell()}")
    return "\n".join(lines)


# -- option parsing ------------------------------------------------------------------------------


def split_paths(values: Sequence[str], positions: Sequence[int], what: str) -> tuple[str | None, dict[int, str]]:
    """``PATH`` for every position and ``N=PATH`` for position N: (the shared path, the per-Spark paths)."""
    shared: str | None = None
    per_spark: dict[int, str] = {}
    for value in values:
        key, separator, rest = value.partition("=")
        if separator and key.strip().isdigit():
            position = int(key)
            if position not in positions:
                raise ServePlanError(f"{what} names Spark {position}, which serves no rank of {list(positions)}")
            per_spark[position] = rest
        elif shared is None:
            shared = value
        else:
            raise ServePlanError(f"{what} is given twice without a Spark position")
    return shared, per_spark


def parse_paths(values: Sequence[str], positions: Sequence[int], what: str) -> dict[int, str]:
    """``PATH`` for every position, or ``N=PATH`` for position N; a per-Spark value wins."""
    shared, per_spark = split_paths(values, positions, what)
    if shared is not None:
        for position in positions:
            per_spark.setdefault(position, shared)
    return per_spark


def parse_positions(text: str, ring_size: int | None = None) -> tuple[int, ...]:
    """``0-3``, ``4,5,6,7``, ``7,0,1,2``, or ``7-2`` (a range across the ring's last cable, given its size)."""
    positions: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        first, _, last = item.partition("-")
        try:
            start, end = int(first), int(last or first)
        except ValueError:
            raise ServePlanError(f"positions {text!r} are not a list of Spark positions") from None
        if end >= start:
            positions.extend(range(start, end + 1))
        elif ring_size is not None and 0 <= end < start < ring_size:
            positions.extend([*range(start, ring_size), *range(0, end + 1)])
        else:
            raise ServePlanError(f"range {item!r} runs backwards; across the ring's last cable it needs the "
                                 "ring's size")
    if not positions:
        raise ServePlanError("no positions given")
    return tuple(positions)


def parse_groups(text: str, ring_size: int | None = None) -> tuple[tuple[int, ...], ...]:
    """Groups separated by ``;``, each in :func:`parse_positions` form: ``0-1;2-3;4-5;6-7``."""
    groups = tuple(parse_positions(part, ring_size) for part in text.split(";") if part.strip())
    if not groups:
        raise ServePlanError("no groups given")
    return groups
