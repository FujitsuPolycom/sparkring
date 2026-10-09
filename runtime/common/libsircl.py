"""libsircl, SIRCL's NCCL-compatible C library, as an image layer and as the installer transport ``libsircl``.

docs/architecture/libsircl.md describes the design. This module holds:

- the facts of the libsircl image layer (``runtime/images/libsircl_layer.py``)
  that installation checks, and ``check_layer``, which verifies an image's
  layer against the ``libsircl`` block of its v3 lock
  (``runtime/common/image_lock.py``);
- ``host_library_path``, where a stock-image deployment's Sparks hold the
  host build of libsircl;
- the transport ``libsircl`` (``sparkring install --transport libsircl``):
  vLLM's PyNccl carries the collectives of every vLLM device communicator on
  libsircl, through the layer's vLLM plugin; SIRCL's vLLM adapter and the
  prepared RoCEnante transport are off. ``runtime/common/transport.py``
  passes a deployment lock's ``transport`` section whose ``backend`` is
  ``libsircl`` to ``validate_section``, ``adapt``, ``plan_lines``,
  ``admit_layer`` and ``check_host_document`` here.

The transport is chosen only by name. It needs an image whose v3 lock lists
``libsircl``; a fabric document that lists ``sircl`` (its relay table is
installed where the fabric has relays) and names the Sparks' devices alike; a
group that SIRCL's group placement accepts (two to eight consecutive Sparks
or the whole cycle, no lane through more than three relays); and a profile
without decode-context parallelism. libsircl supports pairs, paths and cycles
of up to eight ranks; on Sparks only a cabled pair has run.

Each rank's routing settings come from libsircl's own tool,
``spark_transport/libsircl/tools/site_routes.py``, which runs SIRCL's route
planner (``sparkring_sircl.routes``) on the group's explicit layout with
``SIRCL_FABRIC_DOCUMENT`` naming a copy of the fabric document, so the route
maps use the device names setup discovered. Rank ``r`` of the group is
libsircl position ``r``. The section records them per rank, and the lock's
identity covers them.

torch's ProcessGroupNCCL keeps the image's NVIDIA NCCL: the image's
toolchain entrypoint puts it first in ``LD_PRELOAD``. Profile settings whose
collectives would reach it through ``torch.distributed`` are refused (SIRCL's
``nccl_free_problems`` and ``relay_conflicts``), as are vLLM's micro-batching
and a profile that sets a variable this transport owns; the containers log
NCCL's initialization, so a communicator NVIDIA NCCL creates is visible.

Status: research-only. Each Spark keeps libsircl's per-communicator receipts
in the deployment's receipt directory; the installer does not judge them.
"""
from dataclasses import replace
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from runtime.common import fabric_document, image_lock, transport
from runtime.common.container_spec import Bind

ROOT = Path(__file__).resolve().parents[2]
BACKEND = "libsircl"
STATUS = "research-only"
LAYER_SCHEMA = "sparkring-libsircl-layer/v1"
# The library's ELF SONAME: the interface name NCCL programs bind.
SONAME = "libnccl.so.2"
PLUGIN_NAME = "libsircl"
PLUGIN_MODULE = "sparkring_libsircl"
PLUGIN_ENTRY_POINT = ("vllm.general_plugins", PLUGIN_NAME, PLUGIN_MODULE + ":register")
# The notices every source or binary copy of libsircl carries, relative to the vendored tree and to INSTALL_ROOT.
NOTICES = ("LICENSE", "NOTICE", "vendor/NCCL-LICENSE.txt", "vendor/SIRCL-NOTICE", "LICENSES/CUDA-NOTICE.txt",
           "LICENSES/rdma-core-verbs.txt")
INSTALL_ROOT = "/opt/sparkring/libsircl"
# The fail-stop mode: with LIBSIRCL_FAIL_STOP=1 the library ends the process on a recorded asynchronous error. vLLM's
# PyNccl checks only each call's enqueue result, and a wait timeout poisons a communicator without failing the
# step whose output it spoiled, so the transport requires the mode and sets it.
FAIL_STOP_VARIABLE = "LIBSIRCL_FAIL_STOP"
# The NCCL API level libsircl reports and that the transport keeps: it passes vLLM's and torch's feature gates
# without opening newer, unsupported ones.
NCCL_API_VERSION = 22705
# Where every Spark of a stock-image deployment holds the host build of libsircl, by its SHA-256.
HOST_LIBRARY_ROOT = "/var/lib/sparkring/libsircl"
# libsircl's route tool and the SIRCL package it imports as ``sparkring_sircl``.
SITE_ROUTES_TOOL = "spark_transport/libsircl/tools/site_routes.py"
SITE_ROUTES = ROOT / SITE_ROUTES_TOOL
SIRCL_ROOT = ROOT / "spark_transport" / "sircl"
# The routing settings site_routes.py prints for a rank.
ROUTE_VARIABLES = ("LIBSIRCL_POSITION", "SIRCL_PEER_ROUTES", "LIBSIRCL_CHAIN_ORDER", "LIBSIRCL_FORWARD_WINDOWS",
                   "SIRCL_FORWARD_CHUNK_BYTES", "LIBSIRCL_RING_WINDOW", "LIBSIRCL_P2P_WINDOWS", "SIRCL_P2P_CHUNK_BYTES")
LIBRARY_VARIABLE = "SPARKRING_LIBSIRCL_LIBRARY"
DIGEST_VARIABLE = "SPARKRING_LIBSIRCL_SHA256"
# Every communicator's receipt, <prefix>.rank<r>.<pid>.c<n>.json, in the deployment's receipt directory.
RECEIPT_PREFIX = transport.RECEIPT_TARGET + "/libsircl"
# Variables this transport sets; a profile that sets one, or any LIBSIRCL_* variable, is refused.
OWNED = (*ROUTE_VARIABLES, LIBRARY_VARIABLE, DIGEST_VARIABLE, "LIBSIRCL_TRANSPORT", "LIBSIRCL_RECEIPT",
         FAIL_STOP_VARIABLE, "LIBSIRCL_NCCL_API_VERSION", "SIRCL_GID_INDEX", "SIRCL_BOOTSTRAP_ADDR")
# vLLM's collective paths that bypass PyNccl, each turned off so that PyNccl carries the device communicators'
# collectives: PyNccl on, torch and NCCL symmetric memory (and with it NCCL's optional ncclMemAlloc allocator,
# whose callers ignore its errors), FlashInfer's all-reduce and PCIe IPC all-reduce, and B12X's PCIe all-reduce.
# vLLM's custom all-reduce, which covers NVLink and PCIe peer copies, is turned off by CUSTOM_ALL_REDUCE_OFF.
# These values replace the profile's.
INDEPENDENT_TRANSPORTS_OFF = {
    "VLLM_DISABLE_PYNCCL": "0", "VLLM_ALLREDUCE_USE_SYMM_MEM": "0", "VLLM_USE_NCCL_SYMM_MEM": "0",
    "VLLM_ALLREDUCE_USE_FLASHINFER": "0", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "0",
    "VLLM_ENABLE_PCIE_ALLREDUCE": "0",
}
CUSTOM_ALL_REDUCE_OFF = "--disable-custom-all-reduce"
# A profile's SIRCL_* switches (serve.plan.profile_settings: the fused norm and the column gathers) configure
# SIRCL's own adapter, which a libsircl container does not run, so its containers do not take them;
# SIRCL_ENABLED, which the prepared images read, stays as the profile sets it.
PROFILE_SIRCL_KEPT = ("SIRCL_ENABLED",)
SECTION_FIELDS = {"schema", "backend", "status", "image", "fabric", "group", "devices", "routes", "planner",
                  "libsircl"}
GROUP_FIELDS = {"layout", "positions", "shape", "size", "name", "max_relays", "lanes", "cabling"}
PLANNER_FIELDS = {"tool", "layout", "lanes", "chain_order", "ring_problems"}
NOT_QUALIFIED = ("Research-only: no serving A/B has measured libsircl; torch.distributed's own collectives stay on "
                 "the image's NCCL")
EXPECTED = "vLLM's PyNccl carries its collectives on libsircl; the installer does not judge libsircl's receipts"
OFF_TEXT = ("vLLM's custom all-reduce, torch and NCCL symmetric memory, FlashInfer all-reduce and B12X PCIe "
            "all-reduce, so PyNccl carries the device collectives")
TORCH_RELAY_NOTE = ("torch.distributed's NVIDIA NCCL cannot connect this group's ranks that share no cable; a "
                    "collective vLLM sends through torch instead of PyNccl would wait at NCCL's connection setup")


class LibsirclError(transport.TransportError):
    """libsircl cannot run here, or one of its inputs is malformed."""


def _require(condition, text):
    if not condition:
        raise LibsirclError(text)


def _plan():
    """SIRCL's serve-launcher plan module (torch-free): the refusal rules and environments this transport reuses."""
    from spark_transport.sircl.sparkring_sircl.vllm.serve import plan
    return plan


def library_path(version):
    return f"{image_lock.LIBSIRCL_LIBRARY_DIRECTORY}/libsircl.so.{version}"


def pack_architectures(makefile):
    """The GPU architectures of libsircl's kernel packs: ``code=sm_<n>`` of its Makefile's ``NVCCFLAGS``."""
    import re
    flags = re.search(r"^NVCCFLAGS\s*=(.*)$", makefile, re.M)
    _require(flags is not None, "libsircl's Makefile names no NVCCFLAGS")
    found = sorted(set(re.findall(r"code=(sm_[0-9]+)", flags.group(1))))
    _require(bool(found), "libsircl's Makefile names no kernel pack architecture")
    return found


def has_fail_stop(library):
    """Whether the library's bytes name ``LIBSIRCL_FAIL_STOP``, the variable its fail-stop mode reads."""
    return FAIL_STOP_VARIABLE.encode() + b"\0" in library


def host_library_path(sha256, version):
    """The host path of a libsircl build with SHA-256 ``sha256`` (``libsircl_layer.py host-library``)."""
    _require(isinstance(sha256, str) and len(sha256) == 64, "a libsircl build is named by its SHA-256")
    return f"{HOST_LIBRARY_ROOT}/{sha256}/libsircl.so.{version}"


def check_layer(image_id, parent_receipt_sha256, block, *, run):
    """Verify image ``image_id``'s libsircl layer against the lock's ``libsircl`` block.

    The image's own ``verify`` checks every file its external-base receipt
    records (``installer_image.admit``). This reads that receipt and the
    layer receipt in network-less containers and requires the receipt to
    have the SHA-256 ``parent_receipt_sha256``, to record the layer receipt,
    the library and the plugin with the SHA-256 values ``block`` names and
    every file the layer receipt lists, so that verification covered them.
    """
    image_lock.validate_libsircl(block)
    isolated = ["docker", "run", "--rm", "--pull", "never", "--runtime", "runc", "--network", "none",
                "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--entrypoint", "/bin/cat",
                image_id]
    from runtime.common import installer_image
    raw = run([*isolated, installer_image.PARENT_RECEIPT], text=False).stdout
    _require(hashlib.sha256(raw).hexdigest() == parent_receipt_sha256,
             "The image's external-base receipt differs from its lock")
    files = json.loads(raw)["files"]
    layer_raw = run([*isolated, image_lock.LIBSIRCL_RECEIPT], text=False).stdout
    _require(hashlib.sha256(layer_raw).hexdigest() == block["receipt"]["sha256"],
             "The image's libsircl layer receipt differs from its lock")
    _require(files.get(image_lock.LIBSIRCL_RECEIPT) == block["receipt"]["sha256"],
             "The image's verification does not cover its libsircl layer receipt")
    for label, entry in (("library", block["library"]), ("vLLM plugin", block["plugin"])):
        _require(files.get(entry["path"]) == entry["sha256"],
                 f"The image's verification does not cover its libsircl {label} {entry['path']}")
    layer = json.loads(layer_raw)
    _require(layer.get("schema") == LAYER_SCHEMA and layer.get("version") == block["version"]
             and layer.get("snapshot") == block["snapshot"]
             and layer.get("nccl_api_version") == block["nccl_api_version"]
             and layer.get("fail_stop") == block["fail_stop"]
             and (layer.get("library") or {}).get("path") == block["library"]["path"]
             and (layer.get("library") or {}).get("sha256") == block["library"]["sha256"]
             and layer.get("plugin") == block["plugin"],
             "The image's libsircl layer receipt describes another libsircl build")
    installed = layer.get("files")
    _require(isinstance(installed, dict) and installed and all(files.get(path) == digest
                                                               for path, digest in installed.items()),
             "The image's verification does not cover every installed libsircl file")
    return {"schema": "sparkring-libsircl-admission/v1", "image_id": image_id, "version": block["version"],
            "snapshot": block["snapshot"], "receipt_sha256": block["receipt"]["sha256"],
            "files_verified": len(installed)}


# Where the transport runs.

def unavailable(image_value, document):
    """Why libsircl cannot run for this image and fabric, or None."""
    if BACKEND not in image_lock.transports(image_value):
        return f"image {image_value.get('name')} carries no libsircl layer"
    block = image_lock.libsircl(image_value)
    if not block["fail_stop"]:
        return fail_stop_missing(image_library(block["snapshot"]))
    return fabric_unavailable(document)


def image_library(snapshot):
    """How a refusal names the libsircl of an image layer built from vendored snapshot ``snapshot``."""
    return f"the image's libsircl (snapshot {snapshot[:8]})"


def fail_stop_missing(library):
    """Why a libsircl build without the fail-stop mode is refused; ``library`` names that build."""
    return (f"{library} has no fail-stop mode ({FAIL_STOP_VARIABLE}): vLLM's PyNccl checks only that each call was "
            "queued, and a wait timeout that poisons a communicator does not fail the step whose output it spoiled; "
            f"use a libsircl built from a snapshot whose library reads {FAIL_STOP_VARIABLE}")


def fabric_unavailable(document):
    """Why the recorded fabric cannot carry libsircl, or None."""
    if document is None:
        return "this cluster has no fabric document; sudo sparkring setup records one"
    if "sircl" not in document["transports"]:
        return "the fabric's relay table is not installed; sudo sparkring setup installs it"
    if not fabric_document.uniform_names(document):
        return "the Sparks name their fabric devices differently, which SIRCL's route maps do not support"
    return None


def choose(image_value, document, *, nccl=None):
    """``(backend, nccl, reason)`` for an explicit ``--transport libsircl``; LibsirclError where it cannot run."""
    _require(nccl is None, "--nccl applies to deployments on SIRCL; with --transport libsircl, vLLM's PyNccl "
                           "carries every collective on libsircl")
    reason = unavailable(image_value, document)
    _require(reason is None, f"--transport libsircl cannot run here: {reason}")
    return BACKEND, None, None


# The routing settings.

def site_routes(document, layout, lanes, *, run=subprocess.run):
    """libsircl's ``tools/site_routes.py --json`` for ``layout``, with the fabric document's device names.

    The tool runs in its own interpreter with SIRCL's package on
    ``PYTHONPATH`` and ``SIRCL_FABRIC_DOCUMENT`` naming a copy of
    ``document``, so SIRCL's route planner names the devices setup
    discovered; ``-B`` keeps it from writing bytecode into the checkout.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fabric.json"
        path.write_text(fabric_document.encoded(document), encoding="utf-8")
        environment = {"PYTHONPATH": str(SIRCL_ROOT), "SIRCL_FABRIC_DOCUMENT": str(path),
                       "PATH": os.environ.get("PATH", "")}
        if "SYSTEMROOT" in os.environ:
            environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        done = run([sys.executable, "-B", "-s", str(SITE_ROUTES), "--layout", layout, "--lanes", str(lanes), "--json"],
                   capture_output=True, text=True, env=environment, cwd=directory, timeout=120)
    if done.returncode:
        detail = (done.stderr or done.stdout or "").strip().splitlines()
        raise LibsirclError(f"libsircl's {SITE_ROUTES_TOOL} failed for {layout}: "
                            + (detail[-1] if detail else f"exit status {done.returncode}"))
    return json.loads(done.stdout)


def peer_routes(text):
    """``{peer: [device, ...]}`` of a ``SIRCL_PEER_ROUTES`` value."""
    result = {}
    for entry in filter(None, text.split(",")):
        peer, _, devices = entry.partition("=")
        _require(peer.isdigit() and devices and int(peer) not in result, f"SIRCL_PEER_ROUTES entry {entry!r}")
        result[int(peer)] = devices.split("/")
    return result


def group(document, positions, *, dcp=1, run=subprocess.run):
    """``(group, devices, routes, planner)`` of a libsircl group on ``positions`` of ``document``.

    The installer's route code places the group (``transport.group_topology``,
    SIRCL's ``describe_group``), which refuses positions that are not
    consecutive Sparks or the whole cycle and lanes through more than three
    relays; libsircl's ``tools/site_routes.py`` gives every rank's routing
    settings for the group's explicit layout.
    """
    _require(dcp == 1, f"the profile sets decode-context parallelism {dcp}; no libsircl evidence covers vLLM's "
                       "decode-context-parallel groups, so libsircl runs profiles without it")
    _require(2 <= len(positions) <= 8, "a libsircl group spans 2 to 8 Sparks")
    layout = transport.sircl_layout(document)
    topology = transport.group_topology(layout, positions)
    shape, size = topology.fabric.kind, len(topology.members)
    explicit = topology.session_layout()
    planned = site_routes(document, explicit, topology.lane_count, run=run)
    _require(len(planned["ranks"]) == size, f"{SITE_ROUTES_TOOL} planned {len(planned['ranks'])} ranks, not {size}")
    value = {"layout": layout, "positions": [int(position) for position in positions], "shape": shape, "size": size,
             "name": transport.group_name(shape, size), "max_relays": topology.max_relays(),
             "lanes": topology.lane_count, "cabling": topology.nccl_policy.value}
    devices = [transport.rank_devices(document, topology, rank) for rank in range(size)]
    routes = [dict(row["env"]) for row in planned["ranks"]]
    planner = {"tool": SITE_ROUTES_TOOL, "layout": explicit, "lanes": topology.lane_count,
               "chain_order": planned["chain_order"], "ring_problems": list(planned["ring_problems"])}
    return value, devices, routes, planner


def section(image_value, document, positions, *, dcp=1, run=subprocess.run):
    """The deployment lock's ``transport`` section of a libsircl deployment on ``positions`` of ``document``."""
    reason = unavailable(image_value, document)
    _require(reason is None, f"libsircl cannot run: {reason}")
    value, devices, routes, planner = group(document, positions, dcp=dcp, run=run)
    return {"schema": transport.SECTION_SCHEMA, "backend": BACKEND, "status": STATUS, "image": image_value["name"],
            "fabric": {"id": document["id"], "shape": document["shape"], "size": document["size"]},
            "group": value, "devices": devices, "routes": routes, "planner": planner,
            "libsircl": dict(image_lock.libsircl(image_value))}


def check_routes(value, devices, routes, planner, nodes):
    """Refuse routing settings that are not ``site_routes.py``'s shape for a group of ``nodes`` ranks."""
    _require(isinstance(devices, list) and len(devices) == nodes
             and all(isinstance(row, list) and row and all(isinstance(name, str) for name in row) for row in devices),
             "The libsircl group lists each rank's RDMA devices")
    _require(isinstance(routes, list) and len(routes) == nodes, "The libsircl group lists each rank's routing settings")
    for rank, row in enumerate(routes):
        _require(isinstance(row, dict) and set(row) <= set(ROUTE_VARIABLES)
                 and all(isinstance(setting, str) for setting in row.values())
                 and row.get("LIBSIRCL_POSITION") == str(rank) and "SIRCL_PEER_ROUTES" in row,
                 f"Rank {rank}'s libsircl routing settings are those of {SITE_ROUTES_TOOL} at position {rank}")
        peers = peer_routes(row["SIRCL_PEER_ROUTES"])
        _require(sorted(peers) == [peer for peer in range(nodes) if peer != rank]
                 and all(len(lanes) == value["lanes"] and set(lanes) <= set(devices[rank]) for lanes in peers.values()),
                 f"Rank {rank}'s route map names every other rank over its own RDMA devices")
    topology = transport.group_topology(value["layout"], value["positions"])
    _require(isinstance(planner, dict) and set(planner) == PLANNER_FIELDS and planner["tool"] == SITE_ROUTES_TOOL
             and planner["layout"] == topology.session_layout() and planner["lanes"] == value["lanes"],
             "The libsircl group names the route tool and the group's explicit layout it planned")


def validate_section(value, card, image_runtime):
    """A libsircl ``transport`` section of a deployment lock after checking it against the lock's selection."""
    _require(isinstance(value, dict) and set(value) == SECTION_FIELDS and value["schema"] == transport.SECTION_SCHEMA
             and value["backend"] == BACKEND and value["status"] == STATUS,
             f"A libsircl transport section has {', '.join(sorted(SECTION_FIELDS))} and status {STATUS}")
    _require(image_runtime is not None and value["image"] == image_runtime["name"] == card["release"],
             "The transport section names the deployment's installer image")
    fabric = value["fabric"]
    _require(isinstance(fabric, dict) and set(fabric) == {"id", "shape", "size"} and isinstance(fabric["id"], str)
             and fabric["id"].startswith("sha256:") and len(fabric["id"]) == 71,
             "The transport section names the fabric document's identity, shape and size")
    group_ = value["group"]
    _require(isinstance(group_, dict) and set(group_) == GROUP_FIELDS,
             "The transport section's group has " + ", ".join(sorted(GROUP_FIELDS)))
    _require(group_["size"] == len(group_["positions"]) == card["nodes"],
             f"The transport group has {card['nodes']} positions, one per rank")
    topology = transport.group_topology(group_["layout"], group_["positions"])
    _require(group_["layout"] == transport.sircl_layout(fabric) and topology.fabric.kind == group_["shape"]
             and topology.max_relays() == group_["max_relays"] and topology.lane_count == group_["lanes"]
             and topology.nccl_policy.value == group_["cabling"]
             and group_["name"] == transport.group_name(group_["shape"], group_["size"]),
             "The transport group differs from its layout and positions")
    check_routes(group_, value["devices"], value["routes"], value["planner"], card["nodes"])
    image_lock.validate_libsircl(value["libsircl"])
    return value


# The adapter.

def owned_settings(environment_):
    return sorted(key for key in environment_ if key in OWNED or key.startswith("LIBSIRCL_"))


def argument(arguments, *flags):
    """The last value of any of ``flags`` in ``arguments`` (``--flag VALUE`` or ``--flag=VALUE``), or None."""
    found = None
    for index, item in enumerate(arguments):
        for flag in flags:
            if item == flag and index + 1 < len(arguments):
                found = arguments[index + 1]
            elif item.startswith(flag + "="):
                found = item[len(flag) + 1:]
    return found


def _count(arguments, *flags):
    text = argument(arguments, *flags)
    try:
        return int(text) if text is not None else 1
    except ValueError:
        raise LibsirclError(f"{flags[0]} {text!r} is not a number") from None


def capability_problems(arguments):
    """vLLM features that need what libsircl does not carry; empty when the command needs none of them.

    libsircl exports ``ncclCommSuspend`` and ``ncclCommResume`` only as
    refusals; it carries point-to-point only between the two ranks of a
    two-rank communicator; expert parallelism stays on the all-to-all backend
    whose variable all-gathers and reduce-scatters are grouped broadcasts and
    reductions; and the sequence-parallelism pass stays off with the other
    communication fusions.
    """
    problems = []
    if "--enable-sleep-mode" in arguments:
        problems.append("--enable-sleep-mode suspends PyNccl's communicators with ncclCommSuspend and ncclCommResume, "
                        "which libsircl exports only as refusals")
    stages = _count(arguments, "--pipeline-parallel-size", "-pp")
    if stages > 2:
        problems.append(f"pipeline parallelism {stages} sends point-to-point on {stages}-rank communicators; libsircl "
                        "carries point-to-point only between the two ranks of a two-rank communicator")
    if _count(arguments, "--data-parallel-size", "-dp") > 1:
        problems.append("data parallelism, which no libsircl evidence covers")
    backend = argument(arguments, "--all2all-backend")
    if ("--enable-expert-parallel" in arguments or "-ep" in arguments) and backend != "allgather_reducescatter":
        problems.append(f"expert parallelism with the {backend or 'default'} all-to-all backend; libsircl carries "
                        "--all2all-backend allgather_reducescatter, whose variable all-gathers and reduce-scatters are "
                        "grouped broadcasts and reductions")
    try:
        compilation = json.loads(argument(arguments, "--compilation-config", "-cc") or "{}")
    except ValueError:
        compilation = {}
    passes = compilation.get("pass_config") if isinstance(compilation, dict) else None
    if isinstance(passes, dict) and passes.get("enable_sp"):
        problems.append("the recipe's --compilation-config pass_config.enable_sp turns on the sequence-parallelism "
                        "pass, which the libsircl transport keeps off with the other communication fusions")
    return problems


def refusals(profile_environment, arguments):
    """Why a profile cannot run with vLLM's PyNccl on libsircl: settings whose collectives bypass PyNccl, reach
    torch's own NCCL, break the one issuing order libsircl's communicators need, or need what libsircl does not
    carry (``capability_problems``); empty when it can."""
    plan = _plan()
    problems = plan.nccl_free_problems(profile_environment, nccl_mode="never", required=True, arguments=arguments)
    problems += plan.relay_conflicts(arguments, profile_environment)
    batching = plan.micro_batching(arguments)
    if batching:
        problems.append(batching + "; libsircl's communicators need one issuing order too")
    if plan.recipe_dcp(arguments) != 1:
        problems.append("the command sets decode-context parallelism, which no libsircl evidence covers")
    return problems + capability_problems(list(arguments))


def library_environment(block):
    """The libsircl settings of every container: the checked library, fail-stop, its NCCL API level, the verbs
    transport and the receipt prefix."""
    _require(block["fail_stop"], fail_stop_missing(image_library(block["snapshot"])))
    return {LIBRARY_VARIABLE: block["library"]["path"], DIGEST_VARIABLE: block["library"]["sha256"],
            FAIL_STOP_VARIABLE: "1", "LIBSIRCL_NCCL_API_VERSION": str(NCCL_API_VERSION),
            "LIBSIRCL_TRANSPORT": "verbs", "LIBSIRCL_RECEIPT": RECEIPT_PREFIX}


def environment(value, profile_environment, arguments):
    """The settings every rank's container gets besides its routing settings, after the transport's refusals.

    ``profile_environment`` is rank 0's adapted environment and ``arguments``
    its command, which holds the recipe's vLLM arguments.
    """
    plan = _plan()
    owned = owned_settings(profile_environment)
    _require(not owned, f"the profile sets {owned}, which the libsircl transport owns")
    problems = refusals(profile_environment, arguments)
    _require(not problems, "the libsircl transport refuses this profile: " + "; ".join(problems))
    try:
        gid = int(profile_environment.get("NCCL_IB_GID_INDEX", "3"))
    except ValueError:
        raise LibsirclError("the profile's NCCL_IB_GID_INDEX is not a number") from None
    return {
        **library_environment(value["libsircl"]), "SIRCL_GID_INDEX": str(gid),
        **plan.DISABLED_TRANSPORTS, **INDEPENDENT_TRANSPORTS_OFF,
        # NCCL logs every communicator it creates, so one that torch's NVIDIA NCCL makes shows in the model log.
        **plan.nccl_debug_environment(True, {}),
    }


def plugins(value):
    """``VLLM_PLUGINS`` with the libsircl plugin last and without SIRCL's."""
    names = [name for name in value.split(",") if name and name not in ("sircl", PLUGIN_NAME)]
    return ",".join([*names, PLUGIN_NAME])


def adapt(specs, lock):
    """Each rank's container with vLLM's PyNccl on libsircl (the lock's ``transport`` section)."""
    value = lock["transport"]
    rows = lock["site"]["ranks"]
    _require(len(specs) == len(rows) == len(value["routes"]), "one container per rank of the libsircl group")
    common = environment(value, specs[0].environment, specs[0].command)
    result = []
    for rank, spec in enumerate(specs):
        owned = owned_settings(spec.environment)
        _require(not owned, f"rank {rank}: the profile sets {owned}, which the libsircl transport owns")
        _require(spec.environment.get("VLLM_HOST_IP"), f"rank {rank}'s container names no VLLM_HOST_IP")
        profile = {key: item for key, item in spec.environment.items()
                   if not key.startswith("SIRCL_") or key in PROFILE_SIRCL_KEPT}
        # The address a rank publishes in the unique id of a communicator it roots: its own bootstrap address.
        settings = {**profile, **common, **value["routes"][rank],
                    "SIRCL_BOOTSTRAP_ADDR": spec.environment["VLLM_HOST_IP"]}
        settings["VLLM_PLUGINS"] = plugins(settings.get("VLLM_PLUGINS", ""))
        mounts = (*spec.mounts, Bind(transport.receipt_directory(lock), transport.RECEIPT_TARGET, False))
        command = spec.command if CUSTOM_ALL_REDUCE_OFF in spec.command else (*spec.command, CUSTOM_ALL_REDUCE_OFF)
        result.append(replace(spec, environment=settings, mounts=mounts, command=command))
    return result


# Text and identities.

def relays_text(value):
    relays = value["max_relays"]
    return f"at most {relays} relay{'s' if relays != 1 else ''} on a lane" if relays else "no relays"


def plan_lines(value, notes=()):
    """What ``sparkring install`` prints about a libsircl deployment's transport before it asks."""
    block, group_ = value["libsircl"], value["group"]
    lines = [f"Transport: libsircl ({STATUS}): vLLM's PyNccl carries its collectives on libsircl {block['version']}; "
             "SIRCL's adapter and RoCEnante are off",
             f"  libsircl group: {group_['name']} at positions {', '.join(map(str, group_['positions']))}; "
             f"{group_['lanes']} lanes per peer, {relays_text(group_)}",
             f"  Library: {block['library']['path']}, SHA-256 {block['library']['sha256'][:12]}, "
             f"snapshot {block['snapshot'][:8]}; fail-stop on ({FAIL_STOP_VARIABLE}=1)",
             f"  Off: {OFF_TEXT}",
             f"  {NOT_QUALIFIED}"]
    if group_["cabling"] == "none":
        notes = [TORCH_RELAY_NOTE, *notes]
    return lines + [f"  Note: {note}" for note in notes]


def request_identity(value):
    """The part of an installation request that a libsircl section adds to the deployment's identity."""
    return {"backend": BACKEND, "fabric": value["fabric"]["id"], "library": value["libsircl"]["library"]["sha256"]}


def summary(value):
    """The install result's ``transport`` field of a libsircl deployment."""
    group_ = value["group"]
    return {"backend": BACKEND, "status": STATUS, "group": group_["name"], "positions": group_["positions"],
            "fabric": value["fabric"]["id"], "library": dict(value["libsircl"]["library"]), "expected": EXPECTED}


def unjudged_verdict(lock, *, now=None):
    """The receipt verdict of a libsircl deployment: ``unknown``, because the installer does not judge its receipts."""
    from runtime.host import transport_receipts
    moment = (datetime.datetime.fromtimestamp(now, datetime.timezone.utc) if now is not None
              else datetime.datetime.now(datetime.timezone.utc))
    directory = transport.receipt_directory(lock)
    return {"schema": transport_receipts.VERDICT_SCHEMA, "backend": BACKEND, "status": STATUS, "expected": EXPECTED,
            "verdict": "unknown", "nccl_observed": None,
            "problems": [f"the installer does not judge libsircl's receipts; each Spark keeps them in {directory}"],
            "receipts": directory, "checked_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ")}


def verdict_text(verdict):
    """One line for ``sparkring check`` about a libsircl deployment's verdict."""
    return f"Transport: libsircl ({STATUS}); receipts not judged, kept in {verdict.get('receipts')} on each Spark"


def card_text(value):
    """The summary card's Transport value of a libsircl deployment."""
    return f"libsircl ({STATUS}), receipts not judged"


def status_view(value):
    """A libsircl deployment's transport for ``sparkring status``."""
    group_ = value["group"]
    return {"backend": BACKEND, "status": STATUS, "group": group_["name"], "positions": group_["positions"],
            "fabric": value["fabric"]["id"], "library": dict(value["libsircl"]["library"])}


def status_line(view):
    """The ``Transport:`` line of ``sparkring status`` for a libsircl deployment."""
    return f"Transport: libsircl on {view['group']} ({STATUS}); vLLM's PyNccl on libsircl, receipts not judged"


# Admission and host checks.

def admit_layer(lock, *, run):
    """Verify the image's libsircl layer against the deployment's transport section (``check_layer``)."""
    return check_layer(lock["selection"]["image_id"], lock["image_runtime"]["parent_receipt_sha256"],
                       lock["transport"]["libsircl"], run=run)


def check_host_document(value, *, root="/"):
    """Raise LibsirclError unless this Spark's fabric document, from which the routes were planned, has the
    deployment's identity."""
    path = Path(root) / fabric_document.HOST_PATH.lstrip("/")
    try:
        document = fabric_document.load(path)
    except fabric_document.FabricDocumentError as error:
        raise LibsirclError(f"libsircl's routes come from the fabric document, and this Spark's copy cannot be used: "
                            f"{error}") from None
    _require(document["id"] == value["fabric"]["id"],
             f"This deployment was made on fabric {value['fabric']['id'][7:19]}; this Spark records fabric "
             f"{document['id'][7:19]}. Run sudo sparkring install again")
    _require(fabric_document.uniform_names(document), "the Sparks name their fabric devices differently")
    return {"ok": True, "fabric": document["id"]}
