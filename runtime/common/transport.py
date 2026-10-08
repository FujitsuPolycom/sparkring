"""A deployment's collective transport, and the adapter that runs its containers on SIRCL ring sessions.

Every ``sparkring install`` deployment runs on one of two transports:

- ``sircl``: SIRCL ring sessions (``spark_transport/sircl``) carry every
  collective of the tensor-parallel group. The image must carry the SIRCL
  layer (image lock v3, ``runtime/common/image_lock.py``) and the cluster
  must have a fabric document (``runtime/common/fabric_document.py``) that
  lists ``sircl`` among its transports, which a fabric with relays lists once
  its relay table is installed, and whose Sparks name their fabric devices
  alike.
- ``prepared``: the prepared RoCEnante transport of the installer images,
  with NCCL. A lock without a ``transport`` section uses it, so every
  deployment made before this module behaves as it did.

``choose`` makes ``sircl`` the default wherever it can run. NCCL is off on a
SIRCL deployment (``nccl: never``) unless the operator opts in with
``--nccl auto``, which lets NCCL carry the collectives the group's cabling
allows: every collective on a pair, NCCL's ring algorithm on a whole cycle,
none on a group whose members are not all neighbors around it.

The deployment lock records the decision as its ``transport`` section
(``section``, schema ``sparkring-transport/v1``): the fabric document's
identity, the group (SIRCL's layout text, the positions and the shape the
tuning table is keyed by), each rank's RDMA devices, the tuning row in
effect, and the image's SIRCL layer. The lock's identity covers it.

The default tuning table, ``runtime/common/sircl-tuning-defaults.json``
(schema ``sparkring-sircl-tuning/v1``), chooses only among SIRCL's own
settings: per group shape (``pair``, ``path-<n>``, ``cycle-<n>``, else the
row of the shape alone), the session settings that ``serve.plan.Options``
of the SIRCL launcher names (one-shot limit, launch grid, schedules,
minimums, link sizes, capacities and waits), with each row's evidence
(``measured``, ``rules`` for SIRCL's own derivation). ``tables`` may name
measured ``sircl-tuning-table/v1`` files in the repository; each session
takes the one whose key matches its group and the image's SIRCL build.

``adapt`` renders each rank's container from the installer image's adapted
container (``installer_image.adapt``): the environment the SIRCL launcher's
``serve.plan.build_plan`` sets, built from the same helpers and checked by
the same rules (``test_transport.py`` compares the two variable for
variable), with these differences, which the image layer makes possible:

- SIRCL is installed in the image's site-packages, so no source tree is
  mounted and ``PYTHONPATH`` keeps the image's own;
- the two native libraries are the image's prebuilt copies
  (``SIRCL_NATIVE_LIBRARY``, ``SIRCL_P2P_NATIVE_LIBRARY``, and
  ``SIRCL_BUILD_CACHE_DIR`` at their directory), so nothing compiles on a
  Spark;
- ``SIRCL_FABRIC_DOCUMENT`` names the Spark's own copy of the fabric document,
  mounted read-only, so route maps use the device names setup discovered;
- ``SIRCL_RECEIPT_DIR`` is a directory of the deployment's workspace on each
  Spark (``sircl/receipts``), which ``scripts/installer_host.py`` reads;
- the torch.distributed and API addresses stay the installer's.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from runtime.common import fabric_document, fabric_layout, image_lock
from runtime.common.container_spec import Bind

SECTION_SCHEMA = "sparkring-transport/v1"
TUNING_SCHEMA = "sparkring-sircl-tuning/v1"
ROOT = Path(__file__).resolve().parents[2]
TUNING_DEFAULTS = Path(__file__).with_name("sircl-tuning-defaults.json")
# A table that `sudo sparkring fabric tune` measures on this cluster, bound to its fabric and image.
MEASURED_TUNING = "sircl-tuning.json"
BACKENDS = ("sircl", "prepared")
NCCL_MODES = ("never", "auto")
# The SIRCL launcher's other name for auto.
NCCL_ALIASES = {"topology": "auto"}
DEFAULT_NCCL = "never"
# Container paths of the receipts and of measured tuning tables; the fabric
# document keeps its host path.
RECEIPT_TARGET = "/run/sparkring/sircl/receipts"
TABLE_TARGET = "/run/sparkring/sircl/tuning"
# The receipt directory in each Spark's deployment workspace.
RECEIPT_DIRECTORY = "sircl/receipts"
# Tuning row settings: the SIRCL launcher's option names (serve.plan.Options).
INTEGER_SETTINGS = ("capacity", "dispatch", "gather", "oneshot_max", "large_blocks", "chain_min", "ring_min",
                    "link_slots", "ring_gather_stagger", "link_slot", "link_chunk", "gather_link_chunk",
                    "scatter_link_chunk", "reduce_link_chunk", "spin_limit")
SCHEDULE_SETTINGS = ("large_schedule", "gather_schedule", "scatter_schedule")
SECONDS_SETTINGS = ("startup_wait", "serving_wait")
SETTINGS = (*INTEGER_SETTINGS, *SCHEDULE_SETTINGS, *SECONDS_SETTINGS, "large_allreduce")
ROW_SOURCES = re.compile(r"measured|rules|inherited:[a-z0-9-]+")
GROUP_NAME = re.compile(r"pair|(?:path|cycle)(?:-[2-9]|-1[0-6])?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FABRIC_ID = re.compile(r"sha256:[0-9a-f]{64}")


class TransportError(ValueError):
    """The requested transport cannot run, or a transport input is malformed."""


def _require(condition, text):
    if not condition:
        raise TransportError(text)


def _sircl():
    """SIRCL's launcher plan, guard and fabric modules (torch-free)."""
    from spark_transport.sircl.sparkring_sircl.vllm import fabric, guard
    from spark_transport.sircl.sparkring_sircl.vllm.serve import plan
    return plan, guard, fabric


def nccl_mode(value):
    """``--nccl`` as recorded: ``never`` or ``auto`` (``topology`` reads as ``auto``); None stays None."""
    if value is None:
        return None
    value = NCCL_ALIASES.get(value, value)
    _require(value in NCCL_MODES, f"--nccl takes never or auto, not {value!r}")
    return value


# The tuning table.

def encoded(document):
    """Sorted keys, two-space indent and LF: the bytes of a tuning table file."""
    return json.dumps(document, sort_keys=True, indent=2) + "\n"


def tuning_digest(document):
    return hashlib.sha256(encoded(document).encode()).hexdigest()


def _setting(name, value):
    if name in INTEGER_SETTINGS:
        return type(value) is int and value >= 0
    if name in SCHEDULE_SETTINGS:
        return value in ("auto", "chain", "ring", "pieces")
    if name in SECONDS_SETTINGS:
        return type(value) in (int, float) and value > 0
    return value in ("auto", "sircl", "nccl")


def validate_tuning(document, *, root=ROOT):
    """A ``sparkring-sircl-tuning/v1`` document after checking it; TransportError otherwise."""
    _require(isinstance(document, dict) and document.get("schema") == TUNING_SCHEMA, f"expected {TUNING_SCHEMA}")
    _require(set(document) == {"schema", "source", "sircl", "fabric", "image_id", "measured_at", "layouts", "tables"},
             f"a {TUNING_SCHEMA} table has schema, source, sircl, fabric, image_id, measured_at, layouts and tables")
    _require(document["source"] in ("defaults", "measured"), "a tuning table's source is defaults or measured")
    sircl = document["sircl"]
    _require(isinstance(sircl, dict) and set(sircl) == {"version", "abi_version"}
             and isinstance(sircl["version"], str) and type(sircl["abi_version"]) is int,
             "a tuning table names the SIRCL version and ABI it applies to")
    if document["source"] == "measured":
        _require(isinstance(document["fabric"], str) and _FABRIC_ID.fullmatch(document["fabric"])
                 and isinstance(document["image_id"], str),
                 "a measured tuning table names the fabric and the image it was measured on")
    else:
        _require(document["fabric"] is None and document["image_id"] is None,
                 "the default tuning table is bound to no fabric and no image")
    layouts = document["layouts"]
    _require(isinstance(layouts, dict) and layouts, "a tuning table has layout rows")
    for name, row in layouts.items():
        _require(GROUP_NAME.fullmatch(name), f"tuning row {name!r}: rows are pair, path, cycle, path-<n> or cycle-<n>")
        _require(isinstance(row, dict) and set(row) == {"source", "settings"} and isinstance(row["source"], str)
                 and ROW_SOURCES.fullmatch(row["source"]) and isinstance(row["settings"], dict),
                 f"tuning row {name}: source (measured, rules or inherited:<row>) and settings")
        for key, value in row["settings"].items():
            _require(key in SETTINGS and _setting(key, value), f"tuning row {name}: {key}={value!r} is not a "
                     "SIRCL session setting the installer passes")
    tables = document["tables"]
    _require(isinstance(tables, list), "a tuning table lists its measured tables (may be empty)")
    for entry in tables:
        _require(isinstance(entry, dict) and set(entry) == {"path", "sha256"} and isinstance(entry["path"], str)
                 and isinstance(entry["sha256"], str) and _SHA256.fullmatch(entry["sha256"]),
                 "a measured table entry is {path, sha256}")
        relative = PurePosixPath(entry["path"])
        _require(not relative.is_absolute() and ".." not in relative.parts, "a measured table is a repository path")
        path = Path(root) / relative
        _require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"],
                 f"measured table {entry['path']} is missing or differs from its SHA-256")
    return document


def load_tuning(path=TUNING_DEFAULTS, *, root=ROOT):
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TransportError(f"{path} cannot be read: {error}") from None
    return validate_tuning(document, root=root)


def tuning_in_effect(state, document, image_value, *, root=ROOT):
    """``(tuning document, notes)``: the measured table of ``state`` when it is bound to this fabric and
    image, else the default table. A measured table bound elsewhere is named in ``notes``."""
    notes = []
    path = Path(state) / MEASURED_TUNING if state is not None else None
    if path is not None and path.exists():
        try:
            measured = load_tuning(path, root=root)
            if measured["fabric"] == document["id"] and measured["image_id"] == image_value["image_id"]:
                return measured, notes
            notes.append(f"the measured tuning table {path} was measured on another fabric or image; the default "
                         "table applies")
        except TransportError as error:
            notes.append(f"the measured tuning table {path} cannot be used ({error}); the default table applies")
    return load_tuning(root=root), notes


def group_name(shape, size):
    return "pair" if shape == "pair" else f"{shape}-{size}"


def tuning_row(document, shape, size):
    """``(row name, row)`` for a group: ``<shape>-<size>`` (``pair``), else ``<shape>``."""
    for name in (group_name(shape, size), shape):
        if name in document["layouts"]:
            return name, document["layouts"][name]
    raise TransportError(f"the tuning table has no row for a {group_name(shape, size)} group")


# The group and the section of the deployment lock.

def sircl_layout(document):
    """SIRCL's ``SIRCL_FABRIC`` text for a fabric document: ``pair:1``, ``path:<n>`` or ``ring:<n>``."""
    shape, size = document["shape"], document["size"]
    if shape == fabric_layout.PAIR:
        return "pair:1"
    return f"path:{size}" if shape == fabric_layout.PATH else f"ring:{size}"


def group_topology(layout_text, positions):
    _, _, fabric = _sircl()
    try:
        return fabric.describe_group(fabric.Layout.parse(layout_text), list(positions))
    except fabric.FabricError as error:
        raise TransportError(str(error)) from None


def rank_devices(document, topology, rank):
    """The RDMA devices of ``rank``'s lanes to every peer, as the fabric document names them at its position."""
    plan, _, fabric = _sircl()
    roles = {device: role for role, device in fabric_layout.DEVICES.items()}
    named = {row["role"]: rdma for rdma, row in fabric_document.devices(document, topology.members[rank]).items()}
    canonical = plan.rank_devices(topology, rank)
    _require(set(canonical) <= set(roles), f"rank {rank}'s lanes use devices outside the fabric's four functions")
    return [named[roles[device]] for device in canonical]


def unavailable(image_value, document):
    """Why SIRCL cannot run for this image and fabric, or None."""
    name = image_value.get("name")
    if "sircl" not in image_lock.transports(image_value):
        return f"image {name} carries no SIRCL layer"
    if document is None:
        return "this cluster has no fabric document; sudo sparkring setup records one"
    if "sircl" not in document["transports"]:
        return "the fabric's relay table is not installed; sudo sparkring setup installs it"
    if not fabric_document.uniform_names(document):
        return "the Sparks name their fabric devices differently, which SIRCL's route maps do not support"
    return None


def choose(image_value, document, *, backend=None, nccl=None):
    """``(backend, nccl, reason)``: ``sircl`` with NCCL ``never`` wherever SIRCL can run, else ``prepared``.

    ``backend`` and ``nccl`` are the operator's ``--transport`` and ``--nccl``
    or None; ``reason`` says why SIRCL cannot run (None when it can). An
    explicit ``--transport sircl`` that cannot run, and ``--nccl`` with the
    prepared transport, are refused.
    """
    _require(backend in (None, *BACKENDS), f"--transport takes sircl or prepared, not {backend!r}")
    nccl = nccl_mode(nccl)
    reason = unavailable(image_value, document)
    if backend is None:
        backend = "prepared" if reason else "sircl"
    if backend == "sircl" and reason:
        raise TransportError(f"--transport sircl cannot run here: {reason}")
    if backend == "prepared":
        _require(nccl is None, "--nccl applies to deployments on SIRCL; the prepared transport keeps its own NCCL "
                               "settings")
        return backend, None, reason
    return backend, nccl or DEFAULT_NCCL, None


def section(image_value, document, positions, *, nccl, tuning, root=ROOT):
    """The deployment lock's ``transport`` section of a SIRCL deployment on ``positions`` of ``document``."""
    _require(unavailable(image_value, document) is None, "SIRCL cannot run: " + str(unavailable(image_value, document)))
    nccl = nccl_mode(nccl) or DEFAULT_NCCL
    layout = sircl_layout(document)
    topology = group_topology(layout, positions)
    shape, size = topology.fabric.kind, len(topology.members)
    name, row = tuning_row(tuning, shape, size)
    sircl = image_lock.sircl(image_value)
    applies = tuning["sircl"] == {"version": sircl["version"], "abi_version": sircl["abi_version"]}
    tables = []
    if applies and tuning["tables"]:
        from spark_transport.sircl.sparkring_sircl import tuning as sircl_tuning
        # The group's own facts with the image's SIRCL build in place of this checkout's.
        facts = {**sircl_tuning.facts_for_layout(topology.session_layout(), topology.lane_count),
                 **sircl["tuning_key"]}
        for entry in tuning["tables"]:
            table = sircl_tuning.Table(json.loads((Path(root) / entry["path"]).read_text(encoding="utf-8")), entry["path"])
            if not table.mismatches(facts):
                tables.append({**entry, "hash": table.hash})
        _require(len(tables) <= 1, "several measured tuning tables match this group and image")
    return {
        "schema": SECTION_SCHEMA, "backend": "sircl", "nccl": nccl, "image": image_value["name"],
        "fabric": {"id": document["id"], "shape": document["shape"], "size": document["size"]},
        "group": {"layout": layout, "positions": [int(position) for position in positions], "shape": shape,
                  "size": size, "name": group_name(shape, size), "max_relays": topology.max_relays(),
                  "lanes": topology.lane_count, "cabling": topology.nccl_policy.value},
        "devices": [rank_devices(document, topology, rank) for rank in range(size)],
        "tuning": {"source": tuning["source"], "sha256": tuning_digest(tuning), "row": name,
                   "row_source": row["source"] if applies else "rules",
                   "settings": dict(row["settings"]) if applies else {}, "tables": tables,
                   "applies": applies},
        "sircl": sircl,
    }


def validate_section(value, card, image_runtime):
    """The ``transport`` section of a deployment lock after checking it against the lock's selection."""
    _require(isinstance(value, dict) and value.get("schema") == SECTION_SCHEMA and value.get("backend") == "sircl",
             f"A deployment's transport section is a {SECTION_SCHEMA} SIRCL section")
    _require(set(value) == {"schema", "backend", "nccl", "image", "fabric", "group", "devices", "tuning", "sircl"},
             "The transport section has backend, nccl, image, fabric, group, devices, tuning and sircl")
    _require(value["nccl"] in NCCL_MODES, "The transport section's nccl is never or auto")
    _require(image_runtime is not None and value["image"] == image_runtime["name"] == card["release"],
             "The transport section names the deployment's installer image")
    fabric = value["fabric"]
    _require(isinstance(fabric, dict) and set(fabric) == {"id", "shape", "size"}
             and isinstance(fabric["id"], str) and _FABRIC_ID.fullmatch(fabric["id"]),
             "The transport section names the fabric document's identity, shape and size")
    try:
        fabric_layout.layout(fabric["shape"], fabric["size"])
    except ValueError as error:
        raise TransportError(str(error)) from None
    group = value["group"]
    _require(isinstance(group, dict) and set(group) == {"layout", "positions", "shape", "size", "name", "max_relays",
                                                       "lanes", "cabling"},
             "The transport section's group has layout, positions, shape, size, name, max_relays, lanes and cabling")
    _require(group["size"] == len(group["positions"]) == card["nodes"],
             f"The transport group has {card['nodes']} positions, one per rank")
    topology = group_topology(group["layout"], group["positions"])
    _require(group["layout"] == sircl_layout(fabric) and topology.fabric.kind == group["shape"]
             and topology.max_relays() == group["max_relays"] and topology.lane_count == group["lanes"]
             and topology.nccl_policy.value == group["cabling"] and group["name"] == group_name(group["shape"],
                                                                                              group["size"]),
             "The transport group differs from its layout and positions")
    devices = value["devices"]
    _require(isinstance(devices, list) and len(devices) == card["nodes"]
             and all(isinstance(row, list) and row and all(isinstance(name, str) for name in row) for row in devices),
             "The transport section lists each rank's RDMA devices")
    tuning = value["tuning"]
    _require(isinstance(tuning, dict) and set(tuning) == {"source", "sha256", "row", "row_source", "settings", "tables",
                                                         "applies"}
             and isinstance(tuning["sha256"], str) and _SHA256.fullmatch(tuning["sha256"]),
             "The transport section records the tuning table's digest, row and settings")
    for key, setting in tuning["settings"].items():
        _require(key in SETTINGS and _setting(key, setting), f"Tuning setting {key}={setting!r} is not passed")
    image_lock.validate_sircl(value["sircl"])
    return value


# The adapter.

def receipt_directory(lock):
    """The SIRCL receipt directory of the deployment on each Spark."""
    return str(PurePosixPath(lock["site"]["workspace"]) / RECEIPT_DIRECTORY)


def policy(value, environment):
    """``(NCCL policy, reason)`` the SIRCL adapter applies to the tensor-parallel group."""
    _, guard, _ = _sircl()
    topology = group_topology(value["group"]["layout"], value["group"]["positions"])
    return guard.effective_policy(topology.nccl_policy, topology.nccl_reason, nccl_mode=value["nccl"],
                                  environ=environment)


def _options(settings):
    """A namespace with the schedules a tuning row sets (``serve.plan.schedule_environment``)."""
    class Options:
        pass
    options = Options()
    for name in SCHEDULE_SETTINGS:
        setattr(options, name, settings.get(name))
    return options


def environment(value, profile_environment, arguments):
    """The SIRCL variables every rank's container gets, and the checks the SIRCL launcher applies.

    ``profile_environment`` is rank 0's adapted environment and ``arguments``
    its command, which holds the recipe's vLLM arguments.
    """
    plan, _, _ = _sircl()
    settings = value["tuning"]["settings"]
    topology = group_topology(value["group"]["layout"], value["group"]["positions"])
    capacity = settings.get("capacity", plan.DEFAULT_CAPACITY)
    dispatch = settings.get("dispatch", capacity)
    gather = settings.get("gather", plan.DEFAULT_GATHER)
    startup = float(settings.get("startup_wait", plan.DEFAULT_STARTUP_WAIT_S))
    serving = float(settings.get("serving_wait", plan.DEFAULT_SERVING_WAIT_S))
    large = settings.get("large_allreduce", "auto")
    for name, number in (("capacity", capacity), ("dispatch ceiling", dispatch), ("gather capacity", gather)):
        _require(number >= 16 and number % 16 == 0, f"the SIRCL {name} must be a positive multiple of 16 bytes")
    _require(dispatch <= capacity, "the dispatch ceiling cannot exceed the all-reduce capacity")
    for name, seconds in (("startup", startup), ("serving", serving)):
        _require(1e-6 <= seconds <= plan.MAX_WAIT_S, f"the {name} wait must be between 1e-6 and {plan.MAX_WAIT_S:.0f} s")
    spin = settings.get("spin_limit")
    _require(spin is None or 1 <= spin < 1 << 32, "the spin limit must be a positive 32-bit poll count")
    owned = [key for key in plan.OWNED if key in profile_environment]
    _require(not owned, f"the profile sets {owned}, which SIRCL's adapter owns")
    effective, reason = policy(value, profile_environment)
    _require(large != "nccl" or effective.allows("all_reduce"),
             f"SIRCL_LARGE_ALLREDUCE=nccl needs NCCL, which may not all-reduce on this group ({reason})")
    required = value["nccl"] == "never"
    problems = plan.nccl_free_problems(profile_environment, nccl_mode=value["nccl"], required=required,
                                       arguments=arguments)
    _require(not problems, "NCCL communicators would be created outside SIRCL's groups: " + "; ".join(problems))
    batching = plan.micro_batching(arguments)
    _require(batching is None, f"SIRCL's communicator refuses this profile: {batching}")
    if effective.value == "none":
        conflicts = plan.relay_conflicts(arguments, profile_environment)
        _require(not conflicts, f"SIRCL's communicator refuses this profile on {topology.fabric.describe()}, where "
                                f"NCCL may not run ({reason}): " + "; ".join(conflicts))
    links = {name: settings[name] for name in ("link_slot", "link_chunk", "gather_link_chunk", "scatter_link_chunk",
                                               "reduce_link_chunk") if name in settings}
    schedules = {name: settings[name] for name in SCHEDULE_SETTINGS if name in settings}
    try:
        plan.session_problems([plan.session_settings(topology, name="tp", groups="", scoped=True, schedules=schedules,
                                                     link_sizes=links, link_slots=settings.get("link_slots"),
                                                     ring_gather_stagger=settings.get("ring_gather_stagger"))])
        gid = int(profile_environment.get("NCCL_IB_GID_INDEX", "3"))
        common = {
            "SIRCL_MODE": "custom",
            "SIRCL_FABRIC": value["group"]["layout"],
            "SIRCL_RANK_POSITIONS": ",".join(str(position) for position in value["group"]["positions"]),
            "SIRCL_GROUPS": "tp",
            "SIRCL_NCCL": value["nccl"],
            "SIRCL_LARGE_ALLREDUCE": large,
            "SIRCL_SESSION_MODULE": plan.SESSION_MODULE,
            "SIRCL_RECEIPT_DIR": RECEIPT_TARGET,
            "SIRCL_BUILD_CACHE_DIR": image_lock.LIBRARY_DIRECTORY,
            "SIRCL_NATIVE_LIBRARY": value["sircl"]["native"]["path"],
            "SIRCL_P2P_NATIVE_LIBRARY": value["sircl"]["p2p"]["path"],
            "SIRCL_FABRIC_DOCUMENT": fabric_document.HOST_PATH,
            "SIRCL_ALLREDUCE_CAPACITY_BYTES": str(capacity),
            "SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES": str(dispatch),
            "SIRCL_ALLGATHER_MAX_BYTES": str(gather),
            "SIRCL_STARTUP_WAIT_S": f"{startup:g}",
            "SIRCL_SERVING_WAIT_S": f"{serving:g}",
            "SIRCL_GID_INDEX": str(gid),
            **plan.DISABLED_TRANSPORTS,
        }
        if spin is not None:
            common["SIRCL_SPIN_LIMIT"] = str(spin)
        common.update(plan.oneshot_environment(settings.get("oneshot_max"), dispatch))
        common.update(plan.large_blocks_environment(settings.get("large_blocks")))
        common.update(plan.schedule_environment(_options(settings)))
        common.update(plan.chain_min_environment(settings.get("chain_min")))
        common.update(plan.ring_min_environment(settings.get("ring_min")))
        common.update(plan.link_environment(links))
        common.update(plan.link_slots_environment(settings.get("link_slots")))
        common.update(plan.ring_gather_stagger_environment(settings.get("ring_gather_stagger")))
        # Without NCCL every NCCL communicator a container creates is a fault, and NCCL names each one
        # only with INIT logging; the receipt check scans for them (runtime/host/transport_receipts.py).
        common.update(plan.nccl_debug_environment(required, {}))
    except plan.ServePlanError as error:
        raise TransportError(str(error)) from None
    tables = value["tuning"]["tables"]
    if tables:
        common[plan.TUNING_VARIABLE] = ",".join(f"{TABLE_TARGET}/{entry['hash']}.json" for entry in tables)
    return common, effective


def adapt(specs, lock):
    """Each rank's container with SIRCL in front of every collective (the lock's ``transport`` section)."""
    value = lock["transport"]
    plan, _, _ = _sircl()
    rows = lock["site"]["ranks"]
    _require(len(specs) == len(rows) == len(value["devices"]), "one container per rank of the transport group")
    common, effective = environment(value, specs[0].environment, specs[0].command)
    result = []
    for rank, (spec, row) in enumerate(zip(specs, rows, strict=True)):
        environment_ = dict(spec.environment)
        owned = [key for key in plan.OWNED if key in environment_]
        _require(not owned, f"rank {rank}: the profile sets {owned}, which SIRCL's adapter owns")
        environment_.update(common)
        plugins = [item for item in environment_.get("VLLM_PLUGINS", "").split(",") if item]
        environment_["VLLM_PLUGINS"] = ",".join(plugins + ([] if "sircl" in plugins else ["sircl"]))
        if effective.allows("all_reduce"):
            environment_["NCCL_IB_HCA"] = plan.nccl_hca_value(spec.environment.get("NCCL_IB_HCA"), value["devices"][rank])
        mounts = [*spec.mounts,
                  Bind(fabric_document.HOST_PATH, fabric_document.HOST_PATH, True),
                  Bind(receipt_directory(lock), RECEIPT_TARGET, False)]
        mounts += [Bind(str(PurePosixPath(row["repository"]) / entry["path"]), f"{TABLE_TARGET}/{entry['hash']}.json",
                        True) for entry in value["tuning"]["tables"]]
        result.append(replace(spec, environment=environment_, mounts=tuple(mounts)))
    shards = {spec.environment.get(plan.MHC_SHARD, "0").strip() not in ("", "0") for spec in result}
    _require(len(shards) == 1, f"the profile's ranks disagree on {plan.MHC_SHARD}")
    _require(len({spec.environment.get(plan.HC_PREFILL) for spec in result}) == 1,
             f"the profile's ranks disagree on {plan.HC_PREFILL}")
    return result


# Text.

def plan_lines(value, notes=()):
    """What ``sparkring install`` prints about a SIRCL deployment's transport before it asks."""
    tuning = value["tuning"]
    row = tuning["row"]
    if not tuning["applies"]:
        evidence = f"the default table is for another SIRCL build; SIRCL's own rules apply on this {row} group"
    elif tuning["row_source"] == "measured":
        evidence = f"{'measured on this fabric' if tuning['source'] == 'measured' else 'default table'}, {row}"
    elif tuning["row_source"] == "rules":
        evidence = f"default table, {value['group']['name']}: not measured, SIRCL's own rules apply"
    else:
        evidence = f"default table, {value['group']['name']}: {tuning['row_source'].replace(':', ' from ')}"
    if value["nccl"] == "never":
        lines = [f"Transport: sircl on every collective, NCCL off ({evidence})"]
    else:
        allowed = {"all": "every collective", "ring": "NCCL's ring algorithm", "none": "nothing"}[value["group"]["cabling"]]
        lines = [f"Transport: sircl; NCCL may carry what the cabling allows, {allowed} on this group (--nccl auto; "
                 f"{evidence})"]
    group = value["group"]
    relays = group["max_relays"]
    lines.append(f"  SIRCL group: {group['name']} at positions {', '.join(map(str, group['positions']))}; "
                 f"{group['lanes']} lanes per peer, "
                 + (f"at most {relays} relay{'s' if relays != 1 else ''} on a lane" if relays else "no relays"))
    if tuning["settings"]:
        lines.append("  SIRCL settings: " + ", ".join(f"{key} {setting}" for key, setting in
                                                       sorted(tuning["settings"].items())))
    for note in notes:
        lines.append("  Note: " + note)
    return lines


def prepared_line(reason, explicit):
    if explicit:
        return "Transport: prepared (--transport prepared)"
    return f"Transport: prepared, because {reason}" if reason else "Transport: prepared"


# Image admission and host checks.

def admit_layer(lock, *, run):
    """Verify the image's SIRCL layer against the deployment's transport section.

    ``installer_image.admit`` has verified the external-base receipt against
    the lock and run the image's own ``verify``, which checks every file that
    receipt records. This reads the same receipt and the layer receipt in a
    network-less container and requires that the receipt records the layer
    receipt, the wheel's installed files and both libraries with the SHA-256
    the lock records, so the verification covered them.
    """
    from runtime.common import installer_image
    value = lock["transport"]
    sircl = value["sircl"]
    image = lock["selection"]["image_id"]
    isolated = ["docker", "run", "--rm", "--pull", "never", "--runtime", "runc", "--network", "none",
                "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--entrypoint", "/bin/cat",
                image]
    raw = run([*isolated, installer_image.PARENT_RECEIPT], text=False).stdout
    _require(hashlib.sha256(raw).hexdigest() == lock["image_runtime"]["parent_receipt_sha256"],
             "The image's external-base receipt differs from its lock")
    files = json.loads(raw)["files"]
    layer_raw = run([*isolated, image_lock.LAYER_RECEIPT], text=False).stdout
    _require(hashlib.sha256(layer_raw).hexdigest() == sircl["receipt"]["sha256"],
             "The image's SIRCL layer receipt differs from its lock")
    layer = json.loads(layer_raw)
    _require(files.get(image_lock.LAYER_RECEIPT) == sircl["receipt"]["sha256"],
             "The image's verification does not cover its SIRCL layer receipt")
    for kind in image_lock.LIBRARY_STEMS:
        library = sircl[kind]
        _require(files.get(library["path"]) == library["sha256"],
                 f"The image's verification does not cover its SIRCL {kind} library {library['path']}")
    _require(layer.get("schema") == "sparkring-sircl-layer/v1" and layer.get("version") == sircl["version"]
             and layer.get("abi_version") == sircl["abi_version"] and layer.get("wheel") == sircl["wheel"]
             and layer.get("native") == sircl["native"] and layer.get("p2p") == sircl["p2p"]
             and layer.get("tuning_key") == sircl["tuning_key"],
             "The image's SIRCL layer receipt describes another SIRCL build")
    installed = layer.get("files")
    _require(isinstance(installed, dict) and installed and all(files.get(path) == digest
                                                               for path, digest in installed.items()),
             "The image's verification does not cover every installed SIRCL file")
    return {"schema": "sparkring-sircl-admission/v1", "image_id": image, "version": sircl["version"],
            "abi_version": sircl["abi_version"], "receipt_sha256": sircl["receipt"]["sha256"],
            "files_verified": len(installed)}


def check_host_document(value, *, root="/"):
    """Raise TransportError unless this Spark's fabric document has the deployment's identity."""
    path = Path(root) / fabric_document.HOST_PATH.lstrip("/")
    try:
        document = fabric_document.load(path)
    except fabric_document.FabricDocumentError as error:
        raise TransportError(f"SIRCL reads the fabric document, and this Spark's copy cannot be used: {error}") from None
    _require(document["id"] == value["fabric"]["id"],
             f"This deployment was made on fabric {value['fabric']['id'][7:19]}; this Spark records fabric "
             f"{document['id'][7:19]}. Run sudo sparkring install again")
    _require(fabric_document.uniform_names(document), "the Sparks name their fabric devices differently")
    return {"ok": True, "fabric": document["id"]}
