"""A deployment's collective transport, and the adapter that runs its containers on SIRCL ring sessions.

Every ``sparkring install`` deployment runs on one of these transports:

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
- ``libsircl``: vLLM's PyNccl on libsircl, SIRCL's NCCL-compatible C
  library, chosen only by name (research-only). ``runtime/common/libsircl.py``
  owns it; the functions of this module that take a deployment's section
  pass a section whose ``backend`` is ``libsircl`` to it.
- ``nccl``: vLLM's PyNccl alone, on an image with or without the SIRCL
  layer. No SIRCL variable or plugin reaches the container and the RoCEnante
  slot stays off, so PyNccl carries every tensor-parallel and
  expert-parallel collective. It runs only where NCCL's own cabling rule
  holds (``nccl_cabling_rule``): a cabled pair, or a whole cycle, whose
  consecutive ranks share cables around the group. A group whose ranks reach
  each other through relays is refused, because NCCL picks its RDMA devices
  and addresses without the relay table. On a whole cycle the deployment
  adds the NCCL settings of NCCL's ring algorithm (``ring_settings``, the
  SIRCL serve launcher's ``RING_SETTINGS``), so NCCL builds no tree
  connections between Sparks that share no cable, and each rank's
  ``NCCL_IB_HCA`` names the RDMA devices of its own lanes, which the fabric
  document records per position.

``choose`` makes ``sircl`` the default wherever it can run. NCCL is off on a
SIRCL deployment (``nccl: never``) unless the operator opts in with
``--nccl auto``, which lets NCCL carry the collectives the group's cabling
allows: every collective on a pair, NCCL's ring algorithm on a whole cycle,
none on a group whose members are not all neighbors around it. ``topology``
is another name for ``auto``. The modes, their other names and the rule the
plan text states (``nccl_rule``: NCCL is opt-in only, and tables choose among
SIRCL options) are SIRCL's adapter settings (``sparkring_sircl.vllm.settings``).

The deployment lock records the decision as its ``transport`` section
(``section``, schema ``sparkring-transport/v1``): the fabric document's
identity, the group (SIRCL's layout text, the positions and the shape the
tuning table is keyed by), each rank's RDMA devices, the tuning row in
effect, and the image's SIRCL layer. The lock's identity covers it.

A tuning table (schema ``sparkring-sircl-tuning/v1``) chooses only among
SIRCL's own settings: per group shape (``pair``, ``path-<n>``,
``cycle-<n>``, else the row of the shape alone), the session settings that
``serve.plan.Options`` of the SIRCL launcher names (one-shot limit, launch
grid, schedules, minimums, link sizes, capacities and waits), with each
row's evidence (``measured``, ``rules`` for SIRCL's own derivation,
``inherited:<row>``). ``tables`` names measured SIRCL tuning tables
(``sircl-tuning-table/v1``: per collective, size and mode, the fastest SIRCL
algorithm, schedule, piece and launch grid, and the session settings those
choices ran under and need: link slots, link slot, chain slot and
large-message piece); each session takes the one whose key matches its group
and the image's SIRCL build, applies the table's settings that its
environment leaves unset, and its hash joins the session's setup agreement.
The tensor-parallel session and, with decode-context parallelism, the
sessions of the decode-context-parallel groups each take their own table, as
the SIRCL launcher matches them (``serve.plan.tuning_plan``). The section
records each matched table, the sessions that take it and its settings; a
tuning row, whose settings reach the tensor-parallel session only, that sets a
link slot count or link slot below the table's is refused, as the SIRCL
launcher refuses it, because the table's choices could not run. A table's
marks of where NCCL measured faster route no call. Two tables exist:

- the default table, ``runtime/common/sircl-tuning-defaults.json``
  (``source: defaults``), bound to no fabric and no image; its ``tables``
  are repository paths;
- a measured table, ``/var/lib/sparkring/controller/sircl-tuning.json`` on
  Node A (``source: measured``), which ``sudo sparkring fabric tune``
  (``runtime/host/fabric_tune.py``) writes from ring-harness measurements on
  the cluster's own fabric. It is bound to the fabric document's identity,
  the image and its SIRCL build, and each Spark's GPU driver and kernel
  (``binding``); its measured rows are ``measured`` and hold the default
  table's settings for their group shape except the link settings their
  SIRCL table records (``MEASURED_ROW_RULE``, ``measured_row``), and the
  default rows it carries for shapes it did not measure are
  ``default:<their source>``. Its
  ``tables`` are absolute paths under ``HOST_TABLES``, where every Spark holds
  the same bytes. ``tuning_in_effect`` uses it only while every binding
  holds (``measured_problems``) and otherwise falls back to the default table
  and says why.

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
# The measured table in Node A's controller directory: `sudo sparkring fabric tune` writes it from
# measurements on this cluster's fabric, bound to that fabric, the image and the Sparks' drivers.
MEASURED_TUNING = "sircl-tuning.json"
# Every Spark's copies of the measured SIRCL tables, named by their SHA-256; written once, never changed.
HOST_TABLES = "/etc/sparkring/fabric/sircl-tuning"
DOCUMENT_FIELDS = frozenset({"schema", "source", "sircl", "fabric", "image_id", "measured_at", "layouts", "tables"})
BINDING_FIELDS = frozenset({"image", "tuning_key", "drivers", "defaults_sha256", "harness"})
# What a measured table records of each Spark's software: the NVIDIA GPU driver and the kernel release,
# which carries the ConnectX (mlx5) driver. None where a Spark did not report it.
DRIVER_FIELDS = ("gpu", "kernel")
DRIVER_LABELS = {"gpu": "GPU driver", "kernel": "kernel"}
BACKENDS = ("sircl", "prepared", "nccl", "libsircl")
LIBSIRCL = "libsircl"
# The transports whose containers write receipts to the deployment's receipt directory and whose ranks check the
# fabric's relay table at their own fabric positions: SIRCL ring sessions and libsircl. The prepared and nccl
# transports do neither.
SESSION_BACKENDS = ("sircl", LIBSIRCL)
# SIRCL's NCCL modes and their other names (sparkring_sircl.vllm.settings NCCL_MODES and NCCL_MODE_ALIASES),
# kept here for the commands' --nccl choices; test_transport.py compares the two.
NCCL_MODES = ("never", "auto")
NCCL_ALIASES = {"topology": "auto"}
DEFAULT_NCCL = "never"
# The sessions that take a SIRCL tuning table, as the SIRCL launcher names them (serve.plan.SESSION_TITLES).
SESSION_TITLES = {"tp": "tensor-parallel session", "dcp": "decode-context-parallel sessions"}
# The tuning-row settings that set a variable of tuning.SETTINGS, which the session then takes instead of the
# table's: the row's link slot count and link slot.
ROW_TABLE_SETTINGS = {"link_slots": "SIRCL_LINK_SLOTS", "link_slot": "SIRCL_LINK_SLOT_BYTES"}
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
# A row's evidence: measured on a fabric; design, settings the SIRCL install design sets that no measurement
# confirmed; rules, SIRCL's own derivation; inherited:<row>, another row's settings.
ROW_SOURCES = re.compile(r"measured|design|rules|inherited:[a-z0-9-]+")
# A measured table's rows for group shapes it did not measure: the default table's row and its evidence.
CARRIED = "default:"
GROUP_NAME = re.compile(r"pair|(?:path|cycle)(?:-[2-9]|-1[0-6])?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHORT = re.compile(r"[0-9a-f]{16}")
_FABRIC_ID = re.compile(r"sha256:[0-9a-f]{64}")
_HOST_TABLE = re.compile(re.escape(HOST_TABLES) + r"/([0-9a-f]{64})\.json")


class TransportError(ValueError):
    """The requested transport cannot run, or a transport input is malformed."""


def _require(condition, text):
    if not condition:
        raise TransportError(text)


def session_backend(lock):
    """Whether the deployment of ``lock`` runs on a transport of ``SESSION_BACKENDS``."""
    return (lock.get("transport") or {}).get("backend") in SESSION_BACKENDS


def _sircl():
    """SIRCL's launcher plan, guard and fabric modules (torch-free)."""
    from spark_transport.sircl.sparkring_sircl.vllm import fabric, guard
    from spark_transport.sircl.sparkring_sircl.vllm.serve import plan
    return plan, guard, fabric


def _adapter_settings():
    from spark_transport.sircl.sparkring_sircl.vllm import settings
    return settings


def nccl_mode(value):
    """``--nccl`` as recorded: the mode SIRCL's adapter names, ``never`` or ``auto`` (``topology`` is ``auto``);
    None stays None."""
    if value is None:
        return None
    settings = _adapter_settings()
    value = settings.nccl_mode_name(value)
    _require(value in settings.NCCL_MODES, f"--nccl takes never or auto, not {value!r}")
    return value


def nccl_rule():
    """The NCCL rule SIRCL's plans, bundles and receipts state: ``NCCL: opt-in only (auto); tables choose among
    SIRCL options``."""
    return _adapter_settings().NCCL_RULE


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


def _validate_binding(document):
    """The ``binding`` of a measured table: what its measurements depend on besides the fabric and the image."""
    binding = document["binding"]
    _require(isinstance(binding, dict) and set(binding) == BINDING_FIELDS,
             "a measured tuning table's binding names the image, its SIRCL tuning key, each Spark's drivers, the "
             "default table it carries rows of and the harness runs")
    _require(isinstance(binding["image"], str) and binding["image"], "a measured tuning table names its image")
    key = binding["tuning_key"]
    sircl = document["sircl"]
    _require(isinstance(key, dict) and set(key) == {"native", "kernels", "sircl"}
             and all(isinstance(key[name], str) and _SHORT.fullmatch(key[name]) for name in ("native", "kernels"))
             and key["sircl"] == f"{sircl['version']}/abi{sircl['abi_version']}",
             "a measured tuning table records the SIRCL tuning key of its image (native and kernel source hashes "
             "and <version>/abi<n>)")
    drivers = binding["drivers"]
    _require(isinstance(drivers, dict) and drivers
             and all(isinstance(position, str) and position.isdigit() and isinstance(row, dict)
                     and set(row) == set(DRIVER_FIELDS)
                     and all(value is None or isinstance(value, str) for value in row.values())
                     for position, row in drivers.items()),
             "a measured tuning table records each Spark's GPU driver and kernel by position")
    _require(isinstance(binding["defaults_sha256"], str) and _SHA256.fullmatch(binding["defaults_sha256"]),
             "a measured tuning table records the SHA-256 of the default table it carries rows of")
    _require(isinstance(binding["harness"], dict), "a measured tuning table records its harness runs")


def validate_tuning(document, *, root=ROOT, host_root="/"):
    """A ``sparkring-sircl-tuning/v1`` document after checking it; TransportError otherwise.

    ``root`` holds the repository tables a ``tables`` entry names by a
    relative path; ``host_root`` holds this Spark's ``HOST_TABLES``, which a
    measured table names by absolute path.
    """
    _require(isinstance(document, dict) and document.get("schema") == TUNING_SCHEMA, f"expected {TUNING_SCHEMA}")
    measured = document.get("source") == "measured"
    _require(set(document) == (DOCUMENT_FIELDS | {"binding"} if measured else DOCUMENT_FIELDS),
             f"a {TUNING_SCHEMA} table has schema, source, sircl, fabric, image_id, measured_at, layouts and tables, "
             "and a measured one also its binding")
    _require(document["source"] in ("defaults", "measured"), "a tuning table's source is defaults or measured")
    sircl = document["sircl"]
    _require(isinstance(sircl, dict) and set(sircl) == {"version", "abi_version"}
             and isinstance(sircl["version"], str) and type(sircl["abi_version"]) is int,
             "a tuning table names the SIRCL version and ABI it applies to")
    if measured:
        _require(isinstance(document["fabric"], str) and _FABRIC_ID.fullmatch(document["fabric"])
                 and isinstance(document["image_id"], str),
                 "a measured tuning table names the fabric and the image it was measured on")
        _validate_binding(document)
    else:
        _require(document["fabric"] is None and document["image_id"] is None,
                 "the default tuning table is bound to no fabric and no image")
    layouts = document["layouts"]
    _require(isinstance(layouts, dict) and layouts, "a tuning table has layout rows")
    for name, row in layouts.items():
        _require(GROUP_NAME.fullmatch(name), f"tuning row {name!r}: rows are pair, path, cycle, path-<n> or cycle-<n>")
        source = row.get("source") if isinstance(row, dict) else None
        own = source.removeprefix(CARRIED) if isinstance(source, str) and measured else source
        _require(isinstance(row, dict) and set(row) == {"source", "settings"} and isinstance(own, str)
                 and ROW_SOURCES.fullmatch(own) and isinstance(row["settings"], dict),
                 f"tuning row {name}: source (measured, design, rules or inherited:<row>; in a measured table also "
                 "default:<source>) and settings")
        for key, value in row["settings"].items():
            _require(key in SETTINGS and _setting(key, value), f"tuning row {name}: {key}={value!r} is not a "
                     "SIRCL session setting the installer passes")
    tables = document["tables"]
    _require(isinstance(tables, list), "a tuning table lists its measured tables (may be empty)")
    for entry in tables:
        _require(isinstance(entry, dict) and set(entry) == {"path", "sha256"} and isinstance(entry["path"], str)
                 and isinstance(entry["sha256"], str) and _SHA256.fullmatch(entry["sha256"]),
                 "a measured table entry is {path, sha256}")
        named = PurePosixPath(entry["path"])
        if named.is_absolute():
            found = _HOST_TABLE.fullmatch(entry["path"])
            _require(measured and found is not None and found.group(1) == entry["sha256"],
                     f"a measured table is a repository path, or in a measured tuning table "
                     f"{HOST_TABLES}/<sha256>.json")
            path = Path(host_root) / entry["path"].lstrip("/")
        else:
            _require(".." not in named.parts, "a measured table is a repository path")
            path = Path(root) / named
        _require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"],
                 f"measured table {entry['path']} is missing or differs from its SHA-256")
    return document


def load_tuning(path=TUNING_DEFAULTS, *, root=ROOT, host_root="/"):
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TransportError(f"{path} cannot be read: {error}") from None
    return validate_tuning(document, root=root, host_root=host_root)


def measured_problems(measured, document, image_value, *, drivers=None):
    """Why the measured table ``measured`` does not apply here; empty when it does.

    ``document`` is the recorded fabric document (or None) and
    ``image_value`` the lock of the image the installation uses; its SIRCL
    build is compared when the lock carries one. ``drivers`` maps positions to
    their observed ``{"gpu", "kernel"}``; a field either side does not know is
    not compared.
    """
    problems = []
    binding = measured["binding"]
    if document is None:
        problems.append("this cluster has no fabric document")
    elif measured["fabric"] != document["id"]:
        problems.append(f"it was measured on fabric {measured['fabric'][7:19]}, and the recorded fabric is "
                        f"{document['id'][7:19]}")
    if measured["image_id"] != image_value.get("image_id"):
        problems.append(f"it was measured with image {binding['image']}, and this installation uses "
                        f"{image_value.get('name') or image_value.get('image_id')}")
    sircl = image_lock.sircl(image_value)
    if sircl is not None:
        if measured["sircl"] != {"version": sircl["version"], "abi_version": sircl["abi_version"]}:
            problems.append(f"it was measured with SIRCL {measured['sircl']['version']} (ABI "
                            f"{measured['sircl']['abi_version']}), and the image carries SIRCL {sircl['version']} "
                            f"(ABI {sircl['abi_version']})")
        elif binding["tuning_key"] != sircl["tuning_key"]:
            problems.append("the image's SIRCL native or kernel sources differ from the measured build")
    for position, observed in sorted((drivers or {}).items(), key=lambda item: int(item[0])):
        recorded = binding["drivers"].get(str(position)) or {}
        for field in DRIVER_FIELDS:
            before, now = recorded.get(field), (observed or {}).get(field)
            if before and now and before != now:
                problems.append(f"the {DRIVER_LABELS[field]} of position {position} changed from {before} to {now}")
    return problems


def tuning_in_effect(state, document, image_value, *, root=ROOT, host_root="/", drivers=None):
    """``(tuning document, notes)``: the measured table of ``state`` while it applies here, else the default.

    The measured table applies while ``measured_problems`` finds nothing
    against the recorded fabric document, the installation's image and
    ``drivers``; ``notes`` says why it does not, or why it cannot be read.
    """
    notes = []
    path = Path(state) / MEASURED_TUNING if state is not None else None
    if path is not None and path.exists():
        try:
            measured = load_tuning(path, root=root, host_root=host_root)
        except TransportError as error:
            notes.append(f"the measured tuning table {path} cannot be used ({error}); the default table applies")
        else:
            problems = measured_problems(measured, document, image_value, drivers=drivers)
            if not problems:
                return measured, notes
            notes.append(f"the measured tuning table no longer applies: {'; '.join(problems)}. The default table "
                         "applies; sudo sparkring fabric tune measures this fabric again")
    return load_tuning(root=root), notes


# How a measured row's settings are made (measured_row); the plan text and the install reference state it.
MEASURED_ROW_RULE = ("the default table's settings for this group shape, except the link slots and link slot "
                     "that the measured SIRCL table records, which replace them")
_ROW_GROUP = re.compile(r"(path|cycle)-([2-9]|1[0-6])")


def row_group(name):
    """``(shape, size)`` of the groups tuning row ``name`` serves: ``pair``, ``path-<n>`` or ``cycle-<n>``."""
    if name == "pair":
        return "pair", 2
    found = _ROW_GROUP.fullmatch(name)
    _require(found is not None, f"tuning row {name!r} names no group size: pair, path-<n> or cycle-<n>")
    return found.group(1), int(found.group(2))


def row_identity(name):
    """``(shape, world)`` of the SIRCL table key that serves the groups of tuning row ``name``."""
    shape, size = row_group(name)
    return ("pair" if shape == "pair" else f"{shape}:{size}"), size


def measured_row(defaults, name, settings, table=None):
    """The settings of measured row ``name`` (``MEASURED_ROW_RULE``).

    The default table's row for the row's groups (``tuning_row``: the row of
    that size, else of the shape) holds settings the measurement does not set,
    such as the capacity, dispatch ceiling and one-shot limit; those stay. The
    row settings whose variable the measured SIRCL table ``table`` records
    (``ROW_TABLE_SETTINGS``: link slots, link slot) are dropped, so the
    session applies the table's values. The measurement's own ``settings``
    replace any setting they name.
    """
    shape, size = row_group(name)
    try:
        _, row = tuning_row(defaults, shape, size)
        base = dict(row["settings"])
    except TransportError:
        base = {}
    recorded = set(((table or {}).get("settings") or {}))
    kept = {key: value for key, value in base.items() if ROW_TABLE_SETTINGS.get(key) not in recorded}
    return {**kept, **settings}


def table_identity(table_document):
    """``(shape, world, lanes, max_relays)`` of a ``sircl-tuning-table/v1`` key: the groups it can serve."""
    key = table_document["key"]
    return (str(key["shape"]), int(key["world"]), int(key["lanes"]), int(key["max_relays"]))


def measured_document(defaults, rows, tables, *, fabric, image_value, measured_at, binding, root=ROOT):
    """A measured ``sparkring-sircl-tuning/v1`` table over the default table ``defaults``.

    ``rows`` maps measured group names to their settings and ``tables`` the
    SHA-256 of each measured SIRCL table to its parsed document. A measured
    row's settings follow ``measured_row`` with the measured table of its
    groups (``row_identity``). Every row of ``defaults`` the measurement did not
    replace is carried as
    ``default:<source>``; a repository table of ``defaults`` is carried unless a
    measured table serves the same groups. ``binding`` holds every field of
    ``BINDING_FIELDS`` but ``defaults_sha256``, which is ``defaults``' digest.
    The result is not validated here: its tables must first reach ``HOST_TABLES``.
    """
    layouts = {name: {"source": CARRIED + row["source"], "settings": dict(row["settings"])}
               for name, row in defaults["layouts"].items()}
    for name, settings in rows.items():
        _require(GROUP_NAME.fullmatch(name), f"measured row {name!r} is not a group shape")
        own = [table for table in tables.values() if table_identity(table)[:2] == row_identity(name)]
        _require(len(own) <= 1, f"several measured SIRCL tables serve the {name} groups")
        layouts[name] = {"source": "measured",
                         "settings": measured_row(defaults, name, dict(settings), own[0] if own else None)}
    served = {table_identity(table) for table in tables.values()}
    entries = [{"path": f"{HOST_TABLES}/{digest}.json", "sha256": digest} for digest in sorted(tables)]
    for entry in defaults["tables"]:
        carried = json.loads((Path(root) / entry["path"]).read_text(encoding="utf-8"))
        if table_identity(carried) not in served:
            entries.append(dict(entry))
    sircl = image_lock.sircl(image_value)
    _require(sircl is not None, f"image {image_value.get('name')} carries no SIRCL layer")
    return {"schema": TUNING_SCHEMA, "source": "measured",
            "sircl": {"version": sircl["version"], "abi_version": sircl["abi_version"]},
            "fabric": fabric, "image_id": image_value["image_id"], "measured_at": measured_at,
            "layouts": dict(sorted(layouts.items())), "tables": entries,
            "binding": {**binding, "defaults_sha256": tuning_digest(defaults)}}


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
    explicit ``--transport sircl`` that cannot run, ``--nccl`` with the
    prepared or nccl transport, and ``--transport nccl`` without a fabric
    document are refused.
    """
    if backend == LIBSIRCL:
        from runtime.common import libsircl
        return libsircl.choose(image_value, document, nccl=nccl_mode(nccl))
    _require(backend in (None, *BACKENDS), f"--transport takes sircl, prepared, nccl or libsircl, not {backend!r}")
    nccl = nccl_mode(nccl)
    reason = unavailable(image_value, document)
    if backend is None:
        backend = "prepared" if reason else "sircl"
    if backend == "sircl" and reason:
        raise TransportError(f"--transport sircl cannot run here: {reason}")
    if backend == "nccl":
        _require(nccl is None, "--nccl applies to deployments on SIRCL; the nccl transport is NCCL itself")
        _require(document is not None, "--transport nccl reads the fabric document that names each Spark's RDMA "
                                       "devices; this cluster has none; sudo sparkring setup records one")
        return backend, None, None
    if backend == "prepared":
        _require(nccl is None, "--nccl applies to deployments on SIRCL; the prepared transport keeps its own NCCL "
                               "settings")
        return backend, None, reason
    return backend, nccl or DEFAULT_NCCL, None


def table_file(entry, *, root=ROOT, host_root="/"):
    """Where a tuning table's ``tables`` entry lies here: a repository path under ``root``, or a Spark's
    ``HOST_TABLES`` copy under ``host_root``."""
    if PurePosixPath(entry["path"]).is_absolute():
        return Path(host_root) / entry["path"].lstrip("/")
    return Path(root) / entry["path"]


def section(image_value, document, positions, *, nccl, tuning, dcp=1, root=ROOT, host_root="/"):
    """The deployment lock's ``transport`` section of a SIRCL deployment on ``positions`` of ``document``.

    ``tuning`` is the table in effect (``tuning_in_effect``) and ``dcp`` the
    profile's decode-context parallelism (``profile_dcp``). The section records
    the table's digest and the group's row, and each SIRCL table of ``tuning``
    whose key matches a session of the deployment and the image's SIRCL build:
    the tensor-parallel session's group or, with ``dcp`` above 1, the first
    decode-context-parallel group, as the SIRCL launcher matches them. Each
    entry names the sessions that take it and the settings it records.
    """
    _require(unavailable(image_value, document) is None, "SIRCL cannot run: " + str(unavailable(image_value, document)))
    nccl = nccl_mode(nccl) or DEFAULT_NCCL
    layout = sircl_layout(document)
    topology = group_topology(layout, positions)
    shape, size = topology.fabric.kind, len(topology.members)
    name, row = tuning_row(tuning, shape, size)
    sircl = image_lock.sircl(image_value)
    applies = tuning["sircl"] == {"version": sircl["version"], "abi_version": sircl["abi_version"]}
    sessions = {"tp": topology}
    if dcp > 1:
        sessions["dcp"] = dcp_topologies(layout, positions, dcp)[0]
    elif dcp != 1:
        dcp_topologies(layout, positions, dcp)
    tables = []
    if applies and tuning["tables"]:
        from spark_transport.sircl.sparkring_sircl import tuning as sircl_tuning
        # Each session's own facts with the image's SIRCL build in place of this checkout's.
        facts = {kind: {**sircl_tuning.facts_for_layout(group.session_layout(), group.lane_count),
                        **sircl["tuning_key"]} for kind, group in sessions.items()}
        for entry in tuning["tables"]:
            path = table_file(entry, root=root, host_root=host_root)
            table = sircl_tuning.Table(json.loads(path.read_text(encoding="utf-8")), entry["path"])
            takers = [kind for kind, own in facts.items() if not table.mismatches(own)]
            if takers:
                tables.append({**entry, "hash": table.hash, "settings": dict(table.settings), "sessions": takers})
        for kind in sessions:
            _require(sum(kind in entry["sessions"] for entry in tables) <= 1,
                     f"several measured tuning tables match the {SESSION_TITLES[kind]} of this group and image")
    return {
        "schema": SECTION_SCHEMA, "backend": "sircl", "nccl": nccl, "image": image_value["name"],
        "fabric": {"id": document["id"], "shape": document["shape"], "size": document["size"]},
        "group": {"layout": layout, "positions": [int(position) for position in positions], "shape": shape,
                  "size": size, "name": group_name(shape, size), "max_relays": topology.max_relays(),
                  "lanes": topology.lane_count, "cabling": topology.nccl_policy.value, "dcp": dcp},
        "devices": [rank_devices(document, topology, rank) for rank in range(size)],
        "tuning": {"source": tuning["source"], "sha256": tuning_digest(tuning), "row": name,
                   "row_source": row["source"] if applies else "rules",
                   "settings": dict(row["settings"]) if applies else {}, "tables": tables,
                   "applies": applies,
                   **({"measured_at": tuning["measured_at"]} if tuning["source"] == "measured" else {})},
        "sircl": sircl,
    }


def validate_section(value, card, image_runtime):
    """The ``transport`` section of a deployment lock after checking it against the lock's selection."""
    if isinstance(value, dict) and value.get("backend") == LIBSIRCL:
        from runtime.common import libsircl
        return libsircl.validate_section(value, card, image_runtime)
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
                                                       "lanes", "cabling", "dcp"},
             "The transport section's group has layout, positions, shape, size, name, max_relays, lanes, cabling and "
             "dcp")
    _require(type(group["dcp"]) is int and group["dcp"] >= 1 and group["size"] % group["dcp"] == 0,
             "The transport group's decode-context parallelism divides its ranks")
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
    fields = {"source", "sha256", "row", "row_source", "settings", "tables", "applies"}
    _require(isinstance(tuning, dict) and set(tuning) in (fields, fields | {"measured_at"})
             and isinstance(tuning["sha256"], str) and _SHA256.fullmatch(tuning["sha256"]),
             "The transport section records the tuning table's digest, row and settings")
    _require(("measured_at" in tuning) == (tuning["source"] == "measured"),
             "The transport section dates a measured tuning table and only a measured one")
    from spark_transport.sircl.sparkring_sircl import tuning as sircl_tuning
    for entry in tuning["tables"]:
        _require(isinstance(entry, dict) and set(entry) == {"path", "sha256", "hash", "settings", "sessions"}
                 and isinstance(entry["sha256"], str) and _SHA256.fullmatch(entry["sha256"])
                 and isinstance(entry["hash"], str) and _SHORT.fullmatch(entry["hash"])
                 and isinstance(entry["path"], str)
                 and (not PurePosixPath(entry["path"]).is_absolute()
                      or _HOST_TABLE.fullmatch(entry["path"]) is not None),
                 "The transport section names each SIRCL tuning table by path, SHA-256, hash, settings and the "
                 "sessions that take it")
        kinds = ("tp", "dcp") if group["dcp"] > 1 else ("tp",)
        _require(isinstance(entry["sessions"], list) and entry["sessions"]
                 and all(kind in kinds for kind in entry["sessions"]),
                 "A SIRCL tuning table is taken by the tensor-parallel session or, with decode-context parallelism, "
                 "by the decode-context-parallel sessions")
        _require(isinstance(entry["settings"], dict)
                 and all(name in sircl_tuning.SETTINGS and type(number) is int and number >= 1
                         for name, number in entry["settings"].items()),
                 "A SIRCL tuning table's settings are positive integers of " + ", ".join(sircl_tuning.SETTINGS))
    for kind in ("tp", "dcp"):
        _require(sum(kind in entry["sessions"] for entry in tuning["tables"]) <= 1,
                 f"The transport section names at most one SIRCL tuning table for the {SESSION_TITLES[kind]}")
    for key, setting in tuning["settings"].items():
        _require(key in SETTINGS and _setting(key, setting), f"Tuning setting {key}={setting!r} is not passed")
    image_lock.validate_sircl(value["sircl"])
    return value


# The nccl transport.

def ring_settings():
    """The NCCL settings under which NCCL's ring algorithm runs on a cycle of Sparks: the SIRCL serve launcher's
    ``RING_SETTINGS`` (``sparkring_sircl.vllm.serve.bundle``), which the bundle sets whenever it lets NCCL run there."""
    from spark_transport.sircl.sparkring_sircl.vllm.serve import bundle
    return dict(bundle.RING_SETTINGS)


# Why the deployment adds each NCCL ring setting on a whole cycle; the plan prints one line per setting.
NCCL_SETTING_REASONS = {
    "NCCL_ALGO": "NCCL's ring algorithm carries every collective on this cycle",
    "NCCL_SKIP_TREE_CONNECT": "NCCL must not build tree connections between Sparks that share no cable; this is the "
                              "repository's patched-NCCL cycle contract",
}
# What an nccl deployment's receipt and summary state to expect.
NCCL_EXPECTED = ("vLLM's PyNccl carries every tensor-parallel and expert-parallel collective; SIRCL is not loaded "
                 "and the RoCEnante slot is off")


def nccl_cabling_rule():
    """The cabling rule NCCL itself obeys, which the nccl transport requires of a group and of every
    decode-context-parallel group of its profile."""
    return ("NCCL cannot connect Sparks that share no cable; it runs on a cabled pair, or on a group whose "
            "consecutive ranks share cables around the whole group")


def nccl_refusal(reason):
    """The refusal for a group whose ranks reach each other through relays, worded as SIRCL's refusals are."""
    return (f"--transport nccl cannot run on this group: {reason}. {nccl_cabling_rule()}. SIRCL ring sessions run "
            "the Sparks between them (--transport sircl)")


def nccl_section(image_value, document, positions, *, dcp=1):
    """The deployment lock's ``transport`` section of an nccl deployment on ``positions`` of ``document``.

    The section records the group, each rank's RDMA devices as the fabric
    document names them at its position, and, on a whole cycle, the NCCL
    settings of NCCL's ring algorithm (``ring_settings``) with the reason the
    deployment adds each one. A group whose ranks reach each other through
    relays is refused (``nccl_refusal``), and so is a profile whose
    decode-context-parallel groups do.
    """
    _require(document is not None, "--transport nccl reads the fabric document that names each Spark's RDMA "
                                   "devices; this cluster has none; sudo sparkring setup records one")
    layout = sircl_layout(document)
    _, _, sircl_fabric = _sircl()
    try:
        raw_policy, raw_reason = sircl_fabric.nccl_policy_of(sircl_fabric.Layout.parse(layout), positions)
    except sircl_fabric.FabricError as error:
        raise TransportError(str(error)) from None
    _require(raw_policy is not sircl_fabric.NcclPolicy.NONE, nccl_refusal(raw_reason))
    topology = group_topology(layout, positions)
    shape, size = topology.fabric.kind, len(topology.members)
    if dcp != 1:
        for group in dcp_topologies(layout, positions, dcp):
            _require(group.nccl_policy is not sircl_fabric.NcclPolicy.NONE,
                     f"--transport nccl cannot run this profile's decode-context-parallel groups of {dcp} ranks: "
                     f"{group.nccl_reason}. {nccl_cabling_rule()}; run the profile with decode-context parallelism 1, "
                     "or on SIRCL ring sessions (--transport sircl)")
    settings = {} if topology.nccl_policy is sircl_fabric.NcclPolicy.ALL else ring_settings()
    return {
        "schema": SECTION_SCHEMA, "backend": "nccl", "image": image_value["name"],
        "fabric": {"id": document["id"], "shape": document["shape"], "size": document["size"]},
        "group": {"layout": layout, "positions": [int(position) for position in positions], "shape": shape,
                  "size": size, "name": group_name(shape, size), "max_relays": topology.max_relays(),
                  "lanes": topology.lane_count, "cabling": topology.nccl_policy.value, "dcp": dcp},
        "devices": [rank_devices(document, topology, rank) for rank in range(size)],
        "nccl": {"settings": settings, "reasons": {name: NCCL_SETTING_REASONS[name] for name in settings}},
    }


def validate_nccl_section(value, card, image_runtime):
    """The ``transport`` section of an nccl deployment lock after checking it against the lock's selection."""
    _, _, sircl_fabric = _sircl()
    _require(isinstance(value, dict) and value.get("schema") == SECTION_SCHEMA and value.get("backend") == "nccl",
             f"A deployment's transport section is a {SECTION_SCHEMA} nccl section")
    _require(set(value) == {"schema", "backend", "image", "fabric", "group", "devices", "nccl"},
             "The transport section has schema, backend, image, fabric, group, devices and nccl")
    _require(image_runtime is not None and value["image"] == image_runtime["name"] == card["release"],
             "The transport section names the deployment's installer image")
    fabric_value = value["fabric"]
    _require(isinstance(fabric_value, dict) and set(fabric_value) == {"id", "shape", "size"}
             and isinstance(fabric_value["id"], str) and _FABRIC_ID.fullmatch(fabric_value["id"]),
             "The transport section names the fabric document's identity, shape and size")
    try:
        fabric_layout.layout(fabric_value["shape"], fabric_value["size"])
    except ValueError as error:
        raise TransportError(str(error)) from None
    group = value["group"]
    _require(isinstance(group, dict) and set(group) == {"layout", "positions", "shape", "size", "name", "max_relays",
                                                       "lanes", "cabling", "dcp"},
             "The transport group has layout, positions, shape, size, name, max_relays, lanes, cabling and dcp")
    _require(type(group["dcp"]) is int and group["dcp"] >= 1 and group["size"] % group["dcp"] == 0,
             "The transport group's decode-context parallelism divides its ranks")
    _require(group["size"] == len(group["positions"]) == card["nodes"],
             f"The transport group has {card['nodes']} positions, one per rank")
    _require(group["layout"] == sircl_layout(fabric_value),
             "The transport group differs from its layout and positions")
    layout = group["layout"]
    topology = group_topology(layout, group["positions"])
    _require(topology.fabric.kind == group["shape"] and topology.max_relays() == group["max_relays"]
             and topology.lane_count == group["lanes"] and topology.nccl_policy.value == group["cabling"]
             and group["name"] == group_name(group["shape"], group["size"]),
             "The transport group differs from its layout and positions")
    _require(topology.nccl_policy is not sircl_fabric.NcclPolicy.NONE,
             "The transport group's ranks reach each other through relays: " + nccl_cabling_rule())
    devices = value["devices"]
    _require(isinstance(devices, list) and len(devices) == card["nodes"]
             and all(isinstance(row, list) and row and all(isinstance(name, str) for name in row) for row in devices),
             "The transport section lists each rank's RDMA devices")
    nccl = value["nccl"]
    _require(isinstance(nccl, dict) and set(nccl) == {"settings", "reasons"},
             "The transport section's nccl has settings and reasons")
    settings = nccl["settings"]
    _require(isinstance(settings, dict) and set(settings) <= set(ring_settings())
             and all(type(setting) is str for setting in settings.values()),
             "The transport section's nccl settings are the NCCL ring settings, as strings")
    if topology.nccl_policy is sircl_fabric.NcclPolicy.ALL:
        _require(settings == {}, "A cabled pair needs no NCCL setting added")
    else:
        _require(settings == ring_settings(),
                 "NCCL's ring on this cycle needs " + ", ".join(f"{name}={setting}" for name, setting in
                                                                sorted(ring_settings().items())))
    _require(isinstance(nccl["reasons"], dict) and set(nccl["reasons"]) == set(settings)
             and all(isinstance(text, str) and text for text in nccl["reasons"].values()),
             "The transport section's nccl gives the reason each added setting is needed")
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


def added_ring_settings(value, profile_environment):
    """The NCCL settings this deployment adds on a whole cycle with ``--nccl auto`` (``ring_settings``).

    The cycle contract needs exactly these settings, so NCCL builds no tree
    connections between Sparks that share no cable. The profile's own value
    stays when it already names the setting; a profile that names another
    value is refused.
    """
    if value["nccl"] != "auto" or value["group"]["cabling"] != "ring":
        return {}
    added = {}
    for key, setting in ring_settings().items():
        current = (profile_environment.get(key) or "").strip()
        _require(not current or current.lower() == setting.lower(),
                 f"the profile sets {key}={profile_environment.get(key)!r}; NCCL's ring on this cycle needs "
                 f"{key}={setting}, so NCCL builds no tree connections between Sparks that share no cable")
        if current != setting:
            added[key] = setting
    return added


def _options(settings):
    """A namespace with the schedules a tuning row sets (``serve.plan.schedule_environment``)."""
    class Options:
        pass
    options = Options()
    for name in SCHEDULE_SETTINGS:
        setattr(options, name, settings.get(name))
    return options


def profile_dcp(profile):
    """The decode-context parallelism of installer profile ``profile`` (its ``decode_context_parallel_size``)."""
    from runtime.common import profiles
    try:
        return int(profiles.resolve(profile)["serving"].get("decode_context_parallel_size") or 1)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise TransportError(f"the decode-context parallelism of profile {profile} cannot be read: {error}") from None


def dcp_topologies(layout_text, positions, size):
    """vLLM's decode-context-parallel groups of ``size`` consecutive ranks over ``positions``, each placed
    inside the tensor-parallel group, as SIRCL's adapter places them (``serve.plan.dcp_groups``)."""
    _, _, fabric = _sircl()
    positions = list(positions)
    _require(type(size) is int and size >= 1 and len(positions) % size == 0,
             f"decode-context parallelism {size} must divide the group's {len(positions)} ranks")
    try:
        layout = fabric.Layout.parse(layout_text)
        return [fabric.describe_group(layout, positions[start:start + size], parent=positions)
                for start in range(0, len(positions), size)]
    except fabric.FabricError as error:
        raise TransportError(str(error)) from None


def dcp_groups(value, arguments):
    """The decode-context-parallel groups of the profile's command, each placed inside the transport group.

    vLLM forms ``tensor-parallel size / dcp`` groups of ``dcp`` consecutive
    ranks (``--decode-context-parallel-size``). With ``dcp`` above 1 each
    group needs a SIRCL session of its own (``SIRCL_GROUPS=tp,dcp``), which
    routes over the tensor-parallel group's cables; an empty list for ``dcp``
    1. The command's size must be the one the section was made for
    (``group.dcp``), whose tables the sessions take.
    """
    plan, _, _ = _sircl()
    size = plan.recipe_dcp(arguments)
    _require(size == value["group"]["dcp"], f"the profile's command sets decode-context parallelism {size}; its "
                                            f"transport section was made for {value['group']['dcp']}")
    if size <= 1:
        return []
    return dcp_topologies(value["group"]["layout"], value["group"]["positions"], size)


def session_table(value, kind="tp"):
    """The entry of the SIRCL tuning table that the ``kind`` sessions (``tp`` or ``dcp``) of the transport
    section ``value`` take, or None when they take none and their rules choose."""
    return next((entry for entry in value["tuning"]["tables"] if kind in entry["sessions"]), None)


def table_settings(value, kind="tp"):
    """The session settings (``tuning.SETTINGS``) of the SIRCL tuning table the ``kind`` sessions of the
    transport section ``value`` take; empty when they take none."""
    entry = session_table(value, kind)
    return dict(entry["settings"]) if entry is not None else {}


def expected_session_settings(value):
    """``{stats() field: value}`` the tensor-parallel session of ``value`` reports for the variables of
    ``tuning.SETTINGS`` that the tuning row or the table sets: the row's value where it sets one, else the
    table's, which the session applies where its environment leaves the variable unset."""
    from spark_transport.sircl.sparkring_sircl import tuning as sircl_tuning
    found = dict(table_settings(value))
    row = value["tuning"]["settings"]
    found.update({variable: row[name] for name, variable in ROW_TABLE_SETTINGS.items() if name in row})
    return {sircl_tuning.SETTING_STATS[variable]: number for variable, number in sorted(found.items())}


def dcp_problems(value, groups, profile_environment, arguments, effective):
    """Refuse decode-context parallelism where the SIRCL launcher's ``--dcp-size`` refuses it.

    With mHC prefill row ownership on (``VLLM_GLM53_MHC_PREFILL_SHARD``),
    GLM-5.3-Flash starts only at the tensor and decode-context sizes a pinned
    vLLM build admits (``pins.MHC_ADMITS``, ``serve.plan.mhc_dcp_problem``).
    On a tensor-parallel group where NCCL may run, decode-context-parallel
    groups where it may not run refuse the settings that need NCCL there
    (``serve.plan.relay_conflicts``). The launcher's ``--dcp-size`` also
    requires a checkpoint of its list (``serve.plan.DCP_MODELS``) and the B12X
    attention backend; installer profiles set decode-context parallelism in
    their own recipe and are not checked against that list.
    """
    plan, guard, _ = _sircl()
    tensor, size = len(value["group"]["positions"]), len(groups[0].members)
    mhc = profile_environment.get(plan.MHC_SHARD, "0").strip() not in ("", "0")
    _require(plan.mhc_dcp_problem(tensor, size, mhc) is None,
             f"the profile turns on GLM-5.3-Flash's mHC prefill row ownership ({plan.MHC_SHARD}=1), which no pinned "
             f"vLLM build starts at TP{tensor} with decode-context parallelism {size} (sparkring_sircl.vllm.pins)")
    policy_, reason = guard.effective_policy(groups[0].nccl_policy, groups[0].nccl_reason, nccl_mode=value["nccl"],
                                             environ=profile_environment)
    if policy_.value == "none" and effective.value != "none":
        conflicts = plan.relay_conflicts(arguments, profile_environment)
        _require(not conflicts, "SIRCL's communicator refuses this profile's decode-context-parallel groups, where "
                                f"NCCL may not run ({reason}): " + "; ".join(conflicts))


def environment(value, profile_environment, arguments):
    """The SIRCL variables every rank's container gets, and the checks the SIRCL launcher applies.

    ``profile_environment`` is rank 0's adapted environment and ``arguments``
    its command, which holds the recipe's vLLM arguments.
    """
    plan, _, _ = _sircl()
    from spark_transport.sircl.sparkring_sircl import tuning as sircl_tuning
    settings = value["tuning"]["settings"]
    taken = table_settings(value)
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
    ring = added_ring_settings(value, profile_environment)
    effective, reason = policy(value, {**profile_environment, **ring})
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
        # The tensor-parallel session applies its table's settings where the row leaves them unset.
        sessions = [plan.session_settings(topology, name="tp", groups="", scoped=True, schedules=schedules,
                                          link_sizes=links, link_slots=settings.get("link_slots"),
                                          ring_gather_stagger=settings.get("ring_gather_stagger"),
                                          table_settings=taken)]
        dcp = dcp_groups(value, arguments)
        if dcp:
            # Each decode-context-parallel group gets a session of its own with the session defaults and the
            # settings of the table it takes: the tuning row's schedules and link sizes apply to the
            # tensor-parallel session only.
            sessions.append(plan.session_settings(dcp[0], name="dcp", scoped=False,
                                                  groups=plan.dcp_groups_text(len(topology.members),
                                                                              len(dcp[0].members)),
                                                  table_settings=table_settings(value, "dcp")))
            dcp_problems(value, dcp, profile_environment, arguments, effective)
        plan.session_problems(sessions)
        gid = int(profile_environment.get("NCCL_IB_GID_INDEX", "3"))
        common = {
            "SIRCL_MODE": "custom",
            "SIRCL_FABRIC": value["group"]["layout"],
            "SIRCL_RANK_POSITIONS": ",".join(str(position) for position in value["group"]["positions"]),
            "SIRCL_GROUPS": "tp,dcp" if dcp else "tp",
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
        # The NCCL settings the cycle contract needs, added where the profile leaves them unset.
        common.update(ring)
    except plan.ServePlanError as error:
        raise TransportError(str(error)) from None
    tables = value["tuning"]["tables"]
    if tables:
        # Every session picks the listed table whose key matches its own facts.
        common[plan.TUNING_VARIABLE] = ",".join(f"{TABLE_TARGET}/{entry['hash']}.json" for entry in tables)
    if taken:
        # As the SIRCL launcher refuses it (serve.plan.TuningPlan.conflicts): a row setting below the table's
        # would leave the table's choices that need more unable to run. The row reaches the tensor-parallel
        # session alone.
        conflicts = sircl_tuning.settings_conflicts(taken, common)
        _require(not conflicts, f"the tensor-parallel session takes SIRCL tuning table {session_table(value)['hash']}, "
                                f"whose choices need more than the tuning row {value['tuning']['row']} sets: "
                                + ", ".join(conflicts))
    return common, effective


def dcp_interleave(lock, arguments):
    """The ``--cp-kv-cache-interleave-size`` value every rank's command adds, or None.

    As the SIRCL launcher's ``--dcp-size`` does (``serve.plan.dcp_interleave``):
    with decode-context parallelism above 1, a model whose attention needs the
    KV cache interleaved in blocks of a multiple of its size (GLM-5.3-Flash: 4,
    ``serve.plan.DCP_INTERLEAVE``) gets that size where the recipe leaves the
    flag unset, and a recipe that gives another value is refused.
    """
    plan, _, _ = _sircl()
    size = plan.recipe_dcp(arguments)
    card = lock["selection"]
    checkpoint = f"{card['model_repository']}@{card['model_revision']}"
    try:
        return plan.dcp_interleave(size, checkpoint, arguments)
    except plan.ServePlanError:
        model = plan.DCP_MODELS[card["model_repository"]]
        raise TransportError(f"{model}'s attention under decode-context parallelism {size} needs "
                             f"{plan.INTERLEAVE_FLAG} divisible by {plan.DCP_INTERLEAVE[model]}; the profile's recipe "
                             "gives another value") from None


def adapt(specs, lock):
    """Each rank's container with SIRCL in front of every collective (the lock's ``transport`` section)."""
    value = lock["transport"]
    if value.get("backend") == LIBSIRCL:
        from runtime.common import libsircl
        return libsircl.adapt(specs, lock)
    if value["backend"] == "nccl":
        return nccl_adapt(specs, lock)
    plan, _, _ = _sircl()
    rows = lock["site"]["ranks"]
    _require(len(specs) == len(rows) == len(value["devices"]), "one container per rank of the transport group")
    common, effective = environment(value, specs[0].environment, specs[0].command)
    interleave = dcp_interleave(lock, specs[0].command)
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
        # A measured table's copy lies at the same absolute path on every Spark; a repository table in the
        # rank's own checkout of the deployment's source.
        mounts += [Bind(entry["path"] if PurePosixPath(entry["path"]).is_absolute()
                        else str(PurePosixPath(row["repository"]) / entry["path"]),
                        f"{TABLE_TARGET}/{entry['hash']}.json", True) for entry in value["tuning"]["tables"]]
        command = (*spec.command, plan.INTERLEAVE_FLAG, interleave) if interleave is not None else spec.command
        result.append(replace(spec, environment=environment_, mounts=tuple(mounts), command=command))
    shards = {spec.environment.get(plan.MHC_SHARD, "0").strip() not in ("", "0") for spec in result}
    _require(len(shards) == 1, f"the profile's ranks disagree on {plan.MHC_SHARD}")
    _require(len({spec.environment.get(plan.HC_PREFILL) for spec in result}) == 1,
             f"the profile's ranks disagree on {plan.HC_PREFILL}")
    return result


def nccl_adapt(specs, lock):
    """Each rank's container with vLLM's PyNccl carrying every collective (the lock's ``transport`` section).

    No SIRCL variable or plugin reaches the container, and the RoCEnante slot
    and the prepared transports stay off (the SIRCL launcher's
    ``DISABLED_TRANSPORTS``). Each rank's ``NCCL_IB_HCA`` names the RDMA
    devices of its own lanes, in the style of the profile's own value
    (``serve.plan.nccl_hca_value``), which keeps the profile's port suffixes;
    a whole cycle takes the section's NCCL ring settings. The profile's other
    NCCL settings are kept.
    """
    plan, _, _ = _sircl()
    value = lock["transport"]
    rows = lock["site"]["ranks"]
    _require(len(specs) == len(rows) == len(value["devices"]), "one container per rank of the transport group")
    result = []
    for rank, (spec, row) in enumerate(zip(specs, rows, strict=True)):
        environment_ = {key: item for key, item in spec.environment.items() if not key.startswith("SIRCL_")}
        owned = [key for key in plan.OWNED if key in environment_]
        _require(not owned, f"rank {rank}: the profile sets {owned}, which the nccl transport owns")
        plugins = [item for item in environment_.get("VLLM_PLUGINS", "").split(",") if item and item != "sircl"]
        environment_["VLLM_PLUGINS"] = ",".join(plugins)
        environment_.update(plan.DISABLED_TRANSPORTS)
        environment_["NCCL_IB_HCA"] = plan.nccl_hca_value(spec.environment.get("NCCL_IB_HCA"),
                                                          value["devices"][rank])
        environment_.update(value["nccl"]["settings"])
        result.append(replace(spec, environment=environment_))
    return result


# Text.

def plan_lines(value, notes=()):
    """What ``sparkring install`` prints about a deployment's transport before it asks."""
    if value.get("backend") == LIBSIRCL:
        from runtime.common import libsircl
        return libsircl.plan_lines(value, notes)
    if value["backend"] == "nccl":
        return nccl_plan_lines(value, notes)
    tuning = value["tuning"]
    row = tuning["row"]
    # A measured table's row for a shape it did not measure is the default table's row.
    source = tuning["row_source"].removeprefix(CARRIED)
    if not tuning["applies"]:
        evidence = f"the default table is for another SIRCL build; SIRCL's own rules apply on this {row} group"
    elif source == "measured" and tuning["row_source"] == "measured" and tuning["source"] == "measured":
        evidence = f"measured on this fabric {tuning['measured_at']}, {row}"
    elif source == "measured":
        evidence = f"default table, {row}"
    elif source == "rules":
        evidence = f"default table, {value['group']['name']}: not measured, SIRCL's own rules apply"
    elif source == "design":
        evidence = f"default table, {row}: the design's settings, not measured"
    else:
        evidence = f"default table, {value['group']['name']}: {source.replace(':', ' from ')}"
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
    # SIRCL's plans, bundles and receipts state the same rule.
    lines.append(f"  {nccl_rule()}")
    if value["nccl"] != "never" and value["group"]["cabling"] == "ring":
        lines.append("  NCCL on this cycle: " + ", ".join(f"{name}={setting}" for name, setting in
                                                           sorted(ring_settings().items()))
                     + "; the installer adds what the profile leaves unset, so NCCL's ring runs but builds no "
                       "tree connections between Sparks that share no cable")
    if tuning["source"] == "measured" and tuning["row_source"] == "measured":
        lines.append(f"  Measured row {row}: {MEASURED_ROW_RULE}")
    if tuning["settings"]:
        lines.append("  SIRCL settings: " + ", ".join(f"{key} {setting}" for key, setting in
                                                       sorted(tuning["settings"].items())))
    for entry in tuning["tables"]:
        applied = ", ".join(f"{name}={number}" for name, number in sorted(entry["settings"].items()))
        takers = " and the ".join(SESSION_TITLES[kind] for kind in entry["sessions"])
        where = " where the row leaves them unset" if "tp" in entry["sessions"] else ""
        lines.append(f"  SIRCL tuning table {entry['hash']} for the {takers}: the measured algorithm, schedule, piece "
                     "and launch grid per collective and size" + (f"; its sessions apply {applied}{where}"
                                                                  if applied else ""))
    for note in notes:
        lines.append("  Note: " + note)
    return lines


def prepared_line(reason, explicit):
    if explicit:
        return "Transport: prepared (--transport prepared)"
    return f"Transport: prepared, because {reason}" if reason else "Transport: prepared"


def nccl_plan_lines(value, notes=()):
    """What ``sparkring install`` prints about an nccl deployment's transport before it asks."""
    group = value["group"]
    relays = group["max_relays"]
    lines = ["Transport: nccl on every collective; SIRCL is not loaded and the RoCEnante slot is off "
             "(--transport nccl)"]
    lines.append(f"  NCCL group: {group['name']} at positions {', '.join(map(str, group['positions']))}; "
                 f"{group['lanes']} lanes per peer, "
                 + (f"at most {relays} relay{'s' if relays != 1 else ''} on a lane" if relays else "no relays"))
    for name, reason in sorted(value["nccl"]["reasons"].items()):
        lines.append(f"  {name}={value['nccl']['settings'][name]}: {reason}")
    if not value["nccl"]["reasons"]:
        lines.append("  NCCL settings: the profile's own; a cabled pair needs none added")
    for note in notes:
        lines.append("  Note: " + note)
    return lines


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
    if lock["transport"].get("backend") == LIBSIRCL:
        from runtime.common import libsircl
        return libsircl.admit_layer(lock, run=run)
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
    """Raise TransportError unless this Spark's fabric document has the deployment's identity and — on a SIRCL
    deployment, which records tuning — the Spark holds every measured tuning table the deployment mounts from
    ``HOST_TABLES``, byte for byte. A libsircl deployment's check is ``libsircl.check_host_document``."""
    if value.get("backend") == LIBSIRCL:
        from runtime.common import libsircl
        return libsircl.check_host_document(value, root=root)
    path = Path(root) / fabric_document.HOST_PATH.lstrip("/")
    try:
        document = fabric_document.load(path)
    except fabric_document.FabricDocumentError as error:
        raise TransportError(f"SIRCL reads the fabric document, and this Spark's copy cannot be used: {error}") from None
    _require(document["id"] == value["fabric"]["id"],
             f"This deployment was made on fabric {value['fabric']['id'][7:19]}; this Spark records fabric "
             f"{document['id'][7:19]}. Run sudo sparkring install again")
    _require(fabric_document.uniform_names(document), "the Sparks name their fabric devices differently")
    for entry in (value.get("tuning") or {}).get("tables") or []:
        if not PurePosixPath(entry["path"]).is_absolute():
            continue
        path = table_file(entry, host_root=root)
        _require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"],
                 f"This Spark lacks the measured SIRCL tuning table {entry['sha256'][:12]} ({entry['path']}) that "
                 "this deployment uses; sudo sparkring fabric tune --distribute copies it to every Spark")
    return {"ok": True, "fabric": document["id"]}
