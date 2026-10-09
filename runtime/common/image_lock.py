"""Installer image locks of every schema, and the image ``sparkring install`` uses without ``--image``.

Three lock schemas exist. ``runtime/common/installer_image.py`` validates and
admits the first two (``sparkring-installer-image/v1`` and ``/v2``); its
bytes are part of every generated Compose export's identity
(``compose.source_inventory``), so it stays the v1/v2 adapter and this module
adds the third:

``sparkring-installer-image/v3`` holds every v2 field and:

- ``line``: the image line, ``kraken`` (images built on Local Inference Lab's
  ``karmic-kraken-beta`` vLLM and B12X branches);
- ``transports``: the collective transports the image carries, sorted:
  ``libsircl`` (SIRCL's NCCL-compatible C library under vLLM's PyNccl),
  ``prepared`` (the prepared RoCEnante transport) and ``sircl`` (SIRCL ring
  sessions). This package admits only v3 locks that list ``prepared``, whose
  v2 fields then keep their meaning;
- ``sircl``: the SIRCL layer, or null when ``transports`` lacks ``sircl``: the
  package version, the native ABI, the wheel installed in site-packages, the
  two prebuilt native libraries (``roce_proxy-<digest>.so`` and
  ``p2p_proxy-<digest>.so`` under ``/opt/sparkring/sircl/lib``, where
  ``<digest>`` is the first 16 hex digits of the SHA-256 of the C source they
  were built from), the layer receipt
  ``/opt/sparkring/receipts/sircl-layer.json``, the key a measured SIRCL
  tuning table must match (``sparkring_sircl.tuning.KEY_FIELDS``: native and
  kernel source hashes and ``<version>/abi<n>``), and the pinned vLLM builds
  of ``sparkring_sircl.vllm.pins`` the image's vLLM matches;
- ``tuning_defaults_sha256``: the SHA-256 of the default SIRCL tuning table the
  image was released with (``runtime/common/sircl-tuning-defaults.json``);
- ``archived``: whether the release is archived; an archived image stays
  selectable by name;
- ``libsircl``, present only when ``transports`` lists ``libsircl``: the
  libsircl layer (``runtime/images/libsircl_layer.py``): the library version,
  the source it was built from, the library under
  ``/opt/sparkring/libsircl/lib`` with its SHA-256, the NCCL API level it
  reports, whether it has the fail-stop mode (it reads
  ``LIBSIRCL_FAIL_STOP``), the vLLM plugin module that selects it with its
  SHA-256, and the layer receipt
  ``/opt/sparkring/receipts/libsircl-layer.json``. The source is named by
  exactly one field: ``source_tree``, the git tree id of
  ``spark_transport/libsircl`` at the commit the layer built, or, in a lock
  of a layer built while this repository vendored libsircl snapshots (such
  as image ``27e9f75c0d09``'s), ``snapshot``, that snapshot's tree digest.
  A v3 lock without the layer has no such field, so it validates as it did
  before the field existed;
- ``vllm_plugins``, present only when a derived layer added vLLM general
  plugins to the image (``runtime/images/derived_layer.py``, ``Layer.plugins``):
  each plugin's entry-point name in ``vllm.general_plugins`` and its
  distribution version, which the layer's build probed in the built image.
  The plugins every installer image or its transport layers carry
  (``BUILT_IN_PLUGINS``) are not listed. A lock without added plugins has no
  such field.

A profile whose serving environment's ``VLLM_PLUGINS`` names a plugin outside
``BUILT_IN_PLUGINS`` runs only on an image whose lock lists that plugin in
``vllm_plugins`` (``plugin_problem``). vLLM loads only the named plugins it
finds installed and skips the others without an error, so ``for_profile``
refuses such an image instead of serving without the plugin.

``v2_view`` gives ``installer_image`` the v2 fields of a v3 lock, so image
admission, the deployment lock's ``image_runtime`` and storage planning stay
those of v2; the deployment lock's ``transport`` section
(``runtime/common/transport.py``) carries the SIRCL layer.

The default image of ``sparkring install`` (``default``) is the newest
kraken-line v3 release that carries SIRCL and that a GitHub release publishes
(``installer-releases.json``); without one it is ``installer_image``'s
default. Compose exports keep ``installer_image``'s default.
"""
import re

from runtime.common import installer_image, profiles

SCHEMA_V3 = "sparkring-installer-image/v3"
LINES = ("kraken",)
TRANSPORTS = ("libsircl", "prepared", "sircl")
V3_FIELDS = (installer_image.FIELDS[installer_image.SCHEMA] - {"schema"}) | {
    "schema", "line", "transports", "sircl", "tuning_defaults_sha256", "archived"}
# Where the libsircl layer installs the library and its receipt.
LIBSIRCL_LIBRARY_DIRECTORY = "/opt/sparkring/libsircl/lib"
LIBSIRCL_RECEIPT = "/opt/sparkring/receipts/libsircl-layer.json"
LIBSIRCL_FIELDS = {"version", "library", "nccl_api_version", "fail_stop", "plugin", "receipt"}
# The fields that can name a libsircl layer's source, one per block, and the form of each: the git tree id
# of spark_transport/libsircl (SHA-1 or SHA-256 repositories), or the tree digest of a vendored libsircl
# snapshot (the SHA-256 of its FILES.sha256 list), which layers built before the source moved into this
# repository record.
LIBSIRCL_SOURCES = {"source_tree": re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}"), "snapshot": re.compile(r"[0-9a-f]{64}")}
# Where the SIRCL layer puts its prebuilt native libraries and its receipt.
LIBRARY_DIRECTORY = "/opt/sparkring/sircl/lib"
LAYER_RECEIPT = "/opt/sparkring/receipts/sircl-layer.json"
LIBRARY_STEMS = {"native": "roce_proxy", "p2p": "p2p_proxy"}
SIRCL_FIELDS = {"version", "abi_version", "wheel", "native", "p2p", "receipt", "tuning_key", "vllm_pins"}
# Optional fields of a v3 lock.
OPTIONAL_FIELDS = ("libsircl", "vllm_plugins")
# vLLM general plugins that a lock's vllm_plugins does not list: the installer images' own
# (installer_image.PLUGINS), SIRCL's (the SIRCL layer) and libsircl's (the libsircl layer).
BUILT_IN_PLUGINS = (*installer_image.PLUGINS, "sircl", "libsircl")
_PLUGIN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHORT = re.compile(r"[0-9a-f]{16}")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_PIN = re.compile(r"[a-z0-9][a-z0-9.-]{0,95}")


def schema(value):
    return value.get("schema") if isinstance(value, dict) else None


def _require(condition, text):
    if not condition:
        raise ValueError(text)


def _digest(value, label):
    _require(isinstance(value, str) and _SHA256.fullmatch(value), f"The SIRCL layer records a SHA-256 of {label}")


def validate_sircl(value):
    """The ``sircl`` block of a v3 lock after checking its fields; ValueError otherwise."""
    _require(isinstance(value, dict) and set(value) == SIRCL_FIELDS,
             "The SIRCL layer records " + ", ".join(sorted(SIRCL_FIELDS)))
    version, abi = value["version"], value["abi_version"]
    _require(isinstance(version, str) and _VERSION.fullmatch(version), "The SIRCL layer records its package version")
    _require(type(abi) is int and abi > 0, "The SIRCL layer records its native ABI version")
    wheel = value["wheel"]
    _require(isinstance(wheel, dict) and set(wheel) == {"name", "sha256"}
             and wheel["name"] == f"sparkring_sircl-{version}-py3-none-any.whl",
             "The SIRCL layer's wheel is sparkring_sircl-<version>-py3-none-any.whl")
    _digest(wheel["sha256"], "its wheel")
    for kind, stem in LIBRARY_STEMS.items():
        library = value[kind]
        _require(isinstance(library, dict) and set(library) == {"path", "sha256", "source_digest"},
                 f"The SIRCL layer records the path, SHA-256 and source digest of its {kind} library")
        _require(isinstance(library["source_digest"], str) and _SHORT.fullmatch(library["source_digest"])
                 and library["path"] == f"{LIBRARY_DIRECTORY}/{stem}-{library['source_digest']}.so",
                 f"The SIRCL layer's {kind} library is {LIBRARY_DIRECTORY}/{stem}-<source digest>.so")
        _digest(library["sha256"], f"its {kind} library")
    receipt = value["receipt"]
    _require(isinstance(receipt, dict) and set(receipt) == {"path", "sha256"} and receipt["path"] == LAYER_RECEIPT,
             f"The SIRCL layer's receipt is {LAYER_RECEIPT}")
    _digest(receipt["sha256"], "its receipt")
    key = value["tuning_key"]
    _require(isinstance(key, dict) and set(key) == {"native", "kernels", "sircl"}
             and key["native"] == value["native"]["source_digest"]
             and isinstance(key["kernels"], str) and _SHORT.fullmatch(key["kernels"])
             and key["sircl"] == f"{version}/abi{abi}",
             "The SIRCL layer's tuning key names its native and kernel source hashes and <version>/abi<n>")
    pins = value["vllm_pins"]
    _require(isinstance(pins, list) and pins == sorted(set(pins))
             and all(isinstance(pin, str) and _PIN.fullmatch(pin) for pin in pins),
             "The SIRCL layer lists the pinned vLLM builds its image matches, sorted")
    return value


def validate_libsircl(value):
    """The ``libsircl`` block of a v3 lock after checking its fields; ValueError otherwise."""
    sources = set(value) & set(LIBSIRCL_SOURCES) if isinstance(value, dict) else set()
    _require(isinstance(value, dict) and len(sources) == 1 and set(value) - sources == LIBSIRCL_FIELDS,
             "The libsircl layer records " + ", ".join(sorted(LIBSIRCL_FIELDS)) + " and its source: source_tree "
             "or snapshot")
    version = value["version"]
    _require(isinstance(version, str) and _VERSION.fullmatch(version), "The libsircl layer records its version")
    field = sources.pop()
    _require(isinstance(value[field], str) and LIBSIRCL_SOURCES[field].fullmatch(value[field]),
             "The libsircl layer records the git tree id of its source (spark_transport/libsircl)"
             if field == "source_tree" else "The libsircl layer records its snapshot's tree digest")
    _require(type(value["nccl_api_version"]) is int and value["nccl_api_version"] > 0,
             "The libsircl layer records the NCCL API level its library reports")
    _require(type(value["fail_stop"]) is bool, "The libsircl layer records whether its library has the fail-stop mode")
    library = value["library"]
    _require(isinstance(library, dict) and set(library) == {"path", "sha256"}
             and library["path"] == f"{LIBSIRCL_LIBRARY_DIRECTORY}/libsircl.so.{version}"
             and isinstance(library["sha256"], str) and _SHA256.fullmatch(library["sha256"]),
             f"The libsircl layer's library is {LIBSIRCL_LIBRARY_DIRECTORY}/libsircl.so.<version> with its SHA-256")
    plugin = value["plugin"]
    _require(isinstance(plugin, dict) and set(plugin) == {"name", "path", "sha256"} and plugin["name"] == "libsircl"
             and isinstance(plugin["path"], str) and plugin["path"].endswith("-packages/sparkring_libsircl.py")
             and isinstance(plugin["sha256"], str) and _SHA256.fullmatch(plugin["sha256"]),
             "The libsircl layer records its vLLM plugin libsircl: sparkring_libsircl.py in site-packages and its "
             "SHA-256")
    receipt = value["receipt"]
    _require(isinstance(receipt, dict) and set(receipt) == {"path", "sha256"} and receipt["path"] == LIBSIRCL_RECEIPT
             and isinstance(receipt["sha256"], str) and _SHA256.fullmatch(receipt["sha256"]),
             f"The libsircl layer's receipt is {LIBSIRCL_RECEIPT}")
    return value


def validate_vllm_plugins(value):
    """The ``vllm_plugins`` block of a v3 lock after checking it; ValueError otherwise."""
    _require(isinstance(value, dict) and value
             and all(isinstance(name, str) and _PLUGIN.fullmatch(name) and name not in BUILT_IN_PLUGINS
                     and isinstance(version, str) and _VERSION.fullmatch(version) for name, version in value.items()),
             "The image's added vLLM plugins map each entry-point name, other than "
             + ", ".join(BUILT_IN_PLUGINS) + ", to its version")
    return value


def validate_v3(value, profile):
    """A v3 lock after checking its v3 fields and, through ``v2_view``, its v2 fields for ``profile``."""
    _require(isinstance(value, dict) and V3_FIELDS <= set(value) <= V3_FIELDS | set(OPTIONAL_FIELDS),
             f"Expected a complete {SCHEMA_V3} lock")
    _require(value["line"] in LINES, "A v3 image lock names its image line: " + ", ".join(LINES))
    transports = value["transports"]
    _require(isinstance(transports, list) and transports == sorted(set(transports)) and set(transports) <= set(TRANSPORTS),
             "A v3 image lock lists the transports it carries, sorted: " + ", ".join(TRANSPORTS))
    _require("prepared" in transports,
             "This package admits v3 images that also carry the prepared transport, whose fields the lock keeps")
    if "sircl" in transports:
        validate_sircl(value["sircl"])
    else:
        _require(value["sircl"] is None, "A v3 image lock without the sircl transport records no SIRCL layer")
        _require(not sircl_only(value), f"{', '.join(sircl_only(value))} run only on SIRCL ring sessions; a v3 "
                                        "image lock lists them only when its image carries the SIRCL layer")
    if "libsircl" in transports:
        _require("libsircl" in value, "A v3 image lock that lists the libsircl transport records its libsircl layer")
        validate_libsircl(value["libsircl"])
    else:
        _require("libsircl" not in value, "A v3 image lock without the libsircl transport records no libsircl layer")
    if "vllm_plugins" in value:
        validate_vllm_plugins(value["vllm_plugins"])
    _digest(value["tuning_defaults_sha256"], "the default tuning table")
    _require(type(value["archived"]) is bool, "A v3 image lock says whether it is archived")
    installer_image.validate(v2_view(value), profile)
    return value


# Checkpoints whose files only some vLLM builds read, by ``repository@revision``, with the pinned
# vLLM builds (``sparkring_sircl.vllm.pins``) one of which the image's vLLM must match; the lock's
# ``sircl.vllm_pins`` lists the builds an image matches. The CSF checkpoint of GLM-5.3-Flash stores
# its routed experts' block scales compressed, which the nvfp4_csf quantization and loader of the
# kraken-beta build of 2026-10-07 read.
CHECKPOINT_BUILDS = {
    "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd33efa73c3bb7c95b74eee10cad34f9be":
        ("sparkring-kraken-beta-20261007-bc9ea774",),
}


# The status of a checkpoint in CHECKPOINT_BUILDS until an installation of it passes the installer's checks;
# the plan states it (checkpoint_notice), because a profile installs such a checkpoint without --checkpoint on
# an image that reads it. Remove an entry when a record shows an installation of that checkpoint that passed.
CHECKPOINT_STATUS = {
    "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd33efa73c3bb7c95b74eee10cad34f9be":
        "research-only: no installation of it has passed the installer's checks",
}


def checkpoint_notice(value, profile, card, *, preferred):
    """The plan's note on the checkpoint of ``card`` when CHECKPOINT_STATUS records its status, else None.

    ``preferred`` says the installer chose the checkpoint without ``--checkpoint``, as the profile's
    preferred checkpoint on the image of lock ``value``; the note then names the flag that installs the
    profile's default checkpoint."""
    status = CHECKPOINT_STATUS.get(f"{card['model_repository']}@{card['model_revision']}")
    if status is None:
        return None
    text = (f"Checkpoint {card['target_variant']} ({card['model_repository']} at {card['model_revision'][:12]}) "
            f"is {status}")
    if preferred:
        from runtime.common import setup
        default = setup.selection(profile)["target_variant"]
        text += (f". {profile} installs it without --checkpoint on image {value.get('name')}, whose vLLM reads "
                 f"it; --checkpoint {default} installs the profile's default checkpoint")
    return text + "."


def checkpoint_problem(value, card):
    """Why the image of lock ``value`` cannot read the checkpoint of ``card``, or None."""
    key = f"{card['model_repository']}@{card['model_revision']}"
    builds = CHECKPOINT_BUILDS.get(key)
    if not builds:
        return None
    pins = list((sircl(value) or {}).get("vllm_pins") or ())
    if set(builds) & set(pins):
        return None
    named = f" (--checkpoint {card['target_variant']})" if card.get("target_variant") else ""
    return (f"Checkpoint {card['model_repository']} at {card['model_revision'][:12]}{named} "
            f"needs an image whose vLLM is the pinned build {' or '.join(builds)}; image {value.get('name')} "
            + (f"matches {', '.join(pins)}" if pins else "records no pinned vLLM build"))


def preferred_checkpoint(value, profile):
    """The checkpoint that installer profile ``profile`` installs without ``--checkpoint`` on the image of
    lock ``value``: its ``preferred_checkpoint`` when that image reads it, else None for its default.

    A profile prefers a checkpoint that only some vLLM builds read (CHECKPOINT_BUILDS), such as
    GLM-5.3-Flash's CSF checkpoint. Its default checkpoint, whose settings are the profile's own, is
    what every other image installs, so a profile whose admitted images differ keeps one installable
    default on each of them.
    """
    from runtime.common import qwen_flash_next, setup
    card = setup.selection(profile)
    name = qwen_flash_next.preferred_checkpoint(profiles.read_json(profiles.local_path(card["configuration"])))
    if name is None:
        return None
    return None if checkpoint_problem(value, setup.selection(profile, name)) else name


def sircl_only(value):
    """The profiles a lock lists that only SIRCL ring sessions run (``installer_image.SIRCL_ONLY``)."""
    listed = value.get("profiles") if isinstance(value, dict) else None
    return [name for name in listed if name in installer_image.SIRCL_ONLY] if isinstance(listed, list) else []


def validate(value, profile):
    """Any lock schema; v1 and v2 through ``installer_image.validate``, which may not list SIRCL-only profiles."""
    if schema(value) == SCHEMA_V3:
        return validate_v3(value, profile)
    _require(not sircl_only(value), f"{', '.join(sircl_only(value))} run only on SIRCL ring sessions; only an "
                                    f"image lock {SCHEMA_V3} whose image carries the SIRCL layer lists them")
    return installer_image.validate(value, profile)


def v2_view(value):
    """The v2 fields of a v3 lock with the v2 schema; a v1 or v2 lock unchanged."""
    if schema(value) != SCHEMA_V3:
        return value
    fields = installer_image.FIELDS[installer_image.SCHEMA]
    return {**{key: value[key] for key in fields if key != "schema"}, "schema": installer_image.SCHEMA}


def transports(value):
    """The transports the image of ``value`` carries: ``("prepared",)`` for a v1 or v2 lock."""
    return tuple(value["transports"]) if schema(value) == SCHEMA_V3 else ("prepared",)


def sircl(value):
    """The SIRCL layer of ``value``, or None."""
    return value.get("sircl") if schema(value) == SCHEMA_V3 else None


def libsircl(value):
    """The libsircl layer of ``value``, or None."""
    return value.get("libsircl") if schema(value) == SCHEMA_V3 else None


def libsircl_source(block):
    """``(field, value)`` naming the source of a validated libsircl block: ``source_tree`` or ``snapshot``."""
    field = "source_tree" if "source_tree" in block else "snapshot"
    return field, block[field]


def vllm_plugins(value):
    """The vLLM general plugins a derived layer added to the image of ``value``: name -> version; {} for none."""
    return dict(value.get("vllm_plugins") or {}) if schema(value) == SCHEMA_V3 else {}


def required_plugins(profile, environment=None):
    """The plugins ``profile``'s ``VLLM_PLUGINS`` names beyond ``BUILT_IN_PLUGINS``, in its order.

    ``environment`` is the profile's serving environment; it defaults to its configuration's.
    """
    if environment is None:
        environment = installer_image.profile_environment(profile)
    names = [name.strip() for name in environment.get("VLLM_PLUGINS", "").split(",") if name.strip()]
    return [name for name in dict.fromkeys(names) if name not in BUILT_IN_PLUGINS]


def plugin_problem(value, profile, environment=None):
    """Why the image of lock ``value`` cannot run ``profile``'s vLLM plugins, or None."""
    carried = vllm_plugins(value)
    missing = [name for name in required_plugins(profile, environment) if name not in carried]
    if not missing:
        return None
    listed = ", ".join(f"{name} {version}" for name, version in sorted(carried.items())) or "no added vLLM plugin"
    return (f"{profile} loads the vLLM plugins {', '.join(missing)} (VLLM_PLUGINS of its configuration), which image "
            f"{value.get('name')} does not carry: its lock lists {listed}. Select an image lock {SCHEMA_V3} whose "
            "vllm_plugins lists them")


def line(value):
    return value["line"] if schema(value) == SCHEMA_V3 else None


def profiles_of(value):
    return installer_image.profiles_of(v2_view(value))


def _published(rows):
    """Release names that a GitHub release publishes (installer-releases.json)."""
    return set(installer_image.release_tags().values()) & {row["name"] for row in rows}


def default_row(rows=None):
    """The catalog row ``sparkring install`` uses without ``--image``.

    The newest (by release name) kraken-line v3 image that carries SIRCL, is
    not archived and is published by a GitHub release; without one, the row of
    ``installer_image.DEFAULT_LOCK``.
    """
    rows = installer_image.catalog() if rows is None else rows
    published = _published(rows)
    candidates = [row for row in rows if schema(row["lock"]) == SCHEMA_V3 and "sircl" in row["lock"]["transports"]
                  and not row["lock"]["archived"] and row["name"] in published]
    if candidates:
        return max(candidates, key=lambda row: row["name"])
    return next(row for row in rows if row["path"] == installer_image.DEFAULT_LOCK)


def default():
    """The lock of the image ``sparkring install`` uses without ``--image``."""
    return default_row()["lock"]


def catalog():
    """``installer_image.catalog()`` rows with ``default`` meaning the install default, and ``line``,
    ``transports`` and ``archived`` of each lock; the default first, then newest name first."""
    rows = installer_image.catalog()
    chosen = default_row(rows)["name"]
    result = []
    for row in rows:
        value = row["lock"]
        result.append({**row, "default": row["name"] == chosen, "line": line(value),
                       "transports": list(transports(value)),
                       "archived": bool(value.get("archived")) if schema(value) == SCHEMA_V3 else False})
    result.sort(key=lambda row: row["name"], reverse=True)
    return sorted(result, key=lambda row: not row["default"])


def lock_path(name):
    """The lock file of the image ``name`` selects; None when it selects the install default.

    ``name`` is a release name, the GitHub release tag that published it, or a
    part of a release name between hyphens that only one image has, as for
    ``installer_image.lock_path``.
    """
    rows = catalog()
    by_name = {row["name"]: row for row in rows}
    tags = installer_image.release_tags()
    if name in tags:
        if tags[name] not in by_name:
            raise ValueError(f"Release {name} published {tags[name]}, whose lock this package does not carry")
        row = by_name[tags[name]]
    elif name in by_name:
        row = by_name[name]
    else:
        matches = [row for row in rows if f"-{name}-" in f"-{row['name']}-"]
        if len(matches) > 1:
            raise ValueError(f"Image {name} matches several images: {', '.join(row['name'] for row in matches)}; "
                             "give one full name")
        if not matches:
            raise ValueError(f"No installer image is named {name}; sparkring images lists them")
        row = matches[0]
    return None if row["default"] else row["path"]


def for_profile(profile, explicit=None):
    """The image lock (any schema) for one installer profile: ``explicit``, else the install default.

    A lock whose image lacks a vLLM plugin the profile loads is refused (``plugin_problem``).
    """
    replaced = profiles.replacement_message(profile)
    if replaced:
        raise ValueError(replaced)
    value = explicit if explicit is not None else default()
    if schema(value) != SCHEMA_V3:
        if profile in installer_image.SIRCL_ONLY:
            others = [row["name"] for row in catalog() if profile in profiles_of(row["lock"])]
            raise ValueError(f"{profile} runs only on SIRCL ring sessions, and image {value.get('name')} carries no "
                             f"SIRCL layer; images that run it: {', '.join(others) or 'none in this package'}. A "
                             "development image lock that lists it is selected with --image-lock")
        _require(not sircl_only(value), f"{', '.join(sircl_only(value))} run only on SIRCL ring sessions; only an "
                                        f"image lock {SCHEMA_V3} whose image carries the SIRCL layer lists them")
        # v1 and v2 locks keep installer_image's selection, refusals and messages.
        selected = installer_image.for_profile(profile, explicit)
        problem = plugin_problem(selected, profile)
        _require(problem is None, problem)
        return selected
    try:
        selected = validate(value, profile)
    except ValueError as error:
        if explicit is None or schema(value) not in (installer_image.SCHEMA, SCHEMA_V3) \
                or profile in value.get("profiles", ()):
            raise
        others = [row["name"] for row in catalog() if profile in profiles_of(row["lock"])]
        raise ValueError(f"{error}; images that run it: {', '.join(others) or 'none'}") from None
    problem = plugin_problem(selected, profile)
    _require(problem is None, problem)
    return selected
