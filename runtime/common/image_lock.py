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
  selectable by name.

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
TRANSPORTS = ("prepared", "sircl")
V3_FIELDS = (installer_image.FIELDS[installer_image.SCHEMA] - {"schema"}) | {
    "schema", "line", "transports", "sircl", "tuning_defaults_sha256", "archived"}
# Where the SIRCL layer puts its prebuilt native libraries and its receipt.
LIBRARY_DIRECTORY = "/opt/sparkring/sircl/lib"
LAYER_RECEIPT = "/opt/sparkring/receipts/sircl-layer.json"
LIBRARY_STEMS = {"native": "roce_proxy", "p2p": "p2p_proxy"}
SIRCL_FIELDS = {"version", "abi_version", "wheel", "native", "p2p", "receipt", "tuning_key", "vllm_pins"}
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


def validate_v3(value, profile):
    """A v3 lock after checking its v3 fields and, through ``v2_view``, its v2 fields for ``profile``."""
    _require(isinstance(value, dict) and set(value) == V3_FIELDS, f"Expected a complete {SCHEMA_V3} lock")
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


def checkpoint_problem(value, card):
    """Why the image of lock ``value`` cannot read the checkpoint of ``card``, or None."""
    key = f"{card['model_repository']}@{card['model_revision']}"
    builds = CHECKPOINT_BUILDS.get(key)
    if not builds:
        return None
    pins = list((sircl(value) or {}).get("vllm_pins") or ())
    if set(builds) & set(pins):
        return None
    return (f"Checkpoint {card['target_variant']} ({card['model_repository']} at {card['model_revision'][:12]}) "
            f"needs an image whose vLLM is the pinned build {' or '.join(builds)}; image {value.get('name')} "
            + (f"matches {', '.join(pins)}" if pins else "records no pinned vLLM build"))


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
    """The image lock (any schema) for one installer profile: ``explicit``, else the install default."""
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
        return installer_image.for_profile(profile, explicit)
    try:
        return validate(value, profile)
    except ValueError as error:
        if explicit is None or schema(value) not in (installer_image.SCHEMA, SCHEMA_V3) \
                or profile in value.get("profiles", ()):
            raise
        others = [row["name"] for row in catalog() if profile in profiles_of(row["lock"])]
        raise ValueError(f"{error}; images that run it: {', '.join(others) or 'none'}") from None
