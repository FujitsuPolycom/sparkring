"""libsircl, SIRCL's NCCL-compatible C library, as an image layer and as the installer transport ``libsircl``.

docs/architecture/libsircl.md describes the design. This module holds:

- the facts of the libsircl image layer (``runtime/images/libsircl_layer.py``)
  that installation checks, and ``check_layer``, which verifies an image's
  layer against the ``libsircl`` block of its v3 lock
  (``runtime/common/image_lock.py``);
- ``host_library_path``, where a stock-image deployment's Sparks hold the
  host build of libsircl.

Status: research-only.
"""
import hashlib
import json

from runtime.common import image_lock

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
# Where every Spark of a stock-image deployment holds the host build of libsircl, by its SHA-256.
HOST_LIBRARY_ROOT = "/var/lib/sparkring/libsircl"


class LibsirclError(ValueError):
    """libsircl cannot run here, or one of its inputs is malformed."""


def _require(condition, text):
    if not condition:
        raise LibsirclError(text)


def library_path(version):
    return f"{image_lock.LIBSIRCL_LIBRARY_DIRECTORY}/libsircl.so.{version}"


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
