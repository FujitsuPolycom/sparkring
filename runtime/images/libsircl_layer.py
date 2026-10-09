"""Add libsircl to a kraken-line installer image with a v3 lock, and record the derived image's v3 lock.

libsircl (``spark_transport/libsircl``, the vendored snapshot that
``scripts/sync_libsircl.py`` maintains) is SIRCL's NCCL-compatible C
library. The layer stacks on a parent with a ``sparkring-installer-image/v3``
lock, normally the SIRCL image, and adds:

- ``/opt/sparkring/libsircl/lib/libsircl.so.<version>``, built from the
  vendored tree by libsircl's own Makefile (``make -j BUILD=build``, then
  ``make check``) in a network-less container of the parent image at the
  fixed path ``/tmp/libsircl`` with ``LD_PRELOAD`` unset. The build embeds
  the prebuilt kernel packs, so it needs the image's gcc, make, Python 3 and
  rdma-core headers and no nvcc; the fixed path keeps the library's debug
  information, and so its bytes, independent of the build host;
- the notices every binary copy carries under ``/opt/sparkring/libsircl``;
- the vLLM general plugin ``libsircl``
  (``integrations/vllm/libsircl/sparkring_libsircl.py``) in the serving
  interpreter's site-packages, with a dist-info directory whose entry point
  registers it in ``vllm.general_plugins``. vLLM loads it only when a
  container's ``VLLM_PLUGINS`` names ``libsircl``;
- the layer receipt ``/opt/sparkring/receipts/libsircl-layer.json``
  (``sparkring-libsircl-layer/v1``).

Every added file is recorded in the image's external-base receipt, so the
image's ``verify`` checks its bytes.

Actions (none pushes or publishes an image):

- ``natives --parent-lock LOCK --output DIR``: check the vendored tree, build
  the library in the local parent image and save ``natives.json``.
- ``host-library --builder-image ID --output DIR``: the same build in another
  local image that has the build tools, for the stock-image option
  (``runtime/common/stock_image.py``), which mounts the library from the host.
- ``prepare --parent-lock LOCK --natives DIR --output CONTEXT``: write the
  Docker build context, reading the parent's two receipts from
  ``--base-receipt``/``--toolchain-receipt`` or the local parent image.
- ``record --context CONTEXT --image ID --name NAME --output LOCK``: probe the
  built image, check its layer as installation does, admit it for every
  profile of its lock and write its v3 lock.
- ``build --context CONTEXT --tag TAG --name NAME --output LOCK``: tag the
  parent, build, then record.

Status: research-only; no image built from it has been measured.
"""
import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runtime.common import image_lock, installer_image, libsircl  # noqa: E402
from runtime.images import derived_layer, sircl_layer  # noqa: E402
from scripts import sync_libsircl  # noqa: E402

TREE = sync_libsircl.TARGET
PLUGIN = ROOT / "integrations" / "vllm" / "libsircl" / "sparkring_libsircl.py"
NATIVES_SCHEMA = "sparkring-libsircl-natives/v1"
PLAN_SCHEMA = "sparkring-libsircl-layer-plan/v1"
PACKS = ("sircl_kernels", "sircl_fold", "sircl_links")
NCCL_API_VERSION = 22705
BUILD_PATH = "/tmp/libsircl"
SOURCE_MOUNT = "/libsircl-src"
OUTPUT_MOUNT = "/libsircl-out"
GENERATOR = "SparkRing runtime/images/libsircl_layer.py"
PROBE_MARK = "LIBSIRCL-LAYER-PROBE "
# Run in the built image: the plugin's entry point, the library's identity and the plugin's selection.
PROBE = r'''
import ctypes, importlib.metadata, json, os, sys
library, digest = sys.argv[1:3]
points = sorted([point.name, point.value] for point in importlib.metadata.entry_points(group="vllm.general_plugins")
                if point.name == "libsircl")
handle = ctypes.CDLL(library)
version = ctypes.c_int()
code = handle.ncclGetVersion(ctypes.byref(version))
handle.sirclGetInfo.restype = ctypes.c_char_p
info = json.loads(handle.sirclGetInfo().decode())
os.environ.update(SPARKRING_LIBSIRCL_LIBRARY=library, SPARKRING_LIBSIRCL_SHA256=digest)
import sparkring_libsircl
sparkring_libsircl.register()
print("LIBSIRCL-LAYER-PROBE " + json.dumps({"entry_points": points, "nccl_get_version": [code, version.value],
      "library": info.get("library"), "version": info.get("version"), "selected": os.environ.get("VLLM_NCCL_SO_PATH"),
      "plugin_file": sparkring_libsircl.__file__}))
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(data):
    return data.replace(b"\r\n", b"\n")


# The vendored tree.

def tree_facts(tree=TREE):
    """``(summary, {path: bytes})`` of the vendored tree after ``sync_libsircl.check``: every vendored file."""
    tree = Path(tree)
    summary = sync_libsircl.check(tree)
    entries = sync_libsircl.parse_manifest((tree / sync_libsircl.MANIFEST).read_bytes())
    files = {path: (tree / path).read_bytes() for path in sorted(entries) if sync_libsircl.excluded(path) is None}
    return summary, files


def kernel_packs(files):
    """``{pack: sha256}`` of the prebuilt kernel packs, each equal to its ``.sha256`` file."""
    packs = {}
    for name in PACKS:
        path = f"kernels/prebuilt/{name}.fatbin"
        require(path in files and path + ".sha256" in files, f"the tree lacks the kernel pack {path}")
        recorded = files[path + ".sha256"].decode().split()[0]
        require(digest(files[path]) == recorded, f"{path} differs from its .sha256 file")
        packs[name] = recorded
    return packs


def architectures(files):
    """The GPU architectures of the kernel packs, from the Makefile's ``NVCCFLAGS`` (``code=sm_<n>``)."""
    flags = re.search(r"^NVCCFLAGS\s*=(.*)$", files["Makefile"].decode(), re.M)
    require(flags is not None, "the tree's Makefile names no NVCCFLAGS")
    found = sorted(set(re.findall(r"code=(sm_[0-9]+)", flags.group(1))))
    require(found, "the tree's Makefile names no kernel pack architecture")
    return found


# The library build.

def build_script(version):
    """The shell script the build container runs; logs and the library land in the output mount."""
    name = f"libsircl.so.{version}"
    return "\n".join([
        "set -eu",
        f"rm -rf {BUILD_PATH}",
        f"cp -R {SOURCE_MOUNT} {BUILD_PATH}",
        f"cd {BUILD_PATH}",
        "unset LD_PRELOAD",
        f"make -j BUILD=build > {OUTPUT_MOUNT}/build.log 2>&1",
        f"make check BUILD=build > {OUTPUT_MOUNT}/check.log 2>&1",
        f"cp build/{name} {OUTPUT_MOUNT}/{name}",
        f"gcc --version | head -n 1 > {OUTPUT_MOUNT}/compiler.txt",
    ]) + "\n"


def _run(command, text=True):
    return subprocess.run(command, capture_output=True, check=True, text=text)


def build_natives(image_id, output, *, tree=TREE, run=_run, role="parent_image_id"):
    """Build libsircl from the vendored tree in a network-less container of local image ``image_id``."""
    output = Path(output).resolve()
    require(not output.exists(), "The natives directory must not exist")
    summary, files = tree_facts(tree)
    packs, arches = kernel_packs(files), architectures(files)
    version = summary["version"]
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as source:
        for path, data in files.items():
            target = Path(source) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        command = ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "/bin/sh",
                   "--mount", f"type=bind,src={source},dst={SOURCE_MOUNT},readonly",
                   "--mount", f"type=bind,src={output},dst={OUTPUT_MOUNT}"]
        if hasattr(os, "getuid"):
            # The output stays the invoking account's, not root's.
            command += ["--user", f"{os.getuid()}:{os.getgid()}"]
        run([*command, image_id, "-c", build_script(version)])
    name = f"libsircl.so.{version}"
    library = (output / name).read_bytes()
    check = (output / "check.log").read_bytes()
    result = {"schema": NATIVES_SCHEMA, role: image_id, "snapshot": summary["tree_digest"], "version": version,
              "library": {"name": name, "sha256": digest(library)},
              "compiler": (output / "compiler.txt").read_text(encoding="utf-8").strip(),
              "kernel_packs": packs, "architectures": arches,
              "build": {"commands": ["make -j BUILD=build", "make check BUILD=build"], "path": BUILD_PATH,
                        "check_log_sha256": digest(check)}}
    (output / "natives.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def read_natives(natives, *, tree=TREE, role="parent_image_id", image_id=None):
    """``(record, library bytes)`` of a natives directory built from this checkout's vendored tree."""
    natives = Path(natives)
    record = json.loads((natives / "natives.json").read_text(encoding="utf-8"))
    summary, files = tree_facts(tree)
    require(record.get("schema") == NATIVES_SCHEMA and record.get("snapshot") == summary["tree_digest"]
            and record.get("version") == summary["version"],
            "natives.json was built from another libsircl snapshot than this checkout's")
    require(image_id is None or record.get(role) == image_id, "natives.json was built in another image")
    require(record.get("kernel_packs") == kernel_packs(files), "natives.json names other kernel packs")
    data = (natives / record["library"]["name"]).read_bytes()
    require(digest(data) == record["library"]["sha256"], "The library differs from natives.json")
    return record, data


# The layer.

def plugin_files(site, version):
    """``{image path: bytes}`` of the plugin module and its dist-info directory in ``site``."""
    module = _text(PLUGIN.read_bytes())
    info = f"{libsircl.PLUGIN_MODULE}-{version}.dist-info"
    group, name, value = libsircl.PLUGIN_ENTRY_POINT
    files = {
        f"{libsircl.PLUGIN_MODULE}.py": module,
        f"{info}/METADATA": (f"Metadata-Version: 2.4\nName: sparkring-libsircl\nVersion: {version}\n"
                             "Summary: vLLM general plugin libsircl: vLLM's PyNccl loads libsircl\n"
                             "License-Expression: Apache-2.0\nRequires-Python: >=3.11\n").encode(),
        f"{info}/WHEEL": (f"Wheel-Version: 1.0\nGenerator: {GENERATOR}\nRoot-Is-Purelib: true\n"
                          "Tag: py3-none-any\n").encode(),
        f"{info}/entry_points.txt": f"[{group}]\n{name} = {value}\n".encode(),
        f"{info}/top_level.txt": f"{libsircl.PLUGIN_MODULE}\n".encode(),
    }
    rows = []
    for path in sorted(files):
        hashed = base64.urlsafe_b64encode(hashlib.sha256(files[path]).digest()).rstrip(b"=").decode()
        rows.append(f"{path},sha256={hashed},{len(files[path])}")
    rows.append(f"{info}/RECORD,,")
    files[f"{info}/RECORD"] = ("\n".join(rows) + "\n").encode()
    return {site + path: data for path, data in files.items()}


def layer_files(lock, natives, site, *, tree=TREE):
    """``({image path: bytes}, layer receipt)``: the library, the notices, the plugin and the receipt."""
    record, library = read_natives(natives, tree=tree, image_id=lock["image_id"])
    summary, files = tree_facts(tree)
    version = summary["version"]
    payloads = {libsircl.library_path(version): library}
    for name in libsircl.NOTICES:
        require(name in files, f"the vendored tree lacks the notice {name}")
        payloads[f"{libsircl.INSTALL_ROOT}/{name}"] = files[name]
    payloads.update(plugin_files(site, version))
    plugin = site + libsircl.PLUGIN_MODULE + ".py"
    receipt = {"schema": libsircl.LAYER_SCHEMA, "version": version, "snapshot": summary["tree_digest"],
               "library": {"path": libsircl.library_path(version), "sha256": digest(library),
                           "soname": libsircl.SONAME},
               "nccl_api_version": NCCL_API_VERSION, "kernel_packs": record["kernel_packs"],
               "architectures": record["architectures"],
               "plugin": {"name": libsircl.PLUGIN_NAME, "path": plugin, "sha256": digest(payloads[plugin])},
               "compiler": record["compiler"], "build": record["build"], "parent_image_id": lock["image_id"],
               "site_packages": site, "files": {path: digest(data) for path, data in sorted(payloads.items())}}
    payloads[image_lock.LIBSIRCL_RECEIPT] = derived_layer.canonical_json(receipt)
    return payloads, receipt


def load_lock(path):
    """A parent v3 lock after validating it for every profile it lists; it must not carry libsircl yet."""
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    require(image_lock.schema(lock) == image_lock.SCHEMA_V3, "The libsircl layer derives from a v3 image lock")
    for profile in image_lock.profiles_of(lock):
        image_lock.validate(lock, profile)
    require(derived_layer.IMAGE_ID.fullmatch(lock["image_id"]), "The parent lock must name a local image ID")
    require("libsircl" not in lock["transports"], "The parent image already carries a libsircl layer")
    return lock


def prepare(lock, read, natives, output, *, tree=TREE):
    """Write the build context: the layer's files and the re-recorded external-base and toolchain receipts."""
    output = Path(output).resolve()
    require(not output.exists(), "Build context must not exist")
    require(image_lock.schema(lock) == image_lock.SCHEMA_V3 and "libsircl" not in lock["transports"],
            "The libsircl layer derives from a v3 image lock without a libsircl layer")
    base_raw, toolchain_raw = read(derived_layer.BASE_RECEIPT), read(derived_layer.TOOLCHAIN_RECEIPT)
    require(digest(base_raw) == lock["parent_receipt_sha256"], "External-base receipt differs from the parent lock")
    require(digest(toolchain_raw) == lock["toolchain_receipt_sha256"], "Toolchain receipt differs from the parent lock")
    base, toolchain = json.loads(base_raw), json.loads(toolchain_raw)
    require(toolchain.get("parent_receipt_sha256") == digest(base_raw),
            "Toolchain receipt does not record the external-base receipt")
    site = sircl_layer.site_packages(base)
    payloads, receipt = layer_files(lock, natives, site, tree=tree)
    recorded = sorted(path for path in payloads if path in base["files"])
    require(not recorded, "The parent image already records " + ", ".join(recorded[:5]))
    derived = copy.deepcopy(base)
    for path, data in payloads.items():
        derived["files"][path] = digest(data)
    derived.setdefault("capabilities", {})["libsircl"] = {
        "version": receipt["version"], "snapshot": receipt["snapshot"], "receipt": image_lock.LIBSIRCL_RECEIPT,
        "receipt_sha256": digest(payloads[image_lock.LIBSIRCL_RECEIPT])}
    base_out = derived_layer.canonical_json(derived)
    toolchain["parent_receipt_sha256"] = digest(base_out)
    toolchain_out = derived_layer.canonical_json(toolchain, sort_keys=False)
    written = {**payloads, derived_layer.BASE_RECEIPT: base_out, derived_layer.TOOLCHAIN_RECEIPT: toolchain_out}
    output.mkdir(parents=True)
    for target, raw in written.items():
        path = output / "files" / target.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (output / "Dockerfile").write_text(derived_layer.dockerfile())
    plan = {"schema": PLAN_SCHEMA, "parent_lock": lock, "added": sorted(payloads),
            "receipts": {derived_layer.BASE_RECEIPT: digest(base_out),
                         derived_layer.TOOLCHAIN_RECEIPT: digest(toolchain_out)},
            "layer": receipt, "layer_sha256": digest(payloads[image_lock.LIBSIRCL_RECEIPT]),
            "payload_bytes": sum(len(raw) for raw in written.values())}
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return {"context": str(output), "files": len(payloads), "receipts": plan["receipts"],
            "library": receipt["library"],
            "build": ["docker", "build", "--build-arg", "PARENT_IMAGE=" + derived_layer.parent_tag(lock["image_id"]),
                      "-t", "<tag>", str(output)]}


def libsircl_block(plan):
    layer = plan["layer"]
    return {"version": layer["version"], "snapshot": layer["snapshot"],
            "library": {"path": layer["library"]["path"], "sha256": layer["library"]["sha256"]},
            "nccl_api_version": layer["nccl_api_version"], "plugin": dict(layer["plugin"]),
            "receipt": {"path": image_lock.LIBSIRCL_RECEIPT, "sha256": plan["layer_sha256"]}}


def v3_lock(plan, image, name, profiles=None):
    """The v3 lock of the built image: the parent's v3 lock bound to the image, with the libsircl layer."""
    lock = copy.deepcopy(plan["parent_lock"])
    lock.update(name=name, image_id=image["Id"], image_reference=image["Id"], image_bytes=image["Size"],
                download_bytes=lock["download_bytes"] + plan["payload_bytes"],
                parent_receipt_sha256=plan["receipts"][derived_layer.BASE_RECEIPT],
                toolchain_receipt_sha256=plan["receipts"][derived_layer.TOOLCHAIN_RECEIPT],
                transports=sorted({*lock["transports"], "libsircl"}), libsircl=libsircl_block(plan),
                archived=False)
    if profiles:
        lock["profiles"] = sorted(set(profiles))
    for profile in image_lock.profiles_of(lock):
        image_lock.validate(lock, profile)
    return lock


def parse_probe(text):
    lines = [line[len(PROBE_MARK):] for line in text.splitlines() if line.startswith(PROBE_MARK)]
    require(len(lines) == 1, "The libsircl probe printed no record")
    return json.loads(lines[0])


def probe_problems(record, block):
    """What the built image's probe found wrong, given the lock's ``libsircl`` block."""
    group, name, value = libsircl.PLUGIN_ENTRY_POINT
    expected = {"entry_points": [[name, value]], "nccl_get_version": [0, block["nccl_api_version"]],
                "library": "libsircl", "version": block["version"], "selected": block["library"]["path"],
                "plugin_file": block["plugin"]["path"]}
    return [f"{key}: {record.get(key)!r}, expected {wanted!r}" for key, wanted in expected.items()
            if record.get(key) != wanted]


def probe_image(image_id, block, *, run=_run):
    command = ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--read-only", "--entrypoint",
               "python3", image_id, "-I", "-c", PROBE, block["library"]["path"], block["library"]["sha256"]]
    return parse_probe(run(command).stdout)


def record(context, image_id, name, output, *, run=_run, profiles=None):
    """Probe the built image, check its layer, admit it for every profile of its lock and write its v3 lock."""
    plan = json.loads((Path(context) / "plan.json").read_text(encoding="utf-8"))
    output = Path(output)
    require(not output.exists(), "Output lock must not exist")
    parent = plan["parent_lock"]["image_id"]
    for target in plan["added"]:
        try:
            run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "/bin/sh",
                 parent, "-c", 'test ! -e "$1" && test ! -L "$1"', "sh", target])
        except subprocess.CalledProcessError as error:
            raise ValueError("The parent image already has an added path: " + target) from error
    image = json.loads(run(["docker", "image", "inspect", image_id]).stdout)[0]
    lock = v3_lock(plan, image, name, profiles)
    problems = probe_problems(probe_image(image["Id"], lock["libsircl"], run=run), lock["libsircl"])
    require(not problems, "The built image's libsircl probe found: " + "; ".join(problems))
    libsircl.check_layer(lock["image_id"], lock["parent_receipt_sha256"], lock["libsircl"], run=run)
    view = image_lock.v2_view(lock)
    for profile in image_lock.profiles_of(lock):
        installer_image.admit(view, run=run, profile=profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return {"lock": str(output), "image_id": lock["image_id"], "libsircl": lock["libsircl"]["version"],
            "nccl_api_version": lock["libsircl"]["nccl_api_version"], "library": lock["libsircl"]["library"],
            "transports": lock["transports"], "serving_qualified": False}


def build(context, tag, name, output, *, run=_run, profiles=None):
    plan = json.loads((Path(context) / "plan.json").read_text(encoding="utf-8"))
    parent = plan["parent_lock"]["image_id"]
    run(["docker", "tag", parent, derived_layer.parent_tag(parent)])
    run(["docker", "build", "-q", "--build-arg", "PARENT_IMAGE=" + derived_layer.parent_tag(parent), "-t", tag,
         str(context)])
    image_id = json.loads(run(["docker", "image", "inspect", tag]).stdout)[0]["Id"]
    return record(context, image_id, name, output, run=run, profiles=profiles)


def host_library(builder_image, output, *, tree=TREE, run=_run):
    """Build the library a stock image mounts from the host, in local image ``builder_image``.

    The result's ``install`` names the host path every Spark of a stock
    deployment holds it at (``libsircl.host_library_path``).
    """
    result = build_natives(builder_image, output, tree=tree, run=run, role="builder_image_id")
    path = libsircl.host_library_path(result["library"]["sha256"], result["version"])
    result["install"] = {"path": path, "commands": [
        f"sudo install -D -m 0444 {shlex.quote(str(Path(output) / result['library']['name']))} {path}"]}
    (Path(output) / "natives.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    actions = parser.add_subparsers(dest="action", required=True)
    natives = actions.add_parser("natives", help="build the library in the local parent image")
    natives.add_argument("--parent-lock", required=True, type=Path)
    natives.add_argument("--output", required=True, type=Path)
    host = actions.add_parser("host-library", help="build the library a stock image mounts, in a builder image")
    host.add_argument("--builder-image", required=True, help="a local image with gcc, make, Python 3 and rdma-core "
                                                             "headers, such as an installer image")
    host.add_argument("--output", required=True, type=Path)
    prepared = actions.add_parser("prepare", help="write the build context; does not build")
    prepared.add_argument("--parent-lock", required=True, type=Path)
    prepared.add_argument("--natives", required=True, type=Path)
    prepared.add_argument("--base-receipt", type=Path, help="the parent's " + derived_layer.BASE_RECEIPT)
    prepared.add_argument("--toolchain-receipt", type=Path, help="the parent's " + derived_layer.TOOLCHAIN_RECEIPT)
    prepared.add_argument("--output", required=True, type=Path)
    recorded = actions.add_parser("record", help="probe and admit a built image and write its v3 lock")
    recorded.add_argument("--image", required=True)
    built = actions.add_parser("build", help="build a prepared context, then record it")
    built.add_argument("--tag", required=True)
    for action in (recorded, built):
        action.add_argument("--context", required=True, type=Path)
        action.add_argument("--name", required=True, help="release name of the v3 lock")
        action.add_argument("--output", required=True, type=Path)
        action.add_argument("--profiles", help="comma-separated profiles of the v3 lock; default: the parent lock's")
    args = parser.parse_args(argv)
    try:
        if args.action == "natives":
            lock = load_lock(args.parent_lock)
            result = build_natives(lock["image_id"], args.output)
        elif args.action == "host-library":
            result = host_library(args.builder_image, args.output)
        elif args.action == "prepare":
            lock = load_lock(args.parent_lock)
            require(bool(args.base_receipt) == bool(args.toolchain_receipt), "Supply both copied receipts or neither")
            read = (derived_layer.receipt_reader(args.base_receipt, args.toolchain_receipt)
                    if args.base_receipt else derived_layer.docker_reader(lock["image_id"]))
            result = prepare(lock, read, args.natives, args.output)
        else:
            profiles = args.profiles.split(",") if args.profiles else None
            if args.action == "record":
                result = record(args.context, args.image, args.name, args.output, profiles=profiles)
            else:
                result = build(args.context, args.tag, args.name, args.output, profiles=profiles)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError, sync_libsircl.SnapshotError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
