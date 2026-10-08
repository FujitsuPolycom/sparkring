"""Add SIRCL ring sessions to a kraken-line installer image and record its v3 lock.

The layer adds to a parent image whose lock carries the prepared transport
(``sparkring-installer-image/v2``):

- the SIRCL package installed in the serving interpreter's site-packages, from
  a reproducible wheel ``sparkring_sircl-<version>-py3-none-any.whl``
  (``wheel``): every Python, C, header and JSON file of
  ``spark_transport/sircl/sparkring_sircl``, SparkRing's RoCE GID resolver
  ``integrations/vllm/spark_roce_gid.py`` as the top-level module
  ``spark_roce_gid`` (which ``sparkring_sircl.roce_gid`` imports), and a
  dist-info directory whose entry points register the ``sircl`` platform and
  general plugins. The plugins stay inert until a container's
  ``VLLM_PLUGINS`` names ``sircl``, so deployments on the prepared transport
  are unchanged;
- the two native libraries, ``roce_proxy-<digest>.so`` and
  ``p2p_proxy-<digest>.so`` under ``/opt/sparkring/sircl/lib``, compiled by
  SIRCL's own build code (``sparkring_sircl.vllm.serve.probe --build``) in a
  network-less container of the parent image, so they link against that
  image's glibc and ``libibverbs.so.1`` and no Spark compiles them;
- the layer receipt ``/opt/sparkring/receipts/sircl-layer.json``
  (``sparkring-sircl-layer/v1``): version, ABI, wheel, libraries, tuning key,
  the compiler and every installed file with its SHA-256.

Every added file is recorded in the image's external-base receipt, so the
image's own ``verify``, which installer admission runs, checks its bytes.

Actions (none pushes or publishes an image):

- ``wheel --output DIR``: write the wheel; offline.
- ``natives --parent-lock LOCK --wheel FILE --output DIR``: build both
  libraries in the local parent image and save ``natives.json``.
- ``prepare --parent-lock LOCK --wheel FILE --natives DIR --output CONTEXT``:
  write the Docker build context, reading the parent's two receipts from
  ``--base-receipt``/``--toolchain-receipt`` or the local parent image.
- ``record --context CONTEXT --image ID --name NAME --output LOCK``: admit the
  built image for every profile of the parent lock, probe it, and write its
  v3 lock.
- ``build --context CONTEXT --tag TAG --name NAME --output LOCK``: tag the
  parent, build, then record.
"""

import argparse
import base64
import copy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runtime.common import image_lock, installer_image, transport  # noqa: E402
from runtime.images import derived_layer  # noqa: E402

PROJECT = ROOT / "spark_transport" / "sircl"
PACKAGE = PROJECT / "sparkring_sircl"
RESOLVER = ROOT / "integrations" / "vllm" / "spark_roce_gid.py"
LAYER_SCHEMA = "sparkring-sircl-layer/v1"
PLAN_SCHEMA = "sparkring-sircl-layer-plan/v1"
SUFFIXES = (".py", ".c", ".h", ".json")
# A fixed timestamp keeps the wheel's bytes a function of its contents.
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
GENERATOR = "SparkRing runtime/images/sircl_layer.py"
BUILD_SOURCE = "/sircl-wheel"
BUILD_OUTPUT = "/sircl-out"
PROBE = "sparkring_sircl.vllm.serve.probe"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


# The wheel.

def project():
    return tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def version():
    value = project()["version"]
    from spark_transport.sircl.sparkring_sircl import __version__
    require(value == __version__, f"pyproject.toml names SIRCL {value}, the package {__version__}")
    return value


def _text(data):
    """Source bytes with CRLF line ends as LF, so the wheel is the same on every platform."""
    return data.replace(b"\r\n", b"\n")


def package_files():
    """``{wheel path: bytes}`` of the package and the resolver."""
    files = {}
    for path in sorted(PACKAGE.rglob("*")):
        relative = path.relative_to(PROJECT)
        if not path.is_file() or "__pycache__" in relative.parts or path.suffix not in SUFFIXES:
            continue
        files[relative.as_posix()] = _text(path.read_bytes())
    files[RESOLVER.name] = _text(RESOLVER.read_bytes())
    return files


def _entry_points(metadata):
    sections = {"console_scripts": metadata.get("scripts", {}), **metadata.get("entry-points", {})}
    blocks = []
    for group in sorted(sections):
        blocks.append("\n".join([f"[{group}]"] + [f"{name} = {target}" for name, target in sorted(sections[group].items())]))
    return ("\n\n".join(blocks) + "\n").encode()


def _metadata(metadata):
    lines = ["Metadata-Version: 2.4", f"Name: {metadata['name']}", f"Version: {metadata['version']}",
             f"Summary: {metadata['description']}", f"License-Expression: {metadata['license']}",
             f"Requires-Python: {metadata['requires-python']}"]
    for name in metadata.get("license-files", []):
        lines.append(f"License-File: {name}")
    for extra, requirements in sorted(metadata.get("optional-dependencies", {}).items()):
        lines.append(f"Provides-Extra: {extra}")
        lines += [f"Requires-Dist: {requirement}; extra == \"{extra}\"" for requirement in requirements]
    return ("\n".join(lines) + "\n").encode()


def wheel_contents():
    """``(name, {path: bytes})`` of the wheel, RECORD included."""
    metadata = project()
    name = f"sparkring_sircl-{version()}-py3-none-any.whl"
    info = f"sparkring_sircl-{metadata['version']}.dist-info"
    files = package_files()
    files[f"{info}/METADATA"] = _metadata(metadata)
    files[f"{info}/WHEEL"] = (f"Wheel-Version: 1.0\nGenerator: {GENERATOR}\nRoot-Is-Purelib: true\n"
                              "Tag: py3-none-any\n").encode()
    files[f"{info}/entry_points.txt"] = _entry_points(metadata)
    files[f"{info}/top_level.txt"] = b"spark_roce_gid\nsparkring_sircl\n"
    for license_file in metadata.get("license-files", []):
        files[f"{info}/licenses/{license_file}"] = _text((PROJECT / license_file).read_bytes())
    rows = []
    for path in sorted(files):
        hashed = base64.urlsafe_b64encode(hashlib.sha256(files[path]).digest()).rstrip(b"=").decode()
        rows.append(f"{path},sha256={hashed},{len(files[path])}")
    rows.append(f"{info}/RECORD,,")
    files[f"{info}/RECORD"] = ("\n".join(rows) + "\n").encode()
    return name, files


def wheel_bytes():
    """``(name, bytes)``: an uncompressed zip with sorted entries and fixed timestamps."""
    name, files = wheel_contents()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(files):
            entry = zipfile.ZipInfo(path, date_time=ZIP_TIME)
            entry.external_attr = 0o644 << 16
            entry.create_system = 3
            archive.writestr(entry, files[path])
    return name, buffer.getvalue()


def write_wheel(output):
    name, data = wheel_bytes()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / name).write_bytes(data)
    return {"wheel": str(output / name), "name": name, "sha256": digest(data), "files": len(wheel_contents()[1])}


def read_wheel(path):
    """``(name, sha256, {path: bytes})`` of a wheel this module wrote, checked against its RECORD."""
    path = Path(path)
    data = path.read_bytes()
    expected_name, expected = wheel_bytes()
    require(path.name == expected_name and data == expected,
            f"{path.name} is not the wheel this checkout writes ({expected_name}); run the wheel action again")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        files = {entry.filename: archive.read(entry) for entry in archive.infolist()}
    return path.name, digest(data), files


# The native libraries.

def _run(command, text=True):
    return subprocess.run(command, capture_output=True, check=True, text=text)


def build_natives(lock, wheel, output, *, run=_run):
    """Build both native libraries in the local parent image with SIRCL's own build code; save ``natives.json``."""
    output = Path(output).resolve()
    require(not output.exists(), "The natives directory must not exist")
    name, sha256, files = read_wheel(wheel)
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as unpacked:
        for path, data in files.items():
            target = Path(unpacked) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        command = ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "python3",
                   "--mount", f"type=bind,src={unpacked},dst={BUILD_SOURCE},readonly",
                   "--mount", f"type=bind,src={output},dst={BUILD_OUTPUT}",
                   "--env", f"PYTHONPATH={BUILD_SOURCE}", "--env", f"SIRCL_BUILD_CACHE_DIR={BUILD_OUTPUT}",
                   lock["image_id"], "-m", PROBE, "--build"]
        record = parse_probe(run(command).stdout)
        compiler = run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "gcc",
                        lock["image_id"], "--version"]).stdout.splitlines()[0]
    libraries = {}
    for kind, key in (("native", "library"), ("p2p", "p2p_library")):
        entry = record.get(key) or {}
        require(entry.get("exists") and not entry.get("error"), f"The {kind} library was not built: {entry}")
        built = output / PurePosixPath(entry["path"]).name
        require(built.is_file(), f"The {kind} library {built.name} is not in {output}")
        stem = image_lock.LIBRARY_STEMS[kind]
        source = built.name.removeprefix(stem + "-").removesuffix(".so")
        libraries[kind] = {"path": f"{image_lock.LIBRARY_DIRECTORY}/{built.name}", "sha256": digest(built.read_bytes()),
                           "source_digest": source}
    result = {"schema": "sparkring-sircl-natives/v1", "parent_image_id": lock["image_id"], "wheel": name,
              "wheel_sha256": sha256, "compiler": compiler, **libraries, "probe": record}
    (output / "natives.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_probe(text):
    from spark_transport.sircl.sparkring_sircl.vllm.serve import probe
    record = probe.parse(text)
    require(record is not None, "The SIRCL probe printed no record")
    return record


# The layer.

def tuning_key():
    from spark_transport.sircl.sparkring_sircl import tuning
    from spark_transport.sircl.sparkring_sircl.oneshot._proxy import ABI_VERSION
    return {"native": tuning.native_hash(PACKAGE), "kernels": tuning.kernels_hash(PACKAGE),
            "sircl": f"{version()}/abi{ABI_VERSION}"}, ABI_VERSION


def site_packages(base):
    """The serving interpreter's site-packages directory: where the parent receipt records ``vllm/__init__.py``."""
    found = sorted({path[:-len("vllm/__init__.py")] for path in base["files"] if path.endswith("/vllm/__init__.py")})
    require(len(found) == 1, f"The parent receipt records vllm/__init__.py in {len(found)} directories, not one")
    return found[0]


def layer_files(lock, wheel, natives, site):
    """``({image path: bytes}, layer receipt)`` of the layer: the wheel installed in ``site``, both libraries and
    the receipt."""
    name, sha256, files = read_wheel(wheel)
    record = json.loads((Path(natives) / "natives.json").read_text(encoding="utf-8"))
    require(record.get("schema") == "sparkring-sircl-natives/v1" and record["parent_image_id"] == lock["image_id"]
            and record["wheel_sha256"] == sha256,
            "natives.json was built from another parent image or wheel")
    key, abi = tuning_key()
    require(record["native"]["source_digest"] == key["native"],
            "The native library was built from another source than this checkout's")
    payloads = {site + path: data for path, data in files.items()}
    for kind in image_lock.LIBRARY_STEMS:
        library = record[kind]
        data = (Path(natives) / PurePosixPath(library["path"]).name).read_bytes()
        require(digest(data) == library["sha256"], f"The {kind} library differs from natives.json")
        payloads[library["path"]] = data
    receipt = {"schema": "sparkring-sircl-layer/v1", "version": version(), "abi_version": abi,
               "wheel": {"name": name, "sha256": sha256},
               "native": dict(record["native"]), "p2p": dict(record["p2p"]), "tuning_key": key,
               "compiler": record["compiler"], "parent_image_id": lock["image_id"], "site_packages": site,
               "files": {path: digest(data) for path, data in sorted(payloads.items())}}
    payloads[image_lock.LAYER_RECEIPT] = derived_layer.canonical_json(receipt)
    return payloads, receipt


def prepare(lock, read, wheel, natives, output):
    """Write the build context: the layer's files and the re-recorded external-base and toolchain receipts."""
    output = Path(output).resolve()
    require(not output.exists(), "Build context must not exist")
    require(lock["schema"] == installer_image.SCHEMA, "The SIRCL layer derives from a v2 kraken-line image lock")
    base_raw, toolchain_raw = read(derived_layer.BASE_RECEIPT), read(derived_layer.TOOLCHAIN_RECEIPT)
    require(digest(base_raw) == lock["parent_receipt_sha256"], "External-base receipt differs from the parent lock")
    require(digest(toolchain_raw) == lock["toolchain_receipt_sha256"], "Toolchain receipt differs from the parent lock")
    base, toolchain = json.loads(base_raw), json.loads(toolchain_raw)
    require(toolchain.get("parent_receipt_sha256") == digest(base_raw),
            "Toolchain receipt does not record the external-base receipt")
    site = site_packages(base)
    payloads, receipt = layer_files(lock, wheel, natives, site)
    recorded = sorted(path for path in payloads if path in base["files"])
    require(not recorded, "The parent image already records " + ", ".join(recorded[:5]))
    derived = copy.deepcopy(base)
    for path, data in payloads.items():
        derived["files"][path] = digest(data)
    derived.setdefault("capabilities", {})["sircl"] = {
        "version": receipt["version"], "abi_version": receipt["abi_version"],
        "receipt": image_lock.LAYER_RECEIPT, "receipt_sha256": digest(payloads[image_lock.LAYER_RECEIPT])}
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
            "layer": receipt, "layer_sha256": digest(payloads[image_lock.LAYER_RECEIPT]),
            "payload_bytes": sum(len(raw) for raw in written.values()),
            "tuning_defaults_sha256": transport.tuning_digest(transport.load_tuning())}
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return {"context": str(output), "files": len(payloads), "receipts": plan["receipts"],
            "build": ["docker", "build", "--build-arg", "PARENT_IMAGE=" + derived_layer.parent_tag(lock["image_id"]),
                      "-t", "<tag>", str(output)]}


def sircl_block(plan, probe_record):
    layer = plan["layer"]
    return {"version": layer["version"], "abi_version": layer["abi_version"], "wheel": layer["wheel"],
            "native": layer["native"], "p2p": layer["p2p"],
            "receipt": {"path": image_lock.LAYER_RECEIPT, "sha256": plan["layer_sha256"]},
            "tuning_key": layer["tuning_key"],
            "vllm_pins": sorted(set((probe_record.get("vllm") or {}).get("matches") or []))}


def v3_lock(plan, image, name, probe_record):
    """The v3 lock of the built image: the parent's v2 contract bound to the image, with the SIRCL layer."""
    lock = {key: value for key, value in plan["parent_lock"].items() if key != "schema"}
    lock.update(schema=image_lock.SCHEMA_V3, name=name, image_id=image["Id"], image_reference=image["Id"],
                image_bytes=image["Size"], download_bytes=lock["download_bytes"] + plan["payload_bytes"],
                parent_receipt_sha256=plan["receipts"][derived_layer.BASE_RECEIPT],
                toolchain_receipt_sha256=plan["receipts"][derived_layer.TOOLCHAIN_RECEIPT],
                line="kraken", transports=["prepared", "sircl"], sircl=sircl_block(plan, probe_record),
                tuning_defaults_sha256=plan["tuning_defaults_sha256"], archived=False)
    for profile in image_lock.profiles_of(lock):
        image_lock.validate(lock, profile)
    return lock


def probe_image(image_id, *, run=_run):
    """The SIRCL probe in the built image: package and entry points from site-packages, prebuilt libraries."""
    command = ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--read-only", "--entrypoint",
               "python3", "--env", f"SIRCL_BUILD_CACHE_DIR={image_lock.LIBRARY_DIRECTORY}", image_id, "-m", PROBE]
    return parse_probe(run(command).stdout)


def record(context, image_id, name, output, *, run=_run):
    """Admit the built image for every profile of its parent lock, probe SIRCL in it, and write its v3 lock."""
    from spark_transport.sircl.sparkring_sircl.vllm.serve import probe
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
    probed = probe_image(image["Id"], run=run)
    blockers, _ = probe.evaluate(probed, staged_root=plan["layer"]["site_packages"].rstrip("/"),
                                 library=PurePosixPath(plan["layer"]["native"]["path"]).name)
    require(not blockers, "The built image's SIRCL probe found: " + "; ".join(blockers))
    lock = v3_lock(plan, image, name, probed)
    view = image_lock.v2_view(lock)
    for profile in image_lock.profiles_of(lock):
        installer_image.admit(view, run=run, profile=profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return {"lock": str(output), "image_id": lock["image_id"], "sircl": lock["sircl"]["version"],
            "vllm_pins": lock["sircl"]["vllm_pins"], "serving_qualified": False}


def build(context, tag, name, output, *, run=_run):
    plan = json.loads((Path(context) / "plan.json").read_text(encoding="utf-8"))
    parent = plan["parent_lock"]["image_id"]
    run(["docker", "tag", parent, derived_layer.parent_tag(parent)])
    run(["docker", "build", "-q", "--build-arg", "PARENT_IMAGE=" + derived_layer.parent_tag(parent), "-t", tag,
         str(context)])
    image_id = json.loads(run(["docker", "image", "inspect", tag]).stdout)[0]["Id"]
    return record(context, image_id, name, output, run=run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    actions = parser.add_subparsers(dest="action", required=True)
    wheel = actions.add_parser("wheel", help="write the SIRCL wheel; offline")
    wheel.add_argument("--output", required=True, type=Path)
    natives = actions.add_parser("natives", help="build both native libraries in the local parent image")
    prepared = actions.add_parser("prepare", help="write the build context; does not build")
    for action in (natives, prepared):
        action.add_argument("--parent-lock", required=True, type=Path)
        action.add_argument("--wheel", required=True, type=Path)
        action.add_argument("--output", required=True, type=Path)
    prepared.add_argument("--natives", required=True, type=Path)
    prepared.add_argument("--base-receipt", type=Path, help="the parent's " + derived_layer.BASE_RECEIPT)
    prepared.add_argument("--toolchain-receipt", type=Path, help="the parent's " + derived_layer.TOOLCHAIN_RECEIPT)
    recorded = actions.add_parser("record", help="admit a built image and write its v3 lock")
    recorded.add_argument("--image", required=True)
    built = actions.add_parser("build", help="build a prepared context, then record it")
    built.add_argument("--tag", required=True)
    for action in (recorded, built):
        action.add_argument("--context", required=True, type=Path)
        action.add_argument("--name", required=True, help="release name of the v3 lock")
        action.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "wheel":
            result = write_wheel(args.output)
        elif args.action in ("natives", "prepare"):
            lock = derived_layer.load_lock(args.parent_lock)
            if args.action == "natives":
                result = build_natives(lock, args.wheel, args.output)
            else:
                require(bool(args.base_receipt) == bool(args.toolchain_receipt), "Supply both copied receipts or neither")
                read = (derived_layer.receipt_reader(args.base_receipt, args.toolchain_receipt)
                        if args.base_receipt else derived_layer.docker_reader(lock["image_id"]))
                result = prepare(lock, read, args.wheel, args.natives, args.output)
        elif args.action == "record":
            result = record(args.context, args.image, args.name, args.output)
        else:
            result = build(args.context, args.tag, args.name, args.output)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
