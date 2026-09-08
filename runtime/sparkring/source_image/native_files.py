"""Install only the source-bound native artifacts declared by the source lock."""
import hashlib
import io
import json
from pathlib import PurePosixPath
import re
import tarfile

from archive_utils import inventory, read_archive, sha, under

ARCHIVE = "native-files.tar"
METADATA_ROOT = "/opt/sparkring/native-files"
LIBRARIES = {
    "native/libnccl.so.2.30.7": ("nccl", "/opt/sparkring/nccl-pci/libnccl.so.2.30.7"),
    "native/libspark_cache_snapshot.so": ("snapshot", "/opt/sparkcache-native/libspark_cache_snapshot.so"),
}
MEMBERS = set(LIBRARIES) | {
    "licenses/nccl/LICENSE.txt", "licenses/nccl/ThirdPartyNotices.txt",
    "licenses/sparkcache/LICENSE", "provenance.json",
}
PIN_FIELDS = {"schema", "sha256", "bytes", "members", "nccl_tree", "nccl_archive_sha256",
              "sparkcache_native_tree", "sparkcache_native_file_map_sha256", "url"}


def mode(manifest):
    value = manifest.get("native_mode", "compile")
    if value not in ("compile", "pinned"):
        raise ValueError("Unsupported native input mode")
    if (value == "pinned") != ("native_files" in manifest):
        raise ValueError("Native mode and artifact declaration disagree")
    return value


def _digest(value, length):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value)


def expected_record(lock):
    pin = lock.get("native_files")
    if not isinstance(pin, dict) or pin.get("schema") != "sparkring-pinned-native-archive/v1":
        raise ValueError("Source lock must declare the pinned native archive")
    if set(pin) - PIN_FIELDS:
        raise ValueError("Unknown pinned native lock metadata")
    if (not _digest(pin.get("sha256"), 64) or type(pin.get("bytes")) is not int
            or pin["bytes"] <= 0 or not isinstance(pin.get("members"), dict)
            or set(pin["members"]) != MEMBERS):
        raise ValueError("Invalid pinned native archive inventory")
    for member in pin["members"].values():
        if (not isinstance(member, dict) or set(member) != {"sha256", "bytes"} or not _digest(member["sha256"], 64)
                or type(member["bytes"]) is not int or member["bytes"] <= 0):
            raise ValueError("Invalid pinned native member identity")
    if (not _digest(pin.get("nccl_tree"), 40)
            or pin["nccl_tree"] != lock["nccl_build"]["result_tree"]
            or not _digest(pin.get("nccl_archive_sha256"), 64)
            or not _digest(pin.get("sparkcache_native_tree"), 40)
            or not _digest(pin.get("sparkcache_native_file_map_sha256"), 64)):
        raise ValueError("Pinned native source identities differ from the source lock")
    installed = {}
    for name, member in pin["members"].items():
        if name in LIBRARIES:
            component, destination = LIBRARIES[name]
            if (lock["runtime"][component + "_path"] != destination
                    or lock["runtime"][component + "_sha256"] != member["sha256"]):
                raise ValueError("Pinned native library differs from runtime path/hash")
        else:
            destination = METADATA_ROOT + "/" + name
        installed[destination] = member["sha256"]
    return {"schema": "sparkring-native-files-witness/v1", "archive_sha256": pin["sha256"],
            "archive_bytes": pin["bytes"], "members": {k: dict(v) for k, v in pin["members"].items()},
            "nccl_tree": pin["nccl_tree"], "nccl_archive_sha256": pin["nccl_archive_sha256"],
            "sparkcache_native_tree": pin["sparkcache_native_tree"],
            "sparkcache_native_file_map_sha256": pin["sparkcache_native_file_map_sha256"],
            "installed_files": installed}


def git_tree(files):
    """Hash regular-file bytes and executable modes using Git's tree encoding."""
    root = {}
    for name, (data, mode_bits) in files.items():
        node = root
        parts = PurePosixPath(name).parts
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).digest()
        node[parts[-1]] = (b"100755" if mode_bits & 0o111 else b"100644", blob)

    def encode(node):
        entries = []
        for name, value in node.items():
            directory = isinstance(value, dict)
            kind, digest = (b"40000", encode(value)) if directory else value
            entries.append((name.encode() + (b"/" if directory else b""),
                            kind + b" " + name.encode() + b"\0" + digest))
        data = b"".join(value for _, value in sorted(entries))
        return hashlib.sha1(b"tree " + str(len(data)).encode() + b"\0" + data).digest()

    return encode(root).hex()


def _describe(data, lock, manifest, source_archives):
    """Validate all artifact/source bytes before describing fixed install targets."""
    record = expected_record(lock)
    if len(data) != record["archive_bytes"] or sha(data) != record["archive_sha256"]:
        raise ValueError("Pinned native archive hash/size differs")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        members = archive.getmembers()
        if (len(members) != len(MEMBERS) or {m.name for m in members} != MEMBERS
                or any(not m.isfile() or m.linkname for m in members)):
            raise ValueError("Pinned native archive requires exactly six regular files")
        if ([m.name for m in members] != sorted(MEMBERS)
                or any(m.uid != 0 or m.gid != 0 or m.mode != 0o644 or m.pax_headers for m in members)):
            raise ValueError("Pinned native archive header policy differs")
    files = read_archive(data, {name: value["sha256"] for name, value in record["members"].items()})
    if any(len(files[name][0]) != value["bytes"] for name, value in record["members"].items()):
        raise ValueError("Pinned native member size differs")
    for name in LIBRARIES:
        binary = files[name][0]
        if (len(binary) < 64 or binary[:6] != b"\x7fELF\x02\x01"
                or int.from_bytes(binary[16:18], "little") != 3
                or int.from_bytes(binary[18:20], "little") != 183):
            raise ValueError("Pinned native input is not an AArch64 shared ELF")
    provenance = json.loads(files["provenance.json"][0])
    if (provenance.get("schema") != "sparkring-pinned-native-inputs/v1"
            or provenance.get("files") != {k: v for k, v in record["members"].items() if k != "provenance.json"}):
        raise ValueError("Pinned native provenance inventory differs")
    headers = provenance["archive_headers"]
    if (headers != {"compression": "none", "format": "ustar", "uid": 0, "gid": 0,
                   "mode": "0644", "mtime": headers.get("mtime"), "ordering": "member path, ascending"}
            or type(headers["mtime"]) is not int or headers["mtime"] < 0
            or any(m.mtime != headers["mtime"] for m in members)):
        raise ValueError("Pinned native provenance header policy differs")
    nccl = provenance["sources"]["nccl"]
    if (nccl["patched_tree"] != record["nccl_tree"]
            or nccl["base_revision"] != lock["nccl_build"]["base_revision"]
            or nccl["patch_sha256"] != lock["nccl_build"]["patch_sha256"]
            or manifest["nccl_build"]["tree"] != record["nccl_tree"]):
        raise ValueError("Pinned NCCL source identity differs")
    for name in ("nccl", "sparkcache"):
        source = manifest["nccl_build"] if name == "nccl" else manifest["sources"][name]
        if sha(source_archives[name]) != source["archive_sha256"]:
            raise ValueError("Retained native source archive differs")
        if name == "nccl" and source["archive_sha256"] != record["nccl_archive_sha256"]:
            raise ValueError("Retained NCCL archive differs from pinned source identity")
        source_files = read_archive(source_archives[name], source["files"])
        if name == "sparkcache":
            prefix = "sparkcache/native/"
            native = {p: v for p, v in source_files.items() if p.startswith(prefix)}
            file_map = {p: sha(v[0]) for p, v in native.items()}
            map_hash = sha(json.dumps(file_map, sort_keys=True, separators=(",", ":")).encode())
            tree = git_tree({p[len(prefix):]: v for p, v in native.items()})
            declared = provenance["sources"]["sparkcache"]
            if (not native or tree != record["sparkcache_native_tree"]
                    or map_hash != record["sparkcache_native_file_map_sha256"]
                    or declared["qualified_native_tree"] != tree or declared["selected_native_tree"] != tree
                    or declared["native_source_directory"] != "sparkcache/native"
                    or declared["native_file_count"] != len(native)
                    or declared["native_file_map_sha256"] != map_hash):
                raise ValueError("SparkCache native source bytes/modes differ from pinned provenance")
    return record


def describe(data, lock, manifest, source_archives):
    try:
        return _describe(data, lock, manifest, source_archives)
    except (KeyError, TypeError, AttributeError, tarfile.TarError) as error:
        raise ValueError("Malformed pinned native archive or source declaration") from error


def validate_inputs(root, manifest):
    if mode(manifest) == "compile":
        if (root / ARCHIVE).exists():
            raise ValueError("Undeclared native archive in compile mode")
        return None
    lock_bytes = (root / "source-lock.json").read_bytes()
    if sha(lock_bytes) != manifest["source_lock_sha256"]:
        raise ValueError("Native input source lock changed")
    record = describe((root / ARCHIVE).read_bytes(), json.loads(lock_bytes), manifest,
                      {name: (root / (name + ".tar")).read_bytes() for name in ("nccl", "sparkcache")})
    if manifest["native_files"] != record:
        raise ValueError("Native artifact declaration differs from verified inputs")
    return record


def install(root, manifest, rootfs):
    record = validate_inputs(root, manifest)
    if record is None:
        return None
    files = read_archive((root / ARCHIVE).read_bytes())
    targets = {}
    for name in files:
        absolute = LIBRARIES[name][1] if name in LIBRARIES else METADATA_ROOT + "/" + name
        path = under(rootfs, absolute[1:])
        if path.exists():
            raise ValueError("Pinned native destination already exists")
        targets[name] = path
    for name, path in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[name][0])
        path.chmod(0o755 if name in LIBRARIES else 0o644)
    return record


def verify(root, manifest, state, rootfs):
    selected = mode(manifest)
    if state.get("native_mode", "compile") != selected:
        raise ValueError("Installed native mode differs from manifest")
    record = validate_inputs(root, manifest)
    if state.get("native_files") != record:
        raise ValueError("Installed native artifact witness differs")
    if record is not None:
        for absolute, digest in record["installed_files"].items():
            path = under(rootfs, absolute[1:])
            if not path.is_file() or sha(path.read_bytes()) != digest:
                raise ValueError("Installed pinned native file differs")
        expected = {p[len(METADATA_ROOT) + 1:]: h for p, h in record["installed_files"].items()
                    if p.startswith(METADATA_ROOT + "/")}
        if inventory(under(rootfs, METADATA_ROOT[1:])) != expected:
            raise ValueError("Installed native metadata file set differs")
    return record
