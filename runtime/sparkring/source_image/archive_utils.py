"""Strict, deterministic package archives; standard library only."""
import hashlib
import io
from pathlib import Path, PurePosixPath
import tarfile

STARTUP_PATHS = {
    "runtime/glm53-flash-jj-r8-gb10/serve_with_warmup.py": "/opt/sparkring/bin/serve-with-warmup.py",
    "runtime/glm53-flash-jj-r8-gb10/warmup_dflash.py": "/opt/sparkring/bin/warmup_dflash.py",
    "runtime/glm53-flash-jj-r8-gb10/startup_admission.py": "/opt/sparkring/bin/startup_admission.py",
    "runtime/glm53-flash-jj-r8-gb10/scheduler_liveness.py": "/opt/sparkring/bin/scheduler_liveness.py",
}
WARMUP_SOURCE = "runtime/glm53-flash-jj-r8-gb10/warmup_dflash.py"
WARMUP_TRANSFORM = "glm53-thinking-low/v1"


def transform_warmup(data, transform):
    if transform != WARMUP_TRANSFORM:
        raise ValueError("Unsupported warmup-only transform")
    needle = b'"chat_template_kwargs": {"enable_thinking": False},'
    if data.count(needle) != 1:
        raise ValueError("Warmup transform requires exactly one source preimage")
    result = data.replace(needle, b'"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low",', 1)
    compile(result, WARMUP_SOURCE, "exec")
    return result


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative_path(name):
    p = PurePosixPath(name)
    if (not name or name != p.as_posix() or p.is_absolute()
            or any(x in ("", ".", "..") for x in name.split("/"))
            or "\\" in name or ":" in name or "\x00" in name):
        raise ValueError(f"Unsafe relative path: {name!r}")
    return p


def under(root, name):
    p = relative_path(name)
    root = Path(root).resolve()
    result = root.joinpath(*p.parts)
    for part in (result, *result.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"Symlink in destination path: {part}")
    if not result.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes root: {name}")
    return result


def read_archive(data, expected=None):
    files = {}
    folded = set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        for item in archive:
            name = item.name.rstrip("/") if item.isdir() else item.name
            relative_path(name)
            if item.isdir():
                continue
            if not item.isfile() or item.linkname:
                raise ValueError(f"Archive contains a link/special file: {name}")
            if name.casefold() in folded:
                raise ValueError(f"Duplicate archive path: {name}")
            folded.add(name.casefold())
            files[name] = (archive.extractfile(item).read(), item.mode & 0o777)
    for name in files:
        if any(parent.as_posix().casefold() in folded for parent in PurePosixPath(name).parents
               if parent.as_posix() != "."):
            raise ValueError(f"Archive file is also a parent directory: {name}")
    if expected is not None:
        actual = {name: sha(value[0]) for name, value in files.items()}
        if actual != expected:
            raise ValueError("Archive file set or content differs from manifest")
    return files


def make_archive(files, epoch):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, (data, mode) in sorted(files.items()):
            relative_path(name)
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o755 if mode & 0o111 else 0o644
            entry.mtime = epoch
            entry.uid = entry.gid = 0
            archive.addfile(entry, io.BytesIO(data))
    return output.getvalue()


def extract_checked(data, root, expected):
    files = read_archive(data, expected)
    # Validate every destination before creating any file.
    paths = {name: under(root, name) for name in files}
    for name, (payload, mode) in files.items():
        path = paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(mode)


def inventory(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected package symlink: {path}")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            result[name] = sha(path.read_bytes())
    return result
