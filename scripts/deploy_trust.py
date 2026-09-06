"""Build isolated verification commands anchored to reviewed preparation inputs."""

from __future__ import annotations

import inspect
import json


def _verify_preparation(
    workspace,
    expected_digest,
    source_root,
    launch_root,
    preparation_path,
    require_launch,
):
    """Use only the interpreter's standard library before checking staged code."""
    import hashlib
    import json
    import os
    from pathlib import Path, PurePosixPath
    import re
    import stat

    def safe_path(path):
        path = Path(path)
        if not path.is_absolute():
            raise ValueError("Verification paths must be absolute")
        if ".." in path.parts:
            raise ValueError("Verification paths cannot contain parent traversal")
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise ValueError("Verification path contains a symlink")
        return path

    def file_digest(path):
        path = safe_path(path)
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("Verified input must be a regular file")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(8 << 20), b""):
                digest.update(data)
        return digest.hexdigest()

    def file_map(value):
        if not isinstance(value, dict) or not value:
            raise ValueError("Preparation requires a nonempty authenticated file map")
        for name, digest in value.items():
            relative = PurePosixPath(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or str(relative) != name
                or "\\" in name
                or name in ("", ".")
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValueError("Invalid authenticated file-map entry")
        return value

    path = safe_path(preparation_path)
    # Check file type before parsing; a device or FIFO is not a preparation file.
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Preparation must be a regular file")
    document = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        json.dumps(
            {key: value for key, value in document.items() if key != "sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if digest != expected_digest:
        raise ValueError("Preparation differs from reviewed inputs")
    if document["spec"]["workspace"] != workspace:
        raise ValueError("Workspace differs from reviewed inputs")
    source = safe_path(source_root)
    sources = file_map(document["source"]["files"])
    for name, expected in sources.items():
        if file_digest(source / name) != expected:
            raise ValueError("Authenticated source file changed: " + name)
    # Added package initializers or native modules must not shadow verified files.
    for current, directories, files in os.walk(source, followlinks=False):
        directories[:] = [name for name in directories if name != "__pycache__"]
        for name in directories:
            safe_path(Path(current) / name)
        for name in files:
            candidate = Path(current) / name
            relative = candidate.relative_to(source).as_posix()
            if (
                candidate.suffix in (".py", ".pyc", ".pyo", ".so", ".pyd", ".dll")
                and relative not in sources
            ):
                raise ValueError("Unlisted executable source: " + relative)
    if require_launch:
        launch = safe_path(launch_root)
        files = file_map(document.get("launch_files"))
        required = {"site.json", "fabric.json", "launch-rank.sh", "fabric-plan.json"}
        required.update(f"rank{rank}.env" for rank in range(4))
        if (
            set(files) != required
            or {item.name for item in launch.iterdir()} != required
        ):
            raise ValueError("Rendered launch file list differs from reviewed inputs")
        for name, expected in files.items():
            if file_digest(launch / name) != expected:
                raise ValueError("Authenticated launch file changed: " + name)
    return document


def trusted_check(
    workspace,
    preparation_digest,
    *,
    source_root=None,
    launch_root=None,
    preparation_path=None,
    require_launch=True,
    python="python3",
    after="",
    after_args=(),
):
    """Return a sealed-plan-ready command; never import the file being verified."""
    from pathlib import Path

    source_root = str(source_root or (Path(workspace) / "source"))
    launch_root = str(launch_root or (Path(workspace) / "launch"))
    preparation_path = str(preparation_path or (Path(workspace) / "preparation.json"))
    config = [
        str(workspace),
        preparation_digest,
        source_root,
        launch_root,
        preparation_path,
        require_launch,
    ]
    program = inspect.getsource(_verify_preparation)
    program += "\nimport json,sys,tempfile\n"
    program += "config=json.loads(sys.argv[1]);document=_verify_preparation(*config)\n"
    if after:
        program += "with tempfile.TemporaryDirectory(prefix='sparkring-import-cache-') as cache:\n"
        program += " sys.pycache_prefix=cache;sys.dont_write_bytecode=True\n"
        program += " source_root=config[2];launch_root=config[3];arguments=json.loads(sys.argv[2])\n"
        program += " sys.path.insert(0,source_root)\n"
        program += "\n".join(" " + line for line in after.splitlines()) + "\n"
    else:
        program += "print(json.dumps({'verified':True}))\n"
    return [
        python,
        "-I",
        "-B",
        "-c",
        program,
        json.dumps(config),
        json.dumps(list(after_args)),
    ]


def trusted_script(workspace, preparation_digest, script, args, **options):
    """Execute a manifest-listed script only after authenticating its source tree."""
    code = (
        "from pathlib import Path\nimport runpy\n"
        "script=arguments[0]\n"
        "if script not in document['source']['files']: raise ValueError('Script is not authenticated')\n"
        "sys.argv=[str(Path(source_root)/script),*arguments[1:]]\n"
        "sys.path.insert(0,str(Path(sys.argv[0]).parent))\n"
        "runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    return trusted_check(
        workspace, preparation_digest, after=code, after_args=[script, *args], **options
    )
