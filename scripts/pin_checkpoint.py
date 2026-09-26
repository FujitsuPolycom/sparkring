#!/usr/bin/env python3
"""Write the pin manifest and the SHA256SUMS files for one checkpoint revision.

A pin manifest (schema ``sparkring-checkpoint-pins/v1``) describes every file
of one Hugging Face model revision and lives at
``profiles/checkpoints/<owner>--<name>/<revision>.json``:

- ``files`` maps each file name to its ``size``, ``sha256`` and ``git_blob``;
  LFS files also carry ``"lfs": true`` and, when the Hub reports one, their
  ``xet_hash``.
- ``index`` names the weight index, ``model.safetensors.index.json``.
- ``weights`` is the sorted set of the index's ``weight_map`` values, so
  consumers never need the index itself to know which files are weights.
- ``optional`` lists the documentation and repository metadata that no serving
  component reads. The required files are ``files`` minus ``optional``.

``runtime.common.installer.checkpoint_pins`` loads and validates a manifest.
Each profile that pins the revision and keeps a ``SHA256SUMS`` file beside its
``profile.json`` gets that file rewritten to list exactly the required files.
Profiles without a ``SHA256SUMS`` file are left alone.

Values come from the Hub's tree listing at the full commit id, which must
resolve to itself. An LFS file's SHA-256 is its LFS object id, and its Git blob
id must equal the blob id of the standard LFS pointer for that SHA-256 and size.
Every other file is downloaded; its Git blob id is recomputed from the bytes and
must equal the listing before its SHA-256 is taken. The index is downloaded to
read its ``weight_map`` and must hash to its pinned SHA-256 first. Nothing else
is downloaded, so no weight file is ever fetched, and every download is limited
to 64 MiB.

Requests are anonymous: no token is read or sent. The script needs network
access and is not part of installation; installations read the manifests that
ship with the package.

Usage::

    python scripts/pin_checkpoint.py --repository OWNER/NAME --revision COMMIT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import profiles  # noqa: E402

SCHEMA = "sparkring-checkpoint-pins/v1"
INDEX = "model.safetensors.index.json"
ENDPOINT = "https://huggingface.co"
USER_AGENT = "sparkring-pin-checkpoint"
# Bounds every download. Configuration, tokenizer and index files are far
# smaller; weight files are larger and are never requested.
DOWNLOAD_LIMIT = 64 * 1024 * 1024
LFS_POINTER = "version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n"
# Hub repository names never contain "--" or "..", so `<owner>--<name>` maps
# back to exactly one repository, as the Hub cache's `models--<owner>--<name>`.
REPOSITORY_PART = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp")
DOCUMENT_SUFFIXES = (".md", ".pdf")


def valid_repository(repository):
    return (isinstance(repository, str) and "--" not in repository and ".." not in repository
            and re.fullmatch(REPOSITORY_PART + "/" + REPOSITORY_PART, repository) is not None)


def hexadecimal(value, length):
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def safe_name(name):
    """A normalized relative POSIX path that stays inside a checkpoint directory.

    Refused: absolute paths, empty, ``.`` and ``..`` components, ``.cache`` and
    ``.git`` components, NUL, CR, LF and backslash. The installer's loader
    applies the same rule.
    """
    return (isinstance(name, str) and bool(name) and not name.startswith("/")
            and not any(character in name for character in "\0\r\n\\")
            and PurePosixPath(name).as_posix() == name
            and not {"", ".", "..", ".cache", ".git"} & set(name.split("/")))


def optional(name):
    """True for documentation and repository metadata no serving component reads.

    These are ``.gitattributes``, names starting with ``README``, ``LICENSE`` or
    ``NOTICE``, top-level Markdown and PDF documents, and image files. Every
    other file, including nested Markdown, is required.
    """
    path = PurePosixPath(name)
    suffix = path.suffix.lower()
    return (path.name == ".gitattributes"
            or path.name.startswith(("README", "LICENSE", "NOTICE"))
            or (len(path.parts) == 1 and suffix in DOCUMENT_SUFFIXES)
            or suffix in IMAGE_SUFFIXES)


def git_blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Hub:
    """Anonymous, read-only access to the public Hugging Face Hub."""

    def __init__(self, endpoint=ENDPOINT, timeout=120):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def _open(self, url):
        # Only the User-Agent header is sent: never a token or credential.
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        return urllib.request.urlopen(request, timeout=self.timeout)

    def commit(self, repository, revision):
        """The commit id that ``revision`` resolves to."""
        with self._open(f"{self.endpoint}/api/models/{repository}/revision/{revision}") as response:
            return json.load(response).get("sha")

    def tree(self, repository, revision):
        """Every entry of the revision, following the listing's pagination links."""
        url = f"{self.endpoint}/api/models/{repository}/tree/{revision}?recursive=true&expand=true"
        items = []
        while url:
            with self._open(url) as response:
                page = json.load(response)
                link = response.headers.get("Link") or ""
            if not isinstance(page, list):
                raise ValueError(f"{repository}@{revision}: the Hub returned an unexpected tree listing")
            items.extend(page)
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = match.group(1) if match else None
            if url is not None and not url.startswith(self.endpoint + "/"):
                raise ValueError(f"{repository}@{revision}: tree pagination leaves {self.endpoint}")
        return items

    def download(self, repository, revision, name, size):
        """At most ``size + 1`` bytes of one file, so an oversized body is detected."""
        url = f"{self.endpoint}/{repository}/resolve/{revision}/{urllib.parse.quote(name)}"
        with self._open(url) as response:
            return response.read(size + 1)


def fetch(hub, repository, revision, name, size):
    if size > DOWNLOAD_LIMIT:
        raise ValueError(f"{name}: {size} bytes exceeds the {DOWNLOAD_LIMIT}-byte download limit")
    data = hub.download(repository, revision, name, size)
    if len(data) != size:
        raise ValueError(f"{name}: downloaded {len(data)} bytes, the Hub lists {size}")
    return data


def manifest(hub, repository, revision):
    """The pin manifest of ``repository`` at the full commit id ``revision``."""
    if not valid_repository(repository):
        raise ValueError(f"{repository!r} is not a Hugging Face owner/name repository")
    if not hexadecimal(revision, 40):
        raise ValueError("Pin a full 40-character commit id, not a branch or tag")
    resolved = hub.commit(repository, revision)
    if resolved != revision:
        raise ValueError(f"{repository}@{revision} resolves to {resolved!r}, not to itself")
    files, contents = {}, {}
    for item in hub.tree(repository, revision):
        kind, name = item.get("type"), item.get("path")
        if kind == "directory":
            continue
        if kind != "file" or not safe_name(name):
            raise ValueError(f"Unsupported tree entry {kind!r} {name!r}")
        if name in files:
            raise ValueError(f"{name}: listed twice")
        size, blob, lfs = item.get("size"), item.get("oid"), item.get("lfs")
        if type(size) is not int or size <= 0:
            raise ValueError(f"{name}: the Hub lists size {size!r}; pins need a positive size")
        if not hexadecimal(blob, 40):
            raise ValueError(f"{name}: the Hub lists Git blob id {blob!r}")
        if lfs:
            digest = lfs.get("oid")
            if lfs.get("size") != size or not hexadecimal(digest, 64):
                raise ValueError(f"{name}: inconsistent LFS object {lfs!r}")
            pointer = LFS_POINTER.format(oid=digest, size=size).encode()
            if git_blob_id(pointer) != blob:
                raise ValueError(f"{name}: Git blob id {blob} is not the LFS pointer of {digest}")
            entry = {"size": size, "sha256": digest, "git_blob": blob, "lfs": True}
            xet = item.get("xetHash")
            if xet is not None:
                if not hexadecimal(xet, 64):
                    raise ValueError(f"{name}: the Hub lists Xet hash {xet!r}")
                entry["xet_hash"] = xet
        else:
            data = fetch(hub, repository, revision, name, size)
            measured = git_blob_id(data)
            if measured != blob:
                raise ValueError(f"{name}: downloaded bytes have Git blob id {measured}, the Hub lists {blob}")
            entry = {"size": size, "sha256": hashlib.sha256(data).hexdigest(), "git_blob": blob}
            contents[name] = data
        files[name] = entry
    if INDEX not in files:
        raise ValueError(f"{repository}@{revision} has no {INDEX}")
    index = contents.get(INDEX)
    if index is None:
        index = fetch(hub, repository, revision, INDEX, files[INDEX]["size"])
        if hashlib.sha256(index).hexdigest() != files[INDEX]["sha256"]:
            raise ValueError(f"{INDEX}: downloaded bytes differ from its LFS SHA-256")
    weight_map = json.loads(index).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map or not all(isinstance(v, str) for v in weight_map.values()):
        raise ValueError(f"{INDEX}: expected a nonempty weight_map of file names")
    weights = sorted(set(weight_map.values()))
    documentation = sorted(name for name in files if optional(name))
    required = set(files) - set(documentation)
    missing = [name for name in [INDEX, "config.json", *weights] if name not in required]
    if missing:
        raise ValueError(f"Required files are absent or classified as documentation: {', '.join(missing)}")
    return {"schema": SCHEMA, "repository": repository, "revision": revision, "index": INDEX,
            "weights": weights, "optional": documentation, "files": {name: files[name] for name in sorted(files)}}


def required_names(pins):
    return sorted(set(pins["files"]) - set(pins["optional"]))


def encode(pins):
    """Manifest text: JSON with one line per pinned file, for readable diffs."""
    lines = ["{"]
    for key in ("schema", "repository", "revision", "index"):
        lines.append(f"  {json.dumps(key)}: {json.dumps(pins[key])},")
    for key in ("weights", "optional"):
        if pins[key]:
            lines.append(f"  {json.dumps(key)}: [")
            lines.append(",\n".join(f"    {json.dumps(name)}" for name in pins[key]))
            lines.append("  ],")
        else:
            lines.append(f"  {json.dumps(key)}: [],")
    lines.append('  "files": {')
    lines.append(",\n".join(f"    {json.dumps(name)}: {json.dumps(entry)}" for name, entry in sorted(pins["files"].items())))
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def checksums(pins):
    """``sha256sum --check`` input listing every required file, sorted by name."""
    return "".join(f"{pins['files'][name]['sha256']}  {name}\n" for name in required_names(pins))


def manifest_path(root, repository, revision):
    return Path(root) / "profiles" / "checkpoints" / repository.replace("/", "--") / (revision + ".json")


def pinning_profiles(root, repository, revision):
    """Catalog profiles that pin this revision and keep a SHA256SUMS file."""
    result = []
    for profile_id in sorted(profiles.catalog(root)):
        if not (Path(root) / "profiles" / profile_id / "SHA256SUMS").is_file():
            continue
        model = profiles.resolve(profile_id, root=root).get("model") or {}
        if (model.get("repository"), model.get("revision")) == (repository, revision):
            result.append(profile_id)
    return result


def check_contracts(root, pins, profile_ids):
    """Refuse a manifest whose config.json or index differs from a profile's pins."""
    for profile_id in profile_ids:
        model = profiles.resolve(profile_id, root=root)["model"]
        for name, key in (("config.json", "config_sha256"), (pins["index"], "index_sha256")):
            if model.get(key) != pins["files"][name]["sha256"]:
                raise ValueError(f"{profile_id} pins {name} as {model.get(key)}, "
                                 f"but the revision holds {pins['files'][name]['sha256']}")


def replace(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)
    return path


def write(root, pins, profile_ids):
    """Write the manifest and each listed profile's SHA256SUMS; return the paths."""
    root = Path(root)
    written = [replace(manifest_path(root, pins["repository"], pins["revision"]), encode(pins))]
    sums = checksums(pins)
    for profile_id in profile_ids:
        written.append(replace(root / "profiles" / profile_id / "SHA256SUMS", sums))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repository", required=True, help="Hugging Face repository, owner/name")
    parser.add_argument("--revision", required=True, help="full 40-character commit id")
    parser.add_argument("--endpoint", default=ENDPOINT, help="Hub endpoint (default: %(default)s)")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository checkout to write into")
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    try:
        profile_ids = pinning_profiles(args.root, args.repository, args.revision)
        pins = manifest(Hub(args.endpoint), args.repository, args.revision)
        check_contracts(args.root, pins, profile_ids)
        written = write(args.root, pins, profile_ids)
    except (ValueError, OSError) as error:
        print(f"pin_checkpoint: {error}", file=sys.stderr)
        return 1
    required = required_names(pins)
    print(f"{args.repository}@{args.revision}: {len(pins['files'])} files, {len(required)} required "
          f"({len(pins['weights'])} weights), {len(pins['optional'])} optional")
    for path in written:
        print(path.relative_to(args.root).as_posix() if path.is_relative_to(args.root) else path)
    if not profile_ids:
        print("No profile with a SHA256SUMS file pins this revision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
