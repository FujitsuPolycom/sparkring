"""Verify preinstalled model files against immutable Hugging Face source metadata."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import PurePosixPath
import re
from urllib.parse import quote, urlsplit
from urllib.request import urlopen


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_existing_assets(value):
    if (not isinstance(value, dict) or set(value) != {"schema", "image", "model_roots"}
            or value.get("schema") != "sparkring-existing-assets/v1"
            or value.get("image") != "all-ranks-preinstalled"):
        raise ValueError("Existing assets require the explicit preinstalled-image and model-root contract")
    roots = value["model_roots"]
    if not isinstance(roots, list) or len(roots) != 4:
        raise ValueError("Existing assets require four rank-ordered model roots")
    for root in roots:
        if (not isinstance(root, str) or not root.startswith("/") or root == "/"
                or any(c in root for c in ("\0", "\n", "\r", "\\", ":"))
                or PurePosixPath(root).as_posix() != root or ".." in root.split("/")):
            raise ValueError("Existing model roots must be normalized absolute Linux paths")
    return roots


def _relative(path):
    if (not isinstance(path, str) or not path or path in (".", "..") or PurePosixPath(path).is_absolute()
            or PurePosixPath(path).as_posix() != path or ".." in path.split("/")
            or any(c in path for c in ("\0", "\n", "\r", "\\", ":"))):
        raise ValueError("Model metadata contains an unsafe path")
    return path


def pinned_model_manifest(repository, revision, *, opener=urlopen):
    """Fetch every file's Git-blob or LFS identity at one full model revision."""
    if not isinstance(repository, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Model repository must be an explicit namespace/name")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Existing model verification requires a full immutable revision")
    base = f"https://huggingface.co/api/models/{quote(repository, safe='/')}/tree/{revision}"
    url = base + "?recursive=true&expand=false"
    pages, files, seen = [], {}, set()
    while url:
        if url in seen or not url.startswith(base + "?") or urlsplit(url).fragment:
            raise ValueError("Model metadata pagination escaped its pinned revision")
        seen.add(url)
        with opener(url, timeout=60) as response:
            if response.geturl() != url:
                raise ValueError("Model metadata redirected away from its pinned source")
            raw = response.read()
            entries = json.loads(raw)
            if not isinstance(entries, list):
                raise ValueError("Model tree response must be a file list")
            pages.append({"url": url, "body_utf8": raw.decode("utf-8"),
                          "body_sha256": hashlib.sha256(raw).hexdigest()})
            following = re.findall(r'<([^>]+)>;\s*rel="next"', response.headers.get("Link", ""))
            if len(following) > 1:
                raise ValueError("Ambiguous model metadata pagination")
            url = following[0] if following else None
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Model tree entry must be an object")
            path = _relative(entry.get("path"))
            if entry.get("type") == "directory":
                continue
            if entry.get("type") != "file" or path in files:
                raise ValueError("Model metadata contains duplicate or unsupported entries")
            size = entry.get("size")
            if type(size) is not int or size < 0:
                raise ValueError("Model file size must be a nonnegative integer")
            lfs = entry.get("lfs")
            if lfs is not None:
                if (not isinstance(lfs, dict) or lfs.get("size") != size
                        or not isinstance(lfs.get("oid"), str)
                        or not re.fullmatch(r"[0-9a-f]{64}", lfs["oid"])):
                    raise ValueError("Model LFS identity is incomplete")
                files[path] = {"size": size, "sha256": lfs["oid"]}
            else:
                oid = entry.get("oid")
                if not isinstance(oid, str) or not re.fullmatch(r"[0-9a-f]{40}", oid):
                    raise ValueError("Model Git-blob identity is incomplete")
                files[path] = {"size": size, "git_blob_sha1": oid}
    if not files or not {"config.json", "model.safetensors.index.json"} <= set(files):
        raise ValueError("Pinned model tree lacks required model metadata")
    return {"schema": "sparkring-model-source-metadata/v1", "repository": repository,
            "revision": revision, "files": files, "raw_pages": pages,
            "metadata_sha256": canonical_sha(pages)}


REMOTE_VERIFY = r'''
import hashlib,json,os,pathlib,stat,sys
root=pathlib.Path(sys.argv[1]);expected=json.load(sys.stdin);files={}
if not root.is_absolute() or root==pathlib.Path('/') or any(p.is_symlink() for p in (root,*root.parents)):
 raise SystemExit('Existing model root is unsafe or contains a symlink')
if not root.is_dir():raise SystemExit('Existing model root is not a directory')
def walk_error(error):raise error
for directory,dirs,names in os.walk(root,followlinks=False,onerror=walk_error):
 for name in dirs+names:
  path=pathlib.Path(directory)/name
  mode=path.lstat().st_mode
  if stat.S_ISLNK(mode):raise SystemExit('Existing model contains a symlink')
  if not stat.S_ISDIR(mode) and not stat.S_ISREG(mode):raise SystemExit('Existing model contains a special file')
 for name in names:
  path=pathlib.Path(directory)/name;relative=path.relative_to(root).as_posix()
  if relative.startswith('.cache/huggingface/'):continue
  files[relative]=path
if set(files)!=set(expected):
 raise SystemExit('Existing model file set differs from pinned revision: missing='+str(sorted(set(expected)-set(files)))+' extra='+str(sorted(set(files)-set(expected))))
result={};total=0
for name,path in sorted(files.items()):
 record=expected[name];fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as stream:
  before=os.fstat(stream.fileno())
  if not stat.S_ISREG(before.st_mode) or before.st_size!=record['size']:raise SystemExit('Model file size/type differs: '+name)
  digest=hashlib.sha256();blob=hashlib.sha1();blob.update(b'blob '+str(before.st_size).encode()+bytes([0]))
  for chunk in iter(lambda:stream.read(8388608),b''):digest.update(chunk);blob.update(chunk)
  after=os.fstat(stream.fileno());located=path.stat(follow_symlinks=False)
  if (located.st_dev,located.st_ino)!=(before.st_dev,before.st_ino):raise SystemExit('Model path changed during verification: '+name)
  if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise SystemExit('Model file changed during verification: '+name)
 actual=digest.hexdigest()
 if 'sha256' in record and actual!=record['sha256']:raise SystemExit('Model LFS content differs: '+name)
 if 'git_blob_sha1' in record and blob.hexdigest()!=record['git_blob_sha1']:raise SystemExit('Model Git-blob content differs: '+name)
 result[name]=actual;total+=before.st_size
print(json.dumps({'model_files':result,'files_verified':len(result),'bytes_verified':total}))
'''


def verify_existing_models(run, hosts, roots, target, *, opener=urlopen):
    """Read every model byte on each rank; never download or modify model files."""
    roots = validate_existing_assets({"schema": "sparkring-existing-assets/v1",
                                     "image": "all-ranks-preinstalled", "model_roots": roots})
    if (len(hosts) != 4 or [h.get("rank") for h in hosts] != list(range(4))
            or len({h.get("host") for h in hosts}) != 4):
        raise ValueError("Existing model verification requires four distinct ordered ranks")
    source = pinned_model_manifest(target["repository"], target["revision"], opener=opener)
    canonical, ranks = None, []
    def verify_rank(host, root):
        return json.loads(run.remote(host["host"], ["python3", "-I", "-c", REMOTE_VERIFY, root],
                                     input=json.dumps(source["files"]).encode(), timeout=7200))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(verify_rank, host, root) for host, root in zip(hosts, roots)]
        results = [future.result() for future in pending]
    for host, root, result in zip(hosts, roots, results):
        actual = result.get("model_files")
        if (not isinstance(actual, dict) or set(actual) != set(source["files"])
                or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in actual.values())
                or result.get("files_verified") != len(source["files"])
                or result.get("bytes_verified") != sum(v["size"] for v in source["files"].values())):
            raise ValueError("Existing model verifier returned an incomplete inventory")
        for name, key in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
            if actual.get(name) != target[key]:
                raise ValueError("Existing model metadata differs from runtime pins")
        if canonical is not None and actual != canonical:
            raise ValueError("Existing model file hashes differ between ranks")
        canonical = actual
        ranks.append({"rank": host["rank"], "host": host["host"], "root": root,
                      "files_verified": result["files_verified"], "bytes_verified": result["bytes_verified"],
                      "manifest_sha256": canonical_sha(actual)})
    return {"model_files": canonical, "model_source_receipt": source, "ranks": ranks}
