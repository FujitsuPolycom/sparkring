"""Land request PO into impl: an externally driven progress loop, so one thread can serve every SIRCL session
of a process (libsircl needs it for processes that hold several communicators; REPORT section 6.5).

- `_roce_proxy.c`: the progress loop's body becomes `progress_pass`; `roce_start_external` and
  `roce_progress` let a caller's thread drive a session; the thread-per-session start is unchanged.
  Native ABI 10.
- `_proxy.py`: ABI 10, `Proxy.start_external()` and `Proxy.progress()`.
- `README.md`: the binding's functions; `tests/test_progress_external.py`: one thread drives a path of
  four with two lanes on the verbs stand-in, one-shot and two-shot ops exact.

The edit script is edit_PO.py (beside this script, or in PO/); PO/base.json records the base it was cut
from. Copy this directory's files to the lead scratchpad: land_PO.py at its root, edit_PO.py and base.json
in PO/.

python land_PO.py --build [--rebase]
    Builds rsPO/sircl from impl's current package, applies edit_PO.py, runs ruff and the CPU suite, and
    records the impl files it was built from (rsPO/BUILD.json). Refuses when impl's version of a file the
    script edits differs from the base it was cut from (another change landed in it), unless --rebase,
    which applies the script anyway; every replacement must still find its old text exactly once.
python land_PO.py
    Copies every changed or new file of rsPO/sircl into impl's package and adds the change to impl's
    STATUS.md. First saves impl's STATUS.md beside this script as STATUS.md.before-PO.

Landing refuses when impl's package or STATUS.md differ from the recorded build (run --build again), or
when the built tree changed after its tests.

Ordering: `_roce_proxy.c` and `_proxy.py` are also the bidirectional all-gather change's files; land PO
after that change with --build --rebase. The tuning key's native hash (`tuning.native_hash`) covers
`_roce_proxy.c`, so a tuning table matches only sessions built from the proxy source it was measured with.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
IMPL = S / "ring8" / "cleanroom" / "impl"
PKG = IMPL / "spark_transport" / "sircl"
BASE = S / "rsPO"
TREE = BASE / "sircl"
RECORD = BASE / "BUILD.json"
CUT_BASE = S / "PO" / "base.json"
SCRIPT = S / "PO" / "edit_PO.py"
SKIP = ("__pycache__", ".ruff_cache", ".pytest_cache", ".benchmarks", "sircl-ring-results")
STATUS_ANCHOR = "| vLLM adapter (`sparkring_sircl.vllm`) | maintained separately |"
STATUS_ROW = (
    "| Shared progress loop: a session started with `start_external` posts from the caller's thread through "
    "`progress()`, one pass per call, so one thread can serve every session of a process (native ABI 10) | "
    "implemented | CPU tests (`tests/test_progress_external.py`: a path of four with two lanes on the verbs "
    "stand-in, every session driven by one thread, one-shot and two-shot ops exact; a session with its own "
    "thread refuses `progress()`); the thread-per-session start is unchanged; not run on a ring |\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_of(root: Path) -> dict[str, str]:
    found = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_file() and not any(part in SKIP for part in rel.parts):
            found[rel.as_posix()] = digest(path)
    return found


def drift() -> list[str]:
    cut = json.loads(CUT_BASE.read_text(encoding="utf-8"))
    return [name for name, value in sorted(cut.items())
            if not (PKG / name).exists() or digest(PKG / name) != value]


def build(rebase: bool) -> None:
    moved = drift()
    if moved and not rebase:
        raise SystemExit(f"impl changed in files PO edits since its script was cut: {moved}; run --build --rebase "
                         "to apply it anyway")
    before, status = files_of(PKG), digest(IMPL / "STATUS.md")
    staging = BASE / "next"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(PKG, staging / "sircl", ignore=shutil.ignore_patterns(*SKIP))
    root = str(staging / "sircl")
    subprocess.run([sys.executable, str(SCRIPT), root], check=True)
    subprocess.run([sys.executable, "-m", "ruff", "check", "--select", "E,F,W", "--ignore", "E501", "."],
                   cwd=root, check=True)
    subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], cwd=root, check=True)
    if files_of(PKG) != before or digest(IMPL / "STATUS.md") != status:
        raise SystemExit("impl changed while the tree was built; run --build again")
    built = files_of(staging / "sircl")
    changed = sorted(name for name, value in built.items() if before.get(name) != value)
    if TREE.exists():
        shutil.rmtree(TREE)
    (staging / "sircl").rename(TREE)
    shutil.rmtree(staging)
    RECORD.write_text(json.dumps({"impl": before, "status": status, "tree": built, "changed": changed,
                                  "rebased": moved}, indent=1, sort_keys=True), encoding="utf-8")
    print("package files PO replaces or adds:", changed)
    print(f"built and tested {TREE}; recorded {RECORD}")


def replace_once(path: Path, old: str, new: str) -> None:
    raw = path.read_bytes()
    crlf = b"\r\n" in raw and raw.count(b"\r\n") == raw.count(b"\n")
    text = raw.decode("utf-8").replace("\r\n", "\n")
    if text.count(old) != 1:
        raise SystemExit(f"{path.name}: found {text.count(old)} of {old[:80]!r}")
    text = text.replace(old, new)
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


def land() -> None:
    if not RECORD.exists():
        raise SystemExit("no recorded build; run --build first")
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    current = files_of(PKG)
    moved = sorted(name for name in set(current) | set(record["impl"]) if current.get(name) != record["impl"].get(name))
    if moved or digest(IMPL / "STATUS.md") != record["status"]:
        raise SystemExit(f"impl changed after the build; run --build again: {moved[:12]}")
    if files_of(TREE) != record["tree"]:
        raise SystemExit("the built tree changed after its tests; run --build again")
    shutil.copyfile(IMPL / "STATUS.md", S / "STATUS.md.before-PO")
    for name in record["changed"]:
        (PKG / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TREE / name, PKG / name)
    replace_once(IMPL / "STATUS.md", STATUS_ANCHOR, STATUS_ROW + STATUS_ANCHOR)
    print(f"landed PO: {len(record['changed'])} package files and STATUS.md")


if __name__ == "__main__":
    build("--rebase" in sys.argv) if "--build" in sys.argv else land()
