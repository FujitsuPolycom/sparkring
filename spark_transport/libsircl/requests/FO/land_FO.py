"""Land request FO into impl: flags-only own items, so a rank whose peer discards its own link items (the
empty direction of libsircl's pair exchange) posts their flags without payload.

- `_roce_proxy.c`: bit 24 of a link op word is accepted (bits 25-31 still refused) and makes every own item
  of the op 0 bytes, posted as its flags only, as a staggered link's empty items are. Native ABI 9, the
  connection record and the wire unchanged.
- `protocol.py`: `RING_OWN_FLAGS` and `ring_op_word(..., own_flags=)`; `README.md`: the op word's bit 24;
  `tests/test_ring_links.py`: the high-bit refusal case on bit 25; `tests/test_link_own_flags.py`: a pair
  on the verbs stand-in, one rank's own items flags only (no payload reaches its peer, 0 link bytes
  posted, every flag arrives, the other direction exact), and without the bit both directions exact.

The edit script is edit_FO.py (beside this script, or in FO/); FO/base.json records the base it was cut
from. Copy this directory's files to the lead scratchpad: land_FO.py at its root, edit_FO.py and base.json
in FO/.

python land_FO.py --build [--rebase]
    Builds rsFO/sircl from impl's current package, applies edit_FO.py, runs ruff and the CPU suite, and
    records the impl files it was built from (rsFO/BUILD.json). Refuses when impl's version of a file the
    script edits differs from the base it was cut from (another change landed in it), unless --rebase,
    which applies the script anyway; every replacement must still find its old text exactly once.
python land_FO.py
    Copies every changed or new file of rsFO/sircl into impl's package and adds the change to impl's
    STATUS.md. First saves impl's STATUS.md beside this script as STATUS.md.before-FO.

Landing refuses when impl's package or STATUS.md differ from the recorded build (run --build again), or
when the built tree changed after its tests.

Ordering: cut against impl after the session-close change (TD; `_roce_proxy.c` SHA-256
5756d703...dfd58fd9a, `roce_destroy` returning the number of failed verbs calls); FO's replacements do not
touch TD's code. Landing also updates the libsircl sentence of TD's STATUS row. The tuning key's native hash (`tuning.native_hash`)
covers `_roce_proxy.c`, so a tuning table matches only sessions built from the proxy source it was
measured with; the key's ABI part (abi9) does not change.
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
BASE = S / "rsFO"
TREE = BASE / "sircl"
RECORD = BASE / "BUILD.json"
CUT_BASE = S / "FO" / "base.json"
SCRIPT = S / "FO" / "edit_FO.py"
SKIP = ("__pycache__", ".ruff_cache", ".pytest_cache", ".benchmarks", "sircl-ring-results")
STATUS_ANCHOR = "| vLLM adapter (`sparkring_sircl.vllm`) | maintained separately |"
# TD's row says how libsircl binds roce_destroy; FO is the change with which libsircl re-vendors the file.
TD_LIBSIRCL = ("libsircl's copy of the native layer (`src/transport/sircl_roce_proxy.c`, SHA-256 `7deb5b1a...f2ae83`) "
               "declares `roce_destroy` without a return value and its engine binds it so: that copy stays "
               "byte-identical until libsircl declares and checks the int return")
TD_LIBSIRCL_NOW = ("libsircl re-vendors this file together with flags-only own items (request FO), and then declares "
                   "and checks the int return; until then its copy (`src/transport/sircl_roce_proxy.c`, SHA-256 "
                   "`7deb5b1a...f2ae83`) declares `roce_destroy` without a return value and its engine binds it so")
STATUS_ROW = (
    "| Flags-only own items: bit 24 of a link op word (`protocol.RING_OWN_FLAGS`) sends every own item of the "
    "op as its flags only, for a rank whose peer discards them (the empty direction of libsircl's pair "
    "exchange); native ABI 9, connection record and wire unchanged | implemented | CPU tests "
    "(`tests/test_link_own_flags.py`: a pair on the verbs stand-in, 12 pieces through 8 slots, the flags-only "
    "rank's payload never reaches its peer and its link bytes posted are 0, every flag arrives and the other "
    "direction is exact; without the bit both directions exact; `tests/test_ring_links.py`: bits 25-31 "
    "refused); in libsircl on ConnectX-7 (two Sparks, nccl-tests v2.21.1, bfloat16 broadcast, reduce, "
    "gather, scatter and all-to-all, 512 KiB to 256 MiB): 133.1 GB per direction against 232.7 GB without "
    "it (NVIDIA NCCL 2.32.3: 133.2 GB), broadcast at 256 MiB 11.00 ms against 11.96 ms (NVIDIA 11.02 ms); "
    "SIRCL's own kernels do not set the bit |\n")


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
        raise SystemExit(f"impl changed in files FO edits since its script was cut: {moved}; run --build --rebase "
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
    print("package files FO replaces or adds:", changed)
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
    shutil.copyfile(IMPL / "STATUS.md", S / "STATUS.md.before-FO")
    for name in record["changed"]:
        (PKG / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TREE / name, PKG / name)
    replace_once(IMPL / "STATUS.md", STATUS_ANCHOR, STATUS_ROW + STATUS_ANCHOR)
    replace_once(IMPL / "STATUS.md", TD_LIBSIRCL, TD_LIBSIRCL_NOW)
    print(f"landed FO: {len(record['changed'])} package files and STATUS.md")


if __name__ == "__main__":
    build("--rebase" in sys.argv) if "--build" in sys.argv else land()
