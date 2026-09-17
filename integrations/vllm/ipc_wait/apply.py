"""Add an optional SHM reader spin interval to two identified vLLM sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENVIRONMENT = "SPARKRING_SHM_BUSY_LOOP_S"
SIGNATURE = b"        busy_loop_s: float = 1,\n"
ASSIGNMENT = b"            self.busy_loop_s = busy_loop_s\n"
REPLACEMENT = b'''            if busy_loop_s is None:
                import os

                # Tune only the idle wait policy; explicit callers retain control.
                value = os.environ.get("SPARKRING_SHM_BUSY_LOOP_S", "1")
                try:
                    busy_loop_s = float(value)
                except ValueError as error:
                    raise ValueError(
                        "SPARKRING_SHM_BUSY_LOOP_S must be finite and between 0 and 1 seconds"
                    ) from error
                if not 0 <= busy_loop_s <= 1:
                    raise ValueError(
                        "SPARKRING_SHM_BUSY_LOOP_S must be finite and between 0 and 1 seconds"
                    )
            self.busy_loop_s = busy_loop_s
'''


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def patch_source(raw):
    records = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))["sources"]
    observed = digest(raw)
    record = next((item for item in records
                   if observed in (item["source_sha256"], item["result_sha256"])), None)
    if record is None:
        raise ValueError("SHM broadcast source is not an admitted preimage or result")
    if observed == record["result_sha256"]:
        return raw, record
    if raw.count(SIGNATURE) != 1 or raw.count(ASSIGNMENT) != 1:
        raise ValueError("Expected one SpinCondition default and reader assignment")
    result = raw.replace(SIGNATURE, b"        busy_loop_s: float | None = None,\n")
    result = result.replace(ASSIGNMENT, REPLACEMENT)
    if digest(result) != record["result_sha256"]:
        raise ValueError("SHM broadcast output differs from its recorded result")
    compile(result, record["path"], "exec")
    return result, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and args.receipt is None:
        parser.error("Applying the patch requires a new receipt path")
    if args.source.is_symlink():
        parser.error("Refusing to replace a source symlink")
    if args.receipt and args.receipt.exists():
        parser.error("Refusing to overwrite an existing receipt")
    before = args.source.read_bytes()
    after, record = patch_source(before)
    receipt = dict(schema="sparkring-ipc-wait-patch/v1", source_name=record["name"],
                   path=record["path"], preimage_sha256=digest(before),
                   result_sha256=digest(after), environment=ENVIRONMENT,
                   default_seconds=1.0, status="research-only")
    if not args.check:
        temporary = args.source.with_name(args.source.name + ".ipc-wait-tmp")
        with temporary.open("xb") as handle:
            handle.write(after)
        try:
            if args.source.read_bytes() != before:
                raise ValueError("SHM source changed during patch preparation")
            os.chmod(temporary, args.source.stat().st_mode)
            os.replace(temporary, args.source)
        finally:
            temporary.unlink(missing_ok=True)
        with args.receipt.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
            handle.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
