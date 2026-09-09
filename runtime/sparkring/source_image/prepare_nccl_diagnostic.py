"""Prepare a compile-only NCCL image that cannot produce a serving receipt."""
import argparse
import json
from pathlib import Path

from archive_utils import make_archive, sha
from verify_image import HERE, IMAGE_ROOT, trusted_closure


def prepare(context, output):
    if output.exists():
        raise ValueError("Diagnostic output directory must be absent")
    lock_bytes = (HERE / "glm53-tp4-lock.json").read_bytes()
    trusted_closure(context, lock_bytes)
    lock = json.loads(lock_bytes)
    source_receipt = json.loads((context / "context-receipt.json").read_bytes())
    payload = (context / "payload.tar").read_bytes()
    if sha(payload) != source_receipt["payload_sha256"]:
        raise ValueError("Diagnostic payload differs from prepared source context")
    entrypoint = ["python3", "-S", "-B", IMAGE_ROOT + "/build_nccl.py",
                  "--diagnostic-output", "/work/nccl-diagnostic.json"]
    docker = (f"FROM {lock['parent']['reference']}\nUSER root\nADD payload.tar /\n"
              'LABEL org.sparkring.nccl.diagnostic="true"\n'
              'LABEL org.sparkring.runtime.status="research-only"\n'
              "ENTRYPOINT " + json.dumps(entrypoint) + "\n").encode()
    archive = make_archive({"Dockerfile": (docker, 0o644), "payload.tar": (payload, 0o644)},
                           lock["source_date_epoch"])
    output.mkdir(parents=True)
    (output / "Dockerfile").write_bytes(docker)
    (output / "payload.tar").write_bytes(payload)
    (output / "image-context.tar").write_bytes(archive)
    receipt = {"schema": "sparkring-nccl-diagnostic-context/v1", "status": "research-only",
               "source_lock_sha256": sha(lock_bytes), "source_context_sha256": source_receipt["context_sha256"],
               "context_sha256": sha(archive), "payload_sha256": sha(payload),
               "runtime_receipt_eligible": False, "gpu_qualified": False,
               "output_receipt": "/work/nccl-diagnostic.json",
               "output_library": "/work/build/lib/libnccl.so.2.30.7",
               "output_build_log": IMAGE_ROOT + "/nccl-build-log.json"}
    (output / "diagnostic-context-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.context, args.output), indent=2))
