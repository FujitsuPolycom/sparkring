"""Compose cache reuse, checkpoint, attribution, mHC, and startup runtime sources.

The installed file inventory is /opt/sparkring/receipts/mtp3-performance.json.
"""

# ruff: noqa: E402
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent
SITE = Path("/usr/local/lib/python3.12/dist-packages")
context = json.loads((SOURCE / "context.json").read_text())
for name, expected in context["files"].items():
    if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Build input checksum differs: {name}")
sys.path.insert(0, str(SOURCE / "reuse"))
import compose

compose.source_bytes(SITE)
import patch_mtp3_barrier as barrier
import patch_mtp3_lease_accounting as accounting
import patch_mtp3_local_lease_preference as preference
import patch_mtp3_sparse_retention as retention
import patch_mtp3_partial_tail_eligibility as eligibility

scheduler = SITE / "vllm/v1/core/sched/scheduler.py"
barrier.apply_patch(SITE / "b12x/attention/dsa_indexer/fused_indexer.py")
accounting.apply_patch(scheduler)
preference.apply_patch(scheduler)
retention.apply_patch(SITE / "vllm")
eligibility.apply_patch(scheduler)
subprocess.run(
    [
        sys.executable,
        "-S",
        "-B",
        str(SOURCE / "checkpoints/install.py"),
        "apply",
        "--manifest-sha256",
        "0970d29ec33e9f8525a2cc55989ab0deb937bff88b5f16d4d5280035359e4c55",
    ],
    check=True,
)
sys.path.insert(0, str(SOURCE / "reasoning"))
from patch_contract import apply

for name, target in (
    ("serve_with_warmup.py", "serve-with-warmup.py"),
    ("warmup_dflash.py", "warmup_dflash.py"),
    ("startup_admission.py", "startup_admission.py"),
    ("scheduler_liveness.py", "scheduler_liveness.py"),
):
    shutil.copyfile(SOURCE / "startup" / name, Path("/opt/sparkring/bin") / target)
apply(SITE, Path("/opt/sparkring/bin/warmup_dflash.py"))
shutil.copytree(SOURCE / "sparkcache", SITE / "sparkcache", dirs_exist_ok=True)
contract = (
    SITE / "sparkcache/runtime_patches/vllm-manager-page-async-contract-55969c16.json"
)
contract_source = SOURCE / "checkpoints/ownership-contract.json"
if (
    hashlib.sha256(contract_source.read_bytes()).hexdigest()
    != "9fbd8e2596287fd4e12aaf042e189bccfee2ea92126de42f58da8e44fb4d9c55"
):
    raise ValueError("Checkpoint ownership-contract template differs")
data = json.loads(contract_source.read_text())
# Validate every checkpoint ownership digest before applying source transforms.
for row in data["files"]:
    path = SITE / row["path"]
    if row["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"Ownership dependency differs: {row['path']}")
# Apply named transforms only after every checkpoint ownership preimage passed.
import importlib.util

continuation_spec = importlib.util.spec_from_file_location(
    "continuation_install", SOURCE / "continuation/install.py"
)
continuation = importlib.util.module_from_spec(continuation_spec)
continuation_spec.loader.exec_module(continuation)
continuation_transform = continuation.apply(SITE, data)
sys.path.insert(0, str(SOURCE / "attribution"))
from patch_scheduler import apply as apply_attribution

attribution_transform = apply_attribution(scheduler)
for row in data["files"]:
    if row["path"] == "vllm/v1/core/sched/scheduler.py":
        if row["sha256"] != attribution_transform["before_sha256"]:
            raise ValueError("Attribution ownership scheduler preimage differs")
        row["sha256"] = attribution_transform["after_sha256"]
mhc_spec = importlib.util.spec_from_file_location(
    "mhc_prefill_install", SOURCE / "mhc-prefill/install.py"
)
mhc_prefill = importlib.util.module_from_spec(mhc_spec)
mhc_spec.loader.exec_module(mhc_prefill)
mhc_transform = mhc_prefill.apply(SITE, data)
contract.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
shutil.copytree(SOURCE / "bundle", Path("/opt/spark-sircl"), dirs_exist_ok=True)
shutil.copyfile(
    SOURCE / "libspark_cache_placement.so",
    Path(
        "/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so"
    ),
)
files = {}
shutil.copyfile(SOURCE / "verify.py", Path("/opt/sparkring/bin/verify-performance.py"))
shutil.copyfile(SOURCE / "start.py", Path("/opt/sparkring/bin/start-performance.py"))
for root in (
    SITE / "sparkcache",
    SITE / "vllm",
    SITE / "b12x",
    Path("/opt/spark-sircl"),
):
    for path in root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
for path in (
    Path("/opt/sparkring/bin/verify-performance.py"),
    Path("/opt/sparkring/bin/start-performance.py"),
    Path("/opt/sparkring/bin/warmup_dflash.py"),
    Path("/opt/sparkring/bin/serve-with-warmup.py"),
    Path("/opt/sparkring/bin/startup_admission.py"),
    Path("/opt/sparkring/bin/scheduler_liveness.py"),
    Path(
        "/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so"
    ),
):
    files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
receipt = Path("/opt/sparkring/receipts/mtp3-performance.json")
receipt.write_text(
    json.dumps(
        {
            "schema": "sparkring-mtp3-performance-image/v1",
            "status": "research-only",
            "sparkcache_commit": context["sparkcache_commit"],
            "runtime_transforms": {
                "continuation_checkpoints": continuation_transform,
                "request_cache_attribution": attribution_transform,
                "token_sharded_mhc_prefill": mhc_transform,
            },
            "files": files,
        },
        sort_keys=True,
    )
    + "\n"
)
