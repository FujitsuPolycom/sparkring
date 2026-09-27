"""Stage vLLM host-to-device copies through fresh pinned memory.

vLLM's `CpuGpuBuffer.copy_to_gpu` (vllm/v1/utils.py) issues a non-blocking
copy from a pinned host buffer that the next scheduling step rewrites. Under
async scheduling the copy can still be queued when that happens, so it reads
the next step's data. With Qwen3.8-Flash-Next, MTP speculative decoding, async
scheduling and FULL CUDA graphs, this turned about 0.2-0.5% of concurrent
requests into NaN state that decoded as token 8191 repeated
(FujitsuPolycom/sparkring#294). This layer makes the method snapshot the rows
into fresh pinned staging memory before the copy; the caching host allocator
keeps that memory until the copy completes. Without pinned memory the method
is unchanged.

The parent's vllm/v1/utils.py must have SHA-256 5b6015d6...; the result has
SHA-256 194b089a... (both pinned below).
/opt/sparkring/receipts/derived-staging-fix.json records the replaced file.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.images.derived_layer import Layer, main, swap  # noqa: E402

UTILS = "/usr/local/lib/python3.12/dist-packages/vllm/v1/utils.py"
INHERITED = "5b6015d60ea96ab24b0f2e0b7701abaa90d6107bca01b51ac3d4bc361fbdfcff"
RESULT = "194b089acd3f9a4717de88f6ffd2403a3008cc5a418d20996229dbb2c4d74774"
COPY = "        return gpu.copy_(cpu.pin_memory() if PIN_MEMORY else cpu, non_blocking=True)\n"
STAGED_COPY = """\
        if not PIN_MEMORY:
            return gpu.copy_(cpu, non_blocking=True)
        # The host buffer is rewritten for the next step while this copy may
        # still be queued (async scheduling keeps two batches in flight), and a
        # non-blocking copy reads pinned memory when it executes. Snapshot the
        # rows into fresh pinned staging; the caching host allocator keeps it
        # until the copy completes.
        staging = torch.empty_like(cpu, pin_memory=True)
        staging.copy_(cpu)
        return gpu.copy_(staging, non_blocking=True)
"""


def replace(read, receipt):
    return {UTILS: swap(read(UTILS).decode("utf-8"), COPY, STAGED_COPY).encode("utf-8")}


LAYER = Layer(
    name="staging-fix",
    purpose=("CpuGpuBuffer.copy_to_gpu stages each copy through fresh pinned memory, so a host "
             "buffer rewritten for the next step while the copy is still queued under async "
             "scheduling cannot change the copied data (FujitsuPolycom/sparkring#294)"),
    replace=replace,
    provenance="/opt/sparkring/receipts/derived-staging-fix.json",
    pins={UTILS: (INHERITED, RESULT)},
)

if __name__ == "__main__":
    main(LAYER)
