"""Qualify basic GB10 execution and single-rank NCCL ownership for the R33 foundation."""

import hashlib
import json
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist


assert torch.version.git_version == "cf30153c4c131c8164ee7798e5022d810682e2cb"
assert torch.version.cuda == "13.3"
assert torch.cuda.is_available()
device = torch.device("cuda:0")
assert torch.cuda.get_device_capability(device) == (12, 1)

a = torch.arange(4096, dtype=torch.bfloat16, device=device).reshape(64, 64)
b = torch.eye(64, dtype=torch.bfloat16, device=device)
result = a @ b
torch.cuda.synchronize(device)
assert torch.equal(result.cpu(), a.cpu())

with tempfile.TemporaryDirectory(prefix="sparkring-r33-nccl-") as root:
    dist.init_process_group(
        "nccl", init_method=f"file://{root}/store", rank=0, world_size=1
    )
    value = torch.tensor([37.0], device=device)
    dist.all_reduce(value)
    torch.cuda.synchronize(device)
    assert value.item() == 37.0
    dist.destroy_process_group()

mapped = sorted(
    {
        str(Path(line.split()[-1]).resolve())
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libnccl.so" in line
    }
)
assert len(mapped) == 1, mapped
library = Path(mapped[0])
with library.open("rb") as handle:
    library_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
print(
    json.dumps(
        {
            "status": "qualified-single-gpu-single-rank-nccl",
            "torch": torch.__version__,
            "torch_source": torch.version.git_version,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "nccl_library": str(library),
            "nccl_sha256": library_sha256,
            "limits": "One GB10, one process and one-rank collective; no RDMA or serving qualification.",
        },
        indent=2,
    )
)
