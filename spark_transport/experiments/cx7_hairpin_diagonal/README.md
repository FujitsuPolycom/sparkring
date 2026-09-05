# ConnectX-7 hardware-forwarded opposite-peer paths

Status: **research-only**. `fabric.py` implements topology validation and
command planning for a four-node ring. The native source marker supports
managed attachment until signaled, plus bounded diagnostic runs. The MTP3
profile provides an [implemented managed host service](../../../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md);
its [functional record](../../../performance/records/glm53-flash/spark-mtp3-managed-mesh-functional-20260905.md)
qualifies bounded installer, fault/recovery, reload, and recall cases.
Hot marker replacement beneath active RC QPs and unattended high availability
remain unsupported.

## Packet path

Ranks are physically connected as `0-1-2-3-0`. Each physical port exposes two
Socket Direct PCI functions. An opposite-peer RC packet takes two physical
links, with the intermediate ConnectX-7 forwarding in hardware:

```text
origin RC endpoint
  -> RDMA-TX source marker: EtherType 0x0800 becomes 0x88b5
  -> intermediate ASIC: restore 0x0800, rewrite destination MAC, redirect
  -> opposite-rank RC endpoint
```

Reverse RC responses need the corresponding reverse path. A one-way UDP
test or an `in_hw` rule alone does not establish reliable RC operation.
Only the Ethernet envelope changes; IP, UDP, BTH, and payload bytes are
preserved. Endpoints still use CPU-pinned, GPU-mapped host memory and CPU
posting. Hardware transit through the intermediate ASIC does not provide
GPUDirect RDMA to the endpoint GB10 GPU.

The MTP3 profile uses six origin QPs per rank, two paths per peer, eight
intermediate rules, and eight source-marker processes across four ranks.
Direct-neighbor and opposite-peer traffic share physical link capacity; this
is a logical mesh, not additional physical bandwidth.

## Native source marker

`native/mlx5_rdma_tx_rewrite_probe.c` opens one named mlx5 RDMA device and
installs an RDMA-TX action when `--attach` is supplied. The managed service
owns the following invocation; do not launch it independently beneath a
running model:

```bash
/opt/sparkring/bin/mlx5-rdma-tx-rewrite-probe \
  --device RDMA_DEVICE \
  --source-port 65535 \
  --replacement-ethertype 0x88b5 \
  --attach --managed
```

Use the actual device and executable path from the reviewed site plan.
The source-port match applies to **all** RDMA-TX packets using UDP source
port 65535 on that device. It is not constrained by destination IP or QPN.
The selected devices and source port must be reserved during the test.

On a compatible Linux/ARM64 host with rdma-core development headers, compile
from the repository root:

```bash
mkdir -p build/cx7-marker
cc -O2 -Wall -Wextra \
  spark_transport/experiments/cx7_hairpin_diagonal/native/mlx5_rdma_tx_rewrite_probe.c \
  -o build/cx7-marker/mlx5-rdma-tx-rewrite-probe -libverbs -lmlx5
sha256sum build/cx7-marker/mlx5-rdma-tx-rewrite-probe
```

Record the compiler, source digest, library dependencies, and binary digest.
Use that binary digest in the private site document. A rebuild is not
required to have the observed deployment's binary hash; source compatibility
is not byte-for-byte binary reproducibility. The profile pins record both
source and observed binary identities.

With `--managed`, the process owns the installed rule until it receives
SIGINT/SIGTERM or fails. It reports managed attachment with
`lifetime_seconds: null`; no timer removes the rule. Process exit still
removes the source marker, so the host service coordinates model stopping
and cleanup. The process itself does not restore the site's routes,
neighbors, qdiscs, or intermediate TC state.

For isolated diagnostics, `--run-seconds N` instead bounds attachment to at
most 7,200 seconds. Stop dependent test traffic before expiry and clean up
only owned resources. Bounded mode is not the managed serving default.
A successful build is not a hardware test.

## Planning and read-only verification

[`runtime/glm53-spark-mtp3-mesh/profile.py`](../../../runtime/glm53-spark-mtp3-mesh/profile.py)
renders reviewed per-rank route, neighbor, traffic-control, and native marker
argument arrays. `inspect_fabric.py` checks local identities, GID/MTU,
hardware rule placement, and bounded marker lifetime for diagnostic runs.

The generic `fabric.py` orchestration interfaces expect an external host
helper. Do not execute a generic apply/cleanup plan expecting it to install
the MTP3 mesh. The separate
[managed service](../../../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md)
regenerates canonical network commands, verifies ownership, supervises
persistent markers, and authenticates four-rank readiness. Follow the
[operator quickstart](../../../docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md)
for image, model, and transport configuration.

```bash
python -m pytest spark_transport/experiments/cx7_hairpin_diagonal -q
```

Offline tests validate topology and plan invariants. Required hardware gates
include bidirectional RC payload correctness, absence of software forwarding,
rule/drop/error counters, all-rank startup, and failure containment under
marker or peer loss. Aliased Socket Direct counters must not be summed as
independent unique packets.
