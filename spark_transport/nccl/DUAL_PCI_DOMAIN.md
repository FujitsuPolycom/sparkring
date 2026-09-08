# NCCL routing across two host PCIe domains

Status: implemented source patch; research-only deployment profile. Four DGX
Spark ranks have passed collective correctness and serving measurements with
this routing, but the public image builder does not consume this patch yet.

Each Spark exposes NIC functions through two PCIe root domains. Advertising
only two listener GIDs can omit reachable secondary-domain functions. A global
first-reachable fallback can also select a primary-domain function when the
intended function belongs to the secondary domain.

`nccl-2.30.7-dual-pci-domain.patch` provides bounded four-IPv4-GID publication
and a PCI-root-preserving fallback. It includes the switchless cycle changes;
apply it to the unmodified revision in `dual-pci-domain.json`, not after the
existing switchless patch. The flags default off. Generic InfiniBand and IPv6
retain legacy publication. Unknown handle formats are rejected. Compatible
legacy handles do not read the unused tail bytes.

## Offline source preparation

Check out NVIDIA NCCL revision `73cf112295c33aee2b895f329f592f2a9b4b0f97`
with LF line endings. Verify the patch hash against `dual-pci-domain.json`.
From that source checkout, apply the cumulative patch and run:

```sh
g++ -std=c++11 -O2 -Wall -Wextra -Werror tests/routing_handle/compat.cc -o routing-handle-test
./routing-handle-test
make -j8 src.build CUDA_HOME=/opt/cuda-13.3 CUDA_LIB=/opt/cuda-13.3/lib NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121'
```

Keep NCCL LICENSE.txt and ThirdPartyNotices.txt with distributed binaries.
The measured binary hash is provenance, not a guarantee of a byte-identical
rebuild across toolchains. Record the produced binary, compiler, source tree,
patch, CUDA version and image digest in the build receipt.

## Four-Spark cycle selection

The site configuration must enumerate the two neighbor-facing functions under
each host domain and validate the direct-link subnets. Four functions share two
physical NIC ports; this does not create four cables.

```text
NCCL_IB_EXTENDED_IPV4_GIDS=1
NCCL_IB_PRESERVE_PCI_DOMAIN=1
NCCL_IB_ROUTE_DIAGNOSTICS=1
NCCL_ALGO=Ring
NCCL_PROTO=LL,LL128,Simple
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4
NCCL_IB_QPS_PER_CONNECTION=1
```

Both `LD_PRELOAD` and `VLLM_NCCL_SO_PATH` must reference the verified library.
Retain the existing switchless subnet and no-Tree connection contract.
Use INFO NET connection logs to verify final connected QPs use both domains
on every rank. HCA discovery alone does not prove effective selection. QP
counts establish connection placement, not equal byte distribution. Keep
logging out of timed component loops.

The patch does not change the collective algorithm, add ASIC forwarding, or
make shared ring links independent. It affects residual NCCL collectives; the
custom mesh transport has separate routing and submission code.

## Validation boundary

A fresh LF checkout passed `git apply --check` for the packaged patch. Existing
cluster receipts qualify the measured library with TP4/DCP4/MTP3, B12X KDA,
continuation coalescing and token-sharded mHC. Import sanitized raw receipts
before publishing performance claims. Rebuild and image qualification remain
required for the public packaging. Do not turn these settings into defaults
for every pair, model or NCCL version.
