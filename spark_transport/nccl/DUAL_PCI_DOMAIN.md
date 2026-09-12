# NCCL routing across two host PCIe domains

Status: **Development** source patch; **Experimental** shared-image profile.
The build manifest records a measured library hash. The
[shared GLM source builder](../../runtime/sparkring/source_image/README.md)
consumes the cumulative patch; its rebuilt library requires separate
qualification.

Each Spark exposes NIC functions through two PCIe root domains. Advertising
only two listener GIDs can omit reachable secondary-domain functions. A global
first-reachable fallback can also select a primary-domain function when the
intended function belongs to the secondary domain.

The [cumulative patch](nccl-2.30.7-dual-pci-domain.patch) provides bounded four-IPv4-GID publication
and a PCI-root-preserving fallback. It includes the switchless cycle changes;
apply it alone to the unmodified revision in the [build manifest](dual-pci-domain.json).
Do not layer it over the separate
[switchless-cycle patch](nccl-2.30.7-switchless-cycle.patch),
[Tree/PAT patch](nccl-2.30.7-skip-tree-pat.patch), or
[two-GID patch](nccl-2.30.7-advertise-all-listener-gids.patch).
The flags default off. Generic InfiniBand and IPv6
retain legacy publication. Unknown handle formats are rejected. Compatible
legacy handles do not read the unused tail bytes.

## Offline source preparation

Check out NVIDIA NCCL revision `73cf112295c33aee2b895f329f592f2a9b4b0f97`
with LF line endings. Verify the patch hash against `dual-pci-domain.json`.
From that source checkout, apply the cumulative patch and run:

```sh
g++ -std=c++11 -O2 -Wall -Wextra -Werror tests/routing_handle/compat.cc -o routing-handle-test
./routing-handle-test
: "${CUDA_HOME:?Set CUDA_HOME to the installed SM121-capable toolkit root}"
: "${CUDA_LIB:?Set CUDA_LIB to that toolkit's library directory}"
make -j8 src.build CUDA_HOME="$CUDA_HOME" CUDA_LIB="$CUDA_LIB" NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121'
```

Keep NCCL LICENSE.txt and ThirdPartyNotices.txt with distributed binaries.
The measured library used CUDA 13.3 as recorded in the build manifest.
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
Retain the [four-rank cycle environment](README.md#four-rank-cycle), including
subnet-aware routing and the no-Tree connection setting.
Use INFO NET connection logs to verify final connected QPs use both domains
on every rank. HCA discovery alone does not prove effective selection. QP
counts establish connection placement, not equal byte distribution. Keep
logging out of timed component loops.

The patch does not change the collective algorithm, add ASIC forwarding, or
make shared ring links independent. It affects residual NCCL collectives; the
custom mesh transport has separate routing and submission code.

## Validation boundary

`git apply --check` establishes source applicability; the handle test checks
CPU compatibility. Neither qualifies a compiled NCCL library or serving image.
The manifest's measured-library hash identifies prior bytes, but this page
does not provide raw cluster receipts that establish their serving scope.
Qualify a rebuild against its own binary and image receipts before publishing
performance claims. Keep these routing settings specific to the selected
profile, topology and NCCL version.
