#!/usr/bin/env bash
set -euo pipefail

# The GLM image pins NCCL_PROTO=Simple.  Remove that image default so NCCL can
# tune the patched ring and the direct two-rank DCP communicators independently.
unset NCCL_PROTO

: "${NCCL_SKIP_TREE_CONNECT:?must explicitly enable the switchless patch}"
: "${LD_PRELOAD:?must preload the checksum-pinned patched NCCL library}"

exec python3 /opt/sparkring/probe_dcp2_collectives.py
