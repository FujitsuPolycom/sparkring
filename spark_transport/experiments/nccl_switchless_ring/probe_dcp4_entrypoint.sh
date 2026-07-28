#!/usr/bin/env bash
set -euo pipefail

unset NCCL_PROTO

: "${NCCL_SKIP_TREE_CONNECT:?must explicitly enable the switchless patch}"
: "${LD_PRELOAD:?must preload the checksum-pinned patched NCCL library}"

exec python3 /opt/sparkring/probe_dcp4_collectives.py
