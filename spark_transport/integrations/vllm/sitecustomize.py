"""Install the vLLM adapters used by supported SparkRing profiles."""

import importlib
import os
import sys
import traceback
from typing import Any


def _install_required(label: str, module_name: str) -> Any:
    """Install one enabled hook or terminate before vLLM can serve traffic.

    CPython's ``site`` module reports and suppresses ordinary exceptions raised
    while importing ``sitecustomize``.  Explicitly exiting the process here is
    therefore part of the feature contract, not merely nicer error handling.
    """

    try:
        return importlib.import_module(module_name).install()
    except BaseException:
        try:
            print(
                f"FATAL: required Spark startup hook failed: {label}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        finally:
            # Even a closed/broken stderr must not let CPython's site module
            # suppress this required-hook failure and continue serving.
            os._exit(78)
        raise RuntimeError("os._exit unexpectedly returned")


def _four_rank_enabled() -> bool:
    """``SPARK_TP4_ENABLED``: ``0`` turns every four-rank hook off; ``1`` or unset leaves each to its variable.

    A launcher that serves on another transport, such as SIRCL's ring
    sessions, sets ``0`` so that no four-rank hook installs whatever the
    profile's other variables say. Any other value stops the process: ranks
    that read the switch differently would split one collective between two
    transports.
    """

    value = os.getenv("SPARK_TP4_ENABLED", "").strip()
    if value in {"", "1"}:
        return True
    if value == "0":
        return False
    try:
        print(
            f"FATAL: SPARK_TP4_ENABLED must be 0 or 1, got {value!r}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        os._exit(78)
    raise RuntimeError("os._exit unexpectedly returned")


_FOUR_RANK = _four_rank_enabled()

if _FOUR_RANK and os.getenv("VLLM_SPARK_TP4_MODE", "").lower() not in {"", "disabled"}:
    _install_required("TP4 all-reduce backend", "spark_tp4_backend")

if _FOUR_RANK and os.getenv("SPARK_TP4_HEALTH_GATE") == "1":
    _install_required("TP4 post-output health gate", "spark_tp4_health_gate")


if _FOUR_RANK and os.getenv("VLLM_SPARK_TP4_VOCAB_MODE"):
    _install_required(
        "TP4 vocabulary all-gather backend",
        "spark_tp4_vocab_allgather_backend",
    )

if os.getenv("SPARK_CUDAGRAPH_REPLAY_TIMING") == "1":
    _install_required(
        "CUDA graph replay timing",
        "spark_cudagraph_replay_timing",
    )

if _FOUR_RANK and os.getenv("SPARK_TP4_DCP_COLLECTIVE_AUDIT") == "1":
    _install_required("DCP collective audit", "spark_dcp_collective_audit")
