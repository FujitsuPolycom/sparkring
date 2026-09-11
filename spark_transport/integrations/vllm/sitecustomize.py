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


if os.getenv("VLLM_SPARK_TP4_MODE", "").lower() not in {"", "disabled"}:
    _install_required("TP4 all-reduce backend", "spark_tp4_backend")

if os.getenv("SPARK_TP4_HEALTH_GATE") == "1":
    _install_required("TP4 post-output health gate", "spark_tp4_health_gate")


if os.getenv("VLLM_SPARK_TP4_VOCAB_MODE"):
    _install_required(
        "TP4 vocabulary all-gather backend",
        "spark_tp4_vocab_allgather_backend",
    )

if os.getenv("SPARK_CUDAGRAPH_REPLAY_TIMING") == "1":
    _install_required(
        "CUDA graph replay timing",
        "spark_cudagraph_replay_timing",
    )

if os.getenv("SPARK_TP4_DCP_COLLECTIVE_AUDIT") == "1":
    _install_required("DCP collective audit", "spark_dcp_collective_audit")
