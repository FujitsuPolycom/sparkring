"""Install the vLLM adapters used by supported SparkRing profiles."""

import os
import sys
import traceback
from collections.abc import Callable
from typing import Any


def _install_required(label: str, operation: Callable[[], Any]) -> Any:
    """Install one enabled hook or terminate before vLLM can serve traffic.

    CPython's ``site`` module reports and suppresses ordinary exceptions raised
    while importing ``sitecustomize``.  Explicitly exiting the process here is
    therefore part of the feature contract, not merely nicer error handling.
    """

    try:
        return operation()
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


if os.getenv("VLLM_SPARK_TP4_MODE"):
    from spark_tp4_backend import install as install_tp4

    _install_required("TP4 all-reduce backend", install_tp4)

if os.getenv("SPARK_TP4_HEALTH_GATE") == "1":
    from spark_tp4_health_gate import install as install_tp4_health_gate

    _install_required("TP4 post-output health gate", install_tp4_health_gate)


if os.getenv("VLLM_SPARK_TP4_VOCAB_MODE"):
    from spark_tp4_vocab_allgather_backend import (
        install as install_tp4_vocab_allgather,
    )

    _install_required(
        "TP4 vocabulary all-gather backend",
        install_tp4_vocab_allgather,
    )

if os.getenv("SPARK_CUDAGRAPH_REPLAY_TIMING") == "1":
    from spark_cudagraph_replay_timing import (
        install as install_cudagraph_replay_timing,
    )

    _install_required(
        "CUDA graph replay timing",
        install_cudagraph_replay_timing,
    )

if os.getenv("SPARK_TP4_DCP_COLLECTIVE_AUDIT") == "1":
    from spark_dcp_collective_audit import (
        install as install_dcp_collective_audit,
    )

    _install_required("DCP collective audit", install_dcp_collective_audit)
