"""Opt-in startup hooks for Spark vLLM transport experiments."""

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


def _install_adaptive_controller() -> None:
    from adaptive_mtp_controller.runtime_installer import (
        install,
        runtime_installation_snapshot,
    )

    install()
    snapshot = runtime_installation_snapshot()
    if not snapshot["installed"]:
        raise RuntimeError(
            "adaptive-MTP controller returned without owning all runtime hooks: "
            f"{snapshot}"
        )


def _install_mtp_index_reuse() -> None:
    from spark_glm52_mtp_index_reuse import install

    result = install()
    if getattr(result, "status", None) != "installed":
        raise RuntimeError(f"MTP index-reuse installation was not admitted: {result}")


def _install_true_adaptive_draft() -> None:
    from spark_true_adaptive_draft import (
        install,
        true_adaptive_draft_snapshot,
    )

    if install() is not True:
        raise RuntimeError("true adaptive drafting did not report installation")
    snapshot = true_adaptive_draft_snapshot()
    if not snapshot["installed"]:
        raise RuntimeError(
            "true adaptive drafting returned without owning runtime hooks"
        )


def _install_nf3_startup_profile_cap() -> None:
    from spark_nf3_startup_profile_cap import (
        install,
        startup_profile_cap_snapshot,
    )

    install()
    snapshot = startup_profile_cap_snapshot()
    if not snapshot["installed"]:
        raise RuntimeError(
            "NF3 startup-profile cap returned without owning profile_run"
        )
    raw_runner_kind = os.getenv("VLLM_USE_V2_MODEL_RUNNER")
    if raw_runner_kind not in {"0", "1"}:
        raise RuntimeError(
            "NF3 startup-profile cap requires explicit "
            "VLLM_USE_V2_MODEL_RUNNER=0 or 1"
        )
    expected_runner_kind = "v2" if raw_runner_kind == "1" else "v1"
    if snapshot["runner_kind"] != expected_runner_kind:
        raise RuntimeError(
            "NF3 startup-profile cap selected the wrong runner: "
            f"expected={expected_runner_kind}, "
            f"actual={snapshot['runner_kind']}"
        )
    expected_v1_owned = expected_runner_kind == "v1"
    expected_v2_owned = expected_runner_kind == "v2"
    if (
        snapshot["v1_runner_owned"] is not expected_v1_owned
        or snapshot["v2_runner_owned"] is not expected_v2_owned
    ):
        raise RuntimeError(
            "NF3 startup-profile cap runner ownership is invalid: "
            f"{snapshot}"
        )
    if not snapshot["memory_ownership_guard_installed"]:
        raise RuntimeError(
            "NF3 startup-profile cap returned without owning "
            "Worker.determine_available_memory"
        )
    if os.getenv("VLLM_SPARK_NF3_SINGLE_COMPILE_RANGE") == "1":
        if not snapshot["compile_warmup_installed"]:
            raise RuntimeError(
                "NF3 single-range contract returned without owning "
                "Worker._compile_or_warm_up_model_impl"
            )


def _install_nf3_workspace_reserve() -> None:
    from spark_nf3_workspace_reserve import (
        install,
        workspace_reserve_snapshot,
    )

    install()
    snapshot = workspace_reserve_snapshot()
    expected_profile = os.getenv("VLLM_SPARK_NF3_PROFILE")
    raw_reserve = os.getenv(
        "VLLM_SPARK_NF3_WORKSPACE_RESERVE_BYTES"
    )
    expected_reserve = int(raw_reserve) if raw_reserve else None
    if (
        not snapshot["installed"]
        or not snapshot["owned"]
        or snapshot["profile"] != expected_profile
        or snapshot["reserve_bytes"] != expected_reserve
    ):
        raise RuntimeError(
            "NF3 workspace reserve returned without owning the exact "
            f"model_runner lock binding: {snapshot}"
        )


if os.getenv("VLLM_SPARK_NF3_STARTUP_PROFILE_MAX_TOKENS"):
    _install_required(
        "NF3 startup-profile cap",
        _install_nf3_startup_profile_cap,
    )


_NF3_REFERENCE_PROFILES = {
    "reference-four-spark",
    "reference-four-spark-adaptive-2-4",
    "reference-four-spark-adaptive-2-4-c8",
}
if (
    os.getenv("VLLM_SPARK_NF3_PROFILE") in _NF3_REFERENCE_PROFILES
    or os.getenv("VLLM_SPARK_NF3_WORKSPACE_RESERVE_BYTES")
):
    _install_required(
        "NF3 workspace reserve",
        _install_nf3_workspace_reserve,
    )


if os.getenv("SPARK_ADAPTIVE_MTP_CONTROL") == "1":
    _install_required("adaptive-MTP controller", _install_adaptive_controller)

# True selected-depth drafting must wrap this already-composed proposer.
if os.getenv("SPARK_GLM52_MTP_INDEX_REUSE") == "1":
    _install_required("GLM-5.2 MTP index reuse", _install_mtp_index_reuse)

if os.getenv("VLLM_SPARK_TRUE_ADAPTIVE_DRAFT") == "1":
    _install_required("true adaptive drafting", _install_true_adaptive_draft)

if os.getenv("SPARK_Q2R_PROBE") == "1":
    from spark_q2r_probe_bridge import install as install_q2r_probe

    install_q2r_probe()

if os.getenv("SPARK_TP4_FLIGHT_RECORDER") == "1":
    from spark_tp4_flight_recorder import install_b12x_import_hook

    install_b12x_import_hook()

if os.getenv("SPARK_CUDAGRAPH_REPLAY_TIMING") == "1":
    from spark_cudagraph_replay_timing import (
        install as install_cudagraph_replay_timing,
    )

    install_cudagraph_replay_timing()

if os.getenv("VLLM_SPARK_TP4_MODE"):
    from spark_tp4_backend import install as install_tp4

    _install_required("TP4 all-reduce backend", install_tp4)

if os.getenv("VLLM_SPARK_TP4_ALLGATHER_MODE"):
    from spark_tp4_allgather_backend import install as install_tp4_allgather

    _install_required("TP4 all-gather backend", install_tp4_allgather)

if os.getenv("VLLM_SPARK_TP4_VOCAB_MODE"):
    from spark_tp4_vocab_allgather_backend import (
        install as install_tp4_vocab_allgather,
    )

    _install_required(
        "TP4 vocabulary all-gather backend",
        install_tp4_vocab_allgather,
    )

if os.getenv("VLLM_SPARK_TP4_DCP_MODE"):
    from spark_tp4_dcp_backend import install as install_tp4_dcp

    _install_required("TP4 DCP backend", install_tp4_dcp)

if os.getenv("SPARK_TP4_DCP_COLLECTIVE_AUDIT") == "1":
    from spark_dcp_collective_audit import (
        install as install_dcp_collective_audit,
    )

    install_dcp_collective_audit()

if os.getenv("VLLM_SPARK_TRACE_ALLREDUCE") == "1":
    from spark_shape_trace import install as install_trace

    install_trace()
