"""Compose the mounted SIRCL hooks with the virtual-diagonal adapter.

Status: research-only. The private bundle builder installs this file as the
process ``sitecustomize`` and preserves the SIRCL bundle's entry point as
``sircl_sitecustomize.py``.
"""

from __future__ import annotations

import os
import runpy
import sys
import traceback
from pathlib import Path


BASE_SITECUSTOMIZE = Path("/opt/spark-sircl/sircl_sitecustomize.py")


def _required(label: str, operation) -> None:
    try:
        operation()
    except BaseException:
        try:
            print(
                f"FATAL: required private startup hook failed: {label}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        finally:
            os._exit(78)


def _install() -> None:
    if os.getenv("SPARK_TP4_HEALTH_GATE") != "1":
        raise RuntimeError(
            "virtual-diagonal full-model testing requires SPARK_TP4_HEALTH_GATE=1"
        )
    if not BASE_SITECUSTOMIZE.is_file():
        raise RuntimeError(
            f"preserved SIRCL sitecustomize is missing: {BASE_SITECUSTOMIZE}"
        )

    # The base entry point installs every receipt-bound SIRCL, vocabulary,
    # timing, audit, and health hook. The B12X wrapper uses a distinct marker
    # and is installed outside those wrappers.
    runpy.run_path(str(BASE_SITECUSTOMIZE), run_name="_sparkring_sircl_sitecustomize")

    from rocenante_vllm_overlay import install as install_rocenante

    install_rocenante()
    from rocenante_health_gate import install as install_rocenante_health

    install_rocenante_health()


_required("SIRCL and RoCEnante composition", _install)
