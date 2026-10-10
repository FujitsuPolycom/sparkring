"""The catalog of SIRCL's version-pinned vLLM shims, and each shim's status in a vLLM tree.

:data:`ENTRIES` holds, per shim of :mod:`.shims`, what a reader needs to pick
it: its purpose, the models it serves, the vLLM code it wraps or calls, how it
is enabled and disabled, what happens without it (with a measured cost where
one exists), and what it needs from the ring session. The pinned files, the
verified builds and their file hashes come from :mod:`.pins`, and the hook
anchors from :mod:`.hooks`. :func:`document` joins them into the catalog that
``shims.json`` (next to this module) holds; ``python -m
sparkring_sircl.vllm.catalog --write`` regenerates that file, and a CPU test
compares it with :func:`document`.

:func:`tree_status` reads a vLLM package directory and gives each shim one of
three statuses:

- ``verified``: the files the shim touches match a pinned build, so the shim
  installs when its condition holds;
- ``applicable-unverified``: those files, the functions the shim wraps or
  calls and its hook anchors are present, but no pinned build matches the
  files, so the shim refuses to install until a build that matches is pinned
  (``README.md``, section "Shims");
- ``absent``: a file, function or anchor the shim relies on is missing.

Nothing here imports vLLM or torch; the checks read files only.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import hooks, pins, shims

SCHEMA = "sircl-vllm-shims/v1"
STATUS_SCHEMA = "sircl-vllm-shim-status/v1"
CATALOG_FILE = Path(__file__).with_name("shims.json")
STATUSES = {
    "verified": "the files it touches match a pinned build; it installs when its condition holds",
    "applicable-unverified": "its files, functions and anchors are present but match no pinned build; it "
                             "refuses to install until a matching build is pinned",
    "absent": "a file, function or anchor it relies on is missing; it cannot install",
}


@dataclasses.dataclass(frozen=True)
class Code:
    """A vLLM definition a shim relies on: ``wraps`` (replaced or wrapped), ``calls`` (called or checked) or
    ``binding`` (an imported name the shim rebinds)."""

    file: str            # relative to the vllm package
    name: str            # function, Class, Class.method, or the imported name of a binding
    role: str


@dataclasses.dataclass(frozen=True)
class Measurement:
    conditions: str
    result: str


@dataclasses.dataclass(frozen=True)
class Entry:
    name: str
    purpose: str
    models: tuple[str, ...]
    code: tuple[Code, ...]
    enable: str
    disable: str
    without: str
    needs: tuple[str, ...]
    measured: tuple[Measurement, ...] = ()


DCP = "v1/attention/ops/dcp.py"
MHC = "models/glm5next/nvidia/mhc_prefill_sharding.py"
QWEN_HC = "models/qwen4_exp/nvidia/hc_prefill.py"
WORKER = "v1/worker/gpu_worker.py"
NORM = "models/common/ops/fused_allreduce_rms_norm.py"

ENTRIES: tuple[Entry, ...] = (
    Entry(
        name="dcp_all_to_all",
        purpose="Carries vLLM's decode-context-parallel output combine (the LSE-weighted all-to-all) through "
                "SIRCL's communicator instead of torch.distributed on the group's NCCL device group.",
        models=("MLA attention with decode context parallelism and the a2a combine (dcp_comm_backend a2a, "
                "vLLM's default for GLM-5.3): GLM-5.3 and other models on vLLM's DeepSeek-V3.2 code",),
        code=(Code(DCP, "dcp_a2a_lse_reduce", "wraps"), Code(DCP, "_dcp_a2a_lse_pack_dim", "calls"),
              Code(DCP, "_dcp_a2a_send_recv_buffers", "calls"), Code(DCP, "_dcp_a2a_pack_send", "calls"),
              Code(DCP, "_dcp_a2a_unpack_combine", "calls")),
        enable="SIRCL's communicator installs it when it builds a decode-context-parallel group of two or more "
               "ranks with a session (SIRCL_GROUPS=tp,dcp; bundle --session-groups tp,dcp)",
        disable="SIRCL_GROUPS=tp: decode-context-parallel groups get no session",
        without="vLLM's combine calls torch.distributed.all_to_all_single on the group's NCCL device group: "
                "where NCCL may not connect the ranks, SIRCL's tripwire refuses the call; where it may, NCCL "
                "carries it. A decode-context-parallel group with a session fails setup on every rank when the "
                "shim cannot install",
        needs=("the session's all-to-all (scatter collectives, prepare(..., scatter=True))",),
    ),
    Entry(
        name="dcp_b12x_transport",
        purpose="Withholds vLLM's B12X PCIe decode-context-parallel transport, which maps peer memory within one "
                "machine, from groups SIRCL owns, so the MLA DCP manager keeps the a2a combine and the group "
                "coordinator's query all-gather.",
        models=("MLA attention with decode context parallelism, the B12X attention backend and "
                "VLLM_USE_B12X_DCP_A2A=1 (GLM-5.3)",),
        code=(Code("distributed/device_communicators/b12x_dcp.py", "get_b12x_dcp_transport", "wraps"),
              Code(DCP, "MLADCPManager.__init__", "calls")),
        enable="installed with dcp_all_to_all when SIRCL's communicator builds a decode-context-parallel group "
               "of two or more ranks with a session",
        disable="SIRCL_GROUPS=tp: decode-context-parallel groups get no session",
        without="vLLM selects the B12X PCIe transport, whose CUDA IPC handles work within one machine only; a "
                "decode-context-parallel group with a session fails setup on every rank when the shim cannot "
                "install",
        needs=("dcp_all_to_all, installed with it",),
    ),
    Entry(
        name="fused_allreduce_rms_norm",
        purpose="Runs vLLM's eager all-reduce + residual add + RMSNorm helper as one SIRCL collective where the "
                "result is bit-identical to SIRCL's all-reduce followed by vLLM's vllm_c fused_add_rms_norm "
                "kernel.",
        models=("models whose decoder layers, final norm or MTP draft call vLLM's fused_allreduce_rms_norm "
                "helper: GLM-5.3 and other models on vLLM's DeepSeek-V3.2 code",),
        code=(Code(NORM, "fused_allreduce_rms_norm", "wraps"),
              Code("distributed/parallel_state.py", "get_tp_group", "calls"),
              Code("model_executor/layers/layernorm.py", "RMSNorm", "calls")),
        enable="SIRCL_FUSED_NORM=1 (bundle --fused-norm on), research-only: SIRCL's communicator installs it "
               "when it binds the fused kernels to a tensor-parallel session",
        disable="SIRCL_FUSED_NORM=0, the default; the serve launcher keeps it off",
        without="the helper's all-reduce runs through SIRCL's communicator and vLLM's RMSNorm follows",
        needs=("the tensor-parallel session's fused all-reduce + residual add + RMSNorm kernels "
               "(sparkring_sircl.fused_norm)", "vLLM's RMSNorm on the vllm_c provider of fused_add_rms_norm"),
        measured=(Measurement("decode at TP8 on the ring of eight, 4 streams, 1 MiB dispatch ceiling",
                              "41.3 steps/s, 9 % more than with SIRCL_FUSED_NORM=0"),),
    ),
    Entry(
        name="mhc_prefill_shard",
        purpose="Carries GLM-5.3-Flash's mHC prefill row ownership, whose reduce-scatters and all-gathers call "
                "PyNccl directly, through SIRCL's communicator on a tensor-parallel group without PyNccl.",
        models=("GLM-5.3-Flash (model type glm5_next) with VLLM_GLM53_MHC_PREFILL_SHARD=1",),
        code=(Code(MHC, "maybe_create", "wraps"),
              Code("models/glm5next/nvidia/model.py", "maybe_create_mhc_prefill_ownership", "binding"),
              Code(MHC, "PrefillOwnership.reduce_scatter", "calls"),
              Code(MHC, "PrefillOwnership.all_gather", "calls")),
        enable="SIRCL's communicator installs it when it builds a tensor-parallel group without PyNccl, with a "
               "session, while VLLM_GLM53_MHC_PREFILL_SHARD is set (the profile's value; serve "
               "--mhc-prefill-shard profile)",
        disable="serve --mhc-prefill-shard off (VLLM_GLM53_MHC_PREFILL_SHARD=0)",
        without="with the setting on, setup fails on every rank and names VLLM_GLM53_MHC_PREFILL_SHARD=0; with "
                "it off, every rank mixes the mHC streams of every prefill row. With NCCL off (SIRCL_NCCL=never, "
                "the default) the shim serves a pair as well; only where SIRCL_NCCL=topology or auto lets NCCL "
                "run, on a pair, does vLLM's own PyNccl path run without it",
        needs=("the tensor-parallel session's reduce-scatter (scatter_available) and all-gather; without the "
               "session's reduce-scatter the adapter carries each reduce-scatter as an all-reduce and keeps this "
               "rank's rows",),
        measured=(Measurement("GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD, TP4 on the path of Sparks 0-3",
                              "prefill 12-14 % faster than with --mhc-prefill-shard off"),
                  Measurement("GLM-5.3-Flash-NVFP4-Spark, TP4 on the path of Sparks 0-3",
                              "prefill 3.7 % faster than with --mhc-prefill-shard off")),
    ),
    Entry(
        name="qwen_hc_prefill_shard",
        purpose="Carries Qwen3.8's hyper-connection prefill row ownership, whose all-gathers and reduce-scatters "
                "call PyNccl directly, through SIRCL's communicator on a tensor-parallel group without PyNccl.",
        models=("Qwen3.8 (vLLM's qwen4_exp code) with VLLM_QWEN3_8_HC_PREFILL_MODE=shard",),
        code=(Code(QWEN_HC, "create", "wraps"), Code(QWEN_HC, "RowOwnership.gather", "calls"),
              Code(QWEN_HC, "RowOwnership.reduce", "calls")),
        enable="SIRCL's communicator installs it when it builds a tensor-parallel group without PyNccl, with a "
               "session, while VLLM_QWEN3_8_HC_PREFILL_MODE=shard (the profile's value)",
        disable="--env VLLM_QWEN3_8_HC_PREFILL_MODE=off",
        without="with the mode shard, setup fails on every rank and names --env "
                "VLLM_QWEN3_8_HC_PREFILL_MODE=off; with it off, every rank processes every prefill row. With "
                "NCCL off (SIRCL_NCCL=never, the default) the shim serves a pair as well; only where "
                "SIRCL_NCCL=topology or auto lets NCCL run, on a pair, does vLLM's own PyNccl path run without it",
        needs=("the tensor-parallel session's reduce-scatter (scatter_available) and all-gather; without the "
               "session's reduce-scatter the adapter carries each reduce-scatter as an all-reduce and keeps this "
               "rank's rows",),
    ),
    Entry(
        name="roce_slot",
        purpose="Makes vLLM's RoCE all-reduce slot class (B12xRoceAllReduce) SIRCL's ring all-reduce, so a "
                "container that enables vLLM's RoCE slot builds SIRCL there.",
        models=("every model with a tensor-parallel group",),
        code=(Code("distributed/device_communicators/b12x_roce_all_reduce.py", "B12xRoceAllReduce", "wraps"),
              Code("distributed/device_communicators/cuda_communicator.py", "CudaCommunicator.__init__", "calls")),
        enable="the general plugin installs it when VLLM_ENABLE_ROCE_ALLREDUCE=1, or when SIRCL_VLLM_SHIMS "
               "names it",
        disable="VLLM_ENABLE_ROCE_ALLREDUCE=0, which the serve launcher and bundle set",
        without="SIRCL's communicator builds its tensor-parallel slot itself, as the launchers run it; with "
                "VLLM_ENABLE_ROCE_ALLREDUCE=1 and the shim refused, the general plugin stops the process",
        needs=("the tensor-parallel session's all-reduce",),
    ),
    Entry(
        name="worker_regimes",
        purpose="Runs the worker's start-up work (compilation, warm-up, graph capture, memory profiling, sleep, "
                "wake-up, weight loads, profiling) in the sessions' startup flag-wait regime and arms the serving "
                "regime when warm-up returns.",
        models=("every model",),
        code=tuple(Code(WORKER, f"Worker.{method}", "wraps") for method in shims.WORKER_STARTUP_METHODS),
        enable="SIRCL's communicator installs it with the first group that gets a session",
        disable="none: every process with a session tries it",
        without="the sessions keep the startup regime's flag-wait limit (SIRCL_STARTUP_WAIT_S, 600 s) while "
                "serving instead of the serving limit (SIRCL_SERVING_WAIT_S, 20 s), and a warning says so",
        needs=("the sessions' flag-wait regimes (enter_startup, enter_serving)",),
    ),
    Entry(
        name="step_health",
        purpose="Checks every SIRCL session and point-to-point channel set of the process for a recorded failure "
                "(a flag wait that timed out, a progress thread that stopped) when each of the worker's step "
                "methods starts, before the step's SIRCL ops launch; host reads only.",
        models=("every model",),
        code=tuple(Code(WORKER, f"Worker.{method}", "wraps") for method in shims.WORKER_STEP_METHODS),
        enable="SIRCL's communicator installs it with the first group that gets a session or point-to-point "
               "channels",
        disable="none: every process with a session or channels tries it",
        without="a failure raises at the worker's post-step check or at the next eager SIRCL call; a step that "
                "replays a CUDA graph launches its kernels first, and they return without work",
        needs=("the sessions' and channels' check_health (host reads)",),
    ),
)


def entry(name: str) -> Entry:
    return next(item for item in ENTRIES if item.name == name)


def verified_builds(name: str) -> list[pins.VllmBuild]:
    """The pinned builds that have every file the shim touches."""
    files = shims.SHIMS[name].files
    return [build for build in pins.SUPPORTED if all(build.files.get(file) is not None for file in files)]


def shim_hooks(name: str) -> list[hooks.Hook]:
    return [hook for hook in hooks.HOOKS if hook.shim == name]


def document() -> dict[str, Any]:
    """The catalog as ``shims.json`` holds it."""
    records = []
    for item in ENTRIES:
        files = shims.SHIMS[item.name].files
        records.append({
            "name": item.name,
            "purpose": item.purpose,
            "models": list(item.models),
            "code": [dataclasses.asdict(code) for code in item.code],
            "files": list(files),
            "anchors": [{"file": anchor.file, "lines": [anchor.first, anchor.last], "text": anchor.text}
                        for hook in shim_hooks(item.name) for anchor in hook.anchors],
            "verified_builds": {build.name: {file: build.files[file] for file in files}
                                for build in verified_builds(item.name)},
            "enable": item.enable,
            "disable": item.disable,
            "without": item.without,
            "measured": [dataclasses.asdict(measurement) for measurement in item.measured],
            "needs": list(item.needs),
        })
    return {
        "schema": SCHEMA,
        "statuses": dict(STATUSES),
        "builds": [{"name": build.name, "version": build.version, "source": build.source}
                   for build in pins.SUPPORTED],
        "shims": records,
    }


def render_document() -> str:
    return json.dumps(document(), indent=1) + "\n"


# -- status in a tree ------------------------------------------------------------------------------


def defined(text: str, name: str) -> bool:
    """``name`` is a top-level function or class, a ``Class.method``, or a name a top-level import binds."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    owner, _, member = name.partition(".")
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not member:
            if any((alias.asname or alias.name) == name for alias in node.names):
                return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == owner:
            if not member:
                return True
            if isinstance(node, ast.ClassDef) and any(
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == member
                    for item in node.body):
                return True
    return False


def package_root(path: Path) -> Path:
    """The ``vllm`` package directory of ``path``: itself, or its ``vllm`` subdirectory (a checkout)."""
    path = Path(path)
    if not (path / "__init__.py").is_file() and (path / "vllm" / "__init__.py").is_file():
        return path / "vllm"
    return path


def shim_status(root: Path, name: str, matches: Sequence[str] = ()) -> dict[str, Any]:
    """One shim's status in the vLLM package at ``root``: ``status``, the pinned ``build`` its files match
    (preferring a build in ``matches``, the builds the whole tree matches), ``installs`` and what is
    ``missing``."""
    files = shims.SHIMS[name].files
    actual = pins.hashes(root, files)
    candidates = [build for build in verified_builds(name)
                  if all(actual[file] == build.files[file] for file in files)]
    preferred = [build for build in candidates if build.name in matches] or candidates
    relied = (*files, *(code.file for code in entry(name).code),
              *(anchor.file for hook in shim_hooks(name) for anchor in hook.anchors))
    absent_files = [file for file in dict.fromkeys(relied) if not (root / file).is_file()]
    missing = [f"{file} is missing" for file in absent_files]
    texts: dict[str, str] = {}
    for code in entry(name).code:
        if code.file in absent_files:
            continue
        text = texts.setdefault(code.file, (root / code.file).read_text(encoding="utf-8", errors="replace"))
        if not defined(text, code.name):
            missing.append(f"{code.file} defines no {code.name}")
    for result in hooks.verify(root, shim_hooks(name)):
        if result.found_at is None and result.anchor.file not in absent_files:
            missing.append(f"{result.anchor.file} has no {result.anchor.text!r} near lines "
                           f"{result.anchor.first}-{result.anchor.last}")
    missing = list(dict.fromkeys(missing))
    if preferred:
        status = "verified"
    elif not missing:
        status = "applicable-unverified"
    else:
        status = "absent"
    return {"name": name, "status": status, "build": preferred[0].name if preferred else None,
            "installs": bool(preferred), "missing": missing}


def tree_status(root: Path) -> dict[str, Any]:
    """Every catalogued shim's status in the vLLM package at ``root``."""
    root = package_root(root)
    report = pins.check(root)
    return {"schema": STATUS_SCHEMA, "tree": str(root), "matches": list(report.matches),
            "shims": [shim_status(root, item.name, report.matches) for item in ENTRIES]}


def status_line(status: dict[str, Any]) -> str:
    """``name status, ...`` in one line, for the stage step's report."""
    return ", ".join(f"{item['name']} {item['status']}" + (f" ({item['build']})" if item["build"] else "")
                     for item in status.get("shims", ()))


def stage_lines(status: object) -> list[str]:
    """The stage step's report of a probe record's ``shim_status``: one line, then one per shim that is not
    verified, with what it misses."""
    if not isinstance(status, dict):
        return []
    lines = ["shims: " + status_line(status)]
    for item in status.get("shims", ()):
        if item.get("status") != "verified":
            reasons = "; ".join(item.get("missing") or ())
            lines.append(f"shim {item.get('name')} {item.get('status')}" + (f": {reasons}" if reasons else ""))
    return lines


def render_status(status: dict[str, Any]) -> str:
    matches = ", ".join(status.get("matches") or ()) or "no pinned build"
    lines = [f"vLLM at {status.get('tree')}: the whole tree matches {matches}",
             f"  {'shim':<26} {'status':<22} {'pinned build':<41} installs"]
    for item in status.get("shims", ()):
        lines.append(f"  {item['name']:<26} {item['status']:<22} {item['build'] or '-':<41} "
                     f"{'yes' if item['installs'] else 'no'}")
        for reason in item.get("missing") or ():
            lines.append(f"      {reason}")
    lines.append("statuses: " + "; ".join(f"{key}: {meaning}" for key, meaning in STATUSES.items()))
    return "\n".join(lines)


def render_catalog() -> str:
    lines = []
    for record in document()["shims"]:
        lines += [f"{record['name']}: {record['purpose']}",
                  f"  models: {'; '.join(record['models'])}",
                  "  vLLM code: " + "; ".join(f"{code['role']} {code['file']}:{code['name']}"
                                             for code in record["code"]),
                  f"  verified builds: {', '.join(record['verified_builds']) or 'none'}",
                  f"  enable: {record['enable']}",
                  f"  disable: {record['disable']}",
                  f"  without it: {record['without']}",
                  "  measured: " + ("; ".join(f"{item['conditions']}: {item['result']}"
                                             for item in record["measured"]) or "no measurement recorded"),
                  f"  needs: {'; '.join(record['needs'])}", ""]
    return "\n".join(lines).rstrip("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sparkring_sircl.vllm.catalog",
                                     description="write or check shims.json from the catalog")
    parser.add_argument("--write", action="store_true", help=f"write {CATALOG_FILE.name}")
    args = parser.parse_args(argv)
    text = render_document()
    if args.write:
        CATALOG_FILE.write_bytes(text.encode("utf-8"))
        return 0
    current = CATALOG_FILE.read_bytes().decode("utf-8") if CATALOG_FILE.is_file() else ""
    if current != text:
        print(f"{CATALOG_FILE} differs from the catalog; run with --write", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
