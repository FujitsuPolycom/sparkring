"""Start checks: whether the settings that a profile lists in ``start_checks`` took effect at its first start.

Some settings of a research profile have no recorded run on the profile's
checkpoint and image. The GLM-5.3 NVFP4-CSF profile keeps every setting of
``glm53-nvfp4-tp8``, but no run shows that these work through the
``nvfp4_csf`` quantization and loader:

- ``mxfp8-linears``: the online MXFP8 dense linears (``--quantization-config``
  with ``linear`` ``mxfp8``) and the MXFP8 LM head (``VLLM_MXFP8_LM_HEAD``);
- ``draft-quantization``: the MTP draft's load-time quantization
  (``--speculative-config`` ``quantization_config``: MXFP8 linears and
  ``nvfp4_a16`` experts) of the checkpoint's BF16 MTP layer;
- ``column-split-loading``: ``glm53full_speedups``' column-parallel latent
  projection and row-parallel ``eh_proj``, which load through the CSF loader;
- ``csf-w4a16-a4-prefill``: W4A16 decode with A4 prefill on the CSF experts
  (``VLLM_B12X_MOE_FP4_FORCE_A16``, ``B12X_W4A16_A4_PREFILL_MIN_TOKENS``);
- ``csf-scale-prefetch``: the side stream that expands the next layer's
  compressed scales (``VLLM_B12X_CSF_SCALE_PREFETCH``, which the profile does
  not set, so the image's default decides);
- ``w4a16-host-barrier-reset``: whether the image's B12X reads
  ``B12X_W4A16_SMALL_M_HOST_BARRIER_RESET`` in the W4A16 expert code that the
  CSF experts run.

After the smoke request, the installer's rank-0 ``smoke`` operation reads the
rank-0 container's log (``docker logs``), the runtime-status endpoint
(``/v1/sparkring/status``), vLLM's metrics (``/metrics``) and a probe run in
the container (``probe_source``), and ``judge`` turns them into one verdict
per check: ``took effect``, ``did not take effect`` or ``unknown``, each with
its evidence and basis. The result (``sparkring-start-checks/v1``) is part of
the smoke action's receipt, is saved as ``start-checks.json`` in rank 0's
installer directory and appears in the install result. A check never fails
the installation; an unreadable source gives ``unknown``.

Bases
-----
- Weights: ``Model loading took N GiB`` of the rank-0 worker. With MXFP8 dense
  linears, glm53-nvfp4-tp8 measured 53.07 GiB per rank on the native
  checkpoint and 57.34 GiB with BF16 linears (TP8, 16K context;
  ``performance/enhancements.md``, ``glm53-mxfp8-linears``). The CSF
  checkpoint keeps the experts' scales compressed on the GPU, which frees an
  estimated 1.91-2.10 GiB per rank, so with every setting in effect the CSF
  start should load about 51.0-51.2 GiB. BF16 dense linears would add about
  4.3 GiB, and BF16 draft experts (256 experts of 6144 x 2048 x 3 at TP8)
  about 1.6 GiB against ``nvfp4_a16``. The verdicts below use the midpoints:
  under ``REFERENCE_GIB`` the linears are MXFP8, at or over ``BF16_LINEARS_GIB``
  they are not; under ``DRAFT_BOUND_GIB`` the draft experts are quantized,
  between it and ``REFERENCE_GIB`` they are not. These are estimates from
  another image and context; an A/B start of glm53-nvfp4-tp8 on the same image
  gives the reference that settles a borderline value.
- The LM heads: the status endpoint's ``target_head_mxfp8`` and
  ``draft_head_quantization`` (``sparkring_runtime_status``), read from the
  loaded modules.
- The draft runs: ``vllm:spec_decode_num_accepted_tokens_total`` above 0 after
  the smoke request.
- The plugin: its own log lines only. Each flag set to 1 needs a ``patched
  for <flag>`` line (the image constructor was replaced) and the line that
  rank 0's helper logs when it builds the split layer (``fused_qkv_a_proj is
  column-parallel over TP8``, ``eh_proj is RowParallelLinear(12288 -> 6144)
  over TP8``). A ``PatchRefused: glm53full_speedups:`` line, the plugin's own
  ``are 0; nothing patched`` line or a ``keeps the ... replicated`` line (the
  helper's fallback at another tensor-parallel size) means the split did not
  take effect. Lines of other plugins do not count: ``glm_dcp_decode_comm``
  logs ``nothing patched`` whenever its items are off, as both GLM-5.3 TP8
  profiles keep them. A split layer that loaded a wrong-sized slice would
  have stopped the start: its loaders narrow each checkpoint tensor to the
  rank's block and check the shapes.
- The CSF experts: the ``NVFP4-CSF layer`` log lines of vLLM's
  ``nvfp4_csf`` method, which name ``W4A16 decode, A4 prefill`` per layer
  (75 layers per rank for GLM-5.3).
- The prefetch flag: ``vllm.envs.VLLM_B12X_CSF_SCALE_PREFETCH`` evaluated in
  the container's environment from the image's own ``vllm/envs.py``; the
  method creates the side stream when it is true and the experts run W4A16.
- The host barrier variable: the image's B12X source files that read it.
"""
from __future__ import annotations

import json
import re

SCHEMA = "sparkring-start-checks/v1"
CHECKS = ("mxfp8-linears", "draft-quantization", "column-split-loading", "csf-w4a16-a4-prefill",
          "csf-scale-prefetch", "w4a16-host-barrier-reset")
TOOK, NOT, UNKNOWN = "took effect", "did not take effect", "unknown"
REFERENCE_GIB = 53.07
BF16_LINEARS_GIB = 55.0
DRAFT_BOUND_GIB = 51.9
CSF_LAYERS = 75
PLUGIN_FLAGS = ("GLM53FULL_LATENT_SHARD", "GLM53FULL_EH_PROJ_TP")
BARRIER = "B12X_W4A16_SMALL_M_HOST_BARRIER_RESET"
# B12X's W4A16 expert code, which both the native and the CSF expert paths run.
W4A16_PATHS = ("b12x/moe/_shared/kernels/w4a16/", "b12x/moe/fused_moe/")
WEIGHTS = re.compile(r"Model loading took ([0-9]+(?:\.[0-9]+)?) GiB")
CSF_LAYER = re.compile(r"NVFP4-CSF layer (\d+) rank (\d+)/(\d+): native NVFP4 (.+?), shared scale scratch")
PATCHED = re.compile(r"glm53full_speedups: (\S+) patched for (\S+)")
# The layer each flag's helper builds on rank 0, logged once per process.
SPLIT_BUILT = {"GLM53FULL_LATENT_SHARD": re.compile(r"glm53full_speedups: \S+\.fused_qkv_a_proj is column-parallel over "
                                                    r"TP8"),
               "GLM53FULL_EH_PROJ_TP": re.compile(r"glm53full_speedups: \S+\.eh_proj is RowParallelLinear\(.*\) over TP8")}
SPLIT_KEPT = re.compile(r"glm53full_speedups: .* keeps the (?:image's )?replicated (fused_qkv_a_proj|eh_proj)")
SPLIT_REFUSED = re.compile(r"PatchRefused: glm53full_speedups:|glm53full_speedups: .* are 0; nothing patched")
ACCEPTED = re.compile(r"^vllm:spec_decode_num_accepted_tokens_total(?:\{[^}]*\})? ([0-9.e+]+)$", re.M)
DRAFTED = re.compile(r"^vllm:spec_decode_num_draft_tokens_total(?:\{[^}]*\})? ([0-9.e+]+)$", re.M)


def probe():
    """Run inside the rank-0 container: the image's prefetch default and its readers of the barrier variable.

    Loads the image's ``vllm/envs.py`` on its own, without importing vLLM (and
    so torch), and evaluates the variable's getter in the container's
    environment. Prints one JSON object.
    """
    import importlib.machinery
    import importlib.util
    import json as _json
    import pathlib
    result = {"scale_prefetch": None, "barrier_readers": None, "errors": []}
    try:
        vllm = pathlib.Path(importlib.machinery.PathFinder.find_spec("vllm").submodule_search_locations[0])
        spec = importlib.util.spec_from_file_location("sparkring_probe_vllm_envs", vllm / "envs.py")
        envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(envs)
        getter = getattr(envs, "environment_variables", {}).get("VLLM_B12X_CSF_SCALE_PREFETCH")
        result["scale_prefetch"] = bool(getter()) if getter is not None else None
        if getter is None:
            result["errors"].append("vllm/envs.py defines no VLLM_B12X_CSF_SCALE_PREFETCH")
    except Exception as error:  # noqa: BLE001 - reported as the probe's evidence
        result["errors"].append(f"vllm/envs.py: {type(error).__name__}: {error}"[:300])
    try:
        b12x = pathlib.Path(importlib.machinery.PathFinder.find_spec("b12x").submodule_search_locations[0])
        result["barrier_readers"] = sorted(
            path.relative_to(b12x.parent).as_posix() for path in b12x.rglob("*.py")
            if "B12X_W4A16_SMALL_M_HOST_BARRIER_RESET" in path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as error:  # noqa: BLE001
        result["errors"].append(f"b12x: {type(error).__name__}: {error}"[:300])
    print(_json.dumps(result))


def probe_source():
    """Self-contained Python source that prints ``probe()`` as JSON."""
    import inspect
    return inspect.getsource(probe) + "\n\nprobe()\n"


def _check(verdict, evidence, basis):
    return {"verdict": verdict, "evidence": evidence, "basis": basis}


def _metric(pattern, text):
    values = [float(value) for value in pattern.findall(text or "")]
    return sum(values) if values else None


def judge(log, status, metrics, facts, environment, checks=CHECKS):
    """The ``sparkring-start-checks/v1`` document from the gathered sources; any of them may be None.

    ``log`` is the rank-0 container's log text, ``status`` the status
    endpoint's JSON, ``metrics`` the ``/metrics`` text, ``facts`` the probe's
    output and ``environment`` the container's environment from its
    specification.
    """
    log = log or ""
    found = WEIGHTS.findall(log)
    weights = float(found[-1]) if found else None
    # The status document lists each worker's snapshot under workers.ranks; its
    # effective facts hold {"state", "source", "value"} per choice.
    workers = status.get("workers") if isinstance(status, dict) else None
    workers = workers.get("ranks") if isinstance(workers, dict) else None
    model = {}
    for worker in workers if isinstance(workers, list) else []:
        choices = worker.get("effective") if isinstance(worker, dict) else None
        if isinstance(choices, dict) and "target_head_mxfp8" in choices:
            model = choices
            break

    def fact(name):
        value = model.get(name)
        return value.get("value") if isinstance(value, dict) else value

    weights_text = f"rank 0 loaded {weights} GiB of weights" if weights is not None else "no weight figure in the log"
    result = {}
    head = fact("target_head_mxfp8")
    if head is False or (weights is not None and weights >= BF16_LINEARS_GIB):
        verdict = NOT
    elif head is True and weights is not None and weights < REFERENCE_GIB:
        verdict = TOOK
    else:
        verdict = UNKNOWN
    result["mxfp8-linears"] = _check(verdict, [f"target_head_mxfp8: {head}", weights_text],
                                     f"MXFP8 LM head from the status endpoint; dense linears by weights per rank under "
                                     f"{REFERENCE_GIB} GiB (MXFP8) or at least {BF16_LINEARS_GIB} GiB (BF16)")
    accepted, drafted = _metric(ACCEPTED, metrics), _metric(DRAFTED, metrics)
    runs = accepted is not None and accepted > 0
    if weights is not None and weights < DRAFT_BOUND_GIB and runs:
        verdict = TOOK
    elif weights is not None and DRAFT_BOUND_GIB <= weights < REFERENCE_GIB:
        verdict = NOT
    else:
        verdict = UNKNOWN
    result["draft-quantization"] = _check(verdict, [
        f"draft_head_quantization: {fact('draft_head_quantization')}",
        f"draft tokens accepted {accepted} of {drafted}" if drafted is not None else "no speculative-decoding metrics",
        weights_text], f"draft experts quantized when weights per rank are under {DRAFT_BOUND_GIB} GiB and the draft's "
                       f"tokens are accepted; BF16 draft experts between {DRAFT_BOUND_GIB} and {REFERENCE_GIB} GiB")
    patched = {flag for _, flag in PATCHED.findall(log)}
    built = {flag for flag, pattern in SPLIT_BUILT.items() if pattern.search(log)}
    kept = sorted(set(SPLIT_KEPT.findall(log)))
    refused = SPLIT_REFUSED.search(log) is not None
    wanted = {flag for flag in PLUGIN_FLAGS if (environment or {}).get(flag) == "1"}
    if refused or kept or (wanted and not patched):
        verdict = NOT
    elif wanted and wanted <= patched and wanted <= built:
        verdict = TOOK
    else:
        verdict = UNKNOWN
    result["column-split-loading"] = _check(
        verdict, [f"patched for: {', '.join(sorted(patched)) or 'none'}",
                  f"split layers built for: {', '.join(sorted(built)) or 'none'}"]
        + (["glm53full_speedups refused to patch or patched nothing"] if refused else [])
        + ([f"glm53full_speedups kept the replicated {', '.join(kept)}"] if kept else []),
        "glm53full_speedups' own log lines: every flag set to 1 patched its constructor and rank 0 built the split "
        "layer, with no refusal and no replicated fallback; the weights loaded, since the API answered the smoke "
        "request")
    layers = CSF_LAYER.findall(log)
    modes = sorted({mode.strip() for _, _, _, mode in layers})
    a4 = sum(1 for _, _, _, mode in layers if mode.strip() == "W4A16 decode, A4 prefill")
    verdict = TOOK if a4 >= CSF_LAYERS else NOT if layers else UNKNOWN
    result["csf-w4a16-a4-prefill"] = _check(verdict, [f"{len(layers)} NVFP4-CSF layer lines on rank 0, modes: "
                                                      f"{', '.join(modes) or 'none'}"],
                                            f"at least {CSF_LAYERS} layers report W4A16 decode, A4 prefill")
    facts = facts if isinstance(facts, dict) else {}
    prefetch = facts.get("scale_prefetch")
    a16 = (environment or {}).get("VLLM_B12X_MOE_FP4_FORCE_A16") == "1"
    verdict = UNKNOWN if prefetch is None else TOOK if prefetch and a16 else NOT
    result["csf-scale-prefetch"] = _check(verdict, [f"VLLM_B12X_CSF_SCALE_PREFETCH evaluates to {prefetch}",
                                                    f"VLLM_B12X_MOE_FP4_FORCE_A16: {(environment or {}).get('VLLM_B12X_MOE_FP4_FORCE_A16')}",
                                                    *facts.get("errors", [])],
                                          "the method creates the prefetch stream when the flag is true and the "
                                          "experts run W4A16")
    readers = facts.get("barrier_readers")
    applies = None if readers is None else any(path.startswith(W4A16_PATHS) for path in readers)
    verdict = UNKNOWN if applies is None else TOOK if applies else NOT
    result["w4a16-host-barrier-reset"] = _check(verdict, [f"{BARRIER} readers: "
                                                          f"{', '.join(readers) if readers else readers}"],
                                                "a reader in B12X's W4A16 expert code, which the CSF experts run")
    return {"schema": SCHEMA, "weights_gib": weights, "checks": {name: result[name] for name in checks},
            "capacity": capacity(status, metrics)}


def capacity(status, metrics):
    """Observed context, request limit and KV tokens; absent facts stay absent.

    Runtime worker facts own effective limits. KV tokens come from the same
    first-engine cache gauge used by the ring dashboard, without summing TP ranks.
    """
    from runtime.host.dashboard import capacity_from_cache_info, parse_metrics

    cache = capacity_from_cache_info(parse_metrics(metrics or "").get("cache_info")) or {}
    result = {"kv_tokens": cache["kv_tokens"]} if cache.get("kv_tokens") is not None else {}
    workers = status.get("workers") if isinstance(status, dict) else None
    workers = workers.get("ranks") if isinstance(workers, dict) else None
    workers = workers if isinstance(workers, list) else []
    for key, name in (("max_model_len", "context_length"), ("max_num_seqs", "max_requests")):
        facts = [worker.get("effective") if isinstance(worker, dict) else None for worker in workers]
        facts = [fact.get(key) if isinstance(fact, dict) else None for fact in facts]
        values = [fact.get("value") for fact in facts if isinstance(fact, dict) and fact.get("state") == "known"]
        if values and len(values) == len(workers) and all(type(value) is int and value > 0 for value in values):
            if len(set(values)) == 1:
                result[name] = values[0]
    return result


def text(document):
    """One line per check for the install summary."""
    return [f"{name}: {item['verdict']} ({'; '.join(item['evidence'])})" for name, item in document["checks"].items()]


def parse_probe(output):
    """The probe's JSON object, or None when its output is not one."""
    try:
        value = json.loads(output)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None
