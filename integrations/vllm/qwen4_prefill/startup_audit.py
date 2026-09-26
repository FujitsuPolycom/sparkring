"""Report Qwen optimization prerequisites without inferring measured speed."""

import ast
import logging
import os
from pathlib import Path


def inspect_settings(environment, *, tp, batch, source):
    """Return explicit configuration/source evidence; never mutate settings."""
    rows = []
    mode = environment.get("VLLM_QWEN3_8_HC_PREFILL_MODE", "off")
    if mode == "shard":
        tree = ast.parse(source)
        wrapper = next(
            (
                n
                for n in tree.body
                if isinstance(n, ast.ClassDef)
                and n.name == "Qwen4ExpForConditionalGeneration"
            ),
            None,
        )
        forward = next(
            (
                n
                for n in getattr(wrapper, "body", [])
                if getattr(n, "name", None) == "forward"
            ),
            None,
        )
        calls = (
            {ast.unparse(n.func) for n in ast.walk(forward) if isinstance(n, ast.Call)}
            if forward
            else set()
        )
        if "self.language_model.model" in calls:
            rows.append(
                (
                    "WARNING",
                    "HC_ROUTE_BYPASS",
                    "HC sharding requested, but multimodal forward bypasses its dispatcher.",
                )
            )
        elif "self.language_model" in calls:
            rows.append(
                (
                    "INFO",
                    "HC_ROUTE_VERIFIED",
                    "HC dispatcher reachable; execution is unverified until an eligible real prefill.",
                )
            )
        else:
            rows.append(
                ("WARNING", "HC_ROUTE_UNKNOWN", "Cannot verify multimodal HC routing.")
            )
        if tp not in (2, 4):
            rows.append(
                (
                    "WARNING",
                    "HC_TOPOLOGY",
                    "This HC sharding implementation requires TP2 or TP4.",
                )
            )
        if batch < 1024:
            rows.append(
                (
                    "WARNING",
                    "HC_BATCH_TOO_SMALL",
                    "Batch limit prevents the >=1024-row HC path.",
                )
            )
    elif mode == "off" and tp == 4:
        rows.append(("WARNING", "HC_DISABLED", "TP4 HC row sharding is disabled."))
    for flag, label in (
        ("VLLM_QWEN3_8_PREFILL_COALESCE", "checkpoint coalescing"),
        ("VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT", "compact MTP"),
        ("VLLM_QWEN3_8_FLASH_NEXT_OVERLAP", "projection overlap"),
    ):
        enabled = environment.get(flag, "0") == "1"
        rows.append(
            (
                "INFO" if enabled else "WARNING",
                flag,
                f"{label}: {'requested; runtime activation not proven' if enabled else 'disabled'}",
            )
        )
    if environment.get("B12X_NVFP4_DYNAMIC_MATERIALIZED") == "1":
        rows.append(
            (
                "WARNING",
                "DYNAMIC_MATERIALIZED",
                "Dynamic materialization is enabled; check its suitability for the GB10 profile.",
            )
        )
    return rows


def emit(config, logger=None):
    """Log before HTTP serving; audit errors warn rather than break inference."""
    logger = logger or logging.getLogger("sparkring.startup_audit")
    try:
        architectures = (
            getattr(config.model_config.hf_config, "architectures", ()) or ()
        )
        if not any(
            "Qwen3_8FlashNext" in name or "Qwen4Exp" in name for name in architectures
        ):
            return
        import vllm

        source = (
            Path(vllm.__file__).parent / "models/qwen4_exp/nvidia/model.py"
        ).read_text()
        rows = inspect_settings(
            os.environ,
            tp=config.parallel_config.tensor_parallel_size,
            batch=config.scheduler_config.max_num_batched_tokens,
            source=source,
        )
        logger.warning(
            "SPARKRING STARTUP AUDIT: configuration and source checks only; not a performance certificate"
        )
        for level, code, message in rows:
            getattr(logger, level.lower())("SPARKRING AUDIT [%s] %s", code, message)
        logger.warning(
            "SPARKRING AUDIT: %d warning(s); settings unchanged. Real HC activation logs QWEN_HC_PREFILL.",
            sum(level == "WARNING" for level, _, _ in rows),
        )
    except Exception as error:
        logger.warning(
            "SPARKRING AUDIT UNAVAILABLE: %s; activation is unverified",
            type(error).__name__,
        )
