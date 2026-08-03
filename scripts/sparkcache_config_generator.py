#!/usr/bin/env python3
"""Generate and verify SparkCache-enabled argv/env for the DCP2 variant.

Pure functions — no docker, no SSH, no host mutation.  Designed to be unit-
tested offline and called by ``live_dcp2_cutover.py`` when the operator
authorizes a cache-enabled cutover.

The canonical ``--kv-transfer-config`` JSON schema (from sparkcache/README.md):

    {
      "kv_connector": "SparkContextCacheConnector",
      "kv_role": "kv_both",
      "kv_connector_module_path": "spark_context_cache_connector",
      "kv_load_failure_policy": "recompute",
      "kv_connector_extra_config": {
        "spark_cache_root": "/cache/context",
        "spark_cache_target_checkpoint_sha256": "...",
        "spark_cache_draft_policy": "separate",
        "spark_cache_draft_checkpoint_sha256": "...",
        "spark_cache_store": true,
        "spark_cache_restore": true,
        "spark_cache_streaming_snapshots": false
      }
    }

The enable switch is the **presence** of the complete ``--kv-transfer-config``
argument.  The connector does NOT consume ``SPARK_CONTEXT_CACHE_ENABLE``;
that variable is a legacy image flag with no connector authority.

The pinned vLLM factory also requires ``--disable-hybrid-kv-cache-manager``
because this connector does not advertise HMA support.
"""

from __future__ import annotations

import json
import sys
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KVTc_ARG = "--kv-transfer-config"
HMA_ARG = "--disable-hybrid-kv-cache-manager"

# DCP2 limits that must survive the rewrite.
DCP2_ARG_CHECKS = {
    "--decode-context-parallel-size": "2",
    "--max-model-len": "524288",
}

# Keys that must be absent (empty) in the DCP2 variant.
RUNTIME_UNSET_ENVIRONMENT = {
    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": "",
}

# Legacy image variable — NOT a connector switch.  We do not emit or
# consume it.  Documented here so the verifier can flag it if someone
# mistakenly adds it to the env.
LEGACY_ENABLE_KEY = "SPARK_CONTEXT_CACHE_ENABLE"

DEFAULT_CACHE_ROOT = "/cache/context"

_HEX64 = set("0123456789abcdef")


class ConfigGeneratorError(ValueError):
    """Raised when the source argv/env is incompatible with SparkCache."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_map(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        name, _, value = item.partition("=")
        result[name] = value
    return result


def _env_list(mapping: dict[str, str]) -> list[str]:
    return [f"{name}={value}" for name, value in sorted(mapping.items())]


def _is_valid_hex64(value: str) -> bool:
    return len(value) == 64 and all(c in _HEX64 for c in value)


def _find_option_positions(cmd: list[str], option: str) -> list[int]:
    return [i for i in range(len(cmd)) if cmd[i] == option]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_sparkcache_argv(
    source_cmd: list[str],
    source_env: list[str],
    *,
    target_checkpoint: str | None = None,
    draft_checkpoint: str | None = None,
    draft_policy: str = "separate",
    streaming_snapshots: bool = False,
    enabled: bool = False,
    cache_root: str = DEFAULT_CACHE_ROOT,
) -> tuple[list[str], list[str]]:
    """Return SparkCache-augmented (cmd, env) from DCP2 source lists.

    Parameters
    ----------
    source_cmd, source_env
        The DCP2 container's ``Cmd`` and ``Env`` as seen by
        ``docker inspect``.  Must already contain DCP2 args.
    target_checkpoint, draft_checkpoint
        64-character lowercase SHA-256 checkpoint identity strings.
        **Required when enabled=True.**  Must be produced by a
        canonical manifest generator run against the mounted model.
        Ignored when enabled=False (disabled output omits identity env).
    draft_policy
        ``"separate"`` or ``"colocated_target"``.
    streaming_snapshots
        Must be ``False`` for the first live gate.
    enabled
        If ``False`` (default), no ``--kv-transfer-config`` and no
        ``--disable-hybrid-kv-cache-manager`` are added, and no
        identity env vars are emitted.  The output is a clean cache-off
        argv identical to the source except for verified DCP2 limits
        and removal of empty VLLM_PREFIX_CACHE_RETENTION_INTERVAL.
    cache_root
        Rank-local store root path (default ``/cache/context``).
    """
    cmd = list(source_cmd)
    env = _env_map(source_env)

    # --- Validate DCP2 limits before touching anything ---
    for option, expected in DCP2_ARG_CHECKS.items():
        positions = _find_option_positions(cmd, option)
        if len(positions) != 1:
            raise ConfigGeneratorError(
                f"expected exactly one {option}, found {len(positions)}"
            )
        value_index = positions[0] + 1
        if value_index >= len(cmd) or cmd[value_index] != expected:
            actual = cmd[value_index] if value_index < len(cmd) else None
            raise ConfigGeneratorError(
                f"{option} expected {expected!r}, got {actual!r}"
            )

    # --- Remove empty VLLM_PREFIX_CACHE_RETENTION_INTERVAL ---
    for name, expected_empty in RUNTIME_UNSET_ENVIRONMENT.items():
        actual = env.get(name)
        if actual is not None and actual != expected_empty:
            raise ConfigGeneratorError(
                f"{name} must be absent or empty, got {actual!r}"
            )
        env.pop(name, None)

    # --- Remove legacy SPARK_CONTEXT_CACHE_ENABLE if present ---
    # It has no connector authority.  We strip it so the generated
    # env is clean; the source image may carry it as a no-op.
    env.pop(LEGACY_ENABLE_KEY, None)

    if not enabled:
        # Disabled: no --kv-transfer-config, no --disable-hybrid-kv-cache-manager,
        # no identity env vars.  Return clean cache-off argv.
        if _find_option_positions(cmd, KVTc_ARG):
            raise ConfigGeneratorError(
                f"source cmd already contains {KVTc_ARG}; remove it or set enabled=True"
            )
        return cmd, _env_list(env)

    # --- Enabled: validate checkpoint identities ---
    if target_checkpoint is None or not _is_valid_hex64(target_checkpoint):
        raise ConfigGeneratorError(
            "enabled=True requires a 64-character lowercase SHA-256"
            " target_checkpoint (from a canonical manifest generator)"
        )
    if draft_policy == "separate":
        if draft_checkpoint is None or not _is_valid_hex64(draft_checkpoint):
            raise ConfigGeneratorError(
                "separate draft policy requires a 64-character lowercase"
                " SHA-256 draft_checkpoint"
            )
    elif draft_policy == "colocated_target":
        if draft_checkpoint and draft_checkpoint != target_checkpoint:
            raise ConfigGeneratorError(
                "colocated_target draft must use the target checkpoint identity;"
                " omit draft_checkpoint or set it equal to target_checkpoint"
            )
        draft_checkpoint = target_checkpoint
    else:
        raise ConfigGeneratorError(
            "draft_policy must be 'separate' or 'colocated_target'"
        )

    if streaming_snapshots:
        raise ConfigGeneratorError(
            "streaming_snapshots must be False for the first live gate"
        )

    # --- Reject duplicate --kv-transfer-config ---
    if _find_option_positions(cmd, KVTc_ARG):
        raise ConfigGeneratorError(
            f"source cmd already contains {KVTc_ARG}; refusing to add a second"
        )

    # --- Build canonical --kv-transfer-config JSON ---
    kv_transfer_config = {
        "kv_connector": "SparkContextCacheConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "spark_context_cache_connector",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "spark_cache_root": cache_root,
            "spark_cache_target_checkpoint_sha256": target_checkpoint,
            "spark_cache_draft_policy": draft_policy,
            "spark_cache_draft_checkpoint_sha256": draft_checkpoint,
            "spark_cache_store": True,
            "spark_cache_restore": True,
            "spark_cache_streaming_snapshots": False,
        },
    }
    cmd.extend([KVTc_ARG, json.dumps(kv_transfer_config)])

    # --- Add --disable-hybrid-kv-cache-manager ---
    if not _find_option_positions(cmd, HMA_ARG):
        cmd.append(HMA_ARG)

    return cmd, _env_list(env)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def verify_sparkcache_argv(
    cmd: list[str],
    env: list[str],
    *,
    expect_enabled: bool = False,
) -> None:
    """Verify that a SparkCache-augmented argv/env satisfies all invariants.

    Raises ``ConfigGeneratorError`` on any violation.
    """
    env_map = _env_map(env)

    # 1. DCP2 limits retained.
    for option, expected in DCP2_ARG_CHECKS.items():
        positions = _find_option_positions(cmd, option)
        if len(positions) != 1:
            raise ConfigGeneratorError(
                f"expected exactly one {option}, found {len(positions)}"
            )
        value_index = positions[0] + 1
        if value_index >= len(cmd) or cmd[value_index] != expected:
            actual = cmd[value_index] if value_index < len(cmd) else None
            raise ConfigGeneratorError(
                f"{option} expected {expected!r}, got {actual!r}"
            )

    # 2. VLLM_PREFIX_CACHE_RETENTION_INTERVAL must be absent.
    if "VLLM_PREFIX_CACHE_RETENTION_INTERVAL" in env_map:
        raise ConfigGeneratorError(
            "VLLM_PREFIX_CACHE_RETENTION_INTERVAL must be absent from env"
        )

    # 3. Legacy SPARK_CONTEXT_CACHE_ENABLE must not be in generated env.
    if LEGACY_ENABLE_KEY in env_map:
        raise ConfigGeneratorError(
            f"{LEGACY_ENABLE_KEY} must not be in generated env;"
            " it is a legacy image variable with no connector authority"
        )

    # 4. Exactly zero or one --kv-transfer-config.
    kvtc_positions = _find_option_positions(cmd, KVTc_ARG)
    if len(kvtc_positions) > 1:
        raise ConfigGeneratorError(
            f"found {len(kvtc_positions)} {KVTc_ARG} args, expected 0 or 1"
        )

    if expect_enabled:
        # 5. Exactly one --kv-transfer-config.
        if len(kvtc_positions) != 1:
            raise ConfigGeneratorError(
                f"expected exactly one {KVTc_ARG} when enabled"
            )
        value_index = kvtc_positions[0] + 1
        if value_index >= len(cmd):
            raise ConfigGeneratorError(f"{KVTc_ARG} has no value")
        raw = cmd[value_index]

        # 6. JSON must be exactly one argv element.
        # (It is — cmd[value_index] is a single string.)

        # 7. Validate full JSON values.
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ConfigGeneratorError(
                f"{KVTc_ARG} is not valid JSON: {error}"
            ) from error

        expected_fields = {
            "kv_connector": "SparkContextCacheConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "spark_context_cache_connector",
            "kv_load_failure_policy": "recompute",
        }
        for key, expected_val in expected_fields.items():
            if parsed.get(key) != expected_val:
                raise ConfigGeneratorError(
                    f"{KVTc_ARG} {key} must be {expected_val!r}, got {parsed.get(key)!r}"
                )

        extra = parsed.get("kv_connector_extra_config")
        if not isinstance(extra, dict):
            raise ConfigGeneratorError(
                f"{KVTc_ARG} kv_connector_extra_config must be a dict"
            )

        extra_checks = {
            "spark_cache_root": str,
            "spark_cache_target_checkpoint_sha256": str,
            "spark_cache_draft_policy": str,
            "spark_cache_draft_checkpoint_sha256": str,
            "spark_cache_store": bool,
            "spark_cache_restore": bool,
            "spark_cache_streaming_snapshots": bool,
        }
        for key, expected_type in extra_checks.items():
            if key not in extra:
                raise ConfigGeneratorError(
                    f"{KVTc_ARG} kv_connector_extra_config missing {key}"
                )
            if not isinstance(extra[key], expected_type):
                raise ConfigGeneratorError(
                    f"{KVTc_ARG} kv_connector_extra_config {key} must be"
                    f" {expected_type.__name__}, got {type(extra[key]).__name__}"
                )

        # 8. Specific value checks.
        if not _is_valid_hex64(extra["spark_cache_target_checkpoint_sha256"]):
            raise ConfigGeneratorError(
                "spark_cache_target_checkpoint_sha256 must be 64 lowercase hex"
            )
        if not _is_valid_hex64(extra["spark_cache_draft_checkpoint_sha256"]):
            raise ConfigGeneratorError(
                "spark_cache_draft_checkpoint_sha256 must be 64 lowercase hex"
            )
        if extra["spark_cache_draft_policy"] not in ("separate", "colocated_target"):
            raise ConfigGeneratorError(
                "spark_cache_draft_policy must be 'separate' or 'colocated_target'"
            )
        if extra["spark_cache_store"] is not True:
            raise ConfigGeneratorError("spark_cache_store must be true")
        if extra["spark_cache_restore"] is not True:
            raise ConfigGeneratorError("spark_cache_restore must be true")
        if extra["spark_cache_streaming_snapshots"] is not False:
            raise ConfigGeneratorError(
                "spark_cache_streaming_snapshots must be false for the first gate"
            )

        # 9. --disable-hybrid-kv-cache-manager must be present.
        if not _find_option_positions(cmd, HMA_ARG):
            raise ConfigGeneratorError(
                f"{HMA_ARG} must be present when SparkCache is enabled"
                " (connector does not advertise HMA support)"
            )
    else:
        # Disabled: no --kv-transfer-config, no --disable-hybrid-kv-cache-manager.
        if kvtc_positions:
            raise ConfigGeneratorError(
                f"{KVTc_ARG} present but cache is disabled; remove it"
            )
        if _find_option_positions(cmd, HMA_ARG):
            raise ConfigGeneratorError(
                f"{HMA_ARG} present but cache is disabled; remove it"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Reads JSON from stdin (``docker inspect`` output), extracts Cmd and
    Env, and prints the generated config as JSON.

    Without ``--enable``, produces a disabled (cache-off) config that
    requires no checkpoint identities.

    With ``--enable``, requires ``--target-checkpoint`` and
    ``--draft-checkpoint`` (64-hex SHA-256 values from a canonical
    manifest generator).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SparkCache-enabled argv/env from a DCP2 container inspect JSON.",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Enable SparkCache (requires --target-checkpoint and --draft-checkpoint)",
    )
    parser.add_argument(
        "--target-checkpoint",
        default=None,
        help="64-char lowercase SHA-256 target checkpoint identity (required with --enable)",
    )
    parser.add_argument(
        "--draft-checkpoint",
        default=None,
        help="64-char lowercase SHA-256 draft checkpoint identity (required with --enable)",
    )
    parser.add_argument(
        "--draft-policy",
        default="separate",
        choices=["separate", "colocated_target"],
    )
    parser.add_argument(
        "--cache-root",
        default=DEFAULT_CACHE_ROOT,
    )
    args = parser.parse_args(argv)

    doc = json.load(sys.stdin)
    config = doc[0]["Config"] if isinstance(doc, list) else doc["Config"]
    source_cmd = config["Cmd"]
    source_env = config["Env"]

    generated_cmd, generated_env = generate_sparkcache_argv(
        source_cmd,
        source_env,
        target_checkpoint=args.target_checkpoint,
        draft_checkpoint=args.draft_checkpoint,
        draft_policy=args.draft_policy,
        enabled=args.enable,
        cache_root=args.cache_root,
    )
    verify_sparkcache_argv(
        generated_cmd, generated_env, expect_enabled=args.enable
    )
    print(json.dumps({"cmd": generated_cmd, "env": generated_env}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
