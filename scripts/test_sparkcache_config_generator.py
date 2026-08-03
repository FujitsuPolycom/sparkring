#!/usr/bin/env python3
"""Unit tests for sparkcache_config_generator.

Proves the effective argv:
  - Contains the canonical --kv-transfer-config JSON schema from sparkcache/README.md
  - Retains DCP2 limits (--decode-context-parallel-size 2, --max-model-len 524288)
  - Removes empty VLLM_PREFIX_CACHE_RETENTION_INTERVAL
  - Disabled by default: no --kv-transfer-config, no identity env vars
  - Enabled: exactly one --kv-transfer-config + --disable-hybrid-kv-cache-manager
  - Streaming snapshots must be false for the first gate
  - SPARK_CONTEXT_CACHE_ENABLE is NOT in generated env (legacy, no connector authority)
  - CLI: disabled run succeeds without checkpoints; enabled fails closed without them
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import sparkcache_config_generator as gen
from sparkcache_config_generator import (
    ConfigGeneratorError,
    generate_sparkcache_argv,
    verify_sparkcache_argv,
    main,
)

# A minimal DCP2 source cmd/env that mimics what docker inspect would return.
_SOURCE_CMD = [
    "/opt/venv/bin/vllm",
    "--model", "/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid",
    "--tensor-parallel-size", "4",
    "--decode-context-parallel-size", "2",
    "--max-model-len", "524288",
    "--gpu-memory-utilization", "0.89",
]

_SOURCE_ENV = [
    "VLLM_SPARK_DCP_SIZE=2",
    "VLLM_SPARK_MAX_MODEL_LEN=524288",
    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=",
    "SPARK_CONTEXT_CACHE_ENABLE=0",
]

_TARGET_ID = "a" * 64
_DRAFT_ID = "b" * 64

_GENERATOR = Path(__file__).resolve().parent / "sparkcache_config_generator.py"


def _env_map(env: list[str]) -> dict[str, str]:
    return {e.split("=", 1)[0]: e.split("=", 1)[1] for e in env}


class DisabledByDefaultTests(unittest.TestCase):

    def test_disabled_has_no_kv_transfer_config(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
        )
        self.assertNotIn(gen.KVTc_ARG, cmd)

    def test_disabled_has_no_disable_hybrid(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
        )
        self.assertNotIn(gen.HMA_ARG, cmd)

    def test_disabled_removes_prefix_cache_retention(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
        )
        self.assertNotIn("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", _env_map(env))

    def test_disabled_removes_legacy_enable(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
        )
        self.assertNotIn(gen.LEGACY_ENABLE_KEY, _env_map(env))

    def test_disabled_emits_no_identity_env_vars(self) -> None:
        """Disabled output must not contain any SPARK_CONTEXT_CACHE_* identity env."""
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
        )
        em = _env_map(env)
        for key in list(em):
            self.assertFalse(
                key.startswith("SPARK_CONTEXT_CACHE"),
                f"disabled output should not emit {key}",
            )

    def test_disabled_retains_dcp2_limits(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
        )
        self.assertEqual(cmd[cmd.index("--decode-context-parallel-size") + 1], "2")
        self.assertEqual(cmd[cmd.index("--max-model-len") + 1], "524288")

    def test_disabled_passes_verifier(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
        )
        verify_sparkcache_argv(cmd, env, expect_enabled=False)

    def test_disabled_does_not_require_checkpoints(self) -> None:
        """Disabled generation works with no checkpoint identities at all."""
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=None,
            draft_checkpoint=None,
        )
        self.assertNotIn(gen.KVTc_ARG, cmd)
        verify_sparkcache_argv(cmd, env, expect_enabled=False)


class EnabledSchemaTests(unittest.TestCase):

    def test_enabled_adds_exactly_one_kv_transfer_config(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        positions = [i for i in range(len(cmd)) if cmd[i] == gen.KVTc_ARG]
        self.assertEqual(len(positions), 1)

    def test_enabled_kv_transfer_config_has_canonical_schema(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        idx = cmd.index(gen.KVTc_ARG)
        parsed = json.loads(cmd[idx + 1])
        self.assertEqual(parsed["kv_connector"], "SparkContextCacheConnector")
        self.assertEqual(parsed["kv_role"], "kv_both")
        self.assertEqual(parsed["kv_connector_module_path"], "spark_context_cache_connector")
        self.assertEqual(parsed["kv_load_failure_policy"], "recompute")
        extra = parsed["kv_connector_extra_config"]
        self.assertIsInstance(extra, dict)
        self.assertEqual(extra["spark_cache_root"], "/cache/context")
        self.assertEqual(extra["spark_cache_target_checkpoint_sha256"], _TARGET_ID)
        self.assertEqual(extra["spark_cache_draft_policy"], "separate")
        self.assertEqual(extra["spark_cache_draft_checkpoint_sha256"], _DRAFT_ID)
        self.assertIs(extra["spark_cache_store"], True)
        self.assertIs(extra["spark_cache_restore"], True)
        self.assertIs(extra["spark_cache_streaming_snapshots"], False)

    def test_enabled_json_is_single_argv_element(self) -> None:
        """The JSON value must be a single element in cmd, not split."""
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        idx = cmd.index(gen.KVTc_ARG)
        # idx+1 is the JSON string, idx+2 should be the next option or end
        raw = cmd[idx + 1]
        json.loads(raw)  # must parse
        # No engine_number or extra_config keys (removed from old schema)
        parsed = json.loads(raw)
        self.assertNotIn("engine_number", parsed)
        self.assertNotIn("extra_config", parsed)

    def test_enabled_adds_disable_hybrid_kv_cache_manager(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        self.assertIn(gen.HMA_ARG, cmd)

    def test_enabled_does_not_add_duplicate_disable_hybrid(self) -> None:
        source_with_hma = _SOURCE_CMD + [gen.HMA_ARG]
        cmd, env = generate_sparkcache_argv(
            source_with_hma, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        positions = [i for i in range(len(cmd)) if cmd[i] == gen.HMA_ARG]
        self.assertEqual(len(positions), 1)

    def test_enabled_retains_dcp2_limits(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        self.assertEqual(cmd[cmd.index("--decode-context-parallel-size") + 1], "2")
        self.assertEqual(cmd[cmd.index("--max-model-len") + 1], "524288")

    def test_enabled_removes_prefix_cache_retention(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        self.assertNotIn("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", _env_map(env))

    def test_enabled_removes_legacy_enable(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        self.assertNotIn(gen.LEGACY_ENABLE_KEY, _env_map(env))

    def test_enabled_emits_no_identity_env_vars(self) -> None:
        """Identity lives only in --kv-transfer-config JSON, not env."""
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        em = _env_map(env)
        for key in list(em):
            self.assertFalse(
                key.startswith("SPARK_CONTEXT_CACHE"),
                f"enabled output should not emit {key} as env var",
            )

    def test_enabled_passes_verifier(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        verify_sparkcache_argv(cmd, env, expect_enabled=True)


class RejectBadInputTests(unittest.TestCase):

    def test_rejects_enabled_without_target_checkpoint(self) -> None:
        with self.assertRaisesRegex(ConfigGeneratorError, "target_checkpoint"):
            generate_sparkcache_argv(
                _SOURCE_CMD, _SOURCE_ENV,
                target_checkpoint=None,
                draft_checkpoint=_DRAFT_ID,
                enabled=True,
            )

    def test_rejects_enabled_with_short_target_checkpoint(self) -> None:
        with self.assertRaisesRegex(ConfigGeneratorError, "target_checkpoint"):
            generate_sparkcache_argv(
                _SOURCE_CMD, _SOURCE_ENV,
                target_checkpoint="abc",
                draft_checkpoint=_DRAFT_ID,
                enabled=True,
            )

    def test_rejects_enabled_with_non_hex_target_checkpoint(self) -> None:
        with self.assertRaisesRegex(ConfigGeneratorError, "target_checkpoint"):
            generate_sparkcache_argv(
                _SOURCE_CMD, _SOURCE_ENV,
                target_checkpoint="z" * 64,
                draft_checkpoint=_DRAFT_ID,
                enabled=True,
            )

    def test_rejects_separate_draft_without_draft_checkpoint(self) -> None:
        with self.assertRaisesRegex(ConfigGeneratorError, "draft_checkpoint"):
            generate_sparkcache_argv(
                _SOURCE_CMD, _SOURCE_ENV,
                target_checkpoint=_TARGET_ID,
                draft_checkpoint=None,
                draft_policy="separate",
                enabled=True,
            )

    def test_rejects_colocated_draft_with_conflicting_checkpoint(self) -> None:
        with self.assertRaisesRegex(ConfigGeneratorError, "colocated_target"):
            generate_sparkcache_argv(
                _SOURCE_CMD, _SOURCE_ENV,
                target_checkpoint=_TARGET_ID,
                draft_checkpoint=_DRAFT_ID,
                draft_policy="colocated_target",
                enabled=True,
            )

    def test_rejects_streaming_snapshots_for_first_gate(self) -> None:
        with self.assertRaisesRegex(ConfigGeneratorError, "streaming_snapshots"):
            generate_sparkcache_argv(
                _SOURCE_CMD, _SOURCE_ENV,
                target_checkpoint=_TARGET_ID,
                draft_checkpoint=_DRAFT_ID,
                streaming_snapshots=True,
                enabled=True,
            )

    def test_rejects_missing_dcp2_arg(self) -> None:
        bad_cmd = [c for c in _SOURCE_CMD if c != "--decode-context-parallel-size"]
        bad_cmd = [c for c in bad_cmd if c != "2"]
        with self.assertRaisesRegex(ConfigGeneratorError, "decode-context-parallel-size"):
            generate_sparkcache_argv(
                bad_cmd, _SOURCE_ENV,
                target_checkpoint=_TARGET_ID,
                draft_checkpoint=_DRAFT_ID,
            )

    def test_rejects_wrong_dcp2_value(self) -> None:
        bad_cmd = list(_SOURCE_CMD)
        idx = bad_cmd.index("--decode-context-parallel-size")
        bad_cmd[idx + 1] = "4"
        with self.assertRaisesRegex(ConfigGeneratorError, "decode-context-parallel-size"):
            generate_sparkcache_argv(
                bad_cmd, _SOURCE_ENV,
                target_checkpoint=_TARGET_ID,
                draft_checkpoint=_DRAFT_ID,
            )

    def test_rejects_nonempty_prefix_cache_retention(self) -> None:
        bad_env = [e for e in _SOURCE_ENV if not e.startswith("VLLM_PREFIX_CACHE_RETENTION_INTERVAL")]
        bad_env.append("VLLM_PREFIX_CACHE_RETENTION_INTERVAL=100")
        with self.assertRaisesRegex(ConfigGeneratorError, "RETENTION_INTERVAL"):
            generate_sparkcache_argv(
                _SOURCE_CMD, bad_env,
                target_checkpoint=_TARGET_ID,
                draft_checkpoint=_DRAFT_ID,
            )

    def test_rejects_existing_kv_transfer_config_when_disabled(self) -> None:
        bad_cmd = _SOURCE_CMD + [gen.KVTc_ARG, '{"kv_connector":"other"}']
        with self.assertRaisesRegex(ConfigGeneratorError, "kv-transfer-config"):
            generate_sparkcache_argv(
                bad_cmd, _SOURCE_ENV,
                target_checkpoint=_TARGET_ID,
                draft_checkpoint=_DRAFT_ID,
                enabled=False,
            )

    def test_rejects_duplicate_kv_transfer_config_when_enabled(self) -> None:
        bad_cmd = _SOURCE_CMD + [gen.KVTc_ARG, '{"kv_connector":"other"}']
        with self.assertRaisesRegex(ConfigGeneratorError, "kv-transfer-config"):
            generate_sparkcache_argv(
                bad_cmd, _SOURCE_ENV,
                target_checkpoint=_TARGET_ID,
                draft_checkpoint=_DRAFT_ID,
                enabled=True,
            )


class VerifyViolationTests(unittest.TestCase):

    def test_rejects_legacy_enable_in_env(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
        )
        env_with_legacy = env + ["SPARK_CONTEXT_CACHE_ENABLE=1"]
        with self.assertRaisesRegex(ConfigGeneratorError, "SPARK_CONTEXT_CACHE_ENABLE"):
            verify_sparkcache_argv(cmd, env_with_legacy, expect_enabled=False)

    def test_rejects_enabled_without_disable_hybrid(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        # Remove --disable-hybrid-kv-cache-manager
        cmd_without_hma = [c for c in cmd if c != gen.HMA_ARG]
        with self.assertRaisesRegex(ConfigGeneratorError, "disable-hybrid"):
            verify_sparkcache_argv(cmd_without_hma, env, expect_enabled=True)

    def test_rejects_enabled_with_streaming_true_in_json(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        idx = cmd.index(gen.KVTc_ARG)
        parsed = json.loads(cmd[idx + 1])
        parsed["kv_connector_extra_config"]["spark_cache_streaming_snapshots"] = True
        cmd[idx + 1] = json.dumps(parsed)
        with self.assertRaisesRegex(ConfigGeneratorError, "streaming_snapshots"):
            verify_sparkcache_argv(cmd, env, expect_enabled=True)

    def test_rejects_enabled_with_wrong_kv_connector(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        idx = cmd.index(gen.KVTc_ARG)
        parsed = json.loads(cmd[idx + 1])
        parsed["kv_connector"] = "OtherConnector"
        cmd[idx + 1] = json.dumps(parsed)
        with self.assertRaisesRegex(ConfigGeneratorError, "kv_connector"):
            verify_sparkcache_argv(cmd, env, expect_enabled=True)

    def test_rejects_enabled_with_wrong_kv_role(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        idx = cmd.index(gen.KVTc_ARG)
        parsed = json.loads(cmd[idx + 1])
        parsed["kv_role"] = "kv_producer"
        cmd[idx + 1] = json.dumps(parsed)
        with self.assertRaisesRegex(ConfigGeneratorError, "kv_role"):
            verify_sparkcache_argv(cmd, env, expect_enabled=True)

    def test_rejects_enabled_with_wrong_load_failure_policy(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        idx = cmd.index(gen.KVTc_ARG)
        parsed = json.loads(cmd[idx + 1])
        parsed["kv_load_failure_policy"] = "abort"
        cmd[idx + 1] = json.dumps(parsed)
        with self.assertRaisesRegex(ConfigGeneratorError, "kv_load_failure_policy"):
            verify_sparkcache_argv(cmd, env, expect_enabled=True)

    def test_rejects_enabled_with_store_false(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        idx = cmd.index(gen.KVTc_ARG)
        parsed = json.loads(cmd[idx + 1])
        parsed["kv_connector_extra_config"]["spark_cache_store"] = False
        cmd[idx + 1] = json.dumps(parsed)
        with self.assertRaisesRegex(ConfigGeneratorError, "spark_cache_store"):
            verify_sparkcache_argv(cmd, env, expect_enabled=True)

    def test_rejects_enabled_with_non_hex_target_in_json(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        idx = cmd.index(gen.KVTc_ARG)
        parsed = json.loads(cmd[idx + 1])
        parsed["kv_connector_extra_config"]["spark_cache_target_checkpoint_sha256"] = "xyz"
        cmd[idx + 1] = json.dumps(parsed)
        with self.assertRaisesRegex(ConfigGeneratorError, "target_checkpoint_sha256"):
            verify_sparkcache_argv(cmd, env, expect_enabled=True)

    def test_rejects_disabled_with_kv_transfer_config(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
            enabled=True,
        )
        with self.assertRaisesRegex(ConfigGeneratorError, "kv-transfer-config"):
            verify_sparkcache_argv(cmd, env, expect_enabled=False)

    def test_rejects_disabled_with_disable_hybrid(self) -> None:
        cmd, env = generate_sparkcache_argv(
            _SOURCE_CMD, _SOURCE_ENV,
            target_checkpoint=_TARGET_ID,
            draft_checkpoint=_DRAFT_ID,
        )
        cmd_with_hma = cmd + [gen.HMA_ARG]
        with self.assertRaisesRegex(ConfigGeneratorError, "disable-hybrid"):
            verify_sparkcache_argv(cmd_with_hma, env, expect_enabled=False)


class CLITests(unittest.TestCase):
    """Test the CLI entry point via direct main() calls."""

    def _make_inspect_json(self) -> str:
        doc = [{
            "Config": {
                "Cmd": list(_SOURCE_CMD),
                "Env": list(_SOURCE_ENV),
            }
        }]
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(doc, f)
        f.close()
        return f.name

    def test_cli_disabled_succeeds_without_checkpoints(self) -> None:
        """A disabled CLI run should succeed even with no checkpoint identities."""
        inspect_path = self._make_inspect_json()
        try:
            old_stdin = sys.stdin
            with open(inspect_path, encoding="utf-8") as f:
                sys.stdin = f
                try:
                    ret = main([])
                finally:
                    sys.stdin = old_stdin
            self.assertEqual(ret, 0)
        finally:
            Path(inspect_path).unlink()

    def test_cli_enabled_fails_closed_without_checkpoints(self) -> None:
        """An enabled CLI run without checkpoint identities must fail."""
        inspect_path = self._make_inspect_json()
        try:
            old_stdin = sys.stdin
            with open(inspect_path, encoding="utf-8") as f:
                sys.stdin = f
                try:
                    with self.assertRaisesRegex(
                        ConfigGeneratorError, "target_checkpoint"
                    ):
                        main(["--enable"])
                finally:
                    sys.stdin = old_stdin
        finally:
            Path(inspect_path).unlink()

    def test_cli_enabled_succeeds_with_valid_checkpoints(self) -> None:
        """An enabled CLI run with valid 64-hex identities should succeed."""
        inspect_path = self._make_inspect_json()
        try:
            old_stdin = sys.stdin
            with open(inspect_path, encoding="utf-8") as f:
                sys.stdin = f
                try:
                    ret = main([
                        "--enable",
                        "--target-checkpoint", _TARGET_ID,
                        "--draft-checkpoint", _DRAFT_ID,
                    ])
                finally:
                    sys.stdin = old_stdin
            self.assertEqual(ret, 0)
        finally:
            Path(inspect_path).unlink()

    def test_cli_subprocess_disabled_succeeds(self) -> None:
        """Subprocess test: disabled CLI run succeeds without checkpoints."""
        inspect_path = self._make_inspect_json()
        try:
            with open(inspect_path, encoding="utf-8") as f:
                result = subprocess.run(
                    [sys.executable, str(_GENERATOR)],
                    stdin=f,
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(result.stdout)
            self.assertNotIn(gen.KVTc_ARG, output["cmd"])
        finally:
            Path(inspect_path).unlink()

    def test_cli_subprocess_enabled_fails_without_checkpoints(self) -> None:
        """Subprocess test: enabled CLI fails closed without checkpoint identities."""
        inspect_path = self._make_inspect_json()
        try:
            with open(inspect_path, encoding="utf-8") as f:
                result = subprocess.run(
                    [sys.executable, str(_GENERATOR), "--enable"],
                    stdin=f,
                    capture_output=True,
                    text=True,
                )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("target_checkpoint", result.stderr)
        finally:
            Path(inspect_path).unlink()


if __name__ == "__main__":
    unittest.main()
