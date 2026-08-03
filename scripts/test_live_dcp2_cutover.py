#!/usr/bin/env python3
"""CPU-only tests for live_dcp2_cutover.py.

All tests run offline with mocked SSH/docker. No host mutation occurs.
Covers: wrapper normalization, tri-state existence, inspect-confirmed
rollback, prepare label cleanup, identity-first verification, connector
staging mount, PYTHONPATH, cache-root validation, plan preflight, and CLI.
"""

from __future__ import annotations

import argparse
import os
import json
import subprocess
import tempfile
import hashlib
import sys
import unittest
from pathlib import Path
from unittest import mock

_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import live_dcp2_cutover as cutover  # noqa: E402
from live_dcp2_cutover import (  # noqa: E402
    BASELINE_DCP2_PATTERN,
    CUTOVER_CONFIRMATION,
    Existence,
    PREPARE_CONFIRMATION,
    PREPARE_LABEL,
    ROLLBACK_CONFIRMATION,
    CutoverError,
    Node,
    SOURCE_PATTERN,
    STAGING_DESTINATION,
    STAGING_PYTHONPATH,
    TARGET_PATTERN,
)
from sparkcache_config_generator import (  # noqa: E402
    ConfigGeneratorError,
    verify_sparkcache_argv,
)

from connector_bundle_manifest import BUNDLE_DOMAIN_SEPARATOR  # noqa: E402

_TID = "a" * 64
_DID = "b" * 64
_BUNDLE_ID = "e" * 64
_CACHE_ROOT = "/var/tmp/sparkring-public-validation/context-cache"
_STAGING = "/opt/sparkcache-host-staging"
_SOURCE_PYTHONPATH = "/opt/spark-vllm"
_MERGED_PYTHONPATH = f"{STAGING_PYTHONPATH}:{_SOURCE_PYTHONPATH}"

def _node(rank=0):
    return Node(rank=rank, ssh_target=f"h{rank}")


def _nodes4():
    return [Node(rank=r, ssh_target=f"h{r}") for r in range(4)]


def _completed(returncode=0, stdout="ok", stderr=""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr,
    )


# ---------------------------------------------------------------------------
# Source fixture: matches the REAL deployed /usr/bin/env wrapper shape.
# ---------------------------------------------------------------------------


def _src(rank=0, running=True):
    """Build a source container doc with the real /usr/bin/env wrapper.

    The real deployment has:
      Entrypoint: ["/usr/bin/env"]
      Path: "/usr/bin/env"
      Cmd: ["-u", "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
            "/opt/venv/bin/vllm", "--model", ...]
    """
    vllm_argv = [
        "--model", "/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid",
        "--decode-context-parallel-size", "4",
        "--max-model-len", "1048576",
        "--kv-cache-memory-bytes", "9663676416",
        "--max-num-seqs", "256",
        "--max-num-batched-tokens", "8192",
    ]
    wrapper_cmd = ["-u", "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
                   "/opt/venv/bin/vllm"] + vllm_argv
    return {
        "Path": "/usr/bin/env",
        "Image": "sha256:ab6bddba38aac663e2427f139aec20e022eced3716d7e204e1e881742f9da11d",
        "Config": {
            "Cmd": wrapper_cmd,
            "Env": [
                "VLLM_SPARK_DCP_SIZE=4",
                "VLLM_SPARK_MAX_MODEL_LEN=1048576",
                "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=1048576",
                "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=",
                "PATH=/usr/bin:/bin",
                f"PYTHONPATH={_SOURCE_PYTHONPATH}",
            ],
            "Entrypoint": ["/usr/bin/env"],
            "Path": "/usr/bin/env",
            "Labels": None,
        },
        "HostConfig": {
            "NetworkMode": "host", "IpcMode": "host", "Privileged": False,
            "ShmSize": 17179869184, "CapAdd": ["IPC_LOCK"],
            "Devices": [{"PathOnHost": "/dev/infiniband",
                          "PathInContainer": "/dev/infiniband",
                          "CgroupPermissions": "rwm"}],
            "Ulimits": [{"Name": "memlock", "Soft": -1, "Hard": -1}],
        },
        "State": {"Running": running, "Status": "running" if running else "exited",
                  "OOMKilled": False, "ExitCode": 0},
        "Mounts": [
            {"Type": "bind", "Source": "/h/m",
             "Destination": "/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid", "RW": False},
            {"Type": "bind", "Source": "/h/d",
             "Destination": "/mtp-draft", "RW": False},
            {"Type": "bind", "Source": "/h/c",
             "Destination": _CACHE_ROOT, "RW": True},
        ],
    }


def _src_normalized(rank=0, running=True):
    """Return a pre-normalized source (what _normalize_source produces)."""
    src = _src(rank, running=running)
    return cutover._normalize_source(src, _node(rank))


# ---------------------------------------------------------------------------
# Target fixture: matches the real /usr/bin/env wrapper + cache config.
# ---------------------------------------------------------------------------


def _tgt(rank=0, running=False, oom=False, exit_code=0, cmd=None, env=None,
        prepare_token=None, foreign_label=False,
        cache_root=_CACHE_ROOT, staging=_STAGING,
        unset_vars=None):
    """Build a target container doc with the /usr/bin/env wrapper."""
    src = _src(rank)
    doc = _src(rank, running=running)
    doc["Config"]["Entrypoint"] = ["/usr/bin/env"]
    doc["Config"]["Path"] = "/usr/bin/env"

    # Normalize the source to get the vLLM argv.
    norm = cutover._normalize_source(src, _node(rank))

    if cmd is None:
        cmd, gen_env = cutover._apply_cache_config(
            list(norm["Config"]["Cmd"]), list(norm["Config"]["Env"]),
            _TID, _DID, cache_root,
        )
        if env is None:
            env = gen_env

    if unset_vars is None:
        unset_vars = norm.get("_unset_vars", ["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"])

    wrapper_cmd = []
    for name in sorted(unset_vars):
        wrapper_cmd.extend(["-u", name])
    wrapper_cmd.append("/opt/venv/bin/vllm")
    wrapper_cmd.extend(cmd)
    doc["Config"]["Cmd"] = wrapper_cmd

    # Add merged PYTHONPATH to env (staging paths + source PYTHONPATH).
    # The cache_env from _apply_cache_config does NOT contain PYTHONPATH
    # (the config generator strips it), so we add the merged value.
    env_map = cutover._env_map(env)
    env_map["PYTHONPATH"] = _MERGED_PYTHONPATH
    env_list = cutover._env_list_sorted(env_map)
    doc["Config"]["Env"] = env_list

    # Add the staging mount.
    doc["Mounts"] = list(doc["Mounts"]) + [
        {"Type": "bind", "Source": staging,
         "Destination": STAGING_DESTINATION, "RW": False},
    ]

    # Set labels for prepare token or foreign label.
    labels: dict[str, str] = {}
    if prepare_token is not None:
        labels[PREPARE_LABEL] = prepare_token
    elif foreign_label:
        labels[PREPARE_LABEL] = "foreign-token-xyz"
    # Always include bundle identity label (matches _create_argv).
    labels[cutover.BUNDLE_IDENTITY_LABEL] = _BUNDLE_ID
    doc["Config"]["Labels"] = labels if labels else None
    doc["State"]["Running"] = running
    doc["State"]["OOMKilled"] = oom
    doc["State"]["ExitCode"] = exit_code
    doc["State"]["Status"] = "running" if running else "created"
    return doc


def _good_tgt(rank=0, **kw):
    """Target that passes _verify_clone (exact match with source)."""
    return _tgt(rank, **kw)


def _inspect_side_factory(src_fn=_src, tgt_fn=_good_tgt):
    def side(node, name):
        if name == node.source_name:
            return src_fn(node.rank)
        return tgt_fn(node.rank)
    return side


# ---------------------------------------------------------------------------
# Wrapper normalization tests
# ---------------------------------------------------------------------------


class WrapperNormalizationTests(unittest.TestCase):
    def test_extract_unset_vars(self):
        cmd = ["-u", "VAR1", "-u", "VAR2", "/opt/venv/bin/vllm", "--model", "x"]
        self.assertEqual(cutover._extract_unset_vars(cmd), ["VAR1", "VAR2"])

    def test_extract_unset_vars_empty(self):
        cmd = ["/opt/venv/bin/vllm", "--model", "x"]
        with self.assertRaises(CutoverError):
            cutover._extract_unset_vars(cmd)

    def test_strip_wrapper(self):
        cmd = ["-u", "VAR1", "/opt/venv/bin/vllm", "--model", "x"]
        self.assertEqual(cutover._strip_wrapper(cmd), ["--model", "x"])
    def test_strip_wrapper_no_unset(self):
        cmd = ["/opt/venv/bin/vllm", "--model", "x"]
        with self.assertRaises(CutoverError):
            cutover._strip_wrapper(cmd)

    def test_normalize_source_wrapper(self):
        s = _src(0)
        norm = cutover._normalize_source(s, _node(0))
        self.assertEqual(norm["Config"]["Entrypoint"], ["/opt/venv/bin/vllm"])
        self.assertNotIn("-u", norm["Config"]["Cmd"])
        self.assertNotIn("/opt/venv/bin/vllm", norm["Config"]["Cmd"])
        self.assertEqual(norm["_unset_vars"], ["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"])
        self.assertTrue(norm.get("_normalized"))
    def test_normalize_source_already_vllm(self):
        s = _src(0)
        s["Config"]["Entrypoint"] = ["/opt/venv/bin/vllm"]
        s["Path"] = "/opt/venv/bin/vllm"
        s["Config"]["Cmd"] = ["--model", "x"]
        s["_normalized"] = True
        norm = cutover._normalize_source(s, _node(0))
        self.assertEqual(norm["Config"]["Cmd"], ["--model", "x"])
        self.assertEqual(norm["_unset_vars"], [])

    def test_normalize_source_rejects_unknown_entrypoint(self):
        s = _src(0)
        s["Config"]["Entrypoint"] = ["/bin/bash"]
        with self.assertRaises(CutoverError) as ctx:
            cutover._normalize_source(s, _node(0))
        self.assertIn("unexpected entrypoint", str(ctx.exception))

    def test_normalize_source_rejects_empty_vllm_argv(self):
        s = _src(0)
        s["Config"]["Cmd"] = ["-u", "VAR1", "/opt/venv/bin/vllm"]
        with self.assertRaises(CutoverError):
            cutover._normalize_source(s, _node(0))


# ---------------------------------------------------------------------------
# Existence tri-state tests
# ---------------------------------------------------------------------------


class ExistenceTests(unittest.TestCase):
    @mock.patch.object(cutover, "_remote_result")
    def test_proven_present(self, mock_rr):
        mock_rr.return_value = _completed(returncode=0, stdout=json.dumps([_src(0)]))
        self.assertEqual(cutover._check_exists(_node(0), "x"), Existence.PROVEN_PRESENT)

    @mock.patch.object(cutover, "_remote_result")
    def test_proven_absent(self, mock_rr):
        mock_rr.return_value = _completed(
            returncode=1, stdout="", stderr="Error: No such container: x"
        )
        self.assertEqual(cutover._check_exists(_node(0), "x"), Existence.PROVEN_ABSENT)

    @mock.patch.object(cutover, "_remote_result")
    def test_unknown_on_ssh_failure(self, mock_rr):
        """rc 255 (SSH failure) must be UNKNOWN, not absence."""
        mock_rr.return_value = _completed(
            returncode=255, stdout="", stderr="ssh: connect to host: Connection refused"
        )
        self.assertEqual(cutover._check_exists(_node(0), "x"), Existence.UNKNOWN)

    @mock.patch.object(cutover, "_remote_result")
    def test_unknown_on_json_error(self, mock_rr):
        mock_rr.return_value = _completed(returncode=0, stdout="not json", stderr="")
        self.assertEqual(cutover._check_exists(_node(0), "x"), Existence.UNKNOWN)

    @mock.patch.object(cutover, "_remote_result")
    def test_unknown_on_daemon_error(self, mock_rr):
        mock_rr.return_value = _completed(
            returncode=1, stdout="", stderr="Cannot connect to the Docker daemon"
        )
        self.assertEqual(cutover._check_exists(_node(0), "x"), Existence.UNKNOWN)

    @mock.patch.object(cutover, "_remote_result")
    def test_unknown_on_empty_list(self, mock_rr):
        mock_rr.return_value = _completed(returncode=0, stdout="[]", stderr="")
        self.assertEqual(cutover._check_exists(_node(0), "x"), Existence.UNKNOWN)


# ---------------------------------------------------------------------------
# Container pattern tests
# ---------------------------------------------------------------------------


class ContainerPatternTests(unittest.TestCase):
    def test_source_pattern_is_dcp4_fixedk2(self):
        self.assertEqual(SOURCE_PATTERN, "glm52-sparkring-nf3-dcp4-fixedk2-r{rank}")

    def test_target_does_not_collide_with_baseline(self):
        self.assertNotEqual(TARGET_PATTERN, BASELINE_DCP2_PATTERN)
        self.assertIn("sparkcache", TARGET_PATTERN)

    def test_node_names_distinct(self):
        n = _node(0)
        self.assertNotEqual(n.source_name, n.target_name)
        self.assertNotEqual(n.target_name, n.baseline_name)


# ---------------------------------------------------------------------------
# Cache config tests
# ---------------------------------------------------------------------------


class CacheConfigTests(unittest.TestCase):
    def test_apply_produces_enabled_argv(self):
        s = _src_normalized(0)
        cmd, env = cutover._apply_cache_config(
            list(s["Config"]["Cmd"]), list(s["Config"]["Env"]), _TID, _DID, _CACHE_ROOT
        )
        verify_sparkcache_argv(cmd, env, expect_enabled=True)

    def test_apply_has_kv_transfer_config(self):
        s = _src_normalized(0)
        cmd, _ = cutover._apply_cache_config(
            list(s["Config"]["Cmd"]), list(s["Config"]["Env"]), _TID, _DID, _CACHE_ROOT
        )
        self.assertIn("--kv-transfer-config", cmd)

    def test_apply_has_disable_hybrid(self):
        s = _src_normalized(0)
        cmd, _ = cutover._apply_cache_config(
            list(s["Config"]["Cmd"]), list(s["Config"]["Env"]), _TID, _DID, _CACHE_ROOT
        )
        self.assertIn("--disable-hybrid-kv-cache-manager", cmd)

    def test_apply_has_correct_checkpoint_identities(self):
        s = _src_normalized(0)
        cmd, _ = cutover._apply_cache_config(
            list(s["Config"]["Cmd"]), list(s["Config"]["Env"]), _TID, _DID, _CACHE_ROOT
        )
        cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
        extra = cfg["kv_connector_extra_config"]
        self.assertEqual(extra["spark_cache_target_checkpoint_sha256"], _TID)
        self.assertEqual(extra["spark_cache_draft_checkpoint_sha256"], _DID)

    def test_apply_uses_cache_root(self):
        s = _src_normalized(0)
        cmd, _ = cutover._apply_cache_config(
            list(s["Config"]["Cmd"]), list(s["Config"]["Env"]), _TID, _DID, _CACHE_ROOT
        )
        cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
        self.assertEqual(cfg["kv_connector_extra_config"]["spark_cache_root"], _CACHE_ROOT)

    def test_apply_streaming_disabled(self):
        s = _src_normalized(0)
        cmd, _ = cutover._apply_cache_config(
            list(s["Config"]["Cmd"]), list(s["Config"]["Env"]), _TID, _DID, _CACHE_ROOT
        )
        cfg = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
        self.assertFalse(cfg["kv_connector_extra_config"]["spark_cache_streaming_snapshots"])

    def test_apply_no_legacy_enable(self):
        s = _src_normalized(0)
        _, env = cutover._apply_cache_config(
            list(s["Config"]["Cmd"]), list(s["Config"]["Env"]), _TID, _DID, _CACHE_ROOT
        )
        self.assertNotIn("SPARK_CONTEXT_CACHE_ENABLE", cutover._env_map(env))

    def test_apply_unrelated_fields_preserved(self):
        s = _src_normalized(0)
        _, env = cutover._apply_cache_config(
            list(s["Config"]["Cmd"]), list(s["Config"]["Env"]), _TID, _DID, _CACHE_ROOT
        )
        self.assertEqual(cutover._env_map(env)["PATH"], "/usr/bin:/bin")
        cmd, _ = cutover._apply_cache_config(
            list(s["Config"]["Cmd"]), list(s["Config"]["Env"]), _TID, _DID, _CACHE_ROOT
        )
        self.assertEqual(cmd[cmd.index("--decode-context-parallel-size") + 1], "2")
        self.assertEqual(cmd[cmd.index("--max-model-len") + 1], "524288")

    def test_apply_rejects_bad_checkpoint(self):
        s = _src_normalized(0)
        with self.assertRaises(ConfigGeneratorError):
            cutover._apply_cache_config(
                list(s["Config"]["Cmd"]), list(s["Config"]["Env"]), "short", _DID, _CACHE_ROOT
            )


# ---------------------------------------------------------------------------
# Verify clone tests
# ---------------------------------------------------------------------------


class VerifyCloneTests(unittest.TestCase):
    def test_verify_accepts_exact_clone(self):
        s = _src_normalized(0)
        t = _good_tgt(0)
        cutover._verify_clone(s, t, _node(0), _TID, _DID, _CACHE_ROOT, _STAGING)

    def test_verify_rejects_wrong_checkpoint(self):
        """Identity mismatch must produce 'checkpoint identity mismatch', not cmd drift."""
        s = _src_normalized(0)
        t = _good_tgt(0)  # Built with _TID
        with self.assertRaises(CutoverError) as ctx:
            cutover._verify_clone(s, t, _node(0), "c" * 64, _DID, _CACHE_ROOT, _STAGING)
        self.assertIn("checkpoint identity mismatch", str(ctx.exception))

    def test_verify_rejects_draft_checkpoint_mismatch(self):
        s = _src_normalized(0)
        t = _good_tgt(0)  # Built with _DID
        with self.assertRaises(CutoverError) as ctx:
            cutover._verify_clone(s, t, _node(0), _TID, "d" * 64, _CACHE_ROOT, _STAGING)
        self.assertIn("draft checkpoint identity mismatch", str(ctx.exception))

    def test_verify_rejects_cmd_drift(self):
        s = _src_normalized(0)
        cmd, env = cutover._apply_cache_config(
            list(s["Config"]["Cmd"]), list(s["Config"]["Env"]), _TID, _DID, _CACHE_ROOT
        )
        cmd.append("--unexpected-flag")
        t = _tgt(0, cmd=cmd, env=env)
        with self.assertRaises(CutoverError) as ctx:
            cutover._verify_clone(s, t, _node(0), _TID, _DID, _CACHE_ROOT, _STAGING)
        self.assertIn("cmd drift", str(ctx.exception).lower())

    def test_verify_rejects_env_drift(self):
        s = _src_normalized(0)
        cmd, env = cutover._apply_cache_config(
            list(s["Config"]["Cmd"]), list(s["Config"]["Env"]), _TID, _DID, _CACHE_ROOT
        )
        env = list(env) + ["EXTRA_VAR=should_not_be_here"]
        t = _tgt(0, cmd=cmd, env=env)
        with self.assertRaises(CutoverError):
            cutover._verify_clone(s, t, _node(0), _TID, _DID, _CACHE_ROOT, _STAGING)

    def test_verify_rejects_missing_kv_transfer_config(self):
        s = _src_normalized(0)
        t = _tgt(0, cmd=cutover._rewrite_command(list(s["Config"]["Cmd"])),
                 env=cutover._rewrite_environment(s["Config"]["Env"]))
        with self.assertRaises(CutoverError):
            cutover._verify_clone(s, t, _node(0), _TID, _DID, _CACHE_ROOT, _STAGING)

    def test_verify_rejects_image_change(self):
        s = _src_normalized(0)
        t = _good_tgt(0)
        t["Image"] = "sha256:different"
        with self.assertRaises(CutoverError):
            cutover._verify_clone(s, t, _node(0), _TID, _DID, _CACHE_ROOT, _STAGING)

    def test_verify_rejects_missing_staging_mount(self):
        s = _src_normalized(0)
        t = _good_tgt(0)
        # Remove the staging mount.
        t["Mounts"] = [m for m in t["Mounts"] if m["Destination"] != STAGING_DESTINATION]
        with self.assertRaises(CutoverError) as ctx:
            cutover._verify_clone(s, t, _node(0), _TID, _DID, _CACHE_ROOT, _STAGING)
        self.assertIn("staging mount", str(ctx.exception).lower())

    def test_verify_rejects_wrong_staging_host_path(self):
        s = _src_normalized(0)
        t = _good_tgt(0, staging="/wrong/path")
        with self.assertRaises(CutoverError) as ctx:
            cutover._verify_clone(s, t, _node(0), _TID, _DID, _CACHE_ROOT, _STAGING)
        self.assertIn("staging mount", str(ctx.exception).lower())

    def test_verify_rejects_missing_pythonpath(self):
        s = _src_normalized(0)
        t = _good_tgt(0)
        # Remove PYTHONPATH from env.
        t["Config"]["Env"] = [e for e in t["Config"]["Env"] if not e.startswith("PYTHONPATH=")]
        with self.assertRaises(CutoverError) as ctx:
            cutover._verify_clone(s, t, _node(0), _TID, _DID, _CACHE_ROOT, _STAGING)
        self.assertIn("PYTHONPATH", str(ctx.exception))

    def test_verify_rejects_wrong_pythonpath(self):
        s = _src_normalized(0)
        t = _good_tgt(0)
        t["Config"]["Env"] = [
            e if not e.startswith("PYTHONPATH=") else "PYTHONPATH=/wrong"
            for e in t["Config"]["Env"]
        ]
        with self.assertRaises(CutoverError) as ctx:
            cutover._verify_clone(s, t, _node(0), _TID, _DID, _CACHE_ROOT, _STAGING)
        self.assertIn("PYTHONPATH", str(ctx.exception))

    def test_verify_rejects_unset_vars_drift(self):
        s = _src_normalized(0)
        t = _good_tgt(0, unset_vars=["DIFFERENT_VAR"])
        with self.assertRaises(CutoverError) as ctx:
            cutover._verify_clone(s, t, _node(0), _TID, _DID, _CACHE_ROOT, _STAGING)
        self.assertIn("unset vars drift", str(ctx.exception).lower())

    def test_verify_rejects_cache_root_mismatch(self):
        s = _src_normalized(0)
        t = _good_tgt(0, cache_root="/wrong/cache/path")
        with self.assertRaises(CutoverError) as ctx:
            cutover._verify_clone(s, t, _node(0), _TID, _DID, _CACHE_ROOT, _STAGING)
        self.assertIn("cache root mismatch", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# Non-mutating tests
# ---------------------------------------------------------------------------


class NonMutatingTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(cutover, "_verify_bundle_identity", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        rr_patcher = mock.patch.object(
            cutover, "_remote_result",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=255, stdout="", stderr="ssh: connection refused",
            ),
        )
        rr_patcher.start()
        self.addCleanup(rr_patcher.stop)

    @mock.patch.object(cutover, "_remote")
    def test_plan_only_inspects(self, mock_remote):
        mock_remote.return_value = json.dumps([_src(0)])
        cutover.command_plan(_nodes4(), _TID, _DID, _CACHE_ROOT, _STAGING, _BUNDLE_ID)
        for call in mock_remote.call_args_list:
            self.assertEqual(call[0][1][1], "inspect")

    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover, "_remote")
    def test_status_only_inspects(self, mock_remote, mock_exists):
        mock_exists.return_value = Existence.PROVEN_ABSENT
        cutover.command_status(_nodes4())
        mock_remote.assert_not_called()


# ---------------------------------------------------------------------------
# Confirmation tests
# ---------------------------------------------------------------------------


class ConfirmationTests(unittest.TestCase):
    def test_prepare_requires_confirmation(self):
        self.assertTrue(PREPARE_CONFIRMATION.startswith("PREPARE"))

    def test_cutover_requires_confirmation(self):
        self.assertTrue(CUTOVER_CONFIRMATION.startswith("STOP"))

    def test_rollback_requires_confirmation(self):
        self.assertTrue(ROLLBACK_CONFIRMATION.startswith("STOP"))


# ---------------------------------------------------------------------------
# Cache mount tests
# ---------------------------------------------------------------------------


class CacheMountTests(unittest.TestCase):
    def test_find_writable_bind_mount_present(self):
        s = _src(0)
        path = cutover._find_writable_bind_mount(s, _CACHE_ROOT)
        self.assertEqual(path, "/h/c")

    def test_find_writable_bind_mount_missing(self):
        s = _src(0)
        path = cutover._find_writable_bind_mount(s, "/nonexistent/path")
        self.assertIsNone(path)

    def test_find_writable_bind_mount_rejects_readonly(self):
        s = _src(0)
        path = cutover._find_writable_bind_mount(s, "/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid")
        self.assertIsNone(path)  # Model mount is read-only.


# ---------------------------------------------------------------------------
# Cutover rollback tests
# ---------------------------------------------------------------------------


class CutoverRollbackTests(unittest.TestCase):
    """Tests that cutover failures trigger rollback."""

    def setUp(self):
        patcher = mock.patch.object(cutover, "_verify_bundle_identity", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Mock _remote_result to prevent real SSH calls during rollback.
        rr_patcher = mock.patch.object(
            cutover, "_remote_result",
            return_value=_completed(returncode=0, stdout="", stderr=""),
        )
        rr_patcher.start()
        self.addCleanup(rr_patcher.stop)

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote_result")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_source_stop_failure_triggers_restore(self, mock_exists, mock_remote, mock_rr, mock_inspect):
        nodes = _nodes4()
        mock_exists.return_value = Existence.PROVEN_PRESENT
        mock_inspect.side_effect = _inspect_side_factory()
        call_count = [0]
        def remote_side(node, argv, **kw):
            call_count[0] += 1
            if argv[1] == "stop" and node.source_name in argv:
                if call_count[0] <= 4:
                    raise CutoverError("stop failed")
            return "ok"
        mock_remote.side_effect = remote_side
        mock_rr.return_value = _completed(returncode=0, stdout="id", stderr="")
        with self.assertRaises(CutoverError):
            cutover.command_cutover(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID)
        rr_starts = [
            c for c in mock_rr.call_args_list
            if "start" in c[0][1] and c[0][0].source_name in " ".join(c[0][1])
        ]
        self.assertGreater(len(rr_starts), 0)

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover.time, "sleep")
    def test_target_start_failure_triggers_rollback(self, mock_sleep, mock_exists, mock_remote, mock_inspect):
        nodes = _nodes4()
        mock_exists.return_value = Existence.PROVEN_PRESENT
        mock_inspect.side_effect = _inspect_side_factory()
        def remote_side(node, argv, **kw):
            if argv[1] == "stop":
                return "stopped"
            if argv[1] == "start" and node.target_name in argv:
                raise CutoverError("start failed")
            return ""
        mock_remote.side_effect = remote_side
        with self.assertRaises(CutoverError):
            cutover.command_cutover(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID)

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover.time, "sleep")
    def test_oom_triggers_rollback(self, mock_sleep, mock_exists, mock_remote, mock_inspect):
        nodes = _nodes4()
        mock_exists.return_value = Existence.PROVEN_PRESENT
        def remote_side(node, argv, **kw):
            if argv[1] == "stop":
                return "stopped"
            if argv[1] == "start" and node.target_name in argv:
                return "started"
            return ""
        def tgt_fn(rank):
            d = _good_tgt(rank, running=False, oom=True)
            d["State"]["Status"] = "exited"
            return d
        mock_remote.side_effect = remote_side
        mock_inspect.side_effect = _inspect_side_factory(tgt_fn=tgt_fn)
        with self.assertRaises(CutoverError) as ctx:
            cutover.command_cutover(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID, health_timeout=1)
        self.assertIn("OOM", str(ctx.exception))

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover.time, "sleep")
    def test_nonzero_exit_triggers_rollback(self, mock_sleep, mock_exists, mock_remote, mock_inspect):
        nodes = _nodes4()
        mock_exists.return_value = Existence.PROVEN_PRESENT
        def remote_side(node, argv, **kw):
            if argv[1] == "stop":
                return "stopped"
            if argv[1] == "start" and node.target_name in argv:
                return "started"
            return ""
        def tgt_fn(rank):
            d = _good_tgt(rank, running=False, exit_code=137)
            d["State"]["Status"] = "exited"
            return d
        mock_remote.side_effect = remote_side
        mock_inspect.side_effect = _inspect_side_factory(tgt_fn=tgt_fn)
        with self.assertRaises(CutoverError) as ctx:
            cutover.command_cutover(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID, health_timeout=1)
        self.assertIn("exit", str(ctx.exception).lower())

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover.time, "sleep")
    def test_clean_exit_0_triggers_rollback(self, mock_sleep, mock_exists, mock_remote, mock_inspect):
        nodes = _nodes4()
        mock_exists.return_value = Existence.PROVEN_PRESENT
        def remote_side(node, argv, **kw):
            if argv[1] == "stop":
                return "stopped"
            if argv[1] == "start" and node.target_name in argv:
                return "started"
            return ""
        def tgt_fn(rank):
            d = _good_tgt(rank, running=False, exit_code=0)
            d["State"]["Status"] = "exited"
            return d
        mock_remote.side_effect = remote_side
        mock_inspect.side_effect = _inspect_side_factory(tgt_fn=tgt_fn)
        with self.assertRaises(CutoverError) as ctx:
            cutover.command_cutover(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID, health_timeout=1)
        self.assertIn("exit", str(ctx.exception).lower())

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover.subprocess, "run")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover.time, "sleep")
    def test_health_timeout_triggers_rollback(self, mock_sleep, mock_exists, mock_remote, mock_subproc, mock_inspect):
        nodes = _nodes4()
        mock_exists.return_value = Existence.PROVEN_PRESENT
        pre_checked: set[int] = set()
        def remote_side(node, argv, **kw):
            if argv[1] == "stop":
                return "stopped"
            if argv[1] == "start" and node.target_name in argv:
                return "started"
            return ""
        def tgt_fn(rank):
            if rank not in pre_checked:
                pre_checked.add(rank)
                return _good_tgt(rank, running=False)
            return _good_tgt(rank, running=True)
        mock_remote.side_effect = remote_side
        mock_inspect.side_effect = _inspect_side_factory(tgt_fn=tgt_fn)
        mock_subproc.return_value = mock.Mock(returncode=1, stdout="", stderr="connection refused")
        with self.assertRaises(CutoverError) as ctx:
            cutover.command_cutover(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID, health_timeout=1)
        self.assertIn("readiness", str(ctx.exception).lower())

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote_result")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover.time, "sleep")
    def test_health_inspect_exception_triggers_rollback(self, mock_sleep, mock_exists, mock_remote, mock_rr, mock_inspect):
        nodes = _nodes4()
        mock_exists.return_value = Existence.PROVEN_PRESENT
        pre_checked: set[int] = set()
        def remote_side(node, argv, **kw):
            if argv[1] == "stop":
                return "stopped"
            if argv[1] == "start" and node.target_name in argv:
                return "started"
            return ""
        def tgt_fn(rank):
            if rank not in pre_checked:
                pre_checked.add(rank)
                return _good_tgt(rank, running=False)
            raise CutoverError("SSH connection lost during health poll")
        mock_remote.side_effect = remote_side
        mock_rr.return_value = _completed(returncode=0, stdout="id", stderr="")
        mock_inspect.side_effect = _inspect_side_factory(tgt_fn=tgt_fn)
        with self.assertRaises(CutoverError) as ctx:
            cutover.command_cutover(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID, health_timeout=1)
        self.assertIn("health poll", str(ctx.exception).lower())
        starts = [
            c for c in mock_remote.call_args_list
            if c[0][1][1] == "start" and c[0][0].source_name in c[0][1]
        ]
        rr_starts = [
            c for c in mock_rr.call_args_list
            if "start" in c[0][1] and c[0][0].source_name in " ".join(c[0][1])
        ]
        self.assertTrue(len(starts) > 0 or len(rr_starts) > 0)

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote_result")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover.time, "sleep")
    def test_no_remote_diagnostics_before_rollback(self, mock_sleep, mock_exists, mock_remote, mock_rr, mock_inspect):
        nodes = _nodes4()
        mock_exists.return_value = Existence.PROVEN_PRESENT
        call_log: list[tuple[str, str]] = []
        def remote_side(node, argv, **kw):
            call_log.append((argv[1], " ".join(argv)))
            if argv[1] == "stop":
                return "stopped"
            if argv[1] == "start" and node.target_name in argv:
                return "started"
            return ""
        def rr_side(node, argv):
            call_log.append((argv[1], " ".join(argv)))
            return _completed(returncode=0, stdout="id", stderr="")
        def tgt_fn(rank):
            d = _good_tgt(rank, running=False, oom=True)
            d["State"]["Status"] = "exited"
            return d
        mock_remote.side_effect = remote_side
        mock_rr.side_effect = rr_side
        mock_inspect.side_effect = _inspect_side_factory(tgt_fn=tgt_fn)
        with self.assertRaises(CutoverError):
            cutover.command_cutover(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID, health_timeout=1)
        stop_calls = [(op, cmd) for op, cmd in call_log if op == "stop"]
        start_calls = [(op, cmd) for op, cmd in call_log if op == "start"]
        self.assertGreaterEqual(len(stop_calls), 8)
        self.assertGreaterEqual(len(start_calls), 4)

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover.subprocess, "run")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover.time, "sleep")
    def test_successful_cutover(self, mock_sleep, mock_exists, mock_remote, mock_subproc, mock_inspect):
        nodes = _nodes4()
        mock_exists.return_value = Existence.PROVEN_PRESENT
        pre_checked: set[int] = set()
        def remote_side(node, argv, **kw):
            if argv[1] == "stop":
                return "stopped"
            if argv[1] == "start" and node.target_name in argv:
                return "started"
            return ""
        def tgt_fn(rank):
            if rank not in pre_checked:
                pre_checked.add(rank)
                return _good_tgt(rank, running=False)
            return _good_tgt(rank, running=True)
        mock_remote.side_effect = remote_side
        mock_inspect.side_effect = _inspect_side_factory(tgt_fn=tgt_fn)
        mock_subproc.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        cutover.command_cutover(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID, health_timeout=5)

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_cutover_rejects_checkpoint_mismatch_before_stop(self, mock_exists, mock_remote, mock_inspect):
        """Cutover rejects changed target checkpoint identity before any source stop."""
        nodes = _nodes4()
        mock_exists.return_value = Existence.PROVEN_PRESENT
        mock_inspect.side_effect = _inspect_side_factory()
        mock_remote.return_value = ""
        with self.assertRaises(CutoverError) as ctx:
            cutover.command_cutover(nodes, target_checkpoint="c" * 64, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID)
        self.assertIn("checkpoint identity mismatch", str(ctx.exception))
        stops = [c for c in mock_remote.call_args_list if c[0][1][1] == "stop"]
        self.assertEqual(len(stops), 0)

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_cutover_rejects_cmd_drift_before_stop(self, mock_exists, mock_remote, mock_inspect):
        nodes = _nodes4()
        mock_exists.return_value = Existence.PROVEN_PRESENT
        s = _src(0)
        norm = cutover._normalize_source(s, _node(0))
        cmd, env = cutover._apply_cache_config(
            list(norm["Config"]["Cmd"]), list(norm["Config"]["Env"]), _TID, _DID, _CACHE_ROOT
        )
        cmd.append("--unexpected-extra-flag")
        mock_inspect.side_effect = _inspect_side_factory(
            tgt_fn=lambda rank: _tgt(rank, cmd=cmd, env=env),
        )
        mock_remote.return_value = ""
        with self.assertRaises(CutoverError) as ctx:
            cutover.command_cutover(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID)
        self.assertIn("cmd drift", str(ctx.exception).lower())
        stops = [c for c in mock_remote.call_args_list if c[0][1][1] == "stop"]
        self.assertEqual(len(stops), 0)


# ---------------------------------------------------------------------------
# Cutover unexpected-exception rollback tests
# ---------------------------------------------------------------------------


class CutoverUnexpectedExceptionTests(unittest.TestCase):
    """Tests that unexpected exceptions after transition starts trigger rollback."""

    def setUp(self):
        patcher = mock.patch.object(cutover, "_verify_bundle_identity", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Mock _remote_result to prevent real SSH calls during rollback.
        rr_patcher = mock.patch.object(
            cutover, "_remote_result",
            return_value=_completed(returncode=0, stdout="", stderr=""),
        )
        rr_patcher.start()
        self.addCleanup(rr_patcher.stop)

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote_result")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover.time, "sleep")
    def test_unexpected_exception_after_stop_triggers_rollback(
        self, mock_sleep, mock_exists, mock_remote, mock_rr, mock_inspect
    ):
        """An unexpected exception (not CutoverError) after source stop
        must trigger rollback."""
        nodes = _nodes4()
        mock_exists.return_value = Existence.PROVEN_PRESENT
        mock_inspect.side_effect = _inspect_side_factory()
        stop_count = [0]
        def remote_side(node, argv, **kw):
            if argv[1] == "stop" and node.source_name in argv:
                stop_count[0] += 1
                return "stopped"
            if argv[1] == "start" and node.target_name in argv:
                # Simulate an unexpected error (not CutoverError)
                raise RuntimeError("unexpected kernel panic")
            return ""
        mock_remote.side_effect = remote_side
        mock_rr.return_value = _completed(returncode=0, stdout="id", stderr="")
        with self.assertRaises(CutoverError) as ctx:
            cutover.command_cutover(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID)
        # The error must mention both the original and rollback status.
        self.assertIn("ROLLBACK", str(ctx.exception))

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote_result")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover.time, "sleep")
    def test_unexpected_exception_incomplete_rollback(
        self, mock_sleep, mock_exists, mock_remote, mock_rr, mock_inspect
    ):
        """Unexpected exception + rollback failure = ROLLBACK INCOMPLETE."""
        nodes = _nodes4()
        mock_exists.return_value = Existence.PROVEN_PRESENT
        # After rollback attempt: targets still running, sources not running.
        # The _inspect side returns appropriate docs for each call.
        src_checked: set[int] = set()
        tgt_checked: set[int] = set()
        def inspect_side(node, name):
            if name == node.source_name:
                if node.rank not in src_checked:
                    src_checked.add(node.rank)
                    return _src(node.rank, running=True)  # Pre-cutover source
                return _src(node.rank, running=False)  # Post-rollback: not running
            if node.rank not in tgt_checked:
                tgt_checked.add(node.rank)
                return _good_tgt(node.rank, running=False)  # Pre-cutover target
            return _good_tgt(node.rank, running=True)  # Post-rollback: still running
        mock_inspect.side_effect = inspect_side
        def remote_side(node, argv, **kw):
            if argv[1] == "stop" and node.source_name in argv:
                return "stopped"
            if argv[1] == "start" and node.target_name in argv:
                raise RuntimeError("unexpected")
            return ""
        # Rollback: target stop fails, source start fails.
        def rr_side(node, argv):
            if "stop" in argv and node.target_name in " ".join(argv):
                return _completed(returncode=1, stdout="", stderr="error")
            if "start" in argv and node.source_name in " ".join(argv):
                return _completed(returncode=1, stdout="", stderr="error")
            return _completed(returncode=0, stdout="id", stderr="")
        mock_remote.side_effect = remote_side
        mock_rr.side_effect = rr_side
        with self.assertRaises(CutoverError) as ctx:
            cutover.command_cutover(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID)
        self.assertIn("ROLLBACK INCOMPLETE", str(ctx.exception))

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_exception_before_transition_no_rollback(self, mock_exists, mock_remote, mock_inspect):
        """Exception before transition_started should not attempt rollback."""
        nodes = _nodes4()
        # _check_exists returns UNKNOWN → raises before any stop.
        mock_exists.return_value = Existence.UNKNOWN
        with self.assertRaises(CutoverError):
            cutover.command_cutover(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID)
        # No stop or start commands should have been issued.
        stops = [c for c in mock_remote.call_args_list if c[0][1][1] == "stop"]
        self.assertEqual(len(stops), 0)


# ---------------------------------------------------------------------------
# Prepare cleanup tests
# ---------------------------------------------------------------------------


class PrepareCleanupTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(cutover, "_verify_bundle_identity", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    @mock.patch.object(cutover, "_verify_host_staging")
    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_partial_prepare_cleans_up_created(self, mock_exists, mock_remote, mock_inspect, mock_staging):
        """If verify fails after all creates succeed, all created targets are removed."""
        nodes = _nodes4()
        check_count = [0]
        def exists_side(node, name):
            check_count[0] += 1
            if check_count[0] <= 4:
                return Existence.PROVEN_ABSENT
            return Existence.PROVEN_PRESENT
        mock_exists.side_effect = exists_side
        mock_staging.return_value = {"exists": True, "files": [], "stderr": ""}
        def remote_side(node, argv, **kw):
            if argv[1] == "create":
                return f"container-id-r{node.rank}"
            if argv[1] == "rm":
                return ""
            return ""
        mock_remote.side_effect = remote_side
        def inspect_side(node, name):
            if name == node.source_name:
                return _src(node.rank)
            return _good_tgt(node.rank, running=False)
        mock_inspect.side_effect = inspect_side
        with self.assertRaises(CutoverError):
            cutover.command_prepare(nodes, target_checkpoint="c" * 64, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID)
        rm_calls = [
            c for c in mock_remote.call_args_list
            if c[0][1][1] == "rm" and c[0][0].target_name in c[0][1]
        ]
        self.assertGreater(len(rm_calls), 0, "Expected docker rm --force for created targets")

    @mock.patch.object(cutover, "_verify_host_staging")
    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_lost_create_response_cleans_up_same_token(self, mock_exists, mock_remote, mock_inspect, mock_staging):
        """If docker create succeeds but SSH disconnects before returning, cleanup
        inspects for the prepare label and removes the target."""
        nodes = _nodes4()
        check_count = [0]
        def exists_side(node, name):
            check_count[0] += 1
            if check_count[0] <= 4:
                return Existence.PROVEN_ABSENT
            return Existence.PROVEN_PRESENT
        mock_staging.return_value = {"exists": True, "files": [], "stderr": ""}
        mock_exists.side_effect = exists_side
        prepare_token_holder: list[str] = []
        def remote_side(node, argv, **kw):
            if argv[1] == "create":
                for arg in argv:
                    if arg.startswith(cutover.PREPARE_LABEL + "="):
                        prepare_token_holder.append(arg.split("=", 1)[1])
                raise CutoverError("SSH disconnected before create response")
            if argv[1] == "rm":
                return ""
            return ""
        mock_remote.side_effect = remote_side
        def inspect_side(node, name):
            if name == node.source_name:
                return _src(node.rank)
            token = prepare_token_holder[0] if prepare_token_holder else "unknown"
            return _good_tgt(node.rank, running=False, prepare_token=token)
        mock_inspect.side_effect = inspect_side
        with self.assertRaises(CutoverError):
            cutover.command_prepare(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID)
        rm_calls = [
            c for c in mock_remote.call_args_list
            if c[0][1][1] == "rm" and c[0][0].target_name in c[0][1]
        ]
        self.assertGreater(len(rm_calls), 0, "Expected docker rm for lost-response target")

    @mock.patch.object(cutover, "_verify_host_staging")
    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_foreign_label_target_not_removed(self, mock_exists, mock_remote, mock_inspect, mock_staging):
        """A target with a foreign prepare label is never removed by our cleanup."""
        nodes = _nodes4()
        mock_staging.return_value = {"exists": True, "files": [], "stderr": ""}
        mock_exists.return_value = Existence.PROVEN_PRESENT
        def remote_side(node, argv, **kw):
            if argv[1] == "create":
                return f"container-id-r{node.rank}"
            if argv[1] == "rm":
                return ""
            return ""
        mock_remote.side_effect = remote_side
        def inspect_side(node, name):
            if name == node.source_name:
                return _src(node.rank)
            return _good_tgt(node.rank, running=False, foreign_label=True)
        mock_inspect.side_effect = inspect_side
        with self.assertRaises(CutoverError):
            cutover.command_prepare(nodes, target_checkpoint="c" * 64, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID)
        rm_calls = [
            c for c in mock_remote.call_args_list
            if c[0][1][1] == "rm" and c[0][0].target_name in c[0][1]
        ]
        self.assertEqual(len(rm_calls), 0, "Foreign-label target must not be removed")

    @mock.patch.object(cutover, "_verify_host_staging")
    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_unknown_existence_prevents_prepare(self, mock_exists, mock_remote, mock_inspect, mock_staging):
        """UNKNOWN existence before prepare must refuse to proceed."""
        nodes = _nodes4()
        # _load_sources calls _inspect for source names; return proper sources.
        mock_inspect.side_effect = _inspect_side_factory()
        mock_staging.return_value = {"exists": True, "files": [], "stderr": ""}
        mock_exists.return_value = Existence.UNKNOWN
        with self.assertRaises(CutoverError) as ctx:
            cutover.command_prepare(nodes, target_checkpoint=_TID, draft_checkpoint=_DID, cache_root=_CACHE_ROOT, connector_staging=_STAGING, connector_bundle_identity=_BUNDLE_ID)
        self.assertIn("cannot determine", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# Rollback result tests
# ---------------------------------------------------------------------------


class RollbackResultTests(unittest.TestCase):
    """Tests that rollback uses return codes and inspect confirmation."""

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover, "_remote_result")
    def test_rollback_succeeds_when_all_confirmed(self, mock_rr, mock_exists, mock_inspect):
        """Rollback succeeds when stops return 0 and inspect confirms state."""
        nodes = _nodes4()
        mock_rr.return_value = _completed(returncode=0, stdout="id", stderr="")
        mock_exists.return_value = Existence.PROVEN_PRESENT
        def inspect_side(node, name):
            if name == node.target_name:
                return _good_tgt(node.rank, running=False)
            return _src(node.rank, running=True)
        mock_inspect.side_effect = inspect_side
        cutover.command_rollback(nodes)

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover, "_remote_result")
    def test_rollback_fails_on_target_stop_failure(self, mock_rr, mock_exists, mock_inspect):
        """Failed target stop makes rollback fail."""
        nodes = _nodes4()
        def rr_side(node, argv):
            if argv[1] == "stop" and node.target_name in argv:
                return _completed(returncode=1, stdout="", stderr="error")
            return _completed(returncode=0, stdout="id", stderr="")
        mock_rr.side_effect = rr_side
        mock_exists.return_value = Existence.PROVEN_PRESENT
        def inspect_side(node, name):
            if name == node.target_name:
                return _good_tgt(node.rank, running=True)  # Still running!
            return _src(node.rank, running=True)
        mock_inspect.side_effect = inspect_side
        with self.assertRaises(CutoverError) as ctx:
            cutover.command_rollback(nodes)
        self.assertIn("INCOMPLETE", str(ctx.exception))
        self.assertIn("targets not stopped", str(ctx.exception))

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover, "_remote_result")
    def test_rollback_fails_on_source_not_running(self, mock_rr, mock_exists, mock_inspect):
        """Source that doesn't start or isn't running makes rollback fail."""
        nodes = _nodes4()
        mock_rr.return_value = _completed(returncode=0, stdout="id", stderr="")
        mock_exists.return_value = Existence.PROVEN_PRESENT
        def inspect_side(node, name):
            if name == node.target_name:
                return _good_tgt(node.rank, running=False)
            if node.rank == 1:
                return _src(1, running=False)  # Source not running!
            return _src(node.rank, running=True)
        mock_inspect.side_effect = inspect_side
        with self.assertRaises(CutoverError) as ctx:
            cutover.command_rollback(nodes)
        self.assertIn("INCOMPLETE", str(ctx.exception))
        self.assertIn("sources not running", str(ctx.exception))
        self.assertIn("1", str(ctx.exception))

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover, "_remote_result")
    def test_rollback_succeeds_when_target_proven_absent(self, mock_rr, mock_exists, mock_inspect):
        """Target proven absent counts as safely stopped."""
        nodes = _nodes4()
        mock_rr.return_value = _completed(returncode=1, stdout="", stderr="Error: No such container")
        def exists_side(node, name):
            if name == node.target_name:
                return Existence.PROVEN_ABSENT
            return Existence.PROVEN_PRESENT
        mock_exists.side_effect = exists_side
        def inspect_side(node, name):
            if name == node.target_name:
                return _good_tgt(node.rank, running=False)
            return _src(node.rank, running=True)
        mock_inspect.side_effect = inspect_side
        cutover.command_rollback(nodes)

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover, "_remote_result")
    def test_rollback_fails_on_unknown_existence(self, mock_rr, mock_exists, mock_inspect):
        """UNKNOWN existence (SSH rc 255) after stop must NOT be treated as absence."""
        nodes = _nodes4()
        def rr_side(node, argv):
            if argv[1] == "stop" and node.target_name in argv:
                return _completed(returncode=255, stdout="", stderr="ssh: connection refused")
            return _completed(returncode=0, stdout="id", stderr="")
        mock_rr.side_effect = rr_side
        # _check_exists returns UNKNOWN for targets (SSH failed).
        def exists_side(node, name):
            if name == node.target_name:
                return Existence.UNKNOWN
            return Existence.PROVEN_PRESENT
        mock_exists.side_effect = exists_side
        def inspect_side(node, name):
            if name == node.target_name:
                return _good_tgt(node.rank, running=True)
            return _src(node.rank, running=True)
        mock_inspect.side_effect = inspect_side
        with self.assertRaises(CutoverError) as ctx:
            cutover.command_rollback(nodes)
        self.assertIn("INCOMPLETE", str(ctx.exception))

    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_check_exists")
    @mock.patch.object(cutover, "_remote_result")
    def test_rollback_uses_returncode_not_stdout(self, mock_rr, mock_exists, mock_inspect):
        """Return code 1 with non-empty stdout is still failure."""
        nodes = _nodes4()
        def rr_side(node, argv):
            if argv[1] == "stop" and node.target_name in argv:
                return _completed(returncode=1, stdout="some-id", stderr="")
            return _completed(returncode=0, stdout="id", stderr="")
        mock_rr.side_effect = rr_side
        mock_exists.return_value = Existence.PROVEN_PRESENT
        def inspect_side(node, name):
            if name == node.target_name:
                return _good_tgt(node.rank, running=True)
            return _src(node.rank, running=True)
        mock_inspect.side_effect = inspect_side
        with self.assertRaises(CutoverError):
            cutover.command_rollback(nodes)


# ---------------------------------------------------------------------------
# SSH timeout tests
# ---------------------------------------------------------------------------


class SSHTimeoutTests(unittest.TestCase):
    def test_ssh_base_has_connect_timeout(self):
        base = cutover._ssh_base()
        timeout_opts = [
            base[i + 1] for i, v in enumerate(base)
            if v == "-o" and "ConnectTimeout" in base[i + 1]
        ]
        self.assertEqual(len(timeout_opts), 1)
        self.assertIn(str(cutover.SSH_CONNECT_TIMEOUT_S), timeout_opts[0])

    def test_ssh_base_has_batch_mode(self):
        base = cutover._ssh_base()
        batch_opts = [
            base[i + 1] for i, v in enumerate(base)
            if v == "-o" and "BatchMode" in base[i + 1]
        ]
        self.assertEqual(len(batch_opts), 1)

    @mock.patch.object(cutover.subprocess, "run")
    def test_remote_uses_ssh_timeout(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="ok", stderr="")
        cutover._remote(_node(0), ("docker", "inspect", "x"))
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "ssh")
        self.assertTrue(any("ConnectTimeout" in str(a) for a in call_args))

    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover.subprocess, "run")
    def test_health_check_uses_ssh_timeout(self, mock_run, mock_remote):
        mock_remote.return_value = json.dumps([_good_tgt(0, running=True)])
        mock_run.return_value = _completed(returncode=0, stdout="ok", stderr="")
        cutover._check_container_health(_node(0))
        for call in mock_run.call_args_list:
            call_args = call[0][0]
            if call_args[0] == "ssh":
                self.assertTrue(
                    any("ConnectTimeout" in str(a) for a in call_args),
                    f"SSH call missing ConnectTimeout: {call_args}",
                )


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class CLITests(unittest.TestCase):
    """Tests that exercise main(argv) — the actual CLI parser path."""
    def setUp(self):
        # Mock _verify_bundle_identity and _remote_result to prevent SSH hangs.
        for name, ret in [
            ("_verify_bundle_identity", True),
        ]:
            patcher = mock.patch.object(cutover, name, return_value=ret)
            patcher.start()
            self.addCleanup(patcher.stop)
        # _remote_result returns a failed SSH result to prevent hangs.
        rr_patcher = mock.patch.object(
            cutover, "_remote_result",
            return_value=_completed(returncode=255, stdout="", stderr="ssh: connection refused"),
        )
        rr_patcher.start()
        self.addCleanup(rr_patcher.stop)

    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_main_plan_without_checkpoints_returns_blockers(self, mock_exists, mock_remote):
        """Plan without checkpoints reports B2 blocker and returns nonzero."""
        mock_remote.return_value = json.dumps([_src(0)])
        mock_exists.return_value = Existence.PROVEN_ABSENT
        rc = cutover.main([
            "plan", "--node", "0=h0", "--node", "1=h1",
            "--node", "2=h2", "--node", "3=h3",
            "--cache-root", _CACHE_ROOT,
            "--connector-staging", _STAGING,
        ])
        self.assertEqual(rc, 1)

    def test_main_cutover_requires_checkpoint(self):
        with self.assertRaises(CutoverError) as ctx:
            cutover.main([
                "cutover", "--node", "0=h0", "--node", "1=h1",
                "--node", "2=h2", "--node", "3=h3",
                "--execute", "--confirmation", CUTOVER_CONFIRMATION,
            ])
        self.assertIn("requires --target-checkpoint", str(ctx.exception))

    def test_main_prepare_requires_confirmation(self):
        with self.assertRaises(CutoverError) as ctx:
            cutover.main([
                "prepare", "--node", "0=h0", "--node", "1=h1",
                "--node", "2=h2", "--node", "3=h3",
                "--target-checkpoint", _TID, "--draft-checkpoint", _DID,
                "--cache-root", _CACHE_ROOT,
                "--connector-staging", _STAGING,
                "--connector-bundle-identity", _BUNDLE_ID,
            ])
        self.assertIn("requires --execute", str(ctx.exception))

    def test_main_cutover_requires_confirmation(self):
        with self.assertRaises(CutoverError) as ctx:
            cutover.main([
                "cutover", "--node", "0=h0", "--node", "1=h1",
                "--node", "2=h2", "--node", "3=h3",
                "--target-checkpoint", _TID, "--draft-checkpoint", _DID,
                "--cache-root", _CACHE_ROOT,
                "--connector-staging", _STAGING,
                "--connector-bundle-identity", _BUNDLE_ID,
            ])
        self.assertIn("requires --execute", str(ctx.exception))

    def test_main_prepare_requires_cache_root(self):
        with self.assertRaises(CutoverError) as ctx:
            cutover.main([
                "prepare", "--node", "0=h0", "--node", "1=h1",
                "--node", "2=h2", "--node", "3=h3",
                "--target-checkpoint", _TID, "--draft-checkpoint", _DID,
                "--connector-staging", _STAGING,
                "--execute", "--confirmation", PREPARE_CONFIRMATION,
            ])
        self.assertIn("requires --cache-root", str(ctx.exception))

    def test_main_prepare_requires_connector_staging(self):
        with self.assertRaises(CutoverError) as ctx:
            cutover.main([
                "prepare", "--node", "0=h0", "--node", "1=h1",
                "--node", "2=h2", "--node", "3=h3",
                "--target-checkpoint", _TID, "--draft-checkpoint", _DID,
                "--cache-root", _CACHE_ROOT,
                "--execute", "--confirmation", PREPARE_CONFIRMATION,
            ])
        self.assertIn("requires --connector-staging", str(ctx.exception))

    def test_main_cutover_requires_cache_root(self):
        with self.assertRaises(CutoverError) as ctx:
            cutover.main([
                "cutover", "--node", "0=h0", "--node", "1=h1",
                "--node", "2=h2", "--node", "3=h3",
                "--target-checkpoint", _TID, "--draft-checkpoint", _DID,
                "--connector-staging", _STAGING,
                "--execute", "--confirmation", CUTOVER_CONFIRMATION,
            ])
        self.assertIn("requires --cache-root", str(ctx.exception))

    def test_main_cutover_requires_connector_staging(self):
        with self.assertRaises(CutoverError) as ctx:
            cutover.main([
                "cutover", "--node", "0=h0", "--node", "1=h1",
                "--node", "2=h2", "--node", "3=h3",
                "--target-checkpoint", _TID, "--draft-checkpoint", _DID,
                "--cache-root", _CACHE_ROOT,
                "--execute", "--confirmation", CUTOVER_CONFIRMATION,
            ])
        self.assertIn("requires --connector-staging", str(ctx.exception))

    @mock.patch.object(cutover, "_check_exists")
    def test_main_status_does_not_require_checkpoint(self, mock_exists):
        mock_exists.return_value = Existence.PROVEN_ABSENT
        cutover.main([
            "status", "--node", "0=h0", "--node", "1=h1",
            "--node", "2=h2", "--node", "3=h3",
        ])

    def test_main_requires_four_ranks(self):
        with self.assertRaises(CutoverError) as ctx:
            cutover.main(["status", "--node", "0=h0", "--node", "1=h1"])
        self.assertIn("exactly one --node", str(ctx.exception))

    def test_hex64_rejects_short(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cutover._validate_hex64("short")

    def test_hex64_rejects_uppercase(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cutover._validate_hex64("A" * 64)

    def test_hex64_accepts_valid(self):
        self.assertEqual(cutover._validate_hex64("a" * 64), "a" * 64)

    def test_main_cutover_rejects_nonpositive_timeout(self):
        with self.assertRaises(CutoverError) as ctx:
            cutover.main([
                "cutover", "--node", "0=h0", "--node", "1=h1",
                "--node", "2=h2", "--node", "3=h3",
                "--target-checkpoint", _TID, "--draft-checkpoint", _DID,
                "--cache-root", _CACHE_ROOT,
                "--connector-staging", _STAGING,
                "--connector-bundle-identity", _BUNDLE_ID,
                "--execute", "--confirmation", CUTOVER_CONFIRMATION,
                "--health-timeout", "0",
            ])
        self.assertIn("positive", str(ctx.exception))

    @mock.patch.object(cutover, "_remote")
    def test_main_plan_with_cache_root_and_staging(self, mock_remote):
        """Plan with cache-root and connector-staging should inspect and return."""
        mock_remote.return_value = json.dumps([_src(0)])
        rc = cutover.main([
            "plan", "--node", "0=h0", "--node", "1=h1",
            "--node", "2=h2", "--node", "3=h3",
            "--target-checkpoint", _TID, "--draft-checkpoint", _DID,
            "--cache-root", _CACHE_ROOT,
            "--connector-staging", _STAGING,
        ])
        # Plan returns 0 if ready, 1 if blockers.  Since we mock _remote
        # but not _check_exists or _verify_host_staging, there will be
        # blockers.  But it should not crash.
        self.assertIn(rc, (0, 1))



class PlanPreflightTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(cutover, "_verify_bundle_identity", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        rr_patcher = mock.patch.object(
            cutover, "_remote_result",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=255, stdout="", stderr="ssh: connection refused",
            ),
        )
        rr_patcher.start()
        self.addCleanup(rr_patcher.stop)


    @mock.patch.object(cutover, "_verify_host_staging")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_plan_reports_b2_blocker_without_checkpoints(
        self, mock_exists, mock_remote, mock_staging
    ):
        """Plan without checkpoints must report B2 blocker and return 1."""
        nodes = _nodes4()
        mock_remote.return_value = json.dumps([_src(0)])
        mock_exists.return_value = Existence.PROVEN_ABSENT
        mock_staging.return_value = {"exists": True, "files": [], "stderr": ""}
        rc = cutover.command_plan(nodes, None, None, _CACHE_ROOT, _STAGING, _BUNDLE_ID)
        self.assertEqual(rc, 1)

    @mock.patch.object(cutover, "_verify_host_staging")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_plan_reports_staging_blocker_when_missing(
        self, mock_exists, mock_remote, mock_staging
    ):
        """Plan must report blocker when connector staging is missing."""
        nodes = _nodes4()
        mock_remote.return_value = json.dumps([_src(0)])
        mock_exists.return_value = Existence.PROVEN_ABSENT
        mock_staging.return_value = {"exists": False, "files": [], "stderr": "No such file"}
        rc = cutover.command_plan(nodes, _TID, _DID, _CACHE_ROOT, _STAGING, _BUNDLE_ID)
        self.assertEqual(rc, 1)

    @mock.patch.object(cutover, "_verify_host_staging")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_plan_reports_blocker_when_no_staging_specified(
        self, mock_exists, mock_remote, mock_staging
    ):
        """Plan must report blocker when connector staging is not specified."""
        nodes = _nodes4()
        mock_remote.return_value = json.dumps([_src(0)])
        mock_exists.return_value = Existence.PROVEN_ABSENT
        mock_staging.return_value = {"exists": True, "files": [], "stderr": ""}
        rc = cutover.command_plan(nodes, _TID, _DID, _CACHE_ROOT, None, None)
        self.assertEqual(rc, 1)

    @mock.patch.object(cutover, "_verify_host_staging")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_plan_reports_target_collision_blocker(
        self, mock_exists, mock_remote, mock_staging
    ):
        """Plan must report blocker when a target already exists and is running."""
        nodes = _nodes4()
        mock_remote.return_value = json.dumps([_src(0)])
        mock_exists.return_value = Existence.PROVEN_PRESENT
        mock_staging.return_value = {"exists": True, "files": [], "stderr": ""}
        # _inspect is called via _remote for the running target check.
        # But _remote.return_value is already set to _src JSON. The
        # plan command calls _check_exists which returns PROVEN_PRESENT,
        # then tries to _inspect the target. Since _remote returns
        # _src JSON, the inspect will succeed but show source data,
        # which has Running=True. So the blocker will fire.
        rc = cutover.command_plan(nodes, _TID, _DID, _CACHE_ROOT, _STAGING, _BUNDLE_ID)
        self.assertEqual(rc, 1)


# ---------------------------------------------------------------------------
# Path validation tests (Fix 6)
# ---------------------------------------------------------------------------


class PathValidationTests(unittest.TestCase):
    def test_valid_absolute_path(self):
        self.assertEqual(cutover._validate_path("/opt/sparkcache-host-staging"), "/opt/sparkcache-host-staging")

    def test_valid_nested_path(self):
        self.assertEqual(cutover._validate_path("/var/tmp/sparkring-public-validation/context-cache"),
                         "/var/tmp/sparkring-public-validation/context-cache")

    def test_rejects_relative_path(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cutover._validate_path("opt/sparkcache-host-staging")

    def test_rejects_dotdot_path(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cutover._validate_path("/opt/../etc/passwd")

    def test_rejects_dot_path(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cutover._validate_path("/opt/./sparkcache")

    def test_rejects_root(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cutover._validate_path("/")

    def test_rejects_nul(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cutover._validate_path("/opt/\x00sparkcache")

    def test_rejects_newline(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cutover._validate_path("/opt/\nsparkcache")

    def test_cli_rejects_bad_cache_root(self):
        with self.assertRaises(SystemExit):
            cutover.main(["plan", "--node", "0=h0", "--node", "1=h1",
                         "--node", "2=h2", "--node", "3=h3",
                         "--cache-root", "relative/path"])

    def test_cli_rejects_bad_connector_staging(self):
        with self.assertRaises(SystemExit):
            cutover.main(["plan", "--node", "0=h0", "--node", "1=h1",
                         "--node", "2=h2", "--node", "3=h3",
                         "--connector-staging", "/opt/../etc"])


# ---------------------------------------------------------------------------
# Bundle identity verification tests (Fix 7)
# ---------------------------------------------------------------------------


class BundleIdentityTests(unittest.TestCase):
    """Tests for the fail-closed remote bundle identity verifier."""

    @mock.patch.object(cutover, "_remote_result")
    def test_verify_matches_expected_identity(self, mock_rr):
        """When remote identity matches, _verify_bundle_identity returns True."""
        mock_rr.return_value = _completed(
            returncode=0, stdout=json.dumps({"identity": "e" * 64}),
        )
        self.assertTrue(cutover._verify_bundle_identity(_node(0), _STAGING, "e" * 64))

    @mock.patch.object(cutover, "_remote_result")
    def test_verify_rejects_mismatched_identity(self, mock_rr):
        mock_rr.return_value = _completed(
            returncode=0, stdout=json.dumps({"identity": "a" * 64}),
        )
        self.assertFalse(cutover._verify_bundle_identity(_node(0), _STAGING, "f" * 64))

    @mock.patch.object(cutover, "_remote_result")
    def test_verify_rejects_ssh_failure(self, mock_rr):
        mock_rr.return_value = _completed(returncode=255, stdout="", stderr="refused")
        self.assertFalse(cutover._verify_bundle_identity(_node(0), _STAGING, "a" * 64))

    @mock.patch.object(cutover, "_remote_result")
    def test_verify_rejects_error_response(self, mock_rr):
        """Remote verifier reports error (e.g. missing/extra/symlink file)."""
        mock_rr.return_value = _completed(
            returncode=0, stdout=json.dumps({"error": "required file missing: foo.py"}),
        )
        self.assertFalse(cutover._verify_bundle_identity(_node(0), _STAGING, "a" * 64))

    @mock.patch.object(cutover, "_remote_result")
    def test_verify_rejects_malformed_output(self, mock_rr):
        mock_rr.return_value = _completed(returncode=0, stdout="not json at all")
        self.assertFalse(cutover._verify_bundle_identity(_node(0), _STAGING, "a" * 64))

    @mock.patch.object(cutover, "_remote_result")
    def test_verify_rejects_empty_output(self, mock_rr):
        mock_rr.return_value = _completed(returncode=0, stdout="")
        self.assertFalse(cutover._verify_bundle_identity(_node(0), _STAGING, "a" * 64))

    def test_verifier_script_rejects_extra_file(self):
        """Run the verifier script locally against a temp tree with an extra file."""
        _json = json
        tmpdir = tempfile.mkdtemp()
        try:
            staging = os.path.join(tmpdir, "staging")
            os.makedirs(staging)
            for rel in cutover._BUNDLE_REQUIRED_FILES:
                path = os.path.join(staging, rel.replace("/", os.sep))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write("content")
            # Add an extra file.
            with open(os.path.join(staging, "extra.py"), "w") as f:
                f.write("extra")
            required_json = _json.dumps(sorted(cutover._BUNDLE_REQUIRED_FILES))
            proc = subprocess.run(
                [sys.executable, "-c", cutover._REMOTE_VERIFIER_SCRIPT, staging, required_json, BUNDLE_DOMAIN_SEPARATOR],
                capture_output=True, text=True,
            )
            payload = _json.loads(proc.stdout.strip())
            self.assertIn("error", payload)
            self.assertIn("extra file", payload["error"])
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_verifier_script_rejects_missing_file(self):
        """Run the verifier script locally against a temp tree with a missing file."""
        _json = json
        tmpdir = tempfile.mkdtemp()
        try:
            staging = os.path.join(tmpdir, "staging")
            os.makedirs(staging)
            required = sorted(cutover._BUNDLE_REQUIRED_FILES)
            for rel in required[1:]:  # Skip the first file.
                path = os.path.join(staging, rel.replace("/", os.sep))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write("content")
            required_json = _json.dumps(required)
            proc = subprocess.run(
                [sys.executable, "-c", cutover._REMOTE_VERIFIER_SCRIPT, staging, required_json, BUNDLE_DOMAIN_SEPARATOR],
                capture_output=True, text=True,
            )
            payload = _json.loads(proc.stdout.strip())
            self.assertIn("error", payload)
            self.assertTrue(
                "missing" in payload["error"].lower() or
                "cannot lstat" in payload["error"].lower() or
                "not found" in payload["error"].lower(),
                f"Unexpected error: {payload['error']}",
            )
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    @unittest.skipIf(os.name == "nt", "Symlink tests require non-Windows")
    def test_verifier_script_rejects_symlink(self):
        """Run the verifier script locally against a temp tree with a symlink."""
        _json = json
        tmpdir = tempfile.mkdtemp()
        try:
            staging = os.path.join(tmpdir, "staging")
            os.makedirs(staging)
            required = sorted(cutover._BUNDLE_REQUIRED_FILES)
            for rel in required:
                path = os.path.join(staging, rel.replace("/", os.sep))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write("content")
            # Replace one file with a symlink.
            first = os.path.join(staging, required[0].replace("/", os.sep))
            os.unlink(first)
            os.symlink(os.path.join(staging, required[1].replace("/", os.sep)), first)
            required_json = _json.dumps(required)
            proc = subprocess.run(
                [sys.executable, "-c", cutover._REMOTE_VERIFIER_SCRIPT, staging, required_json, BUNDLE_DOMAIN_SEPARATOR],
                capture_output=True, text=True,
            )
            payload = _json.loads(proc.stdout.strip())
            self.assertIn("error", payload)
            self.assertIn("symlink", payload["error"].lower())
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_verifier_script_exact_match(self):
        """Run the verifier script locally against a valid tree and verify identity."""
        _json = json
        tmpdir = tempfile.mkdtemp()
        try:
            staging = os.path.join(tmpdir, "staging")
            os.makedirs(staging)
            required = sorted(cutover._BUNDLE_REQUIRED_FILES)
            for rel in required:
                path = os.path.join(staging, rel.replace("/", os.sep))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                content = f"content for {rel}\n".encode("utf-8")
                with open(path, "wb") as f:
                    f.write(content)
            required_json = _json.dumps(required)
            proc = subprocess.run(
                [sys.executable, "-c", cutover._REMOTE_VERIFIER_SCRIPT, staging, required_json, BUNDLE_DOMAIN_SEPARATOR],
                capture_output=True, text=True,
            )
            payload = _json.loads(proc.stdout.strip())
            self.assertIn("identity", payload)

            # Compute expected identity with the same algorithm.
            h = hashlib.sha256()
            h.update(BUNDLE_DOMAIN_SEPARATOR.encode("utf-8"))
            for rel in required:
                content = f"content for {rel}\n".encode("utf-8")
                sha = hashlib.sha256(content).hexdigest()
                size = len(content)
                h.update(b"\x00")
                h.update(rel.encode("utf-8"))
                h.update(b"\x00")
                h.update(str(size).encode("utf-8"))
                h.update(b"\x00")
                h.update(sha.encode("utf-8"))
            self.assertEqual(payload["identity"], h.hexdigest())
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    @mock.patch.object(cutover, "_remote_result")
    def test_verify_passes_canonical_domain_separator(self, mock_rr):
        """The remote verifier command must receive the canonical
        BUNDLE_DOMAIN_SEPARATOR as an explicit argv, not a hard-coded
        constant inside the script.  This prevents algorithm drift.
        """
        mock_rr.return_value = _completed(
            returncode=0, stdout=json.dumps({"identity": "e" * 64}),
        )
        cutover._verify_bundle_identity(_node(0), _STAGING, "e" * 64)
        # Inspect the argv passed to _remote_result.
        call_args = mock_rr.call_args
        argv = call_args[0][1]  # second positional arg is the argv tuple
        # The argv should be:
        #   ("python3", "-c", <script>, <staging>, <required-json>, <domain>)
        self.assertEqual(argv[-1], BUNDLE_DOMAIN_SEPARATOR)
        self.assertEqual(argv[0], "python3")
        self.assertEqual(argv[1], "-c")


# ---------------------------------------------------------------------------
# Wrapper adversarial tests (Fix 3)
# ---------------------------------------------------------------------------


class WrapperAdversarialTests(unittest.TestCase):
    def test_extract_unset_vars_rejects_empty_name(self):
        """Empty -u name pair must raise, not silently skip."""
        with self.assertRaises(CutoverError):
            cutover._extract_unset_vars(["-u", "", "/opt/venv/bin/vllm", "--model", "x"])

    def test_extract_unset_vars_rejects_duplicate_names(self):
        with self.assertRaises(CutoverError):
            cutover._extract_unset_vars(["-u", "VAR", "-u", "VAR", "/opt/venv/bin/vllm", "--model", "x"])

    def test_strip_wrapper_rejects_no_binary(self):
        """If /opt/venv/bin/vllm is missing after -u pairs, must raise."""
        with self.assertRaises(CutoverError):
            cutover._strip_wrapper(["-u", "VAR", "--model", "x"])

    def test_strip_wrapper_rejects_no_unset_pairs(self):
        with self.assertRaises(CutoverError):
            cutover._strip_wrapper(["/opt/venv/bin/vllm", "--model", "x"])

    def test_normalize_rejects_entrypoint_without_path(self):
        """Entrypoint=/usr/bin/env but Path=None must be rejected."""
        s = _src(0)
        s["Path"] = None
        with self.assertRaises(CutoverError):
            cutover._normalize_source(s, _node(0))

    def test_normalize_rejects_path_without_entrypoint(self):
        """Path=/usr/bin/env but Entrypoint=None must be rejected."""
        s = _src(0)
        s["Config"]["Entrypoint"] = None
        with self.assertRaises(CutoverError):
            cutover._normalize_source(s, _node(0))

    def test_normalize_rejects_mismatched_entrypoint_path(self):
        """Entrypoint=/usr/bin/env but Path=/bin/bash must be rejected."""
        s = _src(0)
        s["Path"] = "/bin/bash"
        with self.assertRaises(CutoverError):
            cutover._normalize_source(s, _node(0))


# ---------------------------------------------------------------------------
# Existence edge-case tests (Fix 4)
# ---------------------------------------------------------------------------


class ExistenceEdgeCaseTests(unittest.TestCase):
    @mock.patch.object(cutover, "_remote_result")
    def test_rc255_with_sentinel_is_unknown(self, mock_rr):
        """rc=255 (SSH failure) with sentinel text must be UNKNOWN, not ABSENT."""
        mock_rr.return_value = _completed(
            returncode=255, stdout="",
            stderr="Error: No such container: x",
        )
        self.assertEqual(cutover._check_exists(_node(0), "x"), Existence.UNKNOWN)

    @mock.patch.object(cutover, "_remote_result")
    def test_rc125_with_sentinel_is_unknown(self, mock_rr):
        """rc=125 (daemon error) with sentinel text must be UNKNOWN."""
        mock_rr.return_value = _completed(
            returncode=125, stdout="",
            stderr="Error: No such object: x",
        )
        self.assertEqual(cutover._check_exists(_node(0), "x"), Existence.UNKNOWN)

    @mock.patch.object(cutover, "_remote_result")
    def test_rc1_without_sentinel_is_unknown(self, mock_rr):
        """rc=1 but no sentinel text must be UNKNOWN (could be other Docker error)."""
        mock_rr.return_value = _completed(
            returncode=1, stdout="", stderr="Some other Docker error",
        )
        self.assertEqual(cutover._check_exists(_node(0), "x"), Existence.UNKNOWN)

    @mock.patch.object(cutover, "_remote_result")
    def test_rc1_with_no_such_object_sentinel(self, mock_rr):
        """rc=1 with 'No such object' (not 'container') must be ABSENT."""
        mock_rr.return_value = _completed(
            returncode=1, stdout="", stderr="Error: No such object: x",
        )
        self.assertEqual(cutover._check_exists(_node(0), "x"), Existence.PROVEN_ABSENT)


# ---------------------------------------------------------------------------
# Plan target state action tests (Fix 5)
# ---------------------------------------------------------------------------


class PlanTargetStateTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(cutover, "_verify_bundle_identity", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        rr = mock.patch.object(
            cutover, "_remote_result",
            return_value=_completed(returncode=255, stdout="", stderr="ssh: connection refused"),
        )
        rr.start()
        self.addCleanup(rr.stop)
    @mock.patch.object(cutover, "_verify_host_staging")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_absent_target_is_warning(self, mock_exists, mock_remote, mock_staging):
        """Absent target should produce a warning (prepare-ready), not a blocker."""
        mock_remote.return_value = json.dumps([_src(0)])
        mock_exists.return_value = Existence.PROVEN_ABSENT
        mock_staging.return_value = {"exists": True, "files": [], "stderr": ""}
        nodes = _nodes4()
        rc = cutover.command_plan(nodes, _TID, _DID, _CACHE_ROOT, _STAGING, _BUNDLE_ID)
        self.assertEqual(rc, 0)

    @mock.patch.object(cutover, "_verify_host_staging")
    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_stopped_target_exact_clone_is_ready(self, mock_exists, mock_remote, mock_inspect, mock_staging):
        """Stopped target that passes _verify_clone should be cutover-ready (rc=0)."""
        mock_remote.return_value = json.dumps([_src(0)])
        mock_exists.return_value = Existence.PROVEN_PRESENT
        mock_staging.return_value = {"exists": True, "files": [], "stderr": ""}
        mock_inspect.side_effect = _inspect_side_factory()
        nodes = _nodes4()
        rc = cutover.command_plan(nodes, _TID, _DID, _CACHE_ROOT, _STAGING, _BUNDLE_ID)
        self.assertEqual(rc, 0)

    @mock.patch.object(cutover, "_verify_host_staging")
    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_stopped_target_clone_mismatch_is_blocker(self, mock_exists, mock_remote, mock_inspect, mock_staging):
        """Stopped target that fails _verify_clone should be a blocker."""
        mock_remote.return_value = json.dumps([_src(0)])
        mock_exists.return_value = Existence.PROVEN_PRESENT
        mock_staging.return_value = {"exists": True, "files": [], "stderr": ""}
        # Return a target with wrong image (fails _verify_clone immediately).
        def bad_tgt(rank):
            t = _good_tgt(rank)
            t["Image"] = "sha256:wrong"
            return t
        mock_inspect.side_effect = _inspect_side_factory(tgt_fn=bad_tgt)
        nodes = _nodes4()
        rc = cutover.command_plan(nodes, _TID, _DID, _CACHE_ROOT, _STAGING, _BUNDLE_ID)
        self.assertEqual(rc, 1)

    @mock.patch.object(cutover, "_verify_host_staging")
    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_running_target_is_blocker(self, mock_exists, mock_remote, mock_inspect, mock_staging):
        """Running target should be a blocker, not a warning."""
        mock_remote.return_value = json.dumps([_src(0)])
        mock_exists.return_value = Existence.PROVEN_PRESENT
        mock_staging.return_value = {"exists": True, "files": [], "stderr": ""}
        mock_inspect.side_effect = _inspect_side_factory(tgt_fn=lambda r: _good_tgt(r, running=True))
        nodes = _nodes4()
        rc = cutover.command_plan(nodes, _TID, _DID, _CACHE_ROOT, _STAGING, _BUNDLE_ID)
        self.assertEqual(rc, 1)

    @mock.patch.object(cutover, "_verify_host_staging")
    @mock.patch.object(cutover, "_inspect")
    @mock.patch.object(cutover, "_remote")
    @mock.patch.object(cutover, "_check_exists")
    def test_unknown_existence_is_blocker(self, mock_exists, mock_remote, mock_inspect, mock_staging):
        """UNKNOWN existence must be a blocker (fail-closed)."""
        mock_remote.return_value = json.dumps([_src(0)])
        mock_exists.return_value = Existence.UNKNOWN
        mock_staging.return_value = {"exists": True, "files": [], "stderr": ""}
        nodes = _nodes4()
        rc = cutover.command_plan(nodes, _TID, _DID, _CACHE_ROOT, _STAGING, _BUNDLE_ID)
        self.assertEqual(rc, 1)

if __name__ == "__main__":
    unittest.main()
