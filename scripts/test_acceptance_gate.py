#!/usr/bin/env python3
"""GPU-free, cluster-free tests for the public-functional acceptance gate.

Nothing here touches a GPU, a fabric, or a serving stack: the gate's command
and HTTP surfaces are injected, so every stage is exercised against a scripted
fake cluster. Run with::

    python -m pytest scripts/test_acceptance_gate.py -q

Every site config below is synthetic: RFC1918 addresses that are not in the
reserved documentation ranges, hostnames under the reserved .test TLD, and
obviously fake hex. No real site's identifiers appear in this tree.
"""

from __future__ import annotations

import copy
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import acceptance_gate as gate  # noqa: E402
import collect_evidence as evidence  # noqa: E402

TOKEN_IDS = [10, 2087, 44, 913, 5, 77]

MODEL_REPO = "example-org/example-checkpoint"
MODEL_PATH = "/models/example-checkpoint"
MODEL_REVISION = "0123456789abcdef0123456789abcdef01234567"
CHECKPOINT_SHA = "0123456789abcdef" * 4
IMAGE_DIGEST = "sha256:" + "89abcdef01234567" * 4
RUNTIME_ID = "example-runtime-rc1"


# ---------------------------------------------------------------------------
# Fake cluster: implements both the executor and the HTTP client protocols
# ---------------------------------------------------------------------------


class FakeCluster:
    def __init__(
        self,
        *,
        verify_exit: int = 0,
        verify_ok: bool = True,
        runtime_ids: dict | None = None,
        qualifier_status: str = "qualified",
        qualifier_exit: int = 0,
        probe_exit: int = 0,
        start_exit: int = 0,
        stop_exit: int = 0,
        rollback_exit: int = 0,
        status_exit: int = 0,
        health_status: int = 200,
        models_status: int = 200,
        served_models: list[str] | None = None,
        completion_status: int = 200,
        tokenize_status: int = 200,
        token_ids: list[int] | None = None,
        completion_text: str = "A ring topology connects nodes in a cycle.",
        stream_ttft: float = 0.5,
        stream_total: float = 1.0,
        stream_tokens: int = 8,
        stream_error: str | None = None,
    ) -> None:
        self.verify_exit = verify_exit
        self.verify_ok = verify_ok
        self.runtime_ids = runtime_ids or {}
        self.qualifier_status = qualifier_status
        self.qualifier_exit = qualifier_exit
        self.probe_exit = probe_exit
        self.start_exit = start_exit
        self.stop_exit = stop_exit
        self.rollback_exit = rollback_exit
        self.status_exit = status_exit
        self.health_status = health_status
        self.models_status = models_status
        self.served_models = (
            served_models if served_models is not None else [MODEL_PATH]
        )
        self.completion_status = completion_status
        self.tokenize_status = tokenize_status
        self.token_ids = list(token_ids if token_ids is not None else TOKEN_IDS)
        self.completion_text = completion_text
        self.stream_ttft = stream_ttft
        self.stream_total = stream_total
        self.stream_tokens = stream_tokens
        self.stream_error = stream_error

        self.stopped = False
        self.commands: list[list[str]] = []
        self.requests: list[str] = []
        self._lock = threading.Lock()

    # -- executor -----------------------------------------------------------

    def run(self, argv, timeout=None):  # noqa: ANN001 - protocol shape
        argv = list(argv)
        with self._lock:
            self.commands.append(argv)
        joined = " ".join(argv)

        if "verify-runtime.py" in joined:
            runtime_id = self.runtime_ids.get(self._rank_of(argv), RUNTIME_ID)
            checks = {
                "ok": self.verify_ok,
                "exit_code": 0 if self.verify_ok else 2,
                "checks": [
                    {"name": "manifest_self_hash", "status": "pass", "detail": "ok"},
                    {"name": "native_libs", "status": "pass", "detail": "2 verified"},
                    {
                        "name": "image_digest",
                        "status": "skip",
                        "detail": "WARNING: digest pending",
                    },
                ],
            }
            attestation = {"manifest_self_hash": "a1b2" * 16, "runtime_id": runtime_id}
            stdout = (
                json.dumps(checks, indent=2)
                + "\n"
                + json.dumps(attestation, sort_keys=True, separators=(",", ":"))
            )
            return self._result(argv, self.verify_exit, stdout)

        if "qualify_direct_cable.py" in joined:
            return self._result(
                argv, self.qualifier_exit, json.dumps({"status": self.qualifier_status})
            )
        if "model-down-probe" in joined:
            return self._result(argv, self.probe_exit, "probe ok")
        if "--start" in argv:
            return self._result(argv, self.start_exit, "started 4 ranks")
        if "--stop" in argv:
            self.stopped = True
            return self._result(argv, self.stop_exit, "stopped 4 ranks")
        if "--verify-rollback" in argv:
            return self._result(argv, self.rollback_exit, "rollback ok")
        if "--rank-status" in argv:
            return self._result(argv, self.status_exit, "running")
        return self._result(argv, 0, "")

    @staticmethod
    def _rank_of(argv) -> int:  # noqa: ANN001
        for token in argv:
            if "@" in token:
                digits = "".join(c for c in token.split("@")[1] if c.isdigit())
                if digits:
                    return int(digits[0])
        return 0

    @staticmethod
    def _result(argv, exit_code, stdout, stderr=""):  # noqa: ANN001
        return gate.CommandResult(
            argv=list(argv),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=0.001,
        )

    # -- http ---------------------------------------------------------------

    def get_json(self, url, timeout=30.0):  # noqa: ANN001
        with self._lock:
            self.requests.append(f"GET {url}")
        if self.stopped:
            return 0, "connection refused"
        if url.endswith("/health"):
            return self.health_status, {"status": "ok"}
        if url.endswith("/v1/models"):
            return self.models_status, {
                "data": [{"id": name} for name in self.served_models]
            }
        return 404, {}

    def post_json(self, url, payload, timeout=300.0):  # noqa: ANN001
        with self._lock:
            self.requests.append(f"POST {url}")
        if url.endswith("/v1/completions"):
            return self.completion_status, {
                "choices": [{"text": self.completion_text}],
                "usage": {"completion_tokens": len(self.token_ids)},
            }
        if url.endswith("/tokenize"):
            return self.tokenize_status, {
                "count": len(self.token_ids),
                "tokens": self.token_ids,
            }
        return 404, {}

    def stream_completion(self, url, payload, timeout=1800.0):  # noqa: ANN001
        with self._lock:
            self.requests.append(f"STREAM {url}")
        return gate.StreamSample(
            ttft_seconds=None if self.stream_error else self.stream_ttft,
            total_seconds=self.stream_total,
            tokens=0 if self.stream_error else self.stream_tokens,
            text="x" * self.stream_tokens,
            error=self.stream_error,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


EDGES = (
    ("r0-r1", "10.10.1.0/24", (0, 1)),
    ("r1-r2", "10.10.2.0/24", (1, 2)),
    ("r2-r3", "10.10.3.0/24", (2, 3)),
    ("r3-r0", "10.10.4.0/24", (3, 0)),
)
ARTIFACT_SHA = {
    "transport_library": "1a2b3c4d5e6f7a8b" * 4,
    "nccl_library": "2b3c4d5e6f7a8b9c" * 4,
    "tp4_graph_probe": "3c4d5e6f7a8b9c0d" * 4,
}


def write_lock(
    tmp_path: Path,
    revision: str = MODEL_REVISION,
    name: str = "runtime-lock.json",
) -> Path:
    lock = {
        "schema": "sparkring-runtime-lock/v1",
        "runtime_id": RUNTIME_ID,
        "toolchain": {
            "cuda_version": "13.2",
            "torch_version": "2.12.0+cu132",
            "python_version": "3.12.3",
            "target_platform": "linux/arm64",
        },
        "vllm": {"commit": "d1e2f3a4" * 5},
        "sparkinfer": {"commit": "e1f2a3b4" * 5},
        "flashinfer": {"commit": "f1a2b3c4" * 5},
        "nccl": {"tag": "v2.30.7-1"},
        "model": {
            "repository": MODEL_REPO,
            "revision": revision,
            "config_sha256": CHECKPOINT_SHA,
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return path


def site_document(**overrides) -> dict:
    def ring_ports(rank_id: int) -> list[dict]:
        ports = []
        for index, (edge_id, subnet, endpoints) in enumerate(
            [edge for edge in EDGES if rank_id in edge[2]]
        ):
            octets = subnet.split(".")[:3]
            ports.append(
                {
                    "edge": edge_id,
                    "interface": f"eth{index + 1}",
                    "address": ".".join(octets + [str(10 + rank_id)]),
                    "rdma_device": f"mlx5_{index}",
                    "rdma_port": 1,
                    "roce_gid_index": 3,
                }
            )
        return ports

    def neighbours(rank_id: int) -> list[dict]:
        peers = []
        for _, _, endpoints in EDGES:
            if rank_id in endpoints:
                other = endpoints[0] if endpoints[1] == rank_id else endpoints[1]
                peers.append({"rank": other, "address": f"10.20.30.{10 + other}"})
        return peers

    document = {
        "schema_version": 1,
        "site": {
            "name": "example-four-node-ring",
            "description": "Synthetic test site for the acceptance gate.",
        },
        "topology": {
            "mtu": 9000,
            "link_speed_mbps": 200000,
            "edges": [
                {"id": edge_id, "subnet": subnet, "endpoints": list(endpoints)}
                for edge_id, subnet, endpoints in EDGES
            ],
        },
        "ranks": [
            {
                "id": rank_id,
                "ssh_target": f"ops@node{rank_id}.example-lab.test",
                "management": {
                    "interface": "eth0",
                    "address": f"10.20.30.{10 + rank_id}",
                },
                "ring_ports": ring_ports(rank_id),
                "transport_peers": neighbours(rank_id),
            }
            for rank_id in range(4)
        ],
        "runtime": {
            "container_image": "registry.example.test/example-org/runtime:1.0.0",
            "container_image_digest": IMAGE_DIGEST,
            "model_path": MODEL_PATH,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA,
        },
        "serving": {
            "tensor_parallel_size": 4,
            "decode_context_parallel_size": 4,
            "mtp_mode": "adaptive",
            "mtp_tokens": 4,
            "max_model_len": 458752,
            "kv_cache_bytes_per_rank": 4000000000,
            "max_num_seqs": 8,
            "master_rank": 0,
            "api_port": 8210,
            "master_port": 29501,
        },
        "paths": {
            "jit_cache_dir": "/var/lib/sparkring/jit-cache",
            "context_cache_dir": "/var/lib/sparkring/context-cache",
            "evidence_dir": "./evidence",
            "min_free_bytes": {"jit_cache": 34359738368, "context_cache": 549755813888},
        },
        "artifacts": [
            {
                "name": name,
                "path": f"/opt/sparkring/lib/{name}",
                "sha256": sha,
                "executable": False,
            }
            for name, sha in ARTIFACT_SHA.items()
        ],
        "preflight": {"ssh_timeout_seconds": 45, "required_free_ports": []},
    }
    for key, value in overrides.items():
        document[key] = value
    return document


def write_site(tmp_path: Path, document: dict | None = None) -> Path:
    path = tmp_path / "site.yaml"
    path.write_text(
        json.dumps(document if document is not None else site_document(), indent=2),
        encoding="utf-8",
    )
    return path


def gate_document(tmp_path: Path, **overrides) -> dict:
    document = {
        "schema": gate.GATE_CONFIG_SCHEMA,
        "ssh": {"command": ["ssh", "-o", "BatchMode=yes"]},
        "runtime": {
            "lock_path": str(write_lock(tmp_path)),
            "verify_script": "/opt/sparkring/verify-runtime.py",
            "manifest_path": "/opt/sparkring/runtime-manifest.json",
            "exec_prefix": ["docker", "exec", "sparkring"],
        },
        "fabric": {
            "model_down_probe": {
                "command": ["bash", "/opt/site/model-down-probe.sh"],
                "per_rank": True,
            }
        },
        "launch": {
            "start_command": ["site-launcher", "--start"],
            "stop_command": ["site-launcher", "--stop"],
            "rollback_verify_command": ["site-launcher", "--verify-rollback"],
            "rank_status_command": ["site-launcher", "--rank-status"],
            "ready_timeout_seconds": 60,
            "stop_timeout_seconds": 60,
        },
        "acceptance": {"seed": 20260729, "max_tokens": 64},
        "performance": {
            "cells": [
                {"concurrency": 1, "max_tokens": 32},
                {"concurrency": 8, "max_tokens": 32},
            ]
        },
    }
    return gate.deep_merge(document, overrides)


def write_gate_config(tmp_path: Path, **overrides) -> Path:
    path = tmp_path / "gate-config.json"
    path.write_text(
        json.dumps(gate_document(tmp_path, **overrides), indent=2), encoding="utf-8"
    )
    return path


def write_expected(tmp_path: Path, token_ids=None, **param_overrides) -> Path:
    ids = list(token_ids if token_ids is not None else TOKEN_IDS)
    params = {
        "prompt_id": gate.DEFAULT_PROMPT_ID,
        "prompt_sha256": gate.sha256_hex(gate.DEFAULT_PROMPT.encode("utf-8")),
        "model": MODEL_PATH,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "seed": 20260729,
        "max_tokens": 64,
    }
    params.update(param_overrides)
    document = {
        "schema": gate.EXPECTED_SCHEMA,
        "params": params,
        "token_ids": ids,
        "token_ids_sha256": gate.sha256_hex(gate.canonical_json(ids).encode("utf-8")),
    }
    path = tmp_path / "expected-generation.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def execute(
    tmp_path: Path,
    cluster: FakeCluster,
    *,
    site: dict | None = None,
    extra_argv: list[str] | None = None,
    **gate_overrides,
) -> tuple[int, dict]:
    site_path = write_site(tmp_path, site)
    gate_path = write_gate_config(tmp_path, **gate_overrides)
    bundle_parent = tmp_path / "evidence"
    argv = [
        "--site",
        str(site_path),
        "--gate-config",
        str(gate_path),
        "--repo-root",
        str(tmp_path),
        "--execute",
        "--confirm",
        gate.CONFIRM_TOKEN,
        "--bundle-dir",
        str(bundle_parent),
        "--run-id",
        "test-run",
    ] + (extra_argv or [])
    code = gate.main(argv, executor=cluster, http=cluster)
    result_path = bundle_parent / "test-run" / "result.json"
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.is_file()
        else {}
    )
    return code, result


def normalised_site(tmp_path: Path, document: dict | None = None) -> dict:
    site, _, _ = gate.load_site_config(write_site(tmp_path, document))
    return site


def stage_status(result: dict) -> dict:
    return {entry["id"]: entry["status"] for entry in result["stages"]}


# ---------------------------------------------------------------------------
# Stage sequencing
# ---------------------------------------------------------------------------


def test_stage_ids_and_order_are_stable():
    assert [stage.id for stage in gate.STAGES] == [
        "runtime_attestation",
        "fabric_transport_qualification",
        "rank_startup",
        "api_liveness",
        "deterministic_generation",
        "performance_matrix",
        "shutdown_rollback",
    ]
    assert [stage.order for stage in gate.STAGES] == [1, 2, 3, 4, 5, 6, 7]
    assert "performance_matrix" not in gate.FUNCTIONAL_STAGE_IDS


def test_happy_path_passes_and_result_matches_schema(tmp_path):
    code, result = execute(
        tmp_path,
        FakeCluster(),
        acceptance={"expected_generation_path": str(write_expected(tmp_path))},
        performance={
            "band": {
                "C1": {"ttft_seconds_p50": {"max": 100.0}},
                "C8": {"per_stream_tokens_per_second_p50": {"min": 0.0}},
            }
        },
    )

    assert code == gate.EXIT_OK
    assert result["schema"] == gate.RESULT_SCHEMA
    assert result["functional_verdict"] == gate.FUNCTIONAL_PASS
    assert result["performance_verdict"] == gate.PERFORMANCE_IN_BAND
    assert set(stage_status(result).values()) == {gate.STATUS_PASS}

    for key in (
        "schema",
        "gate_version",
        "run_id",
        "mode",
        "started_at",
        "finished_at",
        "environment_fingerprint",
        "stages",
        "functional_verdict",
        "performance_verdict",
        "exit_code",
        "notes",
    ):
        assert key in result, f"result.json is missing {key!r}"

    assert len(result["stages"]) == len(gate.STAGES)
    for entry in result["stages"]:
        for key in (
            "id",
            "order",
            "title",
            "status",
            "started_at",
            "finished_at",
            "duration_seconds",
            "message",
            "artifacts",
            "details",
        ):
            assert key in entry, f"stage {entry.get('id')} is missing {key!r}"
        assert isinstance(entry["duration_seconds"], float)
        assert isinstance(entry["artifacts"], list)

    fingerprint = result["environment_fingerprint"]
    assert fingerprint["model"]["revision"] == MODEL_REVISION
    assert fingerprint["serving"]["tensor_parallel_size"] == 4
    assert fingerprint["topology"]["edges"] == 4
    assert fingerprint["config"]["site_config_sha256"]
    assert fingerprint["config"]["gate_config_sha256"]


def test_environment_fingerprint_carries_no_site_identifiers(tmp_path):
    _, result = execute(
        tmp_path,
        FakeCluster(),
        acceptance={"expected_generation_path": str(write_expected(tmp_path))},
    )
    blob = json.dumps(result["environment_fingerprint"])
    for identifier in ("ops@", "example-lab.test", "10.20.30.", "10.10.1."):
        assert identifier not in blob


def test_aborts_at_first_failure_and_skips_the_rest(tmp_path):
    cluster = FakeCluster(qualifier_status="failed", qualifier_exit=1)
    code, result = execute(
        tmp_path,
        cluster,
        acceptance={"expected_generation_path": str(write_expected(tmp_path))},
    )

    statuses = stage_status(result)
    assert statuses["runtime_attestation"] == gate.STATUS_PASS
    assert statuses["fabric_transport_qualification"] == gate.STATUS_FAIL
    for stage_id in (
        "rank_startup",
        "api_liveness",
        "deterministic_generation",
        "performance_matrix",
        "shutdown_rollback",
    ):
        assert statuses[stage_id] == gate.STATUS_SKIPPED

    skipped = next(entry for entry in result["stages"] if entry["id"] == "rank_startup")
    assert skipped["message"] == "aborted-after: fabric_transport_qualification"

    assert result["functional_verdict"] == gate.FUNCTIONAL_FAIL
    assert result["performance_verdict"] == gate.PERFORMANCE_NOT_MEASURED
    assert code == gate.EXIT_FUNCTIONAL_FAIL
    assert not any("--start" in argv for argv in cluster.commands)


def test_first_stage_failure_never_starts_anything(tmp_path):
    cluster = FakeCluster(verify_exit=2, verify_ok=False)
    code, result = execute(tmp_path, cluster)

    assert code == gate.EXIT_FUNCTIONAL_FAIL
    assert stage_status(result)["runtime_attestation"] == gate.STATUS_FAIL
    assert not any("qualify_direct_cable.py" in " ".join(a) for a in cluster.commands)
    assert cluster.requests == []


def test_mismatched_runtime_ids_across_ranks_fail(tmp_path):
    code, result = execute(tmp_path, FakeCluster(runtime_ids={2: "other-runtime"}))
    assert code == gate.EXIT_FUNCTIONAL_FAIL
    stage = result["stages"][0]
    assert stage["status"] == gate.STATUS_FAIL
    assert "same frozen runtime" in stage["message"]


def test_model_down_probe_failure_blocks_startup(tmp_path):
    cluster = FakeCluster(probe_exit=1)
    code, result = execute(tmp_path, cluster)
    assert code == gate.EXIT_FUNCTIONAL_FAIL
    assert stage_status(result)["fabric_transport_qualification"] == gate.STATUS_FAIL
    assert not any("--start" in argv for argv in cluster.commands)


def test_failed_preflight_evidence_blocks_the_run(tmp_path):
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema": gate.PREFLIGHT_SCHEMA,
                "passed": False,
                "failed_check_ids": ["ring.mtu"],
                "failed_ranks": [2],
            }
        ),
        encoding="utf-8",
    )
    code, result = execute(
        tmp_path, FakeCluster(), extra_argv=["--preflight", str(preflight)]
    )
    assert code == gate.EXIT_FUNCTIONAL_FAIL
    assert "preflight" in result["stages"][0]["message"]


def test_passing_preflight_evidence_is_recorded(tmp_path):
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema": gate.PREFLIGHT_SCHEMA,
                "passed": True,
                "failed_check_ids": [],
                "failed_ranks": [],
            }
        ),
        encoding="utf-8",
    )
    _, result = execute(
        tmp_path,
        FakeCluster(),
        extra_argv=["--preflight", str(preflight)],
        acceptance={"expected_generation_path": str(write_expected(tmp_path))},
    )
    assert result["stages"][0]["details"]["preflight"]["passed"] is True


# ---------------------------------------------------------------------------
# Functional vs performance verdict separation
# ---------------------------------------------------------------------------


def test_performance_miss_is_not_a_functional_failure(tmp_path):
    code, result = execute(
        tmp_path,
        FakeCluster(),
        acceptance={"expected_generation_path": str(write_expected(tmp_path))},
        performance={"band": {"C1": {"ttft_seconds_p50": {"max": 0.001}}}},
    )

    statuses = stage_status(result)
    assert statuses["performance_matrix"] == gate.STATUS_PERFORMANCE_OUT_OF_BAND
    # The run continued: shutdown/rollback still executed and passed.
    assert statuses["shutdown_rollback"] == gate.STATUS_PASS
    assert result["functional_verdict"] == gate.FUNCTIONAL_PASS
    assert result["performance_verdict"] == gate.PERFORMANCE_OUT_OF_BAND
    assert code == gate.EXIT_PERFORMANCE_NOT_IN_BAND
    assert code != gate.EXIT_FUNCTIONAL_FAIL


def test_missing_band_records_a_candidate_instead_of_asserting(tmp_path):
    code, result = execute(
        tmp_path,
        FakeCluster(),
        acceptance={"expected_generation_path": str(write_expected(tmp_path))},
    )
    assert stage_status(result)["performance_matrix"] == gate.STATUS_BASELINE_RECORDED
    assert result["performance_verdict"] == gate.PERFORMANCE_BASELINE_RECORDED
    assert result["functional_verdict"] == gate.FUNCTIONAL_PASS
    assert code == gate.EXIT_PERFORMANCE_NOT_IN_BAND
    cell_dir = tmp_path / "evidence" / "test-run" / "stages" / "06-performance_matrix"
    assert (cell_dir / "candidate-band.json").is_file()
    assert (cell_dir / "cell-C1.json").is_file()
    assert (cell_dir / "cell-C8.json").is_file()


def test_verdicts_are_independent_top_level_fields(tmp_path):
    _, result = execute(
        tmp_path,
        FakeCluster(),
        acceptance={"expected_generation_path": str(write_expected(tmp_path))},
    )
    assert result["functional_verdict"] == gate.FUNCTIONAL_PASS
    assert result["performance_verdict"] != result["functional_verdict"]


def test_benchmark_request_errors_are_a_hard_failure(tmp_path):
    code, result = execute(
        tmp_path,
        FakeCluster(stream_error="connection reset"),
        acceptance={"expected_generation_path": str(write_expected(tmp_path))},
    )
    assert stage_status(result)["performance_matrix"] == gate.STATUS_FAIL
    assert code == gate.EXIT_FUNCTIONAL_FAIL


# ---------------------------------------------------------------------------
# BASELINE-RECORDED semantics
# ---------------------------------------------------------------------------


def test_missing_expected_output_file_records_a_baseline_and_does_not_pass(tmp_path):
    code, result = execute(
        tmp_path,
        FakeCluster(),
        acceptance={"expected_generation_path": str(tmp_path / "does-not-exist.json")},
    )

    statuses = stage_status(result)
    assert statuses["deterministic_generation"] == gate.STATUS_BASELINE_RECORDED
    assert statuses["deterministic_generation"] != gate.STATUS_PASS
    # BASELINE-RECORDED does not abort: later stages still produced evidence.
    assert statuses["shutdown_rollback"] == gate.STATUS_PASS
    assert result["functional_verdict"] == gate.FUNCTIONAL_BASELINE_RECORDED
    assert code == gate.EXIT_BASELINE_RECORDED
    assert code != gate.EXIT_OK

    baseline = tmp_path / "evidence" / "test-run" / "expected" / "generation-baseline.json"
    assert baseline.is_file()
    recorded = json.loads(baseline.read_text(encoding="utf-8"))
    assert recorded["token_ids"] == TOKEN_IDS
    assert recorded["token_ids_sha256"] == gate.sha256_hex(
        gate.canonical_json(TOKEN_IDS).encode("utf-8")
    )
    assert "NOT a pass" in next(
        entry["message"]
        for entry in result["stages"]
        if entry["id"] == "deterministic_generation"
    )


def test_unset_expected_output_path_also_records_a_baseline(tmp_path):
    code, result = execute(tmp_path, FakeCluster())
    assert stage_status(result)["deterministic_generation"] == (
        gate.STATUS_BASELINE_RECORDED
    )
    assert code == gate.EXIT_BASELINE_RECORDED


def test_token_id_mismatch_is_a_functional_failure(tmp_path):
    expected = write_expected(tmp_path, token_ids=[10, 2087, 44, 999, 5, 77])
    code, result = execute(
        tmp_path,
        FakeCluster(),
        acceptance={"expected_generation_path": str(expected)},
    )
    stage = next(
        entry for entry in result["stages"] if entry["id"] == "deterministic_generation"
    )
    assert stage["status"] == gate.STATUS_FAIL
    assert stage["details"]["first_divergence_index"] == 3
    assert code == gate.EXIT_FUNCTIONAL_FAIL


def test_parameter_drift_invalidates_the_comparison(tmp_path):
    expected = write_expected(tmp_path, seed=1)
    code, result = execute(
        tmp_path,
        FakeCluster(),
        acceptance={"expected_generation_path": str(expected)},
    )
    stage = next(
        entry for entry in result["stages"] if entry["id"] == "deterministic_generation"
    )
    assert stage["status"] == gate.STATUS_FAIL
    assert "drift" in stage["message"]
    assert code == gate.EXIT_FUNCTIONAL_FAIL


def test_missing_token_ids_never_falls_back_to_text_hashing(tmp_path):
    code, result = execute(tmp_path, FakeCluster(tokenize_status=404))
    stage = next(
        entry for entry in result["stages"] if entry["id"] == "deterministic_generation"
    )
    assert stage["status"] == gate.STATUS_FAIL
    assert "token ids" in stage["message"]
    assert code == gate.EXIT_FUNCTIONAL_FAIL


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_is_the_default_and_mutates_nothing(tmp_path, capsys):
    site_path = write_site(tmp_path)
    gate_path = write_gate_config(tmp_path)
    before = set(tmp_path.rglob("*"))

    code = gate.main(
        [
            "--site",
            str(site_path),
            "--gate-config",
            str(gate_path),
            "--repo-root",
            str(tmp_path),
            "--bundle-dir",
            str(tmp_path / "evidence"),
        ]
    )
    after = set(tmp_path.rglob("*"))

    assert code == gate.EXIT_OK
    assert before == after, "dry run created or removed files"
    assert not (tmp_path / "evidence").exists()

    plan = json.loads(capsys.readouterr().out)
    assert plan["schema"] == gate.PLAN_SCHEMA
    assert plan["mode"] == "dry-run"
    assert [entry["id"] for entry in plan["stages"]] == [
        stage.id for stage in gate.STAGES
    ]
    assert all(entry["actions"] for entry in plan["stages"])
    joined = json.dumps(plan)
    assert "qualify_direct_cable.py" in joined
    assert "verify-runtime.py" in joined


def test_dry_run_never_touches_the_executor_or_http(tmp_path):
    cluster = FakeCluster()
    code = gate.main(
        [
            "--site",
            str(write_site(tmp_path)),
            "--gate-config",
            str(write_gate_config(tmp_path)),
            "--repo-root",
            str(tmp_path),
        ],
        executor=cluster,
        http=cluster,
    )
    assert code == gate.EXIT_OK
    assert cluster.commands == []
    assert cluster.requests == []


def test_refusing_executor_raises_rather_than_running():
    refusing = gate.RefusingExecutor()
    with pytest.raises(RuntimeError):
        refusing.run(["echo", "hi"])
    with pytest.raises(RuntimeError):
        refusing.get_json("http://example.test/health")


def test_execute_requires_the_confirmation_token(tmp_path):
    code = gate.main(
        [
            "--site",
            str(write_site(tmp_path)),
            "--gate-config",
            str(write_gate_config(tmp_path)),
            "--repo-root",
            str(tmp_path),
            "--execute",
            "--bundle-dir",
            str(tmp_path / "evidence"),
        ],
        executor=FakeCluster(),
        http=FakeCluster(),
    )
    assert code == gate.EXIT_CONFIG_ERROR
    assert not (tmp_path / "evidence").exists()


def test_execute_refuses_a_site_config_full_of_placeholders(tmp_path):
    document = site_document()
    document["runtime"]["model_revision"] = "0" * 40
    document["runtime"]["checkpoint_sha256"] = "f" * 64
    lock = json.loads(write_lock(tmp_path, revision="0" * 40).read_text("utf-8"))
    lock["model"]["config_sha256"] = "f" * 64
    (tmp_path / "runtime-lock.json").write_text(json.dumps(lock), encoding="utf-8")

    code = gate.main(
        [
            "--site",
            str(write_site(tmp_path, document)),
            "--gate-config",
            str(write_gate_config(tmp_path)),
            "--repo-root",
            str(tmp_path),
            "--execute",
            "--confirm",
            gate.CONFIRM_TOKEN,
            "--bundle-dir",
            str(tmp_path / "evidence"),
        ],
        executor=FakeCluster(),
        http=FakeCluster(),
    )
    assert code == gate.EXIT_CONFIG_ERROR
    assert not (tmp_path / "evidence").exists()


# ---------------------------------------------------------------------------
# Matrix / config validation
# ---------------------------------------------------------------------------


def problems_for(tmp_path: Path, mutate=None, lock_revision=MODEL_REVISION) -> list[str]:
    site = normalised_site(tmp_path)
    if mutate is not None:
        mutate(site)
    gate_config, _ = gate.load_gate_config(write_gate_config(tmp_path))
    lock = json.loads(
        write_lock(tmp_path, revision=lock_revision).read_text(encoding="utf-8")
    )
    return gate.validate_configuration(site, gate_config, lock)


def test_valid_configuration_has_no_problems(tmp_path):
    assert problems_for(tmp_path) == []


def test_mutable_model_revision_is_refused(tmp_path):
    problems = problems_for(tmp_path, lock_revision="pending")
    assert any("immutable" in problem for problem in problems)

    pending_lock = write_lock(tmp_path, "pending", name="runtime-lock-pending.json")
    code = gate.main(
        [
            "--site",
            str(write_site(tmp_path)),
            "--gate-config",
            str(write_gate_config(tmp_path, runtime={"lock_path": str(pending_lock)})),
            "--repo-root",
            str(tmp_path),
        ]
    )
    assert code == gate.EXIT_CONFIG_ERROR


@pytest.mark.parametrize(
    "revision", ["main", "latest", "v1.0", "", "pending", "B" * 40 + "x"]
)
def test_only_immutable_revisions_are_accepted(revision):
    ok, message, _ = gate.check_model_pin(
        {
            "model": {
                "repository": MODEL_REPO,
                "revision": revision,
                "config_sha256": CHECKPOINT_SHA,
            }
        }
    )
    assert not ok
    assert "immutable" in message or "config_sha256" in message


def test_site_and_lock_must_pin_the_same_checkpoint(tmp_path):
    def mutate(site):
        site["runtime"]["model_repo"] = "someone-else/other-checkpoint"

    assert any("does not match" in p for p in problems_for(tmp_path, mutate))


def test_wrong_rank_count_is_refused(tmp_path):
    def mutate(site):
        site["ranks"] = site["ranks"][:3]

    assert any("exactly 4 ranks" in p for p in problems_for(tmp_path, mutate))


def test_non_ring_topology_is_refused(tmp_path):
    def mutate(site):
        site["topology"]["edges"][3]["endpoints"] = [0, 1]

    assert any("degree" in p for p in problems_for(tmp_path, mutate))


def test_shared_subnet_across_cables_is_refused(tmp_path):
    def mutate(site):
        site["topology"]["edges"][1]["subnet"] = site["topology"]["edges"][0]["subnet"]

    assert any("DISTINCT" in p for p in problems_for(tmp_path, mutate))


def test_wrong_mtu_is_refused(tmp_path):
    def mutate(site):
        site["topology"]["mtu"] = 1500

    assert any("topology.mtu" in p for p in problems_for(tmp_path, mutate))


@pytest.mark.parametrize(
    "key,value",
    [
        ("tensor_parallel_size", 2),
        ("decode_context_parallel_size", 1),
        ("mtp_mode", "off"),
        ("mtp_tokens", 2),
        ("max_model_len", 131072),
        ("kv_cache_bytes_per_rank", 3_000_000_000),
        ("max_num_seqs", 16),
    ],
)
def test_serving_shape_must_match_the_matrix(tmp_path, key, value):
    def mutate(site):
        site["serving"][key] = value

    assert any(f"serving.{key}" in problem for problem in problems_for(tmp_path, mutate))


def test_gate_config_requires_launcher_and_probe_commands(tmp_path):
    site = normalised_site(tmp_path)
    lock = json.loads(write_lock(tmp_path).read_text(encoding="utf-8"))
    bare = copy.deepcopy(gate.DEFAULT_GATE_CONFIG)
    problems = gate.validate_configuration(site, bare, lock)
    assert any("launch.start_command" in problem for problem in problems)
    assert any("launch.stop_command" in problem for problem in problems)
    assert any("model_down_probe" in problem for problem in problems)


def test_gate_config_requires_the_c1_and_c8_cells(tmp_path):
    site = normalised_site(tmp_path)
    lock = json.loads(write_lock(tmp_path).read_text(encoding="utf-8"))
    config, _ = gate.load_gate_config(
        write_gate_config(tmp_path, performance={"cells": [{"concurrency": 1}]})
    )
    problems = gate.validate_configuration(site, config, lock)
    assert any("C1 and C8" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Delegated-tool parsers and band evaluation
# ---------------------------------------------------------------------------


def test_verify_runtime_stdout_is_split_into_checks_and_attestation():
    stdout = (
        json.dumps({"ok": True, "checks": []}, indent=2)
        + "\n"
        + '{"manifest_self_hash":"aa","runtime_id":"rid"}'
    )
    checks, attestation = gate.parse_verify_runtime_stdout(stdout)
    assert checks == {"ok": True, "checks": []}
    assert attestation["runtime_id"] == "rid"

    checks, attestation = gate.parse_verify_runtime_stdout(
        json.dumps({"ok": False, "checks": []}, indent=2)
    )
    assert checks == {"ok": False, "checks": []}
    assert attestation is None


def test_band_evaluation_supports_bounds_and_tolerance():
    observed = {"C1": {"tok_s": 10.0, "ttft": 1.0}}
    _, violations = gate.evaluate_band(observed, {"C1": {"tok_s": {"min": 5.0}}})
    assert violations == []

    _, violations = gate.evaluate_band(
        observed, {"C1": {"tok_s": {"expected": 100.0, "tolerance_pct": 5.0}}}
    )
    assert violations and "outside" in violations[0]

    _, violations = gate.evaluate_band(observed, {"C1": {"missing": {"min": 1.0}}})
    assert violations and "not measured" in violations[0]


# ---------------------------------------------------------------------------
# Evidence redaction
# ---------------------------------------------------------------------------


# Synthetic identifiers only. The RFC1918 /16 most home and lab networks use is
# assembled from octets so that no such literal exists anywhere in this tree,
# while still proving the whole IPv4 class is stripped.
RFC1918_SYNTHETIC = ".".join(("192", "168", "77", "21"))
BLOCKLISTED = [
    "ops@node0.example-lab.test",
    "node0.example-lab.test",
    "node3.example-lab.test",
    "10.20.30.10",
    "10.10.1.10",
    RFC1918_SYNTHETIC,
    "/srv/models/example-private-checkpoint",
    "/home/ops/glm-jit-cache",
    "hunter2-not-a-real-token",
]


def redaction_site_document() -> dict:
    document = site_document()
    document["management"] = {
        "head_ip": RFC1918_SYNTHETIC,
        "hostname": "node0.example-lab.test",
        "ssh_user": "ops",
        "api_token": "hunter2-not-a-real-token",
    }
    document["runtime"]["model_path"] = "/srv/models/example-private-checkpoint"
    document["paths"]["jit_cache_dir"] = "/home/ops/glm-jit-cache"
    return document


def test_evidence_bundle_strips_every_blocklisted_identifier_class(tmp_path):
    site_path = tmp_path / "site.json"
    site_path.write_text(json.dumps(redaction_site_document(), indent=2), encoding="utf-8")

    log = tmp_path / "serve-rank0.log"
    log.write_text(
        "\n".join(
            [
                f"INFO connecting to {RFC1918_SYNTHETIC}:8210",
                "INFO ssh ops@node0.example-lab.test docker ps",
                "INFO peer 10.10.1.10 -> 10.10.1.11 established",
                "INFO management address 10.20.30.10 up",
                "INFO loading /srv/models/example-private-checkpoint/config.json",
                "INFO cache dir /home/ops/glm-jit-cache",
                "INFO rank3 host node3.example-lab.test ready",
                "INFO bearer hunter2-not-a-real-token",
                "INFO nccl 2.30.7+cuda13.0 NET/IB Connected all rings",
            ]
        ),
        encoding="utf-8",
    )

    result_json = tmp_path / "result.json"
    result_json.write_text(
        json.dumps(
            {
                "schema": gate.RESULT_SCHEMA,
                "functional_verdict": "PASS",
                "performance_verdict": "BASELINE-RECORDED",
                "stages": [
                    {
                        "id": "api_liveness",
                        "message": "rank 0 at http://10.20.30.10:8210 answered",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "bundle"
    code = evidence.main(
        [
            "--site",
            str(site_path),
            "--acceptance-result",
            str(result_json),
            "--log",
            str(log),
            "--out",
            str(out),
            "--repo-root",
            str(tmp_path),
        ]
    )
    assert code == 0

    emitted = [path for path in out.rglob("*") if path.is_file()]
    assert emitted
    blob = "\n".join(path.read_text(encoding="utf-8") for path in emitted)
    for identifier in BLOCKLISTED:
        assert identifier not in blob, f"{identifier!r} survived redaction"
    assert "example-lab.test" not in blob
    assert "<redacted-" in blob
    # Useful, non-identifying content survives.
    assert "Connected all rings" in blob
    assert "2.30.7" in blob


def test_secret_shaped_keys_are_dropped_and_site_id_pseudonymised(tmp_path):
    site_path = tmp_path / "site.json"
    site_path.write_text(json.dumps(redaction_site_document(), indent=2), encoding="utf-8")
    out = tmp_path / "bundle"
    assert (
        evidence.main(
            ["--site", str(site_path), "--out", str(out), "--repo-root", str(tmp_path)]
        )
        == 0
    )
    redacted = json.loads((out / "site-config.redacted.json").read_text(encoding="utf-8"))
    assert redacted["management"]["api_token"] == evidence.PLACEHOLDER_SECRET
    # Non-identifying matrix facts must survive so the report is still useful.
    assert redacted["topology"]["mtu"] == 9000
    assert redacted["serving"]["tensor_parallel_size"] == 4
    assert redacted["serving"]["max_model_len"] == 458752


def test_self_check_refuses_to_write_an_unredacted_bundle():
    redactor = evidence.Redactor(
        literals={"node0.example-lab.test": "<redacted-host-1>"}
    )
    leaked = evidence.BundleFile("leak.txt", "connect to node0.example-lab.test", "leak")
    with pytest.raises(evidence.RedactionError):
        evidence.verify_files([leaked], redactor)

    clean = evidence.BundleFile("clean.txt", "connect to <redacted-host-1>", "clean")
    evidence.verify_files([clean], redactor)


def test_identifier_classes_are_caught_without_a_site_literal():
    redactor = evidence.Redactor()
    text = redactor.text(
        f"admin@unknown-host.test reached {RFC1918_SYNTHETIC} via /home/admin/keys"
    )
    assert RFC1918_SYNTHETIC not in text
    assert "admin@unknown-host.test" not in text
    assert "/home/admin/keys" not in text
    assert redactor.residues(text) == []


def test_versions_and_public_references_are_preserved():
    redactor = evidence.Redactor()
    kept = (
        "https://github.com/vllm-project/vllm @ fcc6141 ; "
        "nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 ; "
        "torch 2.12.0+cu132 ; runtime/runtime-lock.json ; verify-runtime.py"
    )
    assert redactor.text(kept) == kept


def test_digests_are_kept_by_default_and_redacted_on_request():
    digest = "sha256:" + "a1b2c3d4" * 8
    assert digest in evidence.Redactor().text(f"image {digest}")
    assert digest not in evidence.Redactor(redact_digests=True).text(f"image {digest}")


def test_hostname_heuristic_distinguishes_files_from_hosts():
    assert evidence.looks_like_hostname("node0.example-lab.test")
    assert not evidence.looks_like_hostname("runtime-lock.json")
    assert not evidence.looks_like_hostname("verify-runtime.py")
    assert not evidence.looks_like_hostname("2.30.7")
    assert not evidence.looks_like_hostname("0.11.2.dev279")
    assert not evidence.looks_like_hostname("github.com")
    assert not evidence.looks_like_hostname("download.pytorch.org")


def test_evidence_bundle_carries_a_manifest_and_a_review_warning(tmp_path):
    site_path = tmp_path / "site.json"
    site_path.write_text(json.dumps(site_document(), indent=2), encoding="utf-8")
    out = tmp_path / "bundle"
    assert (
        evidence.main(
            ["--site", str(site_path), "--out", str(out), "--repo-root", str(tmp_path)]
        )
        == 0
    )
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == evidence.BUNDLE_SCHEMA
    assert manifest["redaction"]["self_check"] == "passed"
    assert manifest["files"]
    assert all(entry["sha256"] for entry in manifest["files"])
    assert "Review" in manifest["warning"]
    assert (out / "README.txt").is_file()
