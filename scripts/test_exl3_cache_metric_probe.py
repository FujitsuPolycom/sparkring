from __future__ import annotations

import hashlib
import json
from pathlib import Path

import exl3_cache_metric_probe as probe
import pytest
from exl3_attribution_cache_contract import build_live_arm_receipt, cache_salt_for_arm


ATTRIBUTION_ARM = "d-mtp2-apc1-lmcache1"
CACHE_SALT = cache_salt_for_arm(ATTRIBUTION_ARM)


@pytest.fixture(autouse=True)
def offline_live_arm_revalidation(monkeypatch):
    monkeypatch.setattr(probe, "load_site", lambda _path: object())
    monkeypatch.setattr(
        probe.attribution.exl3, "load_profile", lambda _path: object()
    )
    monkeypatch.setattr(
        probe.attribution,
        "live_arm_revalidation_actions",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        probe.attribution,
        "revalidate_live_arm",
        lambda *_args, **_kwargs: {
            "status": "live-arm-re-attested",
            "rank_count": 4,
            "runtime_instances": [],
        },
    )


def activation_args(directory: Path) -> list[str]:
    profile_path = directory / "launch.json"
    receipt_path = directory / "live-arm-receipt.json"
    profile = {
        "profile_id": "glm52-exl3-tr3-3.25bpw-lmcache-cs512",
        "image_id": "sha256:" + "a" * 64,
        "model_repository": "willfalco/GLM-5.2-EXL3-TR3-3.25bpw",
        "model_revision": "b" * 40,
        "container_name": "glm52-sparkring-exl3-lmcache-cs512",
        "environment": {
            "VLLM_SPARK_KV_CACHE_MEMORY_BYTES": "4500000000",
            "VLLM_SPARK_MAX_MODEL_LEN": "524288",
        },
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    receipt = build_live_arm_receipt(
        arm_id=ATTRIBUTION_ARM,
        canonical_profile_id=profile["profile_id"],
        canonical_profile_file_sha256=hashlib.sha256(
            profile_path.read_bytes()
        ).hexdigest(),
        image_id=profile["image_id"],
        model_repository=profile["model_repository"],
        model_revision=profile["model_revision"],
        canonical_container_name=profile["container_name"],
        explicit_environment_sha256=[
            hashlib.sha256(f"env-{rank}".encode()).hexdigest()
            for rank in range(4)
        ],
        config_cmd_sha256=[
            hashlib.sha256(f"cmd-{rank}".encode()).hexdigest()
            for rank in range(4)
        ],
        observed_runtime_instances=[
            {
                "container_id": f"{rank + 1:064x}",
                "started_at": f"2026-08-10T03:2{rank}:00.123456789Z",
            }
            for rank in range(4)
        ],
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return [
        "--site",
        str(directory / "site.yaml"),
        "--activation-receipt",
        str(receipt_path),
        "--profile",
        str(profile_path),
    ]


def metrics(*, queries: int, hits: int, external_queries: int = 0, external_hits: int = 0) -> str:
    return f'''vllm:prefix_cache_queries_total{{engine="0",model_name="m"}} {queries}
vllm:prefix_cache_hits_total{{engine="0",model_name="m"}} {hits}
vllm:external_prefix_cache_queries_total{{engine="0",model_name="m"}} {external_queries}
vllm:external_prefix_cache_hits_total{{engine="0",model_name="m"}} {external_hits}
vllm:prompt_tokens_by_source_total{{engine="0",model_name="m",source="local_compute"}} {queries - hits}
vllm:prompt_tokens_by_source_total{{engine="0",model_name="m",source="local_cache_hit"}} {hits}
vllm:prompt_tokens_by_source_total{{engine="0",model_name="m",source="external_kv_transfer"}} {external_hits}
vllm:cache_config_info{{block_size="64",enable_prefix_caching="True",kv_cache_size_tokens="562688",num_gpu_blocks="2198"}} 1
'''


class FakeClient:
    def __init__(self):
        self.metric_bodies = [
            metrics(queries=0, hits=0),
            metrics(queries=1024, hits=0),
            metrics(queries=2048, hits=1024),
        ]
        self.completions = 0
        self.requests = []

    def get_text(self, url, timeout):
        self.requests.append(("GET", url))
        assert url.endswith("/metrics")
        return 200, self.metric_bodies.pop(0)

    def post_json(self, url, payload, timeout):
        self.requests.append(("POST", url))
        if url.endswith("/tokenize"):
            return 200, {"tokens": list(range(1100))}
        assert payload["cache_salt"] == CACHE_SALT
        self.completions += 1
        return 200, {
            "id": f"cmpl-{self.completions}",
            "choices": [{"text": "ok"}],
            "usage": {
                "prompt_tokens": 1100,
                "prompt_tokens_details": {"cached_tokens": 0 if self.completions == 1 else 1024},
            },
        }


def test_metric_snapshot_and_dcp_geometry():
    client = FakeClient()
    result = probe.run_probe(
        client,
        base_url="http://example.invalid:8000",
        model="m",
        prompt="x",
        repetitions=2,
        max_tokens=1,
        seed=1,
        timeout=1,
        dcp_degree=4,
        expected_logical_block_tokens=256,
        lmcache_chunk_tokens=512,
        cache_salt=CACHE_SALT,
    )
    assert result["geometry"] == {
        "physical_block_tokens_per_dcp_rank": 64,
        "dcp_degree": 4,
        "dcp_global_apc_alignment_tokens": 256,
        "lmcache_chunk_tokens": 512,
        "physical_blocks_per_dcp_global_apc_unit_per_rank": 1,
        "dcp_global_apc_units_per_lmcache_chunk": 2,
        "physical_blocks_per_lmcache_chunk_per_rank": 2,
    }
    assert result["prompt_sha256"] == probe.sha256_hex(b"x")
    assert result["observations"][0]["metric_interval_delta"]["counters"]["vllm:prefix_cache_hits_total"] == 0
    assert result["observations"][1]["metric_interval_delta"]["counters"]["vllm:prefix_cache_hits_total"] == 1024


def test_live_geometry_mismatch_fails_closed():
    client = FakeClient()
    try:
        probe.run_probe(
            client,
            base_url="http://example.invalid:8000",
            model="m",
            prompt="x",
            repetitions=2,
            max_tokens=1,
            seed=1,
            timeout=1,
            dcp_degree=2,
            expected_logical_block_tokens=256,
            lmcache_chunk_tokens=512,
            cache_salt=CACHE_SALT,
        )
    except probe.ProbeError as error:
        assert "live logical block is 64 * 2 = 128" in str(error)
    else:
        raise AssertionError("geometry mismatch was accepted")


class UsageClient(FakeClient):
    def __init__(self, usage):
        super().__init__()
        self.usage = usage

    def post_json(self, url, payload, timeout):
        if url.endswith("/tokenize"):
            return 200, {"tokens": list(range(1100))}
        self.completions += 1
        return 200, {
            "id": f"cmpl-{self.completions}",
            "choices": [{"text": "ok"}],
            "usage": self.usage,
        }


@pytest.mark.parametrize(
    ("usage", "message"),
    [
        ({"prompt_tokens": "https://private.invalid"}, "must be a nonnegative integer"),
        ({"prompt_tokens": True}, "must be a nonnegative integer"),
        ({"prompt_tokens": 1099}, "does not match /tokenize"),
        (
            {"prompt_tokens": 1100, "prompt_tokens_details": {"cached_tokens": "https://private.invalid"}},
            "cached_tokens must be a nonnegative integer",
        ),
        (
            {"prompt_tokens": 1100, "prompt_tokens_details": {"cached_tokens": True}},
            "cached_tokens must be a nonnegative integer",
        ),
        (
            {"prompt_tokens": 1100, "prompt_tokens_details": {"cached_tokens": 1101}},
            "cached tokens do not bind",
        ),
        (
            {"prompt_tokens_details": {"cached_tokens": 1}},
            "cached tokens do not bind",
        ),
    ],
)
def test_raw_probe_rejects_unbounded_or_inconsistent_usage(usage, message):
    with pytest.raises(probe.ProbeError, match=message):
        probe.run_probe(
            UsageClient(usage),
            base_url="http://example.invalid:8000",
            model="m",
            prompt="x",
            repetitions=2,
            max_tokens=1,
            seed=1,
            timeout=1,
            dcp_degree=4,
            expected_logical_block_tokens=256,
            lmcache_chunk_tokens=512,
            cache_salt=CACHE_SALT,
        )


def test_cli_refuses_nonfinite_report_before_creating_output(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(probe, "run_probe", lambda *args, **kwargs: {"bad": float("nan")})
    output = tmp_path / "report.json"
    status = probe.main(
        [
            "--base-url", "http://example.invalid:8000",
            "--model", "m",
            "--attribution-arm", ATTRIBUTION_ARM,
            *activation_args(tmp_path),
            "--run-label", "bad-nan",
            "--output", str(output),
            "--prompt-fragment", "a",
            "--prompt-suffix", "z",
            "--execute",
        ],
        client=FakeClient(),
    )
    assert status == 2
    assert "Out of range float values" in capsys.readouterr().err
    assert not output.exists()


def test_cli_plan_does_not_contact_remote(tmp_path, monkeypatch, capsys):
    output = tmp_path / "report.json"
    actions = [
        probe.attribution.exl3.RemoteAction(
            rank,
            f"rank{rank}.test",
            ("sh", "-lc", f"test-rank-{rank}"),
        )
        for rank in range(4)
    ]
    monkeypatch.setattr(
        probe.attribution,
        "live_arm_revalidation_actions",
        lambda *_args, **_kwargs: actions,
    )
    assert probe.main([
        "--base-url", "http://example.invalid:8000",
        "--model", "m",
        "--attribution-arm", ATTRIBUTION_ARM,
        *activation_args(tmp_path),
        "--run-label", "arm-c-probe",
        "--output", str(output),
        "--prompt-fragment", "a",
        "--prompt-suffix", "z",
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["execute_requested"] is False
    assert plan["safety_class"] == "READ-ONLY REMOTE"
    assert [
        item["ssh_target"]
        for item in plan["live_arm_revalidation_before_first_http"]["actions"]
    ] == [f"rank{rank}.test" for rank in range(4)]
    assert plan["attribution_arm"] == ATTRIBUTION_ARM
    assert plan["cache_salt"] == CACHE_SALT
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation", ("image", "model", "kv", "wrong-arm", "missing-label")
)
def test_cli_rejects_activation_receipt_drift_before_requests(
    tmp_path, capsys, mutation
):
    identity = activation_args(tmp_path)
    receipt_path = Path(identity[identity.index("--activation-receipt") + 1])
    profile_path = Path(identity[identity.index("--profile") + 1])
    if mutation in ("wrong-arm", "missing-label"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if mutation == "wrong-arm":
            receipt["arm"] = "e-mtp0-apc0-lmcache1"
        else:
            del receipt["ranks"][0]["labels"]["org.sparkring.component"]
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    else:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if mutation == "image":
            profile["image_id"] = "sha256:" + "c" * 64
        elif mutation == "model":
            profile["model_repository"] = "example/changed-model"
        else:
            profile["environment"]["VLLM_SPARK_KV_CACHE_MEMORY_BYTES"] = (
                "4000000000"
            )
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
    output = tmp_path / "report.json"
    client = FakeClient()
    status = probe.main(
        [
            "--base-url", "http://example.invalid:8000",
            "--model", "m",
            "--attribution-arm", ATTRIBUTION_ARM,
            *identity,
            "--run-label", "receipt-drift",
            "--output", str(output),
            "--prompt-fragment", "a",
            "--prompt-suffix", "z",
            "--execute",
        ],
        client=client,
    )
    assert status == 2
    assert client.completions == 0
    assert len(client.metric_bodies) == 3
    assert not output.exists()
    assert "invalid --activation-receipt" in capsys.readouterr().err


@pytest.mark.parametrize(
    "reason",
    (
        "same-arm receipt is stale after all four containers were replaced",
        "runtime-unique StartedAt mismatch on rank 2",
        "SSH transport failed before rank attestation",
    ),
)
def test_live_revalidation_failure_makes_zero_http_requests(
    tmp_path, monkeypatch, capsys, reason
):
    identity = activation_args(tmp_path)
    output = tmp_path / "report.json"
    client = FakeClient()

    def fail_revalidation(*_args, **_kwargs):
        raise probe.attribution.exl3.ProfileError(reason)

    monkeypatch.setattr(probe.attribution, "revalidate_live_arm", fail_revalidation)
    status = probe.main(
        [
            "--base-url", "http://example.invalid:8000",
            "--model", "m",
            "--attribution-arm", ATTRIBUTION_ARM,
            *identity,
            "--run-label", "stale-receipt",
            "--output", str(output),
            "--prompt-fragment", "a",
            "--prompt-suffix", "z",
            "--execute",
        ],
        client=client,
    )
    assert status == 2
    assert client.requests == []
    assert not output.exists()
    assert "live-arm re-attestation failed" in capsys.readouterr().err


def test_cli_rejects_hidden_or_credentialed_url_state(tmp_path, capsys):
    for index, base_url in enumerate((
        "http://user:secret@example.invalid:8000",
        "http://example.invalid:8000/private",
        "file:///private/socket",
    )):
        assert probe.main([
            "--base-url", base_url,
            "--model", "m",
            "--attribution-arm", ATTRIBUTION_ARM,
            *activation_args(tmp_path),
            "--run-label", f"arm-c-probe-{index}",
            "--output", str(tmp_path / f"report-{index}.json"),
            "--prompt-fragment", "a",
            "--prompt-suffix", "z",
        ]) == 2
        assert "--base-url" in capsys.readouterr().err
