from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request

import pytest

import acceptance_gate
import exl3_attribution_compare as comparison
import exl3_correctness_gate as correctness
from exl3_attribution_cache_contract import (
    build_live_arm_receipt,
    cache_salt_for_arm,
    validate_live_arm_receipt,
)


MODEL = "glm-5.2-exl3-tr3-3.25bpw"
ATTRIBUTION_ARM = "d-mtp2-apc1-lmcache1"
CACHE_SALT = cache_salt_for_arm(ATTRIBUTION_ARM)


@pytest.fixture(autouse=True)
def offline_live_arm_revalidation(monkeypatch):
    monkeypatch.setattr(correctness, "load_site", lambda _path: object())
    monkeypatch.setattr(
        correctness.attribution.exl3, "load_profile", lambda _path: object()
    )
    monkeypatch.setattr(
        correctness.attribution,
        "live_arm_revalidation_actions",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        correctness.attribution,
        "revalidate_live_arm",
        lambda *_args, **_kwargs: {
            "status": "live-arm-re-attested",
            "rank_count": 4,
            "runtime_instances": [],
        },
    )


def activation_files(directory: Path, arm: str = ATTRIBUTION_ARM):
    profile_path = directory / "launch.json"
    receipt_path = directory / "live-arm-receipt.json"
    if not profile_path.exists():
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
    if not receipt_path.exists():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        receipt = build_live_arm_receipt(
            arm_id=arm,
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
    return profile_path, receipt_path


def token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(
        acceptance_gate.canonical_json(token_ids).encode("utf-8")
    ).hexdigest()


def write_config(tmp_path, *, expected="observed", duplicate=False):
    cases = [
        {
            "id": "focused-case",
            "prompt": "Return exactly the word stable.",
            "seed": 20260807,
            "max_tokens": 16,
            "expected_token_ids_sha256": (
                token_hash([10, 20, 30]) if expected == "observed" else expected
            ),
        }
    ]
    if duplicate:
        cases.append(dict(cases[0]))
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps({"schema": correctness.SCHEMA, "cases": cases}),
        encoding="utf-8",
    )
    return path


def argv(path, command="run"):
    profile, receipt = activation_files(path.parent)
    result = [
        "--config",
        str(path),
        "--site",
        str(path.parent / "site.yaml"),
        "--base-url",
        "http://rank0.test:8000",
        "--model",
        MODEL,
        "--attribution-arm",
        ATTRIBUTION_ARM,
        "--activation-receipt",
        str(receipt),
        "--profile",
        str(profile),
    ]
    probe = path.parent / "cache-metric-probe.json"
    if probe.exists():
        result += ["--cache-metric-probe", f"cache-boundary={probe}"]
    return result + [command]


class FakeHttp:
    def __init__(
        self,
        token_runs=None,
        completion_status=200,
        logprobs=None,
        prompt_token_ids=None,
        completion_metadata=None,
    ):
        self.token_runs = iter(token_runs or [[10, 20, 30]] * 3)
        self.prompt_token_ids = list(
            prompt_token_ids if prompt_token_ids is not None else range(32)
        )
        self.completion_status = completion_status
        self.logprobs = logprobs
        self.completion_metadata = completion_metadata or {}
        self.requests = []

    def get_json(self, url, timeout=30.0):
        self.requests.append(("GET", url))
        return 200, {"status": "ok"}

    def post_json(self, url, payload, timeout=30.0):
        self.requests.append(("POST", url, payload))
        if url.endswith("/v1/completions"):
            choice = {"text": "stable"}
            if self.logprobs is not None:
                choice["logprobs"] = self.logprobs
            return self.completion_status, {
                "choices": [choice],
                **self.completion_metadata,
            }
        if url.endswith("/tokenize"):
            if payload.get("prompt") == "stable":
                return 200, {"tokens": next(self.token_runs)}
            return 200, {"tokens": self.prompt_token_ids}
        return 404, {}


def test_plan_is_connection_free_and_allows_focused_case(
    tmp_path, monkeypatch, capsys
):
    path = write_config(tmp_path, expected=None)
    actions = [
        correctness.attribution.exl3.RemoteAction(
            rank,
            f"rank{rank}.test",
            ("sh", "-lc", f"test-rank-{rank}"),
        )
        for rank in range(4)
    ]
    monkeypatch.setattr(
        correctness.attribution,
        "live_arm_revalidation_actions",
        lambda *_args, **_kwargs: actions,
    )
    refusing = acceptance_gate.RefusingExecutor()
    assert correctness.main(argv(path, "plan"), http=refusing) == correctness.EXIT_OK
    plan = json.loads(capsys.readouterr().out)
    assert plan["mutates_remote"] is False
    assert plan["remote_safety_class"] == "READ-ONLY REMOTE"
    assert [
        item["ssh_target"]
        for item in plan["live_arm_revalidation_before_first_http"]["actions"]
    ] == [f"rank{rank}.test" for rank in range(4)]
    assert plan["execute_requested"] is False
    assert plan["contacted_base_url"] == "http://rank0.test:8000"
    assert plan["contacted_http_targets"] == [
        "http://rank0.test:8000/health",
        "http://rank0.test:8000/tokenize",
        "http://rank0.test:8000/v1/completions",
    ]
    assert plan["case_ids"] == ["focused-case"]
    assert plan["run_label"] == "unlabelled"
    assert plan["attribution_arm"] == ATTRIBUTION_ARM
    assert plan["cache_salt"] == CACHE_SALT


def test_run_without_execute_remains_dry(tmp_path, capsys):
    path = write_config(tmp_path)
    assert correctness.main(argv(path), http=acceptance_gate.RefusingExecutor()) == 0
    assert json.loads(capsys.readouterr().out)["execute_requested"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "image", "model", "kv", "wrong-arm", "missing-label",
        "bad-env-digest",
    ),
)
def test_activation_receipt_identity_drift_rejects_before_http(
    tmp_path, capsys, mutation
):
    path = write_config(tmp_path)
    profile_path, receipt_path = activation_files(tmp_path)
    if mutation in ("wrong-arm", "missing-label", "bad-env-digest"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if mutation == "wrong-arm":
            receipt["arm"] = "e-mtp0-apc0-lmcache1"
        elif mutation == "missing-label":
            del receipt["ranks"][0]["labels"]["org.sparkring.managed"]
        else:
            receipt["ranks"][0]["explicit_environment_sha256"] = "bad"
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
    http = FakeHttp()
    assert correctness.main(argv(path, "plan"), http=http) == (
        correctness.EXIT_CONFIG_ERROR
    )
    assert http.requests == []
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
    path = write_config(tmp_path)
    http = FakeHttp()

    def fail_revalidation(*_args, **_kwargs):
        raise correctness.attribution.exl3.ProfileError(reason)

    monkeypatch.setattr(
        correctness.attribution, "revalidate_live_arm", fail_revalidation
    )
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_FUNCTIONAL_FAIL
    assert http.requests == []
    assert "live-arm re-attestation failed" in capsys.readouterr().err


def test_expected_token_hash_passes(tmp_path, capsys):
    path = write_config(tmp_path)
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(args, http=FakeHttp()) == correctness.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "pass"
    assert report["cases"][0]["status"] == "pass"
    assert len(report["cases"][0]["observations"]) == 3


def test_null_expected_hash_records_baseline_but_does_not_pass(tmp_path, capsys):
    path = write_config(tmp_path, expected=None)
    args = argv(path)[:-1] + ["--execute", "run"]
    assert (
        correctness.main(args, http=FakeHttp())
        == correctness.EXIT_BASELINE_RECORDED
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "baseline-recorded"
    assert report["baselines"] == ["focused-case"]


def test_repeat_divergence_is_functional_failure(tmp_path, capsys):
    path = write_config(tmp_path)
    http = FakeHttp(token_runs=[[10, 20, 30], [10, 99], [10, 20, 30]])
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_FUNCTIONAL_FAIL
    report = json.loads(capsys.readouterr().out)
    assert report["failures"] == ["focused-case"]
    assert report["cases"][0]["failure"] == "repetitions diverged"
    case = report["cases"][0]
    assert case["token_id_source"] == "retokenized-completion-text"
    assert case["distinct_output_count"] == 2
    assert case["observations"][1]["repetition"] == 2
    assert case["observations"][0]["run_position"] == "first-in-run"
    assert case["observations"][1]["run_position"] == "subsequent-in-run"
    assert "not necessarily cold" in case["sequence_boundary"]
    assert case["divergences_from_repetition_1"] == [
        {
            "repetition": 2,
            "matches_repetition_1": False,
            "first_divergence_index": 1,
            "reference_token_count": 3,
            "observed_token_count": 2,
        },
        {
            "repetition": 3,
            "matches_repetition_1": True,
            "first_divergence_index": None,
            "reference_token_count": 3,
            "observed_token_count": 3,
        },
    ]


def test_run_label_is_preserved_in_report(tmp_path, capsys):
    path = write_config(tmp_path)
    args = argv(path)[:-1] + [
        "--run-label",
        "mtp0-apc0-lmcache0-rep1",
        "--execute",
        "run",
    ]
    assert correctness.main(args, http=FakeHttp()) == correctness.EXIT_OK
    assert (
        json.loads(capsys.readouterr().out)["run_label"]
        == "mtp0-apc0-lmcache0-rep1"
    )


def test_output_persists_exact_report_without_overwrite(tmp_path, capsys):
    path = write_config(tmp_path)
    output = tmp_path / "evidence" / "arm-a.json"
    args = argv(path)[:-1] + [
        "--run-label",
        "arm-a",
        "--output",
        str(output),
        "--execute",
        "run",
    ]
    assert correctness.main(args, http=FakeHttp()) == correctness.EXIT_OK
    stdout_report = json.loads(capsys.readouterr().out)
    stored_report = json.loads(output.read_text(encoding="utf-8"))
    assert stored_report == stdout_report
    assert stored_report["cases"][0]["observations"][0]["token_ids"] == [
        10,
        20,
        30,
    ]

    refusing = acceptance_gate.RefusingExecutor()
    assert correctness.main(args, http=refusing) == correctness.EXIT_CONFIG_ERROR
    assert "already exists" in capsys.readouterr().err


def test_top_logprobs_are_requested_and_persisted(tmp_path, capsys):
    path = write_config(tmp_path)
    logprobs = {
        "tokens": ["stable"],
        "token_logprobs": [-0.01],
        "top_logprobs": [{"stable": -0.01, "unstable": -4.2}],
        "text_offset": [0],
    }
    http = FakeHttp(logprobs=logprobs)
    args = argv(path)[:-1] + ["--top-logprobs", "2", "--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["top_logprobs"] == 2
    case = report["cases"][0]
    assert case["requested_top_logprobs"] == 2
    assert case["observations"][0]["completion_logprobs"] == logprobs
    completion_payloads = [
        payload
        for method, url, payload in (
            (item[0], item[1], item[2])
            for item in http.requests
            if len(item) == 3
        )
        if method == "POST" and url.endswith("/v1/completions")
    ]
    assert all(payload["logprobs"] == 2 for payload in completion_payloads)
    assert all(payload["cache_salt"] == CACHE_SALT for payload in completion_payloads)


def test_duplicate_case_is_configuration_error(tmp_path, capsys):
    path = write_config(tmp_path, duplicate=True)
    assert correctness.main(argv(path, "plan")) == correctness.EXIT_CONFIG_ERROR
    assert "duplicate correctness case id" in capsys.readouterr().err


def test_bad_expected_hash_is_configuration_error(tmp_path, capsys):
    path = write_config(tmp_path, expected="not-a-hash")
    assert correctness.main(argv(path, "plan")) == correctness.EXIT_CONFIG_ERROR
    assert "must be null or SHA-256" in capsys.readouterr().err


def test_http_failure_is_functional_failure(tmp_path, capsys):
    path = write_config(tmp_path)
    args = argv(path)[:-1] + ["--execute", "run"]
    assert (
        correctness.main(args, http=FakeHttp(completion_status=500))
        == correctness.EXIT_FUNCTIONAL_FAIL
    )
    assert "completion failed with HTTP 500" in capsys.readouterr().err


def cache_config(tmp_path, *, minimum=1024, prompt_token_ids=None):
    profile_path, receipt_path = activation_files(tmp_path)
    prompt_token_ids = list(
        range(1400) if prompt_token_ids is None else prompt_token_ids
    )
    fragment = "alpha beta gamma delta epsilon\n"
    repetitions = 256
    suffix = "Return exactly the word stable."
    prompt = fragment * repetitions + suffix
    document = {
        "schema": correctness.SCHEMA,
        "cases": [
            {
                "id": "cache-boundary",
                "prompt_generator": {
                    "kind": "repeated-prefix-v1",
                    "fragment": fragment,
                    "repetitions": repetitions,
                    "suffix": suffix,
                },
                "cache_attribution": True,
                "minimum_prompt_tokens": minimum,
                "seed": 20260809,
                "max_tokens": 16,
                "expected_token_ids_sha256": None,
            }
        ],
    }
    path = tmp_path / "cache-cases.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    probe = {
        "schema": correctness.METRIC_PROBE_SCHEMA,
        "run_label": "test-cache-probe",
        "model": MODEL,
        "attribution_arm": ATTRIBUTION_ARM,
        "cache_salt": CACHE_SALT,
        "live_arm_receipt": validate_live_arm_receipt(
            receipt_path, profile_path, ATTRIBUTION_ARM
        ),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_token_count": len(prompt_token_ids),
        "prompt_token_ids_sha256": hashlib.sha256(
            acceptance_gate.canonical_json(prompt_token_ids).encode("utf-8")
        ).hexdigest(),
        "geometry": {
            "physical_block_tokens_per_dcp_rank": 64,
            "dcp_degree": 4,
            "dcp_global_apc_alignment_tokens": 256,
            "lmcache_chunk_tokens": 512,
            "physical_blocks_per_dcp_global_apc_unit_per_rank": 1,
            "dcp_global_apc_units_per_lmcache_chunk": 2,
            "physical_blocks_per_lmcache_chunk_per_rank": 2,
        },
        "initial_snapshot": {
            "cache_config": {
                "physical_block_tokens_per_dcp_rank": 64,
                "enable_prefix_caching": "True",
                "kv_cache_size_tokens": 562688,
                "num_gpu_blocks": 2198,
            },
            "counters": {
                "vllm:prefix_cache_queries_total": 0.0,
                "vllm:prefix_cache_hits_total": 0.0,
                "vllm:external_prefix_cache_queries_total": 0.0,
                "vllm:external_prefix_cache_hits_total": 0.0,
            },
            "prompt_tokens_by_source": {
                "local_compute": 0.0,
                "local_cache_hit": 0.0,
                "external_kv_transfer": 0.0,
            },
        },
        "observations": [
            {
                "repetition": repetition,
                "metric_interval_delta": {
                    "interval_ns": 1,
                    "counters": {
                        "vllm:prefix_cache_queries_total": 1.0,
                        "vllm:prefix_cache_hits_total": 0.0,
                        "vllm:external_prefix_cache_queries_total": 0.0,
                        "vllm:external_prefix_cache_hits_total": 0.0,
                    },
                    "prompt_tokens_by_source": {
                        "local_compute": 1.0,
                        "local_cache_hit": 0.0,
                        "external_kv_transfer": 0.0,
                    },
                },
            }
            for repetition in (1, 2)
        ],
        "evidence_scope": {
            "classification": "request-interval-correlated-prometheus-counter-delta"
        },
    }
    (tmp_path / "cache-metric-probe.json").write_text(
        json.dumps(probe), encoding="utf-8"
    )
    return path


def test_published_cache_config_generates_deterministic_long_prefix():
    path = Path("scripts/config/exl3-correctness-cache.example.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    cases, _ = correctness.load_cases(path)
    assert len(cases) == 2
    prompts = [correctness.case_prompt(case, case["id"]) for case in cases]
    assert prompts[0] == correctness.case_prompt(cases[0], cases[0]["id"])
    assert len(prompts[0].split()) > 1024
    assert prompts[0].split("Task:", 1)[0] == prompts[1].split("Task:", 1)[0]
    assert all(case["cache_attribution"] for case in cases)
    assert all(case["ignore_eos"] is True for case in cases)
    assert document["cache_metric_probe_case_ids"] == [
        case["id"] for case in cases
    ]


def test_declared_metric_probe_case_ids_must_match_cache_cases(tmp_path, capsys):
    path = cache_config(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["cache_metric_probe_case_ids"] = ["wrong-case"]
    path.write_text(json.dumps(document), encoding="utf-8")
    assert correctness.main(argv(path, "plan")) == correctness.EXIT_CONFIG_ERROR
    assert "must exactly match" in capsys.readouterr().err


def test_ignore_eos_is_validated_requested_and_recorded(tmp_path, capsys):
    path = cache_config(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["cases"][0]["ignore_eos"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    http = FakeHttp(prompt_token_ids=range(1400))
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_BASELINE_RECORDED
    case = json.loads(capsys.readouterr().out)["cases"][0]
    assert case["ignore_eos"] is True
    completion_payloads = [
        item[2]
        for item in http.requests
        if len(item) == 3 and item[1].endswith("/v1/completions")
    ]
    assert completion_payloads
    assert all(payload["ignore_eos"] is True for payload in completion_payloads)


def test_ignore_eos_must_be_boolean(tmp_path, capsys):
    path = cache_config(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["cases"][0]["ignore_eos"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")
    assert correctness.main(argv(path, "plan")) == correctness.EXIT_CONFIG_ERROR
    assert "ignore_eos must be boolean" in capsys.readouterr().err


def test_cache_case_reports_prompt_boundaries_when_qualified(tmp_path, capsys):
    path = cache_config(tmp_path)
    http = FakeHttp(prompt_token_ids=range(1400))
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_BASELINE_RECORDED
    case = json.loads(capsys.readouterr().out)["cases"][0]
    assert case["prompt_token_count"] == 1400
    assert case["cache_attribution"] is True
    assert case["cache_boundaries"] == {
        "physical_block_tokens_per_dcp_rank": 64,
        "dcp_degree": 4,
        "dcp_global_apc_alignment_tokens": 256,
        "lmcache_chunk_tokens": 512,
        "minimum_prompt_tokens": 1024,
        "reusable_prompt_tokens": 1399,
        "reusable_dcp_global_apc_units": 5,
        "reusable_lmcache_chunks": 2,
        "has_reusable_dcp_global_apc_unit": True,
        "has_reusable_lmcache_chunk": True,
        "qualifies_for_cache_attribution": True,
    }
    assert len(case["observations"]) == 3
    assert case["cache_evidence_scope"]["causal_cache_claim"] == (
        "not-claimed-store-evidence-unavailable"
    )
    assert case["cache_evidence_scope"]["request_correlated_hit_evidence_count"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (("prompt_token_ids_sha256", "e" * 64), ("prompt_token_count", 1399)),
)
def test_cache_metric_probe_must_bind_exact_correctness_prompt(
    tmp_path, capsys, field, value
):
    path = cache_config(tmp_path)
    probe_path = tmp_path / "cache-metric-probe.json"
    document = json.loads(probe_path.read_text(encoding="utf-8"))
    document[field] = value
    probe_path.write_text(json.dumps(document), encoding="utf-8")
    http = FakeHttp(prompt_token_ids=range(1400))
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_FUNCTIONAL_FAIL
    assert not any(
        len(request) == 3 and request[1].endswith("/v1/completions")
        for request in http.requests
    )
    assert f"metric probe probe_{field}" in capsys.readouterr().err


def test_cache_metric_probe_text_mismatch_is_rejected_before_http(tmp_path, capsys):
    path = cache_config(tmp_path)
    probe_path = tmp_path / "cache-metric-probe.json"
    document = json.loads(probe_path.read_text(encoding="utf-8"))
    document["prompt_sha256"] = "f" * 64
    probe_path.write_text(json.dumps(document), encoding="utf-8")
    http = FakeHttp(prompt_token_ids=range(1400))
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_CONFIG_ERROR
    assert http.requests == []
    assert "prompt text does not match" in capsys.readouterr().err


def test_request_correlated_cached_usage_is_bound_but_store_claim_stays_closed(
    tmp_path, capsys
):
    path = cache_config(tmp_path)
    http = FakeHttp(
        prompt_token_ids=range(1400),
        completion_metadata={
            "id": "cmpl-private-request-id",
            "usage": {
                "prompt_tokens": 1400,
                "prompt_tokens_details": {"cached_tokens": 1024},
            },
        },
    )
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_BASELINE_RECORDED
    case = json.loads(capsys.readouterr().out)["cases"][0]
    observation = case["observations"][0]
    assert observation["request_evidence"]["response_id_sha256"] == hashlib.sha256(
        b"cmpl-private-request-id"
    ).hexdigest()
    assert observation["request_evidence"]["usage_cached_prompt_tokens"] == 1024
    assert "cmpl-private-request-id" not in json.dumps(case)
    assert case["cache_evidence_scope"]["request_correlated_hit_evidence_count"] == 3
    assert case["cache_evidence_scope"]["request_correlated_store_evidence_count"] == 0
    assert case["cache_evidence_scope"]["causal_cache_claim"] == (
        "not-claimed-store-evidence-unavailable"
    )


def test_zero_cached_tokens_is_metadata_but_not_positive_hit_evidence(
    tmp_path, capsys
):
    path = cache_config(tmp_path)
    http = FakeHttp(
        prompt_token_ids=range(1400),
        completion_metadata={
            "id": "cmpl-zero-cache",
            "usage": {
                "prompt_tokens": 1400,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
    )
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_BASELINE_RECORDED
    report = json.loads(capsys.readouterr().out)
    assert report["contacted_base_url"] == "http://rank0.test:8000"
    case = report["cases"][0]
    assert case["cache_evidence_scope"]["request_correlated_hit_evidence_count"] == 0
    assert case["observations"][0]["request_evidence"]["hit_evidence_source"] is None
    assert case["cache_evidence_scope"]["request_correlated_store_evidence_count"] == 0


@pytest.mark.parametrize(
    "base_url",
    [
        "http://user:secret@rank0.test:8000",
        "http://rank0.test:8000/private",
        "file:///private/socket",
        "http://%31%39%32%2e%31%36%38%2e%30%2e%31%39%33:8000",
        "http://rank0%2etest:8000",
        "http://rank 0.test:8000",
        "http://rank0.test\t:8000",
        "http://rank_0.test:8000",
        "http://-rank0.test:8000",
        "http://rank0..test:8000",
        "http://tést.test:8000",
        "http://rank0.test\\evil:8000",
    ],
)
def test_base_url_plan_is_exact_and_rejects_unsafe_forms(
    tmp_path, capsys, base_url
):
    path = write_config(tmp_path, expected=None)
    args = argv(path, "plan")
    args[args.index("--base-url") + 1] = base_url
    assert correctness.main(args) == correctness.EXIT_CONFIG_ERROR
    assert "--base-url" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://192.0.2.193:8000", "http://192.0.2.193:8000"),
        ("HTTP://RANK0.TEST:8000/", "http://rank0.test:8000"),
        ("https://[::1]:8443", "https://[::1]:8443"),
    ],
)
def test_base_url_plan_matches_urllib_authority_exactly(
    tmp_path, capsys, base_url, expected
):
    path = write_config(tmp_path, expected=None)
    args = argv(path, "plan")
    args[args.index("--base-url") + 1] = base_url
    assert correctness.main(args) == correctness.EXIT_OK
    plan = json.loads(capsys.readouterr().out)
    assert plan["contacted_base_url"] == expected
    assert Request(f"{expected}/health").host == expected.split("//", 1)[1]


@pytest.mark.parametrize(
    "run_label",
    [
        "../../private",
        "arm-a?token=secret",
        "secret value",
        "https://private.example/run",
        "a" * 129,
        "-leading-dash",
        "",
    ],
)
def test_run_label_rejects_paths_queries_whitespace_urls_and_unbounded_values(
    tmp_path, capsys, run_label
):
    path = write_config(tmp_path, expected=None)
    args = argv(path, "plan")
    args[0:0] = [f"--run-label={run_label}"]
    assert correctness.main(args) == correctness.EXIT_CONFIG_ERROR
    assert "--run-label" in capsys.readouterr().err


def test_run_label_accepts_bounded_public_identifier_and_emits_exactly(
    tmp_path, capsys
):
    path = write_config(tmp_path, expected=None)
    args = argv(path, "plan")
    args[0:0] = ["--run-label", "arm-b.mtp2_apc0-20260809"]
    assert correctness.main(args) == correctness.EXIT_OK
    plan = json.loads(capsys.readouterr().out)
    assert plan["run_label"] == "arm-b.mtp2_apc0-20260809"


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 1399},
        {"prompt_tokens": True},
        {
            "prompt_tokens": 1400,
            "prompt_tokens_details": {"cached_tokens": True},
        },
        {
            "prompt_tokens": 1400,
            "prompt_tokens_details": {"cached_tokens": 1401},
        },
    ],
)
def test_request_cache_metadata_must_bind_prompt_and_use_real_counts(
    tmp_path, capsys, usage
):
    path = cache_config(tmp_path)
    http = FakeHttp(
        prompt_token_ids=range(1400),
        completion_metadata={"id": "cmpl-id", "usage": usage},
    )
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_FUNCTIONAL_FAIL
    assert "case cache-boundary" in capsys.readouterr().err


def test_cache_case_below_boundary_fails_before_completion(tmp_path, capsys):
    path = cache_config(tmp_path, prompt_token_ids=range(511))
    http = FakeHttp(prompt_token_ids=range(511))
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_FUNCTIONAL_FAIL
    report = json.loads(capsys.readouterr().out)
    case = report["cases"][0]
    assert case["status"] == "fail"
    assert case["prompt_token_count"] == 511
    assert case["cache_boundaries"]["has_reusable_dcp_global_apc_unit"] is True
    assert case["cache_boundaries"]["has_reusable_lmcache_chunk"] is False
    assert case["cache_boundaries"]["qualifies_for_cache_attribution"] is False
    assert case["observations"] == []
    assert not any(
        len(item) == 3 and item[1].endswith("/v1/completions")
        for item in http.requests
    )


@pytest.mark.parametrize(
    ("prompt_tokens", "apc", "lmcache", "status"),
    [
        (256, False, False, correctness.EXIT_FUNCTIONAL_FAIL),
        (257, True, False, correctness.EXIT_FUNCTIONAL_FAIL),
        (512, True, False, correctness.EXIT_FUNCTIONAL_FAIL),
        (513, True, True, correctness.EXIT_BASELINE_RECORDED),
    ],
)
def test_cache_reuse_boundaries_use_prompt_tokens_minus_one(
    tmp_path, capsys, prompt_tokens, apc, lmcache, status
):
    path = cache_config(
        tmp_path, minimum=512, prompt_token_ids=range(prompt_tokens)
    )
    args = argv(path)[:-1] + ["--execute", "run"]
    assert correctness.main(
        args, http=FakeHttp(prompt_token_ids=range(prompt_tokens))
    ) == status
    case = json.loads(capsys.readouterr().out)["cases"][0]
    assert case["cache_boundaries"]["reusable_prompt_tokens"] == prompt_tokens - 1
    assert case["cache_boundaries"]["has_reusable_dcp_global_apc_unit"] is apc
    assert case["cache_boundaries"]["has_reusable_lmcache_chunk"] is lmcache


def test_cache_case_requires_one_named_metric_probe_mapping(tmp_path, capsys):
    path = cache_config(tmp_path)
    (tmp_path / "cache-metric-probe.json").unlink()
    assert correctness.main(argv(path, "plan")) == correctness.EXIT_CONFIG_ERROR
    assert "missing --cache-metric-probe mappings" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mapping", "message"),
    (
        ("unknown-case=unused.json", "maps unknown case"),
        ("focused-case=unused.json", "maps non-cache-attribution case"),
        ("not-a-mapping", "CASE_ID=PATH syntax"),
    ),
)
def test_invalid_metric_probe_mapping_is_rejected_before_http(
    tmp_path, capsys, mapping, message
):
    path = write_config(tmp_path)
    http = FakeHttp()
    args = argv(path)[:-1] + ["--cache-metric-probe", mapping, "--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_CONFIG_ERROR
    assert http.requests == []
    assert message in capsys.readouterr().err


def test_duplicate_metric_probe_mapping_is_rejected_before_http(tmp_path, capsys):
    path = cache_config(tmp_path)
    mapping = f"cache-boundary={tmp_path / 'cache-metric-probe.json'}"
    http = FakeHttp()
    args = argv(path)[:-1] + ["--cache-metric-probe", mapping, "--execute", "run"]
    assert correctness.main(args, http=http) == correctness.EXIT_CONFIG_ERROR
    assert http.requests == []
    assert "duplicate --cache-metric-probe" in capsys.readouterr().err


def test_published_multi_case_probes_produce_comparator_accepted_v4_report(
    tmp_path, capsys
):
    generated_config = cache_config(tmp_path)
    template_probe_path = tmp_path / "cache-metric-probe.json"
    template_probe = json.loads(template_probe_path.read_text(encoding="utf-8"))
    template_probe_path.unlink()
    published = json.loads(
        Path("scripts/config/exl3-correctness-cache.example.json").read_text(
            encoding="utf-8"
        )
    )
    generated_config.write_text(json.dumps(published), encoding="utf-8")
    prompt_ids = list(range(1400))
    mappings = []
    for case in published["cases"]:
        prompt = correctness.case_prompt(case, case["id"])
        probe = dict(template_probe)
        probe["run_label"] = f"probe-{case['id']}"
        probe["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        probe["prompt_token_count"] = len(prompt_ids)
        probe["prompt_token_ids_sha256"] = hashlib.sha256(
            acceptance_gate.canonical_json(prompt_ids).encode("utf-8")
        ).hexdigest()
        probe_path = tmp_path / f"{case['id']}-probe.json"
        probe_path.write_text(json.dumps(probe), encoding="utf-8")
        mappings.extend(["--cache-metric-probe", f"{case['id']}={probe_path}"])
    args = argv(generated_config)[:-1] + mappings + ["--execute", "run"]
    http = FakeHttp(
        prompt_token_ids=prompt_ids,
        token_runs=[[10, 20, 30]] * (3 * len(published["cases"])),
    )
    assert correctness.main(args, http=http) == correctness.EXIT_BASELINE_RECORDED
    report = json.loads(capsys.readouterr().out)
    assert [case["id"] for case in report["cases"]] == [
        case["id"] for case in published["cases"]
    ]
    compared = comparison.compare_reports(
        report,
        report,
        left_sha256="a" * 64,
        right_sha256="b" * 64,
    )
    assert compared["comparison_status"] == "exact-match"
    assert all(
        case["cache_geometry_evidence"]["scope"]
        == "validated-metric-probe-and-cache-namespace-bound-v4"
        for case in compared["cases"]
    )


def test_legacy_probe_geometry_is_rejected_as_unbound(tmp_path):
    cache_config(tmp_path)
    probe_path = tmp_path / "cache-metric-probe.json"
    document = json.loads(probe_path.read_text(encoding="utf-8"))
    document["geometry"] = {
        "physical_block_tokens_per_dcp_rank": 64,
        "dcp_degree": 4,
        "logical_apc_block_tokens": 256,
        "lmcache_chunk_tokens": 512,
        "physical_blocks_per_lmcache_chunk_per_rank": 2,
    }
    probe_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(correctness.ConfigError, match="does not bind"):
        correctness.load_metric_probe(probe_path)


def test_metric_probe_rejects_duplicate_json_keys(tmp_path):
    cache_config(tmp_path)
    probe_path = tmp_path / "cache-metric-probe.json"
    rendered = probe_path.read_text(encoding="utf-8")
    probe_path.write_text(
        rendered.replace(
            '"schema": "sparkring-exl3-cache-metric-probe/v2",',
            '"schema": "wrong", "schema": "sparkring-exl3-cache-metric-probe/v2",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(correctness.ConfigError, match="duplicate JSON object key"):
        correctness.load_metric_probe(probe_path)


def test_metric_probe_rejects_cache_salt_not_derived_from_layout(tmp_path):
    cache_config(tmp_path)
    probe_path = tmp_path / "cache-metric-probe.json"
    document = json.loads(probe_path.read_text(encoding="utf-8"))
    document["cache_salt"] = cache_salt_for_arm(
        "e-mtp0-apc0-lmcache1"
    )
    probe_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(correctness.ConfigError, match="does not bind"):
        correctness.load_metric_probe(probe_path)


def test_metric_probe_binds_apc_off_external_cache_evidence(tmp_path):
    cache_config(tmp_path)
    probe_path = tmp_path / "cache-metric-probe.json"
    document = json.loads(probe_path.read_text(encoding="utf-8"))
    document["initial_snapshot"]["cache_config"]["enable_prefix_caching"] = "False"
    for observation in document["observations"]:
        delta = observation["metric_interval_delta"]
        delta["counters"]["vllm:prefix_cache_queries_total"] = 0.0
        delta["counters"]["vllm:external_prefix_cache_queries_total"] = 900.0
        delta["counters"]["vllm:external_prefix_cache_hits_total"] = 512.0
        delta["prompt_tokens_by_source"] = {
            "local_compute": 388.0,
            "local_cache_hit": 0.0,
            "external_kv_transfer": 512.0,
        }
    probe_path.write_text(json.dumps(document), encoding="utf-8")
    probe = correctness.load_metric_probe(probe_path)
    assert probe["native_prefix_caching_enabled"] is False
    assert probe["observed_cache_layers"] == ["external-kv-transfer"]
    assert probe["aggregate_prompt_tokens_by_source"] == {
        "external_kv_transfer": 1024.0,
        "local_cache_hit": 0.0,
        "local_compute": 776.0,
    }


@pytest.mark.parametrize(
    ("family", "name"),
    (
        ("counters", "vllm:prefix_cache_hits_total"),
        ("prompt_tokens_by_source", "local_cache_hit"),
    ),
)
def test_metric_probe_rejects_apc_off_with_native_hit_evidence(
    tmp_path, family, name
):
    cache_config(tmp_path)
    probe_path = tmp_path / "cache-metric-probe.json"
    document = json.loads(probe_path.read_text(encoding="utf-8"))
    document["initial_snapshot"]["cache_config"]["enable_prefix_caching"] = "False"
    document["observations"][0]["metric_interval_delta"]["counters"][
        "vllm:prefix_cache_queries_total"
    ] = 0.0
    document["observations"][1]["metric_interval_delta"]["counters"][
        "vllm:prefix_cache_queries_total"
    ] = 0.0
    document["observations"][0]["metric_interval_delta"][family][name] = 1.0
    probe_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(correctness.ConfigError, match="native prefix|local cache-hit"):
        correctness.load_metric_probe(probe_path)


@pytest.mark.parametrize(
    ("family", "key", "value", "message"),
    [
        ("counters", "vllm:prefix_cache_queries_total", -1, "finite nonnegative"),
        ("counters", "vllm:prefix_cache_queries_total", True, "finite nonnegative"),
        ("counters", "vllm:prefix_cache_queries_total", "https://private.invalid", "finite nonnegative"),
        ("prompt_tokens_by_source", "local_compute", float("nan"), "non-finite JSON"),
        ("prompt_tokens_by_source", "local_compute", float("inf"), "non-finite JSON"),
    ],
)
def test_metric_probe_rejects_malicious_initial_counter_values(
    tmp_path, family, key, value, message
):
    cache_config(tmp_path)
    probe_path = tmp_path / "cache-metric-probe.json"
    document = json.loads(probe_path.read_text(encoding="utf-8"))
    document["initial_snapshot"][family][key] = value
    probe_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(correctness.ConfigError, match=message):
        correctness.load_metric_probe(probe_path)


@pytest.mark.parametrize("value", [-1, True, "https://private.invalid", float("nan"), float("inf"), 1e309])
def test_metric_probe_rejects_malicious_interval_delta_values(tmp_path, value):
    cache_config(tmp_path)
    probe_path = tmp_path / "cache-metric-probe.json"
    document = json.loads(probe_path.read_text(encoding="utf-8"))
    document["observations"][0]["metric_interval_delta"]["counters"][
        "vllm:prefix_cache_hits_total"
    ] = value
    probe_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        correctness.ConfigError,
        match="non-finite JSON|finite nonnegative",
    ):
        correctness.load_metric_probe(probe_path)


@pytest.mark.parametrize(
    "generator",
    [
        {"kind": "unknown", "fragment": "x", "repetitions": 2, "suffix": "y"},
        {"kind": "repeated-prefix-v1", "fragment": "x", "repetitions": 0, "suffix": "y"},
    ],
)
def test_malformed_prompt_generator_is_configuration_error(
    tmp_path, capsys, generator
):
    path = cache_config(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["cases"][0]["prompt_generator"] = generator
    path.write_text(json.dumps(document), encoding="utf-8")
    assert correctness.main(argv(path, "plan")) == correctness.EXIT_CONFIG_ERROR
    assert "prompt_generator" in capsys.readouterr().err
