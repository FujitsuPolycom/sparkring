from __future__ import annotations

import hashlib
import json

import exl3_teacher_forced_margin_probe as probe
from exl3_attribution_cache_contract import build_live_arm_receipt, cache_salt_for_arm
import pytest


ARM = "a-mtp0-apc0-lmcache0"
MODEL = "glm-5.2-exl3-tr3-3.25bpw"


def activation_files(directory):
    profile_path = directory / "launch.json"
    receipt_path = directory / "receipt.json"
    profile = {
        "profile_id": "glm52-exl3-tr3-3.25bpw-lmcache-cs512",
        "image_id": "sha256:" + "a" * 64,
        "model_repository": "willfalco/GLM-5.2-EXL3-TR3-3.25bpw",
        "model_revision": "d7d79c2d14599dfce7a5d12b85f7ad73f40e623d",
        "container_name": "glm52-sparkring-exl3-lmcache-cs512",
        "environment": {
            "VLLM_SPARK_KV_CACHE_MEMORY_BYTES": "4500000000",
            "VLLM_SPARK_MAX_MODEL_LEN": "524288",
        },
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    receipt = build_live_arm_receipt(
        arm_id=ARM,
        canonical_profile_id=profile["profile_id"],
        canonical_profile_file_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        image_id=profile["image_id"],
        model_repository=profile["model_repository"],
        model_revision=profile["model_revision"],
        canonical_container_name=profile["container_name"],
        explicit_environment_sha256=[
            hashlib.sha256(f"env-{rank}".encode()).hexdigest() for rank in range(4)
        ],
        config_cmd_sha256=[
            hashlib.sha256(f"cmd-{rank}".encode()).hexdigest() for rank in range(4)
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


def cli_args(tmp_path, command="plan"):
    profile, receipt = activation_files(tmp_path)
    config = tmp_path / "cases.json"
    config.write_text(
        json.dumps(
            {
                "schema": "sparkring-exl3-correctness-cases/v1",
                "cases": [
                    {
                        "id": "focused-case",
                        "prompt": "prompt",
                        "seed": 20260809,
                        "max_tokens": 140,
                        "ignore_eos": True,
                        "expected_token_ids_sha256": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return [
        "--config",
        str(config),
        "--case-id",
        "focused-case",
        "--site",
        str(tmp_path / "site.yaml"),
        "--profile",
        str(profile),
        "--activation-receipt",
        str(receipt),
        "--base-url",
        "http://rank0.test:8000",
        "--model",
        MODEL,
        "--attribution-arm",
        ARM,
        "--discovery-repetitions",
        "2",
        "--teacher-forced-repetitions",
        "2",
        "--focus-generated-index",
        "1",
        "--window-before",
        "0",
        "--window-after",
        "0",
        command,
    ]


class FakeHttp:
    def __init__(self):
        self.requests = []
        self.discovery = iter(([10, 20, 30], [10, 21, 31]))
        self.teacher_values = iter(
            (
                {"20": {"logprob": -1.0, "rank": 1}, "21": {"logprob": -1.25, "rank": 2}},
                {"20": {"logprob": -1.25, "rank": 2}, "21": {"logprob": -1.0, "rank": 1}},
            )
        )

    def post_json(self, url, payload, timeout):
        self.requests.append((url, payload))
        if url.endswith("/tokenize"):
            return 200, {"tokens": [101, 102]}
        if payload["max_tokens"] != 0:
            return 200, {
                "choices": [
                    {
                        "text": "ignored",
                        "token_ids": list(next(self.discovery)),
                        "logprobs": {
                            "tokens": [],
                            "token_logprobs": [],
                            "top_logprobs": [],
                            "text_offset": [],
                        },
                    }
                ]
            }
        values = next(self.teacher_values)
        prompt_ids = payload["prompt"]
        prompt_logprobs = [None] * len(prompt_ids)
        prompt_logprobs[3] = values
        return 200, {
            "choices": [
                {
                    "text": "",
                    "prompt_token_ids": prompt_ids,
                    "prompt_logprobs": prompt_logprobs,
                }
            ]
        }


def test_teacher_forces_exact_token_context_and_detects_margin_flip():
    http = FakeHttp()
    result = probe.run_probe(
        http,
        base_url="http://rank0.test:8000",
        model="glm-5.2-exl3-tr3-3.25bpw",
        prompt="prompt",
        case_id="focused-case",
        seed=20260809,
        discovery_repetitions=2,
        discovery_max_tokens=3,
        focus_generated_index=1,
        window_before=0,
        window_after=0,
        teacher_forced_repetitions=2,
        top_logprobs=20,
        timeout=1.0,
        cache_salt=cache_salt_for_arm(ARM),
    )

    teacher_requests = [
        payload
        for url, payload in http.requests
        if url.endswith("/v1/completions") and payload["max_tokens"] == 0
    ]
    assert [request["prompt"] for request in teacher_requests] == [
        [101, 102, 10, 20],
        [101, 102, 10, 20],
    ]
    assert all(request["add_special_tokens"] is False for request in teacher_requests)
    assert all(request["prompt_logprobs"] == 20 for request in teacher_requests)
    position = result["teacher_forced_positions"][0]
    assert position["generated_index"] == 1
    assert position["absolute_prompt_index"] == 3
    assert position["summary"]["distinct_top1_count"] == 2
    assert position["summary"]["candidate_margin_sign_changes"] is True
    assert position["summary"]["classification"] == (
        "same-context-forward-ranking-nondeterminism"
    )
    assert result["diagnostic_classification"] == (
        "cache-not-required-forward-ranking-nondeterminism-observed"
    )


def test_summary_reports_bounded_topk_distribution_drift():
    result = probe.run_probe(
        FakeHttp(),
        base_url="http://rank0.test:8000",
        model="glm-5.2-exl3-tr3-3.25bpw",
        prompt="prompt",
        case_id="focused-case",
        seed=20260809,
        discovery_repetitions=2,
        discovery_max_tokens=3,
        focus_generated_index=1,
        window_before=0,
        window_after=0,
        teacher_forced_repetitions=2,
        top_logprobs=20,
        timeout=1.0,
        cache_salt=cache_salt_for_arm(ARM),
    )

    summary = result["teacher_forced_positions"][0]["summary"]
    assert summary["minimum_pairwise_topk_jaccard"] == 1.0
    assert summary["max_abs_common_token_value_delta"] == 0.25
    assert summary["maximum_conditional_common_support_symmetric_kl"] > 0
    assert summary["distribution_metric_scope"] == (
        "truncated-returned-common-support-renormalized-not-full-vocabulary-kld"
    )


def test_later_position_top1_flip_dominates_stable_focus_classification():
    positions = [
        {
            "generated_index": 1,
            "summary": {
                "classification": "teacher-forced-top1-stable-returned-values-vary"
            },
        },
        {
            "generated_index": 2,
            "summary": {"classification": "same-context-top1-nondeterminism"},
        },
    ]

    assert probe._diagnostic_classification(positions, 1) == (
        "cache-not-required-top1-nondeterminism-observed"
    )


def test_plan_is_private_and_makes_no_http_requests(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(probe, "load_site", lambda _path: object())
    monkeypatch.setattr(
        probe.attribution.exl3,
        "load_profile",
        lambda _path: probe.attribution.exl3.Profile({"extra_vllm_args": []}),
    )
    monkeypatch.setattr(
        probe.attribution, "live_arm_revalidation_actions", lambda *_args, **_kwargs: []
    )

    class NoHttp:
        def get_json(self, *_args, **_kwargs):
            raise AssertionError("plan contacted HTTP")

        def post_json(self, *_args, **_kwargs):
            raise AssertionError("plan contacted HTTP")

    assert probe.main(cli_args(tmp_path), http=NoHttp()) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["schema"] == probe.PLAN_SCHEMA
    assert plan["remote_safety_class"] == "READ-ONLY REMOTE"
    assert plan["execute_requested"] is False
    assert plan["evidence_policy"]["raw_plan_and_report_are_private"] is True
    assert plan["live_arm_revalidation_before_first_http"]["required"] is True
    assert plan["runtime_source_pins"]["vllm_commit"] == (
        "668275901b55230f4a70841a9aac1c0be22ef8d3"
    )
    assert plan["runtime_source_pins"]["evidence_scope"] == (
        "declared-canonical-pins-bound-to-receipt-model-not-live-binary-introspection"
    )


def test_execute_reattests_before_http_and_exclusive_creates_report(
    tmp_path, monkeypatch
):
    events = []
    monkeypatch.setattr(probe, "load_site", lambda _path: object())
    monkeypatch.setattr(
        probe.attribution.exl3,
        "load_profile",
        lambda _path: probe.attribution.exl3.Profile({"extra_vllm_args": []}),
    )
    monkeypatch.setattr(
        probe.attribution, "live_arm_revalidation_actions", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        probe.attribution,
        "revalidate_live_arm",
        lambda *_args, **_kwargs: events.append("reattest")
        or {"status": "live-arm-re-attested", "rank_count": 4},
    )

    class OrderedHttp(FakeHttp):
        def get_json(self, url, timeout):
            events.append(("GET", url))
            # The real vLLM /health endpoint has an empty, non-JSON body.
            return 200, probe.StrictJsonFailure("empty response body")

        def post_json(self, url, payload, timeout):
            events.append(("POST", url))
            return super().post_json(url, payload, timeout)

    output = tmp_path / "private-report.json"
    args = cli_args(tmp_path, command="run")
    args = args[:-1] + ["--output", str(output), "--execute", args[-1]]
    assert probe.main(args, http=OrderedHttp()) == 0
    assert events[0] == "reattest"
    assert events[1] == ("GET", "http://rank0.test:8000/health")
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == probe.REPORT_SCHEMA
    assert report["runtime_identity"]["live_arm_revalidation"]["status"] == (
        "live-arm-re-attested"
    )
    assert report["evidence_policy"]["raw_report_is_private"] is True

    assert probe.main(args, http=OrderedHttp()) == probe.EXIT_CONFIG_ERROR


def test_forced_target_may_be_returned_outside_requested_topk():
    http = FakeHttp()
    http.discovery = iter(([10, 30, 40], [10, 30, 40]))
    outside = {
        "20": {"logprob": -1.0, "rank": 1},
        "21": {"logprob": -1.25, "rank": 2},
        "30": {"logprob": -8.0, "rank": 30},
    }
    http.teacher_values = iter((outside, outside))
    result = probe.run_probe(
        http,
        base_url="http://rank0.test:8000",
        model=MODEL,
        prompt="prompt",
        case_id="focused-case",
        seed=20260809,
        discovery_repetitions=2,
        discovery_max_tokens=3,
        focus_generated_index=1,
        window_before=0,
        window_after=0,
        teacher_forced_repetitions=2,
        top_logprobs=20,
        timeout=1.0,
        cache_salt=cache_salt_for_arm(ARM),
    )
    observation = result["teacher_forced_positions"][0]["observations"][0]
    assert observation["forced_token_id"] == 30
    assert observation["forced_token_rank"] == 30
    assert result["autoregressive_discovery"]["earliest_divergence_index"] is None


def test_nonfinite_teacher_forced_value_fails_closed():
    http = FakeHttp()
    malformed = {
        "20": {"logprob": float("nan"), "rank": 1},
        "21": {"logprob": -1.25, "rank": 2},
    }
    http.teacher_values = iter((malformed, malformed))
    with pytest.raises(probe.RequestFailure, match="invalid value or rank"):
        probe.run_probe(
            http,
            base_url="http://rank0.test:8000",
            model=MODEL,
            prompt="prompt",
            case_id="focused-case",
            seed=20260809,
            discovery_repetitions=2,
            discovery_max_tokens=3,
            focus_generated_index=1,
            window_before=0,
            window_after=0,
            teacher_forced_repetitions=2,
            top_logprobs=20,
            timeout=1.0,
            cache_salt=cache_salt_for_arm(ARM),
        )


def test_real_client_does_not_collapse_duplicate_json_keys(monkeypatch):
    raw = b'{"choices":[],"choices":[{"text":"wrong"}]}'

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return raw

    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    status, body = probe.StrictUrllibHttpClient().get_json("http://rank0.test:8000")
    assert status == 200
    assert isinstance(body, probe.StrictJsonFailure)
    assert "duplicate JSON object key" in body.reason


def test_plan_rejects_profile_that_returns_raw_logits(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "load_site", lambda _path: object())
    monkeypatch.setattr(
        probe.attribution.exl3,
        "load_profile",
        lambda _path: probe.attribution.exl3.Profile(
            {"extra_vllm_args": ["--logprobs-mode", "raw_logits"]}
        ),
    )
    monkeypatch.setattr(
        probe.attribution, "live_arm_revalidation_actions", lambda *_args, **_kwargs: []
    )
    assert probe.main(cli_args(tmp_path)) == probe.EXIT_CONFIG_ERROR


def test_raw_logprob_contract_rejects_positive_logit_like_values():
    http = FakeHttp()
    logits = {
        "20": {"logprob": 3.0, "rank": 1},
        "21": {"logprob": 2.0, "rank": 2},
    }
    http.teacher_values = iter((logits, logits))
    with pytest.raises(probe.RequestFailure, match="raw-logprob contract"):
        probe.run_probe(
            http,
            base_url="http://rank0.test:8000",
            model=MODEL,
            prompt="prompt",
            case_id="focused-case",
            seed=20260809,
            discovery_repetitions=2,
            discovery_max_tokens=3,
            focus_generated_index=1,
            window_before=0,
            window_after=0,
            teacher_forced_repetitions=2,
            top_logprobs=20,
            timeout=1.0,
            cache_salt=cache_salt_for_arm(ARM),
        )


def test_stable_teacher_forced_distribution_points_away_from_forward_ranking():
    http = FakeHttp()
    stable = {
        "20": {"logprob": -1.0, "rank": 1},
        "21": {"logprob": -1.25, "rank": 2},
    }
    http.teacher_values = iter((stable, stable))
    result = probe.run_probe(
        http,
        base_url="http://rank0.test:8000",
        model=MODEL,
        prompt="prompt",
        case_id="focused-case",
        seed=20260809,
        discovery_repetitions=2,
        discovery_max_tokens=3,
        focus_generated_index=1,
        window_before=0,
        window_after=0,
        teacher_forced_repetitions=2,
        top_logprobs=20,
        timeout=1.0,
        cache_salt=cache_salt_for_arm(ARM),
    )
    assert result["diagnostic_classification"] == (
        "autoregressive-divergence-without-teacher-forced-top1-flip"
    )
    assert result["teacher_forced_positions"][0]["summary"]["classification"] == (
        "teacher-forced-top1-and-returned-values-stable"
    )
