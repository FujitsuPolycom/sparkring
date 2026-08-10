from __future__ import annotations

import copy
import hashlib
import json

import pytest

from compare_sealed_c8_abba import EvidenceError, compare_abba, main
from llm_decode_prompt_manifest import PromptManifestSession, load_summary


MODEL = "glm-5.2-exl3-tr3-3.25bpw"


def workload() -> dict:
    return {
        "harness": {"name": "llm_decode_bench.py", "version": "0.4.31"},
        "model": MODEL,
        "contexts": [16384],
        "concurrencies": [8],
        "duration_seconds": 25,
        "max_tokens": 2048,
        "temperature": 0.0,
        "reasoning_effort": None,
        "ignore_eos": True,
        "unique_context_percent": 100.0,
        "decode_warmup_seconds": 3,
        "cell_warmup_timeout_seconds": 0,
        "dcp_size": 4,
        "token_targeting": "estimate",
    }


def ordered_call_descriptors() -> list[dict]:
    return [
        {
            "phase": "decode-warmup",
            "request_kind": "prefix-scout",
            "context_tokens": 16384,
            "concurrency": 1,
            "benchmark_mode": "request-count",
        },
        {
            "phase": "decode-warmup",
            "request_kind": "decode-stream-template",
            "context_tokens": 16384,
            "concurrency": 1,
            "benchmark_mode": "duration",
        },
        {
            "phase": "decode-measured",
            "request_kind": "prefix-scout",
            "context_tokens": 16384,
            "concurrency": 1,
            "benchmark_mode": "request-count",
        },
        {
            "phase": "decode-measured",
            "request_kind": "decode-stream-template",
            "context_tokens": 16384,
            "concurrency": 8,
            "benchmark_mode": "duration",
        },
    ]


def create_manifest(tmp_path, *, workload_override: dict | None = None):
    path = tmp_path / "sealed-prompts.json"
    session = PromptManifestSession.export(path, workload_override or workload())
    for sequence, descriptor in enumerate(ordered_call_descriptors()):
        payloads = [
            {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": f"sealed call {sequence} stream {stream}",
                    }
                ],
                "stream": True,
                "max_tokens": 1 if descriptor["request_kind"] == "prefix-scout" else 2048,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream_options": (
                    {"include_usage": True}
                    if descriptor["request_kind"] == "prefix-scout"
                    else {"include_usage": True, "continuous_usage_stats": True}
                ),
            }
            for stream in range(descriptor["concurrency"])
        ]
        session.resolve_call(descriptor, payloads)
    session.finish()
    return path, load_summary(path)


def arm(tps: float = 70.0) -> dict:
    output_tokens = int(tps * 25)
    return {
        "metadata": {
            "version": "0.4.31",
            "engine": "vllm",
            "model": MODEL,
            "decode_mode": "duration",
            "duration_per_test": 25,
            "max_tokens": 2048,
            "temperature": 0.0,
            "reasoning_effort": None,
            "ignore_eos": True,
            "dcp_size": 4,
            "unique_context_percent": 100.0,
            "shared_context_percent": 0.0,
            "decode_warmup_seconds": 3,
            "cell_warmup_timeout_seconds": 0,
            "context_lengths": [16384],
            "concurrency_levels": [8],
            "skip_prefill": True,
            "token_targeting": "estimate",
            "request_count": 0,
            "run_burst": False,
            "prefill_only": False,
            "standalone_prefill": False,
            "prompt_manifest": {
                "schema": "sparkring-llm-decode-prompt-manifest-summary/v1",
                "manifest_schema": "sparkring-llm-decode-prompt-manifest/v1",
                "mode": "import",
                "content_sha256": "0" * 64,
                "payload_sequence_sha256": "0" * 64,
                "call_count": 4,
                "consumed_call_count": 4,
                "consumed_all": True,
            },
        },
        "prefill": {},
        "results": [
            {
                "context_tokens": 16384,
                "concurrency": 8,
                "benchmark_mode": "duration",
                "aggregate_source": "openai_continuous_usage",
                "client_output_tokens": output_tokens,
                "total_tokens": output_tokens,
                "aggregate_tps": tps,
                "measurement_seconds": 25.0,
                "num_errors": 0,
                "effective_concurrency": 8,
                "underfilled": False,
                "warmup_timed_out": False,
                "capacity_limited": False,
            }
        ],
        "burst_results": [],
    }


def evidence_bundle(tmp_path, values=(70, 77, 75, 72)):
    manifest_path, summary = create_manifest(tmp_path)
    documents = [arm(value) for value in values]
    for document in documents:
        document["metadata"]["prompt_manifest"].update(
            content_sha256=summary["content_sha256"],
            payload_sequence_sha256=summary["payload_sequence_sha256"],
            call_count=summary["call_count"],
            consumed_call_count=summary["call_count"],
        )
    return manifest_path, summary, documents


def compare(tmp_path, documents: list[dict], summary: dict | None = None) -> dict:
    if summary is None:
        _manifest_path, summary, generated = evidence_bundle(tmp_path)
        for document, source in zip(documents, generated):
            document["metadata"]["prompt_manifest"] = copy.deepcopy(
                source["metadata"]["prompt_manifest"]
            )
    return compare_abba(
        documents,
        [str(index) * 64 for index in range(1, 5)],
        manifest_summary=summary,
        manifest_artifact_sha256="f" * 64,
    )


def test_sealed_abba_comparison_requires_and_reports_same_exact_payloads(tmp_path):
    _path, summary, documents = evidence_bundle(tmp_path)
    report = compare(tmp_path, documents, summary)

    assert report["passed"] is True
    assert report["order"] == ["A-open", "B-first", "B-second", "A-close"]
    assert report["exact_prompt_replay"]["workload"] == workload()
    assert report["exact_prompt_replay"]["ordered_call_descriptors"] == (
        ordered_call_descriptors()
    )
    assert report["comparison"]["a_mean_aggregate_tps"] == 71
    assert report["comparison"]["b_mean_aggregate_tps"] == 76
    assert report["comparison"]["b_vs_a_percent"] == pytest.approx(7.0422535211)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value["metadata"]["prompt_manifest"].update(mode="export"), "mode"),
        (lambda value: value["metadata"]["prompt_manifest"].update(consumed_all=False), "consumed_all"),
        (lambda value: value["metadata"]["prompt_manifest"].update(consumed_call_count=3), "completely consumed"),
        (lambda value: value["metadata"]["prompt_manifest"].update(payload_sequence_sha256="c" * 64), "commitment differs"),
        (lambda value: value["metadata"]["prompt_manifest"].update(private_path="C:/private"), "private/unknown"),
        (lambda value: value["metadata"].update(temperature=0.1), "temperature"),
        (lambda value: value["metadata"].update(concurrency_levels=[1, 8]), "concurrency_levels"),
        (lambda value: value["results"][0].update(effective_concurrency=7), "sustain"),
        (lambda value: value["results"][0].update(underfilled=True), "readiness"),
        (lambda value: value["results"].append(copy.deepcopy(value["results"][0])), "exactly one"),
        (lambda value: value["metadata"].update(version="0.4.32"), "version"),
        (lambda value: value["metadata"].update(engine="sglang"), "engine"),
        (lambda value: value["metadata"].update(model="/mnt/private/model"), "model"),
        (lambda value: value["results"][0].update(aggregate_source="prometheus_fallback"), "aggregate_source"),
        (lambda value: value["results"][0].update(client_output_tokens=0, total_tokens=0), "client_output_tokens"),
        (lambda value: value["results"][0].update(aggregate_tps=999), "aggregate_tps"),
        (lambda value: value["results"][0].update(measurement_seconds=20), "25s window"),
    ],
)
def test_sealed_abba_fails_closed(tmp_path, mutation, match):
    _path, summary, documents = evidence_bundle(tmp_path)
    mutation(documents[2])
    with pytest.raises(EvidenceError, match=match):
        compare(tmp_path, documents, summary)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda summary: summary["workload"].update(model="other-model"), "authoritative"),
        (lambda summary: summary["workload"].update(duration_seconds=20), "duration_seconds"),
        (lambda summary: summary["ordered_call_descriptors"].reverse(), "ordered call"),
        (lambda summary: summary["workload"]["harness"].update(version="0.4.32"), "harness"),
    ],
)
def test_manifest_semantics_cannot_be_replaced_by_matching_commitments(
    tmp_path, mutation, match
):
    _path, summary, documents = evidence_bundle(tmp_path)
    mutation(summary)
    with pytest.raises(EvidenceError, match=match):
        compare(tmp_path, documents, summary)


def test_sealed_abba_rejects_embedded_commitment_not_bound_to_manifest(tmp_path):
    _path, summary, documents = evidence_bundle(tmp_path)
    summary["content_sha256"] = "c" * 64
    with pytest.raises(EvidenceError, match="supplied sealed manifest"):
        compare(tmp_path, documents, summary)


def test_cli_writes_exclusive_report_without_source_paths(tmp_path):
    manifest_path, _summary, documents = evidence_bundle(tmp_path)
    paths = []
    for index, document in enumerate(documents):
        document["metadata"]["server"] = "10.0.0.42:8000"
        document["metadata"]["operator_output_path"] = "C:/Users/private/result.json"
        path = tmp_path / f"private-arm-{index}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "report.json"
    argv = [
        "--a-open", str(paths[0]),
        "--b-first", str(paths[1]),
        "--b-second", str(paths[2]),
        "--a-close", str(paths[3]),
        "--prompt-manifest", str(manifest_path),
        "--output", str(output),
    ]
    assert main(argv) == 0
    rendered = output.read_text(encoding="utf-8")
    report = json.loads(rendered)
    assert "private-arm" not in rendered
    assert "10.0.0.42" not in rendered
    assert "C:/Users/private" not in rendered
    assert report["arms"][0]["artifact_sha256"] == hashlib.sha256(
        paths[0].read_bytes()
    ).hexdigest()
    assert report["exact_prompt_replay"]["manifest_artifact_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert main(argv) == 2


@pytest.mark.parametrize("forged", ["NaN", "Infinity"])
def test_cli_rejects_nonfinite_benchmark_evidence(tmp_path, forged):
    path = tmp_path / "arm.json"
    path.write_text('{"metadata": {"forged": ' + forged + "}}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    argv = [
        "--a-open", str(path),
        "--b-first", str(path),
        "--b-second", str(path),
        "--a-close", str(path),
        "--prompt-manifest", str(manifest_path),
    ]
    assert main(argv) == 2


def test_cli_rejects_duplicate_benchmark_evidence_keys(tmp_path):
    path = tmp_path / "arm.json"
    path.write_text('{"metadata": {}, "metadata": {}}', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    argv = [
        "--a-open", str(path),
        "--b-first", str(path),
        "--b-second", str(path),
        "--a-close", str(path),
        "--prompt-manifest", str(manifest_path),
    ]
    assert main(argv) == 2
