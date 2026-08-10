from __future__ import annotations

import copy
import json

import pytest

from llm_decode_prompt_manifest import (
    PromptManifestError,
    PromptManifestSession,
    load_summary,
    sha256_json,
)


def workload() -> dict:
    return {
        "harness": {"name": "llm_decode_bench.py", "version": "0.4.31"},
        "model": "GLM-5.2",
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


def payloads(marker: str = "exported", *, max_tokens: int = 2048) -> list[dict]:
    return [
        {
            "model": "GLM-5.2",
            "messages": [
                {"role": "user", "content": f"[BENCH_{marker}_CTX_16384] lane={index}"}
            ],
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
            },
        }
        for index in range(8)
    ]


def descriptor(phase: str = "decode-measured") -> dict:
    return {
        "phase": phase,
        "request_kind": "decode-stream-template",
        "context_tokens": 16384,
        "concurrency": 8,
        "benchmark_mode": "duration",
    }


def create_manifest(tmp_path):
    path = tmp_path / "prompts.json"
    session = PromptManifestSession.export(path, workload())
    session.resolve_call(descriptor(), payloads())
    summary = session.finish()
    return path, summary


def test_export_records_exact_payloads_and_commitments(tmp_path):
    path, summary = create_manifest(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    call = document["content"]["calls"][0]

    assert document["content_sha256"] == sha256_json(document["content"])
    assert call["payloads"][3]["payload"] == payloads()[3]
    assert call["payloads"][3]["payload_sha256"] == sha256_json(payloads()[3])
    assert call["payload_set_sha256"] == sha256_json(
        [entry["payload_sha256"] for entry in call["payloads"]]
    )
    assert summary["consumed_all"] is True
    assert summary["mode"] == "export"
    loaded = load_summary(path)
    assert loaded["content_sha256"] == document["content_sha256"]
    assert loaded["workload"] == workload()
    assert loaded["ordered_call_descriptors"] == [descriptor()]


def test_export_reserves_name_and_refuses_overwrite(tmp_path):
    path, _summary = create_manifest(tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        PromptManifestSession.export(path, workload())
    assert path.read_bytes() == original


def test_import_replays_messages_but_requires_other_request_fields(tmp_path):
    path, exported = create_manifest(tmp_path)
    session = PromptManifestSession.import_(path, workload())
    current = payloads("different-random-run")
    replayed = session.resolve_call(descriptor(), current)
    imported = session.finish()

    assert replayed == payloads("exported")
    assert replayed is not payloads("exported")
    assert imported["content_sha256"] == exported["content_sha256"]
    assert imported["payload_sequence_sha256"] == exported["payload_sequence_sha256"]
    assert imported["consumed_all"] is True


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda values: values[0].update(max_tokens=1024), "non-prompt request settings drifted"),
        (lambda values: values.pop(), "payload count"),
    ],
)
def test_import_fails_closed_on_request_drift(tmp_path, mutator, match):
    path, _summary = create_manifest(tmp_path)
    session = PromptManifestSession.import_(path, workload())
    current = payloads("different-random-run")
    mutator(current)
    with pytest.raises(PromptManifestError, match=match):
        session.resolve_call(descriptor(), current)


def test_import_fails_closed_on_workload_or_call_order_drift(tmp_path):
    path, _summary = create_manifest(tmp_path)
    changed = workload()
    changed["temperature"] = 0.1
    with pytest.raises(PromptManifestError, match="workload"):
        PromptManifestSession.import_(path, changed)

    session = PromptManifestSession.import_(path, workload())
    with pytest.raises(PromptManifestError, match="descriptor"):
        session.resolve_call(descriptor("decode-warmup"), payloads("new"))


def test_import_rejects_corruption_before_returning_payload(tmp_path):
    path, _summary = create_manifest(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["content"]["calls"][0]["payloads"][0]["payload"]["messages"][0][
        "content"
    ] += "corrupt"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PromptManifestError, match="content_sha256"):
        PromptManifestSession.import_(path, workload())


def test_import_rejects_rehashed_payload_with_stale_inner_commitment(tmp_path):
    path, _summary = create_manifest(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["content"]["calls"][0]["payloads"][0]["payload"]["messages"][0][
        "content"
    ] += "corrupt"
    document["content_sha256"] = sha256_json(document["content"])
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PromptManifestError, match="payload_sha256"):
        PromptManifestSession.import_(path, workload())


def test_partial_export_cannot_be_imported(tmp_path):
    path = tmp_path / "partial.json"
    session = PromptManifestSession.export(path, workload())
    session.resolve_call(descriptor(), payloads())
    with pytest.raises(PromptManifestError, match="incomplete"):
        PromptManifestSession.import_(path, workload())


def test_import_must_consume_every_call_and_cannot_consume_extra(tmp_path):
    path = tmp_path / "two-calls.json"
    export = PromptManifestSession.export(path, workload())
    export.resolve_call(descriptor("decode-warmup"), payloads("warmup"))
    export.resolve_call(descriptor(), payloads())
    export.finish()

    incomplete = PromptManifestSession.import_(path, workload())
    incomplete.resolve_call(descriptor("decode-warmup"), payloads("fresh-warmup"))
    with pytest.raises(PromptManifestError, match="only 1 of 2"):
        incomplete.finish()

    complete = PromptManifestSession.import_(path, workload())
    complete.resolve_call(descriptor("decode-warmup"), payloads("fresh-warmup"))
    complete.resolve_call(descriptor(), payloads("fresh"))
    with pytest.raises(PromptManifestError, match="extra unsealed call"):
        complete.resolve_call(descriptor(), payloads("extra"))


def test_returned_replay_is_detached_from_manifest_state(tmp_path):
    path, _summary = create_manifest(tmp_path)
    session = PromptManifestSession.import_(path, workload())
    replay = session.resolve_call(descriptor(), payloads("fresh"))
    manifest_before = copy.deepcopy(session.document)
    replay[0]["messages"][0]["content"] = "mutated by caller"
    assert session.document == manifest_before


def test_import_and_summary_reject_duplicate_keys(tmp_path):
    path, _summary = create_manifest(tmp_path)
    rendered = path.read_text(encoding="utf-8")
    path.write_text(
        rendered.replace(
            '"schema": "sparkring-llm-decode-prompt-manifest/v1"',
            '"schema": "wrong", "schema": "sparkring-llm-decode-prompt-manifest/v1"',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(PromptManifestError, match="duplicate JSON object key"):
        PromptManifestSession.import_(path, workload())
    with pytest.raises(PromptManifestError, match="duplicate JSON object key"):
        load_summary(path)


def test_import_rejects_nonfinite_json_constant(tmp_path):
    path, _summary = create_manifest(tmp_path)
    rendered = path.read_text(encoding="utf-8")
    path.write_text(
        rendered.replace('"complete": true', '"forged": NaN, "complete": true', 1),
        encoding="utf-8",
    )
    with pytest.raises(PromptManifestError, match="non-finite JSON number"):
        PromptManifestSession.import_(path, workload())


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda document: document["content"]["workload"].update(
                operator_path="C:/Users/private/prompts.json"
            ),
            "unexpected/private",
        ),
        (
            lambda document: document["content"]["workload"].update(
                model="/mnt/private/model"
            ),
            "served-model identifier",
        ),
        (
            lambda document: document["content"]["calls"][0]["descriptor"].update(
                host="10.0.0.42"
            ),
            "unexpected/private",
        ),
    ],
)
def test_public_summary_rejects_private_or_path_fields(tmp_path, mutator, match):
    path, _summary = create_manifest(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    mutator(document)
    document["content_sha256"] = sha256_json(document["content"])
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PromptManifestError, match=match):
        load_summary(path)


def test_public_summary_binds_payload_settings_to_workload(tmp_path):
    path = tmp_path / "mismatched-settings.json"
    session = PromptManifestSession.export(path, workload())
    session.resolve_call(descriptor(), payloads(max_tokens=1024))
    session.finish()
    with pytest.raises(PromptManifestError, match="do not match the sealed workload"):
        load_summary(path)
