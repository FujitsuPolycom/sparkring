import json
from pathlib import Path


RECEIPT = (
    Path(__file__).parent
    / "receipts/glm53-flash/adaptive-mtp-vs-dflash7-20260829/diagnostic.json"
)
RECORD = (
    Path(__file__).parent
    / "records/glm53-flash/adaptive-mtp-vs-dflash7-decode-diagnostic-20260829.md"
)


def test_glm53_decode_log_diagnostic_preserves_bounded_claims() -> None:
    document = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert document["schema"] == "sparkring-glm53-decode-log-diagnostic/v1"
    assert document["status"] == "research-only"
    adaptive = document["adaptive_mtp"]
    observations = adaptive["observations"]
    assert adaptive["log_interval_utc"] == {
        "start": observations[0]["timestamp"],
        "end": observations[-1]["timestamp"],
    }
    assert {item["depth"] for item in observations} == {2, 3, 4}
    assert min(item["draft_acceptance_percent"] for item in observations) == 48.9
    assert max(item["draft_acceptance_percent"] for item in observations) == 81.2
    assert min(item["generation_tokens_per_second"] for item in observations) == 24.5
    assert max(item["generation_tokens_per_second"] for item in observations) == 38.7
    assert min(document["dflash7"]["generation_tokens_per_second"]) == 40.6
    assert max(document["dflash7"]["generation_tokens_per_second"]) == 136.0
    post_switch = document["dflash7"]["post_switch_validation"]
    post_switch_observations = post_switch["observations"]
    assert post_switch["log_interval_utc"] == {
        "start": post_switch_observations[0]["timestamp"],
        "end": post_switch_observations[-1]["timestamp"],
    }
    assert min(
        item["generation_tokens_per_second"] for item in post_switch_observations
    ) == 48.8
    assert max(
        item["generation_tokens_per_second"] for item in post_switch_observations
    ) == 74.5
    assert min(
        item["draft_acceptance_percent"] for item in post_switch_observations
    ) == 47.6
    assert max(
        item["draft_acceptance_percent"] for item in post_switch_observations
    ) == 67.0
    assert post_switch["sparkcache_events_in_interval"] == 0
    assert adaptive["sparkcache_events_in_interval"] == 0
    assert document["dflash7"]["sparkcache_events_in_interval"] == 0
    assert document["prompt_receipt_present"] is False
    assert document["output_receipt_present"] is False
    assert (
        document["user_reported_matched_coding_peak"]["receipt_present"] is False
    )


def test_glm53_decode_record_has_required_sections_and_live_receipt_link() -> None:
    text = RECORD.read_text(encoding="utf-8")
    for heading in (
        "## Conditions",
        "## Measurement",
        "## Result",
        "## Conclusion",
        "## Limitations",
        "## Provenance",
    ):
        assert heading in text
    linked_receipt = (
        RECORD.parent
        / "../../receipts/glm53-flash/adaptive-mtp-vs-dflash7-20260829/diagnostic.json"
    ).resolve()
    assert linked_receipt == RECEIPT.resolve()
    assert linked_receipt.is_file()
