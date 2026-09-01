from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SUMMARY = (
    ROOT
    / "performance"
    / "receipts"
    / "glm53-flash"
    / "dcp4-24g-default-20260901"
    / "summary.json"
)
RECORD = Path(__file__).with_name("dcp4-24g-default-20260901.md")


def test_dcp4_default_summary_binds_runtime_and_results() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    conditions = summary["conditions"]

    assert summary["status"] == "research-only"
    assert conditions["decode_context_parallel_size"] == 4
    assert conditions["kv_cache_memory_bytes_per_rank"] == 24 * 1024**3
    assert conditions["kv_capacity_tokens"] == 4_321_618
    assert conditions["max_num_batched_tokens"] == 8192
    assert conditions["prefill_schedule_interval"] == 8
    assert conditions["sparkcache_enabled"] is True
    assert summary["prefill_c1"]["131072"]["tokens_per_second"] == 2565.0
    assert summary["aggregate_decode_tokens_per_second"]["8192"]["16"] == 170.22
    assert "16" not in summary["aggregate_decode_tokens_per_second"]["32768"]
    assert summary["coding_peak"]["median_tokens_per_second"] == 71.67


def test_dcp4_default_public_record_is_sanitized() -> None:
    text = RECORD.read_text(encoding="utf-8") + SUMMARY.read_text(encoding="utf-8")

    assert re.search(r"(?i)\b[A-Z]:\\", text) is None
    assert re.search(r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)", text) is None
    assert "DESKTOP-" not in text
    assert "api_key" not in text.casefold()
