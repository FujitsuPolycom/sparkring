import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runtime" / "ports" / "glm52-port-manifest.json"


def test_port_manifest_is_complete_and_pinned() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema"] == "sparkring-glm52-port-manifest/v1"
    assert len(payload["audited_against"]["vllm_commit"]) == 40

    expected = {
        "sm121-sparse-mla",
        "packed-low-bit-mla-kv",
        "hybrid-mxfp4-checkpoint",
        "adaptive-mtp",
    }
    capabilities = payload["capabilities"]
    assert {entry["id"] for entry in capabilities} == expected
    assert [entry["priority"] for entry in capabilities] == [1, 2, 3, 4]

    sources = payload["sources"]
    for source in sources.values():
        assert source["repository"].startswith("https://github.com/")
        assert len(source["commit"]) == 40
        assert source["status"].startswith("public-pinned")

    for capability in capabilities:
        assert capability["reference_delta_targets"]
        assert capability["acceptance"]
        for source_file in capability["public_sources"]:
            assert source_file["source"] in sources
            assert not Path(source_file["path"]).is_absolute()
            assert len(source_file["sha256"]) == 64
            int(source_file["sha256"], 16)


def test_fast_port_excludes_nonessential_reference_features() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    excluded = {entry["area"] for entry in payload["excluded_from_fast_port"]}
    assert "pipeline-parallel speculative-decode fixes" in excluded
    assert "InstantTensor loader modifications" in excluded
