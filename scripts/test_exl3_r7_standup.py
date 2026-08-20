"""Tests for the EXL3 R7 stand-up chain: schema, semantic delta, rollback,
placeholder rejection, dry-run behavior, and docs links."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_exl3_r7_candidate as gen  # noqa: E402
import generate_exl3_r7_stock_dcp4 as stock_gen  # noqa: E402
import prepare_exl3_r7_mtp2 as mtp2  # noqa: E402
import prepare_exl3_r7_mtp3 as mtp3  # noqa: E402
import prepare_exl3_r7_mtp3_kv925 as kv925  # noqa: E402
import prepare_exl3_r7_mtp4 as mtp4  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _template() -> dict:
    return json.loads(gen.TEMPLATE_PATH.read_text(encoding="utf-8"))


def _pins() -> dict:
    return json.loads(gen.PINS_PATH.read_text(encoding="utf-8"))


def _recipe() -> dict:
    return json.loads(gen.RECIPE_PATH.read_text(encoding="utf-8"))


def _stock_profile() -> dict:
    return stock_gen.derive_stock_profile(_template(), _pins(), _recipe())


def _mtp2_profile() -> dict:
    return mtp2.derive_candidate(_stock_profile())


def _mtp3_profile() -> dict:
    return mtp3.derive_candidate(_stock_profile(), _mtp2_profile())


def _mtp3_kv925_profile() -> dict:
    profile = _mtp3_profile()
    # The generator output already has --max-cudagraph-capture-size 32
    return profile


def _mtp4_profile() -> dict:
    return mtp4.derive_candidate(_mtp3_kv925_profile())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolved_standup_inputs(tmp_path: Path) -> tuple[Path, Path]:
    image = "sparkring/glm52-exl3-r7-3.5bpw:test"
    image_id = "sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513"
    site_text = (ROOT / "scripts/config/exl3-r7-site.example.yaml").read_text(
        encoding="utf-8"
    )
    site_text = site_text.replace(
        "sparkring/glm52-exl3-r7-3.5bpw:REPLACE", image
    ).replace("sha256:" + "1" * 64, image_id).replace(" - REPLACE.", ".")
    site = tmp_path / "site.yaml"
    site.write_text(site_text, encoding="utf-8")

    template_document = _template()
    template_document.update(
        {
            "image": image,
            "image_id": image_id,
            "model_host_path": "/models/glm52-exl3-r7-3.5bpw",
            "jit_cache_host_path": "/var/lib/sparkring/jit-cache",
        }
    )
    template = tmp_path / "candidate.json"
    template.write_text(json.dumps(template_document), encoding="utf-8")
    return site, template


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_public_pins_have_correct_schema() -> None:
    pins = _pins()
    assert pins["schema"] == "sparkring-exl3-r7-pins/v1"
    assert pins["schema_version"] == 1


def test_pins_match_recipe_model_identity() -> None:
    pins = _pins()
    recipe = _recipe()
    assert pins["model"]["repository"] == recipe["model"]["repository"]
    assert pins["model"]["revision"] == recipe["model"]["revision"]
    assert pins["model"]["config_sha256"] == recipe["model"]["config_sha256"]
    assert pins["model"]["index_sha256"] == recipe["model"]["index_sha256"]
    assert pins["model"]["shard_count"] == recipe["model"]["shard_count"]
    assert pins["model"]["weight_count"] == recipe["model"]["weight_count"]
    assert pins["model"]["index_total_size"] == recipe["model"]["index_total_size"]


def test_stock_profile_has_correct_schema() -> None:
    stock = _stock_profile()
    assert stock["schema"] == "sparkring-runtime-profile/v1"
    assert stock["model_family"] == "exl3-r7"


def test_recipe_records_candidate_maturity() -> None:
    recipe = _recipe()
    assert recipe["maturity"] == "accepted"
    assert recipe["default"] is False
    assert recipe["publication"]["operator_default"] is True
    assert recipe["serving"]["mtp_policy"] == "fixed-4"
    assert recipe["serving"]["kv_cache_bytes_per_rank"] == 9_250_000_000


# ---------------------------------------------------------------------------
# Exact semantic profile delta tests
# ---------------------------------------------------------------------------

def test_stock_to_mtp2_changes_only_speculation() -> None:
    stock = _stock_profile()
    candidate = _mtp2_profile()
    mtp2.validate_candidate(stock, candidate)
    # The only changes are: profile_id suffix, MTP env, speculative config,
    # shared capture stream overlay, and labels.
    assert candidate["profile_id"].endswith("-fixed-mtp2")
    assert candidate["environment"]["VLLM_SPARK_MTP_TOKENS"] == "2"
    assert candidate["environment"]["VLLM_SPARK_MAX_QUERY_ROWS"] == "24"


def test_mtp2_to_mtp3_changes_only_depth_and_query_rows() -> None:
    stock = _stock_profile()
    control = _mtp2_profile()
    candidate = _mtp3_profile()
    mtp3.validate_candidate(stock, control, candidate)
    assert candidate["profile_id"].endswith("-fixed-mtp3")
    assert candidate["environment"]["VLLM_SPARK_MTP_TOKENS"] == "3"
    assert candidate["environment"]["VLLM_SPARK_MAX_QUERY_ROWS"] == "32"


def test_mtp3_to_kv925_changes_only_site_kv_bytes() -> None:
    source_site = (
        "serving:\n"
        "  tensor_parallel_size: 4\n"
        "  decode_context_parallel_size: 4\n"
        '  mtp_mode: "static"\n'
        "  mtp_tokens: 3\n"
        "  max_model_len: 65536\n"
        "  kv_cache_bytes_per_rank: 9000000000\n"
        "  max_num_seqs: 8\n"
    )
    candidate_site = kv925.derive_candidate_site(source_site)
    assert candidate_site == source_site.replace("9000000000", "9250000000")
    assert kv925.EXPECTED_CAPACITY_TOKENS == 675_840


def test_mtp3_kv925_to_mtp4_changes_only_depth_and_graph_coverage() -> None:
    source = _mtp3_kv925_profile()
    candidate = _mtp4_profile()
    mtp4.validate_candidate(source, candidate)
    assert candidate["profile_id"].endswith("-fixed-mtp4")
    assert candidate["environment"]["VLLM_SPARK_MTP_TOKENS"] == "4"
    assert candidate["environment"]["VLLM_SPARK_MAX_QUERY_ROWS"] == "40"
    args = candidate["extra_vullm_args"] if "extra_vullm_args" in candidate else candidate["extra_vllm_args"]
    comp = json.loads(args[args.index("--compilation-config") + 1])
    assert comp["cudagraph_capture_sizes"] == list(range(1, 41))
    assert args[args.index("--max-cudagraph-capture-size") + 1] == "40"


def test_mtp4_preserves_kv9_25_and_transport_contract() -> None:
    source = _mtp3_kv925_profile()
    candidate = _mtp4_profile()
    # KV cache dtype, DCP, transport, and online K6 must not change
    assert candidate["environment"]["ONLINE_QUANT"] == source["environment"]["ONLINE_QUANT"]
    assert candidate["environment"]["VLLM_EXL3_ONLINE_TRELLIS_BITS"] == source["environment"]["VLLM_EXL3_ONLINE_TRELLIS_BITS"]
    args_c = candidate["extra_vllm_args"]
    args_s = source["extra_vllm_args"]
    assert args_c[args_c.index("--kv-cache-dtype") + 1] == args_s[args_s.index("--kv-cache-dtype") + 1]
    assert args_c[args_c.index("--dcp-comm-backend") + 1] == "ag_rs"


# ---------------------------------------------------------------------------
# Rollback identity tests
# ---------------------------------------------------------------------------

def test_mtp4_rollback_is_byte_identical_to_mtp3_kv925(tmp_path: Path) -> None:
    profile_path = tmp_path / "mtp3-kv925.json"
    site_path = tmp_path / "mtp3-kv925.yaml"
    candidate_profile = tmp_path / "mtp4.json"
    candidate_site = tmp_path / "mtp4.yaml"
    rollback_profile = tmp_path / "rollback.json"
    rollback_site = tmp_path / "rollback.yaml"

    source = _mtp3_kv925_profile()
    profile_bytes = (json.dumps(source, indent=2) + "\n").encode()
    profile_path.write_bytes(profile_bytes)
    site_path.write_text(
        "serving:\n"
        "  tensor_parallel_size: 4\n"
        "  decode_context_parallel_size: 4\n"
        '  mtp_mode: "static"\n'
        "  mtp_tokens: 3\n"
        "  max_model_len: 65536\n"
        "  kv_cache_bytes_per_rank: 9250000000\n"
        "  max_num_seqs: 8\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp4.py"),
            "--mtp3-profile", str(profile_path),
            "--mtp3-site", str(site_path),
            "--expected-mtp3-profile-sha256", _sha256(profile_bytes),
            "--expected-mtp3-site-sha256", _sha256(site_path.read_bytes()),
            "--candidate-profile", str(candidate_profile),
            "--candidate-site", str(candidate_site),
            "--rollback-profile", str(rollback_profile),
            "--rollback-site", str(rollback_site),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert rollback_profile.read_bytes() == profile_bytes
    assert rollback_site.read_bytes() == site_path.read_bytes()


def test_kv925_rollback_is_byte_identical_to_mtp3(tmp_path: Path) -> None:
    profile_path = tmp_path / "mtp3.json"
    site_path = tmp_path / "mtp3.yaml"
    candidate_profile = tmp_path / "kv925.json"
    candidate_site = tmp_path / "kv925.yaml"
    rollback_profile = tmp_path / "rollback.json"
    rollback_site = tmp_path / "rollback.yaml"

    source = _mtp3_profile()
    profile_bytes = (json.dumps(source, indent=2) + "\n").encode()
    profile_path.write_bytes(profile_bytes)
    site_path.write_text(
        "serving:\n"
        "  tensor_parallel_size: 4\n"
        "  decode_context_parallel_size: 4\n"
        '  mtp_mode: "static"\n'
        "  mtp_tokens: 3\n"
        "  max_model_len: 65536\n"
        "  kv_cache_bytes_per_rank: 9000000000\n"
        "  max_num_seqs: 8\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_exl3_r7_mtp3_kv925.py"),
            "--qualified-profile", str(profile_path),
            "--qualified-site", str(site_path),
            "--candidate-profile", str(candidate_profile),
            "--candidate-site", str(candidate_site),
            "--rollback-profile", str(rollback_profile),
            "--rollback-site", str(rollback_site),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert rollback_profile.read_bytes() == profile_bytes
    assert rollback_site.read_bytes() == site_path.read_bytes()


# ---------------------------------------------------------------------------
# Placeholder / private-identifier rejection tests
# ---------------------------------------------------------------------------

def test_site_template_uses_only_documentation_addresses() -> None:
    site_path = ROOT / "scripts" / "config" / "exl3-r7-site.example.yaml"
    content = site_path.read_text(encoding="utf-8")
    # Must not contain RFC1918 private addresses
    assert "192.168." not in content
    assert "10.0." not in content
    assert "172.16." not in content
    # Must use RFC 5737 documentation ranges
    assert "192.0.2." in content
    assert "198.51.100." in content
    assert "203.0.113." in content


def test_site_template_uses_placeholder_image() -> None:
    site_path = ROOT / "scripts" / "config" / "exl3-r7-site.example.yaml"
    content = site_path.read_text(encoding="utf-8")
    assert "REPLACE" in content
    assert "sha256:1111111111111111111111111111111111111111111111111111111111111111" in content


def test_candidate_template_uses_placeholder_image() -> None:
    template = _template()
    assert "REPLACE" in template["image"]
    assert template["image_id"] == (
        "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    )


def test_pins_contain_no_private_paths() -> None:
    pins = _pins()
    pins_text = json.dumps(pins)
    assert "/var/tmp/sparkring" not in pins_text
    assert "/opt/sparkring" not in pins_text
    assert "Documents" not in pins_text


def test_model_pin_matches_task_requirement() -> None:
    pins = _pins()
    assert pins["model"]["repository"] == (
        "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78"
    )
    assert pins["model"]["revision"] == (
        "9ab9579774cc432df91567a36f6e9e863e0d4c9f"
    )
    assert pins["model"]["config_sha256"] == (
        "fabb73eb513ec64f3a365da396b38de8d55b3930edfb11baeecbf34ecafa6126"
    )
    assert pins["model"]["index_sha256"] == (
        "9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd"
    )


# ---------------------------------------------------------------------------
# Dry-run behavior tests
# ---------------------------------------------------------------------------

def test_standup_dry_run_does_not_write_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "must-not-be-created"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "exl3_r7_standup.py"),
            "plan",
            "--output-dir", str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "DRY-RUN" in result.stdout
    # The receipt is a multi-line indented JSON block at the end of stdout
    json_start = result.stdout.rfind("\n{")
    assert json_start >= 0, "no JSON receipt in stdout"
    receipt = json.loads(result.stdout[json_start + 1:])
    # In dry-run, no profile files should exist
    assert not output_dir.exists()
    assert receipt["dry_run"] is True


def test_standup_execute_writes_files(tmp_path: Path) -> None:
    site, template = _resolved_standup_inputs(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "exl3_r7_standup.py"),
            "plan",
            "--execute",
            "--output-dir", str(tmp_path),
            "--site", str(site),
            "--template", str(template),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "EXECUTE" in result.stdout
    assert (tmp_path / "stock-dcp4-profile.json").exists()
    json_start = result.stdout.rfind("\n{")
    assert json_start >= 0, "no JSON receipt in stdout"
    receipt = json.loads(result.stdout[json_start + 1:])
    assert (tmp_path / "mtp4-kv925-rollback.json").exists()
    assert receipt["dry_run"] is False
    assert "mtp4_profile_sha256" in receipt
    assert "rollback_profile_sha256" in receipt
    assert "rollback_identity" in receipt
    validation = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sparkring_generic_launcher.py"),
            "--site", str(tmp_path / "mtp4-kv925-site.yaml"),
            "--profile", str(tmp_path / "mtp4-kv925-profile.json"),
            "validate",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validation.returncode == 0, validation.stderr


def test_standup_execute_rollback_matches_mtp3_kv925(tmp_path: Path) -> None:
    site, template = _resolved_standup_inputs(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "exl3_r7_standup.py"),
            "plan",
            "--execute",
            "--output-dir", str(tmp_path),
            "--site", str(site),
            "--template", str(template),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    rollback = (tmp_path / "mtp4-kv925-rollback.json").read_bytes()
    mtp3_kv925 = (tmp_path / "mtp3-kv925-profile.json").read_bytes()
    assert rollback == mtp3_kv925


def test_standup_execute_rejects_unresolved_default_inputs(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "exl3_r7_standup.py"),
            "plan",
            "--execute",
            "--output-dir", str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "requires a complete ignored --site" in result.stderr


# ---------------------------------------------------------------------------
# Docs links test
# ---------------------------------------------------------------------------

def test_quickstart_doc_exists() -> None:
    assert (ROOT / "docs" / "EXL3_R7_QUICKSTART.md").exists()


def test_quickstart_doc_links_resolve() -> None:
    import re
    doc = (ROOT / "docs" / "EXL3_R7_QUICKSTART.md").read_text(encoding="utf-8")
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    for match in link_pattern.finditer(doc):
        target = match.group(2)
        if target.startswith("http"):
            continue
        dest = ROOT / "docs" / target
        assert dest.exists(), f"broken link: {target}"


def test_quickstart_doc_separates_operator_acceptance_from_rebuild_maturity() -> None:
    doc = (ROOT / "docs" / "EXL3_R7_QUICKSTART.md").read_text(encoding="utf-8")
    assert "operator-accepted" in doc
    assert "clean-checkout rebuild is an" in doc
    assert "offline-validated candidate" in doc
    assert "Acceptance applies to one four-Spark appliance" in doc
    assert "not transfer to a rebuilt image" in doc
    assert "It is not the repository default" in doc

    assert "LMCache CS512" in doc


def test_quickstart_doc_offers_the_published_image_and_binds_identity() -> None:
    """The page offers the pull and says which identity the profile binds to.

    The published image carries the same runtime filesystem as one built from
    runtime/exl3-r7. Container labels differ between them, so the two report
    different configuration digests, and a reader has to know that the identity
    to record is whichever the image they run reports.
    q40_exact_state_attestation_overlay.py takes a required --image-id and
    embeds it, so the exact-Q40 layer binds to that identity rather than to a
    fixed one, and the page must not describe a pulled image as excluded.
    """

    doc = (ROOT / "docs" / "EXL3_R7_QUICKSTART.md").read_text(encoding="utf-8")
    assert "ghcr.io/fujitsupolycom/gb10-vllm-serving" in doc
    assert "--image-id" in doc
    assert "Every section of this page holds for a" in doc
    assert "not published to a registry" not in doc
    assert "refuses it" not in doc
