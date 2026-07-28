"""Static fail-closed contract for the four-rank DCP graph probe runner."""

from pathlib import Path


TEXT = (
    Path(__file__).with_name("run_tp4_dcp_graph_probe.ps1")
    .read_text(encoding="utf-8")
)


def test_runner_requires_model_down_and_identical_artifacts() -> None:
    assert "requires the model-down memory window" in TEXT
    assert "identical_sha256=true" in TEXT
    assert "artifact SHA-256 values differ across ranks" in TEXT
    assert '[string]$Rank0Target = ($env:SPARKRING_TARGETS -split ",")[0]' in TEXT
    assert '[string]$Rank0ProxyJump = ""' in TEXT
    assert '[string]$Rank1Target = ($env:SPARKRING_TARGETS -split ",")[1]' in TEXT
    assert '[string]$Rank2Target = ($env:SPARKRING_TARGETS -split ",")[2]' in TEXT
    assert '[string]$Rank3Target = ($env:SPARKRING_TARGETS -split ",")[3]' in TEXT
    assert '[string]$Rank3ProxyJump = ""' in TEXT
    assert "missing rank SSH targets" in TEXT
    assert "Get-NodeSshArguments" in TEXT


def test_runner_gates_mixed_family_counts_and_exact_sequence() -> None:
    assert "$capturedNodes = if ($SingleQ -eq 0) { 28L } else { 2L }" in TEXT
    assert (
        "$capturedQueryNodes = if ($SingleQ -eq 0) { 14L } else { 1L }"
        in TEXT
    )
    assert "$capturedCombineNodes = $capturedQueryNodes" in TEXT
    assert "single_q=" in TEXT
    assert "published=$expected" in TEXT
    assert "consumed=$expected" in TEXT
    assert "completed=$expected" in TEXT
    assert "overflow=0" in TEXT


def test_runner_gates_correctness_and_affinity() -> None:
    assert "query_byte_mismatches=0" in TEXT
    assert "output_nonfinite=0" in TEXT
    assert "lse_nonfinite=0" in TEXT
    assert "submit_cpu=$SubmitCpu" in TEXT
    assert "progress_cpu=$ProgressCpu" in TEXT
    assert "poll_policy=$PollPolicy" in TEXT
    assert "dedicated_spin=" in TEXT
    assert "passed=true" in TEXT


def test_runner_exposes_only_attested_poll_policies() -> None:
    assert '[ValidateSet("adaptive-yield", "dedicated-spin")]' in TEXT
    assert '[string]$PollPolicy = "adaptive-yield"' in TEXT
    assert "SPARK_TP4_DCP_GRAPH_POLL_POLICY=$PollPolicy" in TEXT
    assert "[switch]$PhaseTrace" in TEXT
    assert "SPARK_TRANSPORT_TRACE=1" in TEXT
    assert '$phaseLines.Count -ne $expected' in TEXT


def test_runner_supports_fail_closed_single_q_soaks() -> None:
    assert (
        "[ValidateSet(0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40)]"
        in TEXT
    )
    assert "[int]$SingleQ = 0" in TEXT
    assert '"--single-q $SingleQ"' in TEXT


def test_runner_uses_direct_cable_control_addresses() -> None:
    for address in (
        "192.0.2.10",
        "192.0.2.11",
        "192.0.2.20",
        "192.0.2.21",
        "192.0.2.30",
        "192.0.2.31",
        "192.0.2.40",
        "192.0.2.41",
    ):
        assert f'"{address}"' in TEXT
