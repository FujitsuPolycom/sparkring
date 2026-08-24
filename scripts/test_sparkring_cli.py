"""Offline tests for the top-level SparkRing operator command."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from scripts import sparkring


def test_doctor_forwards_cluster_and_remaining_flags():
    with mock.patch.object(sparkring.ring_doctor, "main", return_value=0) as doctor:
        result = sparkring.main(
            ["doctor", "--cluster", "/tmp/cluster.yaml", "--verify", "--json"]
        )

    assert result == 0
    doctor.assert_called_once_with(
        ["--cluster", str(Path("/tmp/cluster.yaml")), "--verify", "--json"]
    )


def test_cluster_init_prints_plan_before_initialising(capsys):
    fake = mock.Mock()
    fake.summary_lines.return_value = ["cluster: test"]
    with mock.patch.object(sparkring, "initialise_cluster", return_value=fake) as init:
        result = sparkring.main(
            [
                "cluster",
                "init",
                "--size",
                "4",
                "--head",
                "user@192.0.2.10",
                "--node",
                "user@192.0.2.11",
                "--node",
                "user@192.0.2.12",
                "--node",
                "user@192.0.2.13",
                "--skip-enroll",
                "--yes",
                "--output",
                "/tmp/cluster.yaml",
            ]
        )

    assert result == 0
    assert "SparkRing cluster initialization plan" in capsys.readouterr().out
    assert init.call_args.kwargs["size"] == 4
    assert init.call_args.kwargs["enroll"] is False


def test_cluster_init_rejects_wrong_worker_count(capsys):
    with mock.patch("builtins.input", return_value=""):
        result = sparkring.main(
            [
                "cluster",
                "init",
                "--size",
                "4",
                "--head",
                "user@192.0.2.10",
                "--node",
                "user@192.0.2.11",
                "--node",
                "user@192.0.2.12",
                "--skip-enroll",
                "--yes",
            ]
        )

    assert result == 2
    assert "username@IPv4" in capsys.readouterr().err


def test_cluster_configure_is_plan_only_without_apply(tmp_path, capsys):
    path = tmp_path / "cluster.yaml"
    with mock.patch.object(sparkring, "load_cluster") as load:
        cluster = mock.Mock()
        cluster.ranks = [mock.Mock(id=0, ssh_target="user@192.0.2.10")]
        load.return_value = cluster
        with mock.patch.object(
            sparkring, "render_rank_netplan", return_value="network: {}\n"
        ):
            with mock.patch.object(sparkring, "apply_fabric_network") as apply:
                result = sparkring.main(
                    ["cluster", "configure", "--cluster", str(path)]
                )

    assert result == 0
    assert "Plan only; no node was changed" in capsys.readouterr().out
    apply.assert_not_called()


def test_host_check_forwards_privacy_policy_flag():
    with mock.patch.object(sparkring.spark_doctor, "main", return_value=0) as host:
        result = sparkring.main(
            ["host", "check", "--require-telemetry-disabled", "--json"]
        )

    assert result == 0
    host.assert_called_once_with(["--json", "--require-telemetry-disabled"])
