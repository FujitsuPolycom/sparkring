"""Offline tests for single-Spark host diagnosis."""

from scripts.spark_doctor import CommandResult, run_host_checks


class HealthyRunner:
    def run(self, argv):
        command = tuple(argv)
        if command[:2] == ("nvidia-smi", "-L"):
            return CommandResult(0, "GPU 0: NVIDIA GB10\n")
        if command == ("ibdev2netdev",):
            return CommandResult(
                0,
                "rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)\n"
                "rocep1s0f1 port 1 ==> enp1s0f1np1 (Down)\n",
            )
        if command[:2] == ("sh", "-c") and "lspci" in command[2]:
            return CommandResult(0, "Mellanox Technologies ConnectX-7\n")
        if command[:2] == ("systemctl", "is-enabled"):
            return CommandResult(0, "enabled\n")
        return CommandResult(0, "ok\n")


def test_telemetry_is_reported_without_overriding_consent_policy():
    checks = run_host_checks(HealthyRunner())

    assert all(check.status.value == "pass" for check in checks)
    telemetry = next(
        check for check in checks if check.check_id == "HOST.NVIDIA_TELEMETRY"
    )
    assert telemetry.evidence == "is-enabled=enabled"


def test_telemetry_can_be_required_disabled():
    checks = run_host_checks(
        HealthyRunner(), require_telemetry_disabled=True
    )

    telemetry = next(
        check for check in checks if check.check_id == "HOST.NVIDIA_TELEMETRY"
    )
    assert telemetry.status.value == "fail"
