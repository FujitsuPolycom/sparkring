"""The package inside the SparkRing repository: shared resolver, module lookup and defaults.

- RoCE GID indices come from SparkRing's one resolver,
  ``integrations/vllm/spark_roce_gid.py``, which both staging paths ship
  beside the package;
- the vLLM adapter finds its sessions under the full package name only;
- NCCL carries nothing unless the operator opts in (``SIRCL_NCCL`` and
  ``--nccl`` default to ``never``; ``auto`` opts in; ``topology`` is read as
  ``auto``);
- the serve launcher turns SIRCL's four-rank adapter off with its own switch,
  ``SPARK_TP4_ENABLED=0``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sparkring_sircl import roce_gid

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parents[1]
RESOLVER = REPOSITORY / "integrations" / "vllm" / "spark_roce_gid.py"
in_repository = pytest.mark.skipif(not RESOLVER.is_file(), reason="not inside a SparkRing checkout")


@in_repository
def test_gid_resolution_is_the_repositorys_resolver():
    assert roce_gid.source_file() == RESOLVER.resolve()
    shared = sys.modules[roce_gid.MODULE]
    assert roce_gid.resolve_device_gid_index is shared.resolve_device_gid_index
    assert roce_gid.GidResolutionError is shared.GidResolutionError


@in_repository
def test_the_resolver_resolves_a_fabric_address_from_a_gid_table(tmp_path):
    port = tmp_path / "rocep1s0f0" / "ports" / "1"
    entries = {0: ("fe80:0000:0000:0000:0000:0000:0000:0001", "IB/RoCE v1"),
               2: ("0000:0000:0000:0000:0000:ffff:c612:0001", "IB/RoCE v1"),
               3: ("0000:0000:0000:0000:0000:ffff:c612:0001", "RoCE v2")}
    for index, (gid, kind) in entries.items():
        for directory, text in (("gids", gid), ("gid_attrs/types", kind), ("gid_attrs/ndevs", "enp1s0f0np0")):
            (port / directory).mkdir(parents=True, exist_ok=True)
            (port / directory / str(index)).write_text(text + "\n", encoding="utf-8")
    assert roce_gid.resolve_gid_index("rocep1s0f0", "198.18.0.1", root=tmp_path) == 3


@in_repository
def test_both_staging_paths_ship_the_resolver_beside_the_package():
    from sparkring_sircl.ring import plan as ring_plan
    from sparkring_sircl.vllm.serve import staging

    staged = dict(staging.staged_tree().files)
    assert staged["spark_roce_gid.py"] == RESOLVER.read_bytes()
    assert not any(name.startswith("sparkring_sircl/") and name.endswith("roce_gid.py") and name !=
                   "sparkring_sircl/roce_gid.py" for name in staged)
    harness = dict(ring_plan.staged_files())
    assert harness["spark_roce_gid.py"] == RESOLVER.resolve()
    assert all(name.startswith("sparkring_sircl/") for name in harness if name != "spark_roce_gid.py")


def test_the_adapter_finds_sessions_under_the_full_package_name_only():
    from sparkring_sircl.vllm import settings

    assert settings.session_modules({}) == ("sparkring_sircl.oneshot",)
    assert settings.session_modules({"SIRCL_SESSION_MODULE": "custom.sessions"}) == ("custom.sessions",)


def test_nccl_is_off_unless_the_operator_opts_in():
    from sparkring_sircl.vllm import settings
    from sparkring_sircl.vllm.serve import bundle, cli
    from sparkring_sircl.vllm.serve import plan as plan_mod

    assert settings.nccl_mode({}) == "never"
    assert settings.nccl_mode({"SIRCL_NCCL": "auto"}) == settings.nccl_mode({"SIRCL_NCCL": "topology"}) == "auto"
    with pytest.raises(settings.SettingError, match="SIRCL_NCCL must be one of never, auto, topology"):
        settings.nccl_mode({"SIRCL_NCCL": "always"})
    assert plan_mod.DEFAULT_NCCL_MODE == "never"
    assert plan_mod.Options(positions=(0,)).nccl_mode == bundle.BundleOptions(positions=(0,)).nccl_mode == "never"
    parser = cli.parser()
    for command in (["plan", "--site", "s", "--repository", "r"], ["bundle", "--site", "s", "--positions", "0-7"]):
        assert parser.parse_args(command).nccl == "never"
        assert parser.parse_args([*command, "--nccl", "auto"]).nccl == "auto"
        assert parser.parse_args([*command, "--nccl", "topology"]).nccl == "auto"


def test_the_serve_launcher_turns_the_four_rank_adapter_off_with_its_own_switch():
    from sparkring_sircl.vllm.serve import plan as plan_mod

    disabled = plan_mod.DISABLED_TRANSPORTS
    assert disabled["SPARK_TP4_ENABLED"] == "0"
    assert disabled["VLLM_SPARK_TP4_MODE"] == disabled["VLLM_SPARK_TP4_VOCAB_MODE"] == ""
    assert not any(name.startswith("SIRCL_") for name in disabled)
    assert "SPARK_TP4_ENABLED" in plan_mod.REASONS and "SPARK_TP4_ENABLED" in plan_mod.LAUNCHER_KEYS
