"""CPU placement on hosts with performance and efficiency cores, from fake sysfs trees."""

from __future__ import annotations

from pathlib import Path

import pytest

from sparkring_sircl import cpus

# A GB10 layout: CPUs 0-4 and 10-14 are Cortex-A725, 5-9 and 15-19 Cortex-X925.
EFFICIENCY = [*range(0, 5), *range(10, 15)]
PERFORMANCE = [*range(5, 10), *range(15, 20)]


def _tree(tmp_path: Path, *, capacity: dict[int, int] | None = None, frequency: dict[int, int] | None = None,
          parts: dict[int, int] | None = None, cpus_online: str = "0-19") -> tuple[Path, Path]:
    root = tmp_path / "cpu"
    root.mkdir(parents=True)
    (root / "online").write_text(cpus_online + "\n")
    for cpu in cpus.parse_cpu_list(cpus_online):
        folder = root / f"cpu{cpu}"
        (folder / "cpufreq").mkdir(parents=True)
        if capacity is not None:
            (folder / "cpu_capacity").write_text(f"{capacity[cpu]}\n")
        if frequency is not None:
            (folder / "cpufreq" / "cpuinfo_max_freq").write_text(f"{frequency[cpu]}\n")
    cpuinfo = tmp_path / "cpuinfo"
    lines = []
    for cpu, part in sorted((parts or {}).items()):
        lines += [f"processor\t: {cpu}", "CPU implementer\t: 0x41", f"CPU part\t: {part:#x}", ""]
    cpuinfo.write_text("\n".join(lines))
    return root, cpuinfo


def test_cpu_lists_round_trip():
    assert cpus.parse_cpu_list("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert cpus.format_cpu_list([11, 10, 8, 3, 2, 1, 0]) == "0-3,8,10-11"
    assert cpus.format_cpu_list([]) == ""


def _check_gb10_placement(placement: cpus.Placement) -> None:
    assert list(placement.performance) == PERFORMANCE
    assert list(placement.main) == PERFORMANCE[:-1] and placement.progress == (19,)
    assert not set(placement.main) & set(placement.progress) and placement.dedicated_progress_core
    assert placement.progress_cpu_list == "19"
    assert placement.to_json()["main_cpus"] == "5-9,15-18"


@pytest.mark.parametrize("source", ["cpu_capacity", "cpuinfo_max_freq", "cpu_part"])
def test_performance_cores_are_found_from_each_source(tmp_path, source):
    fast = {cpu: cpu in PERFORMANCE for cpu in range(20)}
    tree = {
        "cpu_capacity": dict(capacity={cpu: 1024 if fast[cpu] else 560 for cpu in range(20)}),
        "cpuinfo_max_freq": dict(frequency={cpu: 3900000 if fast[cpu] else 2808000 for cpu in range(20)}),
        "cpu_part": dict(parts={cpu: 0xD85 if fast[cpu] else 0xD87 for cpu in range(20)}),
    }[source]
    root, cpuinfo = _tree(tmp_path, **tree)
    online, fastest, found = cpus.classify(root, cpuinfo)
    assert found == source and online == list(range(20)) and fastest == PERFORMANCE
    _check_gb10_placement(cpus.plan("performance", range(20), root=root, cpuinfo=cpuinfo))


def test_one_core_slightly_faster_than_its_siblings(tmp_path):
    # GB10 reports the highest capacity for one X925 core only: the class is all ten X925 cores.
    capacity = {cpu: (1024 if cpu == 19 else 1017) if cpu in PERFORMANCE else 580 for cpu in range(20)}
    root, cpuinfo = _tree(tmp_path, capacity=capacity)
    online, fastest, source = cpus.classify(root, cpuinfo)
    assert source == "cpu_capacity" and fastest == PERFORMANCE
    _check_gb10_placement(cpus.plan("performance", range(20), root=root, cpuinfo=cpuinfo))
    # The part numbers decide when present, whatever the capacities say.
    parts = {cpu: 0xD85 if cpu in PERFORMANCE else 0xD87 for cpu in range(20)}
    root, cpuinfo = _tree(tmp_path / "with-parts", capacity={cpu: 1024 if cpu == 3 else 512 for cpu in range(20)},
                          parts=parts)
    assert cpus.classify(root, cpuinfo)[1:] == (PERFORMANCE, "cpu_part")


def test_uniform_hosts_and_the_none_policy(tmp_path):
    root, cpuinfo = _tree(tmp_path, capacity={cpu: 1024 for cpu in range(20)}, cpus_online="0-19")
    placement = cpus.plan("performance", range(20), root=root, cpuinfo=cpuinfo)
    assert placement.source == "uniform" and list(placement.main) == list(range(19))
    assert placement.progress == (19,)
    unpinned = cpus.plan("none", range(20), root=root, cpuinfo=cpuinfo)
    assert unpinned.progress == () and unpinned.progress_cpu_list is None and list(unpinned.main) == list(range(20))
    with pytest.raises(ValueError, match="CPU policy"):
        cpus.plan("fastest", range(20), root=root, cpuinfo=cpuinfo)


def test_the_progress_thread_never_shares_the_main_cpus(tmp_path):
    root, cpuinfo = _tree(tmp_path, parts={cpu: 0xD85 if cpu in PERFORMANCE else 0xD87 for cpu in range(20)})
    single = cpus.plan("performance", [0, 1, 7], root=root, cpuinfo=cpuinfo)
    assert single.main == (7,) and single.progress == (0, 1) and not single.dedicated_progress_core
    assert single.progress_cpu_list == "0-1"
    alone = cpus.plan("performance", [7], root=root, cpuinfo=cpuinfo)
    assert alone.main == (7,) and alone.progress == () and alone.progress_cpu_list is None
    efficiency_only = cpus.plan("performance", [0, 1, 2], root=root, cpuinfo=cpuinfo)
    assert efficiency_only.main == (0, 1) and efficiency_only.progress == (2,)
    for allowed in ([0, 1, 7], [5, 6], [7], list(range(20))):
        placement = cpus.plan("performance", allowed, root=root, cpuinfo=cpuinfo)
        assert not set(placement.main) & set(placement.progress)
