"""The page-cache drop and memory settle before a deployment starts; offline."""
from types import SimpleNamespace

import pytest

from runtime.host import install_workflow, memory_settle

# Taken before the host tests' fixture replaces it (runtime/host/conftest.py).
REAL_SETTLE_MEMORY = install_workflow.settle_memory

GIB = 2 ** 20  # KiB


class Sparks:
    """Spark hosts whose MemAvailable readings follow a list per host; the last reading repeats."""

    def __init__(self, readings, total=121 * GIB):
        self.readings = {host: list(values) for host, values in readings.items()}
        self.total, self.calls = total, []

    def run(self, host, argv):
        self.calls.append((host, argv))
        if argv == memory_settle.DROP_CACHES:
            return ""
        assert argv == memory_settle.MEMINFO
        values = self.readings[host]
        available = values.pop(0) if len(values) > 1 else values[0]
        return f"{self.total}\n{available}\n"


def clock():
    state = {"now": 0.0}

    def sleep(seconds):
        state["now"] += seconds
    return sleep, lambda: state["now"]


def test_the_requested_memory_comes_from_the_vllm_command():
    assert memory_settle.requested(["serve", "/m", "--gpu-memory-utilization", "0.91",
                                    "--kv-cache-memory-bytes=46170898432"]) == (0.91, 46170898432)
    assert memory_settle.requested(["serve", "/m"]) == (None, None)


def test_caches_drop_once_per_spark_and_the_start_waits_for_settled_memory():
    sparks = Sparks({"operator@192.0.2.10": [60 * GIB, 100 * GIB, 115 * GIB, 115 * GIB],
                     "operator@192.0.2.11": [116 * GIB]})
    sleep, now = clock()
    said = []
    rows = memory_settle.settle(["operator@192.0.2.10", "operator@192.0.2.11", "operator@192.0.2.10"], 0.91,
                                run=sparks.run, sleep=sleep, now=now, say=said.append)
    drops = [host for host, argv in sparks.calls if argv == memory_settle.DROP_CACHES]
    assert drops == ["operator@192.0.2.10", "operator@192.0.2.11"]
    assert sparks.calls[:2] == [(host, memory_settle.DROP_CACHES) for host in drops]
    assert [row["available_gib"] for row in rows] == [115.0, 116.0] and all(row["settled"] for row in rows)
    assert rows[0]["asks_gib"] == round(0.91 * 121, 2) and now() == 15
    assert said == ["Memory before the start (settled; vLLM asks for 0.91 of the total): operator@192.0.2.10 "
                    "115.00 GiB available, operator@192.0.2.11 116.00 GiB available."]


def test_a_spark_below_vllms_share_refuses_the_start_with_its_numbers():
    sparks = Sparks({"operator@192.0.2.10": [116 * GIB], "operator@192.0.2.11": [80 * GIB]})
    sleep, now = clock()
    with pytest.raises(memory_settle.MemoryShort) as refused:
        memory_settle.settle(["operator@192.0.2.10", "operator@192.0.2.11"], 0.91, run=sparks.run, sleep=sleep,
                             now=now, say=lambda line: None)
    assert str(refused.value).startswith("Not starting: available memory after dropping caches is below")
    assert "operator@192.0.2.11 (80.00 of 121.00 GiB available, vLLM asks 110.11 GiB)" in str(refused.value)
    assert "192.0.2.10 (" not in str(refused.value)
    # Without a utilization the caches still drop and nothing is refused.
    rows = memory_settle.settle(["operator@192.0.2.11"], None, run=Sparks({"operator@192.0.2.11": [80 * GIB]}).run,
                                sleep=sleep, now=now, say=lambda line: None)
    assert rows[0]["asks_gib"] is None


def connector_command(slot_bytes, *, slots=2, capture=True, restore=True, budget=268435456):
    import json
    extra = {"spark_cache_async_page_capture": capture, "spark_cache_async_page_capture_slot_bytes": slot_bytes,
             "spark_cache_async_page_capture_slot_count": slots, "spark_cache_cuda_restore": restore,
             "spark_cache_cuda_placement_arena_bytes": 67108864,
             "spark_cache_cuda_restore_arena_budget_bytes": budget}
    return ["serve", "/m", "--gpu-memory-utilization", "0.87", "--kv-transfer-config",
            json.dumps({"kv_connector": "SparkContextCacheConnector", "kv_connector_extra_config": extra})]


def test_sparkcache_pinned_memory_is_its_capture_slots_and_restore_arenas():
    assert memory_settle.pinned_bytes(connector_command(8 << 30)) == 16 * 2 ** 30 + 268435456
    assert memory_settle.pinned_bytes(connector_command(512 << 20, slots=3)) == 1536 * 2 ** 20 + 268435456
    assert memory_settle.pinned_bytes(connector_command(8 << 30, capture=False, restore=False)) == 0
    # Without a restore budget, one lane's two placement arenas.
    assert memory_settle.pinned_bytes(connector_command(512 << 20, budget=None)) == (1024 + 128) * 2 ** 20
    assert memory_settle.pinned_bytes(["serve", "/m", "--gpu-memory-utilization", "0.87"]) == 0
    flag, value = connector_command(512 << 20)[-2:]
    assert memory_settle.pinned_bytes(["serve", f"{flag}={value}"]) == 1024 * 2 ** 20 + 268435456


def test_a_share_and_pinned_memory_above_the_total_less_the_headroom_refuse_the_start_before_waiting():
    total = 121.69
    sparks = Sparks({"operator@192.0.2.10": [116 * GIB], "operator@192.0.2.11": [116 * GIB]},
                    total=round(total * GIB))
    sleep, now = clock()
    eight = memory_settle.pinned_bytes(connector_command(8 << 30))
    with pytest.raises(memory_settle.MemoryShort) as refused:
        memory_settle.settle(["operator@192.0.2.10", "operator@192.0.2.11"], 0.87, run=sparks.run,
                             pinned_bytes=eight, sleep=sleep, now=now, say=lambda line: None)
    text = str(refused.value)
    assert text.startswith("Not starting: the model's memory does not fit on operator@192.0.2.10 (vLLM's share "
                           "0.87 of 121.69 GiB is 105.87 GiB, SparkCache pins 16.25 GiB, together 122.12 GiB, "
                           "above 115.69 GiB: the total less 6 GiB for the system), operator@192.0.2.11 (")
    assert "spark_cache_async_page_capture_slot_bytes" in text and now() == 0
    # The same share with two 512 MiB slots fits and settles as before.
    rows = memory_settle.settle(["operator@192.0.2.10"], 0.87, run=sparks.run,
                                pinned_bytes=memory_settle.pinned_bytes(connector_command(512 << 20)),
                                sleep=sleep, now=now, say=lambda line: None)
    assert rows[0]["settled"]
    # Without a utilization there is no share to add to.
    assert memory_settle.budget_problem([("operator@192.0.2.10", round(total * GIB))], None, eight) is None


def test_every_installer_profile_fits_a_gb10_with_its_sparkcache_pinned_memory():
    # GB10 Sparks report 121.69 GiB MemTotal (0.87 of it is the 105.87 GiB the memory settle prints).
    from runtime.common import installer, profiles, toolchain_profiles
    total = round(121.69 * GIB)
    for profile in sorted(installer.INSTALLABLE):
        metadata, _ = profiles.load(profile)
        command = toolchain_profiles.read(profiles.ROOT / metadata["configuration"]["path"])["vllm_args"]
        utilization, _ = memory_settle.requested(command)
        problem = memory_settle.budget_problem([("gb10", total)], utilization, memory_settle.pinned_bytes(command))
        assert problem is None, (profile, problem)


def test_memory_that_does_not_settle_is_read_until_the_deadline():
    sparks = Sparks({"operator@192.0.2.10": [100 * GIB + step * 512 * 1024 for step in range(100)]})
    sleep, now = clock()
    said = []
    rows = memory_settle.settle(["operator@192.0.2.10"], 0.5, run=sparks.run, sleep=sleep, now=now, say=said.append)
    assert not rows[0]["settled"] and memory_settle.SETTLE_SECONDS < now() <= memory_settle.SETTLE_SECONDS + 10
    assert said[0].startswith("Memory before the start (not settled after 180 s; vLLM asks for 0.5 of the total)")


def test_the_installation_reads_vllms_share_from_rank_0s_container(monkeypatch, tmp_path):
    from runtime.common import installer
    lock = {"backend": "compose", "site": {"name": "pair-a", "ranks": [{"rank": 0, "host": "operator@192.0.2.10"},
                                                                       {"rank": 1, "host": "operator@192.0.2.11"}]}}
    monkeypatch.setattr(installer, "load", lambda directory: lock)
    monkeypatch.setattr(installer, "specifications", lambda value, only_rank=None: [
        SimpleNamespace(command=("serve", "/models/target", "--gpu-memory-utilization", "0.85"))])
    seen = {}

    def settle(hosts, utilization, **kwargs):
        seen.update(hosts=hosts, utilization=utilization, pinned=kwargs.get("pinned_bytes"))
        return []
    monkeypatch.setattr(memory_settle, "settle", settle)
    REAL_SETTLE_MEMORY(tmp_path, run=lambda host, argv: "")
    assert seen == {"hosts": ["operator@192.0.2.10", "operator@192.0.2.11"], "utilization": 0.85, "pinned": 0}
    # A SparkCache connector's slots and arenas count beside the share.
    monkeypatch.setattr(installer, "specifications", lambda value, only_rank=None: [
        SimpleNamespace(command=tuple(connector_command(512 << 20)))])
    REAL_SETTLE_MEMORY(tmp_path, run=lambda host, argv: "")
    assert (seen["utilization"], seen["pinned"]) == (0.87, 1024 * 2 ** 20 + 268435456)
    lock["backend"] = "glm-managed"
    REAL_SETTLE_MEMORY(tmp_path, run=lambda host, argv: "")
    assert seen["utilization"] is None


def test_a_repeated_installation_leaves_out_the_sparks_where_its_model_already_runs(monkeypatch, tmp_path):
    from runtime.common import installer
    lock = {"backend": "compose", "site": {"name": "pair-a", "ranks": [{"rank": 0, "host": "operator@192.0.2.10"},
                                                                       {"rank": 1, "host": "operator@192.0.2.11"}]}}
    monkeypatch.setattr(installer, "load", lambda directory: lock)
    monkeypatch.setattr(installer, "specifications", lambda value, only_rank=None: [
        SimpleNamespace(command=("serve", "/models/target", "--gpu-memory-utilization", "0.85"))])
    seen = {}

    def settle(hosts, utilization, **kwargs):
        seen["hosts"] = hosts
        return [{"host": host} for host in hosts]
    monkeypatch.setattr(memory_settle, "settle", settle)
    running = {"operator@192.0.2.10"}
    asked = []

    def run(host, argv):
        asked.append((host, argv[-1]))
        return "true\n" if host in running else ""
    # Rank 0's container of this deployment runs, rank 1's does not: only rank 1's Spark settles.
    assert REAL_SETTLE_MEMORY(tmp_path, run=run, say=lambda line: None) == [{"host": "operator@192.0.2.11"}]
    assert seen["hosts"] == ["operator@192.0.2.11"]
    assert "inspect --format '{{.State.Running}}' sr-pair-a-r0" in asked[0][1]
    # Running on every Spark (the same installation repeated while it serves): nothing settles or refuses.
    running.add("operator@192.0.2.11")
    seen.clear()
    said = []
    assert REAL_SETTLE_MEMORY(tmp_path, run=run, say=said.append) == []
    assert seen == {} and said == ["Memory settle skipped: this deployment's model already runs on every Spark."]
