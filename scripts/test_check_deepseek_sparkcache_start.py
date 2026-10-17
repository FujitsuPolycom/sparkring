"""DeepSeek startup evidence is checked by role, including contradictory records."""

import pytest

from scripts import check_deepseek_sparkcache_start as check


def config(role):
    return (f"sparkcache: config role={role} root={check.CACHE_ROOT} dcp=1 access_mode=read-write"
            " store=True restore=True cuda_restore=False max_span=524288 load_threads=2\n")


LCM = "kv lcm block sizes 256\n"
TOPOLOGY = "sparkcache: topology profile=deepseek-v41-b12x-hma groups=17 policies=full,replay,sliding,transient\n"


def test_scheduler_config_is_required_only_on_rank_zero():
    worker = config("WORKER") + TOPOLOGY
    assert check.check_log(worker + LCM, rank=0) == ["missing SCHEDULER connector config"]
    assert check.check_log(worker + config("SCHEDULER"), rank=0) == ["missing LCM page"]
    assert check.check_log(worker + config("SCHEDULER") + LCM, rank=0) == []
    for rank in (1, 2, 3):
        assert check.check_log(worker, rank=rank) == []
        assert check.check_log("", rank=rank)


@pytest.mark.parametrize("before,after", [
    ("dcp=1", "dcp=2"), ("cuda_restore=False", "cuda_restore=True"),
    ("max_span=524288", "max_span=1048576"), ("store=True", "store=False"),
    (check.CACHE_ROOT, "/cache/other"), ("deepseek-v41-b12x-hma", "deepseek-v4-fp8-hma"),
    ("groups=17", "groups=8"), ("kv lcm block sizes 256", "kv lcm block sizes 128"), ("full,replay,sliding,transient", "full,sliding"),
])
def test_missing_and_contradictory_settings_fail(before, after):
    text = config("WORKER") + config("SCHEDULER") + TOPOLOGY + LCM
    assert check.check_log(text.replace(before, after))
    assert check.check_log(text + text.replace(before, after))


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_worker_only_rank_rejects_contradictory_lcm_when_present(rank):
    assert check.check_log(config("WORKER") + TOPOLOGY, rank=rank) == []
    assert check.check_log(config("WORKER") + TOPOLOGY + LCM + "kv lcm block sizes 128\n", rank=rank)


def test_combined_startup_line_carries_configuration_and_topology():
    def startup(role):
        return ("sparkcache: startup profile=deepseek-v41-b12x-hma groups=17 "
                "policies=full,replay,sliding,transient identity=0123456789ab integrity=fast "
                + config(role).removeprefix("sparkcache: config "))
    text = startup("SCHEDULER") + startup("WORKER") + LCM
    assert check.check_log(text) == []
    assert check.check_log(startup("WORKER"), rank=1) == []
    assert check.check_log(text + startup("WORKER").replace("groups=17", "groups=1"))
    assert check.check_log(text + startup("WORKER").replace("dcp=1", "dcp=2"))
