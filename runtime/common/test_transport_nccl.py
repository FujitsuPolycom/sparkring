"""The nccl transport: vLLM's PyNccl alone on a group NCCL's cabling rule holds for; offline.

``--transport nccl`` runs no SIRCL session and no RoCEnante slot: no SIRCL
variable or plugin reaches a container, and vLLM's PyNccl carries every
tensor-parallel and expert-parallel collective. The tests place deployments on
a simulated fabric document (``test_transport.document``) and check the
environment of every rank, the per-rank ``NCCL_IB_HCA`` the fabric document
gives each position, the refusals for groups whose ranks reach each other
through relays, and what the plan prints.
"""
import json
import re

import pytest

from runtime.common import fabric_document, image_lock, installer, transport
from runtime.common.test_transport import TP2, document, eight_spark_image, install_site

def nccl_deployment(profile, shape, size, positions, *, image=None, dcp=None):
    """An nccl deployment lock of ``profile`` on ``positions`` of a simulated fabric document."""
    image = image or installer.installer_image.default_lock()
    section = transport.nccl_section(image, document(shape, size), positions,
                                     dcp=transport.profile_dcp(profile) if dcp is None else dcp)
    placement = None if list(positions) == list(range(size)) else positions
    lock = installer.make_lock(profile, install_site(len(positions), placement=placement), "1" * 40, "2" * 64,
                               image_runtime=image_lock.v2_view(image), transport=section)
    return lock, section


def test_an_nccl_pair_runs_pynccl_alone_with_the_devices_of_the_cable():
    lock, section = nccl_deployment(TP2, "pair", 2, [0, 1])
    assert section["backend"] == "nccl" and section["group"]["name"] == "pair"
    assert section["group"]["cabling"] == "all" and section["nccl"] == {"settings": {}, "reasons": {}}
    # A pair fabric cables both Sparks through port 0, so both ranks name the same devices.
    assert section["devices"] == [["rocep1s0f0", "roceP2p1s0f0"]] * 2
    for spec in installer.specifications(lock):
        environment = spec.environment
        assert environment["VLLM_ENABLE_ROCE_ALLREDUCE"] == "0"
        assert environment["SPARKRING_TRANSPORT_PROFILE"] == ""
        assert environment["SPARKRING_TRANSPORT_MANIFEST_SHA256"] == ""
        assert environment["SPARK_TP4_ENABLED"] == "0"
        assert environment["VLLM_PLUGINS"].split(",") == ["b12x_loader", "sparkring_status"]
        assert not [key for key in environment if key.startswith("SIRCL_")]
        assert "NCCL_SKIP_TREE_CONNECT" not in environment
    assert [spec.environment["NCCL_IB_HCA"] for spec in installer.specifications(lock)] == \
        ["=rocep1s0f0,roceP2p1s0f0", "=rocep1s0f0,roceP2p1s0f0"]


def test_an_nccl_pair_of_a_cycle_names_the_devices_of_both_cable_directions():
    # Positions 6 and 7 of an eight-Spark cycle: Spark 6 faces Spark 7 through its port 0,
    # Spark 7 faces Spark 6 through its port 1.
    lock, section = nccl_deployment(TP2, "cycle", 8, [6, 7])
    assert section["devices"] == [["rocep1s0f0", "roceP2p1s0f0"], ["rocep1s0f1", "roceP2p1s0f1"]]
    assert [spec.environment["NCCL_IB_HCA"] for spec in installer.specifications(lock)] == \
        ["=rocep1s0f0,roceP2p1s0f0", "=rocep1s0f1,roceP2p1s0f1"]


def test_an_nccl_whole_cycle_arms_the_ring_and_names_each_ranks_own_devices():
    profile = "glm53-flash-csf-tp8"
    lock, section = nccl_deployment(profile, "cycle", 8, list(range(8)), image=eight_spark_image())
    assert section["group"]["name"] == "cycle-8" and section["group"]["cabling"] == "ring"
    assert section["nccl"]["settings"] == {"NCCL_ALGO": "Ring", "NCCL_SKIP_TREE_CONNECT": "1"}
    assert section["nccl"]["reasons"] == transport.NCCL_SETTING_REASONS
    specs = installer.specifications(lock)
    # Spark 0's port 0 faces Spark 1's port 1 and its port 1 faces Spark 7's port 0, so the ranks name their
    # devices in different orders; the profile's single global value would be wrong for some of them.
    assert [spec.environment["NCCL_IB_HCA"] for spec in specs] == [
        "=rocep1s0f0:1,roceP2p1s0f0:1,roceP2p1s0f1:1,rocep1s0f1:1",   # rank 0: port 0 first, then port 1
        "=rocep1s0f1:1,roceP2p1s0f1:1,rocep1s0f0:1,roceP2p1s0f0:1",   # rank 1: port 1 first, then port 0
        "=rocep1s0f1:1,roceP2p1s0f1:1,rocep1s0f0:1,roceP2p1s0f0:1",   # rank 2: faces rank 0 through its port 1
        "=rocep1s0f1:1,roceP2p1s0f1:1,rocep1s0f0:1,roceP2p1s0f0:1",
        "=rocep1s0f1:1,roceP2p1s0f0:1,roceP2p1s0f1:1,rocep1s0f0:1",   # rank 4: the ring turns around
        "=rocep1s0f0:1,roceP2p1s0f0:1,rocep1s0f1:1,roceP2p1s0f1:1",
        "=rocep1s0f0:1,roceP2p1s0f0:1,rocep1s0f1:1,roceP2p1s0f1:1",
        "=rocep1s0f0:1,roceP2p1s0f0:1,rocep1s0f1:1,roceP2p1s0f1:1",   # rank 7: faces rank 0 through its port 0
    ]
    for rank, spec in enumerate(specs):
        environment = spec.environment
        assert environment["NCCL_ALGO"] == "Ring" and environment["NCCL_SKIP_TREE_CONNECT"] == "1"
        # The profile's other NCCL settings are kept.
        assert environment["NCCL_SWITCHLESS_RING_ONLY"] == "1" and environment["NCCL_IB_GID_INDEX"] == "3"
        assert environment["VLLM_ENABLE_ROCE_ALLREDUCE"] == "0"
        assert "sircl" not in environment["VLLM_PLUGINS"].split(",")
        assert not [key for key in environment if key.startswith("SIRCL_")]
        command = list(spec.command)
        assert command[command.index("--tensor-parallel-size") + 1] == "8"


def test_the_plan_names_the_transport_and_why_each_setting_was_added():
    _, section = nccl_deployment("glm53-flash-csf-tp8", "cycle", 8, list(range(8)), image=eight_spark_image())
    lines = transport.plan_lines(section)
    assert lines[0] == ("Transport: nccl on every collective; SIRCL is not loaded and the RoCEnante slot is off "
                        "(--transport nccl)")
    assert "  NCCL group: cycle-8 at positions 0, 1, 2, 3, 4, 5, 6, 7" in lines[1]
    assert lines[2] == f"  NCCL_ALGO=Ring: {transport.NCCL_SETTING_REASONS['NCCL_ALGO']}"
    assert lines[3] == f"  NCCL_SKIP_TREE_CONNECT=1: {transport.NCCL_SETTING_REASONS['NCCL_SKIP_TREE_CONNECT']}"
    _, pair = nccl_deployment(TP2, "pair", 2, [0, 1])
    assert "  NCCL settings: the profile's own; a cabled pair needs none added" in transport.plan_lines(pair)


@pytest.mark.parametrize("positions, described", [([0, 1, 2], "ranks 2-0 (positions 2-0)"),
                                                  ([0, 1, 2, 3], "ranks 3-0 (positions 3-0)")])
def test_an_nccl_path_of_three_or_more_is_refused_with_the_cabling_rule(positions, described):
    image = installer.installer_image.default_lock()
    with pytest.raises(transport.TransportError, match=re.escape(described)) as refused:
        transport.nccl_section(image, document("cycle", 8), positions)
    message = str(refused.value)
    assert message.startswith("--transport nccl cannot run on this group: no cable between")
    assert transport.nccl_cabling_rule() in message
    assert message.endswith("SIRCL ring sessions run the Sparks between them (--transport sircl)")


def test_an_nccl_deployment_whose_decode_context_groups_are_paths_is_refused():
    # glm53-nvfp4-tp8 serves with decode-context parallelism 4: its groups are lines of four Sparks.
    with pytest.raises(transport.TransportError, match="decode-context-parallel groups of 4 ranks"):
        nccl_deployment("glm53-nvfp4-tp8", "cycle", 8, list(range(8)), image=eight_spark_image())
    # Decode-context groups of two are cabled pairs; the whole cycle as one group is a whole cycle.
    lock, section = nccl_deployment("glm53-nvfp4-tp8", "cycle", 8, list(range(8)), image=eight_spark_image(), dcp=2)
    assert section["group"]["dcp"] == 2 and section["group"]["cabling"] == "ring"


def test_the_nccl_transport_needs_the_fabric_document_and_takes_no_nccl_mode():
    image = installer.installer_image.default_lock()
    with pytest.raises(transport.TransportError, match="--transport nccl reads the fabric document"):
        transport.choose(image, None, backend="nccl")
    with pytest.raises(transport.TransportError, match="--nccl applies to deployments on SIRCL"):
        transport.choose(image, document("pair", 2), backend="nccl", nccl="auto")
    assert transport.choose(image, document("pair", 2), backend="nccl") == ("nccl", None, None)
    # SIRCL's availability does not gate the nccl transport: a pair image without the SIRCL layer runs it.
    assert transport.choose(installer.installer_image.default_lock(), document("pair", 2), backend="nccl") == \
        ("nccl", None, None)


@pytest.mark.parametrize("edit, message", [
    (lambda section: section["nccl"]["settings"].update(NCCL_TREE_CONNECT="1"), "NCCL ring settings"),
    (lambda section: section["group"].update(cabling="none"), "differs from its layout"),
    (lambda section: section["group"].update(positions=[0, 1, 2]), "one per rank"),
    (lambda section: section["devices"].pop(), "each rank's RDMA devices"),
    (lambda section: section["nccl"].update(reasons={"NCCL_ALGO": "because"}), "the reason each added setting is needed"),
    (lambda section: section["nccl"]["settings"].update(NCCL_ALGO="Ring", NCCL_SKIP_TREE_CONNECT="1"),
     "cabled pair needs no NCCL setting"),
])
def test_an_nccl_section_that_disagrees_with_the_lock_is_refused(edit, message):
    _, section = nccl_deployment(TP2, "pair", 2, [0, 1])
    edit(section)
    with pytest.raises(ValueError, match=message):
        installer.make_lock(TP2, install_site(2), "1" * 40, "2" * 64,
                            image_runtime=image_lock.v2_view(installer.installer_image.default_lock()),
                            transport=section)


def test_the_lock_records_the_nccl_section_and_its_identity_covers_it():
    lock, section = nccl_deployment(TP2, "pair", 2, [0, 1])
    assert lock["transport"] == section and installer.validate(lock) == lock
    cycle, _ = nccl_deployment("glm53-flash-csf-tp8", "cycle", 8, list(range(8)), image=eight_spark_image())
    plain = installer.make_lock(TP2, install_site(2), "1" * 40, "2" * 64,
                                image_runtime=image_lock.v2_view(installer.installer_image.default_lock()))
    assert len({lock["id"], cycle["id"], plain["id"]}) == 3


def test_an_nccl_deployment_checks_the_fabric_document_but_needs_no_tuning_tables(tmp_path):
    _, section = nccl_deployment(TP2, "pair", 2, [0, 1])
    with pytest.raises(transport.TransportError, match="cannot be used"):
        transport.check_host_document(section, root=tmp_path)
    path = tmp_path / fabric_document.HOST_PATH.lstrip("/")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document("pair", 2)))
    assert transport.check_host_document(section, root=tmp_path)["ok"]
