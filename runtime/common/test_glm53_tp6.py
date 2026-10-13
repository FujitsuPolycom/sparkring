"""Offline admission and six-rank launch contracts; not GPU qualification."""
import json

import pytest

from runtime.common import derived_checkpoint, installer, profiles, toolchain_profiles, transport
from runtime.common.test_transport import eight_spark_image, sircl_deployment

PROFILE = "glm53-nvfp4-csf-tp6"


def option(args, name):
    return args[args.index(name) + 1]


def test_six_rank_launch_uses_path6_and_three_dcp_pairs():
    lock, section = sircl_deployment(PROFILE, "cycle", 8, list(range(6)), image=eight_spark_image())
    assert section["group"]["name"] == "cycle-6"
    assert section["group"]["positions"] == list(range(6))
    assert section["tuning"]["row"] == "cycle"
    assert section["tuning"]["row_source"] == "rules"
    specs = installer.specifications(lock)
    assert len(specs) == 6
    for rank, spec in enumerate(specs):
        args = list(spec.command)
        assert option(args, "--node-rank") == str(rank)
        assert option(args, "--tensor-parallel-size") == option(args, "--nnodes") == "6"
        assert ("--headless" in args) == (rank > 0)
        assert spec.environment["SIRCL_NCCL"] == "never"
        assert spec.environment["SIRCL_GROUPS"] == "tp,dcp"
        assert spec.environment["SIRCL_FABRIC"] == "ring:8"
        assert spec.environment["SIRCL_ROUTE_SCOPE"] == "fabric"
        assert spec.environment["SIRCL_COLUMN_GATHER"] == "0"
        assert "SIRCL_MAX_RELAYS" not in spec.environment
        assert option(args, "--decode-context-parallel-size") == "2"
        assert option(args, "--cp-kv-cache-interleave-size") == "1"
        assert int(option(args, "--max-model-len")) >= 131072 + 2048
        assert json.loads(option(args, "--speculative-config"))["draft_tensor_parallel_size"] == 6
    assert len(transport.dcp_groups(section, list(specs[0].command))) == 3


def test_padding_preserves_model_identity_and_aligns_prefill():
    resolved = profiles.resolve(PROFILE)
    config = toolchain_profiles.read(installer.ROOT / f"profiles/{PROFILE}/config.json")
    assert toolchain_profiles.canonical(config) == config
    pad = json.loads(option(config["vllm_args"], "--hf-overrides"))
    assert pad == {"num_attention_heads": 66, "original_num_attention_heads": 64,
                   "moe_intermediate_size": 2304, "original_moe_intermediate_size": 2048}
    assert pad["num_attention_heads"] // 6 == 11
    assert pad["moe_intermediate_size"] // 6 % 128 == 0
    assert resolved["topology"] == "direct-path-6"
    card = installer.setup.selection(PROFILE)
    manifest = derived_checkpoint.load(card)
    assert manifest["repository"] == "sparkring-derived/GLM-5.3-NVFP4-CSF-Serving"
    assert manifest["revision"] == "bbd8f71def6bde14a201887796692cb1cc4828a8"
    assert config["environment"]["GLM53FULL_LATENT_SHARD"] == "0"
    assert config["environment"]["GLM53FULL_EH_PROJ_TP"] == "0"
    assert not {"mxfp8-linears", "draft-quantization", "column-split-loading"} & set(config["start_checks"])


def test_six_rank_profile_refuses_eight_hosts():
    with pytest.raises(ValueError, match="6"):
        sircl_deployment(PROFILE, "cycle", 8, list(range(8)), image=eight_spark_image())


def test_six_rank_profile_is_research_only_and_requires_explicit_image_admission():
    from runtime.common import compose, installer_image
    assert PROFILE in profiles.research_catalog()
    assert PROFILE in installer_image.SIRCL_ONLY and PROFILE in compose.FABRIC_PROFILES
    assert PROFILE not in installer_image.default_lock()["profiles"]
    assert PROFILE not in compose.SUPPORTED


def test_six_rank_placement_preserves_the_other_pair(tmp_path):
    from runtime.common import fabric_layout
    from runtime.host import placement
    layout = fabric_layout.layout("cycle", 8)
    arc = placement.parse("0-5", layout)
    assert placement.check(arc, layout=layout, profile_nodes=6, profile=PROFILE) == tuple(range(6))
    assert placement.unsupported(layout, arc) is not None  # no generic qualification
    assert PROFILE in placement.fitting_profiles(layout)
    placement.record(tmp_path, (6, 7), tmp_path / "pair")
    assert (6, 7) not in placement.conflicting(tmp_path, arc)
    assert placement.choose(tmp_path, PROFILE, layout, 6) == arc


def test_six_rank_routes_use_the_universal_cycle_table():
    top = transport.group_topology("ring:8", list(range(6)), research=True)
    assert top.max_relays() == 3 and top.lane_count == 2
    assert [lane.positions for lane in top.lanes(5, 0)] == [(5, 6, 7, 0)] * 2
    for size in (2, 3, 6):
        assert len(transport.dcp_topologies("ring:8", list(range(6)), size)) == 6 // size


def test_every_relayed_lane_matches_installed_addresses_devices_and_tags():
    from runtime.common import fabric_document
    from runtime.host import relays
    from runtime.host.test_relays import prepared
    _, document, table = prepared('cycle', 8)
    top = transport.group_topology('ring:8', list(range(6)), research=True)
    for rank in range(6):
        local = fabric_document.devices(document, rank)
        for peer in range(6):
            if peer == rank:
                continue
            remote = fabric_document.devices(document, peer)
            for lane in top.lanes(rank, peer):
                if not lane.relays:
                    continue
                origin = local[lane.device()]
                destination = remote[lane.remote_device()]['address']
                section = table['positions'][rank]
                assert any(row['dst'] == destination + '/32' and row['dev'] == origin['netdev']
                           and row['src'] == origin['address'] and row['relay_positions'] == list(lane.relays)
                           for row in section['routes'])
                marker = next(row for row in section['markers'] if row['rdma'] == lane.device())
                assert {'dst': destination, 'ethertype': relays.hex16(relays.tag(len(lane.relays)))} in marker['rules']
                for index, position in enumerate(lane.relays):
                    remaining = len(lane.relays) - index
                    assert any(rule['handle'] == remaining
                               and rule['protocol'] == relays.hex16(relays.tag(remaining))
                               and rule['actions'][0]['pedit']['eth_type'] == relays.hex16(relays.tag(remaining - 1))
                               for rule in table['positions'][position]['filters'])
