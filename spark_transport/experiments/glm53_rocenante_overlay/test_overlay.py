"""Host-only contracts for the private GLM-5.3 RoCEnante overlay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spark_transport.experiments.cx7_hairpin_diagonal import fabric
from spark_transport.experiments.glm53_rocenante_overlay import build_bundle
from spark_transport.experiments.glm53_rocenante_overlay import plan as overlay_plan
from spark_transport.experiments.glm53_rocenante_overlay.rocenante_vllm_overlay import (
    VirtualDiagonalAdapter,
    load_contract,
)


HERE = Path(__file__).parent


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_locks_six_qps_threshold_routing_and_cpu() -> None:
    contract = load_contract(HERE / "overlay_contract.json")

    assert contract["canonical_hca_order"] == [
        "rocep1s0f0",
        "rocep1s0f1",
        "roceP2p1s0f0",
        "roceP2p1s0f1",
    ]
    assert contract["peer_hca_maps"] == {
        "0": "1=0/2,2=0/3,3=1/3",
        "1": "0=1/3,2=0/2,3=0/3",
        "2": "0=1/2,1=1/3,3=0/2",
        "3": "0=0/2,1=1/2,2=1/3",
    }
    runtime = contract["runtime"]
    assert runtime["opposite_rank_paths"] == 2
    assert runtime["origin_queue_pairs_per_rank"] == 6
    assert runtime["direct_then_diagonal_threshold_bytes"] == 196608
    assert runtime["proxy_cpu"] == 13
    assert runtime["proxy_cpu"] not in runtime["forbidden_proxy_cpus"]
    assert contract["dispatch"]["candidate"]["maximum_query_rows"] == 32
    assert contract["metadata"]["required_backend"] == "gloo"
    assert contract["metadata"]["create_nccl_communicator"] is False
    assert contract["excluded_performance_arms"] == {
        "hardware_qos": {"status": "research-only", "disposition": "rejected"},
        "four_opposite_paths_mesh32": {
            "status": "research-only",
            "disposition": "rejected",
        },
    }


def test_candidate_signature_is_exact_q1_q32_bf16_width4096() -> None:
    torch = pytest.importorskip("torch")
    adapter = VirtualDiagonalAdapter.__new__(VirtualDiagonalAdapter)
    adapter._closed = False
    adapter.device = torch.device("cuda", 0)
    adapter.width = 4096
    adapter.minimum_query_rows = 1
    adapter.maximum_query_rows = 32
    adapter.execution_mode = "both"
    adapter.captured_sircl_rows = frozenset()

    def tensor(shape=(32, 4096), dtype=torch.bfloat16, contiguous=True):
        return SimpleNamespace(
            shape=shape,
            dtype=dtype,
            is_cuda=True,
            device=torch.device("cuda", 0),
            is_contiguous=lambda: contiguous,
        )

    assert adapter.eligible(tensor())
    assert adapter.eligible(tensor((1, 4096)))
    assert not adapter.eligible(tensor((33, 4096)))
    assert not adapter.eligible(tensor((32, 6144)))
    assert not adapter.eligible(tensor(dtype=torch.float16))
    assert not adapter.eligible(tensor(contiguous=False))


def test_sitecustomize_keeps_sircl_then_adds_distinct_b12x_health_gate() -> None:
    source = (HERE / "sitecustomize.py").read_text(encoding="utf-8")

    base = source.index("runpy.run_path")
    overlay = source.index("install_rocenante()")
    install_health = source.index("install_rocenante_health()")
    assert base < overlay < install_health
    health = (HERE / "rocenante_health_gate.py").read_text(encoding="utf-8")
    assert "_rocenante_health_gate" in health
    assert "_sparkring_health_gate" not in health
    assert "output = original(self)" in health
    assert "_checked(require_health)" in health


def test_adapter_wraps_saved_sircl_chain_without_creating_nccl() -> None:
    source = (HERE / "rocenante_vllm_overlay.py").read_text(encoding="utf-8")

    assert "original_all_reduce = CudaCommunicator.all_reduce" in source
    assert "return original_all_reduce(self, tensor)" in source
    assert "communicator.cpu_group" in source
    assert 'backend != "gloo"' in source
    assert "dist.all_gather_object(votes, local_vote, group=self.group)" in source
    assert "new_group" not in source
    assert "ProcessGroupNCCL" not in source
    assert "init_process_group" not in source
    assert '"B12X_ROCE_OPPOSITE_PATHS": "2"' in source
    assert '"B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES": "196608"' in source
    assert "_set_exact_environment(name, expected)" in source
    assert "with _inherited_thread_affinity(self.proxy_cpu)" in source


def test_bundle_copies_base_sircl_and_only_roce_subpackage(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "sircl"
    base.mkdir()
    records = []
    for name in sorted(build_bundle.REQUIRED_SIRCL_FILES):
        path = base / name
        path.write_bytes((name + "\n").encode())
        records.append({"path": name, "sha256": _sha(path)})
    (base / "sparkring-overlay-manifest.json").write_text(
        json.dumps({"schema": "base", "files": records}), encoding="utf-8"
    )
    b12x = tmp_path / "b12x-repo"
    roce = b12x / "b12x" / "comm" / "roce"
    roce.mkdir(parents=True)
    (roce / "__init__.py").write_text("API_VERSION = 1\n", encoding="utf-8")
    (roce / "_roce_proxy.c").write_text("/* six QPs */\n", encoding="utf-8")
    monkeypatch.setattr(
        build_bundle,
        "_git_state",
        lambda repository: {
            "commit": "f" * 40,
            "roce_source_dirty": True,
            "roce_status": [" M b12x/comm/roce/_roce_proxy.c"],
        },
    )
    output = tmp_path / "bundle"

    manifest = build_bundle.build(base, b12x, output)

    assert manifest["schema"] == build_bundle.MANIFEST_SCHEMA
    assert (output / "sitecustomize.py").is_file()
    assert (output / "rocenante_health_gate.py").is_file()
    assert (output / "sircl_sitecustomize.py").read_text() == "sitecustomize.py\n"
    assert (output / "b12x_overlay/b12x/comm/roce/_roce_proxy.c").is_file()
    assert not (output / "b12x_overlay/b12x/__init__.py").exists()
    config = json.loads((output / "rocenante-overlay-config.json").read_text())
    assert config["runtime"]["origin_queue_pairs_per_rank"] == 6
    assert config["artifacts"]["b12x_git"]["roce_source_dirty"] is True


def test_bundle_converts_private_v24_manifest_and_supplies_health_files(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "sircl-v24"
    base.mkdir()
    for name in sorted(
        build_bundle.REQUIRED_SIRCL_FILES - build_bundle.SUPPLIED_SIRCL_SUPPORT
    ):
        (base / name).write_bytes((name + "\n").encode())
    private_manifest = {
        "schema": "sparkring-private-sircl-bundle/v1",
        "date": "2026-09-04",
        "source_commit": "a" * 40,
        "image_id": "sha256:image",
        "library_sha256": _sha(base / "libspark_transport_capi.so"),
        "backend_sha256": _sha(base / "spark_tp4_backend.py"),
        "prefill_exposure": "fused",
        "prefill_rail_mode": "dual",
        "fused_min_query_rows": 128,
        "fused_max_query_rows": 8192,
        "operation_slots": 2,
        "elements_per_row": 4096,
        "public": False,
    }
    (base / "manifest.json").write_text(json.dumps(private_manifest), encoding="utf-8")
    b12x = tmp_path / "b12x-repo"
    roce = b12x / "b12x" / "comm" / "roce"
    roce.mkdir(parents=True)
    (roce / "__init__.py").write_text("API_VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        build_bundle,
        "_git_state",
        lambda repository: {
            "commit": "f" * 40,
            "roce_source_dirty": True,
            "roce_status": [],
        },
    )

    output = tmp_path / "composed"
    manifest = build_bundle.build(base, b12x, output)

    assert manifest["base_sircl_manifest_name"] == "manifest.json"
    assert json.loads((output / "manifest.json").read_text()) == private_manifest
    assert (output / "spark_tp4_capability.py").is_file()
    assert (output / "spark_tp4_health_gate.py").is_file()
    generated = json.loads((output / "sparkring-overlay-manifest.json").read_text())
    roles = {item["path"]: item["role"] for item in generated["files"]}
    assert roles["spark_tp4_capability.py"] == "spark_sircl_support"
    assert roles["spark_tp4_health_gate.py"] == "spark_sircl_support"
    config = json.loads((output / "rocenante-overlay-config.json").read_text())
    assert config["artifacts"]["base_sircl_manifest"] == private_manifest


def test_sidecar_lifetime_binding_preserves_exact_cleanup() -> None:
    manifest = {
        "apply_phases": [
            {
                "name": "source_markers",
                "commands": [
                    {
                        "argv": [
                            "/helper",
                            "marker",
                            "apply",
                            "--run-seconds",
                            "7200",
                        ]
                    }
                    for _ in range(8)
                ],
            }
        ],
        "cleanup": {"phases": ["source_markers", "intermediate_rules"]},
    }

    changed = overlay_plan._bind_marker_lifetime(manifest, 7200)

    for command in changed["apply_phases"][0]["commands"]:
        assert command["argv"][-1] == "7200"
        assert command["required_helper_contract"] == {
            "signal_safe_cleanup": True,
            "maximum_runtime_seconds": 7200,
            "binary_identity_verified": True,
        }
    assert changed["cleanup"] == manifest["cleanup"]

    with pytest.raises(overlay_plan.PlanError, match="marker lifetime differs"):
        overlay_plan._bind_marker_lifetime(manifest, 300)


def test_plan_source_is_non_executing_and_uses_exact_cleanup_builder() -> None:
    source = (HERE / "plan.py").read_text(encoding="utf-8")

    assert "fabric.build_rocenante_plan" in source
    assert "fabric.require_hardware_gate" in source
    assert "fabric.build_cleanup_manifest" in source
    assert '"execution_authorized": False' in source
    for forbidden in (
        "subprocess.run",
        "os.system",
        "paramiko",
        "docker run",
        "ssh ",
    ):
        assert forbidden not in source


def test_full_topology_gate_requires_24_qps_eight_rules_and_cleanup(
    monkeypatch,
) -> None:
    selected = SimpleNamespace(
        topology_sha256="a" * 64,
        sha256="b" * 64,
        shared_diagonal_flow_label=True,
        markers=tuple(
            SimpleNamespace(flow_label=16383, udp_source_port=65535)
            for _ in range(8)
        ),
        routes=(None,) * 8,
        tc_rules=(None,) * 8,
    )
    inventory = {
        "total_origin_qps": 24,
        "direct_origin_qps": 16,
        "forwarded_origin_qps": 8,
        "origin_qps_per_rank": {str(rank): 6 for rank in range(4)},
    }
    monkeypatch.setattr(fabric, "rocenante_inventory", lambda _selected: inventory)
    gate = {
        "schema": fabric.HARDWARE_GATE_SCHEMA,
        "status": "qualified",
        "topology_sha256": selected.topology_sha256,
        "plan_sha256": selected.sha256,
        "source_ethertype_rewrite_in_hw": True,
        "intermediate_ethertype_restore_in_hw": True,
        "remote_payload_byte_match": True,
        "rx_icrc_encapsulated_delta": 0,
        "cleanup_verified": True,
    }

    assert overlay_plan.require_full_topology_gate(selected, gate) == inventory

    with pytest.raises(overlay_plan.PlanError, match="eight hardware restore rules"):
        overlay_plan.require_full_topology_gate(
            SimpleNamespace(**{**selected.__dict__, "tc_rules": (None,) * 7}), gate
        )

    unclean = {**gate, "status": "research-only", "cleanup_verified": False}
    with pytest.raises(fabric.FabricError, match="hardware gate"):
        overlay_plan.require_full_topology_gate(selected, unclean)
