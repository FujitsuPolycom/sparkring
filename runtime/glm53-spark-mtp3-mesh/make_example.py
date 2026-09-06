"""Generate synthetic, non-site-specific mesh examples using benchmark-only addresses."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def topology_example():
    ranks = []
    for rank in range(4):
        ports = {}
        for direction in ("clockwise", "counter_clockwise"):
            clockwise = direction == "clockwise"
            physical_port = 0 if clockwise else 1
            edge = rank + 1 if clockwise else (rank - 1) % 4 + 1
            endpoint = 1 if clockwise else 2
            ports[direction] = [{
                "function": function,
                "netdev": f"en{'P2' if function else ''}p1s0f{physical_port}np{physical_port}",
                "rdma_device": f"roce{'P2' if function else ''}p1s0f{physical_port}",
                "ipv4_cidr": f"198.18.{edge + 100 * function}.{endpoint}/32",
                "mac": f"02:{rank:02x}:{physical_port:02x}:{function:02x}:00:01",
                "peer_rank": (rank + (1 if clockwise else -1)) % 4,
                "peer_direction": "counter_clockwise" if clockwise else "clockwise",
                "peer_function": function,
            } for function in (0, 1)]
        ranks.append({"rank": rank, "ssh_alias": f"spark-r{rank}", "management_netdev": "enP7s7", "ports": ports})
    return {
        "schema": "sparkring-cx7-hardware-diagonal-fabric/v1", "status": "research-only", "group_id": 77,
        "socket_direct_functions": 2, "flow_label_base": 16383, "shared_diagonal_flow_label": True,
        "endpoint_route_strategy": "adjacent_gateway", "standard_ether_type": 2048, "marked_ether_type": 34997,
        "standard_udp_destination_port": 4791, "roce_gid_index": 3, "bounded_runtime_seconds": 7200,
        "expected_ethernet_mtu": 9000, "expected_roce_mtu": 4096,
        "orchestration": {"host_helper_path": "/UNSUPPORTED/external-fabric-orchestrator",
                          "remote_state_root": "/run/sparkring-mtp3-mesh"}, "ranks": ranks,
    }


def site_example():
    return {
        "schema": "sparkring-glm53-mtp3-mesh-site/v1", "topology_file": "fabric.example.json",
        "management_addresses": [f"192.0.2.{10 + rank}" for rank in range(4)],
        "model_roots": ["/srv/models/GLM-5.3-Flash-NVFP4-Spark/df116c4fb16b1d37ae43d2cfd624de26ffbc832e"] * 4,
        "cache_roots": ["/srv/sparkring/glm53-mtp3-cache"] * 4,
        "bundle_root": "/opt/sparkring/mtp3-mesh-bundle", "container_prefix": "glm53-spark-mtp3-mesh",
        "marker_binary": "/opt/sparkring/bin/mlx5-rdma-tx-rewrite-probe",
        "marker_binary_sha256": "e4403012ee0e15cd0e474c689398833868d5596b99e0d396254b6ff06bba5072",
        "state_root": "/run/sparkring-mtp3-mesh",
    }


if __name__ == "__main__":
    for name, document in (("fabric.example.json", topology_example()), ("site.example.json", site_example())):
        (HERE / name).write_text(json.dumps(document, indent=2) + "\n", newline="\n")
