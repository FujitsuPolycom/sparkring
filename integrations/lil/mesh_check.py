"""Bind supervised image trials to the installed four-rank fabric configuration."""

import inspect


def verify_installed_mesh(expected):
    """Run on the target host; never install, start, or stop a service."""
    import json
    from pathlib import Path
    import subprocess
    import sys

    source = Path("/opt/sparkring/managed-mesh")
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "runtime/glm53-spark-mtp3-mesh"))
    import managed_service

    config_path = "/etc/sparkring/managed-mesh/service.json"
    config, site, topology, _, identity = managed_service.load_config(config_path)
    rank = expected["rank"]
    if config["rank"] != rank or config["container_image"] != expected["image"]:
        raise ValueError("Installed mesh rank/image differs from the proposed trial")
    node = topology.rank(rank)
    if node.ssh_alias != expected["host"]:
        raise ValueError("Installed mesh host differs from the proposed trial")
    actual = {
        "HOST_IP": site["management_addresses"][rank],
        "MASTER_ADDR": site["management_addresses"][0],
        "SOCKET_IFNAME": node.management_netdev,
        "NCCL_IB_HCA": ",".join(
            node.port(direction, 0).rdma_device
            for direction in ("clockwise", "counter_clockwise")
        ),
        "NCCL_IB_GID_INDEX": "3",
    }
    directions = (
        ("clockwise", "counter_clockwise")
        if rank % 2 == 0
        else ("counter_clockwise", "clockwise")
    )
    for slot, direction in enumerate(directions):
        for function in (0, 1):
            local = node.port(direction, function)
            if local.peer_rank != rank ^ (1 if slot == 0 else 3):
                raise ValueError("Installed mesh peer ordering is incompatible")
            peer = topology.rank(local.peer_rank).port(
                local.peer_direction, local.peer_function
            )
            prefix = (
                "SPARK_TP4_"
                if function == 0
                else "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_"
            )
            actual[prefix + f"DEVICE{slot}"] = local.rdma_device
            actual[prefix + f"PEER{slot}"] = peer.ipv4
            actual[prefix + f"GID{slot}"] = "3"
    if actual != expected["network"]:
        raise ValueError(
            "Installed mesh addresses/devices/GIDs differ from the proposed trial"
        )
    container = json.loads(
        subprocess.check_output(
            ["docker", "inspect", config["container_id"]], text=True
        )
    )[0]
    if container["Id"] != config["container_id"] or container["State"]["Running"]:
        raise ValueError("Stop the managed model before starting a supervised trial")
    # A healthy fabric does not enroll the trial container with its supervisor.
    # Keep the original model unit stopped throughout a supervised trial.
    result = subprocess.run(
        [
            sys.executable,
            str(source / "runtime/glm53-spark-mtp3-mesh/managed_service.py"),
            "gate",
            "--config",
            config_path,
            "--timeout",
            "60",
        ],
        capture_output=True,
        text=True,
        timeout=65,
        check=True,
    )
    evidence = json.loads(result.stdout)
    if evidence.get("ready") is not True or evidence.get("identity") != identity:
        raise ValueError("Managed mesh readiness identity changed")
    return "mesh configuration and readiness verified; supervised trial only"


def command(rank, host, image, network):
    expected = {"rank": rank, "host": host, "image": image, "network": dict(network)}
    for name in (
        "NCCL_IB_GID_INDEX",
        "SPARK_TP4_GID0",
        "SPARK_TP4_GID1",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID0",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID1",
    ):
        expected["network"].setdefault(name, "3")
    program = inspect.getsource(verify_installed_mesh)
    program += "\nprint(verify_installed_mesh(" + repr(expected) + "))\n"
    return {
        "argv": ["sudo", "-n", "python3", "-c", program],
        "expected": "mesh configuration and readiness verified; supervised trial only",
    }
