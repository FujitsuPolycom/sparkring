# SparkRing prerequisites

What every Spark needs for the manual profile guides in
[Set up SparkRing manually](setup.md). For the normal path,
`sudo sparkring install`, see the requirements in [Install SparkRing](install.md).
[Host preparation](host-preparation.md) has the commands; the
[pair network guide](pair-network.md) or the ring procedure sets up the fabric.

## Hardware and topology

| Setup | Sparks | Data connections |
|---|---:|---|
| Pair, including GLM-5.3-Flash TP2 | 2 | Direct ConnectX link, using the profile's device mapping |
| Four-node ring | 4 | Four cables in a `0-1-2-3-0` cycle |
| Switched | 4 | Switch-connected ports, as the profile specifies |
| Six-node ring (experimental) | 6 | Six cables in a closed cycle |

Keep rank assignments fixed; rank 0 serves the API. Keep a separate management
connection that stays reachable while you configure the data fabric.

## Operating system and storage

- Linux ARM64 with working NVIDIA drivers.
- Docker with NVIDIA Container Toolkit and access to the GPU and `/dev/infiniband`.
- Local disk for the **complete checkpoint on each rank**, the image and caches.
- The same model revision and image ID on every rank.

The profile guide gives model-specific storage and memory needs. Budget space on
the actual destination filesystems with `sparkring setup storage`
([host preparation, step 5](host-preparation.md#5-check-storage-on-every-rank)).
`host check` only requires 20 GiB free on the root filesystem.

## Network requirements

- Working RoCEv2, link state, addressing and MTU on the fabric interfaces.
- SSH from the controller, plus the profile's rendezvous and control ports.
- Rank 0's API port reachable by the intended clients.
- The configured GID index must still resolve to the intended address after a
  reboot; automatic interface configuration can reorder GIDs
  ([secondary-port record](../../performance/records/transport/nccl-dual-domain-deepseek.md#serving-measurements-and-library-compatibility)).

### Four-Spark managed hardware-forwarded mesh

Follow [mesh host setup](../GLM53_SPARK_MESH_HOST_SETUP.md) for cabling, all
four RDMA functions, driver settings and managed services. Pass the
[startup memory checks](../../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md#automatic-startup-memory-preparation)
before loading the model.

### Routing and forwarding across the fabric

Rings need routes, IPv4 forwarding and forwarding rules between fabric
interfaces. Inspect them with [Ring Doctor](fabric-repair.md) before applying repairs.

### Management safety during repair

Keep management separate from the fabric you are changing. The
[repair guide](fabric-repair.md#management-safety-during-repair) covers
controller identity checks and persistent routing rules.

## Local configuration and preflight

Use the site or environment template the profile guide names. Fill in each
rank's addresses, interfaces and paths, and keep private inputs in an ignored
directory such as `.sparkring/`. Run the guide's checks and review its launch
plan before starting containers.

## Safety boundary

Stop affected workloads before changing NICs, routes or services. Keep
independent management access, and use the guide's coordinated start and stop steps.
