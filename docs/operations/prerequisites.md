# SparkRing prerequisites

Choose a [profile](../../profiles/README.md), then check the following on every Spark.

## Hardware and topology

| Setup | Sparks | Data connections |
|---|---:|---|
| Pair, including GLM-5.3-Flash TP2 | 2 | Direct ConnectX link; follow the profile's device mapping |
| Four-node ring | 4 | Four cables in a `0-1-2-3-0` cycle |
| Switched | 4 | Switch-connected ports, as specified by the profile |
| Six-node ring | 6 | Experimental; six cables in a closed cycle |

Keep rank assignments fixed. Rank 0 serves the API. Use a separate management
connection that remains reachable while configuring the data fabric.

## Operating system and storage

- Linux ARM64 with working NVIDIA drivers.
- Docker with NVIDIA Container Toolkit and access to the GPU and `/dev/infiniband`.
- Enough local disk for the **complete checkpoint on each rank**, the image and caches.
- Matching model revisions and image identities across ranks.

The selected quickstart gives model-specific storage and memory requirements.

## Network requirements

- Working RoCEv2, link state, addressing and MTU on the selected fabric interfaces.
- SSH access from the controller, plus the profile's rendezvous and control ports.
- Rank 0's API port reachable by intended clients.

### Four-Spark managed hardware-forwarded mesh

Follow [mesh host setup](../GLM53_SPARK_MESH_HOST_SETUP.md) for cabling,
all four RDMA functions, driver settings and managed services. Pass the
[startup memory checks](../../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md#automatic-startup-memory-preparation)
before loading the model.

### Routing and forwarding across the fabric

Rings need routes, IPv4 forwarding and forwarding rules between fabric interfaces.
Use the [Ring Doctor procedure](fabric-repair.md) to inspect them before applying repairs.

### Management safety during repair

Keep management separate from the fabric being changed. The
[repair guide](fabric-repair.md#management-safety-during-repair) covers controller
identity checks and persistent routing rules.

## Local configuration and preflight

Use the site or environment template named by your quickstart. Fill in each
rank's addresses, interfaces and paths; keep private inputs in an ignored
location such as `.sparkring/`. Run the quickstart's checks and review its
launch plan before starting containers.

## Safety boundary

Stop affected workloads before changing NICs, routes or services. Keep independent
management access and use the quickstart's coordinated startup and shutdown steps.
