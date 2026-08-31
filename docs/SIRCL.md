# SIRCL

SIRCL is the **Switchless Inference RDMA Collective Layer**. It is SparkRing's
native collective transport for a directly cabled four-system cycle. It is not
a separate service or an NCCL fork.

## Implemented boundary

SIRCL maintains RDMA sessions, registered arenas, and device-published command
rings. CUDA graph replay submits pre-established work without Python or host
control work in the replay path.

A four-rank collective is decomposed into two perfect matchings of the physical
cycle. Each step transfers data only between direct neighbours. This schedule
does not claim a generic multi-node collective interface or support outside the
four-rank topology documented in [architecture](ARCHITECTURE.md).

## Admission

SIRCL admission depends on the collective family, width, dtype, topology, and
runtime adapter contract. Unsupported work remains on patched NCCL. A profile
must name its admitted SIRCL families and link the evidence that supports them;
model identity alone never enables a transport path.

The [profile registry](profiles/README.md) records which deployments use SIRCL.
Artifact-specific comparisons and measurements belong under
[`performance/records/`](../performance/records/).

## Failure behavior

Initialization rejects incomplete rank membership, incompatible geometry,
missing peer connectivity, and registration failures. Unsupported or
unqualified collective work must use the profile's NCCL path rather than being
silently admitted to SIRCL.

## Implementation

The native source, integration boundary, and tests are under
[`spark_transport/`](../spark_transport/). Cable and link requirements are in
[cable qualification](../spark_transport/CABLE_QUALIFICATION.md).
