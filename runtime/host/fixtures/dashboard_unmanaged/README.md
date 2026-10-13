# Unmanaged serving observations

`inspect-wrapper.json` contains the `Path`, `Args`, `Name`, entrypoint, command
and running state of both `seqs16-pp2` ranks captured on ring8c on
2026-10-13. Environment, labels, mounts and other Docker metadata are omitted.
The captured rendezvous is `198.18.12.1:29500`, and both ranks serve
`GLM-5.3-Flash-CSF-DCP2-TP2` through the SparkRing toolchain wrapper.

`nodes-missing-six.json` preserves the captured seven-node dashboard cache's
timestamps and container facts. `status-six.json` contains the missing Spark's
later local status, reduced to public model facts. Node IDs and hostnames use
fixture identities; position 7 follows position 5 in the incomplete cache.

The regression tests derive four arms from the captured inspect, changing
the served names and rendezvous addresses. Those four-arm observations and
their cable topology are synthetic. They test both distinct link subnets and
four cables reusing `198.18.0.1:29500`; they are not a claim that the captured
cache already contained four healthy arms. Positions 2, 3 and 7 had no model
facts in that cache. A separate case joins TP4 workers through a relay path
using the head's uniquely owned fabric address.
