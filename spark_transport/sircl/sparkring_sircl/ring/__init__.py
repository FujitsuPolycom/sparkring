"""Standalone ring harness for SIRCL ring sessions.

The harness runs SIRCL's one-shot all-reduce and all-gather on chosen DGX
Sparks of a cabled ring, one process per rank inside the serving image (GPU
0, host networking), without vLLM. It checks every output bit for bit against
a host reference summed in fixed rank order, measures per-size latency
percentiles eagerly and in CUDA graph replay, and records the RDMA error
counters of every Spark before and after every size.

Modules: :mod:`.site` (the operator's description of the Sparks), :mod:`.plan`
(configurations, route maps and the launch plan, offline), :mod:`.remote` (SSH
and container commands), :mod:`.worker` (one rank, inside a container),
:mod:`.counters` (sysfs and ethtool counters), :mod:`.summary` (result
merging) and :mod:`.cli` (``python -m sparkring_sircl.ring``). The procedure
and the safety class of every step are in ``RUNBOOK.md`` next to the package.
"""
