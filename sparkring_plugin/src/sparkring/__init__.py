"""SparkRing: switchless direct-cable RDMA collectives (SIRCL) for vLLM.

This package delivers the SparkRing vLLM adapter as a standard
``vllm.general_plugins`` entry point, replacing the container-era
``sitecustomize``/bind-mount deployment. The runtime modules under
``sparkring/_vendor`` are byte-identical copies of
``spark_transport/integrations/vllm`` in the SparkRing repository,
enforced by ``tests/test_vendor_parity.py``.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"
