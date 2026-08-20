"""SparkRing: switchless direct-cable RDMA collectives (SIRCL) for vLLM.

This package delivers the SparkRing vLLM adapter as a standard
``vllm.general_plugins`` entry point, so a deployment reaches the transport
by installing a wheel rather than by overlaying files onto a container
image. The runtime modules under ``sparkring/_vendor`` are byte-identical
copies of ``spark_transport/integrations/vllm`` in the SparkRing
repository, enforced by ``tests/test_vendor_parity.py``.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"
