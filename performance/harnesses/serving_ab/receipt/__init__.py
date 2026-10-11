"""Run receipts: proof of what a measured serving arm ran and whether two arms compare.

- :mod:`.collect` reads a deployment on the ring, read-only, into ``resolved.json`` (:mod:`.resolved`), from
  vLLM's own log lines (:mod:`.vllm_log`, :mod:`.pyrepr`), the container and process environment, SIRCL
  receipts, ``/metrics`` and GPU clocks.
- :mod:`.compare` diffs two receipts and checks an arm against its control and its declared change.
- :mod:`.method` names measurement methods and checks a benchmark result's recorded settings against one.
- :mod:`.window` refuses to start a measurement on a busy or reserved deployment and records who sent
  requests during one.

Usage: ``python -m performance.harnesses.serving_ab.receipt --help`` and the "Run receipts" section of
``performance/harnesses/serving_ab/README.md``.
"""
