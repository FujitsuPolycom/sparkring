# Recovered GLM-5.2 reference overlay

This directory publishes the Python delta recovered from the runtime that
produced SparkRing's reference GLM-5.2 results.

- Base: `vllm-project/vllm@fcc614141e5e9ab18cb304c476f7feed2a9552e3`
- Modified upstream files: 59 exact unified patches
- New runtime files: 12 exact, hash-pinned additions under `added/`
- Recovery artifact SHA-256:
  `d23e8269a00bca32d23e51163b46cd464164edc5504ab404936f2c984c3ee4d4`

`preimages.json` makes every patch fail closed against the wrong upstream
source. `additions.json` pins every new file and forbids overwriting an
existing path. `provenance.json` records best-effort attribution markers for
each of the 71 included files.

Two historical patches are deliberately absent:

- `vllm/v1/executor/multiproc_executor.py`
- `vllm/v1/core/kv_cache_utils.py`

Those patches returned empty follower results and masked a launch-topology
error. They are unsafe and unnecessary with vLLM's intended headless
multi-node path. See
[`docs/PUBLIC_STARTUP_SHIM_AUDIT.md`](../../../docs/PUBLIC_STARTUP_SHIM_AUDIT.md).

The build applies this component first, then the independently developed
SparkCache compatibility patches. All consumed files are pinned by
`runtime/runtime-lock.json`.
