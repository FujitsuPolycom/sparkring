# R7 overlay source fixtures

These files are test-only snapshots of the Apache-2.0-licensed vLLM result
tree that [`runtime/exl3-r7/pins.json`](../pins.json) names, after
`scripts/glm35_q40/prepare_q40_overlay_inputs.py`
has applied the two exact-Q40 input edits: the W4A16 scratch reservation in
`exl3.py` and the routed-expert capturer from `q40_v2_route_capture.patch`
in `model_runner.py`. The pinned tree alone does not carry either file at
these hashes. The snapshots let offline CI verify the target-only exact-Q40
overlay without fetching source trees or depending on an operator's ignored
`.sparkring` build directory.

| File | SHA-256 | Role |
|---|---|---|
| `vllm/model_executor/layers/quantization/exl3.py.fixture` | `8e0051faf9b8bac9eefd6f38a5f0133a30bca4c0b5ab41962537e2f13cf968f4` | exact input to the target-only capacity-40/block-8 overlay |
| `vllm/v1/worker/gpu/model_runner.py.fixture` | `992486a1a70fd0cb54c54cf3f70af0f09b4bcc6ae3bbf433e66b1296415015b4` | exact input to the pre-graph numerical-attestation overlay |

The production build still applies each overlay to the prepared vLLM tree
and rejects any input hash drift. These snapshots are not installed into a
container image.
