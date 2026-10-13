# GLM-5.3-Flash TP2/DCP2 prefill admission

Status: **Experimental**. The [source transformation](patch.py) admits KDA
prefill coalescing and mHC row ownership at TP2/DCP2, behind
`VLLM_B12X_KDA_PREFILL_COALESCING` and `VLLM_GLM53_MHC_PREFILL_SHARD`.
Both default to off in vLLM. The research profile enables both.

**Source-read:** KDA admission is a parallel-size allowlist in
`vllm/v1/core/recurrent_prefill_checkpoint.py`. The recurrent head count is
partitioned by TP; the sequence is not partitioned by DCP. mHC has two
allowlists in `mhc_prefill_sharding.py`, in `configure` and `validate_model`.
It gathers full token rows before attention and scatters TP projection
results back to row owners. These ownership boundaries do not intersect
DCP's four-token KV stripes. Every other source guard remains in place.

[admission.patch](admission.patch) shows the complete source diff, including
SIRCL pins and hook inventory. [Image composition](../../../runtime/images/compositions/dcp2-prefill/README.md)
owns packaging and receipts. Fixtures are unchanged sources from image
`d52737a109e0`; their preimages are pinned in the composition manifest.
The two vLLM fixtures retain their Apache-2.0 headers; SIRCL fixtures are
covered by [SIRCL's license](../../../spark_transport/sircl/LICENSE).

## Local evidence

**Measured, 2026-10-13:** [gpu_exactness.py](gpu_exactness.py) executes the
image's B12X kernels on an RTX 5090, PyTorch 2.11.0+cu128, CUDA runtime 12.8,
CuTe DSL 4.8.0. Forty comparisons were bit-identical (`torch.equal`, maximum
absolute difference zero), with finite outputs:

- KDA: 32 local heads of width 128, BF16 inputs/output, FP32 recurrent state;
  one 8192-token call exporting 2048/4096/6144 states versus four 2048-token
  calls; two seeds and zero/nonzero initial states. All outputs, interior
  checkpoints and final states match.
- mHC: H4096, 4096 and 8192 full rows versus two contiguous half-row calls;
  pre, fused post/pre, post and final row contraction. The half-row calls
  reuse the full-ceiling prepared geometry, as vLLM's `B12xMHCResidual`
  does with this profile's small graph sizes and compilation mode 0.

**Measured counterexample:** preparing independent mHC plans for full and
half capacities changes TF32 projection split geometry and produced BF16
differences up to 0.03125. This is not the profile's plan selection. Do not
extend the exactness claim to independently tuned half-row plans. Test any
change to capture/compile sizes or preparation before admitting it.

Run the component test under the workstation owner's GPU lock:

```bash
gpu-lock.sh codex-dcp2-prefill python integrations/vllm/dcp2_prefill/gpu_exactness.py \
  --sources /path/to/exported/image/site-packages
```

The CPU suite executes pinned source functions, checks admission/fallbacks,
checkpoint metadata and physical-page alias refusal, and exercises the
derived-layer receipt/lock machinery with synthetic receipts:

```bash
python -m pytest integrations/vllm/dcp2_prefill/test_patch.py -q
```

**Unknown:** full-model logits, distributed all-reduce versus reduce-scatter
rounding, GB10 serving stability, KV reuse across eviction/preemption, image
startup and performance. Single-GPU component equality is not pair serving
qualification. Use the [bounded pair plan](../../../runtime/images/compositions/dcp2-prefill/HARDWARE-PLAN.md).
