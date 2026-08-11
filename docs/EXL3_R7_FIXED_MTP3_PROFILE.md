# GLM-5.2 R7 fixed-depth-three speculative profile

## Status and evidence scope

This configuration is a **public-functional-lane, live-validated intermediate
candidate** for four NVIDIA DGX Sparks / GB10 GPUs. It completed target,
draft-prefill, and draft-decode CUDA graph capture on all four ranks with one
shared capture stream per process and CUDA device. Repeated 128-token and
256-token greedy outputs matched the MTP-disabled control, requested logprobs
were finite, all three speculative positions were active, and the four-rank
transport audit passed. The 9.25 GB/rank derivative also passed startup,
correctness, endpoint, and C1/C2/C4/C8 decode gates.

This profile is not the repository default or an accepted public-functional
matrix. It is the rollback and matched performance control for the later
[fixed-MTP4, 9.25 GB KV candidate](EXL3_R7_FIXED_MTP4_PROFILE.md).

The maintainer-held `prepare_exl3_r7_mtp3.py` utility consumed the exact
stock-DCP4 control and qualified fixed-MTP2 profile. It rejected baseline
drift, emitted a separate fixed-MTP3 candidate and site, copied the fixed-MTP2
input bytes unchanged as the rollback profile, and refused output paths that
aliased any input. A sanitized public generator and complete stock-DCP4 input
chain are not yet published.

## Exact semantic delta

The candidate changes the fixed speculative depth from two to three:

```text
profile_id suffix:             -fixed-mtp2 -> -fixed-mtp3
VLLM_SPARK_MTP_MODE_ID:        fixed-mtp2  -> fixed-mtp3
VLLM_SPARK_MTP_TOKENS:         2           -> 3
num_speculative_tokens:        2           -> 3
site serving.mtp_tokens:       2           -> 3
VLLM_SPARK_MAX_QUERY_ROWS:     24          -> 32
```

The query-row ceiling is capacity, not an independent algorithm change. With
eight active sequences and three speculative tokens, a target verification
step can contain `8 * (3 + 1) = 32` rows. The fixed-MTP2 profile's ceiling of
24 is insufficient for that legal batch. Its existing compilation contract
already captures sizes 1 through 32, so the CUDA graph-width list does not
change.

Everything else remains byte-identical to the fixed-MTP2 control, including:

- target-only online K6 and the attested checkpoint EXL3 routed experts;
- producer BF16 non-expert tensors for the layer-78 MTP draft;
- top-level `fp8_ds_mla` KV cache with no draft `kv_cache_dtype` override;
- TP4 plus DCP4 `ag_rs`, 9 GB KV memory per rank, and interleave size one;
- hybrid Spark custom TP collectives plus patched NCCL-IB fallback;
- stock DCP and indexer collectives;
- `FULL_AND_PIECEWISE` graphs without eager execution;
- the attested read-only shared-capture-stream overlay;
- disabled adaptive-depth control and disabled index reuse.

The speculative JSON remains greedy and explicitly selects TP4, EXL3,
`b12x`, and `B12X_MLA_SPARSE`. It must not contain `quantization_config` or
`kv_cache_dtype`; those fields would respectively violate target-only K6
ownership or reconstruct the draft cache configuration with an unsupported
explicit 16-token block size.

## Generated local artifacts

The preparer writes distinct ignored artifacts:

```text
.sparkring/exl3-r7-candidate/launch-dcp4-mtp3-stock.json
.sparkring/exl3-r7-candidate/site-dcp4-mtp3.yaml
.sparkring/exl3-r7-candidate/launch-dcp4-mtp3-rollback.json
```

The rollback file must be SHA-256 and byte-identical to
`.sparkring/exl3-r7-candidate/launch-dcp4-mtp2-stock.json`.

Offline validation requires the focused parser and derivative tests, generic
runtime-profile validation, site-schema validation, and a four-rank dry launch
plan. A dry plan proves configuration expansion only; it does not authorize or
perform a host mutation or service replacement.
