# GLM-5.2 V2 MTP index reuse compatibility patch

## Conclusion

Reusing MTP step 0's sparse-attention index buffer for serial steps 1+ is the
checkpoint-defined behavior when `index_share_for_mtp_iteration=true`. It is
therefore model-correct for this GLM-5.2 checkpoint. It is not numerically
equivalent to recomputing top-k from each later step's changed hidden state;
the current V2 behavior can select different indices and is the behavior being
corrected.

The source remains an opt-in compatibility adapter rather than an upstream
runtime change. A DCP1/remap/fixed-K4 experiment activated it behind the exact
version/configuration gates. The first lifecycle attempt wedged during startup
prewarm because request-specific reuse state survived beyond a proposal. The
adapter now returns to compute mode after every successful or failed proposal;
the replacement four-rank launch completed startup and served all correctness
and performance gates. This document does not make the adapter a default.

## Read-only source findings

Inspected read-only in the preserved `glm52-trace` container:

- vLLM version:
  `0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea.fi25dd814.cu132.20260626`.
- Both `/hybridmodel/config.json` and `/hybridmodel/mtp-draft/config.json`
  declare `index_share_for_mtp_iteration=true` and `index_topk=2048`.
- The validated launches use V2, four serial MTP tokens, DCP4 or explicitly
  remapped DCP1, draft DCP sharding, and global DCP top-k. CUDA graphs are
  currently disabled.
- `llm_base_proposer` sets `skip_topk=False` around MTP step 0, then
  `skip_topk=True` for serial steps 1+.
- `DeepSeekMultiTokenPredictor.set_skip_topk` propagates the flag to each MTP
  MLA module.
- `MultiHeadLatentAttentionWrapper.forward` suppresses only the indexer when
  `skip_topk=True`. Query/KV projections, sparse attention using the existing
  indices, the MLP/MoE path, logits, and sampling still execute.
- `load_eagle_model` rewires draft buffers only when the target model exposes a
  root `topk_indices_buffer`. The GLM target construction used here threads a
  locally created buffer into its layer modules without guaranteeing that
  root attribute, while DeepSeek MTP allocates its own draft buffer.
- V2 `MTPSpeculator` inherits `_prefill` and `_multi_step_decode` without
  changing `skip_topk`, so it recomputes the index at every serial step.

The draft-local shared buffer remains live across the serial MTP loop, and the
loop is ordered on the same execution stream. With global top-k enabled, the
buffer contains the merged global token IDs consumed by the DCP-sharded sparse
attention backend. No later target forward intervenes between MTP step 0 and
the remaining serial steps.

### Fixed-K4 load failure

The first live opt-in attempt stopped during the runtime gate with a missing
target root buffer and three draft holders that were not identical to that
missing target value. This was a validation bug, not an activation-timing
problem: the legacy proposer explicitly computes MTP step 0's own indices
before later steps reuse them. Serial reuse therefore requires coherent
draft-side ownership, not target/draft aliasing.

The adapter does not create an alias or copy. It now requires every draft
`topk_indices_buffer` holder—including the indexer, indexer operation, and MLA
consumer—to reference one identical non-null buffer of shape `[tokens, 2048]`.
Multiple identities, `None`, or a wrong shape fail before reuse is armed.

## Patch behavior

[`spark_glm52_mtp_index_reuse.py`](spark_glm52_mtp_index_reuse.py) is an
opt-in in-memory compatibility adapter:

1. It requires an exact vLLM version and SHA-256 fingerprints for all relevant
   V2 lifecycle, model, buffer-sharing, and MLA methods. A mismatch raises
   before any class attribute is changed.
2. At draft-model load it additionally requires the exact GLM target and
   DeepSeek MTP draft architectures, one NextN layer, `index_topk=2048`, DCP4
   or DCP1 with `SPARK_B12X_DCP1_PHYSICAL_REMAP=1`, global DCP top-k, draft
   DCP sharding, a callable `set_skip_topk`, and one coherent draft-local
   buffer identity across every holder.
3. It arms index computation before `_prefill`, enables reuse only after a
   successful step-0 prefill, and returns to compute mode after every completed
   or failed proposal. A failed prefill also leaves compute mode armed. This
   scopes request-specific physical indices strictly to serial steps inside one
   proposal, so startup profiling, prewarming, and unrelated direct forwards
   cannot consume stale state.
4. It reports process-local compute/reuse arm counts and completed logical
   forward counts through `get_stats()`. Setting
   `SPARK_GLM52_MTP_INDEX_REUSE_LOG_EVERY=N` logs a snapshot every `N`
   completed proposals.

For CUDA graphs, vLLM captures `_prefill` before the single-step decode graph.
The adapter therefore captures the prefill graph with `skip_topk=False` and
the decode graph with `skip_topk=True`; replay uses those baked branches. The
exact `capture`, `propose`, `_prefill`, and `_multi_step_decode` sources are
all fingerprint-gated. The currently running configuration is eager, so each
proposal also performs the Python transition directly.

## Activation and rollback

Activation requires a reviewed startup hook that calls:

```python
from spark_glm52_mtp_index_reuse import install

install()
```

and explicitly set:

```text
SPARK_GLM52_MTP_INDEX_REUSE=1
```

For DCP1, activation additionally requires:

```text
SPARK_B12X_DCP1_PHYSICAL_REMAP=1
```

That variable is an attestation, not the implementation. The exact-source
startup patch must first keep B12X logical IDs through any TP row gather and
then map them through the consuming rank's block table to physical KV slots.
Without that remap, measured deep-position acceptance collapses; the adapter
therefore refuses DCP1 reuse.

Unset that variable (or set it to `0`) and restart workers for the primary
rollback. `uninstall()` also restores the original class attributes in-process
for tests or pre-traffic rollback.

The source-only compatibility check does not install the patch:

```bash
python spark_glm52_mtp_index_reuse.py
```

## GPU-free evidence

[`test_glm52_mtp_index_reuse.py`](test_glm52_mtp_index_reuse.py) uses a fake
V2 speculator and a fake indexer at the actual method seam. It verifies:

- current V2 behavior: four serial steps produce four indexer calls;
- patched behavior: one indexer call and three shared-buffer reads;
- capture ordering: compute-mode prefill followed by reuse-mode decode;
- unknown source fingerprints fail before patching;
- a missing target root buffer is accepted when all draft holders are coherent;
- split draft buffer identities fail before reuse;
- a failed step-0 prefill leaves compute mode armed, preventing stale reuse;
- a failure after step 0 and a successful proposal both return to compute
  mode before any unrelated forward;
- DCP1 fails closed without the physical-remap attestation and activates with
  it;
- the disabled environment toggle neither imports nor patches vLLM;
- checkpoints without the share flag remain unchanged;
- in-process uninstall restores four-indexer-call behavior.
