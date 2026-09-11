# Explicit recurrent checkpoints during prefill

Status: **implemented**. This source payload lets one eligible fresh prompt
prefill export up to two interior recurrent states. The scheduler specifies
positions explicitly; the model runner, GLM model-state adapter, GDN metadata,
and B12X recurrence kernel carry and materialize the same plan. Convolution
checkpoints preserve logical history rather than physical speculative slots.

The enabled path requires `SPARK_GDN_PREFILL_CHECKPOINTS=2`. It accepts complete,
block-aligned fresh prompts of at most 8192 tokens and at most two required
interior checkpoints. Continuations and ineligible shapes retain the scheduler's
ordinary path. It does not generalize checkpoint coalescing to arbitrary lengths.

## Build interface

`patch-manifest.json` maps image paths to required preimage hashes and exact
replacement payloads. The filesystem identified by image digest
`sha256:3882eccf0b42e26dad399a3ea89a45988e04413802fccfb146d1486f8bb2fc13`
defines the measured parent composition. A source-built parent must satisfy
every runtime preimage even if its image digest differs.

Run these commands from this directory. Installation is a Docker build operation,
not a command to run against a serving container or host installation.

```sh
python3 install.py verify-context --manifest-sha256 0970d29ec33e9f8525a2cc55989ab0deb937bff88b5f16d4d5280035359e4c55
python3 install.py verify-preimages --manifest-sha256 0970d29ec33e9f8525a2cc55989ab0deb937bff88b5f16d4d5280035359e4c55
python3 install.py apply --manifest-sha256 0970d29ec33e9f8525a2cc55989ab0deb937bff88b5f16d4d5280035359e4c55
```

All preimages, payload hashes, Python syntax, and required symbols are checked
before source replacement. Installation preserves file modes and ownership,
refreshes existing bytecode, and normalizes affected timestamps. Its JSON output
includes `postimages`, the map the composition builder must incorporate into
SparkCache's runtime ownership contract. The builder must regenerate the package
source receipt and execute the ownership verifier before declaring the image
usable. This installer does not replace SparkCache package files or its receipts.

`attestation_reference_hashes` records the measured composition's attestations
for provenance; those hashes are not replacement payloads for another SparkCache
revision. `native_hashes` records its cache-library dependencies for the enclosing
image verifier. The original payload manifest is identified by
`source_manifest_sha256`; all 18 retained source payloads preserve its exact bytes.

## Evidence and limits

A four-Spark TP4/DCP4 MTP3 deployment with an 8192-token prefill budget
passed 17 semantic requests covering fresh/repeated pairs, extended triples,
and mixed requests. Repeated requests reported 6144 cached tokens with exact
expected answers. All four ranks reported the 8192-token prompt's checkpoint
positions at 6144 and 7168 across 34 GDN layers. This qualifies those cases;
it does not qualify arbitrary models, context lengths, or rebuilt image compositions.

Offline checks run with `python -m pytest test_checkpoint_package.py`. They check source
identity rejection and execute the packaged checkpoint planner. GPU recurrence,
convolution state, and live semantic validation remain hardware requirements.
