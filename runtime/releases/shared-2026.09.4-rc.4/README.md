# SparkRing 2026.09.4-rc.4

Status: **research-only opt-in prerelease** for ARM64/GB10. This image adds the
audited LIL vLLM/B12X changes and SparkRing integrations to the September 23 eugr
foundation. Stable profile defaults remain unchanged.

```bash
docker pull ghcr.io/fujitsupolycom/sparkring:shared-2026.09.4-rc.4
docker pull ghcr.io/fujitsupolycom/sparkring@sha256:3d3411b66dd2a4f78ea062ec0a923308f302df14f0e0f9a6ae8961946ae603c8
```

Use the [TP2 profile](../../../profiles/qwen38-flash-next-qad-tp2-eugr/README.md)
or [TP4 profile](../../../profiles/qwen38-flash-next-qad-tp4-eugr/README.md), and
follow the [shared deployment procedure](../../../docs/operations/external-qwen.md).
The image has 31 filesystem layers. An anonymous Docker pull completed and
verified all 5,792 receipt-owned runtime files. The verification host already
had foundation layers; this is not an empty-engine pull claim.

## Runtime composition

- vLLM `91f94783190ce716e2238f899b25d44e84e2a525` and B12X
  `a0a0c425e20f45e041de241afd6e76dcfc52d281`, including the audited loader,
  prepared-kernel, Qwen speculation/checkpoint and MiMo component changes.
- The external foundation's native PyTorch/vLLM/FlashInfer binaries, with
  source-bound Python, Triton and CuTe changes. First startup compiles and tunes
  kernels; later starts can still perform tuning and graph capture.
- SparkRing's dual-domain NCCL runtime, prepared RoCEnante transport, Qwen
  prefill integrations and SparkCache publication/restore contracts.
- Explicit HC modes: projection sharding on TP2; prefill token-row ownership
  with replicated HC projections on TP4. The incompatible combination is
  rejected. The TP4 row path uses direct NCCL collectives while the generic
  policy retains eligible RoCEnante operations.
- `GET /v1/sparkring/status`, using the inference API's authentication. It
  reports configured and effective settings, passive worker metadata, missing
  observations and stale/pending RPC results. It does not instrument requests
  or claim that a prepared kernel executed for a particular request.
- Persistent-cache namespaces bound to both composition and serving profile.

The profiles retain MTP3, 16 sequences, capture ceiling 64, 262144 context and
24 GiB FP8 KV per rank. MTP5, MXFP8 LM-head activation and the `w13` scale mode
remain separate experiments. Asynchronous loading is a startup improvement;
its inclusion alone does not establish decode throughput.

## Evidence and limits

The exact [qualification record](qualification.json) is authoritative for live
checks and pending work. Final-image CPU checks passed 482 vLLM, 103 B12X and
30 runtime-status cases. Protected source checks passed 69/187 vLLM
baseline/candidate cases and 23/228 B12X baseline/candidate cases. GPU component
and prior prototype measurements retain their own source and workload scope.

TP2 passed startup, short generation, a 10,508-token retrieval prompt, immediate
prefix reuse and a retained restart that physically restored 8,544 tokens on
both ranks. Its status endpoint reported both workers. This is bounded
correctness evidence, not a throughput, full-context, sustained-load or arbitrary
media qualification. TP4 passed the same retrieval and restart checks, restoring
10,080 tokens physically on each of its four ranks; all four status reports
confirmed HC prefill row ownership.
Full MiMo serving is not qualified by its isolated loader/kernel checks.

On the final TP4 composition, matched cold C1 fixtures measured median prefill
rates of **4,515 tokens/s at 8K** and **4,653 tokens/s at 32K**, with three runs
each, 16 output tokens and zero cached prompt tokens. This retained the earlier
HC row-ownership result. It does not qualify decode throughput, concurrency or
other prompt lengths; exact conditions and individual runs are in the record.

The published image differs from the tested TP2 image only in labels: filesystem
layers and runtime configuration otherwise match. The installer deliberately
reports `serving_qualified: false`; installation verifies bytes, while model and
hardware qualification belong to the separate evidence record.

## Sources and provenance

The [PR disposition inventory](PR-DISPOSITION.md) and its
[machine-readable record](pr-disposition-manifest.json) cover 409 entries at the
recorded audit cutoff. They distinguish integration, equivalence, conditional
scope and exclusions. Source inclusion is not a claim that every optional path
is enabled or qualified.

The [source manifest](source-manifest.json), [source bundle](source-bundle.json),
[composition](composition.json), [component notices](components.md),
[publication](publication.json) and [registry verification](verification.json)
identify immutable inputs. Release assets include complete vLLM/B12X archives,
reconstructable patches and the status wheel with its matching source archive.
Model weights and private site configuration are not included.

The `external-build-context.tar.gz` asset contains the exact source-bound
integration payload and pinned base inventory. On an ARM64 build host, extract
it and run `docker build --network=none -t sparkring:rc4-local external-context`.
Docker still needs the digest-pinned foundation image locally or from its
registry. Installation verifies the expected base and resulting runtime bytes;
locally rebuilt image metadata need not have the published image ID.
