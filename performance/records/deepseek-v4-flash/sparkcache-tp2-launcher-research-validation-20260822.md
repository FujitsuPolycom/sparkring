# DeepSeek-V4 SparkCache TP2 launcher validation

Lane: **public-functional**. Status: **research-only**. Evidence scope:
**live-validated launcher with a separately identified checkpoint
composition**. Hardware: two directly
cabled NVIDIA DGX Spark systems, TP2/DCP1. Evidence scope: guarded candidate
creation, cold store, coordinated restart, external restore, exact final
content, and fresh canary. This record does not reproduce the qualified
`bd6b0117...` checkpoint receipt.

## Conditions

| Input | Verified identity |
|---|---|
| SparkRing TP2 launcher | `scripts/sparkcache_deepseek_tp2.py`; SHA-256 `7253d6adfcd3b2310f3975fa2f8bd89ac01870562cf39c6516a2d7a73b3b5afa` |
| Container entrypoint | `scripts/sparkcache_tp2_entrypoint.sh`; SHA-256 `47e41773c311744d0453f9dabac7c6ea050c1553798082060d32cf003314dd41` |
| Serving image | manifest `sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028`; image ID `sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7` |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731`; complete-manifest SHA-256 `905dca196ad0495e86557df98fb55a35ab8ae3b4a7fbe84a2696382a070e8a78`; 74 files; 166,898,665,144 bytes. The repository revision differs from the qualified recipe and is retained in the private manifest. |
| SparkCache wheel | version `0.1.0a1`; SHA-256 `87c17d8dab5052f5a7833349dc9b99b76a3b6531ca6f0d3deff812f724fecdcc` |
| SparkCache source | commit `5344192526d328b5cda3417c857b7ffb048fca8a`; deployable source-tree SHA-256 `342236206e35ad648fa9603c7259a6dd894da7a285be855b3dabbee73284f631` |
| vLLM overlays | scheduler `2f34aa9d65a495a86d814c90f654fbe1ff754cfdbecd204b98d513652ca3e06d`; config `71c4f9e622dd8b3d665f2a2b5fb932206516ddb82873ff89283c63aa80696005`; receipt `41c8506ae1badce9d3ef1d04be2e4d356fefa351865295090fae455f5a09ffd2` |

The launcher rehashed the complete mounted checkpoint on both ranks, rejected
stale image model identities, emitted explicit DCP1, required host network and
IPC, verified disjoint rank-local cache roots, and created both candidates for
inspection before a coordinated start. The research checkpoint digest was
passed explicitly with `--research`; it was not substituted for the recipe's
qualified digest.

## Measurement

Raw-record location: unavailable in a public artifact; exact responses,
inspections, logs, and manifests remain in the private validation workspace.
The gate ran once, so no repeatability distribution or uncertainty estimate is
available.

The cold miss contained 73,774 prompt tokens and returned exactly
`SPARKCACHE_OK:9540`. Both ranks committed a 73,728-token reusable span with
digest prefix `c6203281dc6a`.

| Rank | Snapshot | Background commit | Verified restore after restart |
|---:|---:|---:|---:|
| 0 | 565.9 ms | 2,610.4 ms | 705.0 ms |
| 1 | 745.5 ms | 2,694.1 ms | 487.0 ms |

## Result

Both containers were stopped and started together without moving or removing
their dedicated roots. Each rank discovered one compatible manifest. The hit
gate reported 73,814 external-cache queried tokens, 73,728 external-cache hit
tokens, exact final content, and `SPARKCACHE_CANARY_OK`; rank-0 health remained
HTTP 200.

## Conclusion

The published SparkRing TP2 launcher and entrypoint correctly
materialize, attest, create, and restart a SparkCache-enabled TP2/DCP1 stack for
an explicitly research-scoped checkpoint composition.

## Limitations

- The checkpoint revision and complete-manifest digest differ from the
  qualified recipe. This run does not reproduce or transfer that qualification.
- The upstream SparkCache tag has no TP2 profile launcher; the tested launcher
  is the SparkRing implementation identified above.
- One cold/restart/hit cycle is not production reliability or a performance
  matrix.

Raw-record location: unavailable in a public artifact; exact responses,
inspections, logs, and manifests remain in the private validation workspace.
The gate ran once, so no repeatability distribution or uncertainty estimate is
available.
