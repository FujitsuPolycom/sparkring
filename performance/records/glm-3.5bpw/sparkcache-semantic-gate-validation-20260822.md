# GLM SparkCache final-content semantic gate validation

Lane: **public-functional**. Status: **research-only**. Evidence scope:
**live-validated predecessor artifact; offline-validated published artifact**.
Hardware: four directly cabled NVIDIA
DGX Spark systems, TP4/DCP4. Evidence scope: one post-restart hit against the
qualified SparkCache `0.1.0a2` operator-image composition.

## Artifact identities

The live run used `scripts/sparkcache_glm_semantic_gate.py` at SHA-256
`f17fae23035aa7aaf5594e2585a68e95db09226dc97906b6b9d882a0c5fca205`.
That artifact required exact final content and finish reason while recording
reasoning/body equality as diagnostic-only.

The published script has SHA-256
`9cbf484968f40638736522373fb53a94f7975625d01a4c7014b7c7b7bdeae9f3`.
Its executable gate conditions are unchanged. The byte difference renames the
arithmetic response field from an overbroad quorum term to
`inventory_publication_request` and documents that all-rank evidence comes
from metrics and logs. The published bytes pass offline regression tests; they
have not run a separate live hit.

## Conditions

- Serving image ID:
  `sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513`.
- SparkCache wheel SHA-256:
  `3345b8c574951a8204377b0c27f53765c84b96ab4f5a8ec1ac147574dba7568b`.
- Legacy reference schema:
  `sparkcache-deepseek-semantic-reference/v1`, 12,000 archive records,
  expected final content `SPARKCACHE_OK:9540`.
- The gate sent the arithmetic inventory-publication request before the long
  hit, then sent a fresh post-restore canary.

## Measurement

Raw-record location: unavailable in a public artifact; response, metric, and
per-rank log artifacts remain in the private validation workspace. The gate ran
once, so no repeatability distribution or uncertainty estimate is available.

The hit returned:

| Condition | Result |
|---|---|
| Inventory-publication response | `2` |
| Long final content | `SPARKCACHE_OK:9540` |
| Finish reason | `stop` |
| Post-restore canary | `SPARKCACHE_CANARY_OK` |
| Combined assistant-body equality | `false`, diagnostic-only |

Separate vLLM metrics reported 225,607 external-cache queried tokens, 225,536
external-cache hit tokens, and zero native-prefix hit tokens for the known
restore. Scheduler and worker logs reported the same 225,536-token digest on
all four physical ranks. Those surfaces, not the arithmetic response alone,
established all-rank external restore.

## Result

Exact final content, finish reason, inventory-publication response, canary,
external-hit metrics, and all-rank logs passed. Combined assistant-body
equality was false and remained diagnostic-only.

## Conclusion

The final-content equivalence policy is live-validated for the predecessor
artifact identified above. The published terminology-corrected artifact is
offline-validated and retains the same semantic pass/fail conditions.

## Limitations

Raw-record location: unavailable in a public artifact; response, metric, and
per-rank log artifacts remain in the private validation workspace. The gate ran
once, so no repeatability distribution or uncertainty estimate is available.
