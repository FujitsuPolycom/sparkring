# Clean-room acceptance audit of SIRCL 0.3.2, the GLM-5.3 plugins and libsircl

Status: **record of verdicts**. The acceptance audit checks that SparkRing's
own sources hold no text derived from the sources that SparkRing's
clean-room rule keeps implementers out of. Its report stays outside the
repository, because it quotes that excluded-origin text; this record states
only the audited trees, the verdicts and their scope.

## Conditions and verdicts

| Source | Audited tree | Verdict | Lines to rewrite |
|---|---|---|---|
| SIRCL 0.3.2, `spark_transport/sircl` | commit `1273183e`, tree `8a54c1abf591` | PASS; 522 counted idioms | 0 |
| `glm_dsa_indexer_split` 1.1.0, `integrations/vllm/glm_dsa_indexer_split` | commit `cf478504`, tree `732600626129` | PASS | not stated |
| `glm53full_speedups` 1.1.0, `integrations/vllm/glm53full_speedups` | commit `cf478504`, tree `5ba140c1d908` | PASS | not stated |
| `glm_dcp_decode_comm`, `integrations/vllm/glm_dcp_decode_comm` | commit `cf478504`, tree `f3bc9bf7ef29` | PASS | not stated |
| libsircl 0.6.0, `spark_transport/libsircl` | commit `c7c35fe0`, tree `dbf3607484dd47df4cf6c8238eb5b3272466effb` | PASS on 113 files; 512 counted, all idioms | 0 |
| libsircl 0.6.0 with the current-device change, `spark_transport/libsircl` | commit `d3d33158`, tree `030419b8a2b61acc1a74010308a1c3e758863e2f` | PASS on 114 files; 515 counted, all idioms (the 3 lines more than tree `dbf36074` are generic POSIX mutex calls in `src/engine.c`) | 0 |
| vLLM general plugin `libsircl`, `integrations/vllm/libsircl` | commit `c7c35fe0`, tree `b6bdd22755bc` | PASS | 0 |

"Counted idioms" is a count the audit reports for SIRCL and libsircl; its
definition is in the audit's report. The verdicts of the three GLM-5.3
plugins state no count of lines to rewrite.

## Scope against the images of release 2026.10.2's evidence

Image `1a8c10354eb0` was built from commit `c7c35fe0`; release 2026.10.2's
image, `d52737a109e0`, from commit `d3d33158`
([release record](../../../runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md)):

- `spark_transport/sircl`, `glm_dsa_indexer_split` and `glm53full_speedups`
  have the audited trees at `c7c35fe0`.
- `glm_dcp_decode_comm` at `c7c35fe0` is version 2.0.1 (tree `6e0bb0e46758`).
  It differs from the audited tree only in its version: the version strings
  of its README, `__init__.py` and dist-info `METADATA`, and the dist-info
  directory's name.
- Image `1a8c10354eb0`'s libsircl is the audited tree
  `dbf3607484dd47df4cf6c8238eb5b3272466effb`, and its vLLM plugin `libsircl`
  is the audited `integrations/vllm/libsircl` of `c7c35fe0`.
- Image `d52737a109e0` has the same SIRCL, plugin and `integrations/vllm/libsircl`
  trees. Its libsircl is the audited tree
  `030419b8a2b61acc1a74010308a1c3e758863e2f`, tree `dbf36074` with the
  current-device change (commits `c4f3a1af` and `05458d35`).
  `integrations/vllm/libsircl` and libsircl's NCCL files are byte-identical
  to tree `dbf36074`'s and were not measured again.

## Limitations

- The audit compares text; it does not establish license compliance of
  other inputs, which [THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md)
  records.
- Sources changed after the audited trees need another audit.
