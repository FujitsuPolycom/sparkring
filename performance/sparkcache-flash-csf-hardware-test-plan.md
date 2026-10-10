# Hardware test plan: SparkCache on GLM-5.3-Flash CSF, TP4 first then TP2

Status: **research-only plan**. Nothing in it has run. The owner runs it on the
cabled Sparks; this document plans the runs and their receipts. It follows the
R33 TP4/DCP4 SparkCache steps
(`performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md`)
and the SparkCache checks of `docs/operations/profile-validation.md:192-224`.
Every run uses one image, one checkpoint, fixed settings and the namespace
recorded in its section.

## Preconditions

- The derived image of
  [runtime/releases/dev-20261010-sparkcache-gab86b70-plugins-status036/](../runtime/releases/dev-20261010-sparkcache-gab86b70-plugins-status036/README.md)
  is built and recorded on one Spark (`runtime/images/derived_layer.py`,
  [layer README](../runtime/images/compositions/installer-sparkcache-gab86b70/README.md)),
  and every Spark of the ring runs that image.
- The external SparkCache CPU conformance for the lease contract passed
  (`sparkcache/runtime_patches/test_gab86b70_contract.py` against the image's
  vLLM sources). CPU conformance does not qualify CUDA behaviour.
- The CSF checkpoint
  `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd33ef` is
  pinned on the node (`profiles/checkpoints/…/dec48abd33ef…json`), and each
  Spark has the blank-Spark free space of the base profile plus the 8 GiB
  `spark_cache_max_bytes` allowance of the variant.
- Reference numbers for the speed step are the cache-off CSF measurements of
  release 2026.10.2's image: TP4 64.2 / 126.6 / 218.7 tokens/s and TP2 39.5 /
  87.1 / 123.2 tokens/s at 1 / 4 / 8 streams at 32K
  (`README.md:62-65`).

## Stage 1 — TP4 (`glm53-flash-nvfp4-sparkcache-tp4`)

Namespace for every receipt: image
`dev-20261010-sparkcache-gab86b70-plugins-status036`, profile
`glm53-flash-nvfp4-sparkcache-tp4`, checkpoint
`dec48abd33efa73c3bb7c95b74eee10cad34f9be`, cache root
`/cache/glm53-flash-sparkcache-ab53c097d7ca-dec48abd33ef`, lease contract
`vllm-connector-jobs-gab86b70.json`, model profile `glm53-flash-hybrid`,
lease mode `connector-jobs`.

1. **Image probe, no GPU.** On one Spark: read the SparkCache entries of
   `/opt/sparkring/receipts/external-base-installed.json` (the four layer files
   under `dist-packages/sparkcache/` and the native libraries under
   `/opt/sparkring/sparkcache/lib/`), import `sparkcache`, and run
   `verify_manager_page_lease_contract` against
   `/usr/local/lib/python3.12/dist-packages/sparkcache/runtime_patches/vllm-connector-jobs-gab86b70.json`
   with `spark_cache_async_page_capture_vllm_root=/usr/local/lib/python3.12/dist-packages`.
   Pass: every hash matches the receipt and the contract verifies.
2. **Cold start with an empty root.** Every rank logs SparkCache's config line
   with DCP degree 1 and read-write access; all ranks report healthy. Record
   the connector state that `sudo sparkring check` prints.
3. **Exact answers.** `sudo sparkring check` (7 functional checks) and three
   temperature-0 known-answer requests.
4. **Cold prefill.** Two cold prompts of about 37K tokens; each reports
   `cached_tokens` = 0 with no `sample_tokens` timeout, and the logs show page
   snapshots committed on every rank.
5. **Prefix hits.** Repeat one prompt with an identical 32K prefix; record
   cached tokens against the aligned span and the wall-time change. Then the
   six suffix cases 515, 513, 531, 1027, 512 and 2048 tokens; each needs an
   exact answer.
6. **Planned restart restore.** A writer stores an 8K–16K-token entry on every
   rank; `sudo sparkring down --execute` then `up --execute`; the reader
   restores it on every rank. Record tokens, milliseconds and MiB per rank;
   API `cached_tokens` must equal the physical restore on each rank.
7. **SIGKILL restore.** `docker kill` every rank's container, then
   `sparkring up`, with automatic recovery confirmed not to have acted first
   (`docs/operations/install-reference.md:1177-1199`). The reader restores on
   every rank.
8. **Restore fixture.** `needle_hunt.py` before and after restart, three
   seeds, at contexts within the configured 64K span.
9. **Failure path.** Corrupt a synthetic run-owned chunk
   (`runtime/images/upgrades/cache_fault.py`); the request must recompute with
   a correct answer.
10. **Receipts.** Per rank: logs, SparkCache counters, process identities and
    this plan. Write the record under `performance/records/images/` with the
    five record sections.
11. **Speed.** Install `glm53-flash-nvfp4-spark-tp4` (cache off) and the
    variant on the same ring, image and checkpoint, two starts each. Measure
    the serving A/B runner's `phase1-16k` metrics through the API: 32K
    context, 1 / 4 / 8 streams, temperature 0, at most 1,024 output tokens,
    30 s cells, and the cold TTFT median of 3 at 8K / 16K / 32K. Decode prompts
    get unique prefixes so cells measure decode, not restore; restored-prefix
    TTFT is reported separately.

## Stage 2 — TP2 (`glm53-flash-nvfp4-sparkcache-tp2`)

Steps 1–11 again on a pair, with the profile's `--async-scheduling` kept on.
Step 7 is the first SparkCache run with asynchronous scheduling at TP2 and the
2,048-token split pages; a failure there is a finding, not a blocker for
Stage 1's record.

## Qualification language

A stage whose steps 1–9 pass on every rank marks the composition `implemented`
on that image and checkpoint for the spans it ran; `qualified` follows
maintainer release qualification. Spans beyond 64K tokens and 1M-token entries
stay unqualified until their own measured runs; full-context restore is not
qualified on any image today.
