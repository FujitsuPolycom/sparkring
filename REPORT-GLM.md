# Report: GLM-5.3 model-side speedup plugins on the all-in-one image

Branch `claude/glm53-plugins`, local commits only (no push, no publish).
Target image `sparkring-dev/kraken:csf-sircl-libsircl-20261008`
(`sha256:816c6d6a…`), whose Python sources are vLLM `bc9ea774` over the
compiled build `0.1.dev21553+gab86b7073` and B12X `cc36aa6f`. Every image
file named below was read from that image only through the read-only
`docker run` on one Spark of the ring, and every SHA-256 quoted was computed inside
it. No run touched a GPU; no running container was touched; SIRCL was not
edited; nothing outside `glm53full-clean` was read from the earlier
workspace.

## Per-plugin disposition

| Earlier plugin | Disposition | Reason |
|---|---|---|
| `glm_dsa_indexer_split` | **Ported** as 1.1.0 (`integrations/vllm/glm_dsa_indexer_split/`) | Its two wrapped methods survive in the image's `b12x_indexer.py` (`e1841ab0…`), and every other file it relies on is unchanged or re-pinned below. |
| `glm53full_speedups` | **Ported** as 1.1.0 (`integrations/vllm/glm53full_speedups/`) | Both patch anchors survive: `DeepseekV32Attention.__init__` builds `fused_qkv_a_proj` with `DeepSeekV2FusedQkvAProjLinear` (image `attention.py` `c1358e67…`, line 299), and `DeepseekV32MultiTokenPredictorLayer.__init__` builds `eh_proj` as a replicated `nn.Linear` (image `mtp.py` `103516ab…`, line 78). |
| `glm53full_backports` | **Dropped: already upstream in the image** | All four items are in the image's own sources; file and line citations below. Porting it would patch nothing and its pins would all refuse. |
| `glm_dcp_prefill_q_replicate` | **Dropped: covered natively in the image** | The image's own full-CKV route and new mixed-batch route remove the prefill query exchange for every batch regime the plugin ever measured; the two environment switches are in "Query replication" below. Porting it would also have to rewrite its eligibility for the image's new route (its query-gather replacement refuses the shapes the mixed route gathers). |

The backports, each against the image's own sources:

1. **DSA merge `auto`** (earlier: b12x `b87a1e1`): `b12x/attention/dsa_indexer/_tuning.py`
   (`15b33511…`) line 165 already restricts the tuning space to
   `"fused_merge": (FUSED_MERGE_AUTO,)` and line 184 already sets
   `candidate_contract_version=5`.
2. **Load split** (earlier: vLLM `f2d1272`): `vllm/v1/worker/gpu_worker.py`
   (`8ce5974c…`) `_scoped_allocator_max_split` already yields without
   `max_split_size_mb` when `PYTORCH_CUDA_ALLOC_CONF` enables
   `expandable_segments:True` (lines 381-385).
3. **DSA shared plans** (earlier: vLLM `d8caed0`):
   `vllm/v1/attention/backends/mla/b12x_indexer.py` `_declare_plan` already
   returns `replace(self._module.plan(...), shared=True)` (lines 621-631).
4. **Threads before warm-up** (earlier: vLLM `b782ed1`): `gpu_worker.py`
   `_compile_or_warm_up_model_after_preparation` already calls
   `set_torch_threads_for_runtime()` before any warm-up (line 1018, "Drop to
   the serving thread count before any warmup").

## Files changed

```
integrations/vllm/glm_dsa_indexer_split/
  README.md                        plugin description, settings, pins, test setup
  glm_dsa_indexer_split/__init__.py  wrapper install, pins (1.1.0)
  glm_dsa_indexer_split/layout.py    row partition and launch cutting (unchanged)
  glm_dsa_indexer_split/runtime.py   split path; adds the key_gather fallback
  dist-info/glm_dsa_indexer_split-1.1.0.dist-info/{METADATA,entry_points.txt,top_level.txt}
  tests/{split_image_env.py,conftest.py,harness.py,register_probe.py,
         test_pins.py,test_register.py,test_layout.py,test_split_exact.py,
         test_fallbacks.py,test_verify.py}
integrations/vllm/glm53full_speedups/
  README.md
  glm53full_speedups/__init__.py     code patches, helpers, pins (1.1.0)
  dist-info/glm53full_speedups-1.1.0.dist-info/{METADATA,entry_points.txt,top_level.txt}
  tests/{speedups_image_env.py,conftest.py,register_probe.py,test_patches.py,
         test_registration.py,test_latent_shard.py,test_eh_proj.py}
runtime/images/derive_glm53_plugins.py     the thin image layer (additions only)
runtime/images/test_derive_glm53_plugins.py
```

Commit `1776f8d8` holds all of the above. `TASK-GLM.md` and this report are
uncommitted working files of the worktree.

### Pins

The plugins pin files by the SHA-256 computed inside `816c6d6a7e96`. Of the
30 files the two plugins record, 21 are byte-identical to the pins the
earlier build recorded; 9 changed with the image's merged sources and were
re-pinned from the image:

| File | SHA-256 in `816c6d6a7e96` |
|---|---|
| `vllm/v1/attention/backends/mla/b12x_indexer.py` | `e1841ab03516e9f58e52400491ed5a11db7826c4a676f49c427a9e6246210ae2` |
| `vllm/models/deepseek_v32/attention.py` | `c1358e677002658c848247758c45ef5543eec9c5122bc3b36e2e5c4447c68bd6` |
| `vllm/models/deepseek_v32/nvidia/mtp.py` | `103516ab5c596f9851172fdaf86c67cb5435d1ca2781fc80268d5d61dbcb3456` |
| `vllm/models/deepseek_v32/nvidia/model.py` | `03c85e81a225a0de3312bf62830ff836c8f0c9d894e7fee4381e5510f366cd10` |
| `vllm/models/deepseek_v32/nvidia/b12x.py` | `2412da9bec4b3597064395c4c78b0d93c1235153fe3e55df4b9dba6dcde4942d` |
| `vllm/model_executor/models/deepseek_v2.py` | `112a2d92e4f80a2e9a3baa567571f0b22c491bf8c18a1dbba2e9259e9b36692a` |
| `vllm/v1/worker/gpu_worker.py` | `8ce5974c30bfff30fb69cd080d8e68dfaf98ce431127687d2e2128c05baa5948` (named only in the dropped backports) |
| `b12x/attention/dsa_indexer/_tuning.py` | `15b335118ec469eb35cceed63af618a1463f0c2f2cb5e16f10b20eff2c7e7b27` (dropped backports only) |

One behavioral change was needed for the split: the image's
`B12xSparseIndexer.forward` gained a gathered-key prefill path
(`_run_gathered_prefill`, selected by `VLLM_DCP_INDEXER_KEY_GATHER`, default
`0`). That path does not score rows per DCP group, so the split's selections
would not be the image's; the ported plugin sends every prefill step of an
indexer built with the variable set to the image's `forward` and counts it as
a new `key_gather` fallback. The flags, the off-by-default state and the
refusal paths are unchanged: every plugin still refuses any other file
version with `PatchRefused` naming the file and what it relies on there.

## Tests

All CPU tests replay the plugins against the target image's own source
files. `SPARKRING_GLM53_IMAGE_SOURCES` names a directory holding the image's
`vllm/` and `b12x/` trees, extracted read-only from `816c6d6a7e96`; without
it the suites fail with the extraction instructions, like a hardware test
without the hardware. Run from the worktree root with a CPU torch:

```bash
SPARKRING_GLM53_IMAGE_SOURCES=<image-sources> \
  python -m pytest integrations/vllm/glm_dsa_indexer_split/tests \
                   integrations/vllm/glm53full_speedups/tests \
                   runtime/images/test_derive_glm53_plugins.py -q
```

Counts (pytest, one CPU process, torch 2.14.1+cpu; the three suites also
pass together in one run: 386 passed in one combined invocation):

| Suite | Tests | Covers |
|---|---:|---|
| `glm_dsa_indexer_split/tests` | 359 | All pins replayed; settings and their refusals; the row partition, all-gather layout and launch cutting; the split against the image's own `forward` on eight thread-emulated ranks — exact index sets on every rank, rows beyond the step untouched, ranks agreeing, prefill rows equal to an independent global top-k, and the kernel-call and all-gather accounting; the fallbacks (decode, key-gather, mixed, short, layout, shape, unprepared, capture) with construction-time refusals; verify mode with a jittering reference kernel (tie-only differences counted apart, image result kept, budget enforced, a wrong split caught); registration in fresh interpreters (flag combinations, import-hook install, every refusal path). |
| `glm53full_speedups/tests` | 21 | Every pin replayed; both patches prepare against the image's files (edit once, line count kept, compiled original equals the whole-file compile); install idempotence and its refusals; the latent shard on eight thread ranks — byte-exact weight loading per rank, identical gathered output on every rank, and the gathered forward equal to the fused projection within GEMM reduction order; the row-parallel `eh_proj` — exact per-rank columns and the reduced forward equal to the replicated linear; registration and refusals in fresh interpreters. |
| `runtime/images/test_derive_glm53_plugins.py` | 6 | The layer's file list, pins, entry points and the plugins' image pins. |

## Packaging

`runtime/images/derive_glm53_plugins.py` is a code layer on the repository's
generic derived-layer path (`runtime/images/derived_layer.py`). It adds ten
files — the two plugin packages and their dist-info directories — and nothing
else: it never edits a `vllm` or `b12x` file, so the image's entrypoint
(`toolchain.py serve` → `external-base.py verify`) keeps passing, and the
layer records every added file in the external-base receipt so `verify`
checks the plugins' bytes too. The layer writes the provenance receipt
`/opt/sparkring/receipts/derived-glm53-plugins.json`.

The plugins load through vLLM's `vllm.general_plugins` entry points and only
when `VLLM_PLUGINS` names them; each item stays off until its own flag is
set, so adding both plugin names to `VLLM_PLUGINS` without the flags changes
nothing.

### Build commands (for the lead; nothing here builds on a Spark)

Run on the image build host, from a copy of the worktree at this revision,
with the target image `816c6d6a7e96` loaded and the v3 lock of
`csf-sircl-libsircl-20261008` at `$W/glm53-parent-lock.json` (the lock the
libsircl build wrote):

```bash
W=~/sparkring-image; cd $W/sparkring
PARENT_LOCK=$W/glm53-parent-lock.json

# 1. Build context from the parent's lock and receipts (reads the parent's
#    files from the local parent image; add --base-receipt/--toolchain-receipt
#    with copied receipts instead, if the parent image is not local).
python3 runtime/images/derive_glm53_plugins.py prepare \
  --parent-lock $PARENT_LOCK --output $W/glm53-plugins-context

# 2. Build, probe and admit; writes the derived v3 lock.
python3 runtime/images/derive_glm53_plugins.py build \
  --context $W/glm53-plugins-context \
  --tag sparkring-dev/kraken:csf-sircl-libsircl-plugins-20261008 \
  --name dev-20261008-kraken-csf-sircl-libsircl-plugins \
  --output $W/glm53-plugins-lock.json
```

`build` tags the parent, builds the context and records; it never pushes or
publishes. Distribute to the other Sparks the way the libsircl layer was
distributed (a `layer_delta.py` delta archive against the parent, loaded on
each Spark that holds the parent image).

## Launch settings: one GLM-5.3 TP8 start with every ported plugin on

Install the derived image with the eight-Spark profile, then set the plugins
and flags in the profile's environment (one `config.json` edit; every rank
must see the same values):

```bash
sudo sparkring install --profile glm53-nvfp4-tp8 --image-lock $W/glm53-plugins-lock.json
```

Environment additions to `profiles/glm53-nvfp4-tp8/config.json` (the profile
already serves GLM-5.3-NVFP4 `b472e4ee` at TP8, DCP 4, 1M context, 8,192-token
prefill chunks, `--load-format b12x`, on SIRCL ring sessions):

| Variable | Value |
|---|---|
| `VLLM_PLUGINS` | `b12x_loader,glm_dsa_indexer_split,glm53full_speedups` |
| `GLM_DSA_INDEXER_SPLIT` | `1` |
| `GLM_DSA_INDEXER_SPLIT_FULL_LAUNCHES` | `1` |
| `GLM53FULL_LATENT_SHARD` | `1` |
| `GLM53FULL_EH_PROJ_TP` | `1` |
| `VLLM_B12X_MLA_CKV_GATHER` | `1` |
| `VLLM_B12X_MLA_CKV_GATHER_MIXED` | `1` |

The `vllm_args` stay as the profile pins them. The earlier build measured
each plugin's gain separately (indexer split: prefill +8-9% at 16K/64K/128K,
decode unchanged; latent shard and `eh_proj`: in the reference build), so no
per-plugin A/B is planned; the per-plugin flags exist only for bisecting if
this start fails or its outputs diverge. If a start fails, first retry with
`GLM_DSA_INDEXER_SPLIT_VERIFY=8` (`GLM_DSA_INDEXER_SPLIT_VERIFY_MIN_CONTEXT`
as desired), which computes the first eight eligible layer-chunks with the
split and twice with the image's path and logs the comparison per rank; every
plugin also logs one line at startup naming what it patched, and its pins
refuse with the file name if the image's sources differ from the values
above.

Unresolved for serving qualification: the plugins' CPU tests cover the pins,
the mechanics and the exactness of loading and selection against the image's
own sources; they do not qualify serving. The indexer split's selections
remain equal to the image's only up to candidates tied at a row's lowest
selected score, and the verify mode above is the on-ring qualification step.

## Query replication

The earlier build's `glm_dcp_prefill_q_replicate` 1.0.0 (clean copy
`q_replicate/site/`) attacked the prefill query exchange from the query side.
After each model's weights are processed it all-gathers, over the DCP group's
CPU (gloo) process group, every member's processed `q_b_proj` (the packed
MXFP8 weight) and `W_UK_T` — 0.44 / 1.33 / 3.11 GiB per rank at DCP 2 / 4 / 8
over the 79 attention modules of target and MTP draft — and for an eager batch
of at least 512 query rows (`GLM_DCP_PREFILL_Q_REPLICATE_MIN_ROWS`, default
512, above the 24-row decode-graph ceiling) with at least one prefill row on
the sparse-MQA path, each rank computes the whole group's absorbed queries
(each member's `q_b_proj` GEMM, `W_UK_T` BMM and `fused_q` RoPE, into one
`[rows, 8 d, 576]` BF16 tensor in group order) and hands them to the image's
`_sparse_indexer_and_attn` through a one-shot replacement of
`dcp_manager.query_gather`. That is the tensor the all-gather would have
returned, so attention and the combine are unchanged; a prefill chunk's query
shard is 72 MiB per rank, layer and 8,192-row chunk. Decode-only
batches and every CUDA-graph capture keep the gather. Its eligibility yields
to the full-CKV gather (`impl.uses_full_ckv_dcp`), so on the earlier build it
was recorded as that gather's complement — mixed batches and batches above the
gather's capacity — and it composed with it. The earlier build's A/B measured
+3.7 / +15.1 / +13.1 % 16K prefill at DCP 2 / 4 / 8; the clean copy's
`STATUS.md` holds the pre-A/B cost model (+4.7 / +15.5 / +13.9 %, from
isolated gather microbenchmarks) and no GPU run.

**The image covers it natively, by gathering the cache instead of the
queries** — the trade of the earlier build's `glm_dsa_ckv_gather` plugin, now
in the image's own sources and extended to mixed batches. Evidence, all in
`816c6d6a7e96`:

1. **Full-CKV prefill route.** `b12x_mla_sparse.py` (`28ba2439…`)
   `_use_b12x_full_ckv_gather` (lines 223-244) selects eager pure-prefill
   batches (`num_decode_tokens == 0`, not speculative, `max_query_len > 1`,
   not a CUDA-graph capture, `b12x_mla_sparse.py` 1015-1028) with DCP above 1
   and a batch token count inside
   [`VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS`, `VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS`]
   (defaults 16 and 524,288, image `envs.py` 234-237 and 1931-1945). Every
   8,192-token prefill chunk of the serving profile is inside that window.
   `uses_full_ckv_dcp` (2022-2034) confirms it per call, again refusing
   captures. A route batch attends its local heads over the DCP group's
   gathered cache: no query gather, no LSE combine.
2. **Mixed-batch route, new in this image.** `_use_b12x_split_ckv_gather`
   (lines 247-272) selects eager mixed batches (decode rows and prefills both
   present) on the same token window over their prefill tokens, when
   `VLLM_B12X_MLA_CKV_GATHER_MIXED` is set (requested at 790-794). The
   attention layer routes them to `_split_ckv_dcp_mqa`
   (`models/deepseek_v32/attention.py` `c1358e67…`, lines 514-560, selected at
   675-685): the decode rows — ordered first — keep the query gather and the
   LSE combine (537-544, tens of microseconds), the prefill rows attend their
   local heads over their own requests' gathered cache through
   `forward_mqa(..., route="ckv_extend")` (551-558), and none of the prefill
   rows' queries or outputs move. The method's own docstring: the prefill
   cache gather "moves far fewer bytes than exchanging every head's queries
   and outputs". This is exactly the plugin's mixed-batch domain.
3. **GLM-5.3 is in scope for both routes.** The backend recognises the model
   (`_GLM_DSA_MODEL_TYPES = frozenset(("glm_moe_dsa",))`, line 63; set at
   752) and both predicates take `is_glm_dsa` (238, 266).
4. **No native query-replication switch exists for this model.**
   `DeepseekV32Attention` builds `q_b_proj` as a plain `ColumnParallelLinear`
   and passes no `dcp_q_replicate` (`attention.py` 306-311, 253-270), so
   `VLLM_DCP_Q_REPLICATE` / `parallel_config.dcp_q_replicate` reach only
   `DeepseekV2MLAAttention` (`model_executor/models/deepseek_v2.py`
   1076-1085, `mla_attention.py` 614 and 1612-1615) and stay inert for
   GLM-5.3 — as on the earlier image. The coverage is the CKV-gather family,
   not query replication.
5. **The residue.** A batch whose gathered cache exceeds the per-rank
   capacity guard keeps the image's query exchange: eligibility is granted
   only while `padded_total_tokens <= max_local_capacity` with
   `max_local_capacity = round_up(ceil(MAX_TOKENS/d) +
   max_reqs × cp_kv_cache_interleave_size)` (`b12x_mla_sparse.py` 1078-1090;
   the flags stay at their `False` initialisers, line 730). With the defaults
   that is roughly 524,288 cached tokens summed over the batch's prefill
   requests at any DCP size, so prefill chunks into requests cached beyond
   about half a million tokens fall back. That is the plugin's one remaining
   domain, and it is unmeasured on every build: the earlier A/B used 16K
   prompts, the image's own routes have never been measured on the ring above
   16K, and the plugin never ran on a GPU at all.

**Disposition: dropped — use the image's switches, port nothing.** The image's
routes cover every batch regime the plugin ever measured, and in the shared
one (16K pure prefill) the earlier build measured the cache gather well ahead
of query replication (+5.3 / +59 / +79 % at DCP 2 / 4 / 8 — 4-bit KV at DCP 2,
8-bit at DCP 8 — `glm_dsa_ckv_gather` A/B, against the plugin's
+3.7 / +15.1 / +13.1 %). A like-for-like port is not
possible either: the plugin's query-gather replacement requires the full local
`[rows, 8, 576]` query and raises on any other shape (clean copy
`runtime.py` 476-485), while the image's mixed route calls `query_gather` with
the decode rows only (`attention.py` 537) — a port would have to add
`uses_split_ckv_dcp` and a mirror of the capacity guard to the eligibility,
i.e. new eligibility code on an unmeasured regime, at 1.33 GiB per rank of
replicated weights (DCP 4), for a plugin that never left research-only status
anywhere. If above-capacity prefill ever matters on the ring, the image's own
`VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS` (default 524,288; the per-rank gathered
workspace scales with it, `b12x_mla_sparse.py` 1083-1089 and 1708) is the
first knob to test, before any port. `VLLM_DCP_INDEXER_KEY_GATHER`
(`b12x_indexer.py`) is the DSA indexer's own prefill key gather; it does not
touch the attention query exchange and is already handled in the indexer
split's port above.

**Tests.** Nothing was ported, so no tests were added; the ported suites keep
their counts (359 + 21 + 6). The clean copy holds 25 of the plugin's 39 files;
its own CPU suite was 54 tests on the earlier image (`STATUS.md` section 4),
and the copied parts cannot run here: `tests/test_plugin.py` (the wrapper
installation tests, 12) was withheld as transport, `test_replication.py` (4)
and the cost model `qrep_budget.py` behind `test_qrep_budget.py` (18) are
absent, and `test_prefill_path.py` (15) and `test_source_facts.py` (5) replay
the earlier image's sources, not this one.

### Launch setting for query replication: GLM-5.3 TP8/DCP4

Set both switches in the profile's environment (table above); nothing else
changes — no plugin name joins `VLLM_PLUGINS`, and no image layer is rebuilt
for this task:

| Variable | Value |
|---|---|
| `VLLM_B12X_MLA_CKV_GATHER` | `1` |
| `VLLM_B12X_MLA_CKV_GATHER_MIXED` | `1` |

Both default to 0 in the image (`envs.py` 234-237). With them on, the backend
allocates its selection-index and DCP bookkeeping buffers at startup
(`b12x_mla_sparse.py` 795-815) and the gathered-cache workspace of the
`ckv_extend` plan (1708) — tens of MiB per rank, sized from the batch limits
and the gather token window; watch the startup memory margin on the first
launch. `VLLM_B12X_MLA_CKV_GATHER_MIXED=0` alone leaves pure prefill covered;
the mixed switch is the piece the earlier plugin would have served. These
routes' gains above 16K contexts are unmeasured on the ring; the first serving
run is their qualification, as for the ported plugins.
