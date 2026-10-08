# Collectives vLLM issues while serving, and where SIRCL carries them

This survey lists every collective that vLLM issues, in the build of serving
image `aba309e4610c`, for three model configurations. For each one it records
where the collective is dispatched and what SIRCL's vLLM adapter
([`README.md`](README.md)) does with it. It is a static reading of source code;
it records no observation on hardware.

The adapter lets a tensor-parallel group run on Sparks that do not all share
cables. With `SIRCL_NCCL=never`, the adapter's default, NCCL runs no
collective on any multi-rank group. With `SIRCL_NCCL=auto`, NCCL runs only
where a group's cabling allows it.

## Scope and conventions

- **Source tree.** The vLLM Python of serving image `aba309e4610c`: the Local
  Inference Lab fork, `vllm 0.1.dev21553+gab86b7073.d20261001`, pinned as
  build `lil-image-aba309e4610c` in [`pins.py`](pins.py). All paths are
  relative to its `vllm/` package directory, and line numbers refer to that
  tree. `pins.py` records the hook-file hashes of the other pinned builds,
  and a CPU test checks the anchors of [`hooks.py`](hooks.py) in every tree
  it is given.
- **Models:**
  - GLM-5.3-Flash at tensor-parallel size 4 (TP4). Model type `glm5_next`:
    hybrid Kimi delta attention (KDA) linear attention and DeepSeek sparse
    attention (DSA) over multi-head latent attention (MLA), mixture-of-experts
    (MoE) layers, a one-layer multi-token-prediction (MTP) draft head, and B12X
    kernels.
  - Qwen3.8-Flash-Next at TP4.
  - GLM-5.3 at TP8 with decode context parallelism of 4 (DCP4). Model type
    `glm_moe_dsa`.
- **`none` group.** A group on which NCCL may run no collective: every
  multi-rank group under `SIRCL_NCCL=never`, and under `SIRCL_NCCL=auto` a
  group with a consecutive pair of ranks that shares no cable, such as four
  consecutive Sparks of an eight-Spark ring (README section "NCCL on groups
  whose Sparks do not all share cables").
- **Notation:**
  - T: tokens of a forward pass, padded to the captured graph size.
  - L: logits rows.
  - R: sequences in a step.
  - k: speculative tokens per step.
  - N: decoder layers.
  - D: the TP session's dispatch ceiling.
  - "bf16": bfloat16.
- **Default runner and graphs.** The V2 model runner (`v1/worker/gpu/`) is the
  default on CUDA (`config/vllm.py:840-889`). CUDA graphs are on.

## 1. How a collective reaches a transport

| Step | Where |
|---|---|
| Model code calls `tensor_model_parallel_all_reduce` / `_all_gather` / `_reduce_scatter` / `_gather` (all on the TP group) or a `GroupCoordinator` method | `distributed/communication_op.py:12-58` |
| On CUDA, all-reduce, all-gather and reduce-scatter become the custom ops `torch.ops.vllm.*`, which look the group up by name | `platforms/cuda.py:751-753`; `distributed/parallel_state.py:260-305, 460-476, 811-812, 841-846, 872-877` |
| The coordinator calls its device communicator | `parallel_state.py:827-830` (all-reduce), `:848-851` (all-gather), `:853-861` (all-gatherv), `:879-889` (reduce-scatter, reduce-scatterv), `:891-905` (gather), `:1447-1461` (send, recv) |
| The in-place all-reduce used by the MoE runner | `parallel_state.py:816-825` → `cuda_communicator.py:514-523` (PyNccl in place) |
| The device communicator class comes from the platform | `parallel_state.py:611-621` (`current_platform.get_device_communicator_cls()`); `platforms/cuda.py:625-628` |
| `CudaCommunicator.all_reduce` tries, in order: B12X PCIe or RoCE slot (`b12x_ar_comm`), NCCL symmetric memory, quick reduce, FlashInfer PCIe IPC, FlashInfer, AITER, custom all-reduce, torch symmetric memory, PyNccl | `distributed/device_communicators/cuda_communicator.py:429-512` |
| All-gather: RoCE slot, NCCL symmetric memory (dim 0 only), PyNccl | `cuda_communicator.py:537-584` |
| Reduce-scatter and reduce-scatterv: PyNccl or NCCL symmetric memory | `cuda_communicator.py:586-674` |
| Gather: `torch.distributed.gather` on the device group | `base_device_communicator.py:318-347` |
| Broadcast through the communicator: PyNccl | `cuda_communicator.py:777-787` |

**What runs on a multi-node GB10 group in vLLM without the adapter.**
Everything ends in PyNccl:

- B12X PCIe all-reduce is single-node only (`b12x_pcie_all_reduce.py:260-262`).
- The RoCE slot is built only with `VLLM_ENABLE_ROCE_ALLREDUCE=1`
  (`cuda_communicator.py:135-137, 193-203`).
- NCCL symmetric memory is off by default.
- FlashInfer all-reduce has no SM12x size table, so it disables itself
  (`flashinfer_all_reduce.py:379-389`;
  `compilation/passes/fusion/allreduce_rms_fusion.py:108-131`).
- Custom all-reduce needs one node or multi-node NVLink
  (`custom_all_reduce.py:173-247`).
- Torch symmetric memory exists only for capabilities 9.0 and 10.x
  (`symm_mem.py:66-79`).

**PyNccl construction.**

- `CudaCommunicator.__init__` calls the module function `_acquire_pynccl`
  for every group of two or more ranks (`cuda_communicator.py:40-64,
  165-173`).
- `PyNcclCommunicator` broadcasts the NCCL unique id over the group's gloo
  group (`pynccl.py:153-157`), calls `ncclCommInitRank` (`:180-182`), and then
  runs a **one-element warm-up all-reduce over the whole group**
  (`:186-191`).
- `VLLM_DISABLE_PYNCCL` skips all of this (`:127-130`).
- vLLM reads its environment live until the worker caches it after loading
  the model (`envs.py:2569-2604`; `v1/executor/multiproc_executor.py:846-848`).

**Groups that get a device communicator.**

- Every group of two or more ranks gets one (`parallel_state.py:611`), except
  the world group, which is built without one (`:1544-1553`).
- Creation order (`parallel_state.py:2142-2345`): TP (`:2151`), engram TP
  (`:2174`, equal to TP for these models), engram DP, DCP (`:2225`), PCP, PP,
  DP, then EP and EPLB.
- **A MoE model always gets an EP group** (`:2288-2321`), even without expert
  parallelism. At DP=1 its ranks are the TP group's ranks. It carries no
  forward traffic at DP=1, but its PyNccl constructor runs the warm-up
  all-reduce, and online quantization of MoE weights reduces weight `amax`
  values over it at startup (section 6).

**Process-group initialization.**

- `torch.distributed.init_process_group(backend="nccl")` creates NCCL
  communicators lazily (`parallel_state.py:1956-1963`).
- With `VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1`, they are created eagerly through
  `split_group` (`:390-425, 1935-1953`).
- Each group also gets an NCCL device group and a gloo CPU group
  (`:563-574`).

**CUDA graph capture.**

- The V2 runner enters `graph_capture()` at `v1/worker/gpu/cudagraph_utils.py:696`.
- `graph_capture()` enters only the TP, PP and DP groups' contexts
  (`parallel_state.py:1718-1745`).
- Each context enters its communicator's `b12x_ar_comm.capture(stream=...)`
  (`:756-763`).
- The target model's logits are computed outside the captured graph
  (`logits_processor.py:303-304`).

**Post-step health check.**

- After `execute_model` (`gpu_worker.py:1517`) and `sample_tokens` (`:1400`),
  the worker calls the TP communicator's `b12x_ar_comm.check_health()`
  (`:1402-1440`).
- No other group is checked.

**Plugins.**

- A worker loads general plugins before it builds any process group
  (`v1/worker/worker_base.py:353-355`).
- vLLM calls them in entry-point discovery order, filtered by `VLLM_PLUGINS`,
  not in `VLLM_PLUGINS` order (`plugins/__init__.py:36-90`).
- A plugin whose import fails is logged and skipped. An exception raised by a
  plugin's function stops the process.
- Platform plugins pass the same filter. One out-of-tree platform plugin
  takes precedence over the built-in CUDA platform (`platforms/__init__.py:233-287`).

## 2. Phases

| Phase | What issues collectives | Traffic |
|---|---|---|
| Process-group and communicator init | `init_process_group`, `new_group` (store); `in_the_same_node_as` (`parallel_state.py:2558-2649`); message-queue broadcaster setup (`shm_broadcast.py:1089-1100`); custom all-reduce probes (`custom_all_reduce.py:173, 232, 447-451`) | gloo or store only |
| | PyNccl unique id broadcast and **warm-up all-reduce** for every multi-rank group (`pynccl.py:153-157, 186-191`) | gloo, then **NCCL** |
| Model construction | GLM-5.3-Flash `mhc_prefill_sharding.py:148` and Qwen `hc_prefill.py:47`: `all_gather_object` on the TP gloo group, unconditional | gloo |
| Load | none on the default loader; online quantization reduces scales on the device group (`model_executor/layers/quantization/utils/quant_utils.py:49-90`) | device |
| Profile run | `_dummy_run` at the maximum batched tokens, dummy sampler (logits all-gather), drafter dummy propose (`v1/worker/gpu/model_runner.py:946-1085`) | device |
| Warm-up | FlashInfer autotune (if FlashInfer and SM≥90): cache exchange on the world gloo group (`model_executor/warmup/flashinfer_autotune_cache.py:102-283`; `kernel_warmup.py:418-439`) and dummy runs; V2 `warmup_kernels` runs real forward + sampling (`v1/worker/gpu/warmup.py:226-493`) | gloo + device |
| Capture | every forward collective of each captured size, inside `graph_capture()` | device (captured) |
| Serving | forward collectives (graph replay or eager) and the eager logits all-gather every step | device |
| KV-cache sizing | no collective: the engine takes the minimum of per-worker RPC results (`v1/core/kv_cache_utils.py:3219-3227`) | — |

## 3. GLM-5.3-Flash at TP4 (`glm5_next`)

**Code path.**

- On CUDA, `models/glm5next/__init__.py` loads `nvidia/model.py` and
  `nvidia/mtp.py`.
- The MTP head drafts through the V2 `MTPSpeculator`
  (`v1/worker/gpu/spec_decode/__init__.py:50-53`).
- GLM is on the breakable-CUDA-graph list, which forces compilation mode NONE
  (`config/vllm.py:131-143, 922-940`). No `torch.compile` collective pass
  runs.
- Activations are bf16 (`config/vllm.py:91`).

**Config** (`transformers_utils/configs/glm5_next.py`):

- vocab 154,880 (`:19`); hidden 4,096 (`:20`); dense intermediate 12,288 (`:22`).
- N=45 (`:23`); 64 heads (`:24`).
- MoE intermediate 2,048 with 288 experts, top-7, and 1 shared expert (`:34-39`).
- MLA kv_lora 512 (`:48-52`).
- One MTP layer (`:54`).
- KDA 64 heads × 128 (`:61-63`).
- Per-rank shards at TP4 (no padding, `model_executor/models/config.py:458-540`):
  vocab 38,720 and expert intermediate 512.

| Collective | Call site | Group | Tensor | When | SIRCL on a `none` TP4 group |
|---|---|---|---|---|---|
| Embedding all-reduce | `models/glm5next/nvidia/model.py:1156`→`:1125`→`model_executor/layers/vocab_parallel_embedding.py:693` | TP | [T,4096] bf16 | every target forward | one op up to D, else the session's large-message all-reduce |
| Attention output projection (MLA and KDA) | MLA `nvidia/attention.py:237`→`model_executor/layers/mla.py:266`; KDA `nvidia/kda.py:363` or `model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py:1770`; both end at `model_executor/layers/linear.py:1934` | TP | [T,4096] bf16, N per forward | every target forward | same |
| MoE output all-reduce (routed + shared summed first) | `model.py:700`→`:348`→`model_executor/layers/fused_moe/runner/moe_runner.py:900-902`→`:540` | TP | [T,4096] bf16 | every MoE layer | same (in-place variant through `all_reduce_in_place`) |
| Dense MLP all-reduce | `model.py:702`→`:212`→`linear.py:1934` | TP | [T,4096] bf16 | dense layers only | same |
| Target logits all-gather | `v1/worker/gpu/model_runner.py:1737`→`model.py:1527`→`model_executor/layers/logits_processor.py:375`→`:281` | TP | [L,38,720]→[L,154,880] bf16, dim −1 | once per step, eager | one op within the gather op size, else the session's large-message all-gather (rows of 77,440 bytes) |
| MTP embedding, o_proj and MoE all-reduces | `v1/worker/gpu/spec_decode/autoregressive/speculator.py:511`→`models/glm5next/nvidia/mtp.py:222`; `mtp.py:109`→`model.py:575`; `model.py:592`→`moe_runner.py:540` | TP | draft prefill [T,4096], later steps [R,4096] | k draft forwards per step | as the target's |
| Draft logits all-gather | `v1/worker/gpu/spec_decode/speculator.py:475`→`mtp.py:432`→`:244`→`logits_processor.py:281` | TP | [R,38,720]→[R,154,880] | k per step | as the target's |
| Local-argmax pair all-gather (opt-in) | `logits_processor.py:423-427` | TP | [R,2] fp32→[R,8] | only with `use_local_argmax_reduction` (`config/speculative.py:452`, default False) | one op |
| Vision tower all-reduces and merger all-gather | `models/glm5next/common/multimodal.py:215` (`:365` chunked), `:120`, `:428`→`linear.py:669`, `:433` | TP | [S,1,1024], [M,1024]→[M,4096] | image/video prefill and encoder profiling | as above |

Per decode step: 1 + 2N all-reduces of [T,4096] and one logits all-gather on
the target, plus k × (3 all-reduces + 1 all-gather) for MTP.

**mHC prefill row ownership: PyNccl directly.**
`VLLM_GLM53_MHC_PREFILL_SHARD` (vLLM default 0, `envs.py:1830`; the
SparkRing GLM-5.3-Flash profiles set 1) splits the hyper-connection (mHC)
mixing of an eager prefill forward among the TP ranks
(`models/glm5next/nvidia/mhc_prefill_sharding.py`, `model.py`):

- admission: `configure` (`mhc_prefill_sharding.py:90-167`, at model
  construction) requires TP2 or TP4, a 4,096- or 8,192-token ceiling, capture
  sizes below it, BF16, hidden 4,096 and a 48-SM GB10, voted with
  `all_gather_object` on the TP gloo group (`:148`); `maybe_create`
  (`:408-513`, every forward, `model.py:1170`) admits only eager forwards of
  exactly the ceiling's rows with pure prefill metadata, checks the TP
  PyNccl communicator's `available`, `disabled`, `world_size`, `rank` and
  `device` (`:450-465`) and votes on the gloo group (`:491`);
- per decoder layer (`model.py:633-698`): the first layer's mixing stays on
  all rows; every later layer all-gathers the owned rows before attention
  (`:657`); attention and the MLP skip their own all-reduce
  (`defer_tp_reduction=True`, `:663`, `:697`) and their partial outputs are
  reduce-scattered (`:672`, `:698`); the MLP input is all-gathered (`:696`);
- after the last layer one all-gather of the hidden states (`:1217`), plus one
  per auxiliary hidden-state layer (`:1198`; none with MTP);
- the collectives: `PrefillOwnership.reduce_scatter` calls
  `self.comm.reduce_scatter(output, partial, stream=stream)` (`:347`) and
  `all_gather` calls `self.comm.all_gather(output, owned, stream=stream)`
  (`:359`), with `comm` the TP group's PyNccl communicator.

Per 8,192-token chunk at TP4 (N=45): 90 reduce-scatters along dimension 0 of
`[8192, 4096]` BF16 (64 MiB in, `[2048, 4096]` = 16 MiB out) and 90
all-gathers of `[2048, 4096]` BF16 shards (16 MiB in, 64 MiB out); the
embedding and the three MTP draft all-reduces of `[8192, 4096]` stay on the
device communicator. On a `none` TP group, which with NCCL off includes a
pair, the adapter carries both collectives through the pinned shim
`mhc_prefill_shard` (section 6).

**Off by default, and their collectives bypass the communicator.**

- B12X DCP paths (`v1/attention/backends/mla/b12x_mla_sparse.py:268-283,
  1690-1698, 1727, 1946`) apply at DCP > 1 only. Without PyNccl they call
  `torch.distributed.all_gather_into_tensor`, which the tripwire's carrier runs
  on a DCP group's session.
- Batch-sharded sampling (`v1/worker/gpu/sample/batch_shard.py:302`) and
  adaptive verification (`v1/worker/gpu/spec_decode/adaptive_verification.py:379`,
  through `parallel_state.py:917`) are off by default.

On a `none` group the adapter refuses batch-sharded sampling at setup (its
all-to-all has uneven splits), and the tripwire's carrier runs adaptive
verification's broadcast on the group's session (section 6).

## 4. Qwen3.8-Flash-Next at TP4

**Code path.**

- The architectures `Qwen3_8FlashNextForCausalLM`, `...ForConditionalGeneration`
  and `...MTP` map to `vllm.models.qwen4_exp`
  (`model_executor/models/registry.py:114-117, 606-609, 711-714`).
- On CUDA they load `models/qwen4_exp/nvidia/model.py` and `mtp.py`
  (`models/qwen4_exp/__init__.py:28-43`).
- The model is a hybrid of gated delta-net linear attention and gated full
  attention, with MoE and a shared expert, four-stream hyper-connections and
  an MTP head (default one layer, `mtp.py:208`).
- The V2 runner is the default.

**Config.**

- From `transformers_utils/configs/qwen3_next.py`: vocab 151,936 (`:188`),
  hidden 2,048 (`:189`), 48 layers with every fourth full attention
  (`:191, 250-254`), 512 experts top-10 (`:212-213`).
- From `models/qwen4_exp/config.py:93-94`: hyper-connection rank 320.

**B12X switch.** `uses_b12x` (`models/qwen4_exp/nvidia/backend.py:8-10`) is
true only with b12x linear or MoE backends (default `auto`,
`config/kernel.py:255-258, 313`). It changes several rows below.

| Collective | Call site | Group | Tensor | When | SIRCL on a `none` TP4 group |
|---|---|---|---|---|---|
| Embedding all-reduce | `vocab_parallel_embedding.py:693` (from `model.py:587`) | TP | [T,H] bf16 | every forward | one op up to D, else the session's large-message all-reduce |
| GDN output projection | `linear.py:1934` via `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1769` (or `:1688`) | TP | [T,H] bf16 | per GDN layer | same |
| Full-attention o_proj | `linear.py:1934` via `model_executor/models/qwen3_next.py:465` | TP | [T,H] bf16 | per full-attention layer | same |
| MoE output | `moe_runner.py:540` | TP | [T,H] bf16 | per MoE layer | same |
| Hyper-connection all-gathers (b12x only) | `models/qwen4_exp/nvidia/hyperconnection.py:690, 703` | TP | [T,r/4]→[T,r] and [T,H/4]→[T,H] bf16, dim −1 | 4N+2 per forward | one op within the gather op size, else the session's large-message all-gather |
| PLE n-gram embedding all-reduce | `models/qwen4_exp/nvidia/b12x_ple.py:688` or `ngram_embedding.py:461-463` | TP (engram TP = TP) | [T,E] bf16, or an **int8 view** of FP8 tables | per PLE layer, when `ple_layer_ids` is non-empty | bf16: as the embedding; int8: all-gather, then a rank-ordered integer sum (exact) |
| Logits all-gather | `logits_processor.py:281` (`model.py:931`) | TP | [N,Vp/4]→[N,Vp] | once per step | one op within the gather op size, else the session's large-message all-gather |
| MTP: embedding all-reduce, fc gather_output all-gathers (non-b12x), o_proj and MoE all-reduces, draft logits all-gather | `mtp.py:645, 610, 614`; `linear.py:669, 1934`; `moe_runner.py:540`; `mtp.py:836`→`logits_processor.py:281` | TP | as the target's | k per step | as above |

**Hyper-connection prefill row ownership.** vLLM's default is off; the
SparkRing Qwen3.8 profiles set `VLLM_QWEN3_8_HC_PREFILL_MODE=shard`. Each
tensor-parallel rank then keeps `rows / W` rows of an eager pure prefill of
at least 1,024 rows (`hc_prefill.py:52-84`, admitted at `model.py:785`; the
mode is accepted at TP2 and TP4 only, `hc_prefill.py:17-49`), and the
ownership object calls **PyNccl directly** (`hc_prefill.py:104-130`):

| Collective | Call site | Tensor | Count per forward of N layers |
|---|---|---|---|
| reduce-scatter (sum), dimension 0 | `model.py:387, 405` → `hc_prefill.py:119` | [T,H]→[T/W,H] bf16 | 2N |
| all-gather, dimension 0 | `model.py:374, 401, 658` → `hc_prefill.py:110` | [T/W,H]→[T,H] bf16 | 2N + 1 |
| all-gather of the multi-stream state | `model.py:353` (per PLE layer), `:660` (MTP) | [T/W,hc·H]→[T,hc·H] bf16 | PLE layers + 1 |

On a `none` TP group, which with NCCL off includes a pair as well as a
four-Spark group, the pinned shim `qwen_hc_prefill_shard` carries both
(section 6).

**Off by default.** Batch-sharded sampling and adaptive verification behave
as for GLM.

## 5. GLM-5.3 at TP8 with DCP4 (`glm_moe_dsa`)

**Code path.**

- `GlmMoeDsaForCausalLM` → `model_executor/models/registry.py:126` →
  `models/deepseek_v32/__init__.py:17-21` → `nvidia/model.py:435`.
- Attention: `models/deepseek_v32/attention.py`, or `nvidia/b12x.py` with the
  B12X backend.
- MoE: `model_executor/models/deepseek_v2.py:292`.
- MTP: `nvidia/mtp.py:271`.
- GLM's config hook defaults `dcp_comm_backend="a2a"`
  (`model_executor/models/config.py:49-56`).

**Groups.**

- TP {0..7}, the whole ring. Under `SIRCL_NCCL=auto` with `NCCL_ALGO=Ring`
  and `NCCL_SKIP_TREE_CONNECT=1`, NCCL may run its ring algorithm only;
  otherwise it is a `none` group.
- DCP {0-3} and {4-7}. Each is a path of four, a `none` group in either mode.
- EP {0..7}, the TP group's ranks at DP=1: online MoE weight quantization's
  `amax` reductions at startup (section 6); no traffic while serving. As a
  `none` group it shares the TP group's session.

**B12X construction.** In this tree, the B12X attention class does not
accept an argument the model passes (`nvidia/model.py:91-97` versus
`nvidia/b12x.py:71`), so serving the B12X path needs a change to this tree
that supplies it. The rows below assume the B12X path.

| Collective | Call site | Group | Tensor | When | SIRCL |
|---|---|---|---|---|---|
| Embedding all-reduce | `nvidia/model.py:273`→`vocab_parallel_embedding.py:693` | TP | [T,d] bf16 (int8 view for FP8 tables) | every forward | TP session up to D; above D the session's large-message all-reduce, or NCCL's ring for eager messages when NCCL runs on the TP group |
| Post-attention all-reduce, then RMSNorm | `model.py:166`→`models/common/ops/fused_allreduce_rms_norm.py:62` (FlashInfer's fused path, `:40-60`, needs a size entry for the device's compute capability, `compilation/passes/fusion/allreduce_rms_fusion.py:108-130`; GB10's 12.1 has none) | TP | [T,d] bf16, N per forward | every forward | same; with `SIRCL_FUSED_NORM=1` one fused SIRCL kernel per qualifying call ([`norm_fusion.py`](norm_fusion.py)) |
| MoE / MLP all-reduce, then RMSNorm | `model.py:152, 327`→`fused_allreduce_rms_norm.py:62` | TP | [T,d] bf16, N per forward | every forward | same |
| DCP query all-gather | `attention.py:614`→`v1/attention/ops/dcp.py:1705-1709` | DCP | [Ta,H/8,576]→[Ta,H/2,576] bf16, dim 1 | every forward | DCP session: one op within the relay-safe gather op size, else the session's large-message all-gather |
| DCP output combine (default `a2a`) | `attention.py:645`→`dcp.py:951-1017`: **`dist.all_to_all_single` on the DCP NCCL group** | DCP | [4,Ta,H/8,514] bf16 | every forward | **pinned shim** `dcp_all_to_all`: same arithmetic, exchange through the communicator (DCP session all-to-all); with `VLLM_USE_B12X_DCP_A2A=1` the shim `dcp_b12x_transport` withholds the B12X PCIe transport (`v1/attention/ops/dcp.py:1501-1544`) so this path stays |
| Alternative combine `ag_rs` | `dcp.py:455` (LSE all-gather), `:495-504` (reduce-scatter) | DCP | [Ta,H/2] fp32; [H/2,Ta,512] bf16 | with `--dcp-comm-backend ag_rs` | session all-gather and reduce-scatter |
| Indexer top-k merge | `v1/attention/backends/mla/b12x_indexer.py:317` (`:689`, `:730`) | DCP | [R,K,2] fp32→[R,4K,2], dim 1 | per indexer layer, prefill chunk and decode | DCP session all-gather |
| Logits all-gather | `deepseek_v2.py:1978`→`logits_processor.py:281` | TP | [Ns,V′/8]→[Ns,V′] | every step, eager | TP session within the gather op size; above it the session's large-message all-gather, or NCCL's ring when NCCL runs on the TP group |
| MTP draft | `nvidia/mtp.py:230, 162, 251`; DCP rows as above | TP, DCP | as the target's | k per step | as above |

**Prefill KV-latent gather.** The context gather across DCP ranks
(`mla_attention.py:3566`→`dcp.py:1754-1759`) calls `torch.distributed.all_gather_into_tensor`
on the DCP device group. It is not reached on the B12X path
(`b12x_mla_sparse.py:1173`). Where it is reached, the tripwire hands it to the
DCP group's carrier, which gathers the bytes on the DCP session.

## 6. Collectives that bypass the device communicator, and what the adapter does

| Bypass | Where | Default | Adapter on a `none` group |
|---|---|---|---|
| PyNccl warm-up all-reduce at construction | `pynccl.py:186-191` | every multi-rank group | PyNccl is not built (`VLLM_DISABLE_PYNCCL` during that group's constructor) |
| DCP `a2a` combine: `dist.all_to_all_single` | `dcp.py:1012-1017` | GLM at DCP>1 | pinned shim `dcp_all_to_all` |
| `GroupCoordinator.broadcast`, `broadcast_object_list`, `broadcast_tensor_dict` on the device group | `parallel_state.py:907-961, 1053-1133` | with adaptive verification, a confidence snapshot every step (`adaptive_verification.py:377-379`); otherwise unused by TP serving at DP=1 and PP=1 | the tripwire's carrier runs `broadcast` on a group with a session; refused on a group without one |
| mHC prefill sharding (GLM-5.3-Flash): PyNccl reduce-scatter and all-gather directly | `mhc_prefill_sharding.py:347, 359`, ownership created at `model.py:1170` | vLLM off; the SparkRing GLM-5.3-Flash profiles on | pinned shim `mhc_prefill_shard`: a PyNccl stand-in that runs SIRCL's plans, on a TP group with a session; refused at setup without a session |
| Hyper-connection prefill row ownership (Qwen): PyNccl reduce-scatter and all-gather directly | `models/qwen4_exp/nvidia/hc_prefill.py:110, 119`, ownership created at `model.py:595` | vLLM off; the SparkRing Qwen3.8 profiles on | pinned shim `qwen_hc_prefill_shard`: the same PyNccl stand-in, on a TP group with a session; refused at setup without a session |
| Batch-sharded sampling: `all_to_all_single` on the TP device group | `v1/worker/gpu/sample/batch_shard.py:302-307` | off | refused at setup |
| Online quantization scale reductions: `torch.distributed.all_reduce(op=MAX)` of weight `amax` on the TP or EP device group | `model_executor/layers/quantization/utils/quant_utils.py:43-90`, called from `online/nvfp4.py:123`, `online/fp8.py:214, 396, 581, 584, 766` and `online/int8.py:79` | at weight loading with online quantization (GLM-5.3's MoE experts); one float32 per expert, two calls per MoE layer | carried on the group's session (own or shared) by the tripwire's carrier; refused on a group without one, naming the remedies |
| Async TP and fused GEMM communication (`fuse_gemm_comms`): symmetric memory | `compilation/passes/fusion/collective_fusion.py:900-989` | off (`config/vllm.py:199, 362-363`) | refused at setup |
| Fused all-reduce + RMSNorm pass (`fuse_allreduce_rms`): FlashInfer or B12X PCIe kernels | `allreduce_rms_fusion.py:1097-1289` | on at O2 only with SM90/SM100 or the B12X PCIe switch (`config/vllm.py:232-255`) | refused at setup |
| B12X DCP PCIe IPC | `distributed/device_communicators/b12x_dcp.py:159, 170`; selected at `dcp.py:1501-1544` with `VLLM_USE_B12X_DCP_A2A=1` | one node only | pinned shim `dcp_b12x_transport`: no transport for a SIRCL group |
| Symmetric-memory DCP paths (`VLLM_USE_DIRECT_DCP_A2A`, `_Q_GATHER`, `_KV_GATHER`) | `dcp.py:1126, 1290, 1410` | off | refused at setup |
| Overlapped DCP prefill context gather (`VLLM_MLA_PREFILL_DCP_OVERLAP=1`): PyNccl directly | `v1/attention/ops/dcp_prefetch.py:36-46` | off | refused at setup |
| PP and DP broadcasts and DP padding agreement | `v1/worker/gpu/pp_utils.py:85-267`; `v1/worker/dp_utils.py:42-75` | PP>1 or DP>1 only | PP: SIRCL point-to-point channels (the sampled-token broadcast as sends from the last stage, through the tripwire's point-to-point carrier); DP: tripwire |

The tripwire covers Python calls to `torch.distributed` collective functions.
Functional collectives issued from C++ inside compiled graphs are not covered;
the compilation passes that create them are refused instead.

## 7. How the repository's four-rank adapters intercept

The repository's four-rank adapter contract is in
[`integrations/vllm/README.md`](../../../../integrations/vllm/README.md).
`integrations/vllm/sitecustomize.py:62-74` installs each module below when its
variable is set, unless `SPARK_TP4_ENABLED=0`; the serve launcher and
`bundle` set `SPARK_TP4_ENABLED=0`.

| Adapter | Interception | Admits | Everything else |
|---|---|---|---|
| Four-rank all-reduce (`spark_tp4_backend.py`; `VLLM_SPARK_TP4_MODE`) | wraps `CudaCommunicator.all_reduce` (`spark_tp4_backend.py:2094-2348`; keeps `_spark_original`) | group `tp:0`, world 4, contiguous bf16 `[Q,W]`: W=6144 Q 1-40, and research W=4096 graph, eager prefill and fused prefill shapes | vLLM's original chain (NCCL) |
| Four-rank vocabulary all-gather (`spark_tp4_vocab_allgather_backend.py`; `VLLM_SPARK_TP4_VOCAB_MODE`) | wraps `GroupCoordinator._all_gather_out_place` (`:684-889`), in front of every device communicator | `[Q,38720]` bf16, dim −1 or 1, Q ≤ `VLLM_SPARK_MAX_QUERY_ROWS` | the original method |
| Health gate (`spark_tp4_health_gate.py`; `SPARK_TP4_HEALTH_GATE=1`) | wraps async output `get_output` and the worker outputs; exits with code 70 on native failure | — | — |

The adapters assume a four-Spark ring, in which both perfect matchings of the
group are direct cables. On four consecutive Sparks of a larger ring they
cannot serve the group. SIRCL's communicator refuses to start when the
all-reduce or vocabulary adapter is installed and enabled for such a group
(`tp4.conflicts`). On a four-Spark ring it composes with the four-rank
all-reduce adapter: admitted signatures go to the four-rank sessions, and all
other signatures follow SIRCL's plans.

## 8. Limits of this survey

- The checkpoints' real dimensions:
  - GLM-5.3-Flash: layer kinds and dense layers.
  - Qwen: L, H, PLE layers, indexer fields, MTP layers.
  - GLM-5.3: H, d, V, K, number of indexer layers.
- Deployment flags:
  - k.
  - B12X backends for Qwen.
  - DCP environment switches.
  - Graph mode.
- Collectives inside FlashInfer's autotuner (world gloo group) and inside
  external libraries.
