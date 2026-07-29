# SparkRing — Public-Functional Lane: supported matrix and acceptance definition

This document defines **one** supported configuration for the
public-functional lane, and defines exactly what "it works" means for that
configuration. It is the contract the acceptance gate
(`scripts/acceptance_gate.py`) enforces.

> **Status: definition only. The gate defined here has never been run.**
> No public-lane acceptance result exists. Nothing in this document asserts
> that the public lane passes, or has passed, any stage. Every open value is
> listed in [§8 Open TBDs](#8-open-tbds-with-owners) with an owner, not
> guessed.

Companion documents: [README.md](../README.md) (two-lane framing),
[QUICKSTART.md](QUICKSTART.md) (fast public trial),
[SETUP.md](SETUP.md) (bring-up procedure), [RESULTS.md](RESULTS.md) (how
measured claims are labelled and gated), [RUNTIME_GAPS.md](RUNTIME_GAPS.md)
(what the public runtime does and does not contain),
[../runtime/README.md](../runtime/README.md) (the attestation contract).

---

## 1. What the public-functional lane promises

The public-functional lane promises exactly one thing:

> On the supported matrix in §2, a runtime built from `runtime/` and pinned by
> `runtime/runtime-lock.json` starts on all four ranks, attests, and answers
> the OpenAI-compatible API correctly and reproducibly.

It promises **nothing about speed**. Functional acceptance is a binary result.
Performance is a separate, measured, banded result reported alongside it and
never merged into it (§4).

The lane is deliberately narrow. One matrix means one thing can be proven, one
set of artifacts has to be produced, and a bug report is comparable to every
other bug report. If your configuration is not the matrix below, you are not in
this lane — see §3, which exists so you can self-select out quickly.

---

## 2. The supported matrix

Everything here is stated as a **requirement**. Where a requirement is
site-specific (addresses, hostnames, users, paths), the requirement is on the
*shape*, and the value is yours; the gate reads it from your site config and
never needs it published. Where the repository already pins a value, the pin is
cited to its source of truth.

### 2.1 Hardware

| Requirement | Value | Source |
|---|---|---|
| Node count | Exactly **4** | ring topology; TP=4/DCP=4 |
| Node type | NVIDIA DGX Spark (GB10), GPU arch **sm_121 / sm_121a** | SETUP.md Stage 2.2 |
| Unified memory per node | ~120–121 GiB, of which the model-down memory floor must leave room for the KV pool in §2.4 | SETUP.md Stage 2.2, Stage 8.1 |
| Free NVMe per node | >= the checkpoint (382 GiB) plus JIT/compile cache and container image headroom | README headline; SETUP.md Stage 6 |
| NIC | One dual-cage ConnectX-7 complex per node (two 200 GbE cages) | SETUP.md Stage 1.1 |
| Cabling | **Four** QSFP28 200 GbE DAC cables forming a 4-cycle over ranks 0–3, **cage-matched at both ends** (cage*i* to cage*i*). **No switch on the data path.** | SETUP.md Stage 1.2 |
| Link speed | Exactly 200,000 Mb/s on every port of every edge, both directions | SETUP.md Stage 3.4 |
| Per-link addressing | One dedicated **/24 per physical cable**, four **distinct** /24s. Not link-local (169.254/16). | SETUP.md Stage 1.3 |
| MTU | **9000** on both fabric interfaces of every node | SETUP.md Stage 3.1 |
| RoCE | **RoCEv2, IPv4 GID index** bound to the netdev (index `3` on the reference adapters; the requirement is "the IPv4 RoCEv2 index for *your* adapter", and RoCEv1 link-local GIDs are rejected) | SETUP.md Stage 3.2 |
| Management network | A **separate** network carrying SSH, NCCL bootstrap and Gloo control traffic only. No fabric payload may appear on it. | SETUP.md Stage 1.3 |

The acceptance gate validates the *structural* half of this from your site
config before it touches anything: four ranks, four edges, every rank of
degree 2, one /24 per edge, four distinct /24s, MTU 9000. The physical half is
proven by stage `fabric_transport_qualification`, which delegates to
`spark_transport/scripts/qualify_direct_cable.py`.

### 2.2 Software

| Component | Requirement | Source of truth |
|---|---|---|
| Host OS | Ubuntu 24.04 (DGX OS), **aarch64** | SETUP.md Stage 2.2 |
| Host kernel | NVIDIA kernel **6.17** on the reference cluster. Minimum supported floor: **TBD-6**. | SETUP.md Stage 2.2 |
| NVIDIA driver | **580.x** (reference cluster 580.173.02). Minimum supported floor: **TBD-5**. | SETUP.md Stage 2.2 |
| Host CUDA | **13.0** | SETUP.md Stage 2.2 |
| Host system NCCL | >= **2.30.4** (the patched build is what actually runs; this is the packaging floor) | SETUP.md Stage 2 verify block |
| Serving NCCL | Built from `NVIDIA/nccl` commit **`73cf112295c33aee2b895f329f592f2a9b4b0f97`** (the `v2.30.7-1` release commit) plus both in-repo patches, each pinned by SHA-256 in the lock | `runtime/runtime-lock.json` → `nccl` |
| Container runtime | Docker CE with `nvidia-container-toolkit`, or podman (`CONTAINER_ENGINE=podman`). Minimum versions: **TBD-7**. | SETUP.md Stage 2.6; `runtime/README.md` |
| Base images | ARM64 manifests resolved from `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04` and `:13.0.1-runtime-ubuntu24.04`, pinned by digest in the lock | `runtime/runtime-lock.json` → `base_image` |
| Container CUDA / toolchain | CUDA **13.2**, `CMAKE_CUDA_ARCHITECTURES=121`, `linux/arm64` | `runtime/runtime-lock.json` → `toolchain` |
| Python (in the image) | **3.12.3** | `runtime/runtime-lock.json` → `toolchain.python_version` |
| Python (gate host, for `scripts/*.py`) | **>= 3.10** plus PyYAML for the documented YAML site file; no GPU or cluster access needed for `--dry-run` | `requirements-dev.txt`; this repo |
| torch | **2.12.0+cu132** from `https://download.pytorch.org/whl/cu132` | lock → `toolchain` |
| vLLM | `vllm-project/vllm` @ **`fcc614141e5e9ab18cb304c476f7feed2a9552e3`** (0.11.2.dev279 lineage) | lock → `vllm` |
| sparkinfer (B12X) | `local-inference-lab/sparkinfer` @ **`284a2eae83754ee1abd31c37b9ca66b68e20b8a8`** | lock → `sparkinfer` |
| FlashInfer | `flashinfer-ai/flashinfer` @ **`25dd814e03791e370f96c3148242f0dc8de504ac`**; wheels `flashinfer-python 0.6.13+cu132`, `flashinfer_jit_cache 0.6.13+cu132` | lock → `flashinfer` |
| DeepGEMM | **2.5.0+2073ddb** from commit **`2073ddb2814892014c33ef4cd1c7d4c148baf1fe`** | lock → `deep_gemm` |
| Full pip set | `runtime/pip-freeze.txt` — no resolver drift permitted | `runtime/README.md` acceptance gate 5 |
| vLLM overlay | The recovered reference delta under `runtime/patches/00-reference-vllm/` (59 safe modified files + 12 additions), followed by the two independently written SparkCache compatibility patches under `runtime/patches/vllm/`. All 73 operations are preimage-pinned and fail closed. | `runtime/README.md`; `runtime/patches/00-reference-vllm/README.md` |
| Image identity | The built serving image, referenced by its post-push registry digest. The launcher must inject that digest as `SPARKRING_IMAGE_DIGEST`; a missing or skipped digest check is an acceptance failure. | `runtime/README.md` verify flow; `scripts/acceptance_gate.py` |

**Non-pinned images and arbitrary vLLM versions are rejected**, not tolerated
(§3). The runtime manifest inside the image, plus `runtime/verify-runtime.py`,
is what makes that enforceable; the gate delegates to it rather than
re-implementing it.

### 2.3 Model

| Requirement | Value |
|---|---|
| Repository | **`aidendle94/GLM-5.2-MXFP4-Experts-GPTQ`** (`runtime/runtime-lock.json` → `model.repository`) |
| Revision | **`46537e0e16fcd156627800139b41b9c497fc7ee2`**, an immutable Hugging Face commit recorded in `model.revision` |
| Identity check | `config.json` of the pinned revision must hash to `ffd30e72ab8bb7e8ad560f2aaab03cc595f3106f0acf793ef96eedaf90f66d69` (`model.config_sha256`) |
| Size | ~382 GiB, present on all four nodes |
| Load format | `safetensors` |

**Mutable references are rejected.** `main`, `master`, `HEAD`, `latest`, a
branch name, a tag, an empty value, or the literal `pending` are all refused by
the gate's model-identity check with a non-zero exit. A checkpoint reference
that can change under you is not a pinned configuration, and a result measured
against one is not reportable.

### 2.4 Serving configuration

The target serving configuration, as recorded in SETUP.md Stage 8.4/8.5 for the
attested reference window. In this lane these are **requirements of the
matrix**, not a claim that the public lane reproduces reference-lane behaviour
with them (see TBD-8).

| Setting | Required value |
|---|---|
| Tensor parallel | `--tensor-parallel-size 4` (TP=4, one rank per node) |
| Decode context parallel | `--decode-context-parallel-size 4 --dcp-comm-backend ag_rs` (DCP=4) |
| Ranks | 4; rank 0 hosts the API, ranks 1–3 run `--headless` |
| Distributed | `--nnodes 4 --node-rank <RANK> --master-addr <rank0-management-ip> --master-port <port>`, `--distributed-executor-backend mp` |
| Supported MTP mode | **adaptive MTP 2/4, window 32** — `num_speculative_tokens: 4`, `adaptive_speculative_tokens_window: 32`, `method: mtp`, adaptive depths `2,4` (`VLLM_SPARK_MTP_MODE_ID=adaptive-mtp2-4-window32`). Fixed-K MTP and MTP-off are **not** part of this matrix. See TBD-9 for the determinism consequence. |
| Max context | `--max-model-len 458752` |
| Batching | `--max-num-batched-tokens 4096`, `--max-num-seqs 8` (Q40 ABI cap: max query rows = seqs x (K+1) = 40) |
| KV cache | `--kv-cache-dtype nvfp4_ds_mla`, `--kv-cache-memory-bytes 4600000000` (4.6 GB/rank), per-token scale mode |
| Attention backend | `--attention-backend B12X_MLA_SPARSE` |
| Memory | `--gpu-memory-utilization 0.88` |
| Graphs | `cudagraph_mode: FULL_AND_PIECEWISE` with the pinned capture-size list. **First bring-up should run eager** (`--enforce-eager`) before graphs are enabled; an eager run is a valid *bring-up* step but is **not** an acceptance run — the matrix is the graph configuration. |
| Prefix caching | enabled |
| API | OpenAI-compatible server on rank 0; served model name and port are site choices recorded in the site config |

**What is machine-checked.** The gate enforces TP=4, DCP=4, `mtp_mode:
adaptive`, `mtp_tokens: 4`, `max_model_len: 458752`,
`kv_cache_bytes_per_rank: 4600000000` and `max_num_seqs: 8` against the site
config, and refuses any other value with the expected one in the message. The
MTP window (32), the attention backend and the KV dtype have no field in the
site schema, so they are documented requirements the gate cannot yet verify —
TBD-13 and TBD-14.

`--attention-backend B12X_MLA_SPARSE` and `--kv-cache-dtype nvfp4_ds_mla` are
the single largest open question for this lane: RUNTIME_GAPS.md records the
SM121/GB10 sparse-MLA backend and the packed low-bit MLA KV record formats as
**not merged upstream** and supplied by the reference-lane overlay, which this
lane does not ship. Whether a public-lane image can satisfy this row at all is
**TBD-8**. Until TBD-8 is resolved, any substitution (a different backend or KV
dtype) must be written into this document *before* a result is reported against
it — a result measured on an undocumented substitution is not a result for this
matrix.

---

## 3. Explicitly unsupported

If any of the following describes you, the public-functional lane does not
cover your configuration. The gate will refuse, or its result will not be
comparable to anyone else's. This list exists so you can stop here rather than
spend a week finding out.

- **Any GPU that is not GB10 / sm_121**, and any GPU count other than four.
  There is no 1-node, 2-node, 8-node, or mixed-node variant of this matrix.
- **Switched topologies.** Any Ethernet/IB switch on the data path, any
  fabric that is not four direct DAC cables in a 4-cycle, any partial mesh,
  any single-cable or three-cable ring. The transport schedules two perfect
  matchings over a 4-cycle; anything else is a different system.
- **Non-cage-matched cabling**, shared/overlapping fabric subnets, MTU != 9000,
  RoCEv1 link-local GIDs, or fabric traffic sharing the management network.
- **Arbitrary vLLM versions.** Only the pinned commit in the lock. Not "a
  recent main", not a release wheel, not a distro package.
- **Non-pinned images.** Not "an image I built last week", not a tag without a
  digest, not an image whose `runtime-manifest.json` fails
  `runtime/verify-runtime.py`. Fresh builds produce new artifact hashes; the
  fail-closed launcher must have those re-pinned before a run counts.
- **Mutable model references.** See §2.3.
- **Mixed runtimes across ranks.** All four ranks must attest the same
  `runtime_id` and the same manifest self-hash.
- **Performance parity with the reference lane.** This is the important one.
  The public lane is *not* claimed to be as fast as the reference lane, and no
  target in this document is a throughput target. The reference lane runs an
  overlay this lane does not ship (RUNTIME_GAPS.md). Expect a different
  performance profile, and expect it to be reported as its own measured band
  (§4), not as a pass/fail against reference-lane numbers.

---

## 4. Acceptance is not performance

Two verdicts. They are computed separately, reported separately, and never
substituted for one another.

### 4.1 Functional verdict (binary)

**Functional acceptance** means, on the matrix in §2, under the pinned runtime:

1. every rank attests the same runtime, artifacts, and model identity;
2. the fabric qualifies and the model-down transport probes pass;
3. all four ranks start;
4. the API is live and reports the pinned served model;
5. a fixed prompt, decoded greedily with a fixed seed and a fixed token budget,
   produces the **expected token ids** — compared by SHA-256 against a
   committed expected-value file.

`functional_verdict` is one of:

| Value | Meaning |
|---|---|
| `PASS` | Every functional stage passed, including an exact match against a committed expected-output file. |
| `FAIL` | A functional stage failed. The run aborted at that stage. |
| `BASELINE-RECORDED` | Every functional stage ran, but no expected-output file existed, so the gate **recorded** the observed token ids as a candidate baseline instead of asserting a pass. This is not a pass. |
| `NOT-RUN` | Dry-run, or the gate never executed. |

There is no partial credit, no "mostly working", and no silent acceptance: a
missing expected-output file yields `BASELINE-RECORDED` and a non-zero exit,
never `PASS`.

### 4.2 Performance verdict (measured band)

**Performance** is measured by a small C1/C8 matrix that emits raw JSON
(aggregate throughput, per-stream throughput, TTFT p50/p99, token counts, cell
duration) and is compared against a **documented tolerance band** carried in
the site config.

`performance_verdict` is one of:

| Value | Meaning |
|---|---|
| `IN-BAND` | Every banded metric fell inside its band. |
| `OUT-OF-BAND` | At least one banded metric fell outside. Stage status is `PERFORMANCE-OUT-OF-BAND`. |
| `BASELINE-RECORDED` | No band was configured, so the observed numbers were recorded as a candidate band. **This is the expected state today** — no public-lane band exists yet (TBD-10). |
| `NOT-MEASURED` | The performance stage did not run. |

A performance miss **does not** abort the run and **does not** change
`functional_verdict`. It gets its own status and its own exit code so CI can
tell the two apart.

### 4.3 The rule about reference-lane numbers

> **Reference-lane historical throughput numbers must never be presented as
> public-lane results.** Not as a target, not as an expectation, not as a
> band, not as "roughly what you should see".

Every number in [RESULTS.md](RESULTS.md) — 834/884/854 tok/s prefill, 63.60
tok/s C8 aggregate, 20.83/19.28/21.43 tok/s C1 decode, 27.2 tok/s coding
median, the 500,224-token KV pool, every transport row — was measured on the
**reference lane**, with an overlay this lane does not ship, and carries a full
configuration label naming that lane. Quoting any of them as a public-lane
figure is a mislabelled claim under the claim discipline in RESULTS.md §4, and
seeding the public-lane tolerance band from them would bake the mislabel into
the gate.

The public-lane band must be established from public-lane runs: record a
`BASELINE-RECORDED` band first, run it enough times to know its spread, then
commit a band with its own label. Until then, `performance_verdict` is
`BASELINE-RECORDED` and the honest statement is "no public-lane performance
band exists".

---

## 5. The acceptance gate

`scripts/acceptance_gate.py` runs the stages below **in order** and **aborts at
the first functional failure** with a non-zero exit and an actionable message.
Each stage emits a stable id, a status, timing, and captured artifacts.

| # | Stage id | What it proves | Delegates to |
|---|---|---|---|
| 1 | `runtime_attestation` | Every rank runs the same attested image and artifacts (manifest self-hash, installed sources, wheel/NCCL/transport `.so` hashes, image digest, `runtime_id`); the model pin is an immutable revision with the pinned `config.json` hash, and the site and the lock name the same checkpoint. | `runtime/verify-runtime.py` per rank (`--json --emit-attestation --expect-runtime-id`), `scripts/preflight.py` evidence when supplied via `--preflight`, plus a lock-side model-identity check |
| 2 | `fabric_transport_qualification` | All four edges qualify at 200 G with verified bidirectional RC writes, correct GID/MTU, zero new PHY/CRC counters; then the model-down four-rank collective probe passes on every rank. | `spark_transport/scripts/qualify_direct_cable.py` per edge; the repo's `probe_dcp4_entrypoint.sh` per rank |
| 3 | `rank_startup` | All four ranks start under the site's launcher and reach a running state within the configured timeout. | the site's `launch.start_command` (the gate never implements launching) |
| 4 | `api_liveness` | `/health` returns 200 and `/v1/models` lists the pinned served model, on every rank that serves the API; every non-serving rank is explicitly declared headless. | HTTP |
| 5 | `deterministic_generation` | A fixed prompt, `temperature=0`, fixed seed, fixed `max_tokens`, produces token ids whose SHA-256 matches the committed expected-value file. | HTTP (`/v1/completions` then `/tokenize`) |
| 6 | `performance_matrix` | A C1 and a C8 cell, raw JSON out, compared against the documented band. | HTTP |
| 7 | `shutdown_rollback` | The stack stops cleanly, the API stops answering, and the site's rollback verification succeeds. | the site's `launch.stop_command` / `rollback_verify_command` |

Stage statuses: `PASS`, `FAIL`, `BASELINE-RECORDED` (stage 5, and stage 6 when
no band is configured), `PERFORMANCE-OUT-OF-BAND` (stage 6 only), `SKIPPED`
(not reached, or dry-run).

Abort semantics:

- `FAIL` aborts immediately; every later stage is `SKIPPED` with
  `aborted-after: <stage id>`.
- `BASELINE-RECORDED` does **not** abort (later stages still produce evidence)
  but it does prevent `functional_verdict: PASS`.
- `PERFORMANCE-OUT-OF-BAND` does **not** abort and does **not** affect the
  functional verdict.
- A benchmark *request* that errors (connection reset, 5xx, no tokens) is not a
  performance miss — it is a functional failure surfacing in stage 6, so it
  aborts. Only a completed measurement that lands outside the band is
  `PERFORMANCE-OUT-OF-BAND`.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Functional `PASS`; performance `IN-BAND` or `NOT-MEASURED`. |
| 2 | Functional `FAIL`. |
| 3 | Configuration or plan error — nothing was executed. |
| 4 | Functional `BASELINE-RECORDED` — an expected-output file was emitted for review; this is not a pass. |
| 5 | Functional `PASS`, but performance `OUT-OF-BAND` or `BASELINE-RECORDED`. |

### 5.1 `--dry-run` is the default

The gate does not execute anything unless you explicitly ask it to:

```bash
# default: validate the configuration and print the ordered plan. Touches nothing.
python scripts/acceptance_gate.py \
    --site scripts/config/site.yaml --gate-config my-gate-config.json

# write the plan out for review (still executes nothing)
python scripts/acceptance_gate.py \
    --site scripts/config/site.yaml --gate-config my-gate-config.json \
    --plan-out plan.json

# actually run it — requires an explicit confirmation token
python scripts/acceptance_gate.py \
    --site scripts/config/site.yaml --gate-config my-gate-config.json \
    --preflight evidence/preflight-<utc>.json \
    --execute --confirm RUN-PUBLIC-ACCEPTANCE-GATE \
    --bundle-dir evidence/acceptance
```

A dry run creates no bundle directory, writes no files (unless you pass
`--plan-out`), opens no connections, and starts and stops nothing. It validates
the matrix constraints (four ranks, a four-edge 4-cycle with every rank of
degree 2, one distinct /24 per cable, MTU 9000, 200 GbE, TP=4/DCP=4, adaptive
MTP 2/4, the pinned context and KV settings, an immutable model revision, and
site/lock agreement on the checkpoint) and prints the exact commands and
endpoints each stage would use.

`--execute` additionally refuses when the site config still contains the
shipped example's placeholders (`sparkring_site.placeholder_warnings()`) — a
half-filled template does not describe a real cluster. **Never point this gate
at a cluster that is serving production traffic** — stages 3 and 7 start and
stop the stack.

### 5.2 Evidence bundle

`--execute` produces one directory:

```text
<bundle-dir>/
  result.json                     schema, per-stage results, environment
                                  fingerprint, both verdicts, exit code
  plan.json                       the ordered plan that was executed
  expected/
    generation-baseline.json      written only when no expected-output file
                                  existed (BASELINE-RECORDED)
  stages/
    01-runtime_attestation/       per-rank verify-runtime JSON + attestation
    02-fabric_transport_qualification/
                                  per-edge qualifier JSON, per-rank probe output
    03-rank_startup/              launcher stdout/stderr, per-rank status
    04-api_liveness/              /health and /v1/models responses per rank
    05-deterministic_generation/  request params, completion, token ids, sha256
    06-performance_matrix/        raw C1 and C8 cell JSON
    07-shutdown_rollback/         stop output, post-stop health probes
```

`result.json` carries a schema version, the per-stage array (id, order, title,
status, timestamps, duration, message, artifact paths, details), an environment
fingerprint (gate host, repo commit, runtime-lock pins, model pin, serving
config, site config SHA-256), and the two verdicts as separate top-level
fields.

**The raw bundle is not automatically publishable.** Stage artifacts contain
whatever your cluster printed — hostnames, SSH targets, fabric addresses, local
model paths. Run it through `scripts/collect_evidence.py` before attaching it
to anything public (§6).

---

## 6. How to report a result

1. **Run the gate** (§5.1) and keep the bundle directory.
2. **Package it for publication:**

   ```bash
   python scripts/collect_evidence.py \
       --site scripts/config/site.yaml \
       --acceptance-result evidence/acceptance/<run-id>/result.json \
       --preflight evidence/preflight-<utc>.json \
       --runtime-manifest runtime-manifest.json \
       --log serve-rank0.log --log serve-rank1.log \
       --out evidence-bundle
   ```

   `collect_evidence.py` redacts mandatorily: every IPv4/IPv6 literal, every
   `user@host` SSH target and email address, every hostname and every path
   literal derived from your site config, and any obviously secret-shaped key.
   It then re-scans every emitted file and **fails** rather than emitting a
   bundle that still contains a blocklisted identifier. Image digests are kept
   by default (they are content addresses and are what makes the report
   comparable); pass `--redact-digests` if your registry path is sensitive.

3. **Read the bundle before you post it.** Redaction is mechanical; only you
   know what else is site-specific.
4. **State the verdicts separately.** "Functional: `PASS`. Performance:
   `BASELINE-RECORDED` (no public-lane band exists)." Never collapse them into
   one word, and never quote a RESULTS.md number as your own (§4.3).
5. **Include the matrix delta.** If anything differed from §2 — a substituted
   attention backend under TBD-8, an eager bring-up rather than graph mode, a
   different driver — say so at the top. A result on a different matrix is
   still useful; a result on a different matrix presented as this matrix is
   not.

For bug reports, attach the redacted bundle and the stage id that failed. The
stage id plus `result.json` is usually enough to reproduce the reasoning
without any access to your cluster.

---

## 7. Configuration: two files

**1. The site config** (`--site`) describes *your cluster*. Its schema is owned
and validated by `scripts/sparkring_site.py`
(`scripts/config/site.example.yaml` is the annotated template) and is shared
with `scripts/preflight.py`. That module's schema is authoritative; the gate
consumes its normalised form via `load_site(path)` and adds only the matrix
checks in §2 on top. When the module is unavailable the gate degrades to
reading an already-normalised JSON document and says so loudly.

The site config supplies: `topology.{mtu, link_speed_mbps, edges[4]}`,
`ranks[4].{ssh_target, management, ring_ports, transport_peers}`,
`runtime.{container_image, container_image_digest, model_path, model_repo,
model_revision, checkpoint_sha256}`, `serving.{tensor_parallel_size,
decode_context_parallel_size, mtp_mode, mtp_tokens, max_model_len,
kv_cache_bytes_per_rank, max_num_seqs, master_rank, api_port, master_port}`,
`paths`, `artifacts[]`, `preflight`. It is the *only* place site identifiers
live, and nothing in it needs to be published.

**2. The gate config** (`--gate-config`) describes *how to run the gate here* —
the things the site schema deliberately does not carry. It is a JSON document
owned by `scripts/acceptance_gate.py`; the full annotated default is in that
script's module docstring. Keys:

```text
ssh.command                          argv prefix for remote execution
runtime.{lock_path, verify_script, manifest_path, expect_runtime_id,
         exec_prefix}                exec_prefix runs verify-runtime.py inside
                                     the container, e.g. ["docker","exec",...]
fabric.{qualifier, probe_binary, iterations,
        model_down_probe.{command, per_rank}}
launch.{start_command, stop_command, rollback_verify_command,
        rank_status_command, ready_timeout_seconds, stop_timeout_seconds}
api.{scheme, rank_bases}             rank_bases overrides the derived
                                     master-rank endpoint when more than one
                                     rank serves an API
acceptance.{expected_generation_path, served_model_name, prompt, seed,
            max_tokens}
performance.{cells[], band}
preflight.result_path
```

`launch.start_command`, `launch.stop_command` and
`fabric.model_down_probe.command` have **no defaults** and must be supplied:
the gate never implements launching or probing, it invokes yours.

The API endpoint is derived from `serving.master_rank`'s management address and
`serving.api_port`; every other rank is treated as headless, which is what the
site schema encodes. Override `api.rank_bases` if your deployment serves the
API on more than one rank.

The served model name defaults to the site's `runtime.model_path`, which is
what vLLM reports unless `--served-model-name` was passed; override it with
`acceptance.served_model_name`.

---

## 8. Open TBDs (with owners)

Nothing below is guessed. Each is a value this lane genuinely does not have
yet, with the workstream that owns closing it.

| Id | Open value | Why it matters | Owner |
|---|---|---|---|
| TBD-4 | The registry digest of each newly built public runtime image | An image cannot contain its own final registry digest. The build operator must push it, record the digest in launch/evidence metadata, and inject it as `SPARKRING_IMAGE_DIGEST`; the acceptance gate now fails rather than accepting an identity skip. | runtime build/release operator |
| TBD-5 | Minimum supported NVIDIA driver version (reference cluster: 580.173.02; requirement stated only as "580.x") | Users cannot tell whether their driver is in-matrix | hardware / bring-up |
| TBD-6 | Minimum supported host kernel version (reference cluster: NVIDIA kernel 6.17) | Same | hardware / bring-up |
| TBD-7 | Minimum Docker CE / podman / `nvidia-container-toolkit` versions | Same; also affects whether the podman path is actually supported or merely believed to work | bring-up |
| TBD-8 | **Source closed; live gate open.** The recovered overlay now supplies `B12X_MLA_SPARSE` and `nvfp4_ds_mla` in the public builder. It still needs an ARM64 build and four-Spark functional acceptance. | Build, attest and execute the matrix before reporting a public-functional result | runtime build / acceptance workstream |
| TBD-9 | Whether the pinned adaptive-MTP 2/4 configuration produces **bitwise-identical token ids** across runs. `scripts/context_cache_gate.py` already records that two consecutive restores of a byte-verified entry produced different phrasings, i.e. observed run-to-run nondeterminism in a speculative-decode configuration | Decides whether stage 5's exact-token-id criterion is achievable as specified, or whether the matrix must pin an MTP-off determinism configuration for that stage only | acceptance gate owner |
| TBD-10 | The public-lane performance tolerance band. None exists, and reference-lane numbers must not be used (§4.3) | Until a band is committed, `performance_verdict` is permanently `BASELINE-RECORDED` | acceptance gate owner, after N public-lane runs |
| TBD-11 | Availability and exact response shape of the `/tokenize` endpoint in the pinned vLLM build (the gate uses it to recover output token ids) | If absent, stage 5 fails with an actionable message rather than falling back to hashing text — the fallback is deliberately not implemented | acceptance gate owner |
| TBD-13 | The site schema pins `serving.mtp_tokens` but not the adaptive depth window; the matrix's `adaptive_speculative_tokens_window: 32` is therefore documented here but not machine-checked | An adaptive-MTP window other than 32 would pass the gate while being off-matrix | site-config workstream + acceptance gate owner |
| TBD-14 | The site schema has no field for the attention backend or KV cache dtype, so §2.4's `B12X_MLA_SPARSE` / `nvfp4_ds_mla` rows are documented but not machine-checked (they are also gated on TBD-8) | Two matrix rows are honour-system until either the site schema carries them or the gate reads them back from the running engine | site-config workstream + acceptance gate owner |

Closed: TBD-1 (immutable model revision), TBD-2 (ARM64 base-image digests), and
TBD-3 (DeepGEMM full commit) are pinned in `runtime/runtime-lock.json`. The
earlier "reconcile site-config key names" item is also resolved:
`scripts/sparkring_site.py` landed and the gate consumes its normalised schema
directly (§7), with gate-specific settings moved into the gate config.

---

## 9. Change control

This document is the definition of the matrix. Changing §2, §3 or §4 changes
what "public-functional acceptance" means, so:

- A matrix change invalidates any previously reported result against the old
  matrix. Say so in the change.
- Closing a TBD means replacing it with a pinned value **and** its source of
  truth, not deleting the row.
- The performance band (TBD-10) may only be set from public-lane measurements
  with their own labels. Importing a reference-lane number into the band is a
  claim-discipline violation, not a shortcut.
