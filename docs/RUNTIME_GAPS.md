# Runtime Gaps: our frozen vLLM runtime vs public upstream

> **Recovery update — 2026-07-29:** the safe measured-runtime delta is now
> published under `runtime/patches/00-reference-vllm/`: 59 modified files and
> 12 additions, all fail-closed and byte-matched to the freeze after newline
> normalization. Two unsafe empty-return startup shims were excluded. The
> remaining gap is build/live validation, not missing source.

Audited 2026-07-27 against the frozen production runtime and current
upstream sources. Companion to the Status section of the README.

## What we run

SparkRing's serving image freezes a vLLM runtime identified as
`0.11.2.dev279+…fcc6141.b12x284a2ea.fi25dd814.cu132.20260626` — a vLLM build for
aarch64 / CUDA 13.2 targeting NVIDIA GB10 (DGX Spark, SM121), used to serve
GLM-5.2 and DeepSeek-V4-family models with sparse attention (DSA), NVFP4/MXFP4
quantization, and DSpark speculative decoding.

An audit of the installed runtime against public sources reached the following
conclusions.

## The three-case framing, and where we stand

We assessed the freeze against three possible situations, from best to worst:

1. **Recoverable ancestry** — every component of the frozen runtime resolves to
   a public commit, so the exact baseline can be re-fetched at any time.
2. **Rebuildable delta** — the parts that differ from public code are captured
   as reviewable patches that can be re-applied (or upstreamed) against a
   fresh build.
3. **Blocked on B12X kernels** — some capability exists only as compiled
   kernels whose source we cannot obtain, making the runtime unreproducible.

**Finding: we are in cases 1 + 2. We found no evidence we are in case 3.**

### Case 1 evidence — the ancestry is fully public

Every pinned component in the version string resolves to a public commit:

| Component | Pin | Public source |
|---|---|---|
| vLLM | `fcc6141` | `vllm-project/vllm` @ `fcc614141e5e…` (2026-06-26) |
| B12X kernel library | `284a2ea` | `local-inference-lab/sparkinfer` @ `284a2eae837…` (2026-06-25; repo formerly known as `b12x`) |
| FlashInfer | `25dd814` | `flashinfer-ai/flashinfer` @ `25dd814e037…` (2026-06-25) |

The installed vLLM wheel itself shows no source divergence from upstream at
that commit: all compiled extension modules correspond to stock upstream build
targets, and every Python file that ships in the wheel is byte-identical to
upstream. (Residual risk: compiled binaries cannot be hash-verified without a
reproducing build; nothing in the audit suggests they differ.) The DeepGEMM companion package pin was recovered from its installed
metadata (`2.5.0+2073ddb`) and resolves publicly as well.

### Case 2 evidence — the delta is small, captured, and re-appliable

Everything that differs from upstream vLLM is a **post-install patch overlay**:
roughly 13k lines across 73 Python files, applied to the installed tree between
2026-06-29 and 2026-07-27. The full set is now captured as per-file unified
diffs (now published as the 71 safe recovered operations) and falls into four groups:

- **B12X/sparse-attention integration (~9.3k lines)** — wires vLLM's sparse-MLA
  path (DeepSeek-V3.2/V4, GLM-5.x family) to the public B12X/sparkinfer kernel
  library on SM120/SM121, including DCP (decode context parallel) all-to-all
  and low-bit MLA KV-cache record formats. Parts of this derive from public
  community overlay work in the B12X ecosystem.
- **DSpark speculative decoding (~3.2k lines)** — draft-module speculators for
  DeepSeek-V4/Qwen3 targets plus scheduler and config plumbing.
- **SparkRing operational shims (small)** — env-gated capture-stream behavior
  and single-node executor guards applied by our container entrypoint.
- **Unattributed support edits (~0.8k lines)** — small changes without
  provenance markers; excluded from any public reproduction until attributed
  or rewritten.

### Why we are not blocked on B12X kernels

- The B12X kernel library the overlay calls into is public
  (`local-inference-lab/sparkinfer`), and the exact pinned commit exists there.
- Upstream vLLM has since absorbed the B12X MoE path via official FlashInfer
  releases (public wheel index), removing the standalone-package dependency on
  current mains.
- The only non-public native component we ship is our own NCCL transport shim,
  which is SparkRing-authored (not a third-party blob) and outside vLLM.

## Gap-by-gap status against current upstream (v0.26.x, checked 2026-07-27)

Where upstream has caught up, we can drop overlay pieces on the next rebase;
where it has not, our overlay remains ahead.

| Capability in our runtime | Upstream vLLM status | Reference |
|---|---|---|
| GLM-5.2 (glm_moe_dsa) serving | Merged (via the DSA/DeepSeek-V3.2 family path) | PRs #47410, #46876; Blackwell decode opts #48597 merged then reverted (#49768) — still settling |
| DSpark speculative decoding | Merged and actively maintained | PRs #47216, #47419, #49415; more in flight |
| DCP for sparse MLA | Merged | PR #46076 |
| NVFP4 KV cache (generic) | Merged | PR #40177 and follow-ups |
| Sparse (DSA) indexer improvements | Partial — FP8 indexer and index caching merged; MXFP4 indexer cache still open | PRs #46168, #45863 merged; #48558 open |
| SM121/GB10 sparse-MLA backend | **Not merged** — our main remaining gap | open PRs #46055, #47629, #48994 |
| SM121/GB10 in published wheels | **Not merged** | open PRs #31740, #38484, #49904 |
| Packed low-bit MLA KV record formats (NVFP4-family) with per-layer scale calibration | **No upstream counterpart found** | community overlay line; candidate for upstreaming |
| Single-checkpoint hybrid MXFP4 quantization | **No upstream counterpart found** | candidate for upstreaming |
| Env-gated FP8 LM-head | Open PRs only | #35696, #41000 |

## Practical consequences

- **Reproducibility**: the runtime can be reconstructed from public commits
  plus our captured overlay patches. No capability depends on unobtainable
  binaries.
- **Rebase path**: on a rebase to current upstream, roughly half the overlay
  (GLM-5.2 enablement, DSpark, DCP, generic NVFP4 KV) is superseded by merged
  upstream work; the SM121/GB10 sparse-MLA backend and the low-bit MLA KV
  record formats are the durable deltas to carry or upstream.
- **Action items**: continue improving attribution for the small support-edit
  tail, rerun the corrected pinned ARM64 image through four-Spark API/request
  acceptance, and track the
  open SM121 PRs upstream as the retirement path for the largest patch.
