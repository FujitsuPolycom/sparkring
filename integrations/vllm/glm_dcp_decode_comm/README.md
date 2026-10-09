# glm_dcp_decode_comm

A vLLM general plugin (distribution `glm-dcp-decode-comm` 2.0.0, entry point
`glm_dcp_decode_comm = glm_dcp_decode_comm:register`) that changes how
GLM-5.3's DSA attention layers run their decode context parallel (DCP)
collectives when those collectives run on a SIRCL DCP session. Every item is
exact: it produces the bits the image's code produces.

The [GLM-5.3 plugin layer](../../../runtime/images/derive_glm53_plugins.py)
installs it beside `glm_dsa_indexer_split` and `glm53full_speedups`, and
profile [`glm53-nvfp4-tp8`](../../../profiles/glm53-nvfp4-tp8/README.md)
loads it with every item off.

Status: **research-only**.

| Evidence | Conditions | Result |
| --- | --- | --- |
| CPU suite (`tests/`) | host only; the image's vLLM and b12x files and the pinned SIRCL tree on disk; SIRCL's CUDA stand-in for the kernel's host side | the five flags in all 32 combinations on indexer and non-indexer layers, the wire format, the code edits, the pins, every refusal path, and the import hook beside `glm53full_speedups`'s hook in either order |
| GPU checks (`tests/gpu_checks.py`), RTX 5090 | one RTX 5090 (SM120, not GB10) under WSL; torch 2.10.0+cu128, Triton 3.6.0; the image's kernels from copies of serving image 816c6d6a7e96's `common/kernels.py` and `dcp.py`; SIRCL's emulated DCP groups over its in-memory verbs stand-in | kernels: 12 of 12 checks, 0 differing words (`rope_cat`: 36 cases; `wire_combine`: 2, 4 and 8 ranks, 4 to 16 heads, both LSE bases). Packed all-to-all on `path:0-3` and `path:0-1`: 11 of 11 checks each, eager at 1 to 31 rows and in CUDA graphs with 3 replays, every rank equal to the image's combine on SIRCL. Registration: not run (no pinned vLLM). |
| GPU checks, GB10 | serving image 816c6d6a7e96 on one Spark, 2026-10-09 (private record: GB10 gate log of these checks) | kernels: 13 of 13 checks, 0 differing words. Packed all-to-all on `path:0-3` and `path:0-1`: 13 of 13 checks each. Registration: 5 of 5 checks in each of the 4 flag sets, vLLM's attention module imported in 10 to 13 s with lazy CUDA module loading. |
| Audit while serving, GB10 | image 27e9f75c0d09 (SIRCL 0.3.1), profile `glm53-nvfp4-tp8` at TP8/DCP4 on the eight-Spark ring through the serving A/B runner, the five switches and `GLM_DCP_DECODE_AUDIT=1`, one warm-up start with CUDA graphs ([record](../../../performance/records/images/dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-decode-ab-20261009.md)) | 0 differing words in each of the five checks on all eight ranks; no `PatchRefused` |
| Decode effect, GB10 | the same image, profile and Sparks, the five switches on against off, arm `S+`, GPU clocks locked, one measured start each ([record](../../../performance/records/images/dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-decode-ab-20261009.md)) | decode engine steps/s -6.2 % to +3.6 % of the switches off across 12 cells: no consistent gain; time to first token at 16K and 32K higher with them on |
| Serving qualification | an installer deployment with the switches on | not run |

The GPU checks ran against SIRCL 0.3.0, the tree of image 816c6d6a7e96,
and on a package that differs from this one in its SIRCL pins, the import
hook's recursion guard (`PatchOnImport.find_spec`), one refusal message and
documentation, with check scripts that differ in module names and in the
loader's lookup of the image sources. The CPU suite covers both package code
changes and passes against SIRCL 0.3.1 ("Pins" lists what that release
changes in the pinned files). The decode record is the only measurement of the plugin's effect on decode
step time; with one measured start per setting it shows no consistent gain.

## Requirements

- Serving image `816c6d6a7e96` (`sparkring-dev/kraken:csf-sircl-libsircl-20261008`):
  - the vLLM `0.1.dev21553+gab86b7073` wheel with the Python sources of vLLM
    bc9ea774 in its CSF-sources layer;
  - b12x 1.5.0 at B12X cc36aa6f;
  - GLM-5.3 (`glm_moe_dsa`) with B12X sparse MLA and a BF16 query.
- The pinned SIRCL build (`SIRCL_VERSION` in `__init__.py`, "Pins" below) as
  the image installs it (`dist-packages/sparkring_sircl`), with SIRCL's vLLM
  adapter owning the DCP group (`SIRCL_GROUPS` including `dcp`; SIRCL's
  default is `tp,dcp`). Image `816c6d6a7e96` itself carries SIRCL 0.3.0, so
  the items run on an image derived from it with the pinned SIRCL layer; on
  any other SIRCL build an item flag refuses at startup.
- `--decode-context-parallel-size` above 1, the `a2a` DCP combine (vLLM's
  default for GLM-5.3), no prefill context parallelism.
- `glm_dcp_decode_comm` in `VLLM_PLUGINS`.

## Items

Each item has its own flag: `0` or unset is off, `1` is on, and any other
value refuses at startup. Every item acts only on decode-only batches, eager
or in CUDA graph capture; prefill and mixed batches keep the image's path.

| Flag | What changes |
| --- | --- |
| `GLM_DCP_DECODE_QUERY_PACK` | One Triton kernel (`kernels.rope_cat`) builds the DCP query `[ql_nope \| RoPE(q_pe)]` right after the `W_UK` absorption. It replaces the `torch.cat` before the query all-gather, and the whole `fused_q` launch on layers without the DSA indexer. The query is gathered straight into the attention workspace's query buffer as one SIRCL all-gather op, so the attention skips its query copy. |
| `GLM_DCP_DECODE_OVERLAP` | Implies the query pack. Every collective of the DCP session runs on a dedicated communication stream. On DSA indexer layers the query all-gather is issued right after `W_UK` and joined only before the attention. |
| `GLM_DCP_DECODE_WK_OVERLAP` | On calls that run the DSA indexer, the indexer's `wk` GEMM runs on a side stream beside the latent projection. |
| `GLM_DCP_DECODE_SELECTION_REUSE` | The conversion of the shared top-k to this rank's physical slots is computed once per DSA indexer layer and reused by the layers that follow it. |
| `GLM_DCP_DECODE_A2A_FUSED` | The all-to-all combine's pack runs inside the scatter kernel's staging (`_scatter_pack_cute.py`, one scatter op on the DCP session). The combine (`kernels.wire_combine`) reads the own share in place: one Triton kernel fewer per layer. It applies only where the image's combine is one scatter op; larger messages keep the image's combine. |

Other settings:

| Variable | Meaning |
| --- | --- |
| `GLM_DCP_DECODE_COMM_PRIORITY` | CUDA priority of the communication stream, -5 to 0 (default -1). |
| `GLM_DCP_DECODE_AUDIT` | `1`: qualification mode (needs an item flag). The image's computation of every value an exact item replaces also runs, and differing words are counted on the device. Results are logged as `glm_dcp_decode_comm audit ...` lines at most every 10 s from an eager forward, and at exit. Eager calls add one reference query all-gather and, with the fused combine, one reference all-to-all per layer, so audit mode is not for timing. |
| `GLM_DCP_DECODE_QUERY_FP8`, `GLM_DCP_DECODE_A2A_FP8`, `GLM_DCP_DECODE_RESEARCH_LOSSY`, `GLM_DCP_DECODE_QUERY_FP8_TILE` | **unsupported**: lossy FP8 payloads that this build does not carry. Any value other than `0` or unset refuses at startup. |

With every item flag off, `register` verifies nothing and patches nothing, so
loading the plugin with its items off cannot refuse and leaves the image's
code unchanged.

## Qualification

Qualification is a serving run with the items on and audit mode, on the
profile's own launch otherwise unchanged. Its container environment adds:

```text
GLM_DCP_DECODE_QUERY_PACK=1
GLM_DCP_DECODE_OVERLAP=1
GLM_DCP_DECODE_WK_OVERLAP=1
GLM_DCP_DECODE_SELECTION_REUSE=1
GLM_DCP_DECODE_A2A_FUSED=1
GLM_DCP_DECODE_AUDIT=1
```

It passes when every rank logs the startup line `glm_dcp_decode_comm 2.0.0:
enabled ...` naming the five items and audit mode, no `PatchRefused`, and
every `glm_dcp_decode_comm audit cuda:N: CHECK: ...` line of every rank
reports `0 differ` with a nonzero call count for each of the five checks
(`query`, `gathered_query`, `wk`, `selection`, `combine`). Inside a CUDA graph
capture, audit mode adds only the comparisons that need no reference
collective; `gathered_query` and `combine` count eager calls. If either shows
no calls, a second audit run with `--enforce-eager` covers them on serving
traffic. The final counters are logged when the workers exit, so stop the
server cleanly.

A timing run uses the same five flags without `GLM_DCP_DECODE_AUDIT`.

## What it refuses

- **At startup (`register`)**, it refuses unless every pinned file matches.
  The pins are 13 image files, 17 SIRCL files, the image's `attention.py`
  that it edits, and its own modules (`FILE_CHECKS`, `ATTENTION_SHA256` and
  `PACKAGE_SHA256` in `__init__.py`). Each pin states the behavior relied on.
  Digests read CRLF line endings as LF. It also refuses when the SIRCL tree's
  `__version__` is not `SIRCL_VERSION`, when another patch changed the edited
  methods, when a wrapper other than vLLM's `eager_break_during_capture`
  wraps them, and on any malformed or unsupported setting.
- **At a DCP layer's first decode call**, every rank of the group raises
  `PatchRefused` unless the group's device communicator, collectives and
  session are the pinned SIRCL build's. That means `SirclCudaCommunicator`,
  its adapter's `SirclDcpCollectives` and that object's
  `RoceOneshotAllReduce` with the scatter collectives, each loaded from the
  verified files.

A model configuration that the items do not cover leaves the layer
unchanged and is logged once per reason:
- no DCP;
- prefill context parallelism;
- an attention backend other than B12X sparse MLA;
- an FP8 query;
- the B12X PCIe DCP transport;
- padded heads;
- latent or rotary widths other than 512 and 64.

`glm53full_speedups` edits `DeepseekV32Attention.__init__` in the same
module. The two plugins' import hooks each patch that module once, in either
registration order.

## Pins

The pins describe two builds:

- the image's vLLM and b12x files: serving image `816c6d6a7e96`;
- SIRCL: package version 0.3.1, SparkRing's
  [`spark_transport/sircl/sparkring_sircl`](../../../spark_transport/sircl/sparkring_sircl)
  at commit `694b94c2` (Git tree `e21f5f3b587a` of `sparkring_sircl`;
  ring-session tree `ed8209a3a698` by the SIRCL sync tool's digest).

Against SIRCL 0.3.0 (image `816c6d6a7e96`), 0.3.1 changes five of the 17
pinned files: the version in `__init__.py`; aligned working buffers around
the public ops of `oneshot/runtime.py` and `oneshot/_scatter_ops.py` for a
rank whose pointers are not 16-byte aligned; the library's local feature
identity in `oneshot/_roce_proxy.c`; and a comment in `vllm/executor.py`. No
attribute the plugin reads, no statement its kernel restates and no op code
it relies on changes (`tests/test_dcp_decode_interfaces.py` reads them from
the pinned source).

The CPU suite checks the SIRCL pins against this checkout's SIRCL tree when
that tree's version is `SIRCL_VERSION`, so a change to a pinned file of that
release fails it until the plugin is re-pinned. A checkout whose SIRCL tree is
another version skips the SIRCL tests and names both versions;
`GLM_DCP_DECODE_SIRCL_ROOT` then names a tree of the pinned version.

### Re-pinning

`tools/refresh_pins.py` records or checks the pins. From the repository root:

```bash
# SIRCL only: the repository's SIRCL tree (no vLLM or b12x tree needed)
python integrations/vllm/glm_dcp_decode_comm/tools/refresh_pins.py --sircl-only \
  --sircl spark_transport/sircl/sparkring_sircl [--check]

# every pin: the image's vLLM and b12x package directories and a SIRCL tree
python integrations/vllm/glm_dcp_decode_comm/tools/refresh_pins.py \
  --vllm IMAGE/vllm --b12x IMAGE/b12x --sircl spark_transport/sircl/sparkring_sircl [--check]

# the package's own modules after an edit of one of them
python integrations/vllm/glm_dcp_decode_comm/tools/refresh_pins.py --package-only
```

`--check` changes nothing and exits 1 when any pin differs. A re-pin is a
different build of the plugin's dependencies: the reason that `FILE_CHECKS`
states for every changed file must still hold in the new file, the SIRCL
version under "Pins" above must be updated, and the CPU suite, the GPU checks
and the qualification run again.

## Files

| Path | Content |
| --- | --- |
| `glm_dcp_decode_comm/__init__.py` | settings, pins, the code edits of `DeepseekV32Attention.forward` and `_sparse_indexer_and_attn`, registration, and the helpers placed into the attention module |
| `glm_dcp_decode_comm/runtime.py` | worker-side behavior: streams, the session check, the query, gather, selection and combine paths, and audit counters |
| `glm_dcp_decode_comm/kernels.py` | the Triton kernels `rope_cat` and `wire_combine`, which restate the image's statements |
| `glm_dcp_decode_comm/_scatter_pack_cute.py` | the CuTe DSL packed all-to-all, the pinned SIRCL build's scatter op with the pack in its staging |
| `glm_dcp_decode_comm/layout.py` | the query and wire-format geometry and the kernel's index helpers (pure Python) |
| `glm_dcp_decode_comm/reference.py` | torch references of the image's computations and of the plugin's replacements |
| `dist-info/glm_dcp_decode_comm-2.0.0.dist-info/` | the installed distribution's metadata and its `vllm.general_plugins` entry point |
| `tools/refresh_pins.py` | records or checks every pin against vLLM, b12x and SIRCL trees |
| `tests/` | the CPU suite (`test_dcp_decode_*.py`), the subprocess driver over SIRCL's CUDA stand-in (`scatter_driver.py`) and the GPU checks (`gpu_checks.py`) |

## Checks

```bash
# CPU, from the repository root. The SIRCL tests use the repository's SIRCL
# tree; the tests that need the image's vLLM and b12x files skip without them.
SPARKRING_GLM53_IMAGE_SOURCES=<image-sources> python -m pytest integrations/vllm/glm_dcp_decode_comm -q -rs

# GPU, one device, the pinned SIRCL on PYTHONPATH
python integrations/vllm/glm_dcp_decode_comm/tests/gpu_checks.py --parts kernels,dcp,register \
  [--check-timeout 300] [--setup-timeout 900]
```

`<image-sources>` is a directory with `vllm/` and `b12x/` subtrees holding
the image's Python sources, as the other GLM-5.3 plugins' tests read it
([extraction](../glm_dsa_indexer_split/README.md#tests)).
`GLM_DCP_DECODE_VLLM_ROOT`, `GLM_DCP_DECODE_B12X_ROOT` and
`GLM_DCP_DECODE_SIRCL_ROOT` name single package directories instead.

The GPU checks print a `START` line before every check and `STEP` lines for
its cases. A check over its bound prints every thread's Python stack and
exits 1. They compile the image's reference kernels from the image's own
source files without importing vLLM. Registration imports vLLM's attention
module in one process per flag set, with lazy CUDA module loading.

## Limits

- The fused combine runs only while one scatter op holds the message. That
  limit is the DCP group's relay-safe op size and the session's piece size;
  31 rows of 8 heads per rank fit in 1 MiB on four ranks.
- The query is gathered into the attention workspace only when B12X runs
  the gathered heads as they are. B12X rounds the head count up to whole
  groups of eight, so TP-padded layouts do not qualify. Those gather into a
  buffer of their own, and the attention copies the query in as the image
  does.
- Selection reuse installs only where the B12X implementation has no
  physical selection provider of its own. GLM-5.3's indexer has none.
- While the overlap is on, every collective of SIRCL's communicator passes
  through one Python wrapper. It moves only collectives of DCP sessions.
- Direct `torch.distributed` calls that SIRCL's tripwire carries on a DCP
  group stay on the current stream. vLLM issues them in chunked prefill only,
  which the items do not touch.
