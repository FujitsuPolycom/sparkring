# SparkRing 2026.10.2

This release's installer image runs SparkRing's collectives on SIRCL ring
sessions with NCCL off, carries libsircl (an NCCL-API library built from
SparkRing's source), and serves models on fabrics of two to eight DGX Sparks.
The image is `dev-20261009-kraken-csf-sircl032-libsircl-plugins`, built on
release 2026.10.1's image. The
[release record](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/runtime/releases/dev-20261009-kraken-csf-sircl032-libsircl-plugins/README.md)
has its identity, layers, evidence and gates.

> **Pending:** lines marked **Pending** are filled in, with their records,
> before this release is published.

## What changed

### SIRCL carries the collectives

- On this image `sudo sparkring install` runs every tensor-parallel and
  decode-context-parallel collective on SIRCL ring sessions, with NCCL off,
  wherever `sudo sparkring setup` recorded the fabric and its relay table.
  The plan says `Transport: sircl on every collective, NCCL off (...)`.
- After the model answers, the installer reads every rank's SIRCL receipts
  and NCCL log lines. The summary card shows the verdict, for example
  `Transport:   sircl, NCCL: absent`.
- `--nccl auto` lets NCCL carry what the cabling allows. `--transport prepared`
  keeps 2026.10.1's prepared B12X RoCE transport, and `--transport nccl` runs
  vLLM's PyNccl alone on a cabled pair or a whole ring
  ([transport and receipts](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/docs/operations/install-reference.md#transport-and-receipts)).
- SIRCL's settings come from a default tuning table with measured rows for a
  pair, a path of four, a ring of four and a ring of eight. `sudo sparkring fabric tune --execute`
  measures one for your cables
  ([measure the tuning table](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/docs/operations/install-reference.md#measure-the-tuning-table)).
- Status: SIRCL's one-shot all-reduce and all-gather are qualified on an
  eight-Spark ring; serving through vLLM on SIRCL is research-only
  ([SIRCL status](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/spark_transport/sircl/STATUS.md)).

### libsircl, an NCCL-API library built from source

- libsircl 0.6.0 implements the public C API of NVIDIA NCCL 2.32 on SIRCL's
  wire protocol. Its SONAME is `libnccl.so.2`, so PyTorch, vLLM and
  nccl-tests load it in place of NCCL without changes. It is an independent
  implementation: it is not NVIDIA NCCL and contains no NCCL implementation
  source.
- The image compiles it from the repository's `spark_transport/libsircl`
  with its four kernel packs and its fail-stop mode.
- `--transport libsircl` runs vLLM's PyNccl on libsircl, and
  `--image REF --transport libsircl --plan` plans a stock ARM64 vLLM image
  with libsircl as its NCCL. Both are research-only
  ([libsircl in the installer](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/docs/architecture/libsircl.md)).

### Installer

- **Two to eight Sparks.** Setup forms a pair, a line or a ring of up to eight
  Sparks, installs one relay table so that every Spark reaches every other
  through ConnectX-7 hardware relays, restores it at every boot and records
  the fabric document. The relay marker ships prebuilt in the package
  ([fabrics](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/docs/operations/install-reference.md#fabrics-of-up-to-eight-sparks)).
- **Several models on one fabric.** `--on 0-3`, `--on 4-7` or `--on 0,1` puts
  a model on consecutive Sparks; each group serves its own API
  ([models on part of the fabric](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/docs/operations/install-reference.md#models-on-part-of-the-fabric)).
- **Checks.** `sudo sparkring fabric verify` checks links, routes and relays.
  `sudo sparkring check` sends functional requests, judges the transport
  receipts and flags a rank whose GPU clock is throttled; `--report DIR`
  writes a test report to attach to an issue
  ([check](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/docs/operations/commands.md#check)).
- **Memory before a start.** Every Spark drops its page cache and waits for
  settled memory before a model starts. A start that vLLM would refuse for
  memory is refused first, with each Spark's available memory.
- **Distribution.** On a recorded fabric, the image and the checkpoint spread
  along the cables from Node A (implemented; tested with simulated Sparks).
- **Existing rings.** `sudo sparkring setup --node ... --adopt` records a ring
  that is already cabled and addressed and adds only the relay table's missing
  routes, neighbors and filters; links, addresses and connections stay as they
  are. `sudo sparkring setup --re-form` sets recabled Sparks up again.
- **Warnings.** Install plans print the enhancement catalog's warnings, such as
  dense MXFP8 linears without `--linear-backend b12x`.

### Fastest measured settings

- **GLM-5.3 on eight Sparks** (`glm53-nvfp4-tp8`): two-token MTP with MXFP8
  draft linears and W4A16 draft experts, MXFP8 dense linears on the B12X
  linear backend, the MXFP8 LM head, the GLM-5.3 plugins, SIRCL's fused
  all-reduce and RMSNorm and column gathers, 16 sequences and a 1M-token
  context. Three draft tokens, adaptive draft counts and the DFlash2 drafter
  measured no faster. `glm53-nvfp4-tp8-dcp1` drops decode-context
  parallelism for a 524,288-token context (research-only).
- **GLM-5.3-Flash on two and four Sparks**: on the CSF checkpoint, two CTAs
  per SM for small-batch W4A16 experts. On two Sparks, asynchronous
  scheduling for every checkpoint and, on CSF, KDA prefill coalescing and the
  four-Spark profile's draft settings.
- **DeepSeek-V4.1-Flash on four Sparks**: SIRCL's fused all-reduce and
  RMSNorm.
- **Qwen3.8-Flash-Next** keeps QAD step 5500 as its default checkpoint.
- **SIRCL tuning rows**: a pair and a path of four run ring schedules above a
  64 KiB one-shot limit with 1 MiB link pieces; a ring of four takes the
  merged tune of two rings of four (8 link slots of 1 MiB; the ring
  all-reduce from about 1.2 MiB per rank)
  ([record](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/performance/records/transport/sircl-cycle4-tune-two-rings-20261009.md));
  a ring of eight runs a 1 MiB all-reduce capacity, a 28 KiB one-shot limit
  and 16 link slots of 512 KiB.

## Measured results

Every result is lane public-functional (images and harnesses of this
repository) and none is serving qualification. The table's results are
research-only, on eight DGX Sparks (GB10) cabled as one ring with ConnectX-7
RoCE, GPU clocks locked at 2,418 MHz, through the serving A/B runner rather
than `sparkring install`; two measured starts each, temperature 0, no prompt
context. The items below it state their own conditions.

| Model, profile | Checkpoint | Sparks | Image | Output tok/s, 1 / 2 / 4 / 8 streams | First token, 32K prompt |
|---|---|---|---|---|---|
| GLM-5.3, `glm53-nvfp4-tp8` | NVFP4 `b472e4ee53f6` | 8, the ring | `27e9f75c0d09` | 50.7 / 73.9 / 102.4 / 151.4 | 24.4 s |
| GLM-5.3-Flash, `glm53-flash-nvfp4-spark-tp4` | CSF `dec48abd33ef` | 4, a path of the ring | `816c6d6a7e96` | 68.3 / 106.5 / 152.4 / 224.3 | 10.5 s |
| GLM-5.3-Flash, `glm53-flash-nvfp4-spark-tp2` | CSF `dec48abd33ef` | 2, a cabled pair | `816c6d6a7e96` | 42.0 / 61.4 / 88.9 / 133.5 | 15.3 s |
| Qwen3.8-Flash-Next, `qwen38-flash-next-qad-tp4` | QAD step 5500, MXFP8 attention | 4, a path of the ring | `816c6d6a7e96` | 91.0 / 143.6 / 218.5 / 310.6 | 6.8 s |
| Qwen3.8-Flash-Next, `qwen38-flash-next-tp2` | QAD step 5500, MXFP8 attention | 2, a cabled pair | `816c6d6a7e96` | 63.0 / 100.5 / 153.0 / 217.0 | 8.4 s |
| DeepSeek-V4.1-Flash, `deepseek-v41-flash-tp4` | `dba1be0a40aa` | 4, a path of the ring | `816c6d6a7e96` | 61.9 / 94.3 / 138.8 / 181.9 | 7.6 s |

- GLM-5.3 at 16K and 32K context: 44.2 / 64.0 / 89.8 / 135.7 and 47.9 /
  65.9 / 99.8 / 143.7 tok/s; first token 10.0 s for an 8K prompt and 11.9 s
  for 16K
  ([record](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/performance/records/images/dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-glm53-tp8-speculation-20261009.md)).
- The two- and four-Spark rows
  ([record](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/performance/records/images/dev-20261008-kraken-csf-sircl-libsircl-tp2-tp4-matrix-20261009.md))
  ran on groups inside the eight-Spark ring. A four-Spark path's end ranks
  reach each other through relays; a four-Spark ring has no measurement here.
- Both images precede the release image: `27e9f75c0d09` has SIRCL 0.3.1 and
  libsircl snapshot `a3477af2`; `816c6d6a7e96` has SIRCL 0.3.0.
- libsircl on the ring of eight (snapshot `a3477af2`, nccl-tests v2.21.1, NVIDIA
  NCCL 2.32.3 as the baseline): every result exact; a 256 MiB all-reduce under
  ring schedules took 19.3 ms (24.3 GB/s bus bandwidth) against NCCL's
  19.9 ms (23.6 GB/s)
  ([libsircl status](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/spark_transport/libsircl/STATUS.md)).
- libsircl 0.6.0 of the release image on the ring of eight, without NCCL in
  the same gate: every check exact; the cycle plan runs ring schedules from
  8 MiB by default; a 256 MiB all-reduce took 19.3 ms (24.3 GB/s) and a 4 KiB
  one 18.2 µs; broadcast stays at 1.9 GB/s
  ([record](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/performance/records/transport/libsircl-ring8-image-1a8c10354eb0-20261009.md)).
- The clean-room acceptance audit passed SIRCL 0.3.2 and libsircl 0.6.0
  with no line to rewrite, and the three GLM-5.3 plugins
  ([record](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/performance/records/repository/clean-room-acceptance-audit-20261009.md)).
- `sparkring install` on the eight-Spark ring with the release image (GPU
  clocks not locked, one installation each): GLM-5.3 on all eight Sparks,
  GLM-5.3-Flash CSF on four and on two, and Qwen3.8-Flash-Next on two; every
  installation passed `sparkring check`'s functional checks and three
  known-answer questions. GLM-5.3 decoded 47.7 tok/s at one stream and 16K
  context. These are the first CSF installations that passed the installer's
  checks ([record](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-installer-ring8-20261009.md)).
- Two rings of four (the eight Sparks recabled): `sudo sparkring setup --re-form`
  verified both fabrics; SIRCL's quick tune of a ring of four passed on both,
  every output exact; libsircl's gate passed on both, with a 256 MiB
  all-reduce of 16.55 ms (24.3 GB/s) on each
  ([setup and tune](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/performance/records/transport/ring4-setup-and-sircl-tune-20261010.md),
  [libsircl](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/performance/records/transport/libsircl-ring4-image-1a8c10354eb0-20261010.md)).
- **Pending:** the installer scenarios on the rings of four.
- **Pending:** in-place TP4 measurements on a ring of four of GLM-5.3-Flash
  CSF, DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next.

## Known limitations

- `--transport libsircl` is research-only, has no serving measurement, runs
  without decode-context parallelism, and its receipts are not judged.
- libsircl's broadcast on a ring of eight reaches 1.9 GB/s against NCCL's
  24.2 GB/s (unsupported). The default `sircl` transport does not use
  libsircl.
- On this image the GLM-5.3-Flash profiles install the CSF checkpoint, which
  is research-only: one installation of each profile passed the installer's
  checks. `--checkpoint nvfp4-spark` installs NVFP4-Spark.
- The eight-Spark profiles are research-only. `qwen38-flash-next-qad-tp8`
  does not run on this image.
- MiMo-V2.6-Flash-MOPD and Swift-1.5 profiles run on SIRCL with no SIRCL
  serving record.
- A ring of four's tuning row comes from one quick tune per ring, with no
  serving comparison against SIRCL's own rules.
- The serving A/B measurements locked GPU clocks; the installer does not.
- The model's status dashboard is runtime-status 0.3.4: it does not show
  SIRCL's sessions or the collective transport. `sudo sparkring status` and
  `sudo sparkring check` do.

## Upgrade from 2026.10.1

1. Install the 2026.10.2 package on Node A:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/2026.10.2/install.sh | bash -s -- --ref 2026.10.2 --package-only
   ```

2. Record the fabric if this cluster has no fabric document. When
   `sudo sparkring fabric show` says so, run `sudo sparkring setup --plan`,
   read it, then `sudo sparkring setup`.
3. Preview the installation and look for this release's image and
   `Transport: sircl`:

   ```bash
   sudo sparkring install --profile PROFILE --plan
   ```

4. Install. A Spark that holds the 2026.10.1 image downloads only the
   layers this image adds (**Pending:** their size). GLM-5.3-Flash profiles
   also download the CSF checkpoint unless you add `--checkpoint nvfp4-spark`.

   ```bash
   sudo sparkring install --profile PROFILE
   sudo sparkring check --report ~/sparkring-report
   ```

5. Optional: tune SIRCL to your cables while no model serves, then run the
   install command again. It takes up to half an hour per group shape.

   ```bash
   sudo sparkring fabric tune --execute
   ```

To go back, `sudo sparkring install --profile PROFILE --image 2026.10.1`
runs the profile on 2026.10.1's image and transport, and
`--transport prepared` keeps this image on the prepared transport
([another image](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/docs/operations/install-reference.md#another-image)).
