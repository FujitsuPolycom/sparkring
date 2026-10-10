# Release and recommendation changes

Merging implementation and recommending a deployment are separate decisions.
Useful code, documentation and research profiles can be reviewed without owning
Spark hardware. Maintainers own release qualification.

For a publication or default promotion, identify the exact source/dependency
pins, model revision and quantization, image digest, TP/DCP, topology and workload.
Verify source/build inputs and installed paths, then perform the relevant
hardware correctness, failure/recovery and serving checks on authorized hosts.
State cold/warm cache behavior, prompt lengths, speculation, concurrent load,
units, repetitions and limitations for performance claims. A component test is
not full-stack evidence.

Preserve public image names and immutable digests. Select a distinct release
when build inputs change; do not rewrite a publication receipt, retag evidence,
or update preserved hashes merely to pass CI. Record a rollback image and its
compatible site configuration before changing an operational default.

A shared installer image lock (`sparkring-installer-image/v2`),
`runtime/releases/<release>/installer-image.json`, pins one image and lists the
installer profiles admitted to run on it. Admitting a profile to that image
changes only the lock's sorted `profiles` list: the lock and its `sha256` entry
in the same directory's `release.json` are rewritten in place, and the release
keeps its identifier because the image and its receipts are unchanged. Any
other change to the lock, such as its image, a receipt digest or a size, needs
a distinct release. Each installer deployment's identity includes the lock, so
after an admission the next `sparkring install` of any profile on the lock
creates a separate deployment. [Contributing an installer profile](installer-profiles.md)
gives the admission steps. A new installer image release also names the
builder of its final layer in the `releases` field of
[builders.json](../../runtime/images/builders.json); the repository layout check
rejects an installer image lock without one. [Installer image builders](../../runtime/images/installer-images.md)
describes the chain. A GitHub release that publishes an installer image adds
its tag and the image's release name to
[installer-releases.json](../../runtime/releases/installer-releases.json), so
`sudo sparkring install --image TAG` selects that image. An image whose own
layer adds a capability that a serving setting needs, such as the
shared-memory reader window of `--save-cpu`, is listed in
[installer-capabilities.json](../../runtime/releases/installer-capabilities.json)
([capabilities](../../runtime/images/installer-images.md#capabilities)); an
image derived from a listed one needs no entry.

An image that carries SIRCL ring sessions has a `sparkring-installer-image/v3`
lock ([image_lock.py](../../runtime/common/image_lock.py)). It keeps every v2
field and adds:

| Field | Meaning |
|---|---|
| `line` | The image line: `kraken`, images built on Local Inference Lab's `karmic-kraken-beta` vLLM and B12X branches |
| `transports` | The collective transports the image carries, sorted: `libsircl`, `prepared` and `sircl`. This package admits v3 locks that list `prepared`, whose v2 fields keep their meaning |
| `sircl` | The SIRCL layer: package version, native ABI, wheel name and SHA-256, the two prebuilt libraries (path, SHA-256, source digest), the layer receipt, the tuning key a measured table must match, and the pinned vLLM builds the image's vLLM matches |
| `tuning_defaults_sha256` | The SHA-256 of [sircl-tuning-defaults.json](../../runtime/common/sircl-tuning-defaults.json) at the image's build |
| `archived` | `true` for an archived release, which `--image` still selects and which is never the default |
| `libsircl` | Present only when `transports` lists `libsircl`: the [libsircl layer](../../runtime/images/installer-images.md#libsircl-layer)'s version, the git tree id of the libsircl source it was built from (`source_tree`; a layer built from a vendored libsircl snapshot records the snapshot's tree digest as `snapshot` instead), library path and SHA-256, NCCL API level, whether the library has the fail-stop mode, vLLM plugin path and SHA-256, and layer receipt |
| `vllm_plugins` | Present only when a [derived layer](../../runtime/images/installer-images.md#derived-layers) added vLLM general plugins: each plugin's entry-point name and version, other than the images' own `b12x_loader`, `sparkring_status`, `sircl` and `libsircl`. `sparkring install` runs a profile whose `VLLM_PLUGINS` names another plugin only on a lock that lists it |

v1 and v2 locks keep validating, and published releases keep their v2 lock
bytes. Compose exports and the Install Builder use a v3 image through its v2
fields, on the prepared transport. The
[SIRCL layer builder](../../runtime/images/installer-images.md#sircl-layer)
writes the v3 lock. `sparkring install` without `--image` uses the newest
kraken-line v3 release that carries SIRCL, is not archived and is published in
[installer-releases.json](../../runtime/releases/installer-releases.json); until
one is, it uses the default v2 image on the prepared transport. Listing a v3
release there therefore changes the default image and transport of every new
installation; record the rollback image, normally the v2 release it derives
from, in the release notes.

Installer image release names read
`dev-DATE-CHANGE-cudaNNNN-ncclNNNN-statusNNN`. `dev` marks SparkRing's
development image series; a GitHub release tag such as `2026.10.1` publishes
one of them through
[installer-releases.json](../../runtime/releases/installer-releases.json), and
the image keeps its name. DATE is the day the image was composed. CHANGE,
where present, names what the image adds over its parent, such as `spinwait`
or `mimovision`, or its image line: `kraken` is the line built on Local Inference Lab's
`karmic-kraken-beta` vLLM and B12X branches, which a v3 lock records as
`line`. `cudaNNNN` and `ncclNNNN` are the CUDA and NCCL releases the image's
toolchain layer selects (`cuda1342` is CUDA 13.4.2, `nccl2323` NCCL 2.32.3),
and `statusNNN` is the runtime-status dashboard version, recorded as the
lock's `status_version` (`status034` is 0.3.4). New names spell CUDA
`cudaNNNN`; the published image
`dev-20261008-kraken-csf-sircl-libsircl-cu1342-nccl2323-status034` spells
CUDA 13.4.2 `cu1342` and keeps that name. Names are identities, not
descriptions: an image whose inputs change takes a new name. Earlier images
and guides use Local Inference Lab's own release numbers: R33, R35 and R37
are releases of its Jovian Judgement vLLM line, `jovian-r33` is SparkRing's
ARM64 rebuild of the R33 composition, and `Jovian Judgement Community R10` is
a separate community release; the `r8` of `sparkring-glm53-jj-r8-gb10`
numbers that builder's interface, not a release. LIL abbreviates Local
Inference Lab in composition names such as `lil-r37-shared`, which build on
R37; `lil` alone names a separate launcher tool
([lil integration](../../integrations/lil/README.md)).

The default tuning table's rows are the accepted defaults until a
measurement replaces them: the `pair`, `path-4`, `cycle-4` and `cycle-8` rows
are measured, each row names its evidence, and other group sizes, such as a
cycle of six, run on SIRCL's own rules through the `path` and `cycle` rows. The
table names the SIRCL version and ABI of `spark_transport/sircl` and, under
`compatible`, any other build of that ABI whose sessions keep the rows, each
with the reason its difference changes no choice a row makes; a release whose
SIRCL layer is neither runs none of them. A change of the package's SIRCL
version fails `test_transport.py` until the table names that version, and
every shipped v3 lock's SIRCL build must be one the table names. To replace one, or to measure a size
that SIRCL's rules serve while no measurement exists, run `sudo sparkring fabric tune --execute` on the owner's
fabric of that shape with the release's image, copy Node A's
`/var/lib/sparkring/controller/sircl-tuning.json` and
`/etc/sparkring/fabric/sircl-tuning/`, and run
`python scripts/promote_sircl_tuning.py --measured COPY --tables DIRECTORY --row cycle-4`.
A row measured on several fabrics of the same shape, such as two separate
cycles of four Sparks, comes from SIRCL ring-harness tunes of each fabric's
whole group with the release's image: run
`python scripts/promote_sircl_tuning.py --row cycle-4 --ring RING_A --ring RING_B --image-lock LOCK`,
where each ring directory holds the fabric document, the Sparks' drivers and
the harness's tune run. The merged SIRCL table judges every candidate by its
slower fabric's median, drops a candidate that some fabric did not measure
exactly, and the command prints every size at which the fabrics' own tables
choose differently. Either command prints the change; `--write` writes the row's SIRCL
table to `runtime/common/sircl-tuning/` and the row, as `measured` with an
`evidence` text naming the fabrics, runs, image, SIRCL build and drivers, to
the default table ([promote_sircl_tuning.py](../../scripts/promote_sircl_tuning.py)).
Sessions of a SIRCL build the table lists as `compatible` take the row's
settings, and only sessions of the measured build take the SIRCL table's
choices. Record the fabric, image, drivers and harness run in the release's
performance record.
The image built afterwards records the new `tuning_defaults_sha256`.

A release's SparkRing package carries the relay marker compiled for arm64.
Build the release package on an arm64 Linux host with `gcc` and
`libibverbs-dev`:

```bash
python3 scripts/build_deb.py --relay-marker require
```

A missing compiler or header, or another host architecture, stops the build.
Beside the package it writes `sparkring-relay-marker-<source>-arm64` and its
`.sha256` file, where `<source>` is the first 12 hexadecimal digits of the
marker source's SHA-256. When
[`relay-marker-artifact.json`](../../spark_transport/fabric/relay-marker-artifact.json)
names no binary for that source, publish the binary as a release asset and
record its SHA-256 and download URL there; package builds without a compiler,
`install.sh` among them, take that binary. When the record names one, the
release build stops if the binary it compiled differs, so every package of one
marker source carries the same binary. A change to `relay_marker.c` updates the
record's `source_sha256` and sets its binary and URL to `null` until the
release build publishes the new binary.

Publishing a GitHub release that is not a prerelease runs the
[Install Builder pages workflow](../../.github/workflows/compose-builder-pages.yml):
it builds the [Install Builder](../operations/compose-builder.md) from the
release tag, compares the page's engine with `compose.build`, and publishes the
page to the repository's GitHub Pages site only when every case matches.

Prepare the candidate and local PR description before requesting adoption.
Pushes, GitHub posts, merges, image publication and cluster operations require
the user's applicable authorization. Normal reviewed Git history makes rollback
possible; repository adoption does not require rewriting main or forcing a push.

A partial lifecycle fix may be accepted as a mitigation. For example, increasing
peer-response silence tolerance does not handle disappearance of the local
management address. Describe the solved condition and retain a follow-up for
remaining behavior before declaring an issue resolved. Do not auto-close issues
solely because a reporter lacks hardware evidence.
