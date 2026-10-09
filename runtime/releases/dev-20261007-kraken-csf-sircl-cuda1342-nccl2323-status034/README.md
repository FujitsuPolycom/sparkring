# Kraken-line image with SIRCL and the CSF-capable sources

Release `dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034` is a
kraken-line installer image whose lock (`sparkring-installer-image/v3`)
carries SIRCL ring sessions and whose vLLM and B12X read the GLM-5.3-Flash CSF
checkpoint without a source overlay. Status: the builders are
**implemented**; this directory selects them ([release.json](release.json),
`source-build`). The image is built on one Spark and loaded on the others;
it is not published to a registry. No hardware check of the built image is
recorded here.

## Layers

| Layer | Builder | Adds |
|---|---|---|
| Parent | `dev-20261004-kraken-cuda1342-nccl2323-status034` (2026.10.1), image `sha256:aba309e4610c…` | vLLM `d51b4181` and B12X `a850d8eb`, CUDA 13.4.2, NCCL 2.32.3, the prepared RoCEnante transport, runtime-status 0.3.4 |
| CSF sources | `installer-kraken-csf-sources`, [derive_kraken_csf_sources.py](../../images/derive_kraken_csf_sources.py) | the 49 Python files that make vLLM `bc9ea774` and B12X `cc36aa6f` ([sources](../../images/compositions/kraken-csf-sources-20261007/README.md)); lock name `dev-20261007-kraken-csf-cuda1342-nccl2323-status034`, kept on the build host |
| SIRCL | `installer-sircl-layer`, [sircl_layer.py](../../images/sircl_layer.py) | the `sparkring_sircl` 0.2.0 wheel in site-packages and the prebuilt `roce_proxy-d6360d260ba540a8.so` and `p2p_proxy-bd01ba45a875e184.so`, compiled in a network-less container of the CSF-sources image ([SIRCL layer](../../images/installer-images.md#sircl-layer)) |

Nothing compiles on a Spark that installs the image: both native libraries
are in the image, and the merged sources need no compiled extension beyond
the parent's. The lock's `sircl.vllm_pins` is
`["sparkring-kraken-beta-20261007-bc9ea774"]`, the build that
`image_lock.checkpoint_problem` requires for the CSF checkpoint.

## Profiles

The lock lists twelve profiles: the nine of the parent lock and three that
run only on SIRCL ring sessions.

| Profiles | Note |
|---|---|
| `deepseek-v41-flash-tp4`, `glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4`, `mimo-v26-flash-mopd-tp2`, `mimo-v26-flash-mopd-tp4`, `qwen38-flash-next-qad-tp4`, `qwen38-flash-next-tp2`, `swift15-qwen38-flash-next-tp2`, `swift15-qwen38-flash-next-tp4` | The parent's profiles, on either transport |
| `glm53-flash-csf-tp8` | Reads the CSF checkpoint with the merged sources |
| `deepseek-v41-flash-tp8`, `glm53-nvfp4-tp8` | Eight-Spark research profiles; `glm53-nvfp4-tp8` also needs the GLM-5.3 plugin layer, which this image lacks, so the installer refuses it on this lock ([image](../../../profiles/glm53-nvfp4-tp8/README.md#image)) |

`qwen38-flash-next-qad-tp8` is not listed: its HC token-row ownership at
eight ranks is refused by installer admission, because the image receipt's
`hc_supported_modes` names two and four ranks only, and by the image's
`vllm/models/qwen4_exp/nvidia/hc_prefill.py`, which accepts tensor-parallel
groups of two and four.

## Build

The commands run on one Spark (`BUILD`) as an account in the `docker` group,
from a copy of the repository at the revision that holds this directory.
They push nothing. Time and space are for a DGX Spark with the parent image
present.

| Step | Time | Peak space on `BUILD` |
|---|---|---|
| 1-3 CSF-sources image | 5-10 min | about 15 MB |
| 4-6 SIRCL image | 7-14 min | about 10 MB |
| 7 build cache removal | under 1 min | none |
| 8 delta archive | 3-8 min | about 30 GiB in Docker's temporary directory while `docker save` runs, then about 10 MB |
| 9 load on each other Spark | under 1 min each | about 20 MB per Spark |

Keep at least 40 GiB free on `BUILD`. The steps start one short container
per check: 49 parent-file reads in step 2 (both receipts and the 47 replaced
files), 145 added-path checks in step 6, and the image's `verify` once per
profile in steps 3 and 6.

```bash
# 0. Copy the repository to BUILD (on a machine with the checkout).
git -c core.autocrlf=false archive --format=tar --prefix=sparkring/ REVISION \
  | ssh BUILD 'mkdir -p ~/sparkring-image && tar -xf - -C ~/sparkring-image'
```

On `BUILD`:

```bash
W=~/sparkring-image; cd $W/sparkring
PARENT=sha256:aba309e4610c711fda219ed7478a1d68d9bf16dfbd83a0653e32afcbd8f0106f
PARENT_LOCK=runtime/releases/dev-20261004-kraken-cuda1342-nccl2323-status034/installer-image.json
RELEASE=dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034
SOURCES=/srv/sparkring/overlays/csf-05ba9d1c6e5a
cache_ids() { docker buildx du --verbose | awk '/^ID:/{print $2}' | sort; }

# 1. Inputs. Expect the parent ID, 40G or more, Python 3.11 or later, and payload
#    05ba9d1c6e5a..., vLLM bc9ea7744bc7..., B12X cc36aa6f1e93..., tree d02d7dcb87b0...
docker image inspect --format '{{.Id}}' $PARENT
df -BG --output=avail / | tail -1
python3 -c 'import sys; print(sys.version.split()[0])'
grep -E '"(payload_sha256|vllm_merge_commit|b12x_merge_commit|tree_sha256)"' $SOURCES/OVERLAY.json
cache_ids > $W/cache-before.txt

# 2. CSF-sources context. Expect "files": 49, "removed": 0, "status_version": "0.3.4" and receipts
#    external-base 5dca988070da4c25caaecc5b296b7e270c1189d7f3a419567717ed1a0943983f,
#    toolchain 7ab8ee6fac079e37f6558ecd407c749f9175d58901aa74b9dc1d19504b532c55.
python3 runtime/images/derive_kraken_csf_sources.py prepare --parent-lock $PARENT_LOCK \
  --sources $SOURCES --output $W/csf-context

# 3. Build and admit it for the parent's nine profiles. Expect "image_id": "sha256:...",
#    the nine profiles, "status_version": "0.3.4", "serving_qualified": false.
python3 runtime/images/derive_kraken_csf_sources.py build --context $W/csf-context \
  --tag sparkring-dev/kraken:csf-20261007 --name dev-20261007-kraken-csf-cuda1342-nccl2323-status034 \
  --output $W/csf-lock.json

# 4. SIRCL wheel and native libraries. Expect the wheel sparkring_sircl-0.2.0-py3-none-any.whl
#    with 142 files, then /opt/sparkring/sircl/lib/roce_proxy-d6360d260ba540a8.so and
#    p2p_proxy-bd01ba45a875e184.so with the image's gcc version line.
python3 runtime/images/sircl_layer.py wheel --output $W/wheel
WHEEL=$W/wheel/sparkring_sircl-0.2.0-py3-none-any.whl
python3 runtime/images/sircl_layer.py natives --parent-lock $W/csf-lock.json --wheel $WHEEL \
  --output $W/natives

# 5. SIRCL context. Expect "files": 145.
python3 runtime/images/sircl_layer.py prepare --parent-lock $W/csf-lock.json --wheel $WHEEL \
  --natives $W/natives --output $W/sircl-context

# 6. Build, probe and admit for twelve profiles; writes this release's v3 lock. Expect
#    "sircl": "0.2.0", "vllm_pins": ["sparkring-kraken-beta-20261007-bc9ea774"].
PROFILES=deepseek-v41-flash-tp4,deepseek-v41-flash-tp8,glm53-flash-csf-tp8,glm53-flash-nvfp4-spark-tp2,\
glm53-flash-nvfp4-spark-tp4,glm53-nvfp4-tp8,mimo-v26-flash-mopd-tp2,mimo-v26-flash-mopd-tp4,\
qwen38-flash-next-qad-tp4,qwen38-flash-next-tp2,swift15-qwen38-flash-next-tp2,swift15-qwen38-flash-next-tp4
python3 runtime/images/sircl_layer.py build --context $W/sircl-context \
  --tag sparkring-dev/kraken:csf-sircl-20261007 --name $RELEASE --profiles $PROFILES \
  --output runtime/releases/$RELEASE/installer-image.json

# 7. Remove the build cache records of steps 3 and 6 (children before parents) and the
#    build inputs. Expect 0. The natives directory is root-owned: the compiler ran as root.
cache_ids | comm -13 $W/cache-before.txt - > $W/cache-added.txt
for round in 1 2 3; do
  while read -r id; do docker builder prune --force --filter "id=$id" >/dev/null; done < $W/cache-added.txt
done
cache_ids | comm -12 $W/cache-added.txt - | wc -l
sudo rm -rf $W/natives && rm -rf $W/csf-context $W/sircl-context $W/wheel

# 8. Delta archive: the image without the parent's layers. Expect "added_layers": 2 and an
#    archive of about 10 MB.
python3 runtime/images/layer_delta.py --image sparkring-dev/kraken:csf-sircl-20261007 \
  --parent $PARENT --output $W/delta.tar
```

Step 7 compares the build cache before and after the build, so a build that
another user runs on `BUILD` at the same time would lose the records it adds
too. The two image tags and the parent tags that the builders add
(`sparkring-dev/parent:<12 hex digits>`) hold no space of their own.

## Load on the other Sparks

[layer_delta.py](../../images/layer_delta.py) leaves out every layer the
parent provides, so each Spark that holds the parent image loads only the two
added layers, under its own tag, with the lock's image ID. From a machine that
reaches every Spark:

```bash
for HOST in SPARK_1 SPARK_2 SPARK_3; do
  ssh BUILD 'cat ~/sparkring-image/delta.tar' | ssh $HOST "sudo -n docker image inspect --format '{{.Id}}' \
    sha256:aba309e4610c711fda219ed7478a1d68d9bf16dfbd83a0653e32afcbd8f0106f >/dev/null && sudo -n docker load \
    && sudo -n docker image inspect --format '{{.Id}}' sparkring-dev/kraken:csf-sircl-20261007"
done
```

Each Spark prints `Loaded image: sparkring-dev/kraken:csf-sircl-20261007`
and the `image_id` of the lock. On a Spark without the parent image the
command stops before `docker load`; stream the whole image to it instead
(`ssh BUILD 'docker save sparkring-dev/kraken:csf-sircl-20261007' | ssh HOST 'sudo -n docker load'`,
about 30 GiB, with as much free in Docker's temporary directory at both
ends). `sudo sparkring install` copies the image with that whole stream to
every Spark of a deployment that lacks it, and requires twice the image size
and 4 GiB free on each, so load every Spark first.

## Record the release

Copy `runtime/releases/<release>/installer-image.json` from `BUILD` into the
repository, add it with its SHA-256 to the `inputs` of
[release.json](release.json), and record the build conditions (host, parent
image, the IDs of both built images, the commit) beside the lock. The lock's
`image_reference` is the local image ID, so `sparkring images` does not list
the image and `--image NAME` does not select it; installations name the lock:

```bash
sudo sparkring install --profile PROFILE --image-lock /path/to/installer-image.json
```

On this image SIRCL ring sessions are the default transport wherever the
fabric document lists `sircl`, with NCCL off. `sparkring install` without
`--image` selects the image once it is published: the registry digest
replaces `image_reference`, `publication.json` records the derivation from
`dev-20261004-kraken-cuda1342-nccl2323-status034`, and a release tag in
[installer-releases.json](../installer-releases.json) names it
([release procedure](../../../docs/development/releases.md)). The rollback
image is 2026.10.1 (`--image 2026.10.1`), on the prepared transport.

## Checks on hardware

On a fabric that `sudo sparkring setup` recorded with `sircl` among its
transports, with Node A running a SparkRing package that carries the SIRCL
transport adapter and the image loaded on every Spark. Installing stops the
model on the named Sparks.

```bash
LOCK=/path/to/installer-image.json
sudo sparkring install --profile glm53-flash-nvfp4-spark-tp2 --on 0,1 --image-lock $LOCK --plan
sudo sparkring install --profile glm53-flash-nvfp4-spark-tp2 --on 0,1 --image-lock $LOCK
sudo sparkring check --on 0,1 --report ~/sparkring-report
sudo sparkring status
```

| Command | Pass condition |
|---|---|
| `install --plan` | `Transport: sircl on every collective, NCCL off (default table, pair: the design's settings, not measured)` and the release name as the image |
| `install` | `Model ready:` and the summary card's `Transport:   sircl, NCCL: absent` |
| `check --report` | Exit status 0; the functional checks pass; the receipt verdict is as expected; `~/sparkring-report/sparkring-report-<time>/` exists |
| `status` | `Image: dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034` and `Transport: sircl, NCCL: absent (checked ...)` |

The same commands with `--on 2,3` check the other pair. A line of four Sparks
uses `--profile glm53-flash-nvfp4-spark-tp4 --on 0-3` or `--on 4-7`; its plan
names the `path-4` group and `not measured, SIRCL's own rules apply`.
