# Communication source selection in a shared image

Status: **implemented**. The image can carry the TP2 adaptive-grid RoCEnante
package alongside the four-rank weighted mesh package. A serving profile
selects one communication implementation before B12X imports it. CPU checks
validate packaging and selection; they do not qualify CUDA or RDMA behavior
in the assembled image.

The `tp2-rocenante-adaptive` bundle contains the exact `b12x.comm.roce` source
at SparkRing commit `60d8d68486540ce9ddb2702dd545fda6b347c087`, including its
Apache-2.0 license. Its manifest binds all nine package files and the license
by SHA-256 and records their Git blobs. Preserve the supplied file bytes:
the repository attributes prevent newline conversion of this bundle.

The two-path TP2 default uses peer map `1=0/2` on rank 0 and `0=0/2` on rank 1.
Indices refer to the full four-function HCA inventory, selecting the two PCI
domains of physical cage p0. The package also contains research-only
four-path support; this profile does not activate it.

The maintained TP4 weighted mesh source remains at
[`runtime/glm53-spark-mtp3-mesh/performance/transport`](../glm53-spark-mtp3-mesh/performance/transport/).
Its import setup and its proxy/kernel ABI belong to that profile. Do not copy
individual TP2 kernel or proxy files over that bundle, and do not enable both
profile hooks in a serving process.

## Package into the shared image

Prepare an exclusive local staging directory without building an image:

```bash
python3 runtime/transport_profiles/package.py --destination /tmp/tp2-transport-context
```

Copy `tp2-rocenante-adaptive/`, `sparkring_transport_selector.py`, and
`entrypoint.py` into `/opt/sparkring/transports/`. Install the generated
`sparkring_transport.pth` in the serving interpreter's `site-packages`
directory. The hook adds `/opt/sparkring/transports` to Python's search path
and runs the selector. Install it in the interpreter used by both vLLM and
its spawned Python workers.

The serving profile sets:

```text
SPARKRING_TRANSPORT_PROFILE=tp2-rocenante-adaptive
SPARKRING_TRANSPORT_MANIFEST_SHA256=eb03cfde826974811be3bfe5d88f36d9de105b73358f3eaa56b9ed44f19127c4
```

The selector validates the manifest and every bundled file, then installs an
import hook for `b12x.comm.roce` and its children. Other B12X modules, including
the model loader and KDA kernels, resolve from the image's installed package.
The hook compiles verified source directly, avoiding unrelated cached Python
bytecode. With no selected profile it leaves import behavior unchanged.

Python normally prints and ignores exceptions from `.pth` files. An invalid
explicit selection raises `SystemExit`, which stops startup. CPU tests run an
actual Python subprocess with an invalid digest and confirm that its consumer
is never reached. Tests also start a fresh Python process with a valid digest
and confirm the selected package origin.

The TP2 launcher uses this container entrypoint:

```text
python3 /opt/sparkring/transports/entrypoint.py serve /models/target ...
```

Before importing vLLM, that entrypoint rechecks source hashes, verifies the
active import finder, and requires the matching installed `.pth` hook for
spawned workers. It refuses to activate after an unrelated transport has
already been imported. These checks establish source selection; they do not
establish compatibility of the image's remaining vLLM/B12X modules.

The common image must separately satisfy the
[TP2 dependency matrix](../profiles/glm53-flash-nvfp4-tp2/dependencies.json).
In particular, keep the public four-checkpoint producer/consumer contract
while adding TP2 admission. The measured two-checkpoint source is provenance
for the reference deployment, not a replacement for the shared implementation.
