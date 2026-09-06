# MTP3 cache-reuse source experiment

Status: **research-only**. These source transforms coordinate recurrent prefix
reuse, GPU-lease accounting, and the fused indexer's histogram barrier for
GLM-5.3 native-MTP3. `compose.py` produces and verifies their exact Python output
from one specified base image. The package includes source inputs and CPU tests
of the allocation, lookup, retention, and accounting rules.

The composition tool is implemented. Serving and performance qualification
require separate evidence for the complete image, SparkCache package, native
libraries, and workload. Published runtime pins and production Dockerfiles are
outside this experiment's output.

The scope is GLM-5.3 native-MTP3 on four GB10 ranks, TP4/DCP4, with 512-token
recurrent/hash pages, 2,048-token attention scheduling alignment, one prefill
lookahead token, and aligned recurrent caching. Other speculation methods,
geometries and images require separate evidence.

## Exact inputs and outputs

The supported source input is local image ID
`sha256:2e41b1e934a85ff7c21b780532db2f0a0e978df081e52f4ae2bf11f8992fb24f`.
This is a local immutable image identity, not a registry pull reference.
`fixtures/manifest.json` lists the ten input Python files, their paths in
that image, byte counts and SHA-256 values. The 155 KiB fixture archive contains
only those files. Source bytes and their existing SPDX/copyright headers are
preserved. Models, compiled libraries, credentials and site configuration are
excluded. Fixtures are test inputs and do not enter a production image build.

| Component | Output or required-library SHA-256 |
|---|---|
| vLLM scheduler | `75efa57e7ff5a77c76714b85e2e4d8e1d7f456d9a9eec6c67ebb11ca382942f9` |
| vLLM single-type KV manager | `d2e35b012e0cf45ab3771f545c35ca48f2a5858549c574a352975607369124e2` |
| B12X fused indexer | `b43a4a2802c7dfc4a049bbb5751fc7e7688b05cb06d4ece716ab7a1d91d23d2a` |
| Separately supplied SparkCache placement library | `2657cdd2e54a097c9544e4c79ae62c0646db6db123ff24e4f0c384238c3a1e8d` |
| Required SparkCache capture library | `4398f18b8913e743e7bf1ed8fe29560d4580e61b6a1e2ab8b16684b19b6573b5` |

The fixture manifest attests each patch script's exact bytes. Each transform
accepts only its declared input SHA-256 or its own output SHA-256. Apply them in
this dependency order:

1. `patch_mtp3_barrier.py` adds the CTA synchronization before the histogram
   arrival signal and increments that kernel's compile-cache revision.
2. `patch_mtp3_lease_accounting.py` includes an attached GPU lease in initial
   prefill cache statistics without labeling it an external transfer.
3. `patch_mtp3_local_lease_preference.py` prefers a strictly longer converged
   local prefix over a shorter GPU lease and reuses that lookup result.
4. `patch_mtp3_sparse_retention.py` pairs an explicit speculative replay
   checkpoint stop with sparse retention of its predecessor state.
5. `patch_mtp3_partial_tail_eligibility.py` decides whether a partial recurrent
   tail requires a stop from the recurrent page size, rather than attention's
   DCP-scaled scheduling unit.

Every transform rejects an unknown input and checks its complete output. Its
own output is idempotent; source produced by another transform must still match
one of those two declared hashes. Use `compose.py --verify-candidate` to check
the composed output tree.

The `original/` directory contains the exact source inputs. The `candidate/`
directory contains transformed output, and `composition.json` records output
hashes and transform receipts. These directory names describe the tool's file
interface, not deployment or qualification status.

## Offline composition and tests

Python 3.10 or later is sufficient for composition. Pytest is needed for tests.
No Docker, vLLM installation, Torch, model files or GPU is required:

```bash
python runtime/glm53-spark-mtp3-mesh/experiments/cache-reuse/compose.py --check
python -m pytest runtime/glm53-spark-mtp3-mesh/experiments/cache-reuse -q -rs
```

To retain the exact source trees and receipt in a fresh local directory:

```bash
python runtime/glm53-spark-mtp3-mesh/experiments/cache-reuse/compose.py \
  --output-root work/mtp3-cache-reuse-composition
python runtime/glm53-spark-mtp3-mesh/experiments/cache-reuse/compose.py \
  --verify-candidate work/mtp3-cache-reuse-composition/candidate
python runtime/glm53-spark-mtp3-mesh/experiments/cache-reuse/check_mtp3_checkpoint_allocations.py \
  --source-root work/mtp3-cache-reuse-composition/original/vllm \
  --candidate-root work/mtp3-cache-reuse-composition/candidate/vllm \
  --output work/mtp3-checkpoint-allocations.json
```

Existing output directories and result files are rejected. `--source-root`
on the composition command accepts an independently extracted original tree,
with `vllm/` and `b12x/` immediately beneath it; every file is checked against
the same manifest before output creation.

The allocator checker executes the actual allocation and block-registration
methods against planned running-state and GDN checkpoint writes. It covers
336 fresh and 336 resumed empty-table replay schedules with varied shared
boundaries, chunk budgets and speculative buffer counts. A selected null slot
is skipped by the real registration method. A selected non-null slot whose
planned state does not match its hash boundary fails the check. This tests
metadata consistency, not completion of CUDA writes. Some resumed schedules
can still miss a prompt predecessor; the report distinguishes those safe misses
from stale-state registration.

The tests also cover lease/API accounting, local-versus-lease selection,
speculative backoff, prompt-length boundaries and recurrent partial-tail
eligibility. One optional test uses the companion SparkCache checkout directly:

```bash
SPARKCACHE_SOURCE_ROOT=/path/to/sparkcache python -m pytest \
  runtime/glm53-spark-mtp3-mesh/experiments/cache-reuse -q -rs
```

That optional test needs the companion checkout's CPU development dependencies.
It is explicitly skipped when `SPARKCACHE_SOURCE_ROOT` is absent.

## Independent fixture extraction

Use only the exact base image above on a local Docker host. A stopped container
can expose its files without launching Python, CUDA or serving. The following
Python fragment creates that stopped container, copies only manifest-listed
files, and removes the temporary container. It changes local Docker metadata;
do not run it against a serving container or a remote Docker context.

```python
import json
from pathlib import Path
import subprocess

package = Path('runtime/glm53-spark-mtp3-mesh/experiments/cache-reuse')
manifest = json.loads((package / 'fixtures/manifest.json').read_text())
output = Path('work/mtp3-extracted-original')
output.mkdir(parents=True, exist_ok=False)
image = manifest['base_image_id']
observed = subprocess.check_output(['docker', 'image', 'inspect', image,
                                    '--format', '{{.Id}}'], text=True).strip()
assert observed == image
container = subprocess.check_output(['docker', 'create', '--entrypoint',
                                     '/bin/true', image], text=True).strip()
try:
    for name, record in manifest['files'].items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['docker', 'cp', container + ':' + record['image_path'],
                        str(target)], check=True)
finally:
    subprocess.run(['docker', 'rm', container], check=True)
```

Then run `compose.py --source-root work/mtp3-extracted-original --check` to
verify the independently copied inputs and complete transform chain.

## Serving composition boundary

To serve with these Python sources, select a SparkCache commit and supply both
native libraries with the hashes listed above. The experimental image also
needs a runtime contract that attests the output scheduler and manager hashes.
The contract is
`sparkcache/runtime_patches/vllm-manager-page-async-contract-55969c16.json`;
the exact scheduler and manager preimage hashes are in the fixture manifest.
Record the resulting contract, full SparkCache source tree, native libraries,
image ID, model identity, topology, namespace and workload settings together.
The source composer does not rewrite that contract or published source pins.

CPU test success establishes the stated source and metadata checks. Throughput,
deployment readiness, and unmeasured preemption/CUDA schedules require hardware
evidence that identifies the complete image and its test conditions.
