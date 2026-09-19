# Source checks for the Kraken integration

The [B12X suite](../b12x-kraken-suite.json) compares the SparkRing 2026.09.3
source baseline with the pinned Kraken candidate. It does not replace or edit
the [retained suite](../b12x-suite.json), which targets a different baseline API.
Suite files are hash-bound; source provenance and paired component identities
must also be verified by the consuming build gate.

The [paired gate adapter](../../kraken_gate.py) verifies the policy-pinned vLLM
coordinator before invoking the ordinary protected-suite runner. Declare the
coordinator file and adapter as hashed policy inputs. The accepted vLLM source
must carry that same coordinator identity in its source-binding inventory;
testing B12X against an unrelated peer file is not a compatible image check.

The [vLLM suite](../vllm-kraken-suite.json) retains its publication cases and
binds their consumer to the merged request-scoped SparkCache implementation.
The adapter verifies the complete connector-file digest, the unchanged boundary
consumer's AST digest, and the retained case file's digest. It executes the
selected connector method; it does not substitute expected results or disable
identity validation. Supply `SPARKRING_SPARKCACHE_SOURCE_ROOT` explicitly for
source-only checks, or use the exact installed connector in the test image.

| Retained check | Kraken contract |
|---|---|
| Common KDA checkpoint behavior | Unchanged, on both baseline and candidate |
| R37-only checkpoint helper import | Not selected: that helper is absent from the 2026.09.3 baseline; common checkpoint behavior remains required |
| Legacy QSA kernel-count and replacement tests | Exact 2026.09.3 inventory, selector identity and effective mutation tests in the [QSA oracle](QSA_RELEASE3_V1.md) |
| Pre-agreement distributed tuning fixture | Production cache-agreement handshake and cancellation in the [version-2 oracle](distributed-cache-v2.md) |
| Other prepared-kernel, PLE, GDN, compiler and lifetime checks | Unchanged |

The QSA exception is limited to the reviewed raw-ring bounds guard and its
runtime row-count argument. It does not admit arbitrary kernel changes.
The paired cache-agreement test requires an explicit vLLM source root in
addition to B12X. CPU success cannot qualify transport delivery, CUDA replay,
memory safety, installed artifacts or serving performance. Run the documented
GPU and sanitizer gates before recommending an image.
