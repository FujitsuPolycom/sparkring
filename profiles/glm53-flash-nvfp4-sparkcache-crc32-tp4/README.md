# GLM-5.3-Flash CSF with SparkCache CRC32 on 4 Sparks

This profile serves [GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD),
created by Local Inference Lab, Inc. and licensed under the Local Inference Lab
License, Version 1.0, at revision `dec48abd33efa73c3bb7c95b74eee10cad34f9be`.

Status: **research-only** (Experimental). No image build or hardware run is
recorded for this variant. It requires the [SparkCache source layer](../../runtime/images/compositions/installer-sparkcache-8abbe9da67b8/README.md)
over release 2026.10.4. No published image lock admits this profile and no
Compose export exists. The layer's build step produces a development lock
that lists `glm53-flash-nvfp4-sparkcache-crc32-tp4`.

Serving settings match [glm53-flash-nvfp4-sparkcache-tp4](../glm53-flash-nvfp4-sparkcache-tp4/README.md)
except restore and capacity:
MTP3, asynchronous scheduling, CUDA page restore, connector-jobs capture,
two 8 GiB slots, a 256 MiB restore budget, 64 MiB placement arenas, eight CUDA restore
I/O workers, 256K-token spans and a 32 GiB persistent cache with a 28 GiB low
watermark. The explicit
`spark_cache_payload_integrity=fast` setting selects CRC32 payload checks;
identity, keys and metadata retain SHA-256. Use a cache directory writable
only by trusted local processes. CRC32 detects accidental corruption and is
not authentication against an attacker who can rewrite cache contents.

The cache root is `/cache/glm53-flash-sparkcache-crc32-tp4-8abbe9da67b8-dec48abd33ef`. It separates this composition from the earlier
variant and from the other TP size. SparkCache's fast publication format
also uses a distinct cache identity, so SHA-256 payload entries miss cleanly.

The recorded target and draft checkpoint identity remains
`6af0343976eefbfa0fb59d87befcf6d10f02fef90ba0e7881dd1201fe0ecd76e`.
Its original preimage is not recorded. SparkCache's GLM profile helper only
validates it; it does not derive it. See the layer's [identity findings](../../runtime/images/compositions/installer-sparkcache-8abbe9da67b8/README.md#checkpoint-identities).

The separate profile ID preserves the earlier variant's configuration and
hardware evidence. Before qualification, run the [hardware test plan](../../performance/sparkcache-flash-csf-hardware-test-plan.md)
against this image, including its native hashes and eleven-file lease-contract
check, restart, SIGKILL, corruption, output equality and two-start cache-off A/B.
No result from the earlier image is serving evidence for this composition.
