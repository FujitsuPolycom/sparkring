# GLM-5.3-Flash CSF with SparkCache CRC32 on 2 Sparks

This profile serves [GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD),
created by Local Inference Lab, Inc. and licensed under the Local Inference Lab
License, Version 1.0, at revision `dec48abd33efa73c3bb7c95b74eee10cad34f9be`.

Status: **research-only** (Experimental). The default image, release
2026.10.5's, admits this profile
([release record](../../runtime/releases/dev-20261016-sparkcache-4a17f1f5198e-status038/README.md)).
One installation on two Sparks of that image composition's recheck build
returned rc=0 and passed `sparkring check`
([record](../../performance/records/images/dev-20261016-sparkcache-4a17f1f5198e-status038-sparkcache-recovery-ring8-20261016.md));
no serving-restore or performance result is recorded. No Compose export
exists.

Serving settings match [glm53-flash-nvfp4-sparkcache-tp2](../glm53-flash-nvfp4-sparkcache-tp2/README.md)
except restore and capacity:
MTP3, asynchronous scheduling, CUDA page restore, connector-jobs capture,
two 512 MiB slots, a 256 MiB restore budget, 64 MiB placement arenas, eight CUDA restore
I/O workers, 256K-token spans and a 32 GiB persistent cache with a 28 GiB low
watermark. The explicit
`spark_cache_payload_integrity=fast` setting selects CRC32 payload checks;
identity, keys and metadata retain SHA-256. Use a cache directory writable
only by trusted local processes. CRC32 detects accidental corruption and is
not authentication against an attacker who can rewrite cache contents.

The cache root is `/cache/glm53-flash-sparkcache-crc32-tp2-8abbe9da67b8-dec48abd33ef`. It separates this composition from the earlier
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
