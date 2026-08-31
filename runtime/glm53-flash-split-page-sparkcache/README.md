# GLM-5.3 split-page SparkCache runtime artifact

Status: **qualified** for the exact local image ID and bounded C8 workload in
[`qualified-artifact.json`](qualified-artifact.json). The artifact combines
split target and recurrent page geometry, SparkCache CUDA placement, tail-only
publication, and authenticated shared-base reads for GLM-5.3 Flash TP4/DCP1.

The image is retained locally and has no published OCI digest. The source
repositories, immutable revisions, model identities, and serving settings are
public and machine-readable in [`pins.json`](pins.json), but rebuilding them
produces a different artifact with **implemented** status. A rebuild does not
inherit the local image's qualification.

The qualified C8 cohort used eight distinct 16,384-token stored roots that
shared one stored 8,192-token base. After one tiny inference delivered worker
manifest inventory to the scheduler, all eight returned the expected result
and restored external state. Each rank performed one physical read of the
shared base and avoided seven duplicate reads. See the
[`validation.json`](../../performance/receipts/glm53-flash/split-page-shared-base-c8-20260830/validation.json)
receipt for exact conditions and limitations.

An HTTP health response is not persistent-cache readiness. Workers discover
manifests during startup, but the scheduler receives their inventory only
after a real model execution. The operator quickstart therefore runs one tiny
inference before asserting persistent hits.

Operators who already have the qualified image on all four ranks can use
[`launch-qualified-rank.sh`](launch-qualified-rank.sh). It rejects any image
whose local image ID differs from the qualified ID. Copy
[`qualified.env.example`](qualified.env.example), set the five required
site-specific addresses and paths, and change any documented operational
setting in that one file. The launcher validates the configuration and builds
all JSON arguments with a JSON encoder. A setting that differs from the
recorded runtime receives the container label
`org.sparkring.qualification-status=user-modified-unqualified`; it does not
inherit the bounded C8 result. The
[operator quickstart](../../docs/GLM53_SPLIT_PAGE_SPARKCACHE_TP4_QUICKSTART.md)
provides copy-paste launch, health, log, and rollback commands.

The retained rollback containers are
`glm53-pr535-sc78-hotpatch-c8-qualified-r{0..3}`. The rollback helper stops the
qualified container on one rank and starts its retained predecessor. Review
and run that mutating operation on all four ranks; never mix runtime generations
inside one TP4 service.
