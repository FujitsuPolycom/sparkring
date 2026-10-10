# dev-20261010-sparkcache-gab86b70-plugins-status036

Status: **research-only**. This release directory holds the build recipe of a
derived installer image that no Spark has built yet; `installer-image.json`
appears here when `runtime/images/derived_layer.py record` or `build` writes it
(see [the layer's README](../../runtime/images/compositions/installer-sparkcache-gab86b70/README.md)).

The image is installer image `sha256:d52737a109e0…` (release
`dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`) plus one
Python-only derived layer,
[id `sparkcache-gab86b70`](../../runtime/images/compositions/installer-sparkcache-gab86b70/descriptor.json),
which adds the SparkCache connector-job lease contract
`vllm-connector-jobs-gab86b70.json` and restates the three SparkCache verifier
files that load it. The layer adds no native file, no transport file and no
vLLM plugin; the derived v3 lock keeps the parent's image line, transports,
SIRCL block (with the pinned vLLM build `sparkring-kraken-beta-20261007-bc9ea774`,
which reads the CSF checkpoint), default tuning digest and added plugins
unchanged, and lists the two SparkCache research-only variants
`glm53-flash-nvfp4-sparkcache-tp2` and `glm53-flash-nvfp4-sparkcache-tp4`
beside the parent's profiles. No other lock lists those profiles, so
`sparkring install` refuses them on every image of this package except this
derived image.

The derived lock changes every installer deployment's identity for the profiles
it lists, like any lock edit; the parent release's published lock and its
inputs stay untouched, and this release introduces its own.
