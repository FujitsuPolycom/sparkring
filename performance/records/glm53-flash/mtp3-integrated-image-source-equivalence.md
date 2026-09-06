# Source equivalence for the native-MTP3 cache/checkpoint image

Status: **research-only**. Image composition and file verification are implemented.

## Conditions

The published image has registry digest
`sha256:11a556a54041fd823d152a7f051ac4f7c617dc539030df26e93008392fee0746`
and config ID
`sha256:6921a6c163ea40b603e19a0332330efe3dbccbf4dce9f6cbbf6b756c9231835a`.
It derives from the published native-MTP3 compute image identified by
`runtime/glm53-spark-mtp3-mesh/performance/prepare.py`, with SparkCache
`48bbd2be4a7b972e56632a2d7b934bac5460f272` and the packaged runtime transforms.

The comparison image, observed running on a four-GB10 TP4/DCP4 cluster, was
`sha256:84c4546d37e8504fc98a359435b262a5d8e7bfe9c892af7429e876db5bc98422`.
It includes explicit recurrent checkpoints and the GLM reasoning contract.
The cluster remained running during the build and read-only comparison.

## Measurement

SHA-256 hashes from the rebuilt image's installed-file receipt were compared
with files in the running container. The population contains every recorded
vLLM and B12X file, the transport bundle, and the warmup helper: 5,308 files.
SparkCache is excluded from this equality claim because the rebuild explicitly
selects its merged main revision. The runtime's checkpoint and source preimage
checks ran before installation; no ownership digest was accepted merely by
recomputing it from an unknown runtime.

## Result

All 5,308 compared files matched. The image verifier checked 5,472 files in the
rebuilt container without loading a model or initializing CUDA. An independent
registry pull and file-verification run on another Spark also passed. Anonymous
registry manifest access verified the config ID recorded in the public receipt.

## Conclusion

The published build preserves the compared deployed runtime components and
incorporates the selected SparkCache source. Repository-relative build inputs,
source inventories, strict patches, and the published-image contract reproduce
the composition without private workspace paths.

## Limitations

No serving soak was run on this exact rebuilt image. Source equivalence does not
transfer every measurement from the comparison image. Earlier cache-pressure
tests used a 2 GiB/rank namespace; 40 GiB multimodal stress behavior remains
unqualified. Native binary rebuilds can differ with toolchains and require
separate verification. This record does not establish unattended availability.
