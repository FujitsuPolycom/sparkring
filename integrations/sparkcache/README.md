# SparkCache compositions

SparkCache is an external project. SparkRing owns deployment composition and
image/connector compatibility, not a fork of the cache engine. The profile
catalog selects the authoritative composition recipes under `profiles/`;
[legacy recipe paths](../../recipes/sparkcache/README.md) remain generated exports.

The shared resolver in `runtime/common/profiles.py` exposes the selected recipe's
runtime and evidence. GLM TP2/TP4 release contracts separately pin native library
and connector inputs. Enabling cache in a different image does not transfer
qualification. The [Qwen TP2 profile](../../profiles/qwen38-flash-next-tp2/README.md)
has an Experimental, published aligned-cache composition with bounded text/media
restore checks. Complete request-boundary persistence is a separate composition;
do not attribute its measurements to the aligned-cache image.

Changes to cache behavior belong in the external SparkCache repository. Changes
to model-specific vLLM hooks belong in `integrations/vllm`; changes to deployment
selection belong in the profile. Keep source, lease-contract and image references
exact and state which combined configuration was tested.
