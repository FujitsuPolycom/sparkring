# SparkCache compositions

SparkCache is an external project. SparkRing owns deployment composition and
image/connector compatibility, not a fork of the cache engine. The profile
catalog selects the authoritative composition recipes under `profiles/`;
[legacy recipe paths](../../recipes/sparkcache/README.md) remain generated exports.

The shared resolver in `runtime/common/profiles.py` exposes the selected recipe's
runtime and evidence. GLM TP2/TP4 release contracts separately pin native library
and connector inputs. Enabling cache in a different image does not transfer
qualification. Qwen with SparkCache remains unsupported.

Changes to cache behavior belong in the external SparkCache repository. Changes
to model-specific vLLM hooks belong in `integrations/vllm`; changes to deployment
selection belong in the profile. Keep source, lease-contract and image references
exact and state which combined configuration was tested.
