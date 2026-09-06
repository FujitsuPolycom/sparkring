# Distribution and source provenance

This runtime is a derived application container, not an Apache-only base
image. SparkRing/SparkCache source terms do not relicense NVIDIA components
or third-party dependencies. No NVIDIA endorsement is claimed.

## Source identity

The public SparkCache pin is
[`360f97dbd00b62b06fc5e7839b87514c12f2908f`](https://github.com/FujitsuPolycom/sparkcache/commit/360f97dbd00b62b06fc5e7839b87514c12f2908f).
Its deployable source hash is
`433a75f4f558aa7192eceba61dfae44e9b9823d5be714f054c3568efed88a0ce`, identical
to the deployed development source. The publication branch includes runtime
code, tests, and companion commit-bound vLLM patches without private site
records or development history. Full GPU-free suite:1049 passed,7 skipped.

The old development commit in inherited image labels is historical provenance,
not the public checkout pin. Runtime changes are bound to vLLM815f839 and
the patch preimages. The complete inherited image build is not reconstructed
solely by that vLLM commit: it contains native binaries and earlier overlays.

## NVIDIA terms

The image preserves `/NGC-DL-CONTAINER-LICENSE`, version September14,2021.
The [official license](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf)
provides a conditional derived-container distribution grant, not unrestricted
redistribution of a standalone NVIDIA container. Distribution must preserve
the applicable terms and notices. Separate component terms control their
components; codec/patent permissions are not automatically supplied by this
container license. This document is an engineering compliance checklist, not
a legal opinion or a representation that every distribution condition has
been satisfied.

Required NVIDIA attribution: This software contains source code provided by
NVIDIA Corporation.

Retain the original NVIDIA license and notices, SparkCache's Apache2.0 license
at `/usr/share/licenses/SparkCache/LICENSE`, third-party package notices, and
the bundled communication-component licenses. Do not replace the image license
expression with only Apache-2.0 when changing the repository name.

## Audit scope

The experimental packaging child contains the exact three hashed files from
the tested source deployment. Hash verification does not qualify runtime
behavior, prove inherited source provenance, or establish license compliance.
Inherited metadata includes obsolete model/topology claims; the runtime
manifest is authoritative for this child's limited support scope.

The final-filesystem SPDX inventory is generated with checksum-verified Syft1.51.1
(Linux arm64 archive SHA256
`a7fd2b784e6664acd44719270574f6cd8c6864fc2b1700bf9099bd1cccda7d7f`).
SBOM packages must be reviewed for unexpected model weights, private content,
noncommercial terms, unknown origins, and missing notices before release.
An SBOM alone is not a security or redistribution approval.

The all-layers SBOM attempt exceeded the serving host's intended audit memory
budget and was stopped. A separate streaming inspection covered107 inherited
layers and160613 regular-file entries, checking credential-like filenames
and selected token/private-key patterns in small root/opt/tmp files. It found
no matches. It does not scan every byte or prove the absence of secrets.
The completed filesystem inventory must not be described as an all-layer SBOM.

Directly checked core licenses: vLLM, B12X, FlashInfer, Transformers,
Humming kernels and InstantTensor include Apache2.0 terms; PyTorch metadata
declares a composite Apache/BSD/BSL/MIT expression. Their bundled notices are
retained. NVIDIA container and individual component terms remain applicable.
No model checkpoint is added by the packaging step; the runtime profile
downloads checkpoint weights separately under their own terms.
