"""Model shape profiles for the persistent context cache.

A profile declares how one model family's registered KV-cache layers map
onto the cache's persistent record vocabulary, plus the identity strings
that pin cached bytes to a layout. Profiles carry no runtime state and no
vllm/torch imports; the connector resolves one profile at construction and
threads its values through the codec and the SparkCache CUDA placement layer.

The profile name itself is never serialized. Cache identity remains the
tuple of layout strings, checkpoint digests, parallel degrees, geometry,
and policies stored in ``CacheIdentity`` — two profiles with equal field
values interoperate, and any field difference forks the storage namespace
into a clean miss (miss-not-alias is the invariant).

The record vocabulary is closed: ``target_ckv``, ``sparse_indexer``,
``mtp_draft_kv``, plus the non-data ``logical_positions`` and the optional
``boundary_hidden``. Profiles map model layers onto these
kinds; unsupported kinds are rejected because the on-disk chunk ABI and the
SparkCache CUDA placement ABI (at most ``MAX_RECORD_KINDS`` data records per chunk)
are frozen.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping

# Data record kinds a profile may persist. LOGICAL_POSITIONS is implicit in
# every chunk and is not part of this vocabulary.
DATA_FAMILIES = frozenset(
    {"target_ckv", "sparse_indexer", "mtp_draft_kv", "boundary_hidden"}
)

# SparkCache CUDA placement ABI ordinals (legacy spark_cache_native RECORD_*).
# Frozen; a
# profile maps families onto them and must never renumber.
NATIVE_RECORD_ORDINALS = {
    "target_ckv": 0,
    "sparse_indexer": 1,
    "mtp_draft_kv": 2,
    "boundary_hidden": 3,
}

# Ceiling from the C++/CUDA chunk descriptor (record_offset/length arrays).
MAX_NATIVE_RECORD_KINDS = 4

_DRAFT_POLICIES = frozenset({"separate", "colocated_target"})
_BOUNDARY_POLICIES = frozenset({"persisted", "live_forward"})


class ProfileError(ValueError):
    """Invalid profile definition or a profile/deployment mismatch."""


@dataclass(frozen=True)
class ModelProfile:
    """Declarative mapping from one model family to the cache ABI.

    ``classification_rules`` is an ordered tuple of ``(substring, family)``
    pairs matched case-insensitively against registered layer names,
    first match wins; unmatched layers fall to ``default_family``.
    ``required_families`` must all receive at least one layer at
    registration (minus ``optional_families`` and any family an active
    policy drops); a chunk that silently omits declared state is the
    failure mode this cache exists to prevent.
    """

    name: str
    description: str
    quantization_layout: str
    rope_layout: str
    boundary_hidden_policy: str
    default_draft_kv_policy: str
    classification_rules: tuple[tuple[str, str], ...]
    required_families: frozenset[str]
    optional_families: frozenset[str] = frozenset()
    default_family: str = "target_ckv"
    chunk_tokens: int = 256
    storage_mode: str = "per_token_rows"
    # True for MLA-style caches where every TP rank holds identical KV bytes.
    # The value is serialized profile metadata. Quorum, rank-local namespaces,
    # and capacity accounting deliberately remain unchanged; no implemented
    # shared-storage interface consumes this field.
    kv_replicated_across_tp: bool = False
    cuda_page_restore: bool = False

    def __post_init__(self) -> None:
        if self.boundary_hidden_policy not in _BOUNDARY_POLICIES:
            raise ProfileError(
                f"profile {self.name}: unknown boundary_hidden_policy"
                f" {self.boundary_hidden_policy!r}"
            )
        if self.default_draft_kv_policy not in _DRAFT_POLICIES:
            raise ProfileError(
                f"profile {self.name}: unknown draft policy"
                f" {self.default_draft_kv_policy!r}"
            )
        if self.chunk_tokens <= 0:
            raise ProfileError(f"profile {self.name}: chunk_tokens must be positive")
        if self.storage_mode not in {"per_token_rows", "block_pages_v1"}:
            raise ProfileError(
                f"profile {self.name}: unknown storage mode {self.storage_mode!r}"
            )
        if self.cuda_page_restore and self.storage_mode != "block_pages_v1":
            raise ProfileError(
                f"profile {self.name}: cuda_page_restore requires block pages"
            )
        families = {family for _, family in self.classification_rules}
        families |= self.required_families | self.optional_families
        families.add(self.default_family)
        unknown = families - DATA_FAMILIES
        if unknown:
            raise ProfileError(
                f"profile {self.name}: unknown record families:"
                f" {', '.join(sorted(unknown))}"
            )
        for rule in self.classification_rules:
            if not rule[0]:
                raise ProfileError(f"profile {self.name}: empty classification pattern")
        if not self.required_families:
            raise ProfileError(f"profile {self.name}: at least one required family")

    @property
    def native_page_restore(self) -> bool:
        """Compatibility alias for :attr:`cuda_page_restore`."""

        return self.cuda_page_restore

    def persisted_families(self, draft_kv_policy: str) -> frozenset[str]:
        """Data families a store must cover under the active draft policy.

        ``colocated_target`` drops ``mtp_draft_kv`` as a distinct record:
        the drafter's state lives unmarked in the target pool.
        ``live_forward`` drops ``boundary_hidden``: boundary activations
        are recomputed by the live forward instead of persisted.
        """
        families = set(self.required_families)
        if draft_kv_policy == "colocated_target":
            families.discard("mtp_draft_kv")
        if self.boundary_hidden_policy == "live_forward":
            families.discard("boundary_hidden")
        return frozenset(families)

    def allow_missing(self, draft_kv_policy: str) -> frozenset[str]:
        """Families ``build_layer_plans`` may find no layer for."""
        dropped = self.required_families - self.persisted_families(draft_kv_policy)
        return frozenset(self.optional_families | dropped)

    def expects_draft_named_layers(self, draft_kv_policy: str) -> bool:
        """Whether registration should see draft-classified layer names.

        Under ``separate`` the drafter registers marked layers that
        classify as ``mtp_draft_kv``; under ``colocated_target`` drafter
        state registers unmarked inside the target pool.
        """
        return (
            draft_kv_policy == "separate" and "mtp_draft_kv" in self.required_families
        )

    def validate_for_deployment(
        self,
        *,
        dcp_degree: int,
        block_size: int,
        min_span_tokens: int,
        cuda_restore: bool | None = None,
        native_restore: bool | None = None,
    ) -> None:
        """Fail startup on geometry a store or restore would corrupt or
        silently truncate later.
        """
        if dcp_degree <= 0:
            raise ProfileError("dcp_degree must be positive")
        if self.chunk_tokens % dcp_degree:
            raise ProfileError(
                f"profile {self.name}: chunk_tokens {self.chunk_tokens} is not"
                f" divisible by dcp_degree {dcp_degree}; per-rank chunk rows"
                " would silently drop owned positions"
            )
        if block_size <= 0 or (
            self.chunk_tokens % block_size and block_size % self.chunk_tokens
        ):
            raise ProfileError(
                f"profile {self.name}: chunk_tokens {self.chunk_tokens} and"
                f" block_size {block_size} are not mutually divisible; span"
                " alignment would disagree with the block table"
            )
        if min_span_tokens < self.chunk_tokens:
            raise ProfileError(
                f"profile {self.name}: min_span_tokens {min_span_tokens} is"
                f" below one chunk ({self.chunk_tokens} tokens)"
            )
        if cuda_restore is not None and native_restore is not None:
            if bool(cuda_restore) != bool(native_restore):
                raise ProfileError(
                    "conflicting cuda_restore and legacy native_restore values"
                )
        if cuda_restore is None and native_restore is not None:
            warnings.warn(
                "native_restore is deprecated; use cuda_restore",
                FutureWarning,
                stacklevel=2,
            )
            cuda_restore = native_restore
        if cuda_restore:
            if self.storage_mode == "block_pages_v1":
                if not self.cuda_page_restore:
                    raise ProfileError(
                        f"profile {self.name}: SparkCache CUDA restore does not"
                        " support this block-page layout"
                    )
                return
            persisted = self.persisted_families(self.default_draft_kv_policy)
            unmapped = {
                family
                for family in DATA_FAMILIES
                if family not in NATIVE_RECORD_ORDINALS
            } & persisted
            if unmapped:
                raise ProfileError(
                    f"profile {self.name}: families lack C++/CUDA ordinals:"
                    f" {', '.join(sorted(unmapped))}"
                )
            if len(persisted) > MAX_NATIVE_RECORD_KINDS:
                raise ProfileError(
                    f"profile {self.name}: {len(persisted)} persisted families"
                    f" exceed the C++/CUDA ABI ceiling of"
                    f" {MAX_NATIVE_RECORD_KINDS}"
                )


PROFILES: Mapping[str, ModelProfile] = {
    # GLM-5.2 NVFP4 four-Spark reference deployment. Field values preserve the
    # deployed identity strings so compatible on-disk entries remain valid.
    "glm52-nvfp4": ModelProfile(
        name="glm52-nvfp4",
        description=(
            "GLM-5.2 sparse-MLA with NVFP4 per-token CKV records, sparse"
            " indexer state, and MTP draft KV colocated in the target pool."
        ),
        quantization_layout="nvfp4_ds_mla-per-token-v1",
        rope_layout="glm52-rope-v1",
        boundary_hidden_policy="live_forward",
        default_draft_kv_policy="colocated_target",
        classification_rules=(
            ("indexer", "sparse_indexer"),
            ("draft", "mtp_draft_kv"),
            ("mtp", "mtp_draft_kv"),
            ("spec", "mtp_draft_kv"),
        ),
        required_families=frozenset({"target_ckv", "sparse_indexer", "mtp_draft_kv"}),
        chunk_tokens=256,
        kv_replicated_across_tp=True,
    ),
    "glm53-flash-hybrid": ModelProfile(
        name="glm53-flash-hybrid",
        description=(
            "GLM-5.3 Flash hybrid sparse-MLA/C4 pages and KDA/GDN recurrent"
            " checkpoints. External speculative-draft state is recomputed;"
            " its checkpoint identity remains namespace-bound."
        ),
        quantization_layout="glm53-flash-hybrid-block-pages-v1",
        rope_layout="glm53-flash-rope-v1",
        boundary_hidden_policy="live_forward",
        default_draft_kv_policy="separate",
        classification_rules=(),
        required_families=frozenset({"target_ckv"}),
        chunk_tokens=256,
        storage_mode="block_pages_v1",
        cuda_page_restore=True,
        kv_replicated_across_tp=True,
    ),
    "qwen38-flash-next-hybrid": ModelProfile(
        name="qwen38-flash-next-hybrid",
        description=(
            "Qwen3.8 Flash Next attention pages and aligned GDN recurrent"
            " checkpoints. External speculative-draft state is recomputed;"
            " its checkpoint identity remains namespace-bound."
        ),
        quantization_layout="qwen38-flash-next-hybrid-block-pages-v1",
        rope_layout="qwen38-flash-next-mrope-v1",
        boundary_hidden_policy="live_forward",
        default_draft_kv_policy="separate",
        classification_rules=(),
        required_families=frozenset({"target_ckv"}),
        chunk_tokens=32,
        storage_mode="block_pages_v1",
        cuda_page_restore=True,
        kv_replicated_across_tp=False,
    ),
    "deepseek-v4-fp8-hma": ModelProfile(
        name="deepseek-v4-fp8-hma",
        description=(
            "DeepSeek-V4 FP8 hybrid MLA/SWA caches stored as opaque pages"
            " under their five hybrid-memory-allocator (HMA) block tables,"
            " including compressor state."
        ),
        quantization_layout="deepseek-v4-fp8-hma-block-pages-v1",
        rope_layout="deepseek-v4-rope-v1",
        boundary_hidden_policy="live_forward",
        default_draft_kv_policy="colocated_target",
        classification_rules=(),
        required_families=frozenset({"target_ckv"}),
        chunk_tokens=256,
        storage_mode="block_pages_v1",
        kv_replicated_across_tp=True,
    ),
}

# Deployment-facing names may be more specific than the persistent cache
# layout they select. Aliases resolve to the same frozen ModelProfile object,
# so adding a deployment spelling cannot fork or alias an on-disk namespace.
PROFILE_ALIASES: Mapping[str, str] = {
    "glm52-exl3-r7-3.5bpw": "glm52-nvfp4",
}


def resolve_profile(name: str) -> ModelProfile:
    canonical_name = PROFILE_ALIASES.get(name, name)
    try:
        return PROFILES[canonical_name]
    except KeyError:
        known = ", ".join(sorted((*PROFILES, *PROFILE_ALIASES)))
        raise ProfileError(
            f"unknown model profile {name!r}; known profiles: {known}"
        ) from None
