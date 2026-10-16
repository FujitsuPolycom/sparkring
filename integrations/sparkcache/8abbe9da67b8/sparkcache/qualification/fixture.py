"""Byte-exact GPU-free fixture for page-tail qualification.

The fixture models one GLM-5.3 Flash TP4/DCP4 rank-local cache namespace: the
``glm53-flash-hybrid`` block-page profile's record schema, the
``page-tail-cow-v1`` publication schema, DCP degree 4, and the
``:manager-pages-v2:`` manager-page namespace digest formed by
:func:`sparkcache.spark_context_cache_config.kv_group_topology_digest`. One
full-attention page group of 256 tokens carries one 1024-byte-page target
layer and one aligned-recurrent group carries one 64-byte
checkpoint page, matching ``_group_block_counts_for_span``'s requirement that
a ``recurrent_align`` group stores exactly one checkpoint page.

Every snapshot is a deterministic function of its page count. The target page
labels repeat byte-identically across extension thresholds; a delta therefore
publishes only its new pages, and complete-snapshot comparison has a known
expected byte image.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sparkcache.spark_context_cache_config import kv_group_topology_digest
from sparkcache.spark_context_cache_hybrid import (
    PageGroup,
    PageLayer,
    PageLayout,
    encode_page_snapshot,
)


RECURRENTPAGE_BYTES = 64
TARGET_PAGE_BYTES = 1024


class FullAttentionSpec:
    """Full-attention cache spec with the attributes kv_group_topology reads.

    The class name is load-bearing: ``_cache_spec_reuse_policy`` classifies
    by MRO class name, so a per-layer spec must present the production
    ``FullAttentionSpec`` name to resolve the ``full`` reuse policy.
    """

    mamba_cache_mode = "none"


class FullAttentionLayerSpec(FullAttentionSpec):
    """Per-layer full-attention spec inside one manager group."""


class MambaSpec:
    """Recurrent cache spec aligned for full-checkpoint page restore."""

    mamba_cache_mode = "align"
    tokens_per_state = 256
    num_speculative_blocks = 0
    num_prefill_checkpoint_blocks = 1


class MambaLayerSpec(MambaSpec):
    """Per-layer recurrent spec; the MRO name resolves recurrent_align."""


class FullAttentionGroup(FullAttentionSpec):
    """Group-level spec carrying group topology attributes."""

    def __init__(self, layer_names: tuple[str, ...]) -> None:
        self.block_size = 256
        self.storage_block_size = 0
        self.page_size_bytes = TARGET_PAGE_BYTES
        self.layer_names = layer_names
        self.kv_cache_specs = {
            name: FullAttentionLayerSpec() for name in layer_names
        }


class MambaGroup(MambaSpec):
    """Group-level aligned-recurrent spec carrying group attributes."""

    def __init__(self, layer_names: tuple[str, ...]) -> None:
        self.block_size = 256
        self.storage_block_size = 0
        self.page_size_bytes = RECURRENTPAGE_BYTES
        self.layer_names = layer_names
        self.kv_cache_specs = {name: MambaLayerSpec() for name in layer_names}


class KvCacheGroup:
    """One KV-cache group as the config topology walk consumes it."""

    def __init__(self, spec: FullAttentionGroup | MambaGroup) -> None:
        self.kv_cache_spec = spec
        self.layer_names = spec.layer_names
        self.is_eagle_group = False


class KvCacheConfig:
    """The qualification fixture's two-group KV-cache configuration."""

    def __init__(self) -> None:
        self.kv_cache_groups = (
            KvCacheGroup(FullAttentionGroup(("tgt_kva",))),
            KvCacheGroup(MambaGroup(("rec_ckvb",))),
        )


@dataclass(frozen=True)
class PageTailFixture:
    """Immutable inputs for one page-tail qualification cohort."""

    identity_salt: str
    layout: PageLayout
    tokens_per_page: int
    target_checkpoint: str
    draft_checkpoint: str
    quantization_layout: str
    rope_layout: str

    def page_target_bytes(self, page_index: int) -> bytes:
        """One deterministic 1024-byte target page for a logical page index.

        Reused pages must be byte-identical across snapshots, so the label is
        fixed per index rather than derived from generation or snapshot
        boundaries.
        """

        return bytes((page_index % 251,)) * TARGET_PAGE_BYTES

    def page_recurrent_bytes(self) -> bytes:
        """The single deterministic recurrent checkpoint page."""

        return bytes(range(RECURRENTPAGE_BYTES))

    def snapshot_bytes(self, page_count: int) -> bytes:
        """Encode the complete SPHP1 snapshot for ``page_count`` pages."""

        target = b"".join(
            self.page_target_bytes(index) for index in range(page_count)
        )
        return encode_page_snapshot(
            self.layout,
            (page_count, 1),
            {"tgt_kva": target, "rec_ckvb": self.page_recurrent_bytes()},
        )


def _manager_page_quantization_layout() -> str:
    """Mirror the connector's block-page quantization-layout namespace.

    The connector forms ``<profile quantization layout>:manager-pages-v2:
    <topology digest>`` from the registered KV-cache groups
    (spark_context_cache_config.py). The fixture replays that construction on
    equivalent full-attention and aligned-recurrent group specs so its
    identity namespace digest is computed by the production function rather
    than restated.
    """

    return "glm53-flash-hybrid-block-pages-v1:manager-pages-v2:" + (
        kv_group_topology_digest(KvCacheConfig(), dcp_degree=4)
    )


def _page_fixture_layout() -> PageLayout:
    return PageLayout(
        (
            PageGroup(
                256,
                (
                    PageLayer(
                        "tgt_kva",
                        "u8",
                        (TARGET_PAGE_BYTES,),
                        TARGET_PAGE_BYTES,
                    ),
                ),
                reuse_policy="full",
            ),
            PageGroup(
                256,
                (
                    PageLayer(
                        "rec_ckvb",
                        "u8",
                        (RECURRENTPAGE_BYTES,),
                        RECURRENTPAGE_BYTES,
                    ),
                ),
                reuse_policy="recurrent_align",
            ),
        )
    )


def build_fixture() -> PageTailFixture:
    """Build the canonical qualification fixture."""

    return PageTailFixture(
        identity_salt="sparkcache-qualification/page-tail-delta",
        layout=_page_fixture_layout(),
        tokens_per_page=256,
        target_checkpoint=hashlib.sha256(b"qualification-target").hexdigest(),
        draft_checkpoint=hashlib.sha256(b"qualification-draft").hexdigest(),
        quantization_layout=_manager_page_quantization_layout(),
        rope_layout="glm53-flash-rope-v1",
    )


def snapshot_digest(snapshot: bytes) -> str:
    """SHA-256 of one encoded snapshot."""

    return hashlib.sha256(snapshot).hexdigest()
