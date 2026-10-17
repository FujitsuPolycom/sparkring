"""Local and live qualification for page-tail publication under DCP.

The package defines a byte-exact GPU-free fixture, the local harness that
drives the page-snapshot and page-delta commit/restore paths of a
:class:`sparkcache.persistent_context_cache.cache_manifest.ManifestStore`,
the shared receipt schema, and the validation contract for receipts produced
by either the local harness or the live runner under ``deploy/``.
"""

from __future__ import annotations

from sparkcache.qualification.fixture import PageTailFixture, build_fixture
from sparkcache.qualification.harness import run_page_tail_qualification
from sparkcache.qualification.receipt import RECEIPT_SCHEMA, validate_receipt

__all__ = [
    "RECEIPT_SCHEMA",
    "PageTailFixture",
    "build_fixture",
    "run_page_tail_qualification",
    "validate_receipt",
]
