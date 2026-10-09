"""Pytest collection for spark_transport.

``libsircl/`` is a vendored libsircl snapshot (scripts/sync_libsircl.py). Its
suites are scripts that its Makefile runs against a built library
(``make check``), not pytest modules, so pytest does not collect them; the
vendored copy's own check is scripts/test_sync_libsircl.py.
"""
collect_ignore = ["libsircl"]
