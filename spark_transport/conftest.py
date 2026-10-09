"""Pytest collection for spark_transport.

``libsircl/`` is the libsircl C library. Its suites are scripts that its
Makefile runs against a built library (``make check``), not pytest modules,
so pytest does not collect them; CI runs them in the ``libsircl`` job.
"""
collect_ignore = ["libsircl"]
