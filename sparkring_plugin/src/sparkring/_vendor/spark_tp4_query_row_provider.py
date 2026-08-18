"""Generic query-row policy resolver for the TP4 adapters.

The eager TP4 all-reduce admits payloads by query-row count at the
default width. Which row counts exist is policy, and this module is the
single owner of that policy for both the admission gate
(spark_tp4_backend) and the reservation namespace
(spark_tp4_port_namespace), so the two can never disagree.

Three mutually exclusive sources, resolved per call:

- ``VLLM_SPARK_TP4_PREFILL_Q512=1``: the broad prefill geometry,
  rows 1..512.
- ``VLLM_SPARK_TP4_QUERY_ROW_PROVIDER=<module>``: an external provider
  module owns the row set. The module is imported lazily, only when the
  variable is set, and must expose
  ``provider_query_rows(environ) -> Iterable[int]``. The returned rows
  are validated here; the provider owns any policy of its own
  (including reading provider-specific environment variables).
- Neither: the contiguous range ``1..MAX_QUERY_ROWS`` from
  spark_tp4_query_contract.

Failure behavior is closed: a configured provider that cannot be
imported, lacks the interface, or returns an invalid row set raises
ValueError naming ``VLLM_SPARK_TP4_QUERY_ROW_PROVIDER`` and the module.
Configuring both Q512 and a provider raises rather than letting two
geometry sources compete silently.

The provider module name is part of the launch identity: every rank
must configure the same value (or none). A mismatch produces differing
port reservations and fails native session establishment; it is not
detectable from a single rank.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable, Mapping

from spark_tp4_query_contract import MAX_QUERY_ROWS

PROVIDER_ENV = "VLLM_SPARK_TP4_QUERY_ROW_PROVIDER"
_PREFILL_Q512_ENV = "VLLM_SPARK_TP4_PREFILL_Q512"

# Rows occupy row-denominated reservation slots below the extension
# span, which begins at slot 512; a provider may not exceed it.
MAX_PROVIDER_QUERY_ROW = 512

_PREFILL_Q512_ROWS = tuple(range(1, 513))

# One resolution per (provider, q512, cap) triple. The environment is
# immutable for the life of a launch, so this is a per-process constant
# in practice; the cache keeps the admission hot path allocation-free.
_cache: dict[tuple[str | None, bool, int], tuple[int, ...]] = {}


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _prefill_q512(environ: Mapping[str, str]) -> bool:
    value = environ.get(_PREFILL_Q512_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(
            f"{_PREFILL_Q512_ENV} must be '0', '1', or unset"
        )
    return value == "1"


def provider_module_name(
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """The configured provider module name, or None when unset/empty."""

    return _environment(environ).get(PROVIDER_ENV, "").strip() or None


def _validated_rows(rows: Iterable[int], *, module_name: str) -> tuple[int, ...]:
    prefix = f"{PROVIDER_ENV}={module_name}"
    try:
        candidates = list(rows)
    except TypeError as error:
        raise ValueError(
            f"{prefix}: provider_query_rows must return an iterable of "
            f"integers, got {type(rows).__name__}"
        ) from error
    if not candidates:
        raise ValueError(f"{prefix}: provider returned no query rows")
    for row in candidates:
        if isinstance(row, bool) or not isinstance(row, int):
            raise ValueError(
                f"{prefix}: query rows must be integers, got {row!r}"
            )
        if not 1 <= row <= MAX_PROVIDER_QUERY_ROW:
            raise ValueError(
                f"{prefix}: query rows must be in "
                f"[1, {MAX_PROVIDER_QUERY_ROW}], got {row}"
            )
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"{prefix}: query rows must be unique")
    return tuple(sorted(candidates))


def _provider_rows(
    module_name: str, environ: Mapping[str, str]
) -> tuple[int, ...]:
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(
            f"{PROVIDER_ENV} names {module_name!r}, which is not "
            f"importable: {error}"
        ) from error
    interface = getattr(module, "provider_query_rows", None)
    if not callable(interface):
        raise ValueError(
            f"{PROVIDER_ENV} names {module_name!r}, which does not "
            "define provider_query_rows(environ)"
        )
    return _validated_rows(interface(environ), module_name=module_name)


def resolve_query_rows(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """The admitted query-row set for this environment, sorted ascending.

    Raises ValueError on conflicting geometry sources or on a configured
    provider that is absent, interface-less, or returns invalid rows.
    """

    environment = _environment(environ)
    q512 = _prefill_q512(environment)
    module_name = provider_module_name(environment)
    if q512 and module_name is not None:
        raise ValueError(
            f"{_PREFILL_Q512_ENV}=1 and {PROVIDER_ENV}={module_name} "
            "are mutually exclusive geometry sources"
        )
    key = (module_name, q512, MAX_QUERY_ROWS)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    if q512:
        rows = _PREFILL_Q512_ROWS
    elif module_name is not None:
        rows = _provider_rows(module_name, environment)
    else:
        rows = tuple(range(1, MAX_QUERY_ROWS + 1))
    _cache[key] = rows
    return rows


def maximum_query_rows(environ: Mapping[str, str] | None = None) -> int:
    """The largest admitted query-row count for this environment."""

    return resolve_query_rows(environ)[-1]
