"""Reuse reconciled B12X tuning selections when tuning is sharded across ranks.

B12X preparation measures candidate kernel configurations and stores each
winner in a per-rank selection cache. When tuning is sharded across tensor-
parallel ranks, a cache hit on one rank and a miss on another would remove that
rank from a shared tuning race. B12X therefore reconciles the ranks' caches at
the start of a sharded preparation job (``SelectionCache.reconcile``); from then
on every rank reads the same agreed records and
``PreparationSession._tuning_cache_synchronized`` is true.

SparkRing's B12X source composition c8e461281a38, the B12X in installer image
dev-20260927-h2dstaging-cuda1342-nccl2323-status031, also skips the cache in
``PreparationJob._lookup`` whenever tuning is sharded, including after
reconciliation. Every multi-rank start therefore measures every kernel family
again although the reconciled cache already holds the selections.

This module replaces ``PreparationJob._lookup`` with the same method plus one
condition: the sharded-tuning skip applies only before reconciliation. It
leaves every file of the B12X package unchanged, which the prepared RoCE
transport requires (it hashes ``b12x/preparation/session.py`` at startup), and
it replaces the method only when the loaded ``session.py`` has SHA-256
``SESSION_SHA256``. Any other B12X source is left unchanged.

``sparkring_b12x_selection_cache.pth`` in the serving interpreter's
site-packages imports this module at interpreter startup, including in spawned
workers.
"""

import hashlib
import importlib.abc
import importlib.machinery
from pathlib import Path
import sys

TARGET = "b12x.preparation.session"
# b12x/preparation/session.py in SparkRing's B12X composition c8e461281a38. The
# replacement below is that file's PreparationJob._lookup with one added
# condition, so it is valid only for exactly these bytes.
SESSION_SHA256 = "83bf552115fdb2bafa399ccfa8d7adbe4958121a3891243eff117a47ea20a31e"
MARKER = "__sparkring_reconciled_lookup__"


def reconciled_lookup(FrozenMapping):
    """Return ``PreparationJob._lookup`` that reads the cache after reconciliation.

    ``FrozenMapping`` is the session module's own class; the method body is the
    composition's method, unchanged except for the last operand of the
    sharded-tuning condition.
    """

    def _lookup(self, obligation, selections):
        if obligation.selection is not None:
            return
        obligation.key = self._choice_key(obligation, selections)
        if obligation.key is None:
            return
        # Before the ranks reconcile their caches, a local hit could remove one
        # participant from a shared tuning race. After reconciliation every
        # rank reads the same agreed records and makes the same decision.
        if (self.autotune and not self.session.cache_only
                and not self.session._stop.is_set()
                and len(self.session._tuning_ranks) > 1
                and not self.session._tuning_cache_synchronized):
            return
        with self._timing.span("cache_lookup"):
            cache = self.session._selection_cache()
            record = cache.get(obligation.key)
        if record is None:
            return
        configuration, contract = obligation.configuration, obligation.request.plan.contract
        assignment = FrozenMapping(record["assignment"])
        configuration.space.validate(assignment)
        config = contract._lower(configuration.query, configuration.device, assignment)
        if contract.config_payload(config) != record["config"]:
            raise ValueError("cached assignment no longer lowers to its saved config")
        obligation.selection = self._selection(obligation, config, "cached", assignment)
        obligation.coverage = dict(record["coverage"])
        self._cache_hits += 1

    return _lookup


def apply(module):
    """Replace ``PreparationJob._lookup`` in a loaded session module.

    Returns True when the method was replaced. A session source with another
    SHA-256, or a module that was already corrected, is left unchanged.
    """
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if not origin or hashlib.sha256(Path(origin).read_bytes()).hexdigest() != SESSION_SHA256:
        sys.stderr.write(
            "SparkRing B12X selection-cache correction not applied: "
            f"{origin} is not the B12X session source it corrects\n"
        )
        return False
    job = module.PreparationJob
    if getattr(job._lookup, MARKER, False):
        return False
    replacement = reconciled_lookup(module.FrozenMapping)
    replacement.__module__ = job._lookup.__module__
    replacement.__qualname__ = job._lookup.__qualname__
    replacement.__doc__ = job._lookup.__doc__
    setattr(replacement, MARKER, True)
    job._lookup = replacement
    sys.stderr.write(
        "SparkRing B12X selection-cache correction: PreparationJob._lookup "
        "reads reconciled selections\n"
    )
    return True


def _apply_after_load(module):
    try:
        apply(module)
    except Exception as error:
        # The unchanged method is correct, only slower; never fail the import.
        sys.stderr.write(f"SparkRing B12X selection-cache correction failed: {error!r}\n")


class Finder(importlib.abc.MetaPathFinder):
    """Correct ``b12x.preparation.session`` after its normal loader executes it."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET or self not in sys.meta_path:
            return None
        later = sys.meta_path[sys.meta_path.index(self) + 1:]
        for finder in later:
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None:
                continue
            loader = spec.loader
            if loader is None or not hasattr(loader, "exec_module"):
                return spec
            original = loader.exec_module

            def exec_module(module, original=original):
                original(module)
                _apply_after_load(module)

            loader.exec_module = exec_module
            return spec
        return None


def install(meta_path=None):
    """Insert one finder immediately before the standard path finder.

    Finders that SparkRing inserts at the front of ``sys.meta_path``, such as
    the RoCE transport selector, keep their position.
    """
    meta_path = sys.meta_path if meta_path is None else meta_path
    if any(isinstance(finder, Finder) for finder in meta_path):
        return
    position = next(
        (index for index, finder in enumerate(meta_path) if finder is importlib.machinery.PathFinder),
        len(meta_path),
    )
    meta_path.insert(position, Finder())
    loaded = sys.modules.get(TARGET)
    if loaded is not None:
        _apply_after_load(loaded)


install()
