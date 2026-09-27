"""Exercise the B12X selection-cache correction with CPU doubles."""

import ast
from contextlib import contextmanager
import hashlib
import importlib
import importlib.machinery
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS

import pytest

from integrations.b12x.selection_cache import sparkring_b12x_selection_cache as correction

# Importing the module installs its finder, as the .pth file does in the image.
# The tests below use private meta paths instead.
sys.meta_path[:] = [item for item in sys.meta_path if not isinstance(item, correction.Finder)]


# PreparationJob._lookup from b12x/preparation/session.py (SHA-256 83bf552115fd)
# in SparkRing's B12X composition c8e461281a38. Set SPARKRING_B12X_SOURCE_ROOT to
# a checkout of that composition to compare against the complete file.
SOURCE = """class PreparationJob:
    def _lookup(self, obligation, selections):
        if obligation.selection is not None:
            return
        obligation.key = self._choice_key(obligation, selections)
        if obligation.key is None:
            return
        # Local hits cannot remove one participant from a shared tuning race.
        # Compiled programs remain reusable; only distributed selection is rerun.
        if (self.autotune and not self.session.cache_only
                and not self.session._stop.is_set()
                and len(self.session._tuning_ranks) > 1):
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
"""

RECORD = {"assignment": {"width": 2}, "config": {"width": 2}, "coverage": {"measured_count": 4}}


class FrozenMapping(dict):
    """Stands in for b12x.preparation.types.FrozenMapping."""


class Cache:
    def __init__(self, records):
        self.records = records
        self.reads = []

    def get(self, key):
        self.reads.append(key)
        return self.records.get(key)


class Timing:
    def __init__(self):
        self.spans = []

    @contextmanager
    def span(self, name):
        self.spans.append(name)
        yield


class Contract:
    def _lower(self, query, device, assignment):
        assert (query, device) == ("query", "device")
        return {"width": assignment["width"]}

    def config_payload(self, config):
        return dict(config)


class Job:
    _lookup = correction.reconciled_lookup(FrozenMapping)

    def __init__(self, session, *, autotune, key):
        self.session = session
        self.autotune = autotune
        self._timing = Timing()
        self._cache_hits = 0
        self._key = key

    def _choice_key(self, obligation, selections):
        assert selections == {"earlier": "selection"}
        return self._key

    def _selection(self, obligation, config, source, assignment):
        return NS(config=config, source=source, assignment=assignment)


def job(*, ranks=(0, 1), synchronized, autotune=True, cache_only=False, stopped=False,
        records=None, key="shared-key"):
    stop = threading.Event()
    if stopped:
        stop.set()
    cache = Cache({"shared-key": RECORD} if records is None else records)
    opened = []

    def selection_cache():
        opened.append(cache)
        return cache

    session = NS(cache_only=cache_only, _stop=stop, _tuning_ranks=ranks,
                 _tuning_cache_synchronized=synchronized, _selection_cache=selection_cache)
    return Job(session, autotune=autotune, key=key), cache, opened


def obligation(selection=None):
    validated = []
    configuration = NS(space=NS(validate=validated.append), query="query", device="device")
    return NS(selection=selection, key=None, coverage={}, configuration=configuration,
              request=NS(plan=NS(contract=Contract()))), validated


def lookup(target, item):
    target._lookup(item, {"earlier": "selection"})


def test_sharded_tuning_uses_the_reconciled_cache():
    target, cache, opened = job(synchronized=True)
    item, validated = obligation()
    lookup(target, item)
    assert item.key == "shared-key"
    assert cache.reads == ["shared-key"] and opened == [cache]
    assert target._timing.spans == ["cache_lookup"]
    assert validated == [{"width": 2}] and isinstance(validated[0], FrozenMapping)
    assert item.selection.source == "cached" and item.selection.config == {"width": 2}
    assert item.coverage == {"measured_count": 4} and item.coverage is not RECORD["coverage"]
    assert target._cache_hits == 1


def test_sharded_tuning_before_reconciliation_keeps_every_rank_in_the_race():
    target, cache, opened = job(synchronized=False)
    item, _ = obligation()
    lookup(target, item)
    assert item.key == "shared-key"
    assert item.selection is None and item.coverage == {}
    assert cache.reads == [] and opened == [] and target._timing.spans == []
    assert target._cache_hits == 0


def test_single_rank_reads_its_cache_without_reconciliation():
    target, cache, _ = job(ranks=(0,), synchronized=False)
    item, _ = obligation()
    lookup(target, item)
    assert cache.reads == ["shared-key"]
    assert item.selection.source == "cached" and target._cache_hits == 1


@pytest.mark.parametrize("settings", (
    {"cache_only": True}, {"stopped": True}, {"autotune": False},
), ids=("cache-only", "tuning-stopped", "autotune-off"))
def test_sharded_sessions_that_cannot_race_read_the_cache(settings):
    target, cache, _ = job(synchronized=False, **settings)
    item, _ = obligation()
    lookup(target, item)
    assert cache.reads == ["shared-key"]
    assert item.selection.source == "cached" and target._cache_hits == 1


def test_a_reconciled_miss_leaves_the_request_to_tuning():
    target, cache, _ = job(synchronized=True, records={})
    item, _ = obligation()
    lookup(target, item)
    assert cache.reads == ["shared-key"]
    assert item.selection is None and target._cache_hits == 0


def test_selected_and_uncacheable_requests_do_not_read_the_cache():
    target, cache, _ = job(synchronized=True)
    chosen = object()
    item, _ = obligation(selection=chosen)
    lookup(target, item)
    assert item.selection is chosen and item.key is None
    target, cache, _ = job(synchronized=True, key=None)
    item, _ = obligation()
    lookup(target, item)
    assert item.key is None and item.selection is None and cache.reads == []


def test_a_record_that_lowers_differently_is_rejected():
    changed = {**RECORD, "config": {"width": 4}}
    target, _, _ = job(synchronized=True, records={"shared-key": changed})
    item, _ = obligation()
    with pytest.raises(ValueError, match="no longer lowers"):
        lookup(target, item)
    assert item.selection is None and target._cache_hits == 0


def composition_lookup():
    source = SOURCE
    root = os.environ.get("SPARKRING_B12X_SOURCE_ROOT")
    if root:
        data = (Path(root) / "b12x/preparation/session.py").read_bytes()
        assert hashlib.sha256(data).hexdigest() == correction.SESSION_SHA256
        source = data.decode("utf-8")
    tree = ast.parse(source)
    (owner,) = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PreparationJob"]
    (method,) = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_lookup"]
    return method


def correction_lookup():
    tree = ast.parse(Path(correction.__file__).read_text(encoding="utf-8"))
    (factory,) = [node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "reconciled_lookup"]
    (method,) = [node for node in ast.walk(factory) if isinstance(node, ast.FunctionDef) and node.name == "_lookup"]
    return method


def test_replacement_is_the_composition_method_with_one_added_condition():
    original, replacement = composition_lookup(), correction_lookup()
    conditions = [node for node in ast.walk(replacement)
                  if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And) and len(node.values) == 5]
    assert len(conditions) == 1
    added = conditions[0].values.pop()
    assert ast.unparse(added) == "not self.session._tuning_cache_synchronized"
    assert ast.dump(replacement) == ast.dump(original)


SESSION = """class FrozenMapping(dict):
    pass


class PreparationJob:
    def _lookup(self, obligation, selections):
        \"\"\"Composition method.\"\"\"
        raise AssertionError("uncorrected")
"""


@pytest.fixture
def fake_b12x(tmp_path, monkeypatch):
    """A minimal importable b12x.preparation.session on a private meta path."""
    session = tmp_path / "b12x/preparation/session.py"
    session.parent.mkdir(parents=True)
    (tmp_path / "b12x/__init__.py").write_text("")
    (tmp_path / "b12x/preparation/__init__.py").write_text("")
    session.write_text(SESSION)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "meta_path", [item for item in sys.meta_path
                                           if not isinstance(item, correction.Finder)])
    for name in [name for name in sys.modules if name == "b12x" or name.startswith("b12x.")]:
        monkeypatch.delitem(sys.modules, name)
    importlib.invalidate_caches()
    yield session
    for name in [name for name in sys.modules if name == "b12x" or name.startswith("b12x.")]:
        del sys.modules[name]


def match(monkeypatch, session):
    monkeypatch.setattr(correction, "SESSION_SHA256", hashlib.sha256(session.read_bytes()).hexdigest())


def test_import_corrects_the_matching_session_source(fake_b12x, monkeypatch, capsys):
    match(monkeypatch, fake_b12x)
    correction.install()
    module = importlib.import_module(correction.TARGET)
    method = module.PreparationJob._lookup
    assert getattr(method, correction.MARKER)
    assert method.__qualname__ == "PreparationJob._lookup" and method.__doc__ == "Composition method."
    assert "reads reconciled selections" in capsys.readouterr().err
    assert correction.apply(module) is False  # already corrected


def test_import_leaves_another_session_source_unchanged(fake_b12x, capsys):
    correction.install()
    module = importlib.import_module(correction.TARGET)
    assert not hasattr(module.PreparationJob._lookup, correction.MARKER)
    assert "not applied" in capsys.readouterr().err
    with pytest.raises(AssertionError, match="uncorrected"):
        module.PreparationJob()._lookup(None, None)


def test_a_failed_correction_does_not_fail_the_import(fake_b12x, monkeypatch, capsys):
    fake_b12x.write_text("class PreparationJob:\n    pass\n")
    match(monkeypatch, fake_b12x)
    correction.install()
    module = importlib.import_module(correction.TARGET)
    assert not hasattr(module, "FrozenMapping")
    assert "correction failed" in capsys.readouterr().err


def test_installing_after_the_session_loaded_corrects_it(fake_b12x, monkeypatch):
    match(monkeypatch, fake_b12x)
    module = importlib.import_module(correction.TARGET)
    assert not hasattr(module.PreparationJob._lookup, correction.MARKER)
    correction.install()
    assert getattr(module.PreparationJob._lookup, correction.MARKER)


def test_the_finder_precedes_the_path_finder_without_displacing_front_finders():
    front = object()
    meta_path = [front, importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter,
                 importlib.machinery.PathFinder]
    correction.install(meta_path)
    correction.install(meta_path)
    assert meta_path[0] is front
    assert sum(isinstance(item, correction.Finder) for item in meta_path) == 1
    assert isinstance(meta_path[3], correction.Finder) and meta_path[4] is importlib.machinery.PathFinder


def test_the_startup_hook_imports_the_correction_module():
    hook = Path(correction.__file__).with_suffix(".pth")
    assert hook.read_text(encoding="utf-8") == "import " + Path(correction.__file__).stem + "\n"
