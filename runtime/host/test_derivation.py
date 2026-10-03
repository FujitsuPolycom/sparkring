"""The derived checkpoint's plan section, its storage figure and its orchestration across simulated Sparks."""
import hashlib
import json
import os
import subprocess
import sys

import pytest

from runtime.host import checkpoint_plan, derivation, install_assets as assets, install_workflow
from runtime.host.install_errors import NeedsInput
from runtime.host.test_install_assets import (DEVICE, NOW, POLICY, PINS, REPO, REV, Fabric, candidate, ranks,
                                              survey)

GIB = 1024 ** 3
DONOR = "d" * 40
ROOT = "/srv/sparkring/test/checkpoints"
OWNED = f"{ROOT}/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/{REV}"
KEPT = {"tokenizer.json": 2000, "model-00001-of-00003.safetensors": 3 * GIB,
        "model-00003-of-00003.safetensors": 3 * GIB}
RECIPE = {"config.json": 1100, "model-00002-of-00003.safetensors": 2 * GIB, "derivation.json": 900}
DONOR_FILES = {"model-00009-of-00009.safetensors": GIB, "model.safetensors.index.json": 3000}


def digest(name):
    return hashlib.sha256(("derived " + name).encode()).hexdigest()


MANIFEST = {"schema": "sparkring-derived-checkpoint/v1", "repository": "sparkring-derived/Fixture-MXFP8",
            "revision": "f" * 40, "base": {"repository": REPO, "revision": REV},
            "donor": {"repository": REPO, "revision": DONOR, "files": sorted(DONOR_FILES)},
            "recipe": {"path": "runtime/common/mxfp8_attention.py", "sha256": "0" * 64}, "transform": {},
            "index": "config.json",
            "files": {**{name: {"origin": "base", "size": size, "sha256": hashlib.sha256(name.encode()).hexdigest()}
                         for name, size in KEPT.items()},
                      **{name: {"origin": "recipe", "size": size, "sha256": digest(name)}
                         for name, size in RECIPE.items()}}}
DONOR_PINS = {"repository": REPO, "revision": DONOR, "optional": [],
              "files": {name: {"size": size, "sha256": hashlib.sha256(("donor " + name).encode()).hexdigest()}
                        for name, size in DONOR_FILES.items()}}
DERIVED = f"{ROOT}/sparkring-derived--Fixture-MXFP8/{'f' * 40}"
DONOR_DIR = f"{ROOT}/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/{DONOR}"


def reading(names=(), *, donor=(), device=DEVICE, free=10 ** 13):
    """A probe of one Spark: its derived directory holding ``names`` and, on Node A, the donor directory."""
    files = {name: [MANIFEST["files"][name]["size"], MANIFEST["files"][name]["sha256"]] for name in names}
    entry = {"state": "owned" if names else "absent", "repository": MANIFEST["repository"],
             "revision": MANIFEST["revision"], "files": files, "device": device, "free_bytes": free}
    donor_entry = {"state": "owned", "repository": REPO, "revision": DONOR, "device": device, "free_bytes": free,
                   "files": {name: [DONOR_FILES[name], DONOR_PINS["files"][name]["sha256"]] for name in donor}}
    return entry, donor_entry


def probes(*held, donor=()):
    result = []
    for rank, names in enumerate(held):
        entry, donor_entry = reading(names, donor=donor)
        result.append({DERIVED: entry, **({DONOR_DIR: donor_entry} if rank == 0 else {})})
    return result


def section(*held, donor=()):
    return derivation.section("fixture-mxfp8", MANIFEST, DONOR_PINS, ranks(len(held)), probes(*held, donor=donor))


def test_without_a_derived_copy_node_a_downloads_the_donor_files_and_derives_for_every_spark():
    plan_section = section((), (), (), ())
    assert plan_section["derive"] and plan_section["source"] == 0
    assert plan_section["donor"]["download"] == sorted(DONOR_FILES)
    assert plan_section["donor"]["download_bytes"] == sum(DONOR_FILES.values())
    first, *others = plan_section["nodes"]
    assert first["derive"] == sorted(RECIPE) and first["link"] == len(KEPT)
    assert first["write_bytes"] == sum(RECIPE.values()) + sum(DONOR_FILES.values())
    # The other Sparks receive the recipe's files along the ring: ranks 1 and 3 from Node A, rank 2 from rank 1.
    assert [(node["rank"], node["receive"][0]["from"], node["write_bytes"]) for node in others] == [
        (1, 0, sum(RECIPE.values())), (2, 1, sum(RECIPE.values())), (3, 0, sum(RECIPE.values()))]
    assert all(node["path"] == DERIVED and node["derive"] == [] for node in others)
    assert plan_section["donor"]["path"] == DONOR_DIR


def test_held_files_are_reused_and_a_spark_that_holds_them_supplies_the_others():
    every = sorted(MANIFEST["files"])
    reused = section(every, every)
    assert not reused["derive"] and reused["donor"]["download"] == []
    assert all(node["write_bytes"] == 0 and node["link"] == 0 for node in reused["nodes"])
    # Node A holds the donor files and the recipe files are on rank 1: no download, no derivation.
    supplied = section((), every, donor=sorted(DONOR_FILES))
    assert not supplied["derive"] and supplied["source"] == 1
    assert supplied["nodes"][0]["receive"][0]["from"] == 1 and supplied["nodes"][0]["link"] == len(KEPT)
    # A derivation downloads only the donor files Node A lacks.
    partial = section((), (), donor=["model.safetensors.index.json"])
    assert partial["donor"]["download"] == ["model-00009-of-00009.safetensors"]


def base_plan(plan_section, *, free=10 ** 13, surveys=None):
    rows = ranks(len(plan_section["nodes"]))
    surveys = surveys or [survey(rank, candidate(f"/data/qwen{rank}", PINS["files"].keys() - {"README.md"}))
                          for rank in range(len(rows))]
    for item in surveys:
        item["owned"]["free_bytes"] = free
    return checkpoint_plan.plan(PINS, surveys, rows, policy=POLICY, now=NOW, derivation=plan_section,
                                request={"profile": "qwen38-flash-next-tp2", "checkpoint": "fixture-mxfp8"})


def test_the_plan_counts_derived_writes_in_each_sparks_free_space_and_download():
    plan_section = section((), ())
    plan = base_plan(plan_section)
    assert plan["problems"] == [] and plan["derivation"] == plan_section
    first, second = plan["nodes"]
    # The base's weight files are linked and its two small files copied; the derived files come on top.
    assert first["write_bytes"] == 3000 and first["storage"]["derived_bytes"] == plan_section["nodes"][0]["write_bytes"]
    assert first["required_bytes"] == checkpoint_plan.required_space(
        [1000, 2000, *plan_section["nodes"][0]["written"]], cache_bytes=POLICY["cache_bytes"])
    assert second["required_bytes"] == checkpoint_plan.required_space(
        [1000, 2000, *plan_section["nodes"][1]["written"]], cache_bytes=POLICY["cache_bytes"])
    assert checkpoint_plan.download_bytes(plan) == sum(DONOR_FILES.values())
    assert [item["kind"] for item in checkpoint_plan.attention(plan)] == ["download"]
    lines = checkpoint_plan.describe(plan)
    assert lines[-1] == f"Downloads {checkpoint_plan._human(sum(DONOR_FILES.values()))} from huggingface.co on Node 0."
    text = "\n".join(lines)
    assert "Derived checkpoint fixture-mxfp8 (sparkring-derived/Fixture-MXFP8 at ffffffffffff): 6 files" in text
    assert f"downloads 2 files of {REPO} at dddddddddddd" in text and "derives 3 files" in text
    assert "receives 3 files" in text and "from Node 0 over the fabric" in text
    assert checkpoint_plan.summary(plan)["derivation"]["download"] == sorted(DONOR_FILES)
    # Too little space for the derived files stops the plan with the Spark named.
    short = base_plan(plan_section, free=plan_section["nodes"][0]["write_bytes"])
    assert [problem["rank"] for problem in short["problems"]] == [0, 1]
    assert "to derive the checkpoint" in short["problems"][0]["message"]


def test_a_base_served_in_place_or_a_derived_directory_on_another_filesystem_stops_the_plan():
    plan_section = section((), ())
    plan = base_plan(plan_section)
    plan["nodes"][1]["mode"] = "in-place"
    found = derivation.problems(plan_section, plan["nodes"])
    assert [(item["field"], item["rank"]) for item in found] == [("model_path", 1)]
    assert "hard-links its base's files" in found[0]["message"]
    plan_section["nodes"][0]["device"] = DEVICE + 1
    plan["nodes"][1]["mode"] = "owned"
    found = derivation.problems(plan_section, plan["nodes"])
    assert [(item["field"], item["rank"]) for item in found] == [("storage", 0)]
    assert "lies on another filesystem" in found[0]["message"]


def test_a_reviewed_derivation_bounds_later_downloads_and_writes():
    reviewed = section((), (), donor=sorted(DONOR_FILES))
    fresh = section((), ())
    items = derivation.envelope(reviewed, fresh)
    assert [item["kind"] for item in items] == ["download"] and items[0]["names"] == sorted(DONOR_FILES)
    assert derivation.envelope(fresh, fresh) == []
    held = sorted(MANIFEST["files"])
    grown = derivation.envelope(section(held, held), fresh)
    assert {(item["kind"], item["rank"]) for item in grown} == {("download", 0), ("writes", 0), ("writes", 1)}
    assert grown[1]["bytes"] == fresh["nodes"][0]["write_bytes"] - sum(DONOR_FILES.values())
    bound = derivation.bounded(fresh, reviewed)
    assert bound["donor"]["download"] == [] and bound["nodes"][0]["write_bytes"] == reviewed["nodes"][0]["write_bytes"]
    other = dict(fresh, revision="e" * 40)
    assert derivation.envelope(reviewed, other)[0]["kind"] == "checkpoint"


def test_the_probe_reads_placed_files_from_the_journal(tmp_path):
    directory = tmp_path / "checkpoints" / "sparkring-derived--Fixture-MXFP8" / ("f" * 40)
    directory.mkdir(parents=True)
    state = directory.parent / ("." + directory.name + ".sparkring")
    state.mkdir()
    (directory / "config.json").write_bytes(b"{}")
    (directory / "changed.json").write_bytes(b"[]")
    info = os.lstat(directory)
    records = {}
    for name in ("config.json", "changed.json"):
        current = os.lstat(directory / name)
        records[name] = {"state": "placed", "identity": [current.st_dev, current.st_ino], "sha256": digest(name),
                         "stats": [current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns,
                                   current.st_ctime_ns], "origin": "derive", "source": None}
    records["changed.json"]["stats"][2] += 1
    (state / "owner.json").write_text(json.dumps({"schema": "sparkring-checkpoint-owner/v1",
                                                  "repository": MANIFEST["repository"], "revision": "f" * 40,
                                                  "path": str(directory), "directory": [info.st_dev, info.st_ino]}))
    (state / "journal.json").write_text(json.dumps({"schema": "sparkring-checkpoint-journal/v1", "files": records}))
    absent = tmp_path / "checkpoints" / "elsewhere" / ("e" * 40)
    done = subprocess.run([sys.executable, "-"], input=derivation.probe_source([str(directory), str(absent)]),
                          capture_output=True, text=True, check=True)
    found = json.loads(done.stdout)
    assert found[str(directory)]["state"] == "owned"
    assert found[str(directory)]["files"] == {"config.json": [2, digest("config.json")]}
    assert found[str(absent)]["state"] == "absent" and found[str(absent)]["device"] == os.lstat(tmp_path).st_dev


def test_the_install_workflow_reads_every_spark_before_planning(monkeypatch):
    card = {"target_variant": "fixture-mxfp8", "configuration": "profiles/fixture/config.json"}
    monkeypatch.setattr(install_workflow.installer.derived_checkpoint, "load",
                        lambda value: MANIFEST if value is card else None)
    monkeypatch.setattr(install_workflow.installer.derived_checkpoint, "donor_card", lambda value, manifest: "donor")
    monkeypatch.setattr(install_workflow.installer, "checkpoint_pins", lambda value: DONOR_PINS)
    seen = []

    def invoke(host, argv, *, data, timeout):
        seen.append((host, argv))
        wanted = [DERIVED] + ([DONOR_DIR] if host.endswith(".1") else [])
        assert all(repr(path) in data for path in wanted) and "def probe(" in data
        entry, donor_entry = reading()
        return json.dumps({DERIVED: entry, **({DONOR_DIR: donor_entry} if host.endswith(".1") else {})})
    surveys = [survey(0), {"error": "timed out"}]
    result = install_workflow.derivation_section(card, ranks(2), surveys, invoke=invoke)
    assert sorted(seen) == [(f"root@198.51.100.{rank}", derivation.PROBE_COMMAND) for rank in (1, 2)]
    assert result["nodes"][0]["hostname"] == "spark-0000" and result["derive"]
    assert install_workflow.derivation_section({"target_variant": "x"}, ranks(2), surveys, invoke=invoke) is None


class Derivations:
    """Rank operations of a derived checkpoint on every Spark, simulated by the names each derived directory holds."""

    def __init__(self, count, held=None, fail=None):
        self.held = {rank: set((held or {}).get(rank, ())) for rank in range(count)}
        self.events, self.pending, self.fail = [], {}, fail
        self.everything = set(MANIFEST["files"])

    def event(self, *item):
        self.events.append(item)

    def remote(self, rank, op, data=None):
        payload = json.loads(data) if data else None
        self.events.append((op, rank, payload))
        if op == self.fail:
            raise ValueError(f"The recipe {MANIFEST['recipe']['path']} wrote config.json with other contents than "
                             "the derived checkpoint's manifest pins on Node 0; nothing was placed for them")
        if op == "derive-link":
            self.held[rank] |= set(KEPT)
            return {"complete": self.held[rank] == self.everything, "linked": sorted(KEPT),
                    "verified": {name: [1, 2, MANIFEST["files"][name]["size"], 4, 5] for name in self.held[rank]}}
        if op == "derive-donor":
            assert rank == 0
            return {"ok": True, "fetched": payload["names"]}
        if op == "derive-run":
            assert rank == 0
            self.held[0] |= set(RECIPE)
            return {"complete": self.held[0] == self.everything, "missing": []}
        if op == "model-transfer-prepare":
            assert (payload["repository"], payload["revision"]) == (MANIFEST["repository"], MANIFEST["revision"])
            self.pending[rank] = sorted(set(payload["files"]) - self.held[rank])
            return {"ok": True, "needed": self.pending[rank]}
        if op == "model-transfer-complete":
            self.held[rank] |= set(payload["files"])
            return {"ok": True, "complete": self.held[rank] == self.everything}
        raise AssertionError("unexpected rank operation " + op)


@pytest.fixture
def derived(monkeypatch):
    monkeypatch.setattr(assets.installer.derived_checkpoint, "load", lambda card: MANIFEST)
    labels = []
    import contextlib

    @contextlib.contextmanager
    def step(message, **kwargs):
        labels.append(message)
        yield {"failed": False}
    monkeypatch.setattr(assets.progress, "step", step)
    return labels


def lock(count):
    return {"id": "qwen-test", "selection": {"model_repository": REPO, "model_revision": REV,
                                             "target_variant": "fixture-mxfp8"}, "site": {"ranks": ranks(count)}}


def approved(*held, donor=()):
    return {"derivation": section(*held, donor=donor), "approval": "command-line", "command": "sudo sparkring install"}


def test_node_a_derives_once_and_the_other_sparks_receive_the_recipe_files(tmp_path, derived):
    sparks = Derivations(4)
    fabric = Fabric(sparks, 4)
    plan = approved((), (), (), ())
    result = fabric.assets(tmp_path).derive(lock(4), sparks, plan=plan, receipts={1: ["/w/installer/model.json"]})
    ops = [(op, rank) for op, rank, _ in sparks.events]
    assert sorted(ops[:4]) == [("derive-link", rank) for rank in range(4)]
    assert ops[4:6] == [("derive-donor", 0), ("derive-run", 0)]
    links = {rank: payload for op, rank, payload in sparks.events if op == "derive-link"}
    assert links[1] == {"receipts": ["/w/installer/model.json"]} and links[0] == {"receipts": []}
    assert sparks.events[4][2] == {"names": sorted(DONOR_FILES)}
    receives = [(event[1], event[2]) for event in sparks.events if event[0] == "receive"]
    assert sorted(receives[:2]) == [(1, sorted(RECIPE)), (3, sorted(RECIPE))] and receives[2] == (2, sorted(RECIPE))
    assert all(held == sparks.everything for held in sparks.held.values())
    assert result["derived"] and result["donor_downloaded"] == sorted(DONOR_FILES) and result["source_rank"] == 0
    assert json.loads((tmp_path / "derivation-result.json").read_text()) == result
    assert {f"Node {rank}: Link the derived checkpoint's files from the base" for rank in range(4)} <= set(derived)
    assert "Node 0: Derive the checkpoint in the installer image" in derived


def test_a_verified_derived_directory_is_reused_without_download_derivation_or_transfer(tmp_path, derived):
    every = sorted(MANIFEST["files"])
    sparks = Derivations(2, held={0: every, 1: every})
    result = Fabric(sparks, 2).assets(tmp_path).derive(lock(2), sparks, plan=approved(every, every))
    assert [op for op, _, _ in sparks.events] == ["derive-link", "derive-link"]
    assert not result["derived"] and result["received"] == [] and result["donor_downloaded"] == []


def test_derived_writes_beyond_the_approved_plan_need_input_before_any_download(tmp_path, derived):
    every = sorted(MANIFEST["files"])
    sparks = Derivations(2)
    with pytest.raises(NeedsInput) as error:
        Fabric(sparks, 2).assets(tmp_path).derive(lock(2), sparks, plan=approved(every, every))
    assert error.value.field == "checkpoint" and "the derived checkpoint needs" in str(error.value)
    assert [op for op, _, _ in sparks.events] == ["derive-link", "derive-link"]
    # Without an approved plan, only a derived checkpoint the Sparks already hold passes.
    with pytest.raises(NeedsInput):
        Fabric(sparks, 2).assets(tmp_path).derive(lock(2), Derivations(2))


def test_a_refused_derivation_names_the_spark_and_the_file(tmp_path, derived):
    sparks = Derivations(2, fail="derive-run")
    with pytest.raises(RuntimeError, match=r"Node 0: The recipe runtime/common/mxfp8_attention\.py wrote config\.json"):
        Fabric(sparks, 2).assets(tmp_path).derive(lock(2), sparks, plan=approved((), ()))
    assert not any(op.startswith("model-transfer") for op, _, _ in sparks.events)


def test_a_plan_for_another_derived_checkpoint_is_refused(tmp_path, derived):
    plan = approved((), ())
    plan["derivation"]["revision"] = "e" * 40
    with pytest.raises(ValueError, match="derives another checkpoint"):
        Fabric(Derivations(2), 2).assets(tmp_path).derive(lock(2), Derivations(2), plan=plan)


def test_the_derive_phase_runs_the_derivation_once_and_reports_its_failure_on_every_rank(tmp_path, monkeypatch):
    import concurrent.futures
    from pathlib import Path
    from scripts import installer_runner
    rows = ranks(2)
    monkeypatch.setattr(installer_runner.Runner, "__init__",
                        lambda self, directory: setattr(self, "lock", lock(2)) or setattr(self, "directory", Path(directory)))
    calls = []
    monkeypatch.setattr(installer_runner.Runner, "_call", lambda self, target, argv, timeout: calls.append(argv) or {
        "returncode": 0, "stdout": "ok", "stderr": "", "uncertain": False})
    images = concurrent.futures.Future()
    images.set_result({"image_id": "sha256:a"})
    current = Fabric(Derivations(2), 2).assets(tmp_path / "assets")
    runs = []

    def derive(lock_value, runner, previous=None, **options):
        runs.append(sorted(options))
        return {"derived": True}
    current.derive = derive
    runner = current.runner(tmp_path, None, images, plan={"derivation": None}, receipts=["/r"])
    for row in rows:
        assert runner._call(row["host"], ["installer", "derive", str(row["rank"])], 60)["returncode"] == 0
    assert runs == [["plan", "receipts"]] and runner.derivation == {"derived": True}
    assert calls == [["installer", "derive", "0"], ["installer", "derive", "1"]]

    def refuse(lock_value, runner, previous=None, **options):
        raise RuntimeError("Node 0: The recipe runtime/common/mxfp8_attention.py wrote config.json with other contents")
    current = Fabric(Derivations(2), 2).assets(tmp_path / "again")
    current.derive = refuse
    runner = current.runner(tmp_path, None, images)
    results = [runner._call(row["host"], ["installer", "derive", str(row["rank"])], 60) for row in rows]
    assert {result["stderr"] for result in results} == {
        "Checkpoint derivation failed: Node 0: The recipe runtime/common/mxfp8_attention.py wrote config.json with "
        "other contents"}
    assert runner.models_error.startswith("Checkpoint derivation failed")


def test_a_derived_or_donor_directory_that_sparkring_did_not_create_stops_the_plan():
    plan_section = section((), ())
    plan_section["nodes"][1]["state"] = "foreign"
    plan_section["donor"]["state"] = "foreign"
    found = derivation.problems(plan_section, base_plan(section((), ()))["nodes"])
    assert [(item["field"], item["rank"]) for item in found] == [("storage", 0), ("storage", 1)]
    assert DONOR_DIR in found[0]["message"] and "was not created by SparkRing" in found[1]["message"]
    # A donor directory matters only when Node A derives.
    every = sorted(MANIFEST["files"])
    reused = section(every, every)
    reused["donor"]["state"] = "foreign"
    assert derivation.problems(reused, base_plan(reused)["nodes"]) == []
