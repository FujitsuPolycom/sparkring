"""`sparkring node assets` prints this Spark's checkpoint survey summary.

The survey itself (discovery, identification, the path guard, the portable
probe) is tested in test_checkpoint_search.py.
"""
import json

from runtime.common import installer, setup
from runtime.host import assets, checkpoint_search

PROFILE = "qwen38-flash-next-tp2"
DIRECTORY = ("/srv/sparkring/{}/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/"
             "60215d26cf5e42c2db6128774032d57fc62678da")


def test_discover_returns_the_local_survey_summary(monkeypatch):
    calls = []
    folder = "/var/tmp/models/qwen"
    document = {
        "schema": "sparkring-checkpoint-survey/v1", "host": "spark-aa42",
        "repository": "local-inference-lab/Qwen3.8-Flash-Next-NVFP4", "revision": "60215d26cf5e42c2db6128774032d57fc62678da",
        "operator": "root", "docker": {"userns": False, "driver": "overlay2"},
        "owned": {"path": DIRECTORY.format("tp2"), "state": "absent", "fstype": "ext4", "free_bytes": 5, "files": {}},
        "search": {"complete": True, "passes": 1, "seconds": 1.2, "entries": 812, "unvisited": {}, "skipped_mounts": [],
                   "large_directories": 0, "unreadable": 0, "errors": ["docker: unavailable: timeout"]},
        "candidates": [{"path": folder, "layout": "local-dir", "found_by": ["folder"], "commit": None, "branches": [],
                        "home": None, "sparkring": False, "mount_id": 29, "device": 66306, "rotational": False,
                        "files": {"config.json": {"state": "match"}}, "counts": {"match": 48}}],
        "not_used": [], "named": []}

    def survey(pins, options):
        calls.append((pins, options))
        return document
    monkeypatch.setattr(checkpoint_search, "survey", survey)
    result = assets.discover(PROFILE, cluster="tp2")
    pins, options = calls[0]
    assert pins == installer.checkpoint_pins(setup.selection(PROFILE))
    assert options == checkpoint_search.options(owned=DIRECTORY.format("tp2"), root="/")
    assert result["owned"] == {"path": DIRECTORY.format("tp2"), "state": "absent", "fstype": "ext4", "free_bytes": 5,
                               "files": 0}
    # Per-file entries stay out of the summary; counts and layouts remain.
    assert result["candidates"] == [{"path": folder, "layout": "local-dir", "commit": None, "branches": [], "home": None,
                                     "sparkring": False, "found_by": ["folder"], "counts": {"match": 48}}]
    assert result["search"]["complete"] and result["errors"] == 1 and result["cluster"] == "tp2"
    json.dumps(result)


def test_checkpoint_directory_names_this_sparks_cluster(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(checkpoint_search, "survey", lambda pins, options: seen.append(options["owned"]) or {})
    # No controller record and no workspace: setup's default cluster name.
    assets.discover(PROFILE, root=str(tmp_path))
    # A worker holds its cluster's workspace; deployment workspaces of managed
    # GLM installations sit beside it.
    (tmp_path / "srv/sparkring/tp4").mkdir(parents=True)
    (tmp_path / "srv/sparkring/tp4-managed").mkdir()
    assets.discover(PROFILE, root=str(tmp_path))
    # Node A's controller record names the cluster.
    record = tmp_path / "var/lib/sparkring/controller/cluster.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"name": "ring"}))
    assets.discover(PROFILE, root=str(tmp_path))
    assert seen == [DIRECTORY.format("sparkring"), DIRECTORY.format("tp4"), DIRECTORY.format("ring")]

