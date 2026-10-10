"""The catalog of SIRCL's vLLM shims (``sparkring_sircl.vllm.catalog``, ``shims.json``) and ``serve shims``.

The catalog must name every shim of ``shims.py`` with the pinned builds of
``pins.py`` and the anchors of ``hooks.py``; ``shims.json`` must be the
generated catalog. Statuses are checked on synthetic vLLM trees, and with
``SIRCL_TEST_VLLM_ROOTS`` (``vllm`` package directories separated by the
platform's path separator) on real trees identified by the pinned build they
match. None of these tests needs torch or vLLM.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sparkring_sircl.vllm import catalog, hooks, pins, shims
from sparkring_sircl.vllm.serve import cli, probe

EVERY_SHIM = set(shims.SHIMS)
# The statuses in the pinned trees: every shim is verified, but the two prefill row-ownership shims are absent from
# the karmic-kraken-beta tree, which has neither ownership module.
EXPECTED = {
    "lil-image-aba309e4610c": dict.fromkeys(EVERY_SHIM, "verified"),
    "lil-karmic-kraken-beta-4a87c588": {**dict.fromkeys(EVERY_SHIM, "verified"),
                                        "mhc_prefill_shard": "absent", "qwen_hc_prefill_shard": "absent"},
    "sparkring-kraken-beta-20261007-bc9ea774": dict.fromkeys(EVERY_SHIM, "verified"),
}


def test_the_catalog_names_every_shim_with_the_builds_and_anchors_of_pins_and_hooks():
    names = [entry.name for entry in catalog.ENTRIES]
    assert len(names) == len(set(names)) and set(names) == EVERY_SHIM
    assert set(EXPECTED) == {build.name for build in pins.SUPPORTED}
    document = catalog.document()
    assert [build["name"] for build in document["builds"]] == [build.name for build in pins.SUPPORTED]
    assert set(document["statuses"]) == {"verified", "applicable-unverified", "absent"}
    for record in document["shims"]:
        name, files = record["name"], shims.SHIMS[record["name"]].files
        assert record["files"] == list(files)
        having = [build for build in pins.SUPPORTED if all(build.files[file] is not None for file in files)]
        assert record["verified_builds"] == {build.name: {file: build.files[file] for file in files}
                                             for build in having}, name
        anchors = [anchor for hook in hooks.HOOKS if hook.shim == name for anchor in hook.anchors]
        assert anchors and record["anchors"] == [
            {"file": anchor.file, "lines": [anchor.first, anchor.last], "text": anchor.text} for anchor in anchors]
        # A hook that names its builds names builds that have the shim's files.
        for hook in hooks.HOOKS:
            if hook.shim == name and hook.builds:
                assert set(hook.builds) <= set(record["verified_builds"]), (name, hook.builds)
        assert record["code"] and all(code["role"] in ("wraps", "calls", "binding") for code in record["code"])
        assert all(code["file"] in pins.HOOK_FILES for code in record["code"]), name
        assert {code["file"] for code in record["code"] if code["role"] == "wraps"} <= set(files), name
        for key in ("purpose", "enable", "disable", "without"):
            assert record[key] and isinstance(record[key], str), (name, key)
        assert record["models"] and record["needs"], name
    measured = {record["name"]: [item["result"] for item in record["measured"]] for record in document["shims"]}
    assert measured["mhc_prefill_shard"] == ["prefill 12-14 % faster than with --mhc-prefill-shard off",
                                             "prefill 3.7 % faster than with --mhc-prefill-shard off"]
    assert measured["fused_allreduce_rms_norm"] == ["41.3 steps/s, 9 % more than with SIRCL_FUSED_NORM=0"]
    assert not any(measured[name] for name in EVERY_SHIM - {"mhc_prefill_shard", "fused_allreduce_rms_norm"})
    assert {hook.shim for hook in hooks.HOOKS if hook.shim} == EVERY_SHIM


def test_shims_json_is_the_generated_catalog():
    assert catalog.CATALOG_FILE.read_bytes().decode("utf-8") == catalog.render_document()
    assert catalog.main([]) == 0
    assert json.loads(catalog.CATALOG_FILE.read_text(encoding="utf-8"))["schema"] == catalog.SCHEMA


def write_tree(root: Path) -> Path:
    """A vLLM package with every hook file of the pins, every anchor of the shims' hooks (as a comment at its
    first line) and every definition the catalog names."""
    lines: dict[str, dict[int, str]] = {}
    for hook in hooks.HOOKS:
        if not hook.shim:
            continue
        for anchor in hook.anchors:
            taken = lines.setdefault(anchor.file, {})
            number = anchor.first
            while number in taken:
                number += 1
            taken[number] = f"# {anchor.text}"
    definitions: dict[str, list[str]] = {}
    classes: dict[str, dict[str, list[str]]] = {}
    for entry in catalog.ENTRIES:
        for code in entry.code:
            owner, _, member = code.name.partition(".")
            if code.role == "binding":
                definitions.setdefault(code.file, []).append(f"from os import path as {code.name}")
            elif member:
                classes.setdefault(code.file, {}).setdefault(owner, []).append(member)
            else:
                definitions.setdefault(code.file, []).append(f"def {code.name}():\n    pass")
    for file in set(pins.HOOK_FILES) | set(lines) | set(definitions) | set(classes):
        numbered = lines.get(file, {})
        body = [numbered.get(number, "") for number in range(1, max(numbered, default=0) + 1)]
        body += definitions.get(file, [])
        for owner, members in classes.get(file, {}).items():
            body.append(f"class {owner}:\n" + "\n".join(f"    def {member}(self):\n        pass"
                                                          for member in dict.fromkeys(members)))
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(body) + "\n", encoding="utf-8")
    (root / "__init__.py").write_text("", encoding="utf-8")
    return root


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The synthetic tree, pinned as the only build."""
    root = write_tree(tmp_path / "checkout" / "vllm")
    build = pins.VllmBuild("synthetic", "0", "test tree", pins.hashes(root, pins.HOOK_FILES))
    monkeypatch.setattr(pins, "SUPPORTED", (build,))
    return root


def statuses(root: Path) -> dict[str, str]:
    return {item["name"]: item["status"] for item in catalog.tree_status(root)["shims"]}


def test_a_shim_is_verified_applicable_unverified_or_absent_in_a_tree(tree):
    assert statuses(tree) == dict.fromkeys(EVERY_SHIM, "verified")
    assert catalog.package_root(tree.parent) == tree and catalog.package_root(tree) == tree
    assert catalog.tree_status(tree.parent)["matches"] == ["synthetic"]
    # A change anywhere in a pinned file leaves the code present but unverified.
    model = tree / "models/glm5next/nvidia/model.py"
    model.write_text(model.read_text(encoding="utf-8") + "# another revision\n", encoding="utf-8")
    status = {item["name"]: item for item in catalog.tree_status(tree)["shims"]}
    assert status["mhc_prefill_shard"] == {"name": "mhc_prefill_shard", "status": "applicable-unverified",
                                           "build": None, "installs": False, "missing": []}
    assert status["worker_regimes"]["status"] == "verified" and status["worker_regimes"]["installs"]
    # A missing module, definition or anchor makes it absent, and says which.
    (tree / "models/glm5next/nvidia/mhc_prefill_sharding.py").unlink()
    status = {item["name"]: item for item in catalog.tree_status(tree)["shims"]}["mhc_prefill_shard"]
    assert status["status"] == "absent" and status["missing"][0] == (
        "models/glm5next/nvidia/mhc_prefill_sharding.py is missing")
    worker = tree / "v1/worker/gpu_worker.py"
    worker.write_text(worker.read_text(encoding="utf-8").replace("def sleep(", "def doze("), encoding="utf-8")
    status = {item["name"]: item for item in catalog.tree_status(tree)["shims"]}["worker_regimes"]
    assert status["status"] == "absent" and "v1/worker/gpu_worker.py defines no Worker.sleep" in status["missing"]
    assert catalog.defined("from a import b as c\nclass K:\n    def m(self): pass\n", "c")
    assert catalog.defined("class K:\n    def m(self): pass\n", "K.m")
    assert not catalog.defined("class K:\n    def m(self): pass\n", "K.n")
    assert not catalog.defined("def (:\n", "x")


def test_the_shims_command_prints_the_catalog_and_each_shims_status(tree, tmp_path, capsys):
    assert cli.main(["shims"]) == 0
    text = capsys.readouterr().out
    assert all(f"{name}: " in text for name in EVERY_SHIM)
    assert "measured: GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD, TP4 on the path of Sparks 0-3: prefill 12-14 % faster" in text
    assert cli.main(["shims", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(catalog.document()))
    model = tree / "models/qwen4_exp/nvidia/model.py"
    model.write_text(model.read_text(encoding="utf-8") + "# another revision\n", encoding="utf-8")
    assert cli.main(["shims", "--vllm-tree", str(tree.parent)]) == 0
    text = capsys.readouterr().out
    assert "the whole tree matches no pinned build" in text
    assert any(line.split()[:4] == ["qwen_hc_prefill_shard", "applicable-unverified", "-", "no"]
               for line in text.splitlines())
    assert any(line.split()[:4] == ["mhc_prefill_shard", "verified", "synthetic", "yes"] for line in text.splitlines())
    assert cli.main(["shims", "--vllm-tree", str(tree), "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == catalog.STATUS_SCHEMA and len(record["shims"]) == len(EVERY_SHIM)
    # The status a stage probe recorded in the serving image.
    saved = tmp_path / "stage-probe.txt"
    saved.write_text("noise\n" + probe.PREFIX + json.dumps({"shim_status": catalog.tree_status(tree)}) + "\n",
                     encoding="utf-8")
    assert cli.main(["shims", "--probe", str(saved)]) == 0
    assert "applicable-unverified" in capsys.readouterr().out
    saved.write_text(probe.PREFIX + json.dumps({"package": "x"}) + "\n", encoding="utf-8")
    assert cli.main(["shims", "--probe", str(saved)]) == 2
    assert "holds no stage probe record" in capsys.readouterr().err
    assert cli.main(["shims", "--vllm-tree", str(tmp_path / "nowhere")]) == 2
    assert "holds no vllm package" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["shims", "--vllm-tree", "a", "--probe", "b"])


def test_the_stage_probe_records_every_shims_status_and_stage_reports_it(tree, monkeypatch):
    monkeypatch.setattr(probe, "serve_path", lambda: [str(tree.parent)])
    record = probe.collect()
    assert record["modules"]["vllm"] == str(tree / "__init__.py")
    assert {item["name"]: item["status"] for item in record["shim_status"]["shims"]} == dict.fromkeys(
        EVERY_SHIM, "verified")
    (tree / "models/qwen4_exp/nvidia/hc_prefill.py").unlink()
    lines = catalog.stage_lines(probe.collect()["shim_status"])
    assert lines[0].startswith("shims: dcp_all_to_all verified (synthetic), ")
    assert "qwen_hc_prefill_shard absent" in lines[0]
    assert lines[1] == "shim qwen_hc_prefill_shard absent: models/qwen4_exp/nvidia/hc_prefill.py is missing"
    assert catalog.stage_lines(None) == []


def _roots():
    raw = os.environ.get("SIRCL_TEST_VLLM_ROOTS", "")
    return [Path(item) for item in raw.split(os.pathsep) if item]


@pytest.mark.skipif(not _roots(), reason="SIRCL_TEST_VLLM_ROOTS is not set")
@pytest.mark.parametrize("root", _roots(), ids=lambda path: f"{path.parent.parent.name}-{path.parent.name}")
def test_the_shims_status_in_every_pinned_vllm_tree(root):
    status = catalog.tree_status(root)
    assert len(status["matches"]) == 1, status["matches"]
    build = status["matches"][0]
    found = {item["name"]: item for item in status["shims"]}
    assert {name: item["status"] for name, item in found.items()} == EXPECTED[build]
    for name, item in found.items():
        if item["status"] == "verified":
            # Every definition and anchor the catalog names is in a tree whose files match a pinned build.
            assert item["installs"] and item["missing"] == [], (name, item["missing"])
            assert item["build"] == build, (name, item["build"])     # the build the whole tree matches
