"""The NVFP4-CSF serving directory's rank operations against real files, hard links and a fake Docker.

The base is a small synthetic checkpoint in the NVFP4-CSF container layout
(``metadata/``, ``tensors/``, ``manifest.json``); the serving directory's
manifest is ``scripts/pin_csf_serving.py``'s for it, and the fake ``docker
run`` runs the recipe, ``runtime/common/csf_serving.py``, as the installer
does. The derived checkpoint has no donor.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from runtime.common import csf_serving, derived_checkpoint
from runtime.host import checkpoint_place as place
from scripts import installer_host as host, pin_csf_serving

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rank operations use Linux hard links, /proc/self/fd, O_NOFOLLOW, O_NOATIME, flock and change times")

REPOSITORY = "fixture-owner/fixture-csf"
BASE = "c" * 40
DEPLOYMENT = "f" * 64
IMAGE = "sha256:" + "a" * 64
NAME = "csf-serving"
SOURCE = b'{\n  "model_type": "glm_moe_dsa",\n  "quantization_config": {"quant_algo": "NVFP4"}\n}\n'
FAKE_DOCKER = r'''#!{python}
"""Stand-in for the docker CLI: the recipe container and the inspections around it."""
import json, os, subprocess, sys
config = json.load(open(os.environ["FAKE_DOCKER"]))
args = sys.argv[1:]
if args[:2] == ["--context", "default"]:
    args = args[2:]
with open(config["log"], "a") as log:
    log.write(json.dumps(args) + "\n")
if args[:2] == ["container", "inspect"]:
    sys.exit(1)
if args[:2] == ["image", "inspect"]:
    print(json.dumps([{"Id": args[2]}]))
    sys.exit(0)
if args[:1] != ["run"]:
    sys.exit(2)
mounts = {}
for position, value in enumerate(args):
    if value == "--mount":
        fields = dict(item.split("=", 1) for item in args[position + 1].split(",") if "=" in item)
        mounts[fields["dst"]] = fields["src"]
start = args.index("-c") + 1
program = [mounts.get(value, value) for value in args[start + 1:]]
sys.exit(subprocess.run([sys.executable, "-I", "-c", args[start], *program]).returncode)
'''


def journal(directory):
    return json.loads((directory.parent / ("." + directory.name + ".sparkring") / "journal.json").read_text())["files"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def container_files():
    index = json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}}).encode()
    return {"config.json": SOURCE, "metadata/config.json": SOURCE, "metadata/model.safetensors.index.json": index,
            "metadata/tokenizer.json": b'{"model": {}}', "metadata/chat_template.jinja": b"{{ messages }}",
            "manifest.json": json.dumps({"schema": csf_serving.LAYOUT, "family": "glm53_744b_nvfp4",
                                         "metadata_sha256": {"config.json": sha256(SOURCE)}}).encode(),
            "build-contract.json": b'{"schema": "lil-nvfp4-csf-checkpoint/1"}',
            "tensors/model-00001-of-00001.safetensors": b"\x08\x00\x00\x00\x00\x00\x00\x00{}      "}


def environment(tmp_path, monkeypatch, *, wrong=None, source_sha256=None):
    """A lock whose default checkpoint is the serving directory of a synthetic container."""
    files = container_files()
    pins = {"schema": "sparkring-checkpoint-pins/v1", "repository": REPOSITORY, "revision": BASE,
            "index": "metadata/model.safetensors.index.json",
            "weights": ["tensors/model-00001-of-00001.safetensors"], "optional": [],
            "files": {name: {"size": len(data), "sha256": sha256(data), "git_blob": "0" * 40}
                      for name, data in files.items()}}
    transform = pin_csf_serving.transform(pins, json.loads(files["manifest.json"]))
    if source_sha256 is not None:
        transform["source_config"]["sha256"] = source_sha256
    written = csf_serving.serving_config(SOURCE)
    config = {"size": len(written), "sha256": "0" * 64 if wrong == "config.json" else sha256(written)}
    manifest = pin_csf_serving.manifest(pins, "sparkring-derived/fixture-csf-serving", transform, config)

    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("29 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n")
    monkeypatch.setattr(place, "MOUNTINFO", str(mountinfo))
    monkeypatch.setattr(place, "RECORDS", str(tmp_path / "records/files"))
    monkeypatch.setattr(host, "CHECKPOINTS", tmp_path / "records")
    monkeypatch.setattr(host.shutil, "disk_usage", lambda path: SimpleNamespace(total=1 << 50, used=0, free=1 << 45))
    monkeypatch.setattr(host.installer, "validate", lambda value: value)
    monkeypatch.setattr(host.installer, "checkpoint_pins", lambda card, **kwargs: pins)

    def contract(card):
        if card["model_repository"] == manifest["repository"]:
            return {"config_sha256": manifest["files"]["config.json"]["sha256"],
                    "index_sha256": manifest["files"]["model.safetensors.index.json"]["sha256"]}
        return {"config_sha256": sha256(SOURCE), "index_sha256": pins["files"][pins["index"]]["sha256"]}
    monkeypatch.setattr(host.installer, "checkpoint_contract", contract)
    model = {"repository": manifest["repository"], "revision": manifest["revision"]}
    monkeypatch.setattr(host.derived_checkpoint, "model_of", lambda card, **kwargs: dict(model)
                        if card.get("target_variant") == NAME else None)
    monkeypatch.setattr(host.derived_checkpoint, "load", lambda card, **kwargs: manifest
                        if card.get("target_variant") == NAME else None)

    fake = tmp_path / "fake-docker"
    fake.mkdir()
    log = fake / "log.jsonl"
    (fake / "config.json").write_text(json.dumps({"log": str(log)}))
    program = fake / "docker"
    program.write_text(FAKE_DOCKER.replace("{python}", sys.executable))
    program.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_DOCKER", str(fake / "config.json"))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".installer-owner.json").write_text(json.dumps({"deployment": DEPLOYMENT}))
    root = tmp_path / "srv/sparkring/ring/checkpoints"
    base_path = root / REPOSITORY.replace("/", "--") / BASE
    lock = {"id": DEPLOYMENT, "backend": "compose", "site_input": {}, "image_runtime": {},
            "selection": {"profile": "fixture", "nodes": 1, "image_id": IMAGE, "image_reference": IMAGE,
                          "configuration": "profiles/fixture/config.json", "target_variant": NAME,
                          "model_repository": REPOSITORY, "model_revision": BASE},
            "site": {"workspace": str(workspace), "ranks": [
                {"rank": 0, "host": "root@spark0", "model": str(base_path), "cache": str(tmp_path / "cache"),
                 "repository": str(workspace / "source"), "reuse_verified_model": False}]}}

    def call(operation, document=None):
        monkeypatch.setattr(sys, "stdin", io.StringIO("" if document is None else json.dumps(document)))
        return host.perform(operation, lock, 0)

    # The staged download: an hf download --local-dir copy, adopted with links and copies.
    staged = tmp_path / "srv/sparkring/downloads" / REPOSITORY.replace("/", "--") / BASE
    for name, data in files.items():
        (staged / name).parent.mkdir(parents=True, exist_ok=True)
        (staged / name).write_bytes(data)
    adopted = call("model-adopt", {"files": {name: {"action": "link" if name.endswith(".safetensors") else "copy",
                                                    "source": str(staged / name), "size": len(data)}
                                             for name, data in files.items()}})
    assert adopted["complete"]

    def runs():
        return [json.loads(line) for line in log.read_text().splitlines()
                if json.loads(line)[:1] == ["run"]] if log.exists() else []
    return SimpleNamespace(call=call, manifest=manifest, base=base_path, runs=runs, staged=staged,
                           serving=root / "sparkring-derived--fixture-csf-serving" / manifest["revision"],
                           state=workspace / "installer")


def test_the_serving_directory_links_the_container_writes_its_config_and_verifies(tmp_path, monkeypatch):
    env = environment(tmp_path, monkeypatch)
    kept = derived_checkpoint.files_of(env.manifest, "base")
    assert {"tokenizer.json", "model.safetensors.index.json", "chat_template.jinja", "manifest.json",
            "tensors/model-00001-of-00001.safetensors"} <= set(kept)
    linked = env.call("derive-link", {"receipts": []})
    assert sorted(linked["linked"]) == sorted(kept)
    assert linked["missing"] == ["config.json", "derivation.json"]
    for name in kept:
        source = derived_checkpoint.source_of(env.manifest, name)
        # Every kept file, the root links of metadata/ among them, is the container's own inode.
        assert os.stat(env.serving / name).st_ino == os.stat(env.base / source).st_ino
    # Each container file is linked once: metadata/ keeps only config.json, which the root's config.json replaces.
    assert sorted(path.name for path in (env.serving / "metadata").iterdir()) == ["config.json"]
    # The weights are also the staged download's inodes: nothing was copied.
    shard = "tensors/model-00001-of-00001.safetensors"
    assert os.stat(env.serving / shard).st_ino == os.stat(env.staged / shard).st_ino
    assert env.call("derive-donor", {"names": []}) == {"ok": True, "fetched": [], "present": []}
    with pytest.raises(ValueError, match="has no donor"):
        env.call("derive-donor", {"names": ["x"]})
    derived = env.call("derive-run")
    assert derived["complete"] and derived["missing"] == []
    config = json.loads((env.serving / "config.json").read_text())
    assert config["quantization_config"] == {"quant_method": "nvfp4_csf", "format_version": 1,
                                             "checkpoint_root": "/models/target",
                                             "source_quantization_config": {"quant_algo": "NVFP4"}}
    for name, item in env.manifest["files"].items():
        assert sha256((env.serving / name).read_bytes()) == item["sha256"]
    origins = {name: entry["origin"] for name, entry in journal(env.serving).items()}
    assert {origins[name] for name in kept} == {"link"} and origins["config.json"] == "derive"
    [run] = env.runs()
    # The recipe runs CPU only, without network, with the container read-only and no donor.
    assert ["--mount", f"type=bind,src={env.base},dst=/base,readonly"] in [run[i:i + 2] for i in range(len(run))]
    assert "/donor" not in json.dumps(run) and "--donor" not in run
    assert run[run.index("-c") + 1] == (Path(host.installer.ROOT) / pin_csf_serving.RECIPE).read_text()
    assert env.call("derive") == {"ok": True} and env.call("derive-check") == {"ok": True}
    receipt = json.loads((env.state / "derived/model.json").read_text())
    assert (receipt["repository"], receipt["path"]) == (env.manifest["repository"], str(env.serving))


def test_a_source_configuration_that_differs_from_the_manifest_stops_the_recipe(tmp_path, monkeypatch):
    env = environment(tmp_path, monkeypatch, source_sha256="0" * 64)
    env.call("derive-link", {"receipts": []})
    with pytest.raises(ValueError, match=r"did not derive the checkpoint on Node 0: csf_serving: The base's "
                                         r"metadata/config\.json has SHA-256"):
        env.call("derive-run")
    assert not (env.serving / "config.json").exists() and not (env.state / "derived/model.json").exists()


def test_a_written_config_that_differs_from_the_manifest_is_not_placed(tmp_path, monkeypatch):
    env = environment(tmp_path, monkeypatch, wrong="config.json")
    env.call("derive-link", {"receipts": []})
    with pytest.raises(ValueError, match=r"The recipe runtime/common/csf_serving\.py wrote config\.json with other "
                                         r"contents than the derived checkpoint's manifest pins"):
        env.call("derive-run")
    assert not (env.serving / "config.json").exists()
    with pytest.raises(ValueError, match="is incomplete on Node 0"):
        env.call("derive")


def test_a_kept_file_whose_base_changed_is_refused(tmp_path, monkeypatch):
    env = environment(tmp_path, monkeypatch)
    # Replace the container's tokenizer behind SparkRing's back: same size, other bytes.
    target = env.base / "metadata/tokenizer.json"
    data = target.read_bytes()
    os.unlink(target)
    target.write_bytes(data[:-1] + b"]")
    with pytest.raises(ValueError, match=r"differs from its recorded files \(metadata/tokenizer\.json\)"):
        env.call("derive-link", {"receipts": []})
    assert not (env.serving / "tokenizer.json").exists()
