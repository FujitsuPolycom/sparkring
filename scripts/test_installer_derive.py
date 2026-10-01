"""A derived checkpoint's rank operations against real files, hard links and a fake Docker that runs the recipe.

The base and donor are small synthetic checkpoints (runtime/common/test_mxfp8_attention.py):
the donor's MXFP8 tensors are the base's own weights quantized by the reference
quantizer, which a stub vLLM module also supplies to the recipe run inside the
fake ``docker run``. The manifest pins the files the recipe produces from them.
"""
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from runtime.common import derived_checkpoint, mxfp8_attention as recipe
from runtime.host import checkpoint_place as place
from scripts import installer_host as host

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rank operations use Linux hard links, /proc/self/fd, O_NOFOLLOW, O_NOATIME, flock and change times")
torch = pytest.importorskip("torch")

from runtime.common import test_mxfp8_attention as synthetic  # noqa: E402

REPOSITORY = "fixture-owner/fixture-model"
BASE = "b" * 40
DONOR = "d" * 40
DEPLOYMENT = "e" * 64
IMAGE = "sha256:" + "a" * 64
NAME = "fixture-derived"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pins(revision, files):
    entries = {name: {"size": len(data), "sha256": sha256(data), "git_blob": "0" * 40} for name, data in files.items()}
    return {"schema": "sparkring-checkpoint-pins/v1", "repository": REPOSITORY, "revision": revision,
            "index": recipe.INDEX, "weights": sorted(n for n in files if n.endswith(".safetensors")), "optional": [],
            "files": entries}


STUB = synthetic.STUB

FAKE_DOCKER = r'''#!{python}
"""Stand-in for the docker CLI: inspections, the pinned image's download client and the recipe container."""
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
if "/out" in mounts:
    program = [mounts.get(value, value) for value in args[start + 1:]]
    environment = dict(os.environ, PYTHONPATH=config["stub"] + os.pathsep + config["path"])
    done = subprocess.run([sys.executable, "-c", args[start], *program], env=environment)
    sys.exit(done.returncode)
repository, revision, *names = args[start + 1:]
for name in names:
    path = os.path.join(mounts["/fetch"], name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as output:
        output.write(bytes.fromhex(config["files"][name]))
'''


def environment(tmp_path, monkeypatch, *, tamper_donor=False, wrong=None):
    """A derived lock over a synthetic base and donor; ``call(operation, document)`` runs a rank operation."""
    built = synthetic.fixture(tmp_path / "published")
    base_dir, donor_dir = built[0], built[1]
    if tamper_donor:
        metadata, tensors, start = recipe.read_header(donor_dir / synthetic.DONOR)
        name = synthetic.MODULES[1] + ".weight_scale"
        sources = {key: (dtype, shape, recipe.read_tensor(donor_dir / synthetic.DONOR, start, begin, end))
                   for key, (dtype, shape, begin, end) in tensors.items()}
        scale = bytearray(sources[name][2])
        scale[0] ^= 1
        sources[name] = (*sources[name][:2], bytes(scale))
        (donor_dir / synthetic.DONOR).unlink()
        recipe.write_shard(donor_dir / synthetic.DONOR, sources, metadata)
    base_files = {path.name: path.read_bytes() for path in sorted(base_dir.iterdir())}
    base_files["tokenizer.json"] = b'{"model": {"vocab": {}}}'
    donor_files = {path.name: path.read_bytes() for path in sorted(donor_dir.iterdir())}
    needed = sorted(donor_files)
    # The donor checkpoint holds more than the recipe reads; its directory stays partial.
    donor_files["config.json"] = b'{"model_type": "donor"}'
    base_pins, donor_pins = pins(BASE, base_files), pins(DONOR, donor_files)
    recipe_path = "runtime/common/mxfp8_attention.py"
    source = (Path(host.installer.ROOT) / recipe_path).read_bytes()
    donor = {"repository": REPOSITORY, "revision": DONOR, "files": needed}
    manifest = {"schema": derived_checkpoint.SCHEMA, "repository": "sparkring-derived/fixture-model",
                "base": {"repository": REPOSITORY, "revision": BASE}, "donor": donor,
                "recipe": {"path": recipe_path, "sha256": sha256(source)},
                "transform": {"check": "fixture", "format": dict(recipe.ENTRY),
                              "modules": {"count": len(synthetic.MODULES),
                                          "sha256": recipe.modules_digest(synthetic.MODULES)},
                              "rewritten_shards": [synthetic.CHANGED]},
                "index": recipe.INDEX, "files": {}}
    manifest["revision"] = derived_checkpoint.identity(manifest["base"], donor, manifest["recipe"])
    expected = tmp_path / "expected"
    expected.mkdir()
    reference = tmp_path / "reference-base"
    reference.mkdir()
    for name, data in base_files.items():
        (reference / name).write_bytes(data)
    if not tamper_donor:
        recipe.derive(str(reference), str(donor_dir), str(expected), derived_checkpoint.record(manifest),
                      synthetic.quantize, log=lambda text: None)
    outputs = {path.name: path.read_bytes() for path in expected.iterdir()}
    for name, data in base_files.items():
        if name not in outputs:
            manifest["files"][name] = {"origin": "base", "size": len(data), "sha256": sha256(data)}
    for name in (synthetic.CHANGED, "config.json", "hf_quant_config.json", recipe.INDEX, recipe.RECORD):
        data = outputs.get(name, b"x")
        manifest["files"][name] = {"origin": "recipe", "size": len(data),
                                   "sha256": "0" * 64 if name == wrong else sha256(data)}
    contracts = {(REPOSITORY, BASE): base_files, (REPOSITORY, DONOR): donor_files,
                 (manifest["repository"], manifest["revision"]): outputs}

    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("29 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n")
    monkeypatch.setattr(place, "MOUNTINFO", str(mountinfo))
    monkeypatch.setattr(place, "RECORDS", str(tmp_path / "records/files"))
    monkeypatch.setattr(host, "CHECKPOINTS", tmp_path / "records")
    monkeypatch.setattr(host.shutil, "disk_usage", lambda path: SimpleNamespace(total=1 << 50, used=0, free=1 << 45))
    monkeypatch.setattr(host.installer, "validate", lambda value: value)
    monkeypatch.setattr(host.installer, "checkpoint_pins", lambda card, **kwargs: {
        BASE: base_pins, DONOR: donor_pins}[card["model_revision"]])

    def contract(card):
        files = contracts[(card["model_repository"], card["model_revision"])]
        return {"config_sha256": sha256(files.get("config.json", b"")),
                "index_sha256": sha256(files.get(recipe.INDEX, b""))}
    monkeypatch.setattr(host.installer, "checkpoint_contract", contract)
    model = {"repository": manifest["repository"], "revision": manifest["revision"]}
    monkeypatch.setattr(host.derived_checkpoint, "model_of", lambda card, **kwargs: dict(model)
                        if card.get("target_variant") == NAME else None)
    monkeypatch.setattr(host.derived_checkpoint, "load", lambda card, **kwargs: manifest
                        if card.get("target_variant") == NAME else None)
    monkeypatch.setattr(host.derived_checkpoint, "donor_card", lambda card, value, **kwargs: {
        **card, "model_repository": REPOSITORY, "model_revision": DONOR, "target_variant": "fixture-donor"})

    stub = tmp_path / "stub" / "vllm/model_executor/layers/quantization/utils"
    stub.mkdir(parents=True)
    for parent in [stub, *stub.parents][:5]:
        (parent / "__init__.py").write_text("")
    (stub / "mxfp8_utils.py").write_text(STUB)
    fake = tmp_path / "fake-docker"
    fake.mkdir()
    log = fake / "log.jsonl"
    (fake / "config.json").write_text(json.dumps({
        "log": str(log), "stub": str(tmp_path / "stub"), "path": os.pathsep.join(sys.path),
        "files": {name: data.hex() for name, data in donor_files.items()}}))
    program = fake / "docker"
    program.write_text(FAKE_DOCKER.replace("{python}", sys.executable))
    program.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_DOCKER", str(fake / "config.json"))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".installer-owner.json").write_text(json.dumps({"deployment": DEPLOYMENT}))
    root = tmp_path / "srv/sparkring/tp2/checkpoints"
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

    copy = tmp_path / "operator-copy"
    copy.mkdir()
    for name, data in base_files.items():
        (copy / name).write_bytes(data)
    adopted = call("model-adopt", {"files": {name: {"action": "link", "source": str(copy / name), "size": len(data)}
                                             for name, data in base_files.items()}})
    assert adopted["complete"]

    def runs():
        return [json.loads(line) for line in log.read_text().splitlines()
                if json.loads(line)[:1] == ["run"]] if log.exists() else []
    return SimpleNamespace(call=call, manifest=manifest, base=base_path, outputs=outputs, runs=runs,
                           derived=root / "sparkring-derived--fixture-model" / manifest["revision"],
                           donor=root / REPOSITORY.replace("/", "--") / DONOR, state=workspace / "installer",
                           donor_files={name: donor_files[name] for name in needed}, recipe=recipe_path)


def journal(directory):
    return json.loads((directory.parent / ("." + directory.name + ".sparkring") / "journal.json").read_text())["files"]


def test_rank_operations_link_download_derive_and_verify(tmp_path, monkeypatch):
    env = environment(tmp_path, monkeypatch)
    kept = derived_checkpoint.files_of(env.manifest, "base")
    linked = env.call("derive-link", {"receipts": []})
    assert not linked["complete"] and sorted(linked["linked"]) == sorted(kept)
    assert linked["missing"] == sorted(derived_checkpoint.files_of(env.manifest, "recipe"))
    for name in kept:
        # Hard links: the derived directory's unchanged files are the base's inodes.
        assert os.stat(env.derived / name).st_ino == os.stat(env.base / name).st_ino
    fetched = env.call("derive-donor", {"names": sorted(env.donor_files)})
    assert sorted(fetched["fetched"]) == sorted(env.donor_files)
    assert {entry["origin"] for entry in journal(env.donor).values()} == {"hub"}
    derived = env.call("derive-run")
    assert derived["complete"] and derived["missing"] == []
    for name, item in env.manifest["files"].items():
        assert sha256((env.derived / name).read_bytes()) == item["sha256"]
    origins = {name: entry["origin"] for name, entry in journal(env.derived).items()}
    assert {origins[name] for name in kept} == {"link"}
    assert {origins[name] for name in derived_checkpoint.files_of(env.manifest, "recipe")} == {"derive"}
    receipt = json.loads((env.state / "derived/model.json").read_text())
    assert (receipt["repository"], receipt["revision"], receipt["path"], receipt["origin"]) == (
        env.manifest["repository"], env.manifest["revision"], str(env.derived), "derived-checkpoint")
    assert not (env.derived.parent / ("." + env.derived.name + ".sparkring") / "derive").exists()
    [download, derivation] = env.runs()
    assert "/out" not in json.dumps(download)
    # The recipe runs CPU only, without network, as root, with the base and donor read-only and only staging writable.
    for option in (["--network", "none"], ["--runtime", "runc"], ["--user", "0:0"], ["--env", "CUDA_VISIBLE_DEVICES="],
                   ["--mount", f"type=bind,src={env.base},dst=/base,readonly"],
                   ["--mount", f"type=bind,src={env.donor},dst=/donor,readonly"]):
        assert any(derivation[i:i + 2] == option for i in range(len(derivation)))
    staging = env.derived.parent / ("." + env.derived.name + ".sparkring") / "derive"
    writable = f"type=bind,src={staging}/out,dst=/out"
    assert derivation[derivation.index(writable) - 1] == "--mount"
    assert [value for value in derivation if value.startswith("type=bind") and "readonly" not in value] == [writable]
    assert derivation[derivation.index("-c") + 1] == (Path(host.installer.ROOT) / env.recipe).read_text()
    assert env.call("derive") == {"ok": True} and env.call("derive-check") == {"ok": True}
    # Linking refreshed the base's receipt, so its verification needs no hashing.
    assert env.call("model-check") == {"ok": True}


def test_a_repeated_installation_reuses_the_verified_derived_directory(tmp_path, monkeypatch):
    env = environment(tmp_path, monkeypatch)
    env.call("derive-link", {"receipts": []})
    env.call("derive-donor", {"names": sorted(env.donor_files)})
    env.call("derive-run")
    runs = len(env.runs())
    again = env.call("derive-link", {"receipts": []})
    assert again["complete"] and again["linked"] == []
    assert env.call("derive-donor", {"names": []})["fetched"] == []
    assert env.call("derive-run")["complete"] and len(env.runs()) == runs
    # Without its receipt, as after automatic release removed the workspace, the journal decides.
    (env.state / "derived/model.json").unlink()
    assert env.call("derive") == {"ok": True}
    assert (env.state / "derived/model.json").is_file() and len(env.runs()) == runs


def test_recipe_output_that_differs_from_the_manifest_is_refused_and_not_placed(tmp_path, monkeypatch):
    env = environment(tmp_path, monkeypatch, wrong="config.json")
    env.call("derive-link", {"receipts": []})
    env.call("derive-donor", {"names": sorted(env.donor_files)})
    with pytest.raises(ValueError, match=r"The recipe runtime/common/mxfp8_attention\.py wrote config\.json with other "
                                         r"contents than the derived checkpoint's manifest pins"):
        env.call("derive-run")
    assert not (env.derived / "config.json").exists()
    assert not (env.derived.parent / ("." + env.derived.name + ".sparkring") / "derive").exists()
    assert not (env.state / "derived/model.json").exists()
    with pytest.raises(ValueError, match="is incomplete on Node 0; sudo sparkring install derives it"):
        env.call("derive")


def test_a_donor_that_is_not_the_base_in_mxfp8_is_refused(tmp_path, monkeypatch):
    env = environment(tmp_path, monkeypatch, tamper_donor=True)
    env.call("derive-link", {"receipts": []})
    env.call("derive-donor", {"names": sorted(env.donor_files)})
    module = synthetic.MODULES[1].replace(".", r"\.")
    with pytest.raises(ValueError, match=r"did not derive the checkpoint on Node 0: mxfp8_attention: " + module
                                         + ": quantizing the base's BF16 weight does not reproduce"):
        env.call("derive-run")
    assert not (env.derived / synthetic.CHANGED).exists()


def test_donor_files_outside_the_approved_downloads_are_refused(tmp_path, monkeypatch):
    env = environment(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="which the approved plan does not download"):
        env.call("derive-donor", {"names": [recipe.INDEX]})
    assert env.runs() == []
    with pytest.raises(ValueError, match="lacks the donor files"):
        env.call("derive-run")


def test_a_link_across_filesystems_stops_without_copying(tmp_path, monkeypatch):
    env = environment(tmp_path, monkeypatch)

    def refuse(*args, **kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link")
    monkeypatch.setattr(place, "place_link", refuse)
    with pytest.raises(ValueError, match="cannot be hard-linked into .* \\(EXDEV\\).*nothing was copied"):
        env.call("derive-link", {"receipts": []})
    assert sorted(os.listdir(env.derived)) == []


def test_transfers_of_the_derived_checkpoint_use_its_own_directory(tmp_path, monkeypatch):
    env = environment(tmp_path, monkeypatch)
    env.call("derive-link", {"receipts": []})
    recipe_files = derived_checkpoint.files_of(env.manifest, "recipe")
    document = {"repository": env.manifest["repository"], "revision": env.manifest["revision"],
                "files": {name: env.manifest["files"][name]["sha256"] for name in recipe_files},
                "sizes": dict(recipe_files)}
    prepared = env.call("model-transfer-prepare", document)
    assert prepared["needed"] == sorted(recipe_files)
    assert prepared["rsync"] == str(env.derived.parent / ("." + env.derived.name + ".sparkring") / "receive/rsync")
    for name, data in env.outputs.items():
        (Path(prepared["rsync"]) / name).write_bytes(data)
    done = env.call("model-transfer-complete", document)
    assert done["complete"] and (env.state / "derived/model.json").is_file()
    assert {journal(env.derived)[name]["origin"] for name in recipe_files} == {"rsync"}
